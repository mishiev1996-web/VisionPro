"""
app.py — FastAPI server: REST API + frontend hosting + background data refresh.

Run via start.bat, or manually:
    uvicorn app:app --reload
"""
from __future__ import annotations

import asyncio
import collections
import datetime as dt
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

import joblib
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scrapers.utils import format_msk

MAX_SSE_EVENTS = 10000

import db
import data_collector
from train import build_features


# ── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple in-memory sliding-window rate limiter per IP."""
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if key not in self._hits:
                self._hits[key] = collections.deque()
            q = self._hits[key]
            while q and q[0] < now - self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True

RATE_LIMITER = _RateLimiter(max_requests=120, window_seconds=60)


MODEL = None
SCHEDULER: Optional[BackgroundScheduler] = None


# ── Job manager (single-job semantics) ────────────────────────────────────────

class Job:
    """Tracks one long-running task (collect or predict-with-refresh).

    Events are appended to `events` and streamed via SSE.
    `cancel` lets the worker break between iterations.
    """
    def __init__(self):
        self.kind: Optional[str] = None       # "collect" | "predict" | None
        self.running: bool = False
        self.continuous: bool = False
        self.cancel = threading.Event()
        self.events: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None
        self.result: Optional[Dict[str, Any]] = None   # final payload (e.g. prediction)
        self.lock = threading.Lock()
        self.job_id: Optional[int] = None

    def reset(self, kind: str) -> None:
        with self.lock:
            self.kind = kind
            self.running = True
            self.continuous = False
            self.cancel.clear()
            self.events = []
            self.result = None
            self.job_id = (self.job_id or 0) + 1

    def emit(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
        event["job_id"] = self.job_id
        with self.lock:
            if len(self.events) > MAX_SSE_EVENTS:
                self.events = self.events[-MAX_SSE_EVENTS // 2:]
            self.events.append(event)

    def finalize(self) -> None:
        with self.lock:
            self.running = False

    def snapshot(self, since: int = 0) -> Tuple[List[Dict], bool]:
        """Thread-safe snapshot: returns (events since index, is_running)."""
        with self.lock:
            return list(self.events[since:]), self.running


JOB = Job()


# ── Server lifecycle ──────────────────────────────────────────────────────────

def _load_model():
    global MODEL
    if os.path.exists("model.pkl"):
        MODEL = joblib.load("model.pkl")
        logger.info("Model loaded from model.pkl")
    else:
        MODEL = None
        logger.warning("No model.pkl — run 'python train.py' to train")


def _scheduled_refresh():
    """APScheduler hook — refresh current season silently in background."""
    if JOB.running:
        return
    try:
        data_collector.refresh_current_season()
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled Understat refresh ok.")
    except Exception as e:
        logger.info(f"[scheduler] understat refresh failed: {e}")


def _scheduled_live_refresh():
    """Fast FlashScore live refresh — every 5 minutes."""
    if JOB.running:
        return
    try:
        data_collector.refresh_live_only()
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled FlashScore live refresh ok.")
    except Exception as e:
        logger.warning(f"[scheduler] live refresh failed: {e}")


def _scheduled_espn_refresh():
    """Refresh ESPN data — every 30 minutes."""
    if JOB.running:
        return
    try:
        from web_scraper import gather_team_data
        # Refresh data for popular teams
        teams = ["Arsenal", "Chelsea", "Liverpool", "Barcelona", "Real Madrid"]
        for team in teams:
            gather_team_data(team)
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled ESPN refresh ok.")
    except Exception as e:
        logger.warning(f"[scheduler] ESPN refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _load_model()

    global SCHEDULER
    SCHEDULER = BackgroundScheduler(daemon=True)
    SCHEDULER.add_job(_scheduled_refresh, "interval", hours=6, id="understat")
    SCHEDULER.add_job(_scheduled_live_refresh, "interval", minutes=5, id="fs_live")
    SCHEDULER.add_job(_scheduled_espn_refresh, "interval", minutes=30, id="espn")
    SCHEDULER.start()
    logger.info("Scheduler started: Understat every 6h, FlashScore live every 5min, ESPN every 30min.")
    # Pre-compute model stats in background
    if MODEL is not None:
        import threading
        def _precompute():
            try:
                api_model_stats()
                logger.info("Model stats pre-computed.")
            except Exception:
                pass
        threading.Thread(target=_precompute, daemon=True).start()
    yield
    if SCHEDULER:
        SCHEDULER.shutdown(wait=False)


app = FastAPI(title="Football AI Predictor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not RATE_LIMITER.is_allowed(client_ip):
        return StreamingResponse(
            content=b'{"error":"Rate limit exceeded. Try again in a minute."}',
            status_code=429,
            media_type="application/json",
        )
    response = await call_next(request)
    return response


@app.get("/")
def root():
    import time as _time
    v = int(_time.time())
    return FileResponse("frontend/index.html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-VisionPRO-Version": str(v),
    })


@app.get("/mini-app")
def mini_app():
    return FileResponse("mini_app.html", media_type="text/html")


class BotPredictRequest(BaseModel):
    home_name: str
    away_name: str


@app.post("/api/bot/predict")
def api_bot_predict(body: BotPredictRequest):
    """Simple predict endpoint for Telegram bot (name-based)."""
    import ai_analyzer
    result = ai_analyzer.search_and_predict(
        body.home_name, body.away_name,
        progress_cb=lambda e: None,
    )
    if not result:
        return {"error": "Команды не найдены"}
    return result


# ── Reference data ────────────────────────────────────────────────────────────

@app.get("/api/leagues")
def api_leagues():
    return {"leagues": db.list_leagues()}


@app.get("/api/teams")
def api_teams(league: Optional[str] = None):
    return {"teams": db.list_teams(league)}


@app.get("/api/search-teams")
def api_search_teams(q: str = Query(min_length=1), limit: int = 10):
    """Substring search for team names (case-insensitive)."""
    q_norm = f"%{q.strip().lower()}%"
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT t.id, t.name, t.short_name, t.league_slug, l.name AS league_name "
            "FROM teams t LEFT JOIN leagues l ON l.slug=t.league_slug "
            "WHERE LOWER(t.name) LIKE ? "
            "ORDER BY "
            "  CASE WHEN LOWER(t.name)=? THEN 0 "
            "       WHEN LOWER(t.name) LIKE ? THEN 1 ELSE 2 END, "
            "  t.name "
            "LIMIT ?",
            (q_norm, q.strip().lower(), f"{q.strip().lower()}%", int(limit)),
        ).fetchall()
        return {"results": [dict(r) for r in rows]}


# ── Tables / lists ────────────────────────────────────────────────────────────

@app.get("/api/standings")
def api_standings(league: str, season: Optional[int] = None):
    if season is None:
        season = data_collector._current_season_year()
    table = db.standings(league, season)
    return {"league": league, "season": season, "table": table}


@app.get("/api/upcoming")
def api_upcoming(league: Optional[str] = None, limit: int = 20):
    items = db.upcoming_matches(league, limit)
    return {"matches": [_with_prediction(m) for m in items]}


@app.get("/api/results")
def api_results(league: Optional[str] = None, limit: int = 20):
    items = db.recent_results(league, limit)
    return {"matches": items}


# ── Direct prediction (no scrape) ─────────────────────────────────────────────

@app.get("/api/predict")
def api_predict(home_id: int, away_id: int):
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")
    return _predict_pair(home_id, away_id, home, away)


@app.get("/api/team-stats")
def api_team_stats(team_id: int, limit: int = 10):
    team = db.get_team(team_id)
    if not team:
        raise HTTPException(404, "Команда не найдена")
    return {"team": team, "recent": db.matches_played(team_id, limit=limit)}


# ── Admin / status ────────────────────────────────────────────────────────────

@app.get("/api/refresh-status")
def api_refresh_status():
    with db.connect() as conn:
        last = db.get_meta(conn, "last_refresh")
        n_matches = conn.execute("SELECT COUNT(*) AS c FROM matches").fetchone()["c"]
        n_teams = conn.execute("SELECT COUNT(*) AS c FROM teams").fetchone()["c"]
        n_leagues = conn.execute("SELECT COUNT(*) AS c FROM leagues").fetchone()["c"]
        n_elo = conn.execute("SELECT COUNT(*) AS c FROM team_elo").fetchone()["c"]
        n_inj = conn.execute("SELECT COUNT(*) AS c FROM injuries").fetchone()["c"]
    fs = db.fs_stats()
    return {
        "last_refresh": last,
        "matches": n_matches, "teams": n_teams, "leagues": n_leagues,
        "elo_teams": n_elo, "injuries": n_inj,
        "fs_matches": fs.get("total") or 0,
        "fs_countries": fs.get("countries") or 0,
        "fs_leagues": fs.get("leagues") or 0,
        "fs_live": fs.get("live") or 0,
        "model_loaded": MODEL is not None,
    }


_MODEL_STATS_CACHE: Optional[Dict[str, Any]] = None
_MODEL_STATS_CACHE_TS: float = 0


@app.get("/api/model-stats")
def api_model_stats():
    """Return model metrics: feature count, model type. Accuracy loaded separately."""
    global _MODEL_STATS_CACHE, _MODEL_STATS_CACHE_TS
    import time as _time

    if MODEL is None:
        return {"model_loaded": False}
    try:
        fmt = MODEL.get("format", "v1")
        features = MODEL.get("features", [])
        n_features = len(features)
        model_type = fmt
        if fmt == "ensemble_v3":
            model_type = "DC Ensemble"
        elif fmt == "ensemble_v2":
            model_type = "XGB+LGBM"

        # Cache accuracy for 5 minutes
        now = _time.time()
        if _MODEL_STATS_CACHE and (now - _MODEL_STATS_CACHE_TS) < 300:
            return {**_MODEL_STATS_CACHE, "model_loaded": True, "format": fmt,
                    "model_type": model_type, "n_features": n_features}

        # Run quick backtest (only if not cached)
        import backtest
        bt = backtest.run_backtest(seasons=2, max_matches_per_league=100)
        overall = bt.get("overall", {})
        result = {
            "accuracy": overall.get("accuracy", 0),
            "total_matches": overall.get("total", 0),
            "avg_log_loss": overall.get("avg_log_loss", 0),
            "top_20pct_accuracy": overall.get("top_20pct_accuracy", 0),
        }
        _MODEL_STATS_CACHE = result
        _MODEL_STATS_CACHE_TS = now

        return {**result, "model_loaded": True, "format": fmt,
                "model_type": model_type, "n_features": n_features}
    except Exception as e:
        return {"model_loaded": True, "error": str(e)}


# ── Injuries / Elo / Weather ──────────────────────────────────────────────────

@app.get("/api/injuries")
def api_injuries(league: Optional[str] = None, team_id: Optional[int] = None):
    if team_id is not None:
        return {"injuries": db.list_injuries(team_id)}
    return {"injuries": db.list_all_injuries(league)}


@app.get("/api/team-elo")
def api_team_elo(team_id: int):
    return {"team_id": team_id, "elo": db.get_team_elo(team_id)}


@app.get("/api/weather")
def api_weather(match_id: int):
    w = db.get_weather(match_id)
    return {"match_id": match_id, "weather": w}


# ── sstats.net enrichment (Glicko + multi-bookmaker odds + text summary) ────

# Hand-curated map from our Understat league slug → sstats.net leagueId.
SSTATS_LEAGUE_ID = {
    "EPL":        39,    # Premier League
    "La_liga":    140,   # La Liga
    "Bundesliga": 78,    # Bundesliga
    "Serie_A":    135,   # Serie A
    "Ligue_1":    61,    # Ligue 1
    "RFPL":       235,   # Russian Premier League
}

# Per-match cache: our match_id → sstats game_id (lazy, populated on first lookup)
_SSTATS_LINK_CACHE: Dict[int, Optional[int]] = {}
_SSTATS_CACHE_MAX = 5000


def _find_sstats_game_id(match: dict) -> Optional[int]:
    """Locate the sstats game_id for a match in our DB by (league, date, home_name)."""
    mid = match["id"]
    if mid in _SSTATS_LINK_CACHE:
        return _SSTATS_LINK_CACHE[mid]

    if len(_SSTATS_LINK_CACHE) > _SSTATS_CACHE_MAX:
        _SSTATS_LINK_CACHE.clear()

    from scrapers import sstats as _ss
    league_id = SSTATS_LEAGUE_ID.get(match["league_slug"])
    if not league_id:
        _SSTATS_LINK_CACHE[mid] = None
        return None

    # Match date — fetch sstats matches on that date and find by team name
    date_iso = match["date"][:10]
    sstats_games = _ss.fetch_games_by_date(date_iso)
    home_name_norm = (match.get("home_name") or "").lower().strip()
    away_name_norm = (match.get("away_name") or "").lower().strip()
    if not home_name_norm:
        team = db.get_team(match["home_id"])
        home_name_norm = (team["name"] if team else "").lower().strip()
    if not away_name_norm:
        team = db.get_team(match["away_id"])
        away_name_norm = (team["name"] if team else "").lower().strip()

    for g in sstats_games:
        h = (g.get("homeTeam") or {}).get("name", "").lower().strip()
        a = (g.get("awayTeam") or {}).get("name", "").lower().strip()
        # Substring match on either side
        if (home_name_norm and (home_name_norm in h or h in home_name_norm)
                and away_name_norm and (away_name_norm in a or a in away_name_norm)):
            _SSTATS_LINK_CACHE[mid] = int(g["id"])
            return int(g["id"])
    _SSTATS_LINK_CACHE[mid] = None
    return None


@app.get("/api/sstats/account")
def api_sstats_account():
    from scrapers import sstats as _ss
    info = _ss.account_info()
    return {"connected": info is not None, "info": info}


@app.get("/api/sstats/enrich")
def api_sstats_enrich(match_id: int):
    """Look up the match in sstats, return Glicko + odds consensus + summary."""
    from scrapers import sstats as _ss
    with db.connect() as conn:
        row = conn.execute(
            "SELECT m.*, th.name AS home_name, ta.name AS away_name "
            "FROM matches m "
            "JOIN teams th ON th.id = m.home_id "
            "JOIN teams ta ON ta.id = m.away_id "
            "WHERE m.id = ?", (match_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Матч не найден")
    match = dict(row)
    game_id = _find_sstats_game_id(match)
    if not game_id:
        return {"match_id": match_id, "sstats_game_id": None,
                "message": "На sstats этот матч не нашёлся"}

    glicko = _ss.fetch_glicko(game_id)
    odds_blocks = _ss.fetch_odds(game_id)
    summary = _ss.fetch_text_summary(game_id)
    consensus = _ss.consensus_odds(odds_blocks) if odds_blocks else None
    dispersion = _ss.market_dispersion(odds_blocks) if odds_blocks else None

    return {
        "match_id": match_id,
        "sstats_game_id": game_id,
        "glicko": glicko,
        "consensus": consensus,
        "dispersion": dispersion,
        "bookmaker_count": len(odds_blocks) if odds_blocks else 0,
        "text_summary": summary,
        "odds_by_bookmaker": odds_blocks,
    }


@app.get("/api/market-compare")
def api_market_compare(home_id: int, away_id: int):
    """Compare our model prediction with bookmaker consensus odds."""
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Команда не может играть сама с собой")

    # Get our prediction
    our = _predict_pair(home_id, away_id, home, away)

    # Try to find bookmaker odds
    from scrapers import sstats as _ss
    SSTATS_LEAGUE_MAP = {
        "EPL": 39, "La_liga": 140, "Bundesliga": 78,
        "Serie_A": 135, "Ligue_1": 61,
    }
    league_id = SSTATS_LEAGUE_MAP.get(home["league_slug"])
    market = None
    if league_id:
        today = dt.date.today().isoformat()
        try:
            games = _ss.fetch_games_by_date(today)
            hn = home["name"].lower().strip()
            an = away["name"].lower().strip()
            for g in games:
                h = (g.get("homeTeam") or {}).get("name", "").lower().strip()
                a = (g.get("awayTeam") or {}).get("name", "").lower().strip()
                if (hn in h or h in hn) and (an in a or a in an):
                    odds = _ss.fetch_odds(int(g["id"]))
                    if odds:
                        market = _ss.consensus_odds(odds)
                    break
        except Exception:
            pass

    result = {
        "home": home,
        "away": away,
        "our_prediction": {
            "home_win": our["probabilities"]["home_win"],
            "draw": our["probabilities"]["draw"],
            "away_win": our["probabilities"]["away_win"],
        },
        "market": None,
        "value_bets": [],
    }

    if market:
        result["market"] = {
            "implied_home": round(market["implied_h"] * 100, 1),
            "implied_draw": round(market["implied_d"] * 100, 1),
            "implied_away": round(market["implied_a"] * 100, 1),
            "avg_odds_home": market["avg_home_odds"],
            "avg_odds_draw": market["avg_draw_odds"],
            "avg_odds_away": market["avg_away_odds"],
            "bookmakers": market["bookmaker_count"],
            "overround": market["overround_pct"],
        }
        # Find value bets (our prob > market implied by >3%)
        thresholds = [
            ("home_win", "implied_h", home["name"]),
            ("draw", "implied_d", "Ничья"),
            ("away_win", "implied_a", away["name"]),
        ]
        for our_key, mkt_key, label in thresholds:
            our_val = our["probabilities"][our_key] / 100
            mkt_val = market[mkt_key]
            diff = round((our_val - mkt_val) * 100, 1)
            if diff > 3:
                result["value_bets"].append({
                    "outcome": label,
                    "our_prob": round(our_val * 100, 1),
                    "market_prob": round(mkt_val * 100, 1),
                    "edge": diff,
                })

    return result


# ── FlashScore worldwide ──────────────────────────────────────────────────────

@app.get("/api/fs/countries")
def api_fs_countries():
    return {"countries": db.fs_countries()}


@app.get("/api/fs/leagues")
def api_fs_leagues(country: str):
    return {"country": country, "leagues": db.fs_leagues_for_country(country)}


@app.get("/api/fs/matches")
def api_fs_matches(country: str, league: str, limit: int = 100):
    return {"country": country, "league": league,
            "matches": db.fs_matches_for(country, league, limit)}


@app.get("/api/fs/live")
def api_fs_live(limit: int = 100):
    return {"matches": db.fs_live_matches(limit)}


# ── Job control: collect ──────────────────────────────────────────────────────

@app.post("/api/collect/start")
def api_collect_start(mode: str = "understat", continuous: bool = False):
    """Kick off a data-collection job.

    mode:
        "understat" — only Understat (3 seasons × 6 leagues). Fast.
        "all"       — every source: Understat + ClubElo + Open-Meteo
                      + Transfermarkt + FlashScore. Slow (5-15 min).
        "flashscore" — only FlashScore worldwide today
        "live"      — only FlashScore live (very fast)
        "continuous" — keeps running, restarts after completion

    continuous: if True, auto-restart after completion
    """
    if JOB.running:
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания или нажмите Стоп")

    current = data_collector._current_season_year()
    seasons = list(range(current - 7, current + 1))   # 8 seasons total

    JOB.reset("collect")
    JOB.continuous = continuous

    def worker():
        try:
            while True:
                if mode == "all" or mode == "continuous":
                    data_collector.collect_all(seasons, progress_cb=JOB.emit,
                                               cancel_event=JOB.cancel)
                elif mode == "flashscore":
                    data_collector.collect_flashscore(progress_cb=JOB.emit,
                                                      cancel_event=JOB.cancel)
                elif mode == "live":
                    data_collector.refresh_live_only(progress_cb=JOB.emit,
                                                     cancel_event=JOB.cancel)
                else:
                    data_collector.collect_understat(seasons, progress_cb=JOB.emit,
                                                     cancel_event=JOB.cancel)

                if not continuous or JOB.cancel.is_set():
                    break

                JOB.emit({"type": "info", "msg": "Повторный сбор через 5 минут…"})
                for _ in range(300):  # 5 minutes in 1-second intervals
                    if JOB.cancel.is_set():
                        break
                    time.sleep(1)

            JOB.emit({"type": "done", "msg": "Сбор завершён"})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Неожиданная ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "collect", "mode": mode, "continuous": continuous, "seasons": seasons, "job_id": JOB.job_id}


@app.post("/api/collect/stop")
def api_collect_stop():
    if not JOB.running:
        return {"ok": True, "msg": "Ничего не запущено"}
    JOB.cancel.set()
    return {"ok": True, "msg": "Сигнал остановки отправлен — закончит текущую лигу"}


@app.get("/api/collect/status")
def api_collect_status():
    return {
        "running": JOB.running,
        "kind": JOB.kind,
        "events": len(JOB.events),
        "result": JOB.result,
    }


@app.get("/api/collect/stream")
async def api_collect_stream(request: Request, since: int = 0):
    """Server-Sent Events: streams JOB.events starting from index `since`.

    Each event is a JSON dict with `type` (info|success|error|done|stopped|start|result),
    `msg`, `ts`, and optional payload. Sends a heartbeat every ~10s to keep proxies happy.
    """
    async def event_gen():
        last_seen = max(0, since)
        last_heartbeat = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            # Thread-safe snapshot
            new_events, is_running = JOB.snapshot(last_seen)
            for ev in new_events:
                yield f"data: {json.dumps({'idx': last_seen, **ev})}\n\n"
                last_seen += 1
            # Terminate only when worker has finished AND we've sent everything
            if not is_running and not new_events:
                return
            # Heartbeat
            if time.monotonic() - last_heartbeat > 10:
                yield ": heartbeat\n\n"
                last_heartbeat = time.monotonic()
            await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ── Predict-with-refresh (the "по названиям" flow) ────────────────────────────

class PredictRequest(BaseModel):
    home_id: int
    away_id: int


@app.post("/api/predict-with-refresh/start")
def api_predict_refresh_start(body: PredictRequest):
    """Start a job that re-scrapes the relevant league then makes a prediction.

    Streams progress via the same /api/collect/stream endpoint.
    The final event has type='result' and includes the prediction payload.
    """
    if JOB.running:
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")

    home = db.get_team(body.home_id)
    away = db.get_team(body.away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if body.home_id == body.away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")

    JOB.reset("predict")

    def worker():
        try:
            JOB.emit({"type": "info", "msg": f"Поиск матча: {home['name']} vs {away['name']}"})
            current = data_collector._current_season_year()

            # Refresh the league(s) of both teams (often same league)
            leagues_to_refresh = list({home["league_slug"], away["league_slug"]})
            JOB.emit({"type": "info",
                      "msg": f"Обновляю свежие данные: {', '.join(leagues_to_refresh)} · "
                             f"сезон {current}/{current+1}"})

            def predict_cb(ev):
                # Rewrite the inner "done" so the SSE stream stays open
                # for the upcoming "result" event.
                if ev.get("type") == "done":
                    ev = {**ev, "type": "info",
                          "msg": "✓ Свежие данные собраны"}
                JOB.emit(ev)

            data_collector.collect([current], leagues=leagues_to_refresh,
                                   progress_cb=predict_cb, cancel_event=JOB.cancel)
            if JOB.cancel.is_set():
                return

            JOB.emit({"type": "info", "msg": "Считаю прогноз…"})
            prediction = _predict_pair(body.home_id, body.away_id, home, away)
            JOB.result = prediction
            JOB.emit({"type": "result",
                      "msg": "Прогноз готов",
                      "prediction": prediction})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "predict", "job_id": JOB.job_id}


# ── Model retrain trigger ─────────────────────────────────────────────────────

@app.post("/api/train")
def api_train():
    """Retrain model on current DB state. Synchronous (takes ~10s)."""
    if JOB.running:
        raise HTTPException(409, "Идёт сбор данных — сначала дождитесь его окончания")
    import train
    try:
        train.main()
    except SystemExit as e:
        raise HTTPException(400, str(e))
    _load_model()
    return {"ok": True, "model_loaded": MODEL is not None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_predict_proba(rows: list, league_slug: str,
                         home_name: Optional[str] = None,
                         away_name: Optional[str] = None):
    """Works across model formats: v3 (DC ensemble), v2 (XGB+LGB), v1 (single)."""
    import pandas as pd
    X = pd.DataFrame(rows, columns=MODEL["features"])
    fmt = MODEL.get("format", "v1")
    if fmt == "ensemble_v3":
        return MODEL["ensemble"].predict_proba(
            X, league_slug=league_slug,
            home_name=home_name, away_name=away_name,
        )
    if fmt == "ensemble_v2":
        return MODEL["ensemble"].predict_proba(X, league_slug=league_slug)
    return MODEL["model"].predict_proba(X)


def _predict_pair(home_id: int, away_id: int, home: dict, away: dict) -> dict:
    if MODEL is None:
        raise HTTPException(503, "Модель не загружена. Нажмите 'Обучить модель' или запустите train.py")

    with db.connect() as conn:
        prior = [dict(r) for r in conn.execute(
            "SELECT * FROM matches WHERE is_result=1 "
            "AND league_slug=? ORDER BY date DESC",
            (home["league_slug"],),
        ).fetchall()]

    today_iso = dt.date.today().isoformat()
    current_season = data_collector._current_season_year()
    features = build_features(
        home_id, away_id, prior,
        match_date=today_iso,
        league_slug=home["league_slug"],
        season=current_season,
    )
    proba = _model_predict_proba([features], home["league_slug"],
                                  home_name=home["name"],
                                  away_name=away["name"])[0]
    h2h = db.head_to_head(home_id, away_id, limit=5)

    return {
        "home": home,
        "away": away,
        "probabilities": {
            "home_win": round(float(proba[2]) * 100, 1),
            "draw":     round(float(proba[1]) * 100, 1),
            "away_win": round(float(proba[0]) * 100, 1),
        },
        "features": {name: (None if (isinstance(v, float) and (v != v))
                              else (float(v) if isinstance(v, float) else v))
                     for name, v in zip(MODEL["features"], features)},
        "h2h_last5": h2h,
    }


def _with_prediction(match: dict) -> dict:
    out = dict(match)
    if match.get("date"):
        out["date_msk"] = format_msk(match["date"])
    if MODEL is not None:
        try:
            with db.connect() as conn:
                prior = [dict(r) for r in conn.execute(
                    "SELECT * FROM matches WHERE is_result=1 "
                    "AND league_slug=? AND date<? ORDER BY date DESC",
                    (match["league_slug"], match["date"]),
                ).fetchall()]
            features = build_features(
                match["home_id"], match["away_id"], prior,
                match_date=match["date"],
                league_slug=match["league_slug"],
                season=match["season"],
            )
            proba = _model_predict_proba(
                [features], match["league_slug"],
                home_name=match.get("home_name"),
                away_name=match.get("away_name"))[0]
            out["our_prediction"] = {
                "home_win": round(float(proba[2]) * 100, 1),
                "draw":     round(float(proba[1]) * 100, 1),
                "away_win": round(float(proba[0]) * 100, 1),
            }
        except Exception:
            out["our_prediction"] = None
    return out


# ── AI Analysis ────────────────────────────────────────────────────────────────

@app.get("/api/ai/analyze")
def api_ai_analyze(home_id: int, away_id: int, model: str = "deepseek/deepseek-v3.2"):
    """Generate AI-powered text analysis for a match pair."""
    import ai_analyzer
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")

    result = ai_analyzer.generate_preview(home_id, away_id, model=model)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.get("/api/ai/models")
def api_ai_models():
    """List available AI models from Polza.ai."""
    import ai_analyzer
    api_key = ai_analyzer._get_api_key()
    if not api_key:
        return {"models": [], "error": "API ключ не настроен. Добавьте ключ в Апи/key.txt"}
    try:
        resp = requests.get(
            f"{ai_analyzer.POLZA_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"type": "chat"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data.get("data", []):
            pricing = m.get("top_provider", {}).get("pricing", {})
            models.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("top_provider", {}).get("context_length"),
                "price_per_1m_prompt": pricing.get("prompt_per_million"),
                "price_per_1m_completion": pricing.get("completion_per_million"),
            })
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.get("/api/ai/quick-analysis")
def api_ai_quick(home_id: int, away_id: int, model: str = "deepseek/deepseek-v3.2"):
    """Short AI verdict — just the key insight, no full analysis."""
    import ai_analyzer
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")

    result = ai_analyzer.generate_preview(home_id, away_id, model=model)
    if "error" in result:
        raise HTTPException(500, result["error"])

    full_analysis = result.get("analysis", "")
    short_prompt = [
        {"role": "system", "content": "Дай краткий вердикт (2-3 предложения) по матчу. Только главный инсайт и ГЛАВНЫЙ ПРОГНОЗ VisionPRO."},
        {"role": "user", "content": full_analysis},
    ]
    short = ai_analyzer._chat(short_prompt, model=model, temperature=0.5, max_tokens=200)

    return {
        "home": home,
        "away": away,
        "verdict": short,
        "model_used": model,
    }


# ── Save / Load Predictions ────────────────────────────────────────────────

class SavePredictionRequest(BaseModel):
    home_name: str
    away_name: str
    league: str = ""
    match_date: str = ""
    analysis: str = ""
    main_bet: str = ""
    confidence: str = ""
    home_win: Optional[float] = None
    draw_prob: Optional[float] = None
    away_win: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    exact_score: str = ""
    model_used: str = ""


@app.post("/api/predictions/save")
def api_save_prediction(body: SavePredictionRequest):
    """Save a prediction to the database."""
    import datetime as _dt
    pred = body.model_dump()
    pred["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    pred_id = db.save_prediction(pred)
    return {"ok": True, "id": pred_id}


@app.get("/api/predictions/list")
def api_list_predictions(limit: int = 50):
    """List saved predictions."""
    return {"predictions": db.list_predictions(limit)}


@app.get("/api/predictions/{pred_id}")
def api_get_prediction(pred_id: int):
    """Get one saved prediction."""
    pred = db.get_prediction(pred_id)
    if not pred:
        raise HTTPException(404, "Прогноз не найден")
    return pred


@app.get("/api/predictions/stats")
def api_predictions_stats():
    """Stats for saved predictions: win/loss, ROI, by league."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    preds = [dict(r) for r in rows]
    total = len(preds)
    if not total:
        return {"total": 0, "win_rate": 0, "by_league": {}, "confidence_breakdown": {},
                "recent_trend": []}

    by_league = {}
    conf_breakdown = {"Высокая": 0, "Средняя": 0, "Низкая": 0, "None": 0}
    recent_trend = []

    for p in preds:
        lg = p.get("league") or "Unknown"
        conf = p.get("confidence") or "None"
        if lg not in by_league:
            by_league[lg] = {"total": 0, "high_conf": 0}
        by_league[lg]["total"] += 1
        if conf == "Высокая":
            by_league[lg]["high_conf"] += 1
        if conf in conf_breakdown:
            conf_breakdown[conf] += 1

        created = p.get("created_at", "")
        recent_trend.append({
            "date": created[:10] if created else "",
            "home": p.get("home_name", ""),
            "away": p.get("away_name", ""),
            "confidence": conf,
            "main_bet": p.get("main_bet", ""),
        })

    return {
        "total": total,
        "by_league": by_league,
        "confidence_breakdown": conf_breakdown,
        "recent_trend": recent_trend[:20],
    }


@app.get("/api/predictions/{pred_id}/print")
def api_prediction_print(pred_id: int):
    """Return a printable HTML page for a saved prediction."""
    pred = db.get_prediction(pred_id)
    if not pred:
        raise HTTPException(404, "Прогноз не найден")
    from fastapi.responses import HTMLResponse
    p = pred
    conf_color = "#22c55e" if p.get("confidence") == "Высокая" else "#eab308" if p.get("confidence") == "Средняя" else "#ef4444"
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>VisionPRO — {p['home_name']} vs {p['away_name']}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#fff;color:#161826;padding:40px;max-width:600px;margin:0 auto}}
  .header{{display:flex;align-items:center;gap:10px;margin-bottom:24px;border-bottom:2px solid #06b6d4;padding-bottom:16px}}
  .logo{{font-size:22px;font-weight:700}} .logo span{{color:#06b6d4}}
  .match{{font-size:24px;font-weight:700;margin-bottom:4px}}
  .meta{{font-size:12px;color:#6b6f8a;margin-bottom:20px}}
  .bet{{background:linear-gradient(135deg,rgba(6,182,212,.08),rgba(139,92,246,.08));border:1px solid rgba(6,182,212,.2);border-radius:12px;padding:16px;margin-bottom:20px}}
  .bet-label{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#06b6d4;font-weight:700;margin-bottom:6px}}
  .bet-text{{font-size:20px;font-weight:700}}
  .bet-conf{{font-size:12px;color:{conf_color};margin-top:4px;text-align:right}}
  .bar{{display:flex;height:28px;border-radius:8px;overflow:hidden;margin:12px 0}}
  .bar div{{display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff}}
  .bar .h{{background:#22c55e}} .bar .d{{background:#eab308}} .bar .a{{background:#ef4444}}
  .analysis{{font-size:13px;line-height:1.7;color:#4b5563;white-space:pre-wrap;margin:16px 0}}
  .footer{{font-size:10px;color:#9a9db3;text-align:center;margin-top:24px;border-top:1px solid #e2e4ee;padding-top:12px}}
  @media print{{body{{padding:20px}}}}
</style></head><body>
  <div class="header"><div class="logo">Vision<span>PRO</span></div><div style="font-size:11px;color:#6b6f8a">AI Football Analytics</div></div>
  <div class="match">{p['home_name']} vs {p['away_name']}</div>
  <div class="meta">{p.get('league','')} · {p.get('model_used','')} · {p.get('created_at','')[:16]}</div>
  {"<div class='bet'><div class='bet-label'>ГЛАВНЫЙ ПРОГНОЗ</div><div class='bet-text'>"+str(p.get('main_bet',''))+"</div><div class='bet-conf'>"+str(p.get('confidence',''))+"</div></div>" if p.get('main_bet') else ""}
  {"<div class='bar'><div class='h' style='width:"+str(p['home_win'])+"%'>"+str(p['home_win'])+"%</div><div class='d' style='width:"+str(p['draw_prob'])+"%'>"+str(p['draw_prob'])+"%</div><div class='a' style='width:"+str(p['away_win'])+"%'>"+str(p['away_win'])+"%</div></div>" if p.get('home_win') is not None else ""}
  {"<div class='analysis'>"+str(p.get('analysis',''))+"</div>" if p.get('analysis') else ""}
  <div class="footer">VisionPRO · Информационный характер · {p.get('model_used','')}</div>
  <script>window.onload=()=>window.print()</script>
</body></html>"""
    return HTMLResponse(content=html)


@app.get("/api/match-detail")
def api_match_detail(home_name: str, away_name: str):
    """Get detailed info for a match: recent form, H2H, prediction."""
    with db.connect() as conn:
        home_rows = conn.execute(
            "SELECT t.id, t.name FROM teams t WHERE LOWER(t.name)=?",
            (home_name.lower().strip(),),
        ).fetchone()
        away_rows = conn.execute(
            "SELECT t.id, t.name FROM teams t WHERE LOWER(t.name)=?",
            (away_name.lower().strip(),),
        ).fetchone()

    if not home_rows or not away_rows:
        raise HTTPException(404, "Команда не найдена в базе")

    home_id = home_rows["id"]
    away_id = away_rows["id"]
    home = db.get_team(home_id)
    away = db.get_team(away_id)

    def _enrich(matches, team_id):
        out = []
        for m in matches:
            d = dict(m)
            if m["home_id"] == team_id:
                opp = db.get_team(m["away_id"])
                d["opponent"] = opp["name"] if opp else "?"
                d["is_home"] = True
            else:
                opp = db.get_team(m["home_id"])
                d["opponent"] = opp["name"] if opp else "?"
                d["is_home"] = False
            out.append(d)
        return out

    recent_home = _enrich(db.matches_played(home_id, limit=5), home_id)
    recent_away = _enrich(db.matches_played(away_id, limit=5), away_id)
    h2h = db.head_to_head(home_id, away_id, limit=5)

    prediction = None
    if MODEL is not None:
        try:
            prediction = _predict_pair(home_id, away_id, home, away)
        except Exception:
            pass

    return {
        "home": home,
        "away": away,
        "recent_home": recent_home,
        "recent_away": recent_away,
        "h2h": h2h,
        "prediction": prediction,
    }


@app.delete("/api/predictions/{pred_id}")
def api_delete_prediction(pred_id: int):
    """Delete a saved prediction."""
    ok = db.delete_prediction(pred_id)
    if not ok:
        raise HTTPException(404, "Прогноз не найден")
    return {"ok": True}


# ── Search & Auto-ingest + AI ────────────────────────────────────────────────

class SearchPredictRequest(BaseModel):
    home_name: str
    away_name: str
    model: str = "deepseek/deepseek-v3.2"


@app.post("/api/ai/search-and-predict")
def api_ai_search_predict(body: SearchPredictRequest):
    """Search for teams by name, auto-ingest from Understat if needed, predict."""
    import ai_analyzer

    result = ai_analyzer.search_and_predict(
        body.home_name, body.away_name,
        model=body.model,
    )
    return result


@app.post("/api/ai/search-and-predict/start")
def api_ai_search_predict_start(body: SearchPredictRequest):
    """Start AI analysis with SSE progress. Use /api/collect/stream for events."""
    if JOB.running:
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")

    JOB.reset("ai")
    import ai_analyzer

    def worker():
        try:
            result = ai_analyzer.search_and_predict(
                body.home_name, body.away_name,
                model=body.model,
                progress_cb=JOB.emit,
            )
            JOB.result = result
            JOB.emit({"type": "result", "msg": "Анализ готов", "prediction": result})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "ai", "job_id": JOB.job_id}


@app.get("/api/ai/search-db")
def api_ai_search_db(q: str = Query(min_length=1)):
    """Quick local DB search for teams (no scraping)."""
    results = db.search_team_fuzzy(q, limit=10)
    return {"results": results}


@app.get("/api/backtest")
def api_backtest(seasons: int = 3):
    """Run backtest on historical matches and return accuracy metrics."""
    import backtest
    result = backtest.run_backtest(seasons=seasons)
    return result


@app.post("/api/backtest/start")
def api_backtest_start(seasons: int = 3):
    """Start backtest with SSE progress. Use /api/collect/stream for events."""
    if JOB.running:
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")

    JOB.reset("backtest")
    import backtest

    def worker():
        try:
            result = backtest.run_backtest(
                seasons=seasons,
                progress_cb=JOB.emit,
            )
            JOB.result = result
            JOB.emit({"type": "result", "msg": "Backtest завершён", "result": result})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка backtest: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "backtest", "job_id": JOB.job_id}


@app.get("/api/feature-importance")
def api_feature_importance():
    """Return feature importance from the trained model."""
    if MODEL is None:
        return {"error": "Model not loaded"}
    try:
        base_xgb = MODEL["ensemble"].global_model["xgb"].calibrated_classifiers_[0].estimator
        importances = sorted(zip(MODEL["features"], base_xgb.feature_importances_),
                             key=lambda x: -x[1])
        return {"features": [{"name": n, "importance": round(float(v), 4)} for n, v in importances]}
    except Exception as e:
        return {"error": str(e)}




