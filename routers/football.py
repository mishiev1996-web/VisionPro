"""
routers/football.py — Football endpoints: leagues, teams, predict, standings,
upcoming, results, injuries, elo, weather, sstats, market-compare, FlashScore,
prematch, backtest, feature-importance.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

import db
import config
import data_collector
import state as _state
from state import JOB
from helpers import (
    SSTATS_LEAGUE_ID, LEAGUE_NAME_MAP, TEAM_NAME_MAP,
    predict_pair, with_prediction, model_predict_proba,
)
from scrapers.utils import format_msk

logger = logging.getLogger("router.football")

router = APIRouter(prefix="/api", tags=["football"])


# ── Reference data ────────────────────────────────────────────────────────────

@router.get("/leagues")
def api_leagues():
    return {"leagues": db.list_leagues()}


@router.get("/teams")
def api_teams(league: Optional[str] = None):
    return {"teams": db.list_teams(league)}


@router.get("/search-teams")
def api_search_teams(q: str = Query(min_length=1), limit: int = 10):
    q_norm = f"%{q.strip().lower()}%"
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT t.id, t.name, t.short_name, t.league_slug, l.name AS league_name "
            "FROM teams t LEFT JOIN leagues l ON l.slug=t.league_slug "
            "WHERE LOWER(t.name) LIKE ? "
            "ORDER BY "
            "  CASE WHEN LOWER(t.name)=? THEN 0 "
            "       WHEN LOWER(t.name) LIKE ? THEN 1 ELSE 2 END, "
            "  t.name LIMIT ?",
            (q_norm, q.strip().lower(), f"{q.strip().lower()}%", int(limit)),
        ).fetchall()
        return {"results": [dict(r) for r in rows]}


# ── Tables / lists ────────────────────────────────────────────────────────────

@router.get("/standings")
def api_standings(league: str, season: Optional[int] = None):
    if season is None:
        season = data_collector._current_season_year()
    table = db.standings(league, season)
    return {"league": league, "season": season, "table": table}


@router.get("/upcoming")
def api_upcoming(league: Optional[str] = None, limit: int = 20):
    items = db.upcoming_matches(league, limit)
    return {"matches": [with_prediction(m) for m in items]}


@router.get("/results")
def api_results(league: Optional[str] = None, limit: int = 20):
    items = db.recent_results(league, limit)
    return {"matches": items}


# ── Direct prediction ────────────────────────────────────────────────────────

@router.get("/predict")
def api_predict(home_id: int, away_id: int):
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")
    try:
        result = predict_pair(home_id, away_id, home, away)
        return result
    except Exception as e:
        print(f"[predict] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


@router.get("/team-stats")
def api_team_stats(team_id: int, limit: int = 10):
    team = db.get_team(team_id)
    if not team:
        raise HTTPException(404, "Команда не найдена")
    return {"team": team, "recent": db.matches_played(team_id, limit=limit)}


# ── Injuries / Elo / Weather ──────────────────────────────────────────────────

@router.get("/injuries")
def api_injuries(league: Optional[str] = None, team_id: Optional[int] = None):
    if team_id is not None:
        return {"injuries": db.list_injuries(team_id)}
    return {"injuries": db.list_all_injuries(league)}


@router.get("/team-elo")
def api_team_elo(team_id: int):
    return {"team_id": team_id, "elo": db.get_team_elo(team_id)}


@router.get("/weather")
def api_weather(match_id: int):
    w = db.get_weather(match_id)
    return {"match_id": match_id, "weather": w}


# ── sstats.net enrichment ────────────────────────────────────────────────────

_SSTATS_LINK_CACHE: Dict[int, Optional[int]] = {}


def _find_sstats_game_id(match: dict) -> Optional[int]:
    mid = match["id"]
    if mid in _SSTATS_LINK_CACHE:
        return _SSTATS_LINK_CACHE[mid]
    if len(_SSTATS_LINK_CACHE) > 5000:
        _SSTATS_LINK_CACHE.clear()
    from scrapers import sstats as _ss
    league_id = SSTATS_LEAGUE_ID.get(match["league_slug"])
    if not league_id:
        _SSTATS_LINK_CACHE[mid] = None
        return None
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
        if (home_name_norm and (home_name_norm in h or h in home_name_norm)
                and away_name_norm and (away_name_norm in a or a in away_name_norm)):
            _SSTATS_LINK_CACHE[mid] = int(g["id"])
            return int(g["id"])
    _SSTATS_LINK_CACHE[mid] = None
    return None


@router.get("/sstats/account")
def api_sstats_account():
    from scrapers import sstats as _ss
    info = _ss.account_info()
    return {"connected": info is not None, "info": info}


@router.get("/sstats/enrich")
def api_sstats_enrich(match_id: int):
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
        "match_id": match_id, "sstats_game_id": game_id,
        "glicko": glicko, "consensus": consensus, "dispersion": dispersion,
        "bookmaker_count": len(odds_blocks) if odds_blocks else 0,
        "text_summary": summary, "odds_by_bookmaker": odds_blocks,
    }


@router.get("/market-compare")
def api_market_compare(home_id: int, away_id: int):
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Команда не может играть сама с собой")
    our = predict_pair(home_id, away_id, home, away)
    from scrapers import sstats as _ss
    SSTATS_MAP = {"EPL": 39, "La_liga": 140, "Bundesliga": 78, "Serie_A": 135, "Ligue_1": 61}
    league_id = SSTATS_MAP.get(home["league_slug"])
    market = None
    if league_id:
        today = dt.date.today().isoformat()
        try:
            games = _ss.fetch_games_by_date(today)
            hn, an = home["name"].lower().strip(), away["name"].lower().strip()
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
        "home": home, "away": away,
        "our_prediction": {
            "home_win": our["probabilities"]["home_win"],
            "draw": our["probabilities"]["draw"],
            "away_win": our["probabilities"]["away_win"],
        },
        "market": None, "value_bets": [],
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

@router.get("/fs/countries")
def api_fs_countries():
    return {"countries": db.fs_countries()}


@router.get("/fs/leagues")
def api_fs_leagues(country: str):
    return {"country": country, "leagues": db.fs_leagues_for_country(country)}


@router.get("/fs/matches")
def api_fs_matches(country: str, league: str, limit: int = 100):
    return {"country": country, "league": league,
            "matches": db.fs_matches_for(country, league, limit)}


@router.get("/fs/live")
def api_fs_live(limit: int = 100):
    return {"matches": db.fs_live_matches(limit)}


# ── Pre-match endpoints ────────────────────────────────────────────────────────

@router.get("/prematch/today")
def api_prematch_today():
    try:
        from scrapers import sstats
        games = sstats.fetch_upcoming_all()
        matches = []
        for g in games:
            home = (g.get("homeTeam") or {}).get("name", "?")
            away = (g.get("awayTeam") or {}).get("name", "?")
            league = (g.get("season") or {}).get("league", {}).get("name", "?")
            league_ru = LEAGUE_NAME_MAP.get(league, league)
            matches.append({
                "game_id": g.get("id"), "home": home, "away": away,
                "home_ru": home, "away_ru": away,
                "time": g.get("date", ""),
                "league": league, "league_ru": league_ru,
                "status": g.get("statusName", "Not Started"),
            })
        return {"date": dt.date.today().isoformat(), "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": str(e), "matches": [], "count": 0}


@router.get("/prematch/live")
def api_prematch_live():
    try:
        from scrapers import sstats
        games = sstats.fetch_live_matches()
        matches = []
        for g in games:
            home = (g.get("homeTeam") or {}).get("name", "?")
            away = (g.get("awayTeam") or {}).get("name", "?")
            league = (g.get("season") or {}).get("league", {}).get("name", "?")
            league_ru = LEAGUE_NAME_MAP.get(league, league)
            home_result = g.get("homeResult")
            away_result = g.get("awayResult")
            score = f"{home_result} - {away_result}" if home_result is not None and away_result is not None else ""
            xg_home = g.get("homeXg")
            xg_away = g.get("awayXg")
            xg_str = f"xG: {xg_home:.1f} - {xg_away:.1f}" if xg_home is not None and xg_away is not None else ""
            matches.append({
                "game_id": g.get("id"), "home": home, "away": away,
                "home_ru": TEAM_NAME_MAP.get(home, home),
                "away_ru": TEAM_NAME_MAP.get(away, away),
                "time": g.get("date", ""),
                "league": league, "league_ru": league_ru,
                "status": g.get("statusName", "Live"),
                "score": score, "home_result": home_result, "away_result": away_result,
                "xg": xg_str,
            })
        return {"matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": str(e), "matches": [], "count": 0}


def _consensus_from_odds(odds_list):
    if not odds_list:
        return None
    home_odds, draw_odds, away_odds = [], [], []
    for market in odds_list:
        if market.get("marketName") == "Match Winner":
            for o in (market.get("odds") or []):
                name = o.get("name", "")
                val = float(o.get("value", 0)) if o.get("value") else 0
                if val > 1:
                    if name in ("Home", "1"):
                        home_odds.append(val)
                    elif name in ("Draw", "X"):
                        draw_odds.append(val)
                    elif name in ("Away", "2"):
                        away_odds.append(val)
    if not home_odds:
        return None
    avg_h = sum(home_odds) / len(home_odds)
    avg_d = sum(draw_odds) / len(draw_odds) if draw_odds else 3.0
    avg_a = sum(away_odds) / len(away_odds) if away_odds else 3.0
    margin = 1/avg_h + 1/avg_d + 1/avg_a
    return {
        "home_odds": round(avg_h, 2), "draw_odds": round(avg_d, 2), "away_odds": round(avg_a, 2),
        "implied_home": round((1/avg_h) / margin, 3),
        "implied_draw": round((1/avg_d) / margin, 3),
        "implied_away": round((1/avg_a) / margin, 3),
        "bookmaker_count": len(home_odds),
    }


@router.get("/prematch/live/{game_id}")
def api_prematch_live_detail(game_id: int):
    try:
        from scrapers import sstats
        raw = sstats.fetch_game(game_id)
        if not raw:
            return {"error": "Match not found"}
        game = raw.get("game", raw)
        home_name = (game.get("homeTeam") or {}).get("name", "?")
        away_name = (game.get("awayTeam") or {}).get("name", "?")
        league = (game.get("season") or {}).get("league", {}).get("name", "?")
        match_date = game.get("date", "")
        status_name = game.get("statusName", "Live")
        home_result = game.get("homeResult")
        away_result = game.get("awayResult")
        home_ht = game.get("homeHTResult")
        away_ht = game.get("awayHTResult")
        odds = game.get("odds", [])
        consensus = _consensus_from_odds(odds)
        statistics = game.get("statistics", {})
        events = game.get("events", [])
        home_lineup = game.get("homeTeam", {}).get("lineup", [])
        away_lineup = game.get("awayTeam", {}).get("lineup", [])
        injuries = []
        try: injuries = sstats.fetch_injuries(game_id)
        except: pass
        glicko = None
        try: glicko = sstats.fetch_glicko(game_id)
        except: pass
        text_summary = None
        try: text_summary = sstats.fetch_text_summary(game_id)
        except: pass
        last_stats = None
        try: last_stats = sstats.fetch_last_games_stats(game_id)
        except: pass
        ai_analysis = None
        try:
            import ai_analyzer
            sstats_enriched = {
                "consensus": consensus, "glicko": glicko,
                "text_summary": text_summary, "last_stats": last_stats,
                "injuries": injuries,
                "current_score": f"{home_result} - {away_result}" if home_result is not None and away_result is not None else None,
                "home_ht": home_ht, "away_ht": away_ht, "statistics": statistics,
            }
            ai_analysis = ai_analyzer.analyze_match(
                {"home_name": home_name, "away_name": away_name,
                 "league_slug": league, "date": match_date,
                 "season": dt.date.today().year if dt.date.today().month >= 7 else dt.date.today().year - 1,
                 "current_score": f"{home_result} - {away_result}" if home_result is not None and away_result is not None else None,
                 "home_ht": home_ht, "away_ht": away_ht,
                 "status": status_name, "statistics": statistics},
                {}, consensus if consensus else {},
                [], [], [],
                elo_home=None, elo_away=None,
                odds=consensus, sstats_data=sstats_enriched,
                model=config.DEFAULT_AI_MODEL, is_live=True,
            )
        except Exception as e:
            print(f"[live] AI analysis error: {e}")
        return {
            "game_id": game_id, "home": home_name, "away": away_name,
            "league": league, "date": match_date, "status": status_name,
            "score": f"{home_result} - {away_result}" if home_result is not None and away_result is not None else "",
            "home_result": home_result, "away_result": away_result,
            "home_ht": home_ht, "away_ht": away_ht,
            "odds": odds[:5], "consensus": consensus,
            "statistics": statistics, "events": events,
            "home_lineup": home_lineup[:11], "away_lineup": away_lineup[:11],
            "injuries": injuries, "glicko": glicko,
            "text_summary": text_summary, "last_stats": last_stats,
            "ai_analysis": ai_analysis,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/prematch/{game_id}")
def api_prematch_detail(game_id: int, model: str = None):
    try:
        from scrapers import sstats
        raw = sstats.fetch_game(game_id)
        if not raw:
            return {"error": "Match not found"}
        game = raw.get("game", raw)
        home_name = (game.get("homeTeam") or {}).get("name", "?")
        away_name = (game.get("awayTeam") or {}).get("name", "?")
        league = (game.get("season") or {}).get("league", {}).get("name", "?")
        match_date = game.get("date", "")
        odds = game.get("odds", [])
        consensus = _consensus_from_odds(odds)
        glicko = None
        try: glicko = sstats.fetch_glicko(game_id)
        except: pass
        text_summary = None
        try: text_summary = sstats.fetch_text_summary(game_id)
        except: pass
        injuries = []
        try: injuries = sstats.fetch_injuries(game_id)
        except: pass
        last_stats = None
        try: last_stats = sstats.fetch_last_games_stats(game_id)
        except: pass
        profits = None
        try: profits = sstats.fetch_profits(game_id)
        except: pass
        sstats_h2h = None
        try:
            home_id_sstats = (game.get("homeTeam") or {}).get("id")
            away_id_sstats = (game.get("awayTeam") or {}).get("id")
            if home_id_sstats and away_id_sstats:
                sstats_h2h = sstats.fetch_h2h(home_id_sstats, away_id_sstats)
        except: pass

        def _find_team(name):
            results = db.search_team_fuzzy(name, limit=5)
            if results:
                q = name.lower().strip()
                for r in results:
                    if r["name"].lower() == q:
                        return r
                for r in results:
                    if r.get("league_slug") in ("EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"):
                        return r
                return results[0]
            words = name.lower().split()
            if len(words) > 1:
                for w in words:
                    if len(w) > 3:
                        r2 = db.search_team_fuzzy(w, limit=3)
                        for r in r2:
                            if w in r["name"].lower():
                                return r
            return None

        home_team = _find_team(home_name)
        away_team = _find_team(away_name)
        home_id = home_team["id"] if home_team else None
        away_id = away_team["id"] if away_team else None

        ml_source, ml_result, features_dict = None, None, {}
        if _state.MODEL is not None and home_id and away_id:
            try:
                pred = predict_pair(home_id, away_id, db.get_team(home_id), db.get_team(away_id))
                ml_source = "model"
                ml_result = {"prediction": pred}
                features_dict = pred.get("features", {})
            except Exception:
                pass
        if not ml_result and consensus:
            ml_source = "sstats_odds"
            ml_result = {"prediction": {"probabilities": {
                "home_win": round(consensus["implied_home"] * 100, 1),
                "draw": round(consensus["implied_draw"] * 100, 1),
                "away_win": round(consensus["implied_away"] * 100, 1),
            }}}

        db_h2h, elo_home, elo_away = [], None, None
        injuries_home, injuries_away = [], []
        if home_id and away_id:
            db_h2h = db.head_to_head(home_id, away_id, limit=5)
            elo_home = db.get_team_elo(home_id)
            elo_away = db.get_team_elo(away_id)
            injuries_home = db.list_injuries(home_id)
            injuries_away = db.list_injuries(away_id)
        all_injuries = []
        if isinstance(injuries, list) and injuries:
            all_injuries = injuries
        elif injuries_home or injuries_away:
            all_injuries = [{"player": {"name": i["player_name"]}, "reason": i.get("reason", "")}
                           for i in (injuries_home + injuries_away)]

        prediction_data = ml_result.get("prediction", {}) if ml_result else {}
        prediction_probs = prediction_data.get("probabilities", prediction_data)
        return {
            "game_id": game_id, "home": home_name, "away": away_name,
            "league": league, "date": match_date,
            "odds": odds[:5], "consensus": consensus,
            "glicko": glicko, "injuries": all_injuries,
            "text_summary": text_summary, "last_stats": last_stats,
            "profits": profits, "h2h_sstats": sstats_h2h,
            "ml_prediction": {"probabilities": prediction_probs} if prediction_probs else None,
            "ml_source": ml_source, "ai_analysis": None,
            "features": features_dict,
            "h2h": [dict(m) for m in db_h2h[:5]] if db_h2h else [],
            "elo_home": elo_home, "elo_away": elo_away,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/prematch/{game_id}/analyze")
def api_prematch_analyze(game_id: int, model: str = "deepseek/deepseek-v4-flash"):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")
    JOB.reset("ai")

    def worker():
        try:
            from scrapers import sstats
            import ai_analyzer
            JOB.emit({"type": "info", "msg": "Загрузка данных матча..."})
            raw = sstats.fetch_game(game_id)
            if not raw:
                JOB.emit({"type": "error", "msg": "Матч не найден на sstats.net"})
                return
            game = raw.get("game", raw)
            home_name = (game.get("homeTeam") or {}).get("name", "?")
            away_name = (game.get("awayTeam") or {}).get("name", "?")
            JOB.emit({"type": "info", "msg": f"Анализирую: {home_name} vs {away_name}"})
            result = ai_analyzer.search_and_predict(home_name, away_name, model=model, progress_cb=JOB.emit, sstats_game_id=game_id)
            if result:
                JOB.result = result
                JOB.emit({"type": "result", "msg": "AI Анализ готов", "prediction": result})
            else:
                JOB.emit({"type": "error", "msg": "Не удалось проанализировать матч"})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = __import__("threading").Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "ai", "game_id": game_id, "job_id": JOB.job_id}


# ── Match detail (by team names) ────────────────────────────────────────────

@router.get("/match-detail")
def api_match_detail(home_name: str, away_name: str):
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
    if _state.MODEL is not None:
        try:
            prediction = predict_pair(home_id, away_id, home, away)
        except Exception:
            pass
    return {
        "home": home, "away": away,
        "recent_home": recent_home, "recent_away": recent_away,
        "h2h": h2h, "prediction": prediction,
    }
