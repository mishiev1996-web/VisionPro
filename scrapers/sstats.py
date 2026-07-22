"""
scrapers/sstats.py — wrapper around the sstats.net Football API.

Authentication via `apikey` header. Endpoints used:
    /Account/Info                  — sanity check
    /Leagues                       — full leagues + seasons catalog
    /Games/list?leagueId=X         — paginated league history
    /Games/list?date=YYYY-MM-DD    — all worldwide matches on a date
    /Games/list?from=YYYY-MM-DD&to=YYYY-MM-DD — date range
    /Games/{id}                    — single match detail
    /Games/glicko/{id}             — Glicko ratings for a match
    /Games/text-summary?id={id}    — pre-built text summary (already in Russian)
    /Odds/{id}                     — odds across 8 bookmakers + multiple markets
    /Teams/list?leagueId=X         — teams in a league

The API uses TLS fingerprinting; use Botasaurus @request for proper handshake.
"""
from __future__ import annotations

import os
import requests as _requests
from typing import Any, Dict, List, Optional


SSTATS_BASE = "https://api.sstats.net"
SSTATS_KEY = os.environ.get("SSTATS_API_KEY", "")
if not SSTATS_KEY:
    _key_path = os.path.join(os.path.dirname(__file__), "..", "Апи", "sstats_key.txt")
    if os.path.exists(_key_path):
        with open(_key_path, "r") as _f:
            SSTATS_KEY = _f.read().strip()

import threading as _threading
import time as _time
import functools
_sstats_lock = _threading.Lock()
_sstats_last_call = 0.0

# ── TTL Cache ────────────────────────────────────────────────────────────────

_SSTATS_CACHE_TTL = 300  # 5 minutes — match data doesn't change faster
_sstats_cache: Dict[str, tuple] = {}  # key → (timestamp, value)
_sstats_cache_lock = _threading.Lock()


def _cache_key(func_name: str, args: tuple, kwargs: tuple) -> str:
    """Build a cache key from function name and arguments."""
    return f"{func_name}:{args}:{kwargs}"


def cached_sstats(ttl: int = _SSTATS_CACHE_TTL):
    """Decorator: cache sstats function results for `ttl` seconds.

    Thread-safe. Cache is per-function+args. Auto-expires entries.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _cache_key(fn.__name__, args, tuple(sorted(kwargs.items())))
            now = _time.monotonic()
            with _sstats_cache_lock:
                if key in _sstats_cache:
                    ts, val = _sstats_cache[key]
                    if now - ts < ttl:
                        return val
            # Cache miss — call the real function
            result = fn(*args, **kwargs)
            with _sstats_cache_lock:
                _sstats_cache[key] = (now, result)
                # Evict expired entries periodically (max 500 entries)
                if len(_sstats_cache) > 500:
                    expired = [k for k, (ts, _) in _sstats_cache.items()
                               if now - ts > ttl]
                    for k in expired:
                        del _sstats_cache[k]
            return result
        return wrapper
    return decorator


def _rate_limit():
    """Ensure minimum 2 seconds between sstats API calls."""
    global _sstats_last_call
    with _sstats_lock:
        now = __import__("time").monotonic()
        elapsed = now - _sstats_last_call
        if elapsed < 2.0:
            __import__("time").sleep(2.0 - elapsed)
        _sstats_last_call = __import__("time").monotonic()


def _fetch_batch(data_list: list) -> list:
    """Generic batch GET. Each item must contain 'url'."""
    headers = {"apikey": SSTATS_KEY, "User-Agent": "Mozilla/5.0 (Football-AI)"}
    results = []
    for data in data_list:
        try:
            r = _requests.get(data["url"], headers=headers, timeout=25)
            if r.status_code == 200:
                results.append(r.json())
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


def _fetch_one(path: str) -> Optional[dict]:
    """Single GET — convenience wrapper."""
    _rate_limit()
    results = _fetch_batch([{"url": SSTATS_BASE + path}])
    if results and results[0]:
        return results[0]
    return None


# ── Public endpoints ─────────────────────────────────────────────────────────

def account_info() -> Optional[dict]:
    data = _fetch_one("/Account/Info")
    return (data or {}).get("data")


def fetch_leagues() -> List[dict]:
    """Full catalog — ~1400 leagues × all their seasons."""
    data = _fetch_one("/Leagues")
    return (data or {}).get("data") or []


@cached_sstats()
def fetch_games_by_date(date_iso: str) -> List[dict]:
    """All worldwide matches on a single date (YYYY-MM-DD)."""
    data = _fetch_one(f"/Games/list?date={date_iso}")
    return (data or {}).get("data") or []


def fetch_upcoming_by_team(team_id: int) -> List[dict]:
    """Upcoming matches for a specific team (status 1,2 + date > now)."""
    data = _fetch_one(f"/Games/list?upcoming=true&team={team_id}")
    return (data or {}).get("data") or []


@cached_sstats()
def fetch_upcoming_all() -> List[dict]:
    """All upcoming worldwide matches — upcoming filter + today's remaining."""
    # 1. Upcoming filter (may miss some due to limit=1000)
    upcoming = _fetch_one("/Games/list?upcoming=true&limit=1000") or {}
    games = (upcoming.get("data") or [])
    seen_ids = {g["id"] for g in games}

    # 2. Also fetch today's Not Started matches (catches what upcoming misses)
    today = __import__("datetime").date.today().isoformat()
    today_data = _fetch_one(f"/Games/list?date={today}&status=2") or {}
    for g in (today_data.get("data") or []):
        if g.get("id") not in seen_ids:
            games.append(g)
            seen_ids.add(g["id"])

    return games


@cached_sstats()
def fetch_live_matches() -> List[dict]:
    """All currently live matches worldwide."""
    data = _fetch_one("/Games/list?live=true&limit=1000") or {}
    return (data or {}).get("data") or []


@cached_sstats()
def fetch_h2h(team1_id: int, team2_id: int) -> List[dict]:
    """Head-to-head: all ended matches between two teams."""
    data = _fetch_one(f"/Games/list?ended=true&bothTeams={team1_id},{team2_id}")
    return (data or {}).get("data") or []


def fetch_games_by_league(league_id: int, page: int = 0) -> List[dict]:
    """Paginated league history (1000 per page)."""
    suffix = f"&page={page}" if page else ""
    data = _fetch_one(f"/Games/list?leagueId={league_id}{suffix}")
    return (data or {}).get("data") or []


@cached_sstats()
def fetch_game(game_id: int) -> Optional[dict]:
    data = _fetch_one(f"/Games/{game_id}")
    return (data or {}).get("data")


@cached_sstats()
def fetch_glicko(game_id: int) -> Optional[dict]:
    data = _fetch_one(f"/Games/glicko/{game_id}")
    return (data or {}).get("data")


@cached_sstats()
def fetch_text_summary(game_id: int) -> Optional[str]:
    """Pre-built Russian text summary with bookmaker predictions."""
    _rate_limit()
    import urllib.request
    headers = {"apikey": SSTATS_KEY, "User-Agent": "Mozilla/5.0 (Football-AI)"}
    url = f"{SSTATS_BASE}/Games/text-summary?id={game_id}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            text = resp.read().decode("utf-8")
            return text if text.strip() else None
    except Exception:
        return None


@cached_sstats()
def fetch_odds(game_id: int) -> List[dict]:
    """All bookmakers' odds + market types for a game."""
    data = _fetch_one(f"/Odds/{game_id}")
    return (data or {}).get("data") or []


def fetch_teams_in_league(league_id: int) -> List[dict]:
    data = _fetch_one(f"/Teams/list?leagueId={league_id}")
    return (data or {}).get("data") or []


def search_team_by_name(team_name: str, limit: int = 5) -> List[dict]:
    """Search for a team by name using /Games/query.
    
    Returns list of matches where the team appears, with team info.
    """
    _rate_limit()
    # Sanitize team_name to prevent injection into the Condition string
    # Escape: single/double quotes, LIKE wildcards (% and _), backslash
    safe_name = (team_name
                 .replace("\\", "\\\\")
                 .replace("'", "''")
                 .replace('"', '""')
                 .replace("%", "\\%")
                 .replace("_", "\\_"))
    body = {
        "Condition": f"HomeTeamName LIKE '%{safe_name}%' OR AwayTeamName LIKE '%{safe_name}%'",
        "Fields": ["Id", "Date", "HomeTeamName", "HomeTeamId", "AwayTeamName", "AwayTeamId",
                   "ScoreHomeFT", "ScoreAwayFT", "Status", "LeagueId", "LeagueName"],
        "Order": "Date DESC",
        "Limit": limit
    }
    headers = {"apikey": SSTATS_KEY, "Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Football-AI)"}
    try:
        r = _requests.post(f"{SSTATS_BASE}/Games/query",
                          headers=headers,
                          json=body, timeout=10)
        if r.status_code == 200:
            return (r.json() or {}).get("data") or []
    except Exception:
        pass
    return []


def search_team_matches(team_name: str, limit: int = 10) -> List[dict]:
    """Search for recent matches involving a team by name.
    
    Returns list of match dicts with team IDs and names.
    """
    return search_team_by_name(team_name, limit)


@cached_sstats()
def fetch_last_games_stats(game_id: int, limit: int = 25,
                           same_league: bool = False,
                           same_season: bool = False,
                           home_away: bool = False) -> Optional[dict]:
    """Pre-match averaged statistics for both teams over their last N games.

    Returns dict with 'home' and 'away' keys, each containing:
    - avg goals scored/conceded (avgScore, avgConceded)
    - avg shots, corners, cards
    - form (W/D/L counts)
    - avg odds (home/draw/away)
    """
    params = f"gameId={game_id}&limit={limit}"
    if same_league:
        params += "&sameLeague=true"
    if same_season:
        params += "&sameSeason=true"
    if home_away:
        params += "&homeAway=true"
    data = _fetch_one(f"/Games/last-games-stats?{params}")
    # API returns data directly, not under 'data' key
    return data


@cached_sstats()
def fetch_injuries(game_id: int) -> List[dict]:
    """Players unavailable due to injury for a specific match."""
    data = _fetch_one(f"/Games/injuries?gameId={game_id}")
    return (data or {}).get("data") or []


def fetch_season_table(league_id: int, year: int = None,
                       limit: int = 1000) -> Optional[dict]:
    """Rating/league table with over/under stats."""
    params = f"league={league_id}&limit={limit}"
    if year:
        params += f"&year={year}"
    data = _fetch_one(f"/Games/season-table?{params}")
    return (data or {}).get("data")


@cached_sstats()
def fetch_profits(game_id: int, limit: int = 25,
                  this_league: bool = False,
                  home_away: bool = False) -> Optional[dict]:
    """Historical betting profitability analysis by bet type."""
    params = f"gameId={game_id}&limit={limit}"
    if this_league:
        params += "&thisLeague=true"
    if home_away:
        params += "&homeAway=true"
    data = _fetch_one(f"/Games/profits?{params}")
    return (data or {}).get("data")


def fetch_query(condition: str, fields: List[str] = None,
                order: str = "Date DESC", fmt: str = "json") -> list:
    """Advanced SQL-like query for bulk data extraction.

    Example:
        fetch_query(
            condition="LeagueId = 330 AND Year = 2025 AND Winner1 > 1.1",
            fields=["Id", "Date", "HomeTeamName", "AwayTeamName",
                    "ScoreHomeFT", "ScoreAwayFT", "Winner1", "WinnerX", "Winner2"],
        )
    """
    body = {
        "Condition": condition,
        "Fields": fields or ["Id", "Date", "HomeTeamName", "AwayTeamName",
                             "ScoreHomeFT", "ScoreAwayFT"],
        "Order": order,
        "Format": fmt,
    }
    _rate_limit()
    import urllib.request
    import json as _json
    headers = {"apikey": SSTATS_KEY, "Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Football-AI)"}
    req = urllib.request.Request(
        f"{SSTATS_BASE}/Games/query",
        data=_json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
            return (result or {}).get("data") or []
    except urllib.error.HTTPError as e:
        if e.code == 429:
            import time, sys
            print(f"  [sstats.query] 429 Too Many Requests — waiting 30s...", file=sys.stderr)
            time.sleep(30)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = _json.loads(resp.read())
                    return (result or {}).get("data") or []
            except Exception:
                return []
        return []
    except Exception as e:
        return []


# ── Higher-level helpers ─────────────────────────────────────────────────────

def consensus_odds(odds_blocks: List[dict]) -> Optional[Dict[str, float]]:
    """Average 1X2 odds across bookmakers + compute implied probabilities.

    `odds_blocks` is the list returned by fetch_odds(). Each entry has
    bookmakerName + odds[] with markets. We want marketName='Match Winner'.
    """
    h_vals, d_vals, a_vals = [], [], []
    for bm in odds_blocks:
        for m in (bm.get("odds") or []):
            if m.get("marketName") not in ("Match Winner", "1X2"):
                continue
            for o in (m.get("odds") or []):
                name = (o.get("name") or "").lower()
                v = o.get("value")
                if v is None or v <= 1.0:
                    continue
                if name == "home": h_vals.append(float(v))
                elif name == "draw": d_vals.append(float(v))
                elif name == "away": a_vals.append(float(v))
    if not (h_vals and d_vals and a_vals):
        return None
    avg_h = sum(h_vals) / len(h_vals)
    avg_d = sum(d_vals) / len(d_vals)
    avg_a = sum(a_vals) / len(a_vals)
    # Convert to implied probabilities (normalised, removes bookmaker margin)
    inv = [1/avg_h, 1/avg_d, 1/avg_a]
    s = sum(inv)
    return {
        "avg_home_odds": round(avg_h, 3),
        "avg_draw_odds": round(avg_d, 3),
        "avg_away_odds": round(avg_a, 3),
        "implied_h": round(inv[0]/s, 4),
        "implied_d": round(inv[1]/s, 4),
        "implied_a": round(inv[2]/s, 4),
        "bookmaker_count": len(h_vals),
        "overround_pct": round((s - 1) * 100, 2),   # bookmaker margin (vig)
    }


def consensus_over_under(odds_blocks: List[dict]) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute Over/Under implied probabilities for each line from bookmaker consensus.

    Returns dict like {"2.5": {"over": 0.58, "under": 0.42, "avg_over_odds": 1.72, ...}, ...}
    or None if no Goals Over/Under data found.
    """
    import re
    if not odds_blocks:
        return None
    # Group by line: {line_str: {"over": [odds], "under": [odds]}}
    by_line: Dict[str, Dict[str, List[float]]] = {}
    for bm in odds_blocks:
        for m in (bm.get("odds") or []):
            if m.get("marketName") != "Goals Over/Under":
                continue
            for o in (m.get("odds") or []):
                name = (o.get("name") or "").strip()
                v = o.get("value")
                if v is None or v <= 1.0:
                    continue
                # Parse "Over 2.5" or "Under 1.5"
                match = re.match(r"(Over|Under)\s+([\d.]+)", name, re.IGNORECASE)
                if not match:
                    continue
                side = match.group(1).lower()
                line = match.group(2)
                by_line.setdefault(line, {"over": [], "under": []})
                by_line[line][side].append(float(v))

    if not by_line:
        return None

    result = {}
    for line, vals in sorted(by_line.items(), key=lambda x: float(x[0])):
        over_odds = vals["over"]
        under_odds = vals["under"]
        if not over_odds or not under_odds:
            continue
        avg_over = sum(over_odds) / len(over_odds)
        avg_under = sum(under_odds) / len(under_odds)
        inv_over = 1 / avg_over
        inv_under = 1 / avg_under
        s = inv_over + inv_under
        result[line] = {
            "over": round(inv_over / s, 4),
            "under": round(inv_under / s, 4),
            "avg_over_odds": round(avg_over, 3),
            "avg_under_odds": round(avg_under, 3),
            "bookmaker_count": max(len(over_odds), len(under_odds)),
        }

    return result if result else None


def consensus_btts(odds_blocks: List[dict]) -> Optional[Dict[str, float]]:
    """Compute BTTS (Both Teams To Score) implied probabilities from bookmaker consensus.

    Returns dict like {"yes": 0.65, "no": 0.35, "avg_yes_odds": 1.54, ...}
    or None if no BTTS data found.
    """
    if not odds_blocks:
        return None
    yes_vals, no_vals = [], []
    for bm in odds_blocks:
        for m in (bm.get("odds") or []):
            if m.get("marketName") != "Both Teams Score":
                continue
            for o in (m.get("odds") or []):
                name = (o.get("name") or "").strip().lower()
                v = o.get("value")
                if v is None or v <= 1.0:
                    continue
                if name == "yes":
                    yes_vals.append(float(v))
                elif name == "no":
                    no_vals.append(float(v))
    if not yes_vals or not no_vals:
        return None
    avg_yes = sum(yes_vals) / len(yes_vals)
    avg_no = sum(no_vals) / len(no_vals)
    inv_yes = 1 / avg_yes
    inv_no = 1 / avg_no
    s = inv_yes + inv_no
    return {
        "yes": round(inv_yes / s, 4),
        "no": round(inv_no / s, 4),
        "avg_yes_odds": round(avg_yes, 3),
        "avg_no_odds": round(avg_no, 3),
        "bookmaker_count": max(len(yes_vals), len(no_vals)),
    }


def market_dispersion(odds_blocks: List[dict]) -> Optional[Dict[str, float]]:
    """How much bookmakers disagree about the outcome. High dispersion = harder match."""
    h_vals, d_vals, a_vals = [], [], []
    for bm in odds_blocks:
        for m in (bm.get("odds") or []):
            if m.get("marketName") not in ("Match Winner", "1X2"):
                continue
            for o in (m.get("odds") or []):
                v = o.get("value")
                if v is None or v <= 1.0:
                    continue
                name = (o.get("name") or "").lower()
                if name == "home": h_vals.append(float(v))
                elif name == "draw": d_vals.append(float(v))
                elif name == "away": a_vals.append(float(v))
    if len(h_vals) < 2:
        return None
    def _var(xs: List[float]) -> float:
        if not xs: return 0.0
        mean = sum(xs) / len(xs)
        return sum((x - mean) ** 2 for x in xs) / len(xs)
    return {
        "home_odds_var": round(_var(h_vals), 4),
        "draw_odds_var": round(_var(d_vals), 4),
        "away_odds_var": round(_var(a_vals), 4),
    }

# ── Live odds changes (line movement) ────────────────────────────────────────

def fetch_live_odds_changes(game_id: int) -> Optional[List[dict]]:
    """Fetch live odds changes (line movement) for a game.
    
    Returns list of change records: {market_id, market_name, outcome_id,
    outcome_name, elapsed_seconds, created_time, value}.
    Only available for live/finished matches, NOT upcoming.
    """
    data = _fetch_one(f"/Odds/live-changes/{game_id}")
    if not data or not isinstance(data.get("data"), list):
        return None
    
    changes = []
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    
    for market in data["data"]:
        market_id = market.get("marketId")
        market_name = market.get("marketName", "")
        for outcome in market.get("outcomes", []):
            outcome_id = outcome.get("outcomeId")
            outcome_name = outcome.get("outcomeName", "")
            for change in outcome.get("changes", []):
                changes.append({
                    "game_id": game_id,
                    "market_id": market_id,
                    "market_name": market_name,
                    "outcome_id": outcome_id,
                    "outcome_name": outcome_name,
                    "elapsed_seconds": change.get("elapsedSeconds", 0),
                    "created_time": change.get("createdTime", ""),
                    "value": change.get("value"),
                    "collected_at": now,
                })
    
    return changes if changes else None


def save_live_odds_changes(changes: List[dict]) -> int:
    """Save live odds changes to DB (append-only, dedup via UNIQUE constraint)."""
    if not changes:
        return 0
    import db
    saved = 0
    with db.connect() as conn:
        for c in changes:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO sstats_live_odds_changes 
                    (game_id, market_id, market_name, outcome_id, outcome_name,
                     elapsed_seconds, created_time, value, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (c["game_id"], c["market_id"], c["market_name"],
                     c["outcome_id"], c["outcome_name"], c["elapsed_seconds"],
                     c["created_time"], c["value"], c["collected_at"])
                )
                saved += 1
            except Exception:
                pass
    return saved


def get_live_odds_features(game_id: int, window_minutes: int = 10) -> Optional[dict]:
    """Compute line movement features for a live match.
    
    Returns dict with:
    - odds_movement_1x2: dict with home/draw/away implied prob change over window
    - odds_direction: 'up'/'down'/'stable' for main favorite
    - odds_volatility: number of changes per minute
    """
    import db
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT outcome_name, elapsed_seconds, value 
            FROM sstats_live_odds_changes 
            WHERE game_id = ? AND market_name IN ('Match Winner', '1X2', 'Fulltime Result')
            ORDER BY elapsed_seconds""",
            (game_id,)
        ).fetchall()
    
    if not rows or len(rows) < 6:
        return None
    
    # Group by outcome
    outcome_history = {}
    for name, elapsed, value in rows:
        if name not in outcome_history:
            outcome_history[name] = []
        outcome_history[name].append((elapsed, value))
    
    # Get latest and earliest values for each outcome
    features = {}
    for name in ["Home", "Draw", "Away"]:
        history = outcome_history.get(name, [])
        if len(history) >= 2:
            latest_val = history[-1][1]
            earliest_val = history[0][1]
            # Convert odds to implied probability
            latest_imp = 1.0 / latest_val if latest_val > 1 else 0.5
            earliest_imp = 1.0 / earliest_val if earliest_val > 1 else 0.5
            features[f"{name.lower()}_odds_change"] = round(latest_val - earliest_val, 3)
            features[f"{name.lower()}_implied_change"] = round(latest_imp - earliest_imp, 3)
    
    # Direction: is favorite strengthening or weakening?
    if "home_implied_change" in features:
        home_change = features["home_implied_change"]
        if home_change > 0.02:
            features["odds_direction"] = "home_strengthening"
        elif home_change < -0.02:
            features["odds_direction"] = "home_weakening"
        else:
            features["odds_direction"] = "stable"
    
    # Volatility: changes per minute
    total_changes = len(rows)
    if rows:
        total_minutes = max(1, (rows[-1][1] - rows[0][1]) / 60)
        features["odds_volatility"] = round(total_changes / total_minutes, 2)
    
    return features if features else None