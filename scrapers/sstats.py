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
from typing import Any, Dict, List, Optional

from botasaurus.request import request, Request


SSTATS_BASE = "https://api.sstats.net"
# Key embedded — user's personal account, project is for personal use only.
# Override via SSTATS_API_KEY env var if needed.
SSTATS_KEY = os.environ.get("SSTATS_API_KEY", "")
if not SSTATS_KEY:
    _key_path = os.path.join(os.path.dirname(__file__), "..", "Апи", "sstats_key.txt")
    if os.path.exists(_key_path):
        with open(_key_path, "r") as _f:
            SSTATS_KEY = _f.read().strip()

# Global rate limiter: max 1 request per 2 seconds to avoid 429
import threading as _threading
_sstats_lock = _threading.Lock()
_sstats_last_call = 0.0


def _rate_limit():
    """Ensure minimum 2 seconds between sstats API calls."""
    global _sstats_last_call
    with _sstats_lock:
        now = __import__("time").monotonic()
        elapsed = now - _sstats_last_call
        if elapsed < 2.0:
            __import__("time").sleep(2.0 - elapsed)
        _sstats_last_call = __import__("time").monotonic()


@request(cache=False, output=None, create_error_logs=False, max_retry=2,
         raise_exception=False, parallel=8)
def _fetch_batch(req: Request, data: dict) -> Optional[dict]:
    """Generic batch GET. `data` must contain 'url' (and optionally other meta)."""
    headers = {"apikey": SSTATS_KEY, "User-Agent": "Mozilla/5.0 (Football-AI)"}
    try:
        r = req.get(data["url"], headers=headers, timeout=25)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


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


def fetch_games_by_date(date_iso: str) -> List[dict]:
    """All worldwide matches on a single date (YYYY-MM-DD)."""
    data = _fetch_one(f"/Games/list?date={date_iso}")
    return (data or {}).get("data") or []


def fetch_games_by_league(league_id: int, page: int = 0) -> List[dict]:
    """Paginated league history (1000 per page)."""
    suffix = f"&page={page}" if page else ""
    data = _fetch_one(f"/Games/list?leagueId={league_id}{suffix}")
    return (data or {}).get("data") or []


def fetch_game(game_id: int) -> Optional[dict]:
    data = _fetch_one(f"/Games/{game_id}")
    return (data or {}).get("data")


def fetch_glicko(game_id: int) -> Optional[dict]:
    data = _fetch_one(f"/Games/glicko/{game_id}")
    return (data or {}).get("data")


def fetch_text_summary(game_id: int) -> Optional[str]:
    """Pre-built Russian text summary with bookmaker predictions."""
    results = _fetch_batch([{"url": f"{SSTATS_BASE}/Games/text-summary?id={game_id}"}])
    if results and results[0]:
        # /Games/text-summary returns plain text, not JSON
        return results[0] if isinstance(results[0], str) else None
    return None


def fetch_odds(game_id: int) -> List[dict]:
    """All bookmakers' odds + market types for a game."""
    data = _fetch_one(f"/Odds/{game_id}")
    return (data or {}).get("data") or []


def fetch_teams_in_league(league_id: int) -> List[dict]:
    data = _fetch_one(f"/Teams/list?leagueId={league_id}")
    return (data or {}).get("data") or []


def fetch_last_games_stats(game_id: int, limit: int = 25,
                           same_league: bool = False,
                           same_season: bool = False,
                           home_away: bool = False) -> Optional[dict]:
    """Pre-match averaged statistics for both teams over their last N games.

    Returns dict with 'home' and 'away' keys, each containing:
    - avg goals scored/conceded
    - avg xG and xG against
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
    return (data or {}).get("data")


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
