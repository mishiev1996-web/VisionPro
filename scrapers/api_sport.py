"""
scrapers/api_sport.py — API-sport.ru scraper (api-sport.io).

Uses API key with 50 requests/day limit.
Fetches: matches, odds, statistics, leagues, teams.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "https://api.api-sport.ru/v2"
API_KEY = os.environ.get("API_SPORT_KEY", "")
if not API_KEY:
    _key_path = Path(__file__).parent.parent / "Апи" / "api_sport_key.txt"
    if _key_path.exists():
        API_KEY = _key_path.read_text().strip()

HEADERS = {"Authorization": API_KEY}
CACHE_DIR = Path(__file__).parent.parent / "data" / "api_sport_cache"
RATE_LIMIT_DELAY = 1.5  # seconds between requests (50/day = ~1 every 29 min)

_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _get(path: str, params: dict = None) -> Optional[dict]:
    """Make authenticated GET request."""
    _rate_limit()
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[api_sport] HTTP {r.status_code}: {path}")
            return None
    except Exception as e:
        print(f"[api_sport] Error: {e}")
        return None


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str, max_age_hours: int = 24) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age_hours * 3600:
        return None
    return json.loads(p.read_text())


def _cache_set(key: str, data: dict):
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False))


# ── Public API ────────────────────────────────────────────────────────────────

def get_matches(sport: str = "football", date: str = None,
                league_id: int = None, status: str = None,
                with_bk_odds: bool = False) -> List[dict]:
    """Get matches for a date or league.
    
    Args:
        sport: football, hockey, basketball, tennis, etc.
        date: YYYY-MM-DD format
        league_id: Tournament ID
        status: notstarted, inprogress, finished
        with_bk_odds: Include bookmaker odds (uses extra request)
    """
    cache_key = f"matches_{sport}_{date}_{league_id}_{status}_{with_bk_odds}"
    cached = _cache_get(cache_key, max_age_hours=1 if status != "finished" else 24)
    if cached:
        return cached

    params = {}
    if date:
        params["date"] = date
    if league_id:
        params["leagueId"] = league_id
    if status:
        params["status"] = status
    if with_bk_odds:
        params["with_bk_odds"] = "true"

    data = _get(f"/{sport}/matches", params)
    matches = (data or {}).get("data") or []
    if matches:
        _cache_set(cache_key, matches)
    return matches


def get_match(sport: str, match_id: int) -> Optional[dict]:
    """Get full match details by ID."""
    cache_key = f"match_{sport}_{match_id}"
    cached = _cache_get(cache_key, max_age_hours=2)
    if cached:
        return cached

    data = _get(f"/{sport}/matches/{match_id}")
    match = (data or {}).get("data")
    if match:
        _cache_set(cache_key, match)
    return match


def get_leagues(sport: str = "football") -> List[dict]:
    """Get all leagues for a sport."""
    cache_key = f"leagues_{sport}"
    cached = _cache_get(cache_key, max_age_hours=168)  # 1 week
    if cached:
        return cached

    data = _get(f"/{sport}/leagues")
    leagues = (data or {}).get("data") or []
    if leagues:
        _cache_set(cache_key, leagues)
    return leagues


def get_teams(sport: str = "football", league_id: int = None) -> List[dict]:
    """Get teams for a league."""
    cache_key = f"teams_{sport}_{league_id}"
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached:
        return cached

    params = {"leagueId": league_id} if league_id else {}
    data = _get(f"/{sport}/teams", params)
    teams = (data or {}).get("data") or []
    if teams:
        _cache_set(cache_key, teams)
    return teams


def get_standings(sport: str = "football", league_id: int = None,
                  season_id: int = None) -> List[dict]:
    """Get league standings/table."""
    cache_key = f"standings_{sport}_{league_id}_{season_id}"
    cached = _cache_get(cache_key, max_age_hours=24)
    if cached:
        return cached

    params = {}
    if league_id:
        params["leagueId"] = league_id
    if season_id:
        params["seasonId"] = season_id

    data = _get(f"/{sport}/standings", params)
    standings = (data or {}).get("data") or []
    if standings:
        _cache_set(cache_key, standings)
    return standings


def search(query: str, sport: str = "football") -> List[dict]:
    """Full-text search for teams, players, tournaments."""
    data = _get(f"/{sport}/search", {"q": query})
    return (data or {}).get("data") or []


# ── Football-specific ─────────────────────────────────────────────────────────

def get_football_matches_today(with_odds: bool = True) -> List[dict]:
    """Get today's football matches with odds if available."""
    return get_matches("football", date=None, status="notstarted",
                       with_bk_odds=with_odds)


def get_football_finished(date: str = None) -> List[dict]:
    """Get finished football matches."""
    return get_matches("football", date=date, status="finished")


# ── Tennis-specific ───────────────────────────────────────────────────────────

def get_tennis_matches_today() -> List[dict]:
    """Get today's tennis matches."""
    return get_matches("tennis", date=None)


def get_tennis_finished(date: str = None) -> List[dict]:
    """Get finished tennis matches with set details."""
    return get_matches("tennis", date=date, status="finished")
