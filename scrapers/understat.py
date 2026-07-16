"""
scrapers/understat.py — Understat API scraper for xG, schedule, results.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import time
import requests as _requests


UNDERSTAT_BASE = "https://understat.com"

LEAGUES: Dict[str, Dict[str, Any]] = {
    "EPL":        {"name": "Premier League",  "country": "England",  "understat": "EPL", "min_season": 2014},
    "La_liga":    {"name": "La Liga",         "country": "Spain",    "understat": "La_liga", "min_season": 2014},
    "Bundesliga": {"name": "Bundesliga",      "country": "Germany",  "understat": "Bundesliga", "min_season": 2014},
    "Serie_A":    {"name": "Serie A",         "country": "Italy",    "understat": "Serie_A", "min_season": 2014},
    "Ligue_1":    {"name": "Ligue 1",         "country": "France",   "understat": "Ligue_1", "min_season": 2014},
    "RFPL":       {"name": "Russian Premier", "country": "Russia",   "understat": "RFPL", "min_season": 2014},
}


def _fetch_batch(data_list: list) -> list:
    """Batch fetch with requests."""
    results = []
    for data in data_list:
        url = data["url"]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            r = _requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                results.append(r.json())
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


def fetch_understat_league(league_slug: str, season: int) -> Dict[str, Any]:
    """Fetch one league/season from Understat's JSON API."""
    if league_slug not in LEAGUES:
        raise ValueError(f"Unknown league: {league_slug}")

    slug = LEAGUES[league_slug]["understat"]
    url = f"{UNDERSTAT_BASE}/getLeagueData/{slug}/{season}"

    results = _fetch_batch([{"url": url}])
    payload = results[0] if results and results[0] else None

    if not payload:
        raise ConnectionError(f"Understat: no data for {league_slug}/{season}")

    return {
        "matches": payload.get("dates", []),
        "teams":   payload.get("teams", {}),
        "players": payload.get("players", []),
    }


def fetch_understat_parallel(tasks: List[Dict[str, Any]],
                             max_workers: int = 4,
                             pause: float = 0.3) -> List[Dict[str, Any]]:
    """Fetch multiple league/season combos in parallel via Botasaurus batch."""
    urls = []
    meta = []
    for t in tasks:
        slug = t["league_slug"]
        season = t["season"]
        if slug not in LEAGUES:
            meta.append(t)
            urls.append({"url": "__invalid__"})
            continue
        uslug = LEAGUES[slug]["understat"]
        urls.append({"url": f"{UNDERSTAT_BASE}/getLeagueData/{uslug}/{season}"})
        meta.append(t)

    results_raw = _fetch_batch(urls)
    results = []

    for i, t in enumerate(meta):
        slug = t["league_slug"]
        season = t["season"]
        payload = results_raw[i] if results_raw and i < len(results_raw) else None
        if payload:
            results.append({
                "league_slug": slug,
                "season": season,
                "data": {
                    "matches": payload.get("dates", []),
                    "teams":   payload.get("teams", {}),
                    "players": payload.get("players", []),
                }
            })
        else:
            results.append({
                "league_slug": slug,
                "season": season,
                "error": f"No data from Understat for {slug}/{season}"
            })

    return results


def fetch_all_leagues(seasons: List[int], pause_sec: float = 0.3) -> Dict[str, Any]:
    """Fetch every (league, season) combination."""
    tasks = [{"league_slug": s, "season": sz}
             for s in LEAGUES for sz in seasons]
    results = fetch_understat_parallel(tasks, max_workers=4, pause=pause_sec)
    out: Dict[str, Any] = {}
    for r in results:
        key = f"{r['league_slug']}/{r['season']}"
        if "error" in r:
            out[key] = {"matches": [], "teams": {}, "players": []}
        else:
            out[key] = r["data"]
    return out


def search_team(name: str, season: int = None) -> List[Dict[str, Any]]:
    """Search for a team by name across all leagues."""
    import datetime as _dt
    if season is None:
        today = _dt.date.today()
        season = today.year if today.month >= 7 else today.year - 1

    q = name.strip().lower()
    results = []
    for league_slug, meta in LEAGUES.items():
        try:
            payload = fetch_understat_league(league_slug, season)
        except Exception:
            continue
        for tid_str, team_obj in payload.get("teams", {}).items():
            team_name = str(team_obj.get("title", "")).lower()
            team_id = int(tid_str)
            if q in team_name or team_name in q:
                team_matches = [
                    m for m in payload.get("matches", [])
                    if int(m.get("h", {}).get("id", 0)) == team_id
                    or int(m.get("a", {}).get("id", 0)) == team_id
                ]
                results.append({
                    "team_id": team_id,
                    "team_name": team_obj.get("title", ""),
                    "league_slug": league_slug,
                    "league_name": meta["name"],
                    "season": season,
                    "matches": team_matches,
                    "team_data": team_obj,
                })
        time.sleep(0.3)
    return results
