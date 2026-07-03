"""
scrapers/thesportsdb.py — fetch worldwide soccer matches via TheSportsDB API.

TheSportsDB is a free public JSON API (no key required for the public tier):
    https://www.thesportsdb.com/api.php

Endpoints we use:
    /eventsday.php?d=YYYY-MM-DD&s=Soccer    — all soccer matches on a date
    /livescore.php?s=Soccer                  — live matches right now
    /eventsnext.php?id=<team_id>             — upcoming for a team
    /search_all_leagues.php?s=Soccer         — list of all soccer leagues

We normalize every match to the same shape as the old FlashScore output so
nothing downstream needs to change:
    {country, league, home, away, score_home, score_away, time, status}
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional


TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"
_UA = "Mozilla/5.0 (Football-AI; learning project) Python-urllib"


def _get(url: str) -> Optional[dict]:
    """Plain GET to TheSportsDB. No Cloudflare here — Botasaurus would be overkill."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _normalize_event(e: dict) -> Optional[Dict[str, str]]:
    home = e.get("strHomeTeam")
    away = e.get("strAwayTeam")
    if not home or not away:
        return None
    sh = e.get("intHomeScore")
    sa = e.get("intAwayScore")
    status_raw = (e.get("strStatus") or "").upper()
    # Map TheSportsDB statuses to our 3-bucket schema
    if status_raw in ("LIVE", "1H", "2H", "HT", "ET", "P", "IN_PLAY"):
        status = "live"
    elif status_raw in ("FT", "AET", "PEN", "FIN", "FINISHED") or (
        sh not in (None, "", "null") and sa not in (None, "", "null")
        and status_raw not in ("NS", "TBD", "")
    ):
        status = "finished"
    else:
        status = "scheduled"
    # Time string: prefer kick-off time of day, fall back to date
    time_str = e.get("strTime") or e.get("strTimeLocal") or ""
    if time_str and len(time_str) >= 5:
        time_str = time_str[:5]
    elif e.get("dateEvent"):
        time_str = str(e["dateEvent"])
    return {
        "country":   str(e.get("strCountry") or ""),
        "league":    str(e.get("strLeague") or ""),
        "home":      str(home),
        "away":      str(away),
        "score_home": str(sh) if sh not in (None, "") else "",
        "score_away": str(sa) if sa not in (None, "") else "",
        "time":      time_str,
        "status":    status,
    }


def fetch_events_for_date(date_iso: str) -> List[Dict[str, str]]:
    """All soccer events for a given date (YYYY-MM-DD)."""
    data = _get(f"{TSDB_BASE}/eventsday.php?d={date_iso}&s=Soccer")
    if not data or not data.get("events"):
        return []
    out = []
    for e in data["events"]:
        norm = _normalize_event(e)
        if norm:
            out.append(norm)
    return out


def fetch_today_matches() -> List[Dict[str, str]]:
    return fetch_events_for_date(dt.date.today().isoformat())


def fetch_yesterday_matches() -> List[Dict[str, str]]:
    y = dt.date.today() - dt.timedelta(days=1)
    return fetch_events_for_date(y.isoformat())


def fetch_tomorrow_matches() -> List[Dict[str, str]]:
    t = dt.date.today() + dt.timedelta(days=1)
    return fetch_events_for_date(t.isoformat())


def fetch_week_matches() -> List[Dict[str, str]]:
    """Yesterday + today + next 6 days — single 'wide net' fetch."""
    out: List[Dict[str, str]] = []
    for offset in range(-1, 7):
        d = (dt.date.today() + dt.timedelta(days=offset)).isoformat()
        out.extend(fetch_events_for_date(d))
    return out


def fetch_live_matches() -> List[Dict[str, str]]:
    data = _get(f"{TSDB_BASE}/livescore.php?s=Soccer")
    if not data:
        return []
    # Some responses key under "events", some under "livescore"
    items = data.get("events") or data.get("livescore") or []
    out = []
    for e in items:
        norm = _normalize_event(e)
        if norm:
            norm["status"] = "live"
            out.append(norm)
    return out


def fetch_all_soccer_leagues() -> List[Dict[str, str]]:
    """Catalog of every soccer league TheSportsDB knows about."""
    data = _get(f"{TSDB_BASE}/search_all_leagues.php?s=Soccer")
    if not data or not data.get("countries"):
        return []
    out = []
    for l in data["countries"]:
        out.append({
            "id": str(l.get("idLeague") or ""),
            "name": str(l.get("strLeague") or ""),
            "country": str(l.get("strCountry") or ""),
        })
    return out
