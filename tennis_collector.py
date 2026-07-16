"""
tennis_collector.py — Collect tennis data from Tennis API (RapidAPI).

Fetches: rankings (ATP/WTA), today's matches, live matches, H2H.
Stores everything in data/tennis.db via tennis_db.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import requests

import config
import tennis_db

logger = logging.getLogger("tennis_collector")

API_HOST = config.TENNIS_API_HOST
API_KEY = config.TENNIS_API_KEY
BASE_URL = f"https://{API_HOST}/api/tennis"

_session = requests.Session()
_session.headers.update({
    "X-RapidAPI-Host": API_HOST,
    "X-RapidAPI-Key": API_KEY,
})


def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = _session.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Tennis API {resp.status_code}: {endpoint}")
        return None
    except Exception as e:
        logger.error(f"Tennis API error: {e}")
        return None


def _emit(cb: Optional[Callable], event: dict):
    if cb:
        try:
            cb(event)
        except Exception:
            pass


# ── Rankings ─────────────────────────────────────────────────────────────────

def fetch_rankings(tour: str = "atp") -> list:
    """Fetch ATP or WTA rankings."""
    data = _get(f"rankings/{tour}")
    if not data:
        return []

    rankings = []
    for item in data.get("rankings", []):
        team = item.get("team", {})
        rankings.append({
            "player_id": team.get("id"),
            "player_name": team.get("name", "Unknown"),
            "ranking": item.get("ranking"),
            "ranking_points": item.get("points"),
        })
    return rankings


def collect_rankings(progress_cb=None) -> dict:
    """Collect both ATP and WTA rankings."""
    tennis_db.init_db()
    summary = {"atp": 0, "wta": 0}

    for tour in ["atp", "wta"]:
        _emit(progress_cb, {"type": "info", "msg": f"Загружаю рейтинги {tour.upper()}..."})
        rankings = fetch_rankings(tour)
        if rankings:
            with tennis_db.connect() as conn:
                # Upsert players
                for r in rankings:
                    if r["player_id"]:
                        tennis_db.upsert_player(conn, {
                            "id": r["player_id"],
                            "name": r["player_name"],
                            "ranking": r["ranking"],
                            "ranking_points": r["ranking_points"],
                        })
                tennis_db.save_rankings(conn, rankings, tour)
            summary[tour] = len(rankings)
            _emit(progress_cb, {"type": "success",
                                "msg": f"{tour.upper()}: {len(rankings)} игроков загружено"})
        else:
            _emit(progress_cb, {"type": "error", "msg": f"Не удалось загрузить {tour.upper()} рейтинг"})

        time.sleep(0.5)

    _emit(progress_cb, {"type": "done", "msg": "Рейтинги обновлены"})
    return summary


# ── Today's Matches ──────────────────────────────────────────────────────────

def _parse_event(event: dict) -> dict:
    """Parse a Tennis API event into our match format."""
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    status_obj = event.get("status", {})

    # Determine match status
    status_type = status_obj.get("type", "")
    if status_type == "inprogress":
        status = "live"
    elif status_type == "finished":
        status = "finished"
    else:
        status = "scheduled"

    # Score
    result = event.get("result", {})
    score = ""
    if result:
        score = result.get("displayScore", result.get("score", ""))

    return {
        "api_event_id": event.get("id"),
        "tournament_name": event.get("tournament", {}).get("name", ""),
        "tournament_id": event.get("tournament", {}).get("uniqueTournament", {}).get("id"),
        "round_name": event.get("roundInfo", {}).get("round", ""),
        "surface": event.get("groundType", ""),
        "date": event.get("startTimestamp", ""),
        "player1_id": home.get("id"),
        "player2_id": away.get("id"),
        "player1_name": home.get("name", "TBD"),
        "player2_name": away.get("name", "TBD"),
        "winner_id": event.get("winnerCode"),
        "score": score,
        "status": status,
        "odds_player1": event.get("homeOdds"),
        "odds_player2": event.get("awayOdds"),
    }


def fetch_day_events(day: int, month: int, year: int) -> list:
    """Fetch all events for a specific date."""
    data = _get(f"event/{day}/{month}/{year}")
    if not data:
        return []

    events = []
    sport = data.get("sportItem", {})
    for cat in sport.get("categories", []):
        for event in cat.get("events", []):
            events.append(_parse_event(event))
    return events


def collect_today(progress_cb=None) -> dict:
    """Collect today's tennis matches."""
    tennis_db.init_db()
    today = dt.date.today()
    _emit(progress_cb, {"type": "info",
                        "msg": f"Загружаю расписание на {today.strftime('%d.%m.%Y')}..."})

    events = fetch_day_events(today.day, today.month, today.year)
    added = 0

    with tennis_db.connect() as conn:
        for ev in events:
            if ev.get("api_event_id"):
                tennis_db.upsert_match(conn, ev)
                added += 1

    _emit(progress_cb, {"type": "success", "msg": f"Загружено {added} матчей на сегодня"})
    _emit(progress_cb, {"type": "done", "msg": "Расписание обновлено"})
    return {"total": added}


# ── Live Matches ─────────────────────────────────────────────────────────────

def collect_live(progress_cb=None) -> dict:
    """Refresh live matches."""
    tennis_db.init_db()
    _emit(progress_cb, {"type": "info", "msg": "Проверяю live-матчи..."})

    data = _get("events/live")
    if not data:
        _emit(progress_cb, {"type": "error", "msg": "Не удалось загрузить live-матчи"})
        return {"total": 0}

    events = []
    for ev in data.get("events", []):
        events.append(_parse_event(ev))

    updated = 0
    with tennis_db.connect() as conn:
        for ev in events:
            if ev.get("api_event_id"):
                tennis_db.upsert_match(conn, ev)
                updated += 1

    _emit(progress_cb, {"type": "success", "msg": f"Обновлено {updated} live-матчей"})
    _emit(progress_cb, {"type": "done", "msg": "Live обновлён"})
    return {"total": updated}


# ── Search ───────────────────────────────────────────────────────────────────

def search(query: str) -> list:
    """Search for players/tournaments."""
    data = _get(f"search/{query}")
    if not data:
        return []

    results = []
    for item in data.get("results", []):
        entity = item.get("entity", {})
        results.append({
            "type": item.get("type", "unknown"),
            "id": entity.get("id"),
            "name": entity.get("name", ""),
            "country": entity.get("country", {}).get("name", ""),
        })
    return results


# ── Combined collector ───────────────────────────────────────────────────────

def collect_all(progress_cb=None, cancel_event=None) -> dict:
    """Full collection: rankings + today + live."""
    tennis_db.init_db()
    summary = {"rankings": {}, "today": 0, "live": 0}

    if cancel_event and cancel_event.is_set():
        return summary

    summary["rankings"] = collect_rankings(progress_cb)

    if cancel_event and cancel_event.is_set():
        return summary

    time.sleep(1)

    summary["today"] = collect_today(progress_cb).get("total", 0)

    if cancel_event and cancel_event.is_set():
        return summary

    time.sleep(1)

    summary["live"] = collect_live(progress_cb).get("total", 0)

    with tennis_db.connect() as conn:
        tennis_db.set_meta(conn, "last_refresh", dt.datetime.now().isoformat(timespec="seconds"))

    return summary


# ── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def _cli_cb(ev):
        print(f"  [{ev.get('type', '?')}] {ev.get('msg', '')}")

    print("=== Tennis Data Collection ===")
    result = collect_all(progress_cb=_cli_cb)
    print(f"\nDone: {result}")
    stats = tennis_db.db_stats()
    print(f"DB stats: {stats}")
