"""
tennis_live.py — Live tennis match context from FlashScore + Tennis API.

Provides real-time scores, set-by-set breakdown, and momentum
for live tennis predictions.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests

import config
import tennis_db

logger = logging.getLogger("tennis_live")


# ── Tennis API live ──────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """Call Tennis API (RapidAPI)."""
    headers = {
        "X-RapidAPI-Host": config.TENNIS_API_HOST,
        "X-RapidAPI-Key": config.TENNIS_API_KEY,
    }
    try:
        resp = requests.get(
            f"https://{config.TENNIS_API_HOST}/api/tennis/{endpoint}",
            headers=headers, params=params, timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Tennis API error: {e}")
    return None


def fetch_live_from_api() -> List[Dict]:
    """Fetch live matches from Tennis API."""
    data = _api_get("events/live")
    if not data:
        return []

    matches = []
    for ev in (data.get("events") or []):
        match = _parse_api_event(ev)
        if match:
            matches.append(match)
    return matches


def _parse_api_event(ev: dict) -> Optional[Dict]:
    """Parse a Tennis API event into a unified match dict."""
    home = ev.get("homeTeam") or {}
    away = ev.get("awayTeam") or {}

    home_name = home.get("name", "")
    away_name = away.get("name", "")
    if not home_name or not away_name:
        return None

    # Extract current score
    home_score = ev.get("homeScore") or {}
    away_score = ev.get("awayScore") or {}

    # Current set scores
    current_sets = []
    for i in range(1, 6):
        hs = home_score.get(f"period{i}")
        as_ = away_score.get(f"period{i}")
        if hs is not None and as_ is not None:
            current_sets.append((int(hs), int(as_)))

    # Current game score in current set
    home_current = home_score.get("current", 0)
    away_current = away_score.get("current", 0)

    # Total sets won
    home_sets = sum(1 for h, a in current_sets if h > a)
    away_sets = sum(1 for h, a in current_sets if a > h)

    # Match status
    status_type = ev.get("type", "")
    status_code = ev.get("status", {})
    if isinstance(status_code, dict):
        status_code = status_code.get("code", "")

    # Tournament info
    tournament = ev.get("tournament") or {}
    tournament_name = tournament.get("name", "")
    tournament_round = tournament.get("round", "")

    # Surface detection
    surface = _detect_surface(tournament_name)

    return {
        "player1": home_name,
        "player2": away_name,
        "player1_id": home.get("id"),
        "player2_id": away.get("id"),
        "tournament": tournament_name,
        "round": tournament_round,
        "surface": surface,
        "status": "live",
        "current_sets": current_sets,
        "home_sets": home_sets,
        "away_sets": away_sets,
        "current_game": f"{home_current}-{away_current}",
        "score_display": _format_score(current_sets, home_current, away_current),
        "is_live": True,
        "source": "tennis_api",
    }


def _format_score(sets: list, current_home: int, current_away: int) -> str:
    """Format score like '6-4 3-2' with current game."""
    parts = []
    for h, a in sets:
        parts.append(f"{h}-{a}")
    if current_home or current_away:
        parts.append(f"{current_home}-{current_away}")
    return " ".join(parts) if parts else "0-0"


def _detect_surface(tournament_name: str) -> str:
    name = tournament_name.lower()
    if "wimbledon" in name:
        return "Grass"
    if "roland garros" in name or "french open" in name:
        return "Clay"
    if any(x in name for x in ["australian open", "us open", "indian wells", "miami"]):
        return "Hard"
    if any(x in name for x in ["monte carlo", "rome", "madrid", "barcelona"]):
        return "Clay"
    return "Hard"


# ── FlashScore live (fallback) ──────────────────────────────────────────────

def fetch_live_from_flashscore() -> List[Dict]:
    """Fetch live matches from FlashScore (headless browser)."""
    try:
        from scrapers.tennis_flashscore import fetch_tennis_live
        raw = fetch_tennis_live()
        matches = []
        for m in raw:
            # Parse score from string like "6-4 3-2 4-3"
            score_str = m.get("score", "")
            sets, current = _parse_score_string(score_str)
            home_sets = sum(1 for h, a in sets if h > a)
            away_sets = sum(1 for h, a in sets if a > h)

            matches.append({
                "player1": m.get("home", ""),
                "player2": m.get("away", ""),
                "tournament": m.get("tournament", ""),
                "surface": _detect_surface(m.get("tournament", "")),
                "status": "live",
                "current_sets": sets,
                "home_sets": home_sets,
                "away_sets": away_sets,
                "current_game": f"{current[0]}-{current[1]}" if current else "0-0",
                "score_display": score_str,
                "is_live": True,
                "source": "flashscore",
            })
        return matches
    except Exception as e:
        logger.warning(f"FlashScore tennis live error: {e}")
        return []


def _parse_score_string(score: str) -> tuple:
    """Parse score string into (sets_list, current_game_tuple)."""
    sets = []
    current = (0, 0)
    parts = score.replace(",", " ").split()
    for part in parts:
        if "-" in part:
            try:
                h, a = part.split("-")
                h, a = int(h), int(a)
                # If both <= 7 and it's not a tiebreak continuation, it's a set
                if h <= 7 and a <= 7:
                    sets.append((h, a))
                else:
                    current = (h, a)
            except ValueError:
                continue
    return sets, current


# ── Combined live fetch ─────────────────────────────────────────────────────

def fetch_live_context(player1_name: str, player2_name: str) -> Optional[Dict]:
    """Fetch live context for a specific match.

    Returns live match data if the match is currently live,
    including current score, sets, and momentum.
    """
    # Try Tennis API first
    live_matches = fetch_live_from_api()

    # Fallback to FlashScore
    if not live_matches:
        live_matches = fetch_live_from_flashscore()

    # Search for matching live match
    p1_lower = player1_name.lower().strip()
    p2_lower = player2_name.lower().strip()

    for m in live_matches:
        m_p1 = m["player1"].lower().strip()
        m_p2 = m["player2"].lower().strip()

        # Match by name substring (handles "J. Sinner" vs "Jannik Sinner")
        p1_match = (p1_lower in m_p1 or m_p1 in p1_lower or
                    any(w in m_p1 for w in p1_lower.split() if len(w) > 2))
        p2_match = (p2_lower in m_p2 or m_p2 in p2_lower or
                    any(w in m_p2 for w in p2_lower.split() if len(w) > 2))

        # Also check reverse order
        p1_match_rev = (p1_lower in m_p2 or m_p2 in p1_lower or
                        any(w in m_p2 for w in p1_lower.split() if len(w) > 2))
        p2_match_rev = (p2_lower in m_p1 or m_p1 in p2_lower or
                        any(w in m_p1 for w in p2_lower.split() if len(w) > 2))

        if (p1_match and p2_match) or (p1_match_rev and p2_match_rev):
            # If reversed, flip the match data
            if p1_match_rev and p2_match_rev:
                m = _flip_match(m)
            return m

    return None


def _flip_match(m: dict) -> dict:
    """Swap player1 and player2 in match data."""
    return {
        **m,
        "player1": m["player2"],
        "player2": m["player1"],
        "player1_id": m.get("player2_id"),
        "player2_id": m.get("player1_id"),
        "home_sets": m.get("away_sets", 0),
        "away_sets": m.get("home_sets", 0),
        "current_game": _flip_game_score(m.get("current_game", "0-0")),
        "current_sets": [(a, h) for h, a in m.get("current_sets", [])],
    }


def _flip_game_score(score: str) -> str:
    """Flip game score '4-3' → '3-4'."""
    parts = score.split("-")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0]}"
    return score


# ── Build live context string for LLM ───────────────────────────────────────

def build_live_context(live_data: dict, ml_prediction: dict = None) -> str:
    """Build a context string for LLM analysis with live data."""
    parts = []

    parts.append(f"ТЕКУЩИЙ СЧЁТ: {live_data.get('score_display', '?')}")
    parts.append(f"СТАТУС: Live — {live_data.get('tournament', '?')}")
    if live_data.get("round"):
        parts.append(f"Раунд: {live_data['round']}")
    if live_data.get("surface"):
        parts.append(f"Покрытие: {live_data['surface']}")

    # Set-by-set breakdown
    sets = live_data.get("current_sets", [])
    if sets:
        parts.append("\n--- ПО НАБОРАМ ---")
        for i, (h, a) in enumerate(sets, 1):
            parts.append(f"  Сет {i}: {h}-{a}")

    # Current game
    current = live_data.get("current_game", "0-0")
    parts.append(f"\nТекущий гейм: {current}")

    # Sets summary
    h_sets = live_data.get("home_sets", 0)
    a_sets = live_data.get("away_sets", 0)
    parts.append(f"Счёт по сетам: {h_sets}:{a_sets}")

    # Momentum analysis
    if sets:
        last_set = sets[-1]
        if last_set[0] > last_set[1]:
            parts.append(f"Инициатива: {live_data['player1']} (выиграл последний сет)")
        elif last_set[1] > last_set[0]:
            parts.append(f"Инициатива: {live_data['player2']} (выиграл последний сет)")
        else:
            parts.append("Инициатива: равная борьба")

    # ML prediction if available
    if ml_prediction:
        parts.append("\n--- ПРОГНОЗ МОДЕЛИ (ДО МАТЧА) ---")
        parts.append(f"  Победа {live_data['player1']}: {ml_prediction.get('player1_win', '?')}%")
        parts.append(f"  Победа {live_data['player2']}: {ml_prediction.get('player2_win', '?')}%")

    return "\n".join(parts)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Tennis Live Context ===")

    live = fetch_live_from_api()
    print(f"API live matches: {len(live)}")
    for m in live[:5]:
        print(f"  {m['player1']} vs {m['player2']} | {m['score_display']} | {m['tournament']}")

    print(f"\nFlashScore live matches: {len(fetch_live_from_flashscore())}")
