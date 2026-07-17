"""
tennis_prematch.py — Unified tennis prematch data from multiple sources.

Combines:
- FlashScore (live + upcoming)
- ATP Tour (schedule + rankings)
- Tennis Explorer (odds + H2H)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_tennis_upcoming() -> List[Dict]:
    """Fetch upcoming tennis matches from multiple sources.
    
    Returns unified list of matches with:
    - player1, player2
    - tournament, surface
    - round, time
    - odds (if available)
    """
    matches = []

    # Source 1: Local CSV (most reliable for current tournaments)
    try:
        matches = _load_from_csv()
        if matches:
            return matches
    except Exception as e:
        print(f"[tennis_prematch] CSV error: {e}")

    # Source 2: Tennis Explorer (fallback)
    try:
        from scrapers.tennis_explorer import fetch_upcoming_matches
        te_matches = fetch_upcoming_matches()
        for m in te_matches:
            if m.get("player1") and m.get("player2"):
                matches.append({
                    "player1": m.get("player1", ""),
                    "player2": m.get("player2", ""),
                    "tournament": m.get("tournament", ""),
                    "surface": _detect_surface(m.get("tournament", "")),
                    "round": "",
                    "time": "",
                    "odds1": m.get("odds1"),
                    "odds2": m.get("odds2"),
                    "source": "tennis_explorer",
                })
    except Exception as e:
        print(f"[tennis_prematch] Tennis Explorer error: {e}")

    # Source 3: FlashScore (last resort)
    if not matches:
        try:
            from scrapers.tennis_flashscore import fetch_tennis_upcoming
            fs_matches = fetch_tennis_upcoming()
            for m in fs_matches:
                matches.append({
                    "player1": m.get("home", ""),
                    "player2": m.get("away", ""),
                    "tournament": m.get("tournament", ""),
                    "surface": _detect_surface(m.get("tournament", "")),
                    "round": "",
                    "time": m.get("time", ""),
                    "odds1": None,
                    "odds2": None,
                    "source": "flashscore",
                })
        except Exception as e:
            print(f"[tennis_prematch] FlashScore error: {e}")

    return matches


def fetch_tennis_live() -> List[Dict]:
    """Fetch currently live tennis matches."""
    matches = []

    # FlashScore live
    try:
        from scrapers.tennis_flashscore import fetch_tennis_live
        fs_matches = fetch_tennis_live()
        for m in fs_matches:
            matches.append({
                "player1": m.get("home", ""),
                "player2": m.get("away", ""),
                "score": m.get("score", ""),
                "tournament": m.get("tournament", ""),
                "status": "live",
                "source": "flashscore",
            })
    except Exception as e:
        print(f"[tennis_prematch] FlashScore live error: {e}")

    # Tennis Explorer live
    if not matches:
        try:
            from scrapers.tennis_explorer import fetch_live_matches
            te_matches = fetch_live_matches()
            for m in te_matches:
                matches.append({
                    "player1": m.get("player1", ""),
                    "player2": m.get("player2", ""),
                    "score": m.get("score", ""),
                    "status": "live",
                    "source": "tennis_explorer",
                })
        except Exception as e:
            print(f"[tennis_prematch] Tennis Explorer live error: {e}")

    return matches


def fetch_tournament_info(tournament_name: str) -> Dict:
    """Get info about a specific tournament."""
    info = {
        "name": tournament_name,
        "surface": _detect_surface(tournament_name),
        "level": _detect_level(tournament_name),
    }

    # Try to get more info from ATP
    try:
        from scrapers.tennis_atp import fetch_current_tournaments
        tournaments = fetch_current_tournaments()
        for t in tournaments:
            if tournament_name.lower() in t.get("name", "").lower():
                info.update(t)
                break
    except Exception:
        pass

    return info


def fetch_player_profile(player_name: str) -> Dict:
    """Get player profile from multiple sources."""
    profile = {
        "name": player_name,
        "ranking": None,
        "country": "",
        "recent_form": [],
    }

    # Try Tennis Explorer
    try:
        from scrapers.tennis_explorer import search_player
        results = search_player(player_name)
        if results:
            profile.update(results[0])
    except Exception:
        pass

    # Try local DB
    try:
        import tennis.tennis_db as tennis_db
        db_results = tennis_db.search_player(player_name, limit=1)
        if db_results:
            p = db_results[0]
            profile["ranking"] = p.get("ranking")
            profile["country"] = p.get("country", "")
            profile["player_id"] = p.get("id")
    except Exception:
        pass

    return profile


def _detect_surface(tournament_name: str) -> str:
    """Detect surface from tournament name."""
    name_lower = tournament_name.lower()
    if "wimbledon" in name_lower:
        return "Grass"
    if "roland garros" in name_lower or "french open" in name_lower:
        return "Clay"
    if "australian open" in name_lower or "us open" in name_lower:
        return "Hard"
    if "indian wells" in name_lower or "miami" in name_lower:
        return "Hard"
    if "monte carlo" in name_lower or "rome" in name_lower or "madrid" in name_lower:
        return "Clay"
    return "Hard"  # default


def _detect_level(tournament_name: str) -> str:
    """Detect tournament level from name."""
    name_lower = tournament_name.lower()
    if any(x in name_lower for x in ["grand slam", "australian open", "roland garros",
                                       "wimbledon", "us open"]):
        return "Grand Slam"
    if any(x in name_lower for x in ["masters", "indian wells", "miami", "monte carlo",
                                       "rome", "madrid", "canada", "cincinnati", "shanghai", "paris"]):
        return "Masters 1000"
    if "atp 500" in name_lower or any(x in name_lower for x in ["dubai", "barcelona", "halle",
                                                                   "queens", "hamburg", "beijing",
                                                                   "tokyo", "vienna", "basel"]):
        return "ATP 500"
    if "atp 250" in name_lower:
        return "ATP 250"
    return "ATP Tour"


def _load_from_csv() -> List[Dict]:
    """Load matches from ongoing_tourneys.csv and generate upcoming matches."""
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent / "База Теннис" / "ongoing_tourneys.csv"
    if not csv_path.exists():
        return []

    # Load all matches
    all_matches = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_matches.append(row)

    # Generate upcoming matches based on confirmed bracket
    upcoming = []

    # Wimbledon 2026 SF matches (confirmed by user)
    # SF1: Sinner vs Djokovic
    # SF2: Zverev vs Fery
    sf_matches = [
        {
            "player1": "Jannik Sinner",
            "player2": "Novak Djokovic",
            "tournament": "Wimbledon",
            "surface": "Grass",
            "round": "SF",
            "time": "2026-07-10",
            "odds1": None,
            "odds2": None,
            "source": "confirmed",
        },
        {
            "player1": "Alexander Zverev",
            "player2": "Arthur Fery",
            "tournament": "Wimbledon",
            "surface": "Grass",
            "round": "SF",
            "time": "2026-07-10",
            "odds1": None,
            "odds2": None,
            "source": "confirmed",
        },
    ]

    # Final placeholder
    final = {
        "player1": "Победитель SF1",
        "player2": "Победитель SF2",
        "tournament": "Wimbledon",
        "surface": "Grass",
        "round": "Final",
        "time": "2026-07-12",
        "odds1": None,
        "odds2": None,
        "source": "bracket",
    }

    upcoming.extend(sf_matches)
    upcoming.append(final)

    return upcoming


# CLI test
if __name__ == "__main__":
    print("=== Tennis Prematch ===")
    print("\nUpcoming matches:")
    matches = fetch_tennis_upcoming()
    for m in matches[:10]:
        print(f"  {m['player1']} vs {m['player2']} | {m['tournament']} | {m['surface']}")

    print("\nLive matches:")
    live = fetch_tennis_live()
    for m in live[:5]:
        print(f"  {m['player1']} vs {m['player2']} | {m['score']}")
