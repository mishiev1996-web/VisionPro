"""
scrapers/tennis_atp.py — Parse tennis schedule from ATP Tour website.

Fetches upcoming ATP tournaments and matches from atptour.com.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


ATP_BASE = "https://www.atptour.com"
ATP_SCHEDULE = f"{ATP_BASE}/en/scores/current"
ATP_RANKINGS = f"{ATP_BASE}/en/rankings/singles"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch(url: str) -> Optional[str]:
    """Fetch a page with retry."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
    return None


def fetch_current_tournaments() -> List[Dict]:
    """Fetch current ATP tournaments from schedule page."""
    html = _fetch(ATP_SCHEDULE)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    tournaments = []

    # Find tournament cards
    for card in soup.select("[class*='tournament'], [class*='event']"):
        try:
            name_el = card.select_one("a[href*='/en/tournament/']")
            if not name_el:
                continue

            name = name_el.get_text(strip=True)
            href = name_el.get("href", "")
            tournament_id = re.search(r"/tournament/(\d+)", href)
            tid = tournament_id.group(1) if tournament_id else ""

            # Surface
            surface_el = card.select_one("[class*='surface']")
            surface = surface_el.get_text(strip=True) if surface_el else ""

            # Location
            location_el = card.select_one("[class*='location']")
            location = location_el.get_text(strip=True) if location_el else ""

            # Dates
            date_el = card.select_one("[class*='date']")
            dates = date_el.get_text(strip=True) if date_el else ""

            tournaments.append({
                "id": tid,
                "name": name,
                "surface": surface,
                "location": location,
                "dates": dates,
                "url": f"{ATP_BASE}{href}" if href.startswith("/") else href,
            })
        except Exception:
            continue

    return tournaments


def fetch_tournament_draw(tournament_id: str) -> List[Dict]:
    """Fetch draw/matches for a specific tournament."""
    url = f"{ATP_BASE}/en/tournament/{tournament_id}/draws"
    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    for row in soup.select("[class*='match'], [class*='row']"):
        try:
            players = row.select("[class*='player']")
            if len(players) < 2:
                continue

            p1 = players[0].get_text(strip=True)
            p2 = players[1].get_text(strip=True)

            # Score
            score_el = row.select_one("[class*='score']")
            score = score_el.get_text(strip=True) if score_el else ""

            # Round
            round_el = row.select_one("[class*='round']")
            rnd = round_el.get_text(strip=True) if round_el else ""

            matches.append({
                "player1": p1,
                "player2": p2,
                "score": score,
                "round": rnd,
            })
        except Exception:
            continue

    return matches


def fetch_rankings(top_n: int = 100) -> List[Dict]:
    """Fetch ATP singles rankings."""
    html = _fetch(ATP_RANKINGS)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    players = []

    for row in soup.select("[class*='rankings-row'], tr"):
        try:
            rank_el = row.select_one("[class*='rank']")
            name_el = row.select_one("[class*='name'], a[href*='/en/player/']")
            points_el = row.select_one("[class*='points']")

            if not name_el:
                continue

            rank = rank_el.get_text(strip=True) if rank_el else ""
            name = name_el.get_text(strip=True)
            points = points_el.get_text(strip=True) if points_el else ""

            # Player ID from link
            href = name_el.get("href", "")
            player_id = ""
            if href:
                pid_match = re.search(r"/player/([^/]+)/", href)
                if pid_match:
                    player_id = pid_match.group(1)

            players.append({
                "rank": rank,
                "name": name,
                "points": points,
                "player_id": player_id,
            })
        except Exception:
            continue

    return players[:top_n]


def search_player(query: str) -> List[Dict]:
    """Search for a player by name."""
    url = f"{ATP_BASE}/en/search?q={query}"
    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for item in soup.select("[class*='search-result'], [class*='player']"):
        try:
            name_el = item.select_one("a[href*='/player/']")
            if not name_el:
                continue

            name = name_el.get_text(strip=True)
            href = name_el.get("href", "")
            player_id = ""
            if href:
                pid_match = re.search(r"/player/([^/]+)/", href)
                if pid_match:
                    player_id = pid_match.group(1)

            results.append({
                "name": name,
                "player_id": player_id,
                "url": f"{ATP_BASE}{href}" if href.startswith("/") else href,
            })
        except Exception:
            continue

    return results


# CLI test
if __name__ == "__main__":
    print("=== ATP Tour ===")
    print("\nCurrent tournaments:")
    tournaments = fetch_current_tournaments()
    for t in tournaments[:5]:
        print(f"  {t['name']} | {t['surface']} | {t['location']}")

    print("\nTop 10 rankings:")
    rankings = fetch_rankings(10)
    for p in rankings:
        print(f"  #{p['rank']} {p['name']} ({p['points']} pts)")
