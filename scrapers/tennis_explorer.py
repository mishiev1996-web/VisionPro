"""
scrapers/tennis_explorer.py — Parse tennis data from Tennis Explorer.

Tennis Explorer provides:
- Upcoming matches with odds
- Player statistics
- H2H records
- Tournament draws
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


TE_BASE = "https://www.tennisexplorer.com"
TE_SCHEDULE = f"{TE_BASE}/matches/"
TE_RANKINGS = f"{TE_BASE}/ranking/atp/"
TE_LIVE = f"{TE_BASE}/matches/live/"

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


def fetch_upcoming_matches(date: str = None) -> List[Dict]:
    """Fetch upcoming tennis matches.
    
    Args:
        date: Optional date in YYYY-MM-DD format. Default: today.
    """
    url = TE_SCHEDULE
    if date:
        url = f"{TE_BASE}/matches/{date.replace('-', '')}/"

    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    # Find match tables
    for table in soup.select("table"):
        for row in table.select("tr"):
            try:
                cells = row.select("td")
                if len(cells) < 5:
                    continue

                # Player names
                p1_el = cells[0].select_one("a")
                p2_el = cells[2].select_one("a") if len(cells) > 2 else None

                if not p1_el or not p2_el:
                    continue

                p1 = p1_el.get_text(strip=True)
                p2 = p2_el.get_text(strip=True)

                # Tournament
                tournament = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                # Odds (if available)
                odds_cells = cells[3:6] if len(cells) > 5 else []
                odds = []
                for oc in odds_cells:
                    txt = oc.get_text(strip=True)
                    try:
                        odds.append(float(txt))
                    except ValueError:
                        odds.append(None)

                # Player IDs
                p1_id = ""
                p2_id = ""
                if p1_el.get("href"):
                    pid = re.search(r"/player/([^/]+)/", p1_el["href"])
                    if pid:
                        p1_id = pid.group(1)
                if p2_el and p2_el.get("href"):
                    pid = re.search(r"/player/([^/]+)/", p2_el["href"])
                    if pid:
                        p2_id = pid.group(1)

                matches.append({
                    "player1": p1,
                    "player2": p2,
                    "tournament": tournament,
                    "odds1": odds[0] if odds else None,
                    "odds2": odds[1] if len(odds) > 1 else None,
                    "odds_draw": odds[2] if len(odds) > 2 else None,
                    "player1_id": p1_id,
                    "player2_id": p2_id,
                })
            except Exception:
                continue

    return matches


def fetch_live_matches() -> List[Dict]:
    """Fetch currently live tennis matches."""
    html = _fetch(TE_LIVE)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    for table in soup.select("table"):
        for row in table.select("tr"):
            try:
                cells = row.select("td")
                if len(cells) < 4:
                    continue

                p1 = cells[0].get_text(strip=True)
                p2 = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                # Score
                score = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                if p1 and p2:
                    matches.append({
                        "player1": p1,
                        "player2": p2,
                        "score": score,
                        "status": "live",
                    })
            except Exception:
                continue

    return matches


def fetch_player_stats(player_id: str) -> Optional[Dict]:
    """Fetch detailed stats for a player."""
    url = f"{TE_BASE}/player/{player_id}/"
    html = _fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    stats = {}

    # Name
    name_el = soup.select_one("h1")
    stats["name"] = name_el.get_text(strip=True) if name_el else ""

    # Rankings
    rank_el = soup.select_one("[class*='ranking']")
    stats["ranking"] = rank_el.get_text(strip=True) if rank_el else ""

    # Recent form
    form_matches = []
    for row in soup.select("[class*='match'], tr"):
        try:
            cells = row.select("td")
            if len(cells) >= 3:
                opponent = cells[0].get_text(strip=True)
                result = cells[1].get_text(strip=True)
                score = cells[2].get_text(strip=True)
                if opponent and result:
                    form_matches.append({
                        "opponent": opponent,
                        "result": result,
                        "score": score,
                    })
        except Exception:
            continue

    stats["recent_form"] = form_matches[:10]
    return stats


def fetch_h2h(player1_id: str, player2_id: str) -> List[Dict]:
    """Fetch head-to-head record between two players."""
    url = f"{TE_BASE}/match/{player1_id}/{player2_id}/"
    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches = []

    for row in soup.select("tr"):
        try:
            cells = row.select("td")
            if len(cells) >= 3:
                date = cells[0].get_text(strip=True)
                winner = cells[1].get_text(strip=True)
                score = cells[2].get_text(strip=True)

                if date and winner:
                    matches.append({
                        "date": date,
                        "winner": winner,
                        "score": score,
                    })
        except Exception:
            continue

    return matches


# CLI test
if __name__ == "__main__":
    print("=== Tennis Explorer ===")
    print("\nUpcoming matches:")
    matches = fetch_upcoming_matches()
    for m in matches[:10]:
        odds_str = ""
        if m.get("odds1"):
            odds_str = f" | Odds: {m['odds1']}/{m.get('odds_draw','?')}/{m.get('odds2','?')}"
        print(f"  {m['player1']} vs {m['player2']} | {m['tournament']}{odds_str}")
