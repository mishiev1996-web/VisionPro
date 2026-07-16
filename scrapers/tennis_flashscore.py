"""
scrapers/tennis_flashscore.py — Parse tennis matches from FlashScore.

Uses Botasaurus for headless browsing with anti-detection.
Fetches upcoming, live, and recent tennis matches worldwide.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from botasaurus.browser import browser, Driver
from botasaurus.soupify import soupify


FS_TENNIS_URL = "https://www.flashscore.com/tennis/"


@browser(
    headless=True,
    block_images=True,
    create_error_logs=False,
    output=None,
    max_retry=2,
    reuse_driver=True,
)
def _fetch_page(driver: Driver, url: str) -> str:
    """Fetch a FlashScore page with anti-detection."""
    driver.google_get(url, bypass_cloudflare=True)
    try:
        driver.short_random_sleep()
        for _ in range(10):
            html = driver.page_html
            if "event__match" in html or "sportName" in html:
                break
            time.sleep(0.5)
    except Exception:
        pass
    return driver.page_html or ""


def _parse_matches(html: str) -> List[Dict]:
    """Parse match rows from FlashScore HTML."""
    soup = soupify(html)
    matches = []

    # Find all match containers
    for row in soup.select("[class*='event__match'], [class*='wclMatch']"):
        try:
            # Extract teams
            home_el = row.select_one("[class*='event__participant--home'], [class*='wclHome']")
            away_el = row.select_one("[class*='event__participant--away'], [class*='wclAway']")
            if not home_el or not away_el:
                continue

            home = home_el.get_text(strip=True)
            away = away_el.get_text(strip=True)

            # Extract score
            score_el = row.select_one("[class*='event__score'], [class*='wclScore']")
            score = score_el.get_text(strip=True) if score_el else ""

            # Extract time/status
            time_el = row.select_one("[class*='event__time'], [class*='wclTime']")
            match_time = time_el.get_text(strip=True) if time_el else ""

            # Extract tournament
            stage = row.select_one("[class*='event__stage'], [class*='wclStage']")
            tournament = stage.get_text(strip=True) if stage else ""

            # Match ID
            match_id = row.get("id", "")
            if match_id:
                match_id = re.sub(r"[^\d]", "", match_id)

            matches.append({
                "id": match_id,
                "home": home,
                "away": away,
                "score": score,
                "time": match_time,
                "tournament": tournament,
                "status": _detect_status(score, match_time),
            })
        except Exception:
            continue

    return matches


def _detect_status(score: str, time_str: str) -> str:
    """Detect match status from score and time."""
    if score and "-" in score:
        # Check if match is finished (all sets complete)
        sets = re.findall(r"\d+-\d+", score)
        if len(sets) >= 2:
            last_set = sets[-1]
            s1, s2 = map(int, last_set.split("-"))
            # If one player reached 6+ and is ahead by 2+, likely finished
            if (s1 >= 6 or s2 >= 6) and abs(s1 - s2) >= 2:
                return "finished"
        return "live"
    if time_str and ":" in time_str:
        return "upcoming"
    return "unknown"


def fetch_tennis_upcoming(limit: int = 50) -> List[Dict]:
    """Fetch upcoming tennis matches from FlashScore."""
    html = _fetch_page(FS_TENNIS_URL)
    matches = _parse_matches(html)
    return [m for m in matches if m["status"] == "upcoming"][:limit]


def fetch_tennis_live(limit: int = 50) -> List[Dict]:
    """Fetch live tennis matches from FlashScore."""
    html = _fetch_page(FS_TENNIS_URL)
    matches = _parse_matches(html)
    return [m for m in matches if m["status"] == "live"][:limit]


def fetch_tennis_all(limit: int = 100) -> List[Dict]:
    """Fetch all tennis matches (upcoming + live + recent)."""
    html = _fetch_page(FS_TENNIS_URL)
    return _parse_matches(html)[:limit]


# CLI test
if __name__ == "__main__":
    print("=== FlashScore Tennis ===")
    matches = fetch_tennis_all()
    print(f"Found {len(matches)} matches")
    for m in matches[:10]:
        print(f"  [{m['status']}] {m['home']} vs {m['away']} | {m['tournament']} | {m['score']}")
