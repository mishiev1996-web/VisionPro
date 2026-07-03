"""
scrapers/flashscore.py — fetch live & upcoming matches worldwide via FlashScore.

FlashScore is protected by Cloudflare and renders match data via JS, so we use
Botasaurus' @browser decorator (headless Chromium with anti-detection).

Two entry points:
    fetch_today_matches()     → list of today's worldwide matches (live + upcoming + finished)
    fetch_league_matches(url) → all visible matches for one league page

The site changes its CSS class names occasionally, so we fall back to multiple
selectors. If parsing fails, we return [] instead of crashing.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from botasaurus.browser import browser, Driver
from botasaurus.soupify import soupify


FS_BASE = "https://www.flashscore.com"


@browser(
    headless=True,
    block_images=True,
    create_error_logs=False,
    output=None,
    max_retry=2,
    reuse_driver=True,
)
def _fetch_page_html(driver: Driver, url: str) -> str:
    """Open a FlashScore page, wait for matches to render, return full HTML."""
    driver.google_get(url, bypass_cloudflare=True)
    # Wait for the match list container to appear (best-effort)
    try:
        driver.short_random_sleep()
        # Most FlashScore pages render match rows under a container with class
        # containing "leagues" / "sportName". Just give it time.
        for _ in range(8):
            html = driver.page_html
            if "event__match" in html or "wclLeagueHeader" in html:
                break
            time.sleep(0.5)
    except Exception:
        pass
    return driver.page_html or ""


def _parse_matches(html: str) -> List[Dict[str, str]]:
    """Walk through FlashScore DOM. Match rows are divs with class 'event__match…'.

    Each league has a header row with class 'event__header' / 'wclLeagueHeader'.
    Returns list of {country, league, home, away, score, status, time}.
    """
    soup = soupify(html)
    matches: List[Dict[str, str]] = []
    current_country = ""
    current_league = ""

    # FlashScore uses many auto-generated class names with hash suffixes.
    # The stable prefixes we rely on are: 'event__header', 'event__match',
    # 'event__participant', 'event__time', 'event__score'.
    for el in soup.find_all(class_=re.compile(r"(event__header|event__match)")):
        cls = " ".join(el.get("class") or [])

        if "event__header" in cls or "wclLeagueHeader" in cls:
            # Two spans typically: country, league name
            spans = el.find_all(["span", "a", "div"])
            text_parts = [s.get_text(strip=True) for s in spans if s.get_text(strip=True)]
            if len(text_parts) >= 2:
                current_country, current_league = text_parts[0], text_parts[1]
            elif text_parts:
                current_league = text_parts[0]
            continue

        if "event__match" not in cls:
            continue

        home = ""
        away = ""
        score_home = score_away = ""
        time_str = ""
        status = ""

        # Participants — home & away
        homes = el.find_all(class_=re.compile(r"event__participant--home"))
        aways = el.find_all(class_=re.compile(r"event__participant--away"))
        if homes: home = homes[0].get_text(strip=True)
        if aways: away = aways[0].get_text(strip=True)
        if not home or not away:
            # Fallback: 2 .event__participant
            parts = el.find_all(class_=re.compile(r"event__participant"))
            if len(parts) >= 2:
                home = home or parts[0].get_text(strip=True)
                away = away or parts[1].get_text(strip=True)

        # Score
        sh = el.find(class_=re.compile(r"event__score--home"))
        sa = el.find(class_=re.compile(r"event__score--away"))
        if sh: score_home = sh.get_text(strip=True)
        if sa: score_away = sa.get_text(strip=True)

        # Time (kick-off or live minute)
        tm = el.find(class_=re.compile(r"event__time"))
        if tm: time_str = tm.get_text(strip=True)

        # Status: live, finished, scheduled — guess from classes
        if "event__match--live" in cls:    status = "live"
        elif "event__match--finished" in cls or score_home or score_away: status = "finished"
        else: status = "scheduled"

        if home and away:
            matches.append({
                "country": current_country,
                "league":  current_league,
                "home":    home,
                "away":    away,
                "score_home": score_home,
                "score_away": score_away,
                "time":    time_str,
                "status":  status,
            })
    return matches


def fetch_today_matches() -> List[Dict[str, str]]:
    """Worldwide matches scheduled / live / finished today."""
    html = _fetch_page_html(FS_BASE + "/")
    return _parse_matches(html)


def fetch_league_matches(league_url: str) -> List[Dict[str, str]]:
    """Schedule + recent results for one league.

    `league_url` is the FlashScore path, e.g. '/football/england/premier-league/'
    or a full URL.
    """
    if not league_url.startswith("http"):
        league_url = FS_BASE + league_url
    html = _fetch_page_html(league_url)
    return _parse_matches(html)


def fetch_live_only() -> List[Dict[str, str]]:
    """Just the live matches right now (smaller, faster)."""
    html = _fetch_page_html(FS_BASE + "/football/?d=live")
    return [m for m in _parse_matches(html) if m["status"] == "live"]
