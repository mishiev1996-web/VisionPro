"""
scrapers/fbref.py — Fetch xG data from FBref.

STATUS: Cloudflare Turnstile blocks ALL automated access (Playwright, curl_cffi,
cloudscraper, nodriver). This module is kept for reference but won't work
until FBref changes their protection or a paid API is used.

For xG data, use Understat (6 leagues) or goals-based estimation (0.9 * goals).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import sync_playwright, Browser, Page


FBREF_BASE = "https://fbref.com"
COOKIES_PATH = Path(__file__).parent.parent / "data" / "fbref_cookies.json"

LEAGUES: Dict[str, Dict[str, str]] = {
    "EPL":           {"name": "Premier League",    "country": "England",    "slug": "9"},
    "La_liga":       {"name": "La Liga",           "country": "Spain",      "slug": "12"},
    "Bundesliga":    {"name": "Bundesliga",        "country": "Germany",    "slug": "20"},
    "Serie_A":       {"name": "Serie A",           "country": "Italy",      "slug": "11"},
    "Ligue_1":       {"name": "Ligue 1",           "country": "France",     "slug": "13"},
    "RFPL":          {"name": "Russian Premier",   "country": "Russia",     "slug": "24"},
    "Eredivisie":    {"name": "Eredivisie",        "country": "Netherlands","slug": "14"},
    "Primeira_Liga": {"name": "Primeira Liga",     "country": "Portugal",   "slug": "23"},
    "Super_Lig":     {"name": "Super Lig",         "country": "Turkey",     "slug": "26"},
    "Championship":  {"name": "Championship",      "country": "England",    "slug": "8"},
    "MLS":           {"name": "MLS",               "country": "USA",        "slug": "21"},
    "Liga_MX":       {"name": "Liga MX",           "country": "Mexico",     "slug": "22"},
}


def _save_cookies(context) -> None:
    """Save browser cookies for reuse."""
    cookies = context.cookies()
    COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COOKIES_PATH, "w") as f:
        json.dump(cookies, f)


def _load_cookies(context) -> bool:
    """Load saved cookies if they exist."""
    if not COOKIES_PATH.exists():
        return False
    try:
        with open(COOKIES_PATH, "r") as f:
            cookies = json.load(f)
        context.add_cookies(cookies)
        return True
    except Exception:
        return False


def _fetch_with_playwright(url: str, timeout: int = 30000) -> Optional[str]:
    """Fetch a page using Playwright with cookie persistence."""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            java_script_enabled=True,
        )

        # Try loading saved cookies
        _load_cookies(context)

        page = context.new_page()

        # Remove webdriver detection
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = {runtime: {}};
        """)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)

            # Wait for Cloudflare challenge to resolve
            for _ in range(30):
                title = page.title()
                content = page.content()
                if ("just a moment" not in title.lower() and
                    "checking" not in title.lower() and
                    "challenge" not in content.lower()[:1000]):
                    break
                time.sleep(1)

            # Wait for tables to render
            try:
                page.wait_for_selector("table", timeout=15000)
            except Exception:
                pass

            time.sleep(2)

            # Save cookies for future use
            _save_cookies(context)

            html = page.content()
            return html

        except Exception as e:
            print(f"  Playwright error: {e}")
            return None
        finally:
            browser.close()


def _parse_fbref_schedule(html: str, league_slug: str) -> List[Dict]:
    """Parse xG data from FBref schedule HTML."""
    from bs4 import BeautifulSoup

    matches = []
    soup = BeautifulSoup(html, "html.parser")

    # Find the schedule table
    table = soup.find("table", id="sched")
    if not table:
        tables = soup.find_all("table", class_=re.compile(r"stats_table"))
        if tables:
            table = tables[0]

    if not table or not table.tbody:
        return matches

    # Get headers to find column positions
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thethead.find_all("th"):
            text = th.get_text(strip=True).lower()
            headers.append(text)

    # Find xG columns
    xg_cols = [i for i, h in enumerate(headers) if "xg" in h]
    score_col = next((i for i, h in enumerate(headers) if "score" in h), None)
    date_col = next((i for i, h in enumerate(headers) if "date" in h), None)
    home_col = next((i for i, h in enumerate(headers) if "home" in h or "squad" in h), None)
    away_col = next((i for i, h in enumerate(headers) if "away" in h or "opponent" in h), None)

    if not xg_cols or len(xg_cols) < 2:
        return matches

    xg_h_col = xg_cols[0]
    xg_a_col = xg_cols[1] if len(xg_cols) > 1 else xg_cols[0]

    # Parse each row
    for tr in table.tbody.find_all("tr"):
        # Skip separator rows
        if tr.get("class") and "thead" in " ".join(tr.get("class", [])):
            continue

        tds = tr.find_all(["td", "th"])
        if len(tds) < max(filter(None, [xg_h_col, xg_a_col, score_col, date_col, home_col, away_col])) + 1:
            continue

        try:
            date_text = tds[date_col].get_text(strip=True) if date_col is not None else ""
            home_text = tds[home_col].get_text(strip=True) if home_col is not None else ""
            away_text = tds[away_col].get_text(strip=True) if away_col is not None else ""
            score_text = tds[score_col].get_text(strip=True) if score_col is not None else ""
            xg_h_text = tds[xg_h_col].get_text(strip=True)
            xg_a_text = tds[xg_a_col].get_text(strip=True)

            # Parse xG
            home_xg = float(xg_h_text) if xg_h_text else None
            away_xg = float(xg_a_text) if xg_a_text else None
            if home_xg is None or away_xg is None:
                continue

            # Parse score
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_text)
            if not score_match:
                continue
            home_goals = int(score_match.group(1))
            away_goals = int(score_match.group(2))

            # Parse date
            date_iso = ""
            dm = re.search(r'(\d{4}-\d{2}-\d{2})', date_text)
            if dm:
                date_iso = dm.group(0)
            else:
                dm = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
                if dm:
                    date_iso = f"{dm.group(3)}-{dm.group(1).zfill(2)}-{dm.group(2).zfill(2)}"

            if not date_iso or not home_text or not away_text:
                continue

            matches.append({
                "home": home_text,
                "away": away_text,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "date": date_iso,
                "league": league_slug,
            })
        except (ValueError, IndexError, AttributeError):
            continue

    return matches


def fetch_league_xg(league_slug: str) -> List[Dict]:
    """Fetch xG data for a league from FBref."""
    if league_slug not in LEAGUES:
        return []

    comp_id = LEAGUES[league_slug]["slug"]
    comp_name = LEAGUES[league_slug]["name"].replace(" ", "-")
    url = f"{FBREF_BASE}/en/comps/{comp_id}/{comp_name}-Stats"

    html = _fetch_with_playwright(url)
    if not html:
        return []

    return _parse_fbref_schedule(html, league_slug)


def fetch_all_leagues_xg(progress_cb=None) -> Dict[str, List[Dict]]:
    """Fetch xG data for all supported leagues."""
    results = {}
    for slug, meta in LEAGUES.items():
        if progress_cb:
            progress_cb({"type": "info", "msg": f"FBref: {meta['name']}..."})
        try:
            matches = fetch_league_xg(slug)
            results[slug] = matches
            if progress_cb:
                progress_cb({"type": "success", "msg": f"✓ {meta['name']}: {len(matches)} matches"})
        except Exception as e:
            results[slug] = []
            if progress_cb:
                progress_cb({"type": "error", "msg": f"✗ {meta['name']}: {e}"})
        time.sleep(3)  # Be polite
    return results
