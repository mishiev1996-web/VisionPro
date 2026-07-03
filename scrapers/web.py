"""
scrapers/web.py — Universal web scraper: Jina Reader API → Playwright fallback.

Jina Reader (https://r.jina.ai/) converts any URL to clean Markdown:
- Free, no API key required
- Bypasses Cloudflare and basic anti-bot
- Returns clean text, not raw HTML
- Rate limit: ~20 req/min on free tier
"""
from __future__ import annotations

import datetime as dt
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus

import requests

from scrapers.utils import format_msk, MSK


HEADERS = {
    "User-Agent": "FootballAI/1.0",
    "Accept": "text/markdown",
}

# Per-URL cache to avoid repeated fetches
_cache: Dict[str, str] = {}
_cache_ttl: Dict[str, float] = {}
CACHE_SECONDS = 300  # 5 min


# ── Jina Reader (primary) ──────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 15, max_chars: int = 8000) -> str:
    """Fetch a URL and return clean Markdown text via Jina Reader.

    Falls back to raw requests if Jina fails.
    Returns empty string on total failure.
    """
    # Check cache
    now = time.time()
    if url in _cache and (now - _cache_ttl.get(url, 0)) < CACHE_SECONDS:
        return _cache[url][:max_chars]

    # 1. Try Jina Reader
    text = _fetch_via_jina(url, timeout)
    if text and len(text) > 50:
        _cache[url] = text
        _cache_ttl[url] = now
        return text[:max_chars]

    # 2. Fallback: raw requests with HTML stripping
    text = _fetch_raw(url, timeout)
    if text:
        _cache[url] = text
        _cache_ttl[url] = now
        return text[:max_chars]

    return ""


def _fetch_via_jina(url: str, timeout: int = 15) -> Optional[str]:
    """Use Jina Reader API to convert URL to Markdown."""
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers=HEADERS,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception:
        pass
    return None


def _fetch_raw(url: str, timeout: int = 10) -> Optional[str]:
    """Fallback: raw HTTP GET with HTML tag stripping."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36",
        })
        resp.raise_for_status()
        html = resp.text
        # Strip scripts, styles, tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return None


# ── Search ──────────────────────────────────────────────────────────────────

def search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web and return structured results.

    Uses Jina Search API (free tier) with DuckDuckGo fallback.
    Returns: [{"title": str, "url": str, "snippet": str}]
    """
    # 1. Try Jina Search
    results = _search_jina(query, num_results)
    if results:
        return results[:num_results]

    # 2. Fallback: DuckDuckGo instant answers
    results = _search_duckduckgo(query, num_results)
    return results[:num_results]


def _search_jina(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Jina Search API — free, returns web search results."""
    try:
        resp = requests.get(
            f"https://s.jina.ai/{quote_plus(query)}",
            headers={
                "User-Agent": "FootballAI/1.0",
                "Accept": "application/json",
            },
            timeout=12,
        )
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            results = []
            for r in data.get("data", [])[:num_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:300],
                })
            if results:
                return results
    except Exception:
        pass
    return []


def _search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo instant answers API."""
    results = []
    try:
        resp = requests.get(
            f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json",
            timeout=8,
            headers={"User-Agent": "FootballAI/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("Abstract", "")
            if abstract:
                results.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": abstract[:300],
                })
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:300],
                    })
    except Exception:
        pass
    return results[:num_results]


# ── Wikipedia ───────────────────────────────────────────────────────────────

def fetch_wikipedia(query: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch Wikipedia article summary as clean text."""
    slug = query.replace(" ", "_")
    urls = [
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}_national_football_team",
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}_(football_club)",
    ]
    for url in urls:
        text = fetch_url(url, timeout=8, max_chars=max_chars)
        if text and "football" in text.lower():
            return text
    return None


# ── Team data gatherer ─────────────────────────────────────────────────────

def gather_team_info(team_name: str, progress_cb=None) -> Dict[str, Any]:
    """Gather comprehensive info about a football team from web sources.

    Priority: Jina Search → Wikipedia → DuckDuckGo
    Returns: {"name", "source", "data", "url"}
    """
    if progress_cb:
        progress_cb({"type": "info", "msg": f"Web: ищу {team_name}…"})

    info = {"name": team_name, "source": "none", "data": "", "url": ""}

    # 1. Search for the team
    results = search(f"{team_name} football team squad results 2025 2026", num_results=5)
    if results:
        best = results[0]
        info["url"] = best["url"]
        info["data"] = best["snippet"]

        # Fetch the best result page
        page_text = fetch_url(best["url"], max_chars=5000)
        if page_text:
            info["data"] = page_text
            info["source"] = "web"
            if progress_cb:
                progress_cb({"type": "success", "msg": f"✓ Web: {len(page_text)} символов"})
            return info

    # 2. Try Wikipedia
    if progress_cb:
        progress_cb({"type": "info", "msg": f"Wikipedia: {team_name}…"})

    wiki = fetch_wikipedia(team_name)
    if wiki:
        info["data"] = wiki
        info["source"] = "wikipedia"
        if progress_cb:
            progress_cb({"type": "success", "msg": f"✓ Wikipedia: {len(wiki)} символов"})
        return info

    if progress_cb:
        progress_cb({"type": "info", "msg": "Web: данные не найдены"})

    return info


def gather_match_info(home: str, away: str, progress_cb=None) -> Dict[str, Any]:
    """Gather info about a specific match from web sources."""
    if progress_cb:
        progress_cb({"type": "info", "msg": f"Web: ищу {home} vs {away}…"})

    query = f"{home} vs {away} football match preview prediction"
    results = search(query, num_results=5)

    info = {
        "home": home, "away": away,
        "results": results,
        "pages": [],
    }

    for r in results[:3]:
        text = fetch_url(r["url"], max_chars=3000)
        if text:
            info["pages"].append({
                "title": r["title"],
                "url": r["url"],
                "text": text,
            })

    if progress_cb:
        progress_cb({"type": "success", "msg": f"Web: {len(info['pages'])} источников"})

    return info


# ── ESPN match scraper ──────────────────────────────────────────────────────

def find_espn_match_id(home: str, away: str) -> Optional[str]:
    """Find ESPN gameId for a match by searching ESPN scoreboard.

    Returns gameId string or None.
    """
    import datetime as _dt
    today = _dt.date.today()

    # Search today ±3 days
    for delta in range(-3, 4):
        d = today + _dt.timedelta(days=delta)
        date_str = d.strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date_str}"
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "FootballAI/1.0"})
            if resp.status_code != 200:
                continue
            data = resp.json()
            for event in data.get("events", []):
                comps = event.get("competitions", [])
                if not comps:
                    continue
                comp = comps[0]
                sides = comp.get("competitors", [])
                if len(sides) < 2:
                    continue
                h = sides[0] if sides[0].get("homeAway") == "home" else sides[1]
                a = sides[1] if sides[0].get("homeAway") == "home" else sides[0]
                h_name = (h.get("team") or {}).get("displayName", "").lower()
                a_name = (a.get("team") or {}).get("displayName", "").lower()
                if (home.lower() in h_name or h_name in home.lower()) and \
                   (away.lower() in a_name or a_name in away.lower()):
                    return event.get("id")
        except Exception:
            continue
    return None


def fetch_espn_match(home_en: str, away_en: str, progress_cb=None) -> Optional[Dict[str, Any]]:
    """Fetch match data from ESPN using ESPN JSON API + Jina Reader.

    Returns dict with match info, recent form, stats, or None.
    """
    if progress_cb:
        progress_cb({"type": "info", "msg": f"ESPN: ищу {home_en} vs {away_en}…"})

    # 1. Find the match ID via ESPN API
    game_id = find_espn_match_id(home_en, away_en)
    if not game_id:
        if progress_cb:
            progress_cb({"type": "info", "msg": "ESPN: матч не найден"})
        return None

    # 2. Get match details from ESPN JSON API (faster, more reliable)
    match_detail = _fetch_espn_match_detail(game_id)

    # 3. Fetch match page via Jina Reader (for news, headlines)
    url = f"https://www.espn.com/soccer/match/_/gameId/{game_id}"
    text = fetch_url(url, max_chars=8000)

    result = {
        "source": "espn",
        "game_id": game_id,
        "url": url,
        "raw_text": text or "",
        "match_info": "",
        "recent_form": "",
        "stats": "",
        "news": "",
    }

    # 4. Build match_info from JSON API data
    if match_detail:
        result["match_info"] = _format_match_info(match_detail, home_en, away_en)

    # 5. Parse news from Jina text
    if text:
        for line in text.split("\n"):
            line_s = line.strip()
            if not line_s or len(line_s) < 20:
                continue
            lower = line_s.lower()
            if any(kw in lower for kw in ["coach", "joke", "preview", "prediction",
                                           "injury", "squad", "lineup", "formation"]):
                result["news"] += line_s[:200] + "\n"

    # 6. Fetch team recent form from ESPN JSON API
    for team_name in [home_en, away_en]:
        team_id = _espn_team_id(team_name)
        if team_id and team_id != "0":
            team_data = _fetch_espn_team_schedule(team_id)
            if team_data:
                result["recent_form"] += f"\n--- {team_name} ---\n{team_data}\n"

    if progress_cb:
        total = len(result["match_info"]) + len(result["recent_form"]) + len(result["news"])
        progress_cb({"type": "success", "msg": f"ESPN: {total} символов данных"})

    return result


def _fetch_espn_match_detail(game_id: str) -> Optional[Dict]:
    """Fetch match detail from ESPN JSON API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={game_id}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "FootballAI/1.0"})
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _format_match_info(detail: dict, home_en: str, away_en: str) -> str:
    """Format match info from ESPN JSON API response."""
    lines = []

    # Header
    header = detail.get("header", {})
    comp = header.get("competitions", [{}])[0] if header.get("competitions") else {}
    competitors = comp.get("competitors", [])

    # Find teams
    home_data = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_data = next((c for c in competitors if c.get("homeAway") == "away"), {})

    home_team = home_data.get("team", {})
    away_team = away_data.get("team", {})

    # Venue
    venue = comp.get("venue", {})
    venue_name = venue.get("fullName", "")
    city = venue.get("address", {}).get("city", "")

    # Date
    date_str = comp.get("date", "")

    # Status
    status = comp.get("status", {})
    status_type = status.get("type", {}).get("description", "")

    lines.append(f"МАТЧ: {home_en} vs {away_en}")
    if venue_name:
        lines.append(f"Стадион: {venue_name}" + (f", {city}" if city else ""))
    if date_str:
        lines.append(f"Дата: {format_msk(date_str)}")
    if status_type:
        lines.append(f"Статус: {status_type}")

    # Headlines / news
    articles = detail.get("article", {}).get("articles", [])
    for art in articles[:3]:
        headline = art.get("headline", "")
        if headline:
            lines.append(f"Новость: {headline[:150]}")

    # Key events / storylines
    storylines = detail.get("header", {}).get("storylines", [])
    for s in storylines[:3]:
        title = s.get("title", "")
        if title:
            lines.append(f"Факт: {title[:150]}")

    return "\n".join(lines)


def _fetch_espn_team_schedule(team_id: str) -> Optional[str]:
    """Fetch team's recent results from ESPN JSON API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/teams/{team_id}/schedule"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "FootballAI/1.0"})
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    events = data.get("events", [])
    if not events:
        return None

    lines = []
    for ev in events[-5:]:  # Last 5 matches
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        sides = comp.get("competitors", [])
        if len(sides) < 2:
            continue

        home = next((c for c in sides if c.get("homeAway") == "home"), sides[0])
        away = next((c for c in sides if c.get("homeAway") == "away"), sides[1])

        home_name = (home.get("team") or {}).get("shortDisplayName", "")
        away_name = (away.get("team") or {}).get("shortDisplayName", "")

        # Handle score: can be dict, string, or number
        home_score_raw = home.get("score", "")
        away_score_raw = away.get("score", "")
        if isinstance(home_score_raw, dict):
            home_score = str(home_score_raw.get("displayValue", home_score_raw.get("value", "")))
        else:
            home_score = str(home_score_raw) if home_score_raw else ""
        if isinstance(away_score_raw, dict):
            away_score = str(away_score_raw.get("displayValue", away_score_raw.get("value", "")))
        else:
            away_score = str(away_score_raw) if away_score_raw else ""

        date = ev.get("date", "")[:10]
        status = comp.get("status", {}).get("type", {}).get("state", "")

        # Safe comparison
        try:
            h_int = int(home_score) if home_score.isdigit() else 0
            a_int = int(away_score) if away_score.isdigit() else 0
        except (ValueError, TypeError):
            h_int = a_int = 0

        result = "W" if status == "post" and h_int > a_int else \
                 "L" if status == "post" and h_int < a_int else \
                 "D" if status == "post" and h_int == a_int else \
                 "-" if status == "post" else "•"

        score = f"{home_score}:{away_score}" if home_score and away_score else "vs"
        lines.append(f"  {date} | {home_name} {score} {away_name} | {result}")

    return "\n".join(lines) if lines else None


_TEAM_ID_CACHE: Dict[str, str] = {}


def _espn_team_id(name: str) -> str:
    """Find team ID by searching ESPN API dynamically. Cached."""
    q = name.lower().strip()
    if q in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[q]

    _LEAGUES = [
        "fifa.world", "uefa.nations",
        "eng.1", "esp.1", "ger.1", "ita.1", "fra.1",
        "ned.1", "por.1", "bel.1", "tur.1", "sui.1",
        "usa.1", "bra.1", "arg.1", "rus.1",
        "eng.2", "esp.2", "ger.2", "ita.2", "fra.2",
        "sco.1", "gre.1", "aut.1", "den.1", "nor.1", "swe.1",
        "ukr.1", "pol.1", "cze.1", "croat.1", "srb.1",
        "rom.1", "bul.1", "hun.1", "fin.1", "ice.1", "irl.1",
        "mex.1", "chi.1", "col.1", "par.1", "uru.1", "ecu.1",
        "ksa.1", "uae.1", "qat.1", "jpn.1", "chn.1", "kor.1", "aus.1",
        "mar.1", "tun.1", "egy.1",
    ]
    for league in _LEAGUES:
        try:
            resp = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams",
                timeout=6, headers={"User-Agent": "FootballAI/1.0"},
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            teams = (data.get("sports") or [{}])[0].get("leagues") or [{}]
            if not teams:
                continue
            for t in teams[0].get("teams") or []:
                team = t.get("team", {})
                dn = (team.get("displayName") or "").lower()
                sn = (team.get("shortDisplayName") or "").lower()
                if q == dn or q == sn or q in dn or dn in q:
                    tid = str(team["id"])
                    _TEAM_ID_CACHE[q] = tid
                    return tid
        except Exception:
            continue
    return "0"
