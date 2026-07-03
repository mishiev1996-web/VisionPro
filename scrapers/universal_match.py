"""
scrapers/universal_match.py — Universal match data scraper.

Automatically finds and parses match data from multiple sources:
  - championat.com — previews, lineups, analytics
  - sports.ru — stats, news
  - euro-football.ru — lineups
  - bombardir.ru — lineups
  - flashscorekz.com — live scores
  - ESPN — stats, form
  - Wikipedia — history, rankings
  - Jina Reader — any URL → clean text

Usage:
    from scrapers.universal_match import gather_all_match_data
    data = gather_all_match_data("France", "Iraq")
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scrapers.web import fetch_url, search


# ── Source parsers ──────────────────────────────────────────────────────────

def _parse_championat(text: str, url: str) -> Dict[str, Any]:
    """Parse championat.com match page."""
    result = {"source": "championat.com", "url": url, "preview": "", "lineups": "", "stats": "", "news": ""}

    lines = text.split("\n")
    buf = []

    for line in lines:
        line_s = line.strip()
        if not line_s or len(line_s) < 15:
            continue
        lower = line_s.lower()

        if any(kw in lower for kw in ["состав", "стартовый", "lineup"]):
            if buf:
                result["lineups"] += "\n".join(buf) + "\n"
            buf = []
        elif any(kw in lower for kw in ["статистика", "владение", "удары", "угловые"]):
            if buf:
                result["stats"] += "\n".join(buf) + "\n"
            buf = []

        buf.append(line_s[:250])

    if buf:
        result["preview"] = "\n".join(buf)[:3000]
    return result


def _parse_sports_ru(text: str, url: str) -> Dict[str, Any]:
    """Parse sports.ru match page."""
    result = {"source": "sports.ru", "url": url, "preview": "", "lineups": "", "stats": "", "news": ""}

    lines = text.split("\n")
    buf = []

    for line in lines:
        line_s = line.strip()
        if not line_s or len(line_s) < 15:
            continue
        lower = line_s.lower()

        if any(kw in lower for kw in ["состав", "lineup", "стартовый"]):
            if buf:
                result["lineups"] += "\n".join(buf) + "\n"
            buf = []
        elif any(kw in lower for kw in ["статистика", "владение", "удары"]):
            if buf:
                result["stats"] += "\n".join(buf) + "\n"
            buf = []

        buf.append(line_s[:250])

    if buf:
        result["preview"] = "\n".join(buf)[:3000]
    return result


def _parse_generic(text: str, url: str) -> Dict[str, Any]:
    """Generic parser for any URL — extracts useful text."""
    result = {"source": url.split("/")[2] if "/" in url else "unknown", "url": url,
              "preview": "", "lineups": "", "stats": "", "news": ""}

    lines = text.split("\n")
    buf = []

    for line in lines:
        line_s = line.strip()
        if not line_s or len(line_s) < 15:
            continue
        lower = line_s.lower()

        if any(kw in lower for kw in ["состав", "lineup", "стартовый", "formation"]):
            if buf:
                result["lineups"] += "\n".join(buf) + "\n"
            buf = []
        elif any(kw in lower for kw in ["статистика", "stat", "владение", "удары", "shots"]):
            if buf:
                result["stats"] += "\n".join(buf) + "\n"
            buf = []

        buf.append(line_s[:250])

    if buf:
        result["preview"] = "\n".join(buf)[:3000]
    return result


# ── URL finders ─────────────────────────────────────────────────────────────

def _build_match_urls(home_en: str, away_en: str) -> List[str]:
    """Build candidate URLs for a match on various Russian football sites."""
    h = home_en.lower().replace(" ", "-")
    a = away_en.lower().replace(" ", "-")

    return [
        f"https://www.championat.com/football/_worldcup/match/{h}-vs-{a}/",
        f"https://www.championat.com/football/match/{h}-vs-{a}/",
        f"https://www.sports.ru/football/match/{h}-vs-{a}/stat/",
        f"https://www.sports.ru/football/match/{h}-vs-{a}/",
        f"https://bombardir.ru/online/{h}-vs-{a}/lineups",
        f"https://bombardir.ru/online/{h}-vs-{a}/",
        f"https://euro-football.ru/online/{h}-{a}/lineups",
        f"https://euro-football.ru/online/{h}-{a}/",
    ]


def _search_match_urls(home_en: str, away_en: str) -> List[str]:
    """Search for match URLs on Russian football sites."""
    urls = []
    try:
        results = search(f"{home_en} vs {away_en} матч статистика состав", num_results=10)
        for r in results:
            url = r.get("url", "")
            if any(domain in url for domain in [
                "championat.com", "sports.ru", "bombardir.ru",
                "euro-football.ru", "flashscorekz.com"
            ]):
                urls.append(url)
    except Exception:
        pass
    return urls[:5]


# ── Universal fetcher ───────────────────────────────────────────────────────

def _fetch_and_parse(url: str) -> Optional[Dict[str, Any]]:
    """Fetch URL and parse with appropriate parser."""
    text = fetch_url(url, max_chars=6000)
    if not text or len(text) < 100:
        return None

    if "championat.com" in url:
        return _parse_championat(text, url)
    elif "sports.ru" in url:
        return _parse_sports_ru(text, url)
    else:
        return _parse_generic(text, url)


# ── Main function ───────────────────────────────────────────────────────────

def gather_all_match_data(home_en: str, away_en: str,
                          progress_cb=None) -> Dict[str, Any]:
    """
    Universal match data gathering.

    Returns:
        {
            "home": str, "away": str,
            "sources": [{"source", "url", "preview", "lineups", "stats", "news"}],
            "preview": str,    # combined previews
            "lineups": str,    # combined lineups
            "stats": str,      # combined stats
            "news": str,       # combined news
            "total_chars": int,
        }
    """
    if progress_cb:
        progress_cb({"type": "info", "msg": f"Собираю данные: {home_en} vs {away_en}…"})

    # 1. Build candidate URLs
    candidate_urls = _build_match_urls(home_en, away_en)

    # 2. Search for additional URLs
    search_urls = _search_match_urls(home_en, away_en)

    # 3. Merge and deduplicate
    all_urls = list(dict.fromkeys(candidate_urls + search_urls))

    # 4. Fetch and parse each
    sources = []
    for url in all_urls[:8]:  # Max 8 sources
        try:
            data = _fetch_and_parse(url)
            if data and (data.get("preview") or data.get("lineups") or data.get("stats")):
                sources.append(data)
                if progress_cb:
                    preview_len = len(data.get("preview", ""))
                    stats_len = len(data.get("stats", ""))
                    progress_cb({"type": "success",
                                "msg": f"  {data['source']}: preview={preview_len} stats={stats_len}"})
        except Exception as e:
            if progress_cb:
                progress_cb({"type": "error", "msg": f"  {url[:50]}: {e}"})

    # 5. Combine results
    combined = {
        "home": home_en,
        "away": away_en,
        "sources": sources,
        "preview": "",
        "lineups": "",
        "stats": "",
        "news": "",
        "total_chars": 0,
    }

    for s in sources:
        combined["preview"] += s.get("preview", "") + "\n"
        combined["lineups"] += s.get("lineups", "") + "\n"
        combined["stats"] += s.get("stats", "") + "\n"
        combined["news"] += s.get("news", "") + "\n"
        combined["total_chars"] += s.get("raw_length", 0) if "raw_length" in s else len(s.get("preview", ""))

    # Trim to reasonable limits
    combined["preview"] = combined["preview"][:5000]
    combined["lineups"] = combined["lineups"][:3000]
    combined["stats"] = combined["stats"][:3000]
    combined["news"] = combined["news"][:1000]

    if progress_cb:
        progress_cb({"type": "success",
                     "msg": f"Собрано: {len(sources)} источников, "
                            f"{len(combined['preview'])}+{len(combined['lineups'])}+{len(combined['stats'])} символов"})

    return combined
