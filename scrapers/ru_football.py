"""
scrapers/ru_football.py — Russian football news & match data scrapers.

Sources:
  - championat.com — match previews, lineups, analytics
  - sports.ru — match stats, news
  - euro-football.ru — lineups, live
  - bombardir.ru — lineups
  - flashscorekz.com — live scores

Uses Jina Reader to parse pages into clean text.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scrapers.web import fetch_url


# ── Championat.com ──────────────────────────────────────────────────────────

def fetch_championat_match(url: str) -> Dict[str, Any]:
    """Parse championat.com match page. Returns structured data."""
    text = fetch_url(url, max_chars=8000)
    if not text:
        return {"source": "championat", "error": "page not loaded"}

    result = {
        "source": "championat",
        "url": url,
        "raw_length": len(text),
        "preview": "",
        "lineups": "",
        "stats": "",
        "news": "",
    }

    lines = text.split("\n")
    capture = None
    buf = []

    for line in lines:
        line_s = line.strip()
        if not line_s or len(line_s) < 10:
            continue

        lower = line_s.lower()

        # Section markers
        if any(kw in lower for kw in ["состав", "lineup", "предматч", "анализ", "превью", "обзор"]):
            if capture and buf:
                result[capture] = "\n".join(buf)
            capture = "preview"
            buf = []
        elif any(kw in lower for kw in ["статистика", "stat", "владение", "удары", "угловые"]):
            if capture and buf:
                result[capture] = "\n".join(buf)
            capture = "stats"
            buf = []
        elif any(kw in lower for kw in ["новости", "news", "травм", "состав宣布"]):
            if capture and buf:
                result[capture] = "\n".join(buf)
            capture = "news"
            buf = []

        if capture:
            buf.append(line_s[:300])

    if capture and buf:
        result[capture] = "\n".join(buf)

    # Extract key info from text
    result["info"] = _extract_match_info(text)

    return result


def _extract_match_info(text: str) -> Dict[str, str]:
    """Extract match info from any Russian football page."""
    info = {}

    # Date patterns
    date_match = re.search(r'(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})', text)
    if date_match:
        info["date_text"] = date_match.group(0)

    # Score patterns
    score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text[:500])
    if score_match:
        info["score"] = f"{score_match.group(1)}:{score_match.group(2)}"

    # Venue/stadium
    venue_match = re.search(r'(стадион|стадіон|stadium|арена)\s+([^\n,]+)', text, re.IGNORECASE)
    if venue_match:
        info["venue"] = venue_match.group(2).strip()

    return info


# ── Sports.ru ───────────────────────────────────────────────────────────────

def fetch_sports_ru_stat(url: str) -> Dict[str, Any]:
    """Parse sports.ru match stat page."""
    text = fetch_url(url, max_chars=8000)
    if not text:
        return {"source": "sports.ru", "error": "page not loaded"}

    result = {
        "source": "sports.ru",
        "url": url,
        "raw_length": len(text),
        "stats": "",
        "lineups": "",
        "news": "",
    }

    lines = text.split("\n")
    buf = []

    for line in lines:
        line_s = line.strip()
        if not line_s or len(line_s) < 10:
            continue
        lower = line_s.lower()

        if any(kw in lower for kw in ["состав", "lineup", "стартовый"]):
            if buf:
                result["lineups"] = "\n".join(buf)
            buf = []
        elif any(kw in lower for kw in ["статистика", "stat", "владение", "удары"]):
            if buf:
                result["stats"] = "\n".join(buf)
            buf = []

        buf.append(line_s[:300])

    if buf:
        result["stats"] = result["stats"] or "\n".join(buf)

    return result


# ── Euro-football.ru ───────────────────────────────────────────────────────

def fetch_eurofootball_lineups(url: str) -> Dict[str, Any]:
    """Parse euro-football.ru lineups page."""
    text = fetch_url(url, max_chars=8000)
    if not text:
        return {"source": "euro-football.ru", "error": "page not loaded"}

    return {
        "source": "euro-football.ru",
        "url": url,
        "raw_length": len(text),
        "lineups": text[:3000],
    }


# ── Bombardir.ru ───────────────────────────────────────────────────────────

def fetch_bombardir_lineups(url: str) -> Dict[str, Any]:
    """Parse bombardir.ru lineups page."""
    text = fetch_url(url, max_chars=8000)
    if not text:
        return {"source": "bombardir.ru", "error": "page not loaded"}

    return {
        "source": "bombardir.ru",
        "url": url,
        "raw_length": len(text),
        "lineups": text[:3000],
    }


# ── Flashscore KZ ──────────────────────────────────────────────────────────

def fetch_flashscore_match(url: str) -> Dict[str, Any]:
    """Parse flashscorekz.com match page."""
    text = fetch_url(url, max_chars=8000)
    if not text:
        return {"source": "flashscorekz", "error": "page not loaded"}

    return {
        "source": "flashscorekz",
        "url": url,
        "raw_length": len(text),
        "live_data": text[:3000],
    }


# ── Batch fetcher ──────────────────────────────────────────────────────────

def fetch_all_match_sources(urls: List[str]) -> Dict[str, Any]:
    """Fetch data from all sources for a match. Returns combined result."""
    sources = []
    for url in urls:
        if "championat.com" in url:
            data = fetch_championat_match(url)
        elif "sports.ru" in url:
            data = fetch_sports_ru_stat(url)
        elif "euro-football.ru" in url:
            data = fetch_eurofootball_lineups(url)
        elif "bombardir.ru" in url:
            data = fetch_bombardir_lineups(url)
        elif "flashscore" in url:
            data = fetch_flashscore_match(url)
        else:
            data = {"source": "unknown", "url": url, "raw_length": 0}
        sources.append(data)

    # Merge into combined context
    combined = {
        "sources": sources,
        "preview": "",
        "lineups": "",
        "stats": "",
        "news": "",
    }

    for s in sources:
        combined["preview"] += s.get("preview", "") + "\n"
        combined["lineups"] += s.get("lineups", "") + "\n"
        combined["stats"] += s.get("stats", "") + "\n"
        combined["news"] += s.get("news", "") + "\n"

    return combined
