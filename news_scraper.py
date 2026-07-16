"""
news_scraper.py — Gather football news from RSS feeds.

All functions return structured data for LLM analysis.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests


HEADERS = {"User-Agent": "FootballAI/1.0"}


# ── RSS Feeds ───────────────────────────────────────────────────────────────

FOOTBALL_RSS = [
    # Fast + reliable
    {"name": "ESPN FC", "url": "https://www.espn.com/espn/rss/soccer/news"},
    {"name": "Sky Sports", "url": "https://www.skysports.com/rss/12040"},
    # Extended sources for broader coverage
    {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/football/rss.xml"},
    {"name": "The Guardian", "url": "https://www.theguardian.com/football/rss"},
    {"name": "L'Equipe", "url": "https://www.lequipe.fr/rss/Football.xml"},
]


def fetch_rss_feed(url: str, limit: int = 15) -> List[Dict[str, str]]:
    """Fetch and parse an RSS feed."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []

        root = ElementTree.fromstring(resp.content)
        items = []

        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")

            desc_clean = re.sub(r'<[^>]+>', '', desc).strip()

            if title:
                items.append({
                    "title": title,
                    "text": desc_clean[:500] if desc_clean else title,
                    "url": link,
                    "date": pub_date,
                })

        return items
    except Exception:
        return []


def fetch_all_football_news(match_query: str = "", limit_per_source: int = 10) -> List[Dict]:
    """Fetch news from all football RSS sources, filtered by match query."""
    all_news = []

    for source in FOOTBALL_RSS:
        items = fetch_rss_feed(source["url"], limit=limit_per_source)
        for item in items:
            item["source_name"] = source["name"]
            all_news.append(item)
        time.sleep(0.2)

    if match_query:
        query_lower = match_query.lower()
        keywords = [w.strip() for w in query_lower.split() if len(w) > 2]
        filtered = []
        for news in all_news:
            text = (news.get("title", "") + " " + news.get("text", "")).lower()
            if any(kw in text for kw in keywords):
                filtered.append(news)
        return filtered[:50]

    return all_news[:50]


def gather_match_news(home_name: str, away_name: str, progress_cb=None) -> Dict[str, Any]:
    """Gather all news about a match from RSS sources."""
    query = f"{home_name} {away_name}"
    data = {"home": home_name, "away": away_name, "news": []}

    if progress_cb:
        progress_cb({"type": "info", "msg": "RSS: ищу новости…"})

    news = fetch_all_football_news(query, limit_per_source=3)
    data["news"] = news
    if progress_cb:
        progress_cb({"type": "success", "msg": f"RSS: {len(news)} новостей"})

    return data


def format_news_for_llm(data: Dict[str, Any]) -> str:
    """Format news data for LLM consumption."""
    parts = []

    news = data.get("news", [])
    if news:
        parts.append("=== НОВОСТИ ===")
        for n in news[:10]:
            source = n.get("source_name", "")
            title = n.get("title", "")
            text = n.get("text", "")[:200]
            parts.append(f"[{source}] {title}")
            if text:
                parts.append(f"  {text}")
            parts.append("")

    return "\n".join(parts) if parts else "Новости не найдены"
