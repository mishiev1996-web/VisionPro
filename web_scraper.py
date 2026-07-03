"""
web_scraper.py — Smart football data gathering with fallback chain.

Strategy (fast → slow):
1. ESPN team search API → find team ID → get schedule
2. Wikipedia → basic info
3. LLM general knowledge
"""
from __future__ import annotations

import datetime as dt
import json
import re
import ssl
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

import requests

_CTX = ssl.create_default_context()
_HEADERS = {"User-Agent": "FootballAI/1.0"}


def _extract_score(raw) -> str:
    """Extract score from ESPN's variable format: dict, string, or number."""
    if isinstance(raw, dict):
        return str(raw.get("displayValue", raw.get("value", "")))
    return str(raw) if raw else ""


def _http_get(url: str, timeout: int = 10) -> Optional[dict]:
    """Safe HTTP GET returning parsed JSON."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        pass
    return None


# ── ESPN: team search + schedule ────────────────────────────────────────────

_ESPN_ALL_LEAGUES = [
    "fifa.world", "uefa.nations", "fifa.worldq.uefa",
    "fifa.worldq.conmebol", "fifa.worldq.afc", "fifa.worldq.caf",
    "uefa.euro",
    "eng.1", "esp.1", "ger.1", "ita.1", "fra.1",
    "ned.1", "por.1", "bel.1", "tur.1", "sui.1",
    "usa.1", "bra.1", "arg.1", "rus.1",
    "eng.2", "esp.2", "ger.2", "ita.2", "fra.2",
    "sco.1", "gre.1", "aut.1", "den.1", "nor.1", "swe.1",
    "ukr.1", "pol.1", "cze.1", "croat.1", "srb.1",
    "rom.1", "bul.1", "hun.1", "fin.1", "ice.1", "irl.1",
    "mex.1", "chi.1", "col.1", "ksa.1", "uae.1", "qat.1",
    "jpn.1", "chn.1", "kor.1", "aus.1",
    "mar.1", "tun.1", "egy.1",
]


_ESPN_CACHE: Dict[str, Dict] = {}


def _espn_search_team(team_name: str) -> Optional[Dict]:
    """Find team in ESPN by name. Returns team dict with id, name, league."""
    q = team_name.lower().strip()
    if q in _ESPN_CACHE:
        return _ESPN_CACHE[q]

    for league in _ESPN_ALL_LEAGUES:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams"
        data = _http_get(url, timeout=6)
        if not data:
            continue

        sports = data.get("sports") or [{}]
        if not sports:
            continue
        leagues_data = sports[0].get("leagues") or [{}]
        if not leagues_data:
            continue
        teams = leagues_data[0].get("teams") or []

        for team in teams:
            t = team.get("team", {})
            name = (t.get("displayName") or "").lower()
            short = (t.get("shortDisplayName") or "").lower()

            if q == name or q == short:
                result = {"id": t["id"], "name": t.get("displayName", team_name), "league": league}
                _ESPN_CACHE[q] = result
                return result
            if q in name or name in q:
                return {"id": t["id"], "name": t.get("displayName", team_name), "league": league}

    return None


def _espn_team_schedule(team_id: str, league: str, limit: int = 15) -> List[Dict]:
    """Get recent + upcoming matches for a team from ESPN."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/schedule"
    data = _http_get(url)
    if not data:
        return []

    matches = []
    events = data.get("events") or []

    for ev in events[-limit:]:  # Last N events
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        sides = comp.get("competitors") or []
        if len(sides) < 2:
            continue

        home = sides[0] if sides[0].get("homeAway") == "home" else sides[1]
        away = sides[1] if sides[0].get("homeAway") == "home" else sides[0]

        home_name = (home.get("team") or {}).get("displayName", "")
        away_name = (away.get("team") or {}).get("displayName", "")
        home_score = _extract_score(home.get("score", ""))
        away_score = _extract_score(away.get("score", ""))

        status_type = (comp.get("status") or {}).get("type", {})
        status = status_type.get("state", "")
        finished = status == "post"

        matches.append({
            "date": (ev.get("date") or "")[:10],
            "home": home_name,
            "away": away_name,
            "score_home": home_score,
            "score_away": away_score,
            "finished": finished,
        })

    return matches


def _espn_search_date_range(league: str, team_name: str,
                             days_back: int = 30, days_forward: int = 7) -> List[Dict]:
    """Search ESPN scoreboard for matches on specific dates.

    Optimized: searches in weekly chunks, stops early when enough data found.
    """
    q = team_name.lower()
    today = dt.date.today()
    matches = []
    checked = set()

    # Search in weekly chunks for efficiency
    for week_start in range(0, days_back, 7):
        for delta in range(min(7, days_back - week_start)):
            d = today - dt.timedelta(days=week_start + delta)
            date_key = d.isoformat()
            if date_key in checked:
                continue
            checked.add(date_key)

            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={d.strftime('%Y%m%d')}"
            data = _http_get(url, timeout=5)
            if not data:
                continue

            for ev in (data.get("events") or []):
                comps = ev.get("competitions") or []
                if not comps:
                    continue
                comp = comps[0]
                sides = comp.get("competitors") or []
                if len(sides) < 2:
                    continue

                home = sides[0] if sides[0].get("homeAway") == "home" else sides[1]
                away = sides[1] if sides[0].get("homeAway") == "home" else sides[0]

                home_name = (home.get("team") or {}).get("displayName", "")
                away_name = (away.get("team") or {}).get("displayName", "")

                if q in home_name.lower() or q in away_name.lower():
                    home_score = _extract_score(home.get("score", ""))
                    away_score = _extract_score(away.get("score", ""))
                    status_type = (comp.get("status") or {}).get("type", {})
                    state = status_type.get("state", "")
                    finished = state == "post" or (home_score and away_score and state != "pre")

                    matches.append({
                        "date": (ev.get("date") or "")[:10],
                        "home": home_name,
                        "away": away_name,
                        "score_home": home_score,
                        "score_away": away_score,
                        "finished": finished,
                    })

        if len(matches) >= 10:
            break

    return matches


def _calculate_stats(matches: List[Dict], team_name: str) -> Dict[str, Any]:
    """Calculate concrete stats from matches."""
    q = team_name.lower()
    finished = [m for m in matches if m["finished"] and m["score_home"]]

    stats = {
        "total": len(finished),
        "wins": 0, "draws": 0, "losses": 0,
        "gf": 0, "ga": 0,
        "form": [],
        "results": [],
    }

    for m in finished:
        try:
            sh, sa = int(m["score_home"]), int(m["score_away"])
        except (ValueError, TypeError):
            continue

        is_home = q in m["home"].lower()
        gf, ga = (sh, sa) if is_home else (sa, sh)

        stats["gf"] += gf
        stats["ga"] += ga

        if gf > ga:
            stats["wins"] += 1
            result = "W"
        elif gf == ga:
            stats["draws"] += 1
            result = "D"
        else:
            stats["losses"] += 1
            result = "L"

        opponent = m["away"] if is_home else m["home"]
        stats["results"].append({
            "date": m["date"],
            "opponent": opponent,
            "score": f"{sh}:{sa}",
            "result": result,
        })

    # Form (last 5)
    stats["form"] = [r["result"] for r in stats["results"][:5]]

    # Averages
    if stats["total"] > 0:
        stats["avg_gf"] = round(stats["gf"] / stats["total"], 2)
        stats["avg_ga"] = round(stats["ga"] / stats["total"], 2)
        stats["win_pct"] = round(stats["wins"] / stats["total"] * 100, 1)

    return stats


# ── Wikipedia fallback ──────────────────────────────────────────────────────

def _wikipedia_search(team_name: str) -> Optional[str]:
    """Get team info from Wikipedia."""
    for query in [
        f"{team_name} national football team",
        f"{team_name} football club",
        team_name,
    ]:
        try:
            slug = query.replace(" ", "_")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
            resp = requests.get(url, timeout=8, headers=_HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("extract", "")
                if text and len(text) > 50:
                    if any(kw in text.lower() for kw in ["football", "soccer", "league", "club", "team"]):
                        return text[:1500]
        except Exception:
            pass
        time.sleep(0.2)
    return None


# ── Main gatherer with fallback chain ───────────────────────────────────────

def gather_team_data(team_name: str, progress_cb=None) -> Dict[str, Any]:
    """Gather data about a team. Tries multiple sources until successful.

    Chain: ESPN team search → ESPN schedule → ESPN date search
    Wikipedia removed — stale data.
    """
    data = {"name": team_name, "stats": None, "matches": [], "source": "none"}

    # 1. Try ESPN: find team + get schedule
    if progress_cb:
        progress_cb({"type": "info", "msg": f"ESPN: ищу {team_name}…"})

    team_info = _espn_search_team(team_name)
    if team_info:
        if progress_cb:
            progress_cb({"type": "info", "msg": f"Нашёл: {team_info['name']} ({team_info['league']})"})

        # Try schedule first
        matches = _espn_team_schedule(team_info["id"], team_info["league"], limit=15)

        # If no finished matches (common for national teams), search by dates
        finished = [m for m in matches if m["finished"]]
        if len(finished) < 3:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"Ищу матчи по датам…"})

            date_matches = _espn_search_date_range(
                team_info["league"], team_name,
                days_back=30, days_forward=7
            )
            # Merge, avoiding duplicates
            existing_dates = {(m["home"], m["away"]) for m in matches}
            for dm in date_matches:
                if (dm["home"], dm["away"]) not in existing_dates:
                    matches.append(dm)

        if matches:
            stats = _calculate_stats(matches, team_name)
            data["stats"] = stats
            data["matches"] = matches
            data["source"] = "espn"
            data["team_name_en"] = team_info["name"]

            if progress_cb:
                progress_cb({"type": "success",
                             "msg": f"✓ ESPN: {stats['total']} матчей, "
                                    f"форма: {''.join(stats['form']) or '—'}"})
        else:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"ESPN: матчи не найдены"})
    else:
        if progress_cb:
            progress_cb({"type": "info", "msg": f"ESPN: команда не найдена"})

    return data


def format_data_for_llm(data: Dict[str, Any]) -> str:
    """Format data for LLM with concrete numbers."""
    parts = []

    stats = data.get("stats")
    if stats:
        name = data.get("team_name_en", data["name"])
        parts.append(f"=== {name.upper()} ===")
        parts.append(f"Матчей проанализировано: {stats['total']}")
        parts.append(f"Победы / Ничьи / Поражения: {stats['wins']} / {stats['draws']} / {stats['losses']}")
        parts.append(f"Голы за / против: {stats['gf']} / {stats['ga']}")
        parts.append(f"Средние голы за матч: {stats['avg_gf']} / {stats['avg_ga']}")
        parts.append(f"Процент побед: {stats['win_pct']}%")
        if stats["form"]:
            parts.append(f"Форма (последние 5): {' '.join(stats['form'])}")
        if stats["results"]:
            parts.append("\nПоследние матчи:")
            for r in stats["results"][:8]:
                parts.append(f"  {r['date']} | {r['opponent']} | {r['score']} | {r['result']}")

    return "\n".join(parts) if parts else "Данные не найдены"
