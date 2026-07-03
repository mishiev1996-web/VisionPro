"""
scrapers/espn.py — fetch worldwide soccer matches via ESPN's public JSON API.

ESPN exposes per-league scoreboard endpoints with NO authentication:
    https://site.api.espn.com/apis/site/v2/sports/soccer/<league>/scoreboard?dates=YYYYMMDD
    https://site.api.espn.com/apis/v2/sports/soccer/leagues       — list of all known leagues

Coverage is excellent: top European leagues, Champions/Europa/Conference,
South American (Libertadores, Brasileirão, Argentina), Mexican, MLS, Saudi,
World Cup qualifiers, women's leagues, plus dozens more.

Returns matches in our normalized shape:
    {country, league, home, away, score_home, score_away, time, status}
"""
from __future__ import annotations

import datetime as dt
import json
import ssl
import urllib.error
import urllib.request
from typing import Dict, List, Optional


ESPN_BASE   = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_INDEX  = "https://site.api.espn.com/apis/v2/sports/soccer/leagues?limit=500"
_UA         = "Mozilla/5.0 (Football-AI; learning) Python-urllib"
# Some corporate / Windows environments lack root CAs; fall back gracefully.
_CTX = ssl.create_default_context()


# Major leagues we always query (ESPN league slug, country, friendly name).
# Used as a fallback if the dynamic /leagues catalog request fails.
DEFAULT_LEAGUES: List[Dict[str, str]] = [
    # Top 6
    {"slug": "eng.1",   "country": "England",     "name": "Premier League"},
    {"slug": "esp.1",   "country": "Spain",       "name": "La Liga"},
    {"slug": "ger.1",   "country": "Germany",     "name": "Bundesliga"},
    {"slug": "ita.1",   "country": "Italy",       "name": "Serie A"},
    {"slug": "fra.1",   "country": "France",      "name": "Ligue 1"},
    {"slug": "rus.1",   "country": "Russia",      "name": "Russian Premier"},
    # Second divisions
    {"slug": "eng.2",   "country": "England",     "name": "Championship"},
    {"slug": "esp.2",   "country": "Spain",       "name": "Segunda División"},
    {"slug": "ger.2",   "country": "Germany",     "name": "2. Bundesliga"},
    {"slug": "ita.2",   "country": "Italy",       "name": "Serie B"},
    {"slug": "fra.2",   "country": "France",      "name": "Ligue 2"},
    # Other European
    {"slug": "ned.1",   "country": "Netherlands", "name": "Eredivisie"},
    {"slug": "por.1",   "country": "Portugal",    "name": "Primeira Liga"},
    {"slug": "bel.1",   "country": "Belgium",     "name": "Belgian Pro League"},
    {"slug": "sco.1",   "country": "Scotland",    "name": "Premiership"},
    {"slug": "tur.1",   "country": "Turkey",      "name": "Süper Lig"},
    {"slug": "gre.1",   "country": "Greece",      "name": "Super League"},
    {"slug": "aut.1",   "country": "Austria",     "name": "Bundesliga"},
    {"slug": "sui.1",   "country": "Switzerland", "name": "Super League"},
    {"slug": "den.1",   "country": "Denmark",     "name": "Superliga"},
    {"slug": "nor.1",   "country": "Norway",      "name": "Eliteserien"},
    {"slug": "swe.1",   "country": "Sweden",      "name": "Allsvenskan"},
    {"slug": "ukr.1",   "country": "Ukraine",     "name": "Premier League"},
    {"slug": "pol.1",   "country": "Poland",      "name": "Ekstraklasa"},
    {"slug": "cze.1",   "country": "Czech Rep.",  "name": "First League"},
    {"slug": "croat.1", "country": "Croatia",     "name": "HNL"},
    {"slug": "srb.1",   "country": "Serbia",      "name": "SuperLiga"},
    {"slug": "rom.1",   "country": "Romania",     "name": "SuperLiga"},
    {"slug": "bul.1",   "country": "Bulgaria",    "name": "First League"},
    {"slug": "hun.1",   "country": "Hungary",     "name": "NB I"},
    {"slug": "slo.1",   "country": "Slovenia",    "name": "PrvaLiga"},
    {"slug": "svn.1",   "country": "Slovakia",    "name": "Super Liga"},
    {"slug": "fin.1",   "country": "Finland",     "name": "Veikkausliiga"},
    {"slug": "ice.1",   "country": "Iceland",     "name": "Úrvalsdeild"},
    {"slug": "irl.1",   "country": "Ireland",     "name": "Premier Division"},
    {"slug": "isr.1",   "country": "Israel",      "name": "Premier League"},
    {"slug": "chn.1",   "country": "China",       "name": "Super League"},
    {"slug": "jpn.1",   "country": "Japan",       "name": "J.League"},
    {"slug": "kor.1",   "country": "South Korea", "name": "K League 1"},
    # Americas
    {"slug": "usa.1",   "country": "USA",         "name": "MLS"},
    {"slug": "mex.1",   "country": "Mexico",      "name": "Liga MX"},
    {"slug": "bra.1",   "country": "Brazil",      "name": "Brasileirão"},
    {"slug": "arg.1",   "country": "Argentina",   "name": "Primera División"},
    {"slug": "chi.1",   "country": "Chile",       "name": "Primera División"},
    {"slug": "col.1",   "country": "Colombia",    "name": "Primera A"},
    {"slug": "par.1",   "country": "Paraguay",    "name": "Primera División"},
    {"slug": "uru.1",   "country": "Uruguay",     "name": "Primera División"},
    {"slug": "ecu.1",   "country": "Ecuador",     "name": "LigaPro"},
    {"slug": "per.1",   "country": "Peru",        "name": "Liga 1"},
    {"slug": "ven.1",   "country": "Venezuela",   "name": "Primera División"},
    # Middle East / Asia
    {"slug": "ksa.1",   "country": "Saudi Arabia","name": "Saudi Pro League"},
    {"slug": "uae.1",   "country": "UAE",         "name": "Pro League"},
    {"slug": "qat.1",   "country": "Qatar",       "name": "Stars League"},
    {"slug": "aus.1",   "country": "Australia",   "name": "A-League"},
    {"slug": "ind.1",   "country": "India",       "name": "ISL"},
    {"slug": "tha.1",   "country": "Thailand",    "name": "Thai League 1"},
    {"slug": "vnm.1",   "country": "Vietnam",     "name": "V.League 1"},
    {"slug": "mas.1",   "country": "Malaysia",    "name": "Super League"},
    {"slug": "phl.1",   "country": "Philippines", "name": "Philippines Football League"},
    # Africa
    {"slug": "mar.1",   "country": "Morocco",     "name": "Botola Pro"},
    {"slug": "tun.1",   "country": "Tunisia",     "name": "Ligue 1"},
    {"slug": "egy.1",   "country": "Egypt",       "name": "Premier League"},
    {"slug": "nga.1",   "country": "Nigeria",     "name": "NPFL"},
    {"slug": "rsa.1",   "country": "South Africa","name": "Premier Soccer League"},
    {"slug": "gha.1",   "country": "Ghana",       "name": "Premier League"},
    {"slug": "sen.1",   "country": "Senegal",     "name": "Ligue 1"},
    # Continental club competitions
    {"slug": "uefa.champions",          "country": "Europe", "name": "UEFA Champions League"},
    {"slug": "uefa.europa",             "country": "Europe", "name": "UEFA Europa League"},
    {"slug": "uefa.europa.conf",        "country": "Europe", "name": "UEFA Conference League"},
    {"slug": "conmebol.libertadores",   "country": "South America", "name": "Copa Libertadores"},
    {"slug": "conmebol.sudamericana",   "country": "South America", "name": "Copa Sudamericana"},
    {"slug": "concacaf.champions",      "country": "N./C. America", "name": "Champions Cup"},
    {"slug": "afc.champions",           "country": "Asia",   "name": "AFC Champions League"},
    # National team
    {"slug": "fifa.world",              "country": "World",  "name": "FIFA World Cup"},
    {"slug": "uefa.euro",               "country": "Europe", "name": "UEFA Euro"},
    {"slug": "uefa.nations",            "country": "Europe", "name": "UEFA Nations League"},
    {"slug": "fifa.worldq.uefa",        "country": "Europe", "name": "WC Qualifier UEFA"},
    {"slug": "fifa.worldq.conmebol",    "country": "South America", "name": "WC Qualifier CONMEBOL"},
    {"slug": "fifa.worldq.concacaf",    "country": "N./C. America", "name": "WC Qualifier CONCACAF"},
    {"slug": "fifa.worldq.afc",         "country": "Asia",   "name": "WC Qualifier AFC"},
    {"slug": "fifa.worldq.caf",         "country": "Africa", "name": "WC Qualifier CAF"},
    {"slug": "fifa.worldq.ofc",         "country": "Oceania","name": "WC Qualifier OFC"},
]


def _get(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15, context=_CTX) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _date_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _status_from_espn(s: dict) -> str:
    """Map ESPN status to our 3-bucket schema."""
    type_obj = (s.get("type") or {})
    state = (type_obj.get("state") or "").lower()
    if state == "in":   return "live"
    if state == "post": return "finished"
    return "scheduled"


def _parse_event(ev: dict, league_meta: Dict[str, str]) -> Optional[Dict[str, str]]:
    comps = ev.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    sides = comp.get("competitors") or []
    if len(sides) < 2:
        return None
    home = next((c for c in sides if c.get("homeAway") == "home"), sides[0])
    away = next((c for c in sides if c.get("homeAway") == "away"), sides[1])
    home_team = (home.get("team") or {}).get("displayName") or ""
    away_team = (away.get("team") or {}).get("displayName") or ""
    if not home_team or not away_team:
        return None
    status = _status_from_espn(comp.get("status") or ev.get("status") or {})
    score_home = str(home.get("score", ""))
    score_away = str(away.get("score", ""))
    # Time string: prefer status time (e.g. "65'") for live, kick-off time otherwise
    st = (comp.get("status") or {}).get("displayClock") or ""
    if status == "live" and st:
        time_str = st.strip()
    else:
        # ev["date"] is ISO with Z; extract HH:MM
        d = ev.get("date") or ""
        if "T" in d:
            time_str = d.split("T")[1][:5]
        else:
            time_str = ""
    return {
        "country":    league_meta["country"],
        "league":     league_meta["name"],
        "home":       str(home_team),
        "away":       str(away_team),
        "score_home": score_home if score_home != "" else "",
        "score_away": score_away if score_away != "" else "",
        "time":       time_str,
        "status":     status,
    }


def fetch_league_scoreboard(league_slug: str, date: dt.date,
                            league_name: str = "", country: str = "") -> List[Dict[str, str]]:
    """Matches for one league on one date. Returns [] on failure."""
    url = f"{ESPN_BASE}/{league_slug}/scoreboard?dates={_date_yyyymmdd(date)}"
    data = _get(url)
    if not data:
        return []
    events = data.get("events") or []
    league_meta = {"slug": league_slug,
                   "name": league_name or league_slug,
                   "country": country or ""}
    # Some responses include league name in "leagues" array — prefer that
    leagues_in_resp = data.get("leagues") or []
    if leagues_in_resp:
        lg = leagues_in_resp[0]
        if not league_name:
            league_meta["name"] = lg.get("name") or league_meta["name"]
    out = []
    for ev in events:
        norm = _parse_event(ev, league_meta)
        if norm:
            out.append(norm)
    return out


def fetch_dynamic_leagues() -> List[Dict[str, str]]:
    """Try ESPN's leagues catalog. Returns DEFAULT_LEAGUES if it fails."""
    data = _get(ESPN_INDEX)
    if not data or "leagues" not in data:
        return DEFAULT_LEAGUES
    out = []
    for l in data["leagues"]:
        out.append({
            "slug":    str(l.get("slug") or ""),
            "name":    str(l.get("name") or ""),
            "country": (l.get("country") or {}).get("name") or "",
        })
    return [x for x in out if x["slug"]] or DEFAULT_LEAGUES


def fetch_all_today(progress_cb=None) -> List[Dict[str, str]]:
    """Hit every default league for today's date. Used for 'live + today' coverage."""
    out = []
    today = dt.date.today()
    for lg in DEFAULT_LEAGUES:
        items = fetch_league_scoreboard(lg["slug"], today, lg["name"], lg["country"])
        if items and progress_cb:
            progress_cb({"type": "info",
                         "msg": f"  ESPN · {lg['country']} {lg['name']}: {len(items)} матчей"})
        out.extend(items)
    return out


def fetch_week(progress_cb=None, cancel_event=None) -> List[Dict[str, str]]:
    """Bigger sweep: yesterday + today + next 6 days, every default league."""
    import time as _time
    out: List[Dict[str, str]] = []
    today = dt.date.today()
    for offset in range(-1, 7):
        if cancel_event and cancel_event.is_set():
            break
        d = today + dt.timedelta(days=offset)
        day_total = 0
        for lg in DEFAULT_LEAGUES:
            if cancel_event and cancel_event.is_set():
                break
            items = fetch_league_scoreboard(lg["slug"], d, lg["name"], lg["country"])
            if items:
                out.extend(items)
                day_total += len(items)
            _time.sleep(0.15)
        if progress_cb:
            progress_cb({"type": "info",
                         "msg": f"  ESPN · {d.isoformat()}: {day_total} матчей"})
    return out


def fetch_live(progress_cb=None) -> List[Dict[str, str]]:
    """Today's matches across every default league, filtered to status=live."""
    import time as _time
    today = dt.date.today()
    out = []
    for lg in DEFAULT_LEAGUES:
        for m in fetch_league_scoreboard(lg["slug"], today, lg["name"], lg["country"]):
            if m["status"] == "live":
                out.append(m)
        _time.sleep(0.15)
    return out
