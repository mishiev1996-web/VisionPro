"""
scrapers/transfermarkt.py — fetch current injuries via Botasaurus @browser.

Transfermarkt sits behind Cloudflare with a JS challenge, so we use the
browser-based decorator. The URL pattern for a club's injury page is:
    https://www.transfermarkt.com/<team-slug>/sperrenundverletzungen/verein/<id>

Maintenance: when a team isn't in TEAM_URL_MAP it will be skipped.
The map covers the ~120 teams in our top-6 leagues.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from botasaurus.browser import browser, Driver
    from botasaurus.soupify import soupify
    HAS_BOTASAURUS = True
except ImportError:
    HAS_BOTASAURUS = False


TM_BASE = "https://www.transfermarkt.com"


# Understat-name → Transfermarkt URL slug (path part /<slug>/sperrenundverletzungen/verein/<id>)
TEAM_URL_MAP: Dict[str, str] = {
    # EPL
    "Arsenal": "fc-arsenal/sperrenundverletzungen/verein/11",
    "Aston Villa": "aston-villa/sperrenundverletzungen/verein/405",
    "Bournemouth": "afc-bournemouth/sperrenundverletzungen/verein/989",
    "Brentford": "fc-brentford/sperrenundverletzungen/verein/1148",
    "Brighton": "brighton-amp-hove-albion/sperrenundverletzungen/verein/1237",
    "Burnley": "fc-burnley/sperrenundverletzungen/verein/1132",
    "Chelsea": "fc-chelsea/sperrenundverletzungen/verein/631",
    "Crystal Palace": "crystal-palace/sperrenundverletzungen/verein/873",
    "Everton": "fc-everton/sperrenundverletzungen/verein/29",
    "Fulham": "fc-fulham/sperrenundverletzungen/verein/931",
    "Leeds": "leeds-united/sperrenundverletzungen/verein/399",
    "Liverpool": "fc-liverpool/sperrenundverletzungen/verein/31",
    "Manchester City": "manchester-city/sperrenundverletzungen/verein/281",
    "Manchester United": "manchester-united/sperrenundverletzungen/verein/985",
    "Newcastle United": "newcastle-united/sperrenundverletzungen/verein/762",
    "Nottingham Forest": "nottingham-forest/sperrenundverletzungen/verein/703",
    "Sunderland": "afc-sunderland/sperrenundverletzungen/verein/289",
    "Tottenham": "tottenham-hotspur/sperrenundverletzungen/verein/148",
    "West Ham": "west-ham-united/sperrenundverletzungen/verein/379",
    "Wolverhampton Wanderers": "wolverhampton-wanderers/sperrenundverletzungen/verein/543",
    # La Liga
    "Real Madrid": "real-madrid/sperrenundverletzungen/verein/418",
    "Barcelona": "fc-barcelona/sperrenundverletzungen/verein/131",
    "Atletico Madrid": "atletico-madrid/sperrenundverletzungen/verein/13",
    "Athletic Club": "athletic-bilbao/sperrenundverletzungen/verein/621",
    "Real Sociedad": "real-sociedad-san-sebastian/sperrenundverletzungen/verein/681",
    "Real Betis": "real-betis-sevilla/sperrenundverletzungen/verein/150",
    "Sevilla": "fc-sevilla/sperrenundverletzungen/verein/368",
    "Villarreal": "fc-villarreal/sperrenundverletzungen/verein/1050",
    "Valencia": "fc-valencia/sperrenundverletzungen/verein/1049",
    "Celta Vigo": "celta-vigo/sperrenundverletzungen/verein/940",
    "Espanyol": "espanyol-barcelona/sperrenundverletzungen/verein/714",
    "Getafe": "fc-getafe/sperrenundverletzungen/verein/3709",
    "Girona": "fc-girona/sperrenundverletzungen/verein/12321",
    "Mallorca": "rcd-mallorca/sperrenundverletzungen/verein/237",
    "Osasuna": "ca-osasuna/sperrenundverletzungen/verein/331",
    "Rayo Vallecano": "rayo-vallecano/sperrenundverletzungen/verein/367",
    "Alaves": "deportivo-alaves/sperrenundverletzungen/verein/1108",
    "Oviedo": "real-oviedo/sperrenundverletzungen/verein/2497",
    "Levante": "ud-levante/sperrenundverletzungen/verein/3368",
    "Elche": "fc-elche/sperrenundverletzungen/verein/1531",
    # Bundesliga
    "Bayern Munich": "fc-bayern-munchen/sperrenundverletzungen/verein/27",
    "Borussia Dortmund": "borussia-dortmund/sperrenundverletzungen/verein/16",
    "RB Leipzig": "rasenballsport-leipzig/sperrenundverletzungen/verein/23826",
    "Bayer Leverkusen": "bayer-04-leverkusen/sperrenundverletzungen/verein/15",
    "Eintracht Frankfurt": "eintracht-frankfurt/sperrenundverletzungen/verein/24",
    "VfB Stuttgart": "vfb-stuttgart/sperrenundverletzungen/verein/79",
    "Borussia M.Gladbach": "borussia-monchengladbach/sperrenundverletzungen/verein/18",
    "Werder Bremen": "sv-werder-bremen/sperrenundverletzungen/verein/86",
    "VfL Wolfsburg": "vfl-wolfsburg/sperrenundverletzungen/verein/82",
    "1.FC Union Berlin": "1-fc-union-berlin/sperrenundverletzungen/verein/89",
    "FC Augsburg": "fc-augsburg/sperrenundverletzungen/verein/167",
    "1.FSV Mainz 05": "1-fsv-mainz-05/sperrenundverletzungen/verein/39",
    "TSG Hoffenheim": "tsg-1899-hoffenheim/sperrenundverletzungen/verein/533",
    "1.FC Heidenheim": "1-fc-heidenheim-1846/sperrenundverletzungen/verein/2036",
    "FC St. Pauli": "fc-st-pauli/sperrenundverletzungen/verein/35",
    "Hamburger SV": "hamburger-sv/sperrenundverletzungen/verein/41",
    "SC Freiburg": "sc-freiburg/sperrenundverletzungen/verein/60",
    "1.FC Koln": "1-fc-koln/sperrenundverletzungen/verein/3",
    # Serie A
    "Inter": "inter-mailand/sperrenundverletzungen/verein/46",
    "AC Milan": "ac-mailand/sperrenundverletzungen/verein/5",
    "Juventus": "juventus-turin/sperrenundverletzungen/verein/506",
    "Napoli": "ssc-neapel/sperrenundverletzungen/verein/6195",
    "Atalanta": "atalanta-bergamo/sperrenundverletzungen/verein/800",
    "Roma": "as-rom/sperrenundverletzungen/verein/12",
    "Lazio": "lazio-rom/sperrenundverletzungen/verein/398",
    "Fiorentina": "ac-florenz/sperrenundverletzungen/verein/430",
    "Bologna": "fc-bologna/sperrenundverletzungen/verein/1025",
    "Torino": "fc-turin/sperrenundverletzungen/verein/416",
    "Udinese": "udinese-calcio/sperrenundverletzungen/verein/410",
    "Genoa": "cfc-genua/sperrenundverletzungen/verein/252",
    "Cagliari": "cagliari-calcio/sperrenundverletzungen/verein/1390",
    "Hellas Verona": "hellas-verona/sperrenundverletzungen/verein/276",
    "Sassuolo": "us-sassuolo-calcio/sperrenundverletzungen/verein/6574",
    "Lecce": "us-lecce/sperrenundverletzungen/verein/1005",
    "Como": "como-1907/sperrenundverletzungen/verein/1047",
    "Parma": "parma-calcio-1913/sperrenundverletzungen/verein/130",
    "Pisa": "pisa-sporting-club/sperrenundverletzungen/verein/204",
    "Cremonese": "us-cremonese/sperrenundverletzungen/verein/2239",
    # Ligue 1
    "Paris Saint Germain": "fc-paris-saint-germain/sperrenundverletzungen/verein/583",
    "Marseille": "olympique-marseille/sperrenundverletzungen/verein/244",
    "Monaco": "as-monaco/sperrenundverletzungen/verein/162",
    "Lille": "osc-lille/sperrenundverletzungen/verein/1082",
    "Lyon": "olympique-lyon/sperrenundverletzungen/verein/1041",
    "Nice": "ogc-nizza/sperrenundverletzungen/verein/417",
    "Rennes": "stade-rennes/sperrenundverletzungen/verein/273",
    "Lens": "rc-lens/sperrenundverletzungen/verein/826",
    "Nantes": "fc-nantes/sperrenundverletzungen/verein/995",
    "Toulouse": "fc-toulouse/sperrenundverletzungen/verein/415",
    "Strasbourg": "rc-strassburg/sperrenundverletzungen/verein/667",
    "Brest": "stade-brest-29/sperrenundverletzungen/verein/3911",
    "Auxerre": "aj-auxerre/sperrenundverletzungen/verein/290",
    "Angers": "sco-angers/sperrenundverletzungen/verein/1420",
    "Le Havre": "le-havre-ac/sperrenundverletzungen/verein/738",
    "Metz": "fc-metz/sperrenundverletzungen/verein/347",
    "Paris FC": "fc-paris/sperrenundverletzungen/verein/2143",
    "Lorient": "fc-lorient/sperrenundverletzungen/verein/1158",
    # Eredivisie
    "Ajax": "ajax-amsterdam/sperrenundverletzungen/verein/615",
    "PSV": "psv-eindhoven/sperrenundverletzungen/verein/367",
    "Feyenoord": "feyenoord-rotterdam/sperrenundverletzungen/verein/234",
    "AZ": "az-alkmaar/sperrenundverletzungen/verein/13",
    "Twente": "fc-twente-enschede/sperrenundverletzungen/verein/362",
    # Primeira Liga
    "Benfica": "sl-benfica/sperrenundverletzungen/verein/294",
    "Porto": "fc-porto/sperrenundverletzungen/verein/518",
    "Sporting CP": "sporting-lissabon/sperrenundverletzungen/verein/336",
    "Braga": "sc-braga/sperrenundverletzungen/verein/1178",
    # Süper Lig
    "Galatasaray": "galatasaray-istanbul/sperrenundverletzungen/verein/141",
    "Fenerbahce": "fenerbahce-istanbul/sperrenundverletzungen/verein/36",
    "Besiktas": "besiktas-istanbul/sperrenundverletzungen/verein/114",
    "Trabzonspor": "trabzonspor/sperrenundverletzungen/verein/154",
    # Belgian Pro League
    "Club Brugge": "fc-brugge/sperrenundverletzungen/verein/624",
    "Anderlecht": "rsc-anderlecht/sperrenundverletzungen/verein/35",
    "Genk": "krc-genk/sperrenundverletzungen/verein/562",
}


@browser(
    headless=True,
    block_images=True,
    create_error_logs=False,
    output=None,
    max_retry=2,
    reuse_driver=True,
)
def _fetch_team_injuries_browser(driver: Driver, team_slug: str) -> List[Dict[str, str]]:
    """Open transfermarkt's injury page for a team, parse the table.

    Returns list of {player, status, since, until, reason}.
    """
    url = f"{TM_BASE}/{team_slug}"
    driver.google_get(url, bypass_cloudflare=True)
    driver.short_random_sleep()

    html = driver.page_html
    soup = soupify(html)

    out: List[Dict[str, str]] = []
    # Transfermarkt's injury table has class "items"
    table = soup.find("table", class_=re.compile(r"items"))
    if not table or not table.tbody:
        return out

    for tr in table.tbody.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 5:
            continue
        # The structure varies, but typically:
        # td[0]=number, td[1]=player+pos, td[2]=DOB/age, td[3]=reason, td[4]=since, td[5]=until
        # We use a tolerant approach
        cells = [td.get_text(strip=True, separator=" ") for td in tds]
        # Find a player name — first cell with an <a> linking to /spieler/
        player = ""
        for td in tds:
            a = td.find("a", href=re.compile(r"/spieler/|/profil/spieler/"))
            if a:
                player = a.get_text(strip=True)
                break
        if not player:
            continue

        out.append({
            "player": player.encode("ascii", "replace").decode("ascii"),
            "reason": (cells[3] if len(cells) > 3 else "").encode("ascii", "replace").decode("ascii"),
            "since":  (cells[4] if len(cells) > 4 else "").encode("ascii", "replace").decode("ascii"),
            "until":  (cells[5] if len(cells) > 5 else "").encode("ascii", "replace").decode("ascii"),
            "status": "injured",
        })
    return out


def fetch_team_injuries(team_name: str) -> List[Dict[str, str]]:
    """Public — get current injuries for one team."""
    slug = TEAM_URL_MAP.get(team_name)
    if not slug:
        return []
    try:
        return _fetch_team_injuries_browser(slug) or []
    except Exception as e:
        print(f"[transfermarkt] {team_name}: {e}")
        return []


def _safe_print(msg: str) -> None:
    """Print with fallback encoding for Windows consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def fetch_injuries_for_teams(team_names: List[str], pause_sec: float = 1.0,
                             progress_cb=None) -> Dict[str, List[Dict[str, str]]]:
    """Bulk fetch — {team_name: [injury_dicts]}. Skips teams not in our URL map."""
    import time
    out: Dict[str, List[Dict[str, str]]] = {}
    for name in team_names:
        if name not in TEAM_URL_MAP:
            continue
        injuries = fetch_team_injuries(name)
        out[name] = injuries
        if progress_cb:
            progress_cb({"type": "info",
                         "msg": f"  Transfermarkt · {name}: {len(injuries)} травм"})
        time.sleep(pause_sec)
    return out
