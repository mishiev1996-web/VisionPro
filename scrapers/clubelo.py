"""
scrapers/clubelo.py — fetch Elo ratings from api.clubelo.com.

ClubElo provides a free public API:
    http://api.clubelo.com/<Club>          → CSV history for one club
    http://api.clubelo.com/<YYYY-MM-DD>    → CSV ranking for that date

No auth, no Cloudflare. Just plain HTTP.
"""
from __future__ import annotations

import csv
import io
import time
from typing import Dict, List, Optional, Tuple

from botasaurus.request import request, Request


CLUBELO_BASE = "http://api.clubelo.com"


# Map our Understat team names → ClubElo URL slugs (no spaces, sometimes truncated)
# ClubElo uses these specific forms — most match, a few are quirky.
NAME_MAP: Dict[str, str] = {
    # EPL
    "Arsenal": "Arsenal", "Aston Villa": "AstonVilla", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Crystal Palace": "CrystalPalace", "Everton": "Everton",
    "Fulham": "Fulham", "Ipswich": "Ipswich", "Leeds": "Leeds",
    "Leicester": "Leicester", "Liverpool": "Liverpool", "Luton": "Luton",
    "Manchester City": "ManCity", "Manchester United": "ManUnited",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Forest",
    "Sheffield United": "SheffieldUnited", "Southampton": "Southampton",
    "Sunderland": "Sunderland", "Tottenham": "Tottenham",
    "West Ham": "WestHam", "Wolverhampton Wanderers": "Wolves",
    # La Liga
    "Real Madrid": "Real", "Barcelona": "Barcelona", "Atletico Madrid": "Atletico",
    "Athletic Club": "Athletic", "Real Sociedad": "Sociedad", "Real Betis": "Betis",
    "Sevilla": "Sevilla", "Villarreal": "Villarreal", "Valencia": "Valencia",
    "Celta Vigo": "Celta", "Espanyol": "Espanyol", "Getafe": "Getafe",
    "Girona": "Girona", "Mallorca": "Mallorca", "Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo", "Las Palmas": "LasPalmas", "Leganes": "Leganes",
    "Alaves": "Alaves", "Valladolid": "Valladolid", "Cadiz": "Cadiz",
    "Almeria": "Almeria", "Granada": "Granada", "Elche": "Elche", "Oviedo": "Oviedo",
    "Levante": "Levante",
    # Bundesliga
    "Bayern Munich": "Bayern", "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "Leipzig", "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Frankfurt", "VfB Stuttgart": "Stuttgart",
    "Borussia M.Gladbach": "Gladbach", "Werder Bremen": "Bremen",
    "VfL Wolfsburg": "Wolfsburg", "1.FC Union Berlin": "UnionBerlin",
    "FC Augsburg": "Augsburg", "1.FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim", "1.FC Heidenheim": "Heidenheim",
    "FC St. Pauli": "StPauli", "Holstein Kiel": "Kiel", "VfL Bochum": "Bochum",
    "SC Freiburg": "Freiburg", "1.FC Koln": "Koeln", "Hamburger SV": "HSV",
    # Serie A
    "Inter": "Inter", "Juventus": "Juventus", "AC Milan": "Milan", "Napoli": "Napoli",
    "Atalanta": "Atalanta", "Roma": "Roma", "Lazio": "Lazio", "Fiorentina": "Fiorentina",
    "Bologna": "Bologna", "Torino": "Torino", "Udinese": "Udinese", "Genoa": "Genoa",
    "Empoli": "Empoli", "Cagliari": "Cagliari", "Hellas Verona": "Verona",
    "Sassuolo": "Sassuolo", "Lecce": "Lecce", "Monza": "Monza", "Como": "Como",
    "Parma": "Parma", "Pisa": "Pisa", "Cremonese": "Cremonese", "Venezia": "Venezia",
    # Ligue 1
    "Paris Saint Germain": "Paris", "Marseille": "Marseille", "Monaco": "Monaco",
    "Lille": "Lille", "Lyon": "Lyon", "Nice": "Nice", "Rennes": "Rennes",
    "Lens": "Lens", "Reims": "Reims", "Nantes": "Nantes", "Toulouse": "Toulouse",
    "Strasbourg": "Strasbourg", "Brest": "Brest", "Auxerre": "Auxerre",
    "Angers": "Angers", "Montpellier": "Montpellier", "Le Havre": "LeHavre",
    "Saint-Etienne": "StEtienne", "Metz": "Metz", "Paris FC": "ParisFC",
    "Lorient": "Lorient",
    # Eredivisie
    "Ajax": "Ajax", "PSV": "PSV", "Feyenoord": "Feyenoord", "AZ": "AZ",
    "Twente": "Twente", "Utrecht": "Utrecht", "Vitesse": "Vitesse",
    "Heerenveen": "Heerenveen", "Groningen": "Groningen",
    # Primeira Liga
    "Benfica": "Benfica", "Porto": "Porto", "Sporting CP": "Sporting",
    "Braga": "Braga", "Vitoria Guimaraes": "Guimaraes",
    "Sporting Braga": "Braga", "Gil Vicente": "GilVicente",
    # Süper Lig
    "Galatasaray": "Galatasaray", "Fenerbahce": "Fenerbahce",
    "Besiktas": "Besiktas", "Trabzonspor": "Trabzonspor",
    "Istanbul Basaksehir": "Basaksehir", "Antalyaspor": "Antalyaspor",
    # Belgian Pro League
    "Club Brugge": "ClubBrugge", "Anderlecht": "Anderlecht",
    "Genk": "Genk", "Union SG": "UnionSG", "Royal Antwerp": "Antwerp",
    # Russian Premier League
    "Zenit": "Zenit", "CSKA Moscow": "CSKA", "Spartak Moscow": "Spartak",
    "Lokomotiv Moscow": "Lokomotiv", "Dynamo Moscow": "DynamoMoscow",
    "Krasnodar": "Krasnodar", "Rostov": "Rostov",
    "Zenit St. Petersburg": "Zenit", "FC Rostov": "Rostov",
    "Dinamo Moscow": "DynamoMoscow", "FC Rotor Volgograd": "Rotor",
    "FK Akhmat": "Akhmat", "Krylya Sovetov Samara": "KrylyaSovetov",
    "Nizhny Novgorod": "NizhnyNovgorod", "PFC Sochi": "Sochi",
    "Fakel": "FakelVoronezh", "Arsenal Tula": "ArsenalTula",
    "FC Ufa": "Ufa", "Anzhi Makhachkala": "Anzhi",
    "FC Yenisey Krasnoyarsk": "Yenisey", "FC Tambov": "Tambov",
    "Rubin Kazan": "Rubin", "Torpedo Moscow": "Torpedo",
    # Eredivisie extra
    "AZ Alkmaar": "AZ", "SC Heerenveen": "Heerenveen",
    "FC Groningen": "Groningen", "Vitesse Arnhem": "Vitesse",
    "Heracles Almelo": "Heracles", "Go Ahead Eagles": "GoAheadEagles",
    "PEC Zwolle": "Zwolle", "Willem II Tilburg": "WillemII",
    "Sparta Rotterdam": "SpartaRotterdam", "NEC Nijmegen": "NEC",
    "Almere City": "AlmereCity", "RKC Waalwijk": "RKC",
    # Primeira Liga extra
    "Vitoria Guimaraes": "Guimaraes", "Sporting Braga": "Braga",
    "Gil Vicente": "GilVicente", "Casa Pia": "CasaPia",
    "Estoril Praia": "Estoril", "Rio Ave": "RioAve",
    "Boavista": "Boavista", "Famalicao": "Famalicao",
    "Arouca": "Arouca", "Chaves": "Chaves",
    "Moreirense": "Moreirense", "Vizela": "Vizela",
    "Portimonense": "Portimonense", "Santa Clara": "SantaClara",
    # Super Lig extra
    "Istanbul Basaksehir": "Basaksehir", "Antalyaspor": "Antalyaspor",
    "Adana Demirspor": "AdanaDemirspor", "Kayserispor": "Kayserispor",
    "Alanyaspor": "Alanyaspor", "Hatayspor": "Hatayspor",
    "Konyaspor": "Konyaspor", "Gazişehir Gaziantep": "Gaziantep",
    "Sivasspor": "Sivasspor", "Kasımpaşa": "Kasimpasa",
    "Giresunspor": "Giresunspor", "Ümraniyespor": "Umraniyespor",
    "MKE Ankaragücü": "Ankaragucu", "İstanbulspor": "Istanbulspor",
    # Belgian Pro League extra
    "Union SG": "UnionSG", "Royal Antwerp": "Antwerp",
    "KRC Genk": "Genk", "KAA Gent": "Gent",
    "Standard Liege": "Standard", "RSC Anderlecht": "Anderlecht",
    "Club Brugge": "ClubBrugge", "KV Mechelen": "Mechelen",
    "AS Eupen": "Eupen", "Cercle Brugge": "CercleBrugge",
    "Oud-Heverlee Leuven": "OHL", "Sint-Truiden": "SintTruiden",
    "SV Zulte Waregem": "ZulteWaregem", "RFC Seraing": "Seraing",
    "K Beerschot VA": "Beerschot", "RWD Molenbeek": "Molenbeek",
    # English Championship extra
    "West Bromwich Albion": "WestBrom", "Middlesbrough": "Middlesbrough",
    "Coventry City": "Coventry", "Norwich City": "Norwich",
    "Sheffield Wednesday": "SheffieldWed", "Swansea City": "Swansea",
    "Watford": "Watford", "Bristol City": "BristolCity",
    "Blackburn Rovers": "Blackburn", "Millwall": "Millwall",
    "Preston North End": "Preston", "Hull City": "Hull",
    "Stoke City": "Stoke", "Cardiff City": "Cardiff",
    "Plymouth Argyle": "Plymouth", "Oxford United": "Oxford",
    "Luton Town": "Luton", "Derby County": "Derby",
    "Brighton": "Brighton", "Leeds": "Leeds",
    # Serie A extra
    "SPAL 2013": "Spal", "Spezia": "Spezia",
    "Salernitana": "Salernitana", "Pisa": "Pisa",
    "Cremonese": "Cremonese", "Empoli": "Empoli",
    # Ligue 1 extra
    "Paris FC": "ParisFC", "Clermont Foot": "Clermont",
    "Troyes": "Troyes", "Nimes": "Nimes",
    "Guingamp": "Guingamp", "Metz": "Metz",
    "Lorient": "Lorient", "Saint-Etienne": "StEtienne",
    # Bundesliga extra
    "Hertha Berlin": "Hertha", "Schalke 04": "Schalke",
    "Hannover 96": "Hannover", "FC Cologne": "Koeln",
    "RasenBallsport Leipzig": "Leipzig", "Paderborn": "Paderborn",
    "Fortuna Duesseldorf": "Duesseldorf", "Nuernberg": "Nuernberg",
    "Arminia Bielefeld": "Bielefeld", "Greuther Fuerth": "Fuerth",
    # La Liga extra
    "SD Huesca": "Huesca", "Real Valladolid": "Valladolid",
    "Real Oviedo": "Oviedo",
}


# parallel=10 → 10× faster. Botasaurus pools sessions correctly so this is safe.
@request(cache=False, output=None, create_error_logs=False, max_retry=2,
         raise_exception=False, parallel=10)
def _fetch_club_batch(req: Request, data: dict) -> Optional[str]:
    name = data["name"]
    try:
        resp = req.get(f"{CLUBELO_BASE}/{name}", timeout=15)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def _fetch_club_batch_direct(names: List[str]) -> List[Optional[str]]:
    """Fetch ClubElo CSV for a batch of club names using direct HTTP."""
    import urllib.request
    results = []
    for name in names:
        try:
            req = urllib.request.Request(
                f"{CLUBELO_BASE}/{name}",
                headers={"User-Agent": "Mozilla/5.0 (Football-AI; learning)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    results.append(resp.read().decode("utf-8", errors="replace"))
                else:
                    results.append(None)
        except Exception:
            results.append(None)
    return results


def fetch_elo_history(team_name: str) -> List[Tuple[str, float]]:
    """Return list of (date, elo) pairs for a team. Empty if name not found."""
    slug = NAME_MAP.get(team_name, team_name.replace(" ", ""))
    csv_text = _fetch_club(slug)
    if not csv_text:
        return []
    rows = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        # CSV columns: Club, Country, Level, Elo, From, To
        try:
            elo = float(r.get("Elo", "0") or 0)
            date = r.get("From", "")
            if elo > 0 and date:
                rows.append((date, elo))
        except (TypeError, ValueError):
            continue
    return rows


def fetch_current_elo(team_name: str) -> Optional[float]:
    """Latest Elo rating for a team, or None."""
    hist = fetch_elo_history(team_name)
    if not hist:
        return None
    # Sorted by From date ascending → take last
    hist.sort(key=lambda x: x[0])
    return hist[-1][1]


def fetch_elo_for_teams(team_names: List[str], pause_sec: float = 0.5,
                        progress_cb=None) -> Dict[str, float]:
    """Bulk fetch — returns {team_name: latest_elo}. Skips teams not on ClubElo."""
    out: Dict[str, float] = {}
    for name in team_names:
        elo = fetch_current_elo(name)
        if elo is not None:
            out[name] = elo
            if progress_cb:
                progress_cb({"type": "info", "msg": f"  ClubElo · {name}: {elo:.0f}"})
        else:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"  ClubElo · {name}: не найдено"})
        time.sleep(pause_sec)
    return out
