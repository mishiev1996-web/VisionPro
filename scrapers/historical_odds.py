"""
scrapers/historical_odds.py — bookmaker odds from football-data.co.uk.

Free, no auth, just CSV downloads:
    https://www.football-data.co.uk/mmz4281/<YYstart-YYend>/<league_code>.csv

Each row has closing odds from multiple bookmakers; we keep Pinnacle closing
(most predictive in literature) and a 1X2 market average as a backup.

League coverage matches Understat almost 1:1 (top-5 European leagues), and we
add a few extras (Eredivisie, Liga Portugal, Championship, etc.) that the model
can learn from if we extend Understat coverage later.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Understat slug → football-data.co.uk league code
LEAGUE_CODES: Dict[str, str] = {
    "EPL":        "E0",
    "La_liga":    "SP1",
    "Bundesliga": "D1",
    "Serie_A":    "I1",
    "Ligue_1":    "F1",
    "Championship": "E1",
    "La_liga_2":  "SP2",
    "Serie_B":    "I2",
    "Ligue_2":    "F2",
    "2_Bundesliga": "D2",
    "Eredivisie": "N1",
    "Primeira_Liga": "P1",
    "Belgian_First": "B1",
    "Super_Lig":  "T1",
    "Greek_Super": "G1",
}


# Football-data uses very short team names. Map them back to Understat names.
# Lowercased keys for case-insensitive match.
NAME_MAP_RAW: Dict[str, str] = {
    # EPL
    "man united":      "Manchester United",
    "man city":        "Manchester City",
    "newcastle":       "Newcastle United",
    "tottenham":       "Tottenham",
    "nott'm forest":   "Nottingham Forest",
    "wolves":          "Wolverhampton Wanderers",
    "brighton":        "Brighton",
    "west ham":        "West Ham",
    "leicester":       "Leicester",
    "ipswich":         "Ipswich",
    "leeds":           "Leeds",
    "burnley":         "Burnley",
    "luton":           "Luton",
    "sheffield united":"Sheffield United",
    "sunderland":      "Sunderland",
    # La Liga
    "ath bilbao":      "Athletic Club",
    "sociedad":        "Real Sociedad",
    "atletico":        "Atletico Madrid",
    "betis":           "Real Betis",
    "vallecano":       "Rayo Vallecano",
    "celta":           "Celta Vigo",
    "las palmas":      "Las Palmas",
    "leganes":         "Leganes",
    "real oviedo":     "Oviedo",
    "alaves":          "Alaves",
    "valladolid":      "Valladolid",
    # Bundesliga
    "bayern munich":   "Bayern Munich",
    "dortmund":        "Borussia Dortmund",
    "leverkusen":      "Bayer Leverkusen",
    "leipzig":         "RB Leipzig",
    "ein frankfurt":   "Eintracht Frankfurt",
    "stuttgart":       "VfB Stuttgart",
    "m'gladbach":      "Borussia M.Gladbach",
    "werder bremen":   "Werder Bremen",
    "wolfsburg":       "VfL Wolfsburg",
    "union berlin":    "1.FC Union Berlin",
    "augsburg":        "FC Augsburg",
    "mainz":           "1.FSV Mainz 05",
    "hoffenheim":      "TSG Hoffenheim",
    "heidenheim":      "1.FC Heidenheim",
    "st pauli":        "FC St. Pauli",
    "holstein kiel":   "Holstein Kiel",
    "bochum":          "VfL Bochum",
    "freiburg":        "SC Freiburg",
    "fc koln":         "1.FC Koln",
    "hamburg":         "Hamburger SV",
    # Serie A
    "ac milan":        "AC Milan",
    "inter":           "Inter",
    "verona":          "Hellas Verona",
    # Ligue 1
    "paris sg":        "Paris Saint Germain",
    "st etienne":      "Saint-Etienne",
    "paris fc":        "Paris FC",
    # Eredivisie
    "ajax":            "Ajax",
    "psv eindhoven":   "PSV",
    "feyenoord":       "Feyenoord",
    "az alkmaar":      "AZ",
    "fc twente":       "Twente",
    "utrecht":         "Utrecht",
    "vitesse":         "Vitesse",
    "heerenveen":      "Heerenveen",
    "groningen":       "Groningen",
    # Primeira Liga
    "benfica":         "Benfica",
    "porto":           "Porto",
    "sporting cp":     "Sporting CP",
    "braga":           "Braga",
    "guimaraes":       "Vitoria Guimaraes",
    # Championship
    "leicester":       "Leicester",
    "leeds":           "Leeds",
    "burnley":         "Burnley",
    "sunderland":      "Sunderland",
    "west brom":       "West Brom",
    "middlesbrough":   "Middlesbrough",
    "coventry":        "Coventry",
    "norwich":         "Norwich",
    "sheffield wed":   "Sheffield Wednesday",
    "swansea":         "Swansea",
    "watford":         "Watford",
    "bristol city":    "Bristol City",
    "blackburn":       "Blackburn",
    "millwall":        "Millwall",
    "preston":         "Preston",
    "hull":            "Hull",
    "ston":            "Stoke",
    "cardiff":         "Cardiff",
    "plymouth":        "Plymouth",
    "oxford":          "Oxford United",
}


def _norm(name: str) -> str:
    return name.strip().lower()


def normalize_team(fd_name: str) -> str:
    """Map football-data.co.uk team name to its Understat counterpart (best effort)."""
    n = _norm(fd_name)
    if n in NAME_MAP_RAW:
        return NAME_MAP_RAW[n]
    # If no mapping, return original — fuzzy matcher in db helper will try substring.
    return fd_name.strip()


def _season_code(year_start: int) -> str:
    """2024 → '2425'."""
    return f"{year_start % 100:02d}{(year_start + 1) % 100:02d}"


def _parse_date(s: str) -> Optional[str]:
    """Parse DD/MM/YYYY or DD/MM/YY → 'YYYY-MM-DD'."""
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _to_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_league_season_odds(league_slug: str, season_year: int
                             ) -> List[Dict[str, object]]:
    """Download one CSV and parse odds rows."""
    code = LEAGUE_CODES.get(league_slug)
    if not code:
        return []
    url = f"{BASE_URL}/{_season_code(season_year)}/{code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":
            "Mozilla/5.0 (Football-AI; learning) Python-urllib"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                return []
            raw = resp.read().decode("utf-8-sig", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return []

    reader = csv.DictReader(io.StringIO(raw))
    out: List[Dict[str, object]] = []
    for r in reader:
        date_iso = _parse_date(r.get("Date", ""))
        if not date_iso:
            continue
        home = normalize_team(r.get("HomeTeam") or "")
        away = normalize_team(r.get("AwayTeam") or "")
        if not home or not away:
            continue

        # Prefer Pinnacle closing → Bet365 → Average. Some leagues lack PSC.
        psh, psd, psa = (_to_float(r.get("PSCH")), _to_float(r.get("PSCD")),
                         _to_float(r.get("PSCA")))
        bsh, bsd, bsa = (_to_float(r.get("B365H")), _to_float(r.get("B365D")),
                         _to_float(r.get("B365A")))
        avh, avd, ava = (_to_float(r.get("AvgH")), _to_float(r.get("AvgD")),
                         _to_float(r.get("AvgA")))

        if psh and psd and psa:
            home_o, draw_o, away_o, source = psh, psd, psa, "pinnacle_close"
        elif bsh and bsd and bsa:
            home_o, draw_o, away_o, source = bsh, bsd, bsa, "bet365"
        elif avh and avd and ava:
            home_o, draw_o, away_o, source = avh, avd, ava, "market_avg"
        else:
            continue   # no usable odds row

        out.append({
            "date": date_iso, "home": home, "away": away,
            "home_odds": home_o, "draw_odds": draw_o, "away_odds": away_o,
            "source": source,
        })
    return out


def odds_to_implied(home_o: float, draw_o: float, away_o: float
                    ) -> Tuple[float, float, float]:
    """Convert decimal odds → implied probabilities, normalized to sum 1.0."""
    p_h = 1.0 / home_o
    p_d = 1.0 / draw_o
    p_a = 1.0 / away_o
    total = p_h + p_d + p_a
    if total <= 0:
        return 0.0, 0.0, 0.0
    return p_h / total, p_d / total, p_a / total
