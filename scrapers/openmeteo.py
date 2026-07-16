"""
scrapers/openmeteo.py — weather forecast for matches via api.open-meteo.com.

Open-Meteo is a free public API (no key required). We use the hourly forecast
endpoint and pick the hour closest to kick-off.

Stadium coordinates are hard-coded for the ~140 teams in our 6 leagues. Unknown
teams are skipped (weather is optional in the model).
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Tuple

import requests as _requests


METEO_BASE = "https://api.open-meteo.com/v1/forecast"


# (lat, lon) for home stadiums of teams in our top-6 leagues.
# Approximate when stadium relocated mid-season; close enough for weather.
STADIUMS: Dict[str, Tuple[float, float]] = {
    # EPL
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5092, -1.8847),
    "Bournemouth": (50.7352, -1.8383), "Brentford": (51.4904, -0.2887),
    "Brighton": (50.8617, -0.0833), "Burnley": (53.7891, -2.2300),
    "Chelsea": (51.4817, -0.1910), "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4388, -2.9663), "Fulham": (51.4750, -0.2218),
    "Ipswich": (52.0550, 1.1450), "Leeds": (53.7779, -1.5722),
    "Leicester": (52.6204, -1.1421), "Liverpool": (53.4308, -2.9608),
    "Luton": (51.8839, -0.4319), "Manchester City": (53.4831, -2.2004),
    "Manchester United": (53.4631, -2.2913), "Newcastle United": (54.9755, -1.6217),
    "Nottingham Forest": (52.9400, -1.1325), "Sheffield United": (53.3702, -1.4709),
    "Southampton": (50.9059, -1.3911), "Sunderland": (54.9145, -1.3884),
    "Tottenham": (51.6043, -0.0664), "West Ham": (51.5386, -0.0166),
    "Wolverhampton Wanderers": (52.5902, -2.1300),
    # La Liga
    "Real Madrid": (40.4530, -3.6883), "Barcelona": (41.3809, 2.1228),
    "Atletico Madrid": (40.4360, -3.5995), "Athletic Club": (43.2641, -2.9494),
    "Real Sociedad": (43.3013, -1.9737), "Real Betis": (37.3564, -5.9819),
    "Sevilla": (37.3839, -5.9706), "Villarreal": (39.9444, -0.1031),
    "Valencia": (39.4747, -0.3583), "Celta Vigo": (42.2117, -8.7397),
    "Espanyol": (41.3478, 2.0786), "Getafe": (40.3258, -3.7144),
    "Girona": (41.9614, 2.8278), "Mallorca": (39.5894, 2.6303),
    "Osasuna": (42.7969, -1.6367), "Rayo Vallecano": (40.3917, -3.6586),
    "Las Palmas": (28.1003, -15.4569), "Leganes": (40.3408, -3.7619),
    "Alaves": (42.8372, -2.6883), "Valladolid": (41.6444, -4.7611),
    "Oviedo": (43.3611, -5.8675), "Levante": (39.4944, -0.3650),
    # Bundesliga
    "Bayern Munich": (48.2188, 11.6248), "Borussia Dortmund": (51.4925, 7.4519),
    "RB Leipzig": (51.3458, 12.3486), "Bayer Leverkusen": (51.0388, 7.0027),
    "Eintracht Frankfurt": (50.0686, 8.6453), "VfB Stuttgart": (48.7925, 9.2322),
    "Borussia M.Gladbach": (51.1736, 6.3856), "Werder Bremen": (53.0664, 8.8378),
    "VfL Wolfsburg": (52.4319, 10.8039), "1.FC Union Berlin": (52.4569, 13.5683),
    "FC Augsburg": (48.3236, 10.8856), "1.FSV Mainz 05": (49.9839, 8.2244),
    "TSG Hoffenheim": (49.2386, 8.8881), "1.FC Heidenheim": (48.6764, 10.1394),
    "FC St. Pauli": (53.5547, 9.9678), "Holstein Kiel": (54.3494, 10.1217),
    "VfL Bochum": (51.4894, 7.2364), "SC Freiburg": (48.0214, 7.8294),
    "1.FC Koln": (50.9336, 6.8750), "Hamburger SV": (53.5872, 9.8989),
    # Serie A
    "Inter": (45.4781, 9.1239), "AC Milan": (45.4781, 9.1239),
    "Juventus": (45.1097, 7.6411), "Napoli": (40.8278, 14.1928),
    "Atalanta": (45.7090, 9.6810), "Roma": (41.9339, 12.4544),
    "Lazio": (41.9339, 12.4544), "Fiorentina": (43.7806, 11.2828),
    "Bologna": (44.4925, 11.3097), "Torino": (45.0414, 7.6504),
    "Udinese": (46.0814, 13.2003), "Genoa": (44.4163, 8.9525),
    "Empoli": (43.7264, 10.9550), "Cagliari": (39.1997, 9.1369),
    "Hellas Verona": (45.4350, 10.9686), "Sassuolo": (44.6253, 10.8497),
    "Lecce": (40.3653, 18.2089), "Monza": (45.5828, 9.3083),
    "Como": (45.8189, 9.0853), "Parma": (44.7944, 10.3389),
    "Pisa": (43.7211, 10.4006), "Cremonese": (45.1389, 10.0339),
    "Venezia": (45.4297, 12.3636),
    # Ligue 1
    "Paris Saint Germain": (48.8414, 2.2530), "Marseille": (43.2697, 5.3956),
    "Monaco": (43.7275, 7.4150), "Lille": (50.6119, 3.1306),
    "Lyon": (45.7653, 4.9819), "Nice": (43.7050, 7.1925),
    "Rennes": (48.1075, -1.7128), "Lens": (50.4328, 2.8147),
    "Reims": (49.2467, 4.0250), "Nantes": (47.2561, -1.5253),
    "Toulouse": (43.5828, 1.4344), "Strasbourg": (48.5600, 7.7547),
    "Brest": (48.4031, -4.4633), "Auxerre": (47.7872, 3.5894),
    "Angers": (47.4606, -0.5314), "Montpellier": (43.6222, 3.8119),
    "Le Havre": (49.4986, 0.1078), "Saint-Etienne": (45.4608, 4.3897),
    "Metz": (49.1100, 6.1597), "Paris FC": (48.8217, 2.3597),
    "Lorient": (47.7486, -3.3700),
    # RFPL
    "Zenit": (59.9728, 30.2208), "CSKA Moscow": (55.7975, 37.5358),
    "Spartak Moscow": (55.8181, 37.4406), "Dynamo Moscow": (55.7903, 37.5567),
    "Lokomotiv Moscow": (55.7714, 37.6731), "Krasnodar": (45.0181, 38.9728),
    "Rostov": (47.2078, 39.7400), "Akhmat Grozny": (43.3197, 45.6989),
    "Rubin Kazan": (55.8253, 49.0436), "Ural": (56.8519, 60.6097),
    "Sochi": (43.5811, 39.7669), "Khimki": (55.8939, 37.4319),
    "Orenburg": (51.7681, 55.0972), "Krylya Sovetov": (53.2403, 50.1900),
    "Nizhny Novgorod": (56.3267, 44.0072), "Fakel Voronezh": (51.6597, 39.2017),
    "Baltika": (54.7167, 20.5167), "Akron": (53.5081, 49.4194),
    "Dynamo Makhachkala": (42.9847, 47.4983), "Pari NN": (56.3267, 44.0072),
    # Eredivisie
    "Ajax": (52.3143, 4.9419), "PSV": (51.4403, 5.4660),
    "Feyenoord": (51.8939, 4.5206), "AZ": (52.5964, 4.7394),
    "Twente": (52.2367, 6.8350), "Utrecht": (52.0837, 5.1397),
    "Vitesse": (51.9631, 5.8931), "Heerenveen": (52.9597, 5.9372),
    "Groningen": (53.2194, 6.5667),
    # Primeira Liga
    "Benfica": (38.7527, -9.1847), "Porto": (41.1614, -8.6327),
    "Sporting CP": (38.6612, -9.1630), "Braga": (41.5440, -8.4439),
    # Süper Lig
    "Galatasaray": (41.0422, 28.9869), "Fenerbahce": (40.9833, 29.0375),
    "Besiktas": (41.0430, 28.9950), "Trabzonspor": (40.9940, 39.7148),
    # Belgian Pro League
    "Club Brugge": (51.1928, 3.2256), "Anderlecht": (50.8344, 4.3336),
    "Genk": (50.9600, 5.5100), "Royal Antwerp": (51.2194, 4.4039),
    # Greek Super League
    "Olympiacos": (37.9486, 23.6650), "Panathinaikos": (37.9647, 23.7870),
    "AEK Athens": (37.9647, 23.7870), "PAOK": (40.6339, 22.9767),
    # Turkish Super Lig additional
    "Istanbul Basaksehir": (41.0700, 28.7900), "Antalyaspor": (36.8800, 30.7100),
    "Adana Demirspor": (37.0000, 35.3213),
    # Championships
    "Leicester": (52.6204, -1.1421), "Leeds": (53.7779, -1.5722),
    "Burnley": (53.7891, -2.2300), "Sunderland": (54.9145, -1.3884),
    "West Brom": (52.5090, -1.9639), "Middlesbrough": (54.5782, -1.2184),
}


def _fetch_forecast(data: dict) -> Optional[dict]:
    lat, lon = data["lat"], data["lon"]
    when = data["when"]   # ISO date "2025-12-15"
    url = (f"{METEO_BASE}?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m,precipitation,windspeed_10m"
           f"&start_date={when}&end_date={when}&timezone=auto")
    try:
        resp = _requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def fetch_match_weather(team_home: str, match_datetime: str
                        ) -> Optional[Dict[str, float]]:
    """Return weather at kick-off for the home team's stadium, or None."""
    coords = STADIUMS.get(team_home)
    if not coords:
        return None
    lat, lon = coords
    # parse "2025-12-15 18:00:00"
    try:
        kt = dt.datetime.fromisoformat(match_datetime.replace(" ", "T"))
    except Exception:
        return None
    # Open-Meteo forecast horizon is ~16 days; beyond that returns nothing useful
    if (kt.date() - dt.date.today()).days > 14:
        return None
    if kt.date() < dt.date.today():
        return None  # historical weather is a different endpoint
    data = _fetch_forecast({"lat": lat, "lon": lon, "when": kt.date().isoformat()})
    if not data or "hourly" not in data:
        return None
    times = data["hourly"]["time"]            # ["2025-12-15T00:00", ...]
    temps = data["hourly"]["temperature_2m"]
    rains = data["hourly"]["precipitation"]
    winds = data["hourly"]["windspeed_10m"]
    target = kt.strftime("%Y-%m-%dT%H:00")
    if target in times:
        i = times.index(target)
    else:
        return None
    return {
        "temp_c":   float(temps[i]) if temps[i] is not None else None,
        "rain_mm":  float(rains[i]) if rains[i] is not None else None,
        "wind_ms":  float(winds[i]) if winds[i] is not None else None,
    }
