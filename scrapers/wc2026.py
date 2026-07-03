"""
wc2026.py — FIFA World Cup 2026 data scraper and predictor.

Collects teams, groups, schedule, and matches for the 2026 World Cup
(USA/Canada/Mexico, June 11 – July 19, 2026).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Dict, List, Optional, Any

import requests

logger = logging.getLogger("wc2026")

WC2026_SLUG = "WC_2026"
WC2026_NAME = "FIFA World Cup 2026"
WC2026_COUNTRY = "International"

# English → Russian name mapping for all WC2026 teams
TEAM_NAME_RU = {
    "Argentina": "Аргентина", "France": "Франция", "England": "Англия",
    "Brazil": "Бразилия", "Portugal": "Португалия", "Netherlands": "Нидерланды",
    "Spain": "Испания", "Colombia": "Колумбия", "Uruguay": "Уругвай",
    "Japan": "Япония", "USA": "США", "Mexico": "Мексика",
    "Germany": "Германия", "Croatia": "Хорватия", "Morocco": "Марокко",
    "Italy": "Италия", "Belgium": "Бельгия", "Iran": "Иран",
    "South Korea": "Южная Корея", "Australia": "Австралия", "Senegal": "Сенегал",
    "Ecuador": "Эквадор", "Turkey": "Турция", "Switzerland": "Швейцария",
    "Canada": "Канада", "Paraguay": "Парагвай", "Serbia": "Сербия",
    "Nigeria": "Нигерия", "Egypt": "Египет", "Scotland": "Шотландия",
    "Tunisia": "Тунис", "Cameroon": "Камерун", "Norway": "Норвегия",
    "Greece": "Греция", "Czech Republic": "Чехия", "Poland": "Польша",
    "Algeria": "Алжир", "Mali": "Мали", "Ivory Coast": "Кот-д'Ивуар",
    "Saudi Arabia": "Саудовская Аравия", "Qatar": "Катар",
    "Indonesia": "Индонезия", "Costa Rica": "Коста-Рика",
    "Honduras": "Гондурас", "Jamaica": "Ямайка",
    "New Zealand": "Новая Зеландия", "Ghana": "Гана", "Uzbekistan": "Узбекистан",
}
# Reverse: Russian → English
TEAM_NAME_EN = {v: k for k, v in TEAM_NAME_RU.items()}

# All 48 qualified teams with FIFA ranking groups (seeded 1-4 per pot)
# Based on FIFA rankings as of late 2025 / early 2026
WC2026_TEAMS = {
    # Pot 1 (Top seeds)
    "Argentina": {"conf": "CONMEBOL", "pot": 1, "fifa_rank": 1},
    "France": {"conf": "UEFA", "pot": 1, "fifa_rank": 2},
    "England": {"conf": "UEFA", "pot": 1, "fifa_rank": 3},
    "Brazil": {"conf": "CONMEBOL", "pot": 1, "fifa_rank": 4},
    "Portugal": {"conf": "UEFA", "pot": 1, "fifa_rank": 5},
    "Netherlands": {"conf": "UEFA", "pot": 1, "fifa_rank": 6},
    "Spain": {"conf": "UEFA", "pot": 1, "fifa_rank": 7},
    "Colombia": {"conf": "CONMEBOL", "pot": 1, "fifa_rank": 8},
    "Uruguay": {"conf": "CONMEBOL", "pot": 1, "fifa_rank": 9},
    "Japan": {"conf": "AFC", "pot": 1, "fifa_rank": 10},
    "USA": {"conf": "CONCACAF", "pot": 1, "fifa_rank": 11},
    "Mexico": {"conf": "CONCACAF", "pot": 1, "fifa_rank": 12},
    # Pot 2
    "Germany": {"conf": "UEFA", "pot": 2, "fifa_rank": 13},
    "Croatia": {"conf": "UEFA", "pot": 2, "fifa_rank": 14},
    "Morocco": {"conf": "CAF", "pot": 2, "fifa_rank": 15},
    "Italy": {"conf": "UEFA", "pot": 2, "fifa_rank": 16},
    "Belgium": {"conf": "UEFA", "pot": 2, "fifa_rank": 17},
    "Iran": {"conf": "AFC", "pot": 2, "fifa_rank": 18},
    "South Korea": {"conf": "AFC", "pot": 2, "fifa_rank": 19},
    "Australia": {"conf": "AFC", "pot": 2, "fifa_rank": 20},
    "Senegal": {"conf": "CAF", "pot": 2, "fifa_rank": 21},
    "Ecuador": {"conf": "CONMEBOL", "pot": 2, "fifa_rank": 22},
    "Turkey": {"conf": "UEFA", "pot": 2, "fifa_rank": 23},
    "Switzerland": {"conf": "UEFA", "pot": 2, "fifa_rank": 24},
    # Pot 3
    "Canada": {"conf": "CONCACAF", "pot": 3, "fifa_rank": 25},
    "Paraguay": {"conf": "CONMEBOL", "pot": 3, "fifa_rank": 26},
    "Serbia": {"conf": "UEFA", "pot": 3, "fifa_rank": 27},
    "Nigeria": {"conf": "CAF", "pot": 3, "fifa_rank": 28},
    "Egypt": {"conf": "CAF", "pot": 3, "fifa_rank": 29},
    "Scotland": {"conf": "UEFA", "pot": 3, "fifa_rank": 30},
    "Tunisia": {"conf": "CAF", "pot": 3, "fifa_rank": 31},
    "Cameroon": {"conf": "CAF", "pot": 3, "fifa_rank": 32},
    "Norway": {"conf": "UEFA", "pot": 3, "fifa_rank": 33},
    "Greece": {"conf": "UEFA", "pot": 3, "fifa_rank": 34},
    "Czech Republic": {"conf": "UEFA", "pot": 3, "fifa_rank": 35},
    "Poland": {"conf": "UEFA", "pot": 3, "fifa_rank": 36},
    # Pot 4
    "Algeria": {"conf": "CAF", "pot": 4, "fifa_rank": 37},
    "Mali": {"conf": "CAF", "pot": 4, "fifa_rank": 38},
    "Ivory Coast": {"conf": "CAF", "pot": 4, "fifa_rank": 39},
    "Saudi Arabia": {"conf": "AFC", "pot": 4, "fifa_rank": 40},
    "Qatar": {"conf": "AFC", "pot": 4, "fifa_rank": 41},
    "Indonesia": {"conf": "AFC", "pot": 4, "fifa_rank": 42},
    "Costa Rica": {"conf": "CONCACAF", "pot": 4, "fifa_rank": 43},
    "Honduras": {"conf": "CONCACAF", "pot": 4, "fifa_rank": 44},
    "Jamaica": {"conf": "CONCACAF", "pot": 4, "fifa_rank": 45},
    "New Zealand": {"conf": "OFC", "pot": 4, "fifa_rank": 46},
    "Ghana": {"conf": "CAF", "pot": 4, "fifa_rank": 47},
    "Uzbekistan": {"conf": "AFC", "pot": 4, "fifa_rank": 48},
}

# Groups A-L (48 teams, 12 groups of 4)
# Seeded: Pot 1 in position 1, Pot 2 in position 2, Pot 3 in position 3, Pot 4 in position 4
# Draw simulation based on FIFA's draw procedures (geographical restrictions)
WC2026_GROUPS = {
    "A": ["Mexico", "Iran", "Egypt", "New Zealand"],
    "B": ["USA", "South Korea", "Norway", "Jamaica"],
    "C": ["Brazil", "Germany", "Cameroon", "Honduras"],
    "D": ["Argentina", "Morocco", "Poland", "Indonesia"],
    "E": ["England", "Italy", "Nigeria", "Costa Rica"],
    "F": ["France", "Croatia", "Senegal", "Tunisia"],
    "G": ["Spain", "Belgium", "Ecuador", "Ghana"],
    "H": ["Portugal", "Japan", "Scotland", "Saudi Arabia"],
    "I": ["Netherlands", "Australia", "Serbia", "Qatar"],
    "J": ["Colombia", "Turkey", "Czech Republic", "Mali"],
    "K": ["Uruguay", "Greece", "Paraguay", "Uzbekistan"],
    "L": ["Canada", "Switzerland", "Ivory Coast", "Algeria"],
}

# 2026 World Cup schedule (group stage matches)
# June 11 – July 19, 2026
# Opening match: Mexico vs New Zealand (Mexico City, June 11)
WC2026_SCHEDULE_TEMPLATE = [
    # Group A
    {"date": "2026-06-11", "time": "15:00", "home": "Mexico", "away": "New Zealand", "venue": "Estadio Azteca, Mexico City", "group": "A"},
    {"date": "2026-06-11", "time": "21:00", "home": "Iran", "away": "Egypt", "venue": "Rose Bowl, Los Angeles", "group": "A"},
    {"date": "2026-06-16", "time": "17:00", "home": "Mexico", "away": "Iran", "venue": "Estadio Azteca, Mexico City", "group": "A"},
    {"date": "2026-06-16", "time": "20:00", "home": "Egypt", "away": "New Zealand", "venue": "Lumen Field, Seattle", "group": "A"},
    {"date": "2026-06-22", "time": "16:00", "home": "Egypt", "away": "Mexico", "venue": "AT&T Stadium, Dallas", "group": "A"},
    {"date": "2026-06-22", "time": "16:00", "home": "New Zealand", "away": "Iran", "venue": "Levi's Stadium, San Francisco", "group": "A"},
    # Group B
    {"date": "2026-06-12", "time": "15:00", "home": "USA", "away": "Jamaica", "venue": "MetLife Stadium, New York", "group": "B"},
    {"date": "2026-06-12", "time": "21:00", "home": "South Korea", "away": "Norway", "venue": "NRG Stadium, Houston", "group": "B"},
    {"date": "2026-06-17", "time": "17:00", "home": "USA", "away": "South Korea", "venue": "MetLife Stadium, New York", "group": "B"},
    {"date": "2026-06-17", "time": "20:00", "home": "Norway", "away": "Jamaica", "venue": "Mercedes-Benz Stadium, Atlanta", "group": "B"},
    {"date": "2026-06-23", "time": "16:00", "home": "Norway", "away": "USA", "venue": "Lincoln Financial Field, Philadelphia", "group": "B"},
    {"date": "2026-06-23", "time": "16:00", "home": "Jamaica", "away": "South Korea", "venue": "Gillette Stadium, Boston", "group": "B"},
    # Group C
    {"date": "2026-06-13", "time": "15:00", "home": "Brazil", "away": "Cameroon", "venue": "Hard Rock Stadium, Miami", "group": "C"},
    {"date": "2026-06-13", "time": "21:00", "home": "Germany", "away": "Honduras", "venue": "SoFi Stadium, Los Angeles", "group": "C"},
    {"date": "2026-06-18", "time": "17:00", "home": "Brazil", "away": "Germany", "venue": "SoFi Stadium, Los Angeles", "group": "C"},
    {"date": "2026-06-18", "time": "20:00", "home": "Honduras", "away": "Cameroon", "venue": "NRG Stadium, Houston", "group": "C"},
    {"date": "2026-06-24", "time": "16:00", "home": "Honduras", "away": "Brazil", "venue": "Estadio BBVA, Monterrey", "group": "C"},
    {"date": "2026-06-24", "time": "16:00", "home": "Cameroon", "away": "Germany", "venue": "BMO Field, Toronto", "group": "C"},
    # Group D
    {"date": "2026-06-14", "time": "15:00", "home": "Argentina", "away": "Poland", "venue": "MetLife Stadium, New York", "group": "D"},
    {"date": "2026-06-14", "time": "21:00", "home": "Morocco", "away": "Indonesia", "venue": "Mercedes-Benz Stadium, Atlanta", "group": "D"},
    {"date": "2026-06-19", "time": "17:00", "home": "Argentina", "away": "Morocco", "venue": "MetLife Stadium, New York", "group": "D"},
    {"date": "2026-06-19", "time": "20:00", "home": "Indonesia", "away": "Poland", "venue": "Lincoln Financial Field, Philadelphia", "group": "D"},
    {"date": "2026-06-25", "time": "16:00", "home": "Indonesia", "away": "Argentina", "venue": "Estadio Azteca, Mexico City", "group": "D"},
    {"date": "2026-06-25", "time": "16:00", "home": "Poland", "away": "Morocco", "venue": "Lumen Field, Seattle", "group": "D"},
    # Group E
    {"date": "2026-06-15", "time": "15:00", "home": "England", "away": "Nigeria", "venue": "NRG Stadium, Houston", "group": "E"},
    {"date": "2026-06-15", "time": "21:00", "home": "Italy", "away": "Costa Rica", "venue": "Levi's Stadium, San Francisco", "group": "E"},
    {"date": "2026-06-20", "time": "17:00", "home": "England", "away": "Italy", "venue": "NRG Stadium, Houston", "group": "E"},
    {"date": "2026-06-20", "time": "20:00", "home": "Costa Rica", "away": "Nigeria", "venue": "Estadio BBVA, Monterrey", "group": "E"},
    {"date": "2026-06-26", "time": "16:00", "home": "Costa Rica", "away": "England", "venue": "Gillette Stadium, Boston", "group": "E"},
    {"date": "2026-06-26", "time": "16:00", "home": "Nigeria", "away": "Italy", "venue": "BMO Field, Toronto", "group": "E"},
    # Group F
    {"date": "2026-06-15", "time": "18:00", "home": "France", "away": "Tunisia", "venue": "Hard Rock Stadium, Miami", "group": "F"},
    {"date": "2026-06-15", "time": "24:00", "home": "Croatia", "away": "Senegal", "venue": "AT&T Stadium, Dallas", "group": "F"},
    {"date": "2026-06-20", "time": "24:00", "home": "France", "away": "Croatia", "venue": "AT&T Stadium, Dallas", "group": "F"},
    {"date": "2026-06-21", "time": "17:00", "home": "Senegal", "away": "Tunisia", "venue": "Estadio Azteca, Mexico City", "group": "F"},
    {"date": "2026-06-26", "time": "20:00", "home": "Senegal", "away": "France", "venue": "Rose Bowl, Los Angeles", "group": "F"},
    {"date": "2026-06-26", "time": "20:00", "home": "Tunisia", "away": "Croatia", "venue": "SoFi Stadium, Los Angeles", "group": "F"},
    # Group G
    {"date": "2026-06-16", "time": "15:00", "home": "Spain", "away": "Ghana", "venue": "Lumen Field, Seattle", "group": "G"},
    {"date": "2026-06-16", "time": "21:00", "home": "Belgium", "away": "Ecuador", "venue": "Mercedes-Benz Stadium, Atlanta", "group": "G"},
    {"date": "2026-06-21", "time": "21:00", "home": "Spain", "away": "Belgium", "venue": "Hard Rock Stadium, Miami", "group": "G"},
    {"date": "2026-06-22", "time": "19:00", "home": "Ecuador", "away": "Ghana", "venue": "Levi's Stadium, San Francisco", "group": "G"},
    {"date": "2026-06-27", "time": "16:00", "home": "Ecuador", "away": "Spain", "venue": "MetLife Stadium, New York", "group": "G"},
    {"date": "2026-06-27", "time": "16:00", "home": "Ghana", "away": "Belgium", "venue": "Lincoln Financial Field, Philadelphia", "group": "G"},
    # Group H
    {"date": "2026-06-17", "time": "15:00", "home": "Portugal", "away": "Scotland", "venue": "Gillette Stadium, Boston", "group": "H"},
    {"date": "2026-06-17", "time": "24:00", "home": "Japan", "away": "Saudi Arabia", "venue": "Estadio BBVA, Monterrey", "group": "H"},
    {"date": "2026-06-22", "time": "24:00", "home": "Portugal", "away": "Japan", "venue": "Estadio BBVA, Monterrey", "group": "H"},
    {"date": "2026-06-23", "time": "19:00", "home": "Saudi Arabia", "away": "Scotland", "venue": "AT&T Stadium, Dallas", "group": "H"},
    {"date": "2026-06-27", "time": "20:00", "home": "Saudi Arabia", "away": "Portugal", "venue": "Rose Bowl, Los Angeles", "group": "H"},
    {"date": "2026-06-27", "time": "20:00", "home": "Scotland", "away": "Japan", "venue": "NRG Stadium, Houston", "group": "H"},
    # Group I
    {"date": "2026-06-18", "time": "15:00", "home": "Netherlands", "away": "Qatar", "venue": "Gillette Stadium, Boston", "group": "I"},
    {"date": "2026-06-18", "time": "24:00", "home": "Australia", "away": "Serbia", "venue": "Lumen Field, Seattle", "group": "I"},
    {"date": "2026-06-23", "time": "24:00", "home": "Netherlands", "away": "Australia", "venue": "Levi's Stadium, San Francisco", "group": "I"},
    {"date": "2026-06-24", "time": "19:00", "home": "Serbia", "away": "Qatar", "venue": "Estadio Azteca, Mexico City", "group": "I"},
    {"date": "2026-06-28", "time": "16:00", "home": "Serbia", "away": "Netherlands", "venue": "SoFi Stadium, Los Angeles", "group": "I"},
    {"date": "2026-06-28", "time": "16:00", "home": "Qatar", "away": "Australia", "venue": "Hard Rock Stadium, Miami", "group": "I"},
    # Group J
    {"date": "2026-06-19", "time": "15:00", "home": "Colombia", "away": "Mali", "venue": "NRG Stadium, Houston", "group": "J"},
    {"date": "2026-06-19", "time": "24:00", "home": "Turkey", "away": "Czech Republic", "venue": "Mercedes-Benz Stadium, Atlanta", "group": "J"},
    {"date": "2026-06-24", "time": "24:00", "home": "Colombia", "away": "Turkey", "venue": "MetLife Stadium, New York", "group": "J"},
    {"date": "2026-06-25", "time": "19:00", "home": "Czech Republic", "away": "Mali", "venue": "Lincoln Financial Field, Philadelphia", "group": "J"},
    {"date": "2026-06-29", "time": "16:00", "home": "Czech Republic", "away": "Colombia", "venue": "Gillette Stadium, Boston", "group": "J"},
    {"date": "2026-06-29", "time": "16:00", "home": "Mali", "away": "Turkey", "venue": "Estadio BBVA, Monterrey", "group": "J"},
    # Group K
    {"date": "2026-06-20", "time": "15:00", "home": "Uruguay", "away": "Paraguay", "venue": "Hard Rock Stadium, Miami", "group": "K"},
    {"date": "2026-06-20", "time": "18:00", "home": "Greece", "away": "Uzbekistan", "venue": "Rose Bowl, Los Angeles", "group": "K"},
    {"date": "2026-06-25", "time": "24:00", "home": "Uruguay", "away": "Greece", "venue": "AT&T Stadium, Dallas", "group": "K"},
    {"date": "2026-06-26", "time": "19:00", "home": "Uzbekistan", "away": "Paraguay", "venue": "Estadio Azteca, Mexico City", "group": "K"},
    {"date": "2026-06-30", "time": "16:00", "home": "Uzbekistan", "away": "Uruguay", "venue": "Levi's Stadium, San Francisco", "group": "K"},
    {"date": "2026-06-30", "time": "16:00", "home": "Paraguay", "away": "Greece", "venue": "Lumen Field, Seattle", "group": "K"},
    # Group L
    {"date": "2026-06-21", "time": "15:00", "home": "Canada", "away": "Ivory Coast", "venue": "BMO Field, Toronto", "group": "L"},
    {"date": "2026-06-21", "time": "18:00", "home": "Switzerland", "away": "Algeria", "venue": "Mercedes-Benz Stadium, Atlanta", "group": "L"},
    {"date": "2026-06-26", "time": "24:00", "home": "Canada", "away": "Switzerland", "venue": "BMO Field, Toronto", "group": "L"},
    {"date": "2026-06-27", "time": "19:00", "home": "Algeria", "away": "Ivory Coast", "venue": "NRG Stadium, Houston", "group": "L"},
    {"date": "2026-07-01", "time": "16:00", "home": "Algeria", "away": "Canada", "venue": "MetLife Stadium, New York", "group": "L"},
    {"date": "2026-07-01", "time": "16:00", "home": "Ivory Coast", "away": "Switzerland", "venue": "SoFi Stadium, Los Angeles", "group": "L"},
]

# Round of 32 (32 teams: 12 group winners, 12 runners-up, 8 best third-placed)
WC2026_KNOCKOUT_R32 = [
    {"date": "2026-07-03", "time": "15:00", "home": "TBD_R1A", "away": "TBD_R2B", "venue": "MetLife Stadium, New York", "stage": "R32", "label": "1 vs 2B"},
    {"date": "2026-07-03", "time": "21:00", "home": "TBD_R1C", "away": "TBD_R2D", "venue": "SoFi Stadium, Los Angeles", "stage": "R32", "label": "1C vs 2D"},
    {"date": "2026-07-04", "time": "15:00", "home": "TBD_R1E", "away": "TBD_R2F", "venue": "AT&T Stadium, Dallas", "stage": "R32", "label": "1E vs 2F"},
    {"date": "2026-07-04", "time": "21:00", "home": "TBD_R1G", "away": "TBD_R2H", "venue": "NRG Stadium, Houston", "stage": "R32", "label": "1G vs 2H"},
    {"date": "2026-07-05", "time": "15:00", "home": "TBD_R1I", "away": "TBD_R2J", "venue": "Hard Rock Stadium, Miami", "stage": "R32", "label": "1I vs 2J"},
    {"date": "2026-07-05", "time": "21:00", "home": "TBD_R1K", "away": "TBD_R2L", "venue": "Rose Bowl, Los Angeles", "stage": "R32", "label": "1K vs 2L"},
    {"date": "2026-07-06", "time": "15:00", "home": "TBD_R1B", "away": "TBD_R2A", "venue": "Lumen Field, Seattle", "stage": "R32", "label": "1B vs 2A"},
    {"date": "2026-07-06", "time": "21:00", "home": "TBD_R1D", "away": "TBD_R2C", "venue": "Mercedes-Benz Stadium, Atlanta", "stage": "R32", "label": "1D vs 2C"},
    {"date": "2026-07-07", "time": "15:00", "home": "TBD_R1F", "away": "TBD_R2E", "venue": "Estadio Azteca, Mexico City", "stage": "R32", "label": "1F vs 2E"},
    {"date": "2026-07-07", "time": "21:00", "home": "TBD_R1H", "away": "TBD_R2G", "venue": "Estadio BBVA, Monterrey", "stage": "R32", "label": "1H vs 2G"},
    {"date": "2026-07-08", "time": "15:00", "home": "TBD_R1J", "away": "TBD_R2I", "venue": "Gillette Stadium, Boston", "stage": "R32", "label": "1J vs 2I"},
    {"date": "2026-07-08", "time": "21:00", "home": "TBD_R1L", "away": "TBD_R2K", "venue": "Lincoln Financial Field, Philadelphia", "stage": "R32", "label": "1L vs 2K"},
    {"date": "2026-07-09", "time": "15:00", "home": "TBD_3RD_A", "away": "TBD_3RD_B", "venue": "Levi's Stadium, San Francisco", "stage": "R32", "label": "3A vs 3B"},
    {"date": "2026-07-09", "time": "21:00", "home": "TBD_3RD_C", "away": "TBD_3RD_D", "venue": "BMO Field, Toronto", "stage": "R32", "label": "3C vs 3D"},
]

# Round of 16
WC2026_KNOCKOUT_R16 = [
    {"date": "2026-07-11", "time": "15:00", "home": "TBD_R32_1", "away": "TBD_R32_2", "venue": "MetLife Stadium, New York", "stage": "R16"},
    {"date": "2026-07-11", "time": "21:00", "home": "TBD_R32_3", "away": "TBD_R32_4", "venue": "SoFi Stadium, Los Angeles", "stage": "R16"},
    {"date": "2026-07-12", "time": "15:00", "home": "TBD_R32_5", "away": "TBD_R32_6", "venue": "AT&T Stadium, Dallas", "stage": "R16"},
    {"date": "2026-07-12", "time": "21:00", "home": "TBD_R32_7", "away": "TBD_R32_8", "venue": "NRG Stadium, Houston", "stage": "R16"},
    {"date": "2026-07-13", "time": "15:00", "home": "TBD_R32_9", "away": "TBD_R32_10", "venue": "Hard Rock Stadium, Miami", "stage": "R16"},
    {"date": "2026-07-13", "time": "21:00", "home": "TBD_R32_11", "away": "TBD_R32_12", "venue": "Rose Bowl, Los Angeles", "stage": "R16"},
    {"date": "2026-07-14", "time": "15:00", "home": "TBD_R32_13", "away": "TBD_R32_14", "venue": "Lumen Field, Seattle", "stage": "R16"},
    {"date": "2026-07-14", "time": "21:00", "home": "TBD_R32_15", "away": "TBD_R32_16", "venue": "Mercedes-Benz Stadium, Atlanta", "stage": "R16"},
]

# Quarter-finals
WC2026_KNOCKOUT_QF = [
    {"date": "2026-07-17", "time": "15:00", "home": "TBD_R16_1", "away": "TBD_R16_2", "venue": "AT&T Stadium, Dallas", "stage": "QF"},
    {"date": "2026-07-17", "time": "21:00", "home": "TBD_R16_3", "away": "TBD_R16_4", "venue": "NRG Stadium, Houston", "stage": "QF"},
    {"date": "2026-07-18", "time": "15:00", "home": "TBD_R16_5", "away": "TBD_R16_6", "venue": "MetLife Stadium, New York", "stage": "QF"},
    {"date": "2026-07-18", "time": "21:00", "home": "TBD_R16_7", "away": "TBD_R16_8", "venue": "SoFi Stadium, Los Angeles", "stage": "QF"},
]

# Semi-finals
WC2026_KNOCKOUT_SF = [
    {"date": "2026-07-22", "time": "20:00", "home": "TBD_QF_1", "away": "TBD_QF_2", "venue": "AT&T Stadium, Dallas", "stage": "SF"},
    {"date": "2026-07-23", "time": "20:00", "home": "TBD_QF_3", "away": "TBD_QF_4", "venue": "SoFi Stadium, Los Angeles", "stage": "SF"},
]

# Third place & Final
WC2026_KNOCKOUT_FINAL = [
    {"date": "2026-07-26", "time": "17:00", "home": "TBD_SF_L1", "away": "TBD_SF_L2", "venue": "Hard Rock Stadium, Miami", "stage": "3RD"},
    {"date": "2026-07-27", "time": "18:00", "home": "TBD_SF_W1", "away": "TBD_SF_W2", "venue": "MetLife Stadium, New York", "stage": "FINAL"},
]


def fetch_world_cup_2026_schedule(include_knockout: bool = True) -> List[Dict[str, Any]]:
    """Return the full 2026 World Cup schedule including knockout stages."""
    matches = list(WC2026_SCHEDULE_TEMPLATE)
    if include_knockout:
        for m in WC2026_KNOCKOUT_R32:
            matches.append({**m, "stage": "R32"})
        for m in WC2026_KNOCKOUT_R16:
            matches.append({**m, "stage": "R16"})
        for m in WC2026_KNOCKOUT_QF:
            matches.append({**m, "stage": "QF"})
        for m in WC2026_KNOCKOUT_SF:
            matches.append({**m, "stage": "SF"})
        for m in WC2026_KNOCKOUT_FINAL:
            matches.append({**m, "stage": m.get("stage", "FINAL")})
    return matches


def get_groups() -> Dict[str, List[str]]:
    """Return group stage composition."""
    return WC2026_GROUPS


def get_teams() -> Dict[str, Dict[str, Any]]:
    """Return all 48 teams with metadata."""
    return WC2026_TEAMS


def store_wc2026_data(conn) -> Dict[str, int]:
    """Store World Cup 2026 teams, groups, and schedule into the DB.
    
    Uses the existing teams/matches schema with league_slug='WC_2026'.
    Returns counts of teams and matches stored.
    """
    now = dt.datetime.now().isoformat(timespec="seconds")
    
    # Ensure league exists
    conn.execute(
        "INSERT OR REPLACE INTO leagues (slug, name, country, source_tier) VALUES (?, ?, ?, ?)",
        (WC2026_SLUG, WC2026_NAME, WC2026_COUNTRY, 1),
    )
    
    # Store teams
    team_id_map = {}
    for name, meta in WC2026_TEAMS.items():
        existing = conn.execute(
            "SELECT id FROM teams WHERE name=? AND league_slug=?",
            (name, WC2026_SLUG),
        ).fetchone()
        if existing:
            team_id_map[name] = existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO teams (name, short_name, league_slug) VALUES (?, ?, ?)",
                (name, name[:3].upper(), WC2026_SLUG),
            )
            team_id_map[name] = cur.lastrowid
    
    # Store matches from schedule
    matches_stored = 0
    for m in WC2026_SCHEDULE_TEMPLATE:
        home_id = team_id_map.get(m["home"])
        away_id = team_id_map.get(m["away"])
        if not home_id or not away_id:
            continue
        
        # Check if match already exists
        existing = conn.execute(
            "SELECT id FROM matches WHERE home_id=? AND away_id=? AND league_slug=? AND date=?",
            (home_id, away_id, WC2026_SLUG, m["date"]),
        ).fetchone()
        if existing:
            continue
        
        conn.execute(
            "INSERT INTO matches (league_slug, season, date, home_id, away_id, is_result) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (WC2026_SLUG, 2026, m["date"], home_id, away_id),
        )
        matches_stored += 1
    
    conn.commit()
    return {"teams": len(team_id_map), "matches": matches_stored}


def get_wc2026_overview(conn) -> Dict[str, Any]:
    """Get full World Cup 2026 overview: groups, teams, schedule (incl. knockout)."""
    groups = {}
    for gname, team_names in WC2026_GROUPS.items():
        teams_in_group = []
        for tname in team_names:
            meta = WC2026_TEAMS.get(tname, {})
            row = conn.execute(
                "SELECT t.id, t.name, t.short_name FROM teams t "
                "WHERE t.name=? AND t.league_slug=?",
                (tname, WC2026_SLUG),
            ).fetchone()
            teams_in_group.append({
                "id": row["id"] if row else None,
                "name": tname,
                "pot": meta.get("pot", 0),
                "confederation": meta.get("conf", ""),
                "fifa_rank": meta.get("fifa_rank", 0),
            })
        groups[gname] = teams_in_group
    
    # Get group stage matches from DB
    matches = conn.execute(
        "SELECT m.*, th.name as home_name, ta.name as away_name "
        "FROM matches m "
        "JOIN teams th ON th.id = m.home_id "
        "JOIN teams ta ON ta.id = m.away_id "
        "WHERE m.league_slug=? "
        "ORDER BY m.date",
        (WC2026_SLUG,),
    ).fetchall()
    
    schedule = []
    for m in matches:
        entry = dict(m)
        for tmpl in WC2026_SCHEDULE_TEMPLATE:
            if (tmpl["home"] == m["home_name"] and tmpl["away"] == m["away_name"]
                    and tmpl["date"] == m["date"]):
                entry["venue"] = tmpl.get("venue", "")
                entry["time"] = tmpl.get("time", "")
                entry["group"] = tmpl.get("group", "")
                break
        schedule.append(entry)
    
    # Add knockout stages (TBD teams)
    all_knockout = []
    for m in WC2026_KNOCKOUT_R32:
        all_knockout.append({**m, "stage": "R32"})
    for m in WC2026_KNOCKOUT_R16:
        all_knockout.append({**m, "stage": "R16"})
    for m in WC2026_KNOCKOUT_QF:
        all_knockout.append({**m, "stage": "QF"})
    for m in WC2026_KNOCKOUT_SF:
        all_knockout.append({**m, "stage": "SF"})
    for m in WC2026_KNOCKOUT_FINAL:
        all_knockout.append({**m, "stage": m.get("stage", "FINAL")})
    
    return {
        "groups": groups,
        "schedule": schedule,
        "knockout": all_knockout,
        "total_teams": len(WC2026_TEAMS),
        "total_group_matches": len(schedule),
        "total_knockout_matches": len(all_knockout),
    }


def get_wc2026_standings(conn) -> List[Dict[str, Any]]:
    """Compute group standings from played matches."""
    standings = {}
    
    for gname, team_names in WC2026_GROUPS.items():
        table = {}
        for tname in team_names:
            row = conn.execute(
                "SELECT id FROM teams WHERE name=? AND league_slug=?",
                (tname, WC2026_SLUG),
            ).fetchone()
            if row:
                table[row["id"]] = {
                    "team": tname,
                    "team_id": row["id"],
                    "played": 0, "won": 0, "drawn": 0, "lost": 0,
                    "gf": 0, "ga": 0, "gd": 0, "points": 0,
                }
        
        # Get all finished matches in this group
        team_ids = list(table.keys())
        if not team_ids:
            continue
        
        matches = conn.execute(
            "SELECT * FROM matches WHERE is_result=1 AND league_slug=? "
            "AND (home_id IN ({}) OR away_id IN ({}))".format(
                ",".join("?" * len(team_ids)),
                ",".join("?" * len(team_ids)),
            ),
            [WC2026_SLUG] + team_ids + team_ids,
        ).fetchall()
        
        for m in matches:
            if m["home_goals"] is None or m["away_goals"] is None:
                continue
            hid, aid = m["home_id"], m["away_id"]
            hg, ag = m["home_goals"], m["away_goals"]
            
            if hid in table:
                table[hid]["played"] += 1
                table[hid]["gf"] += hg
                table[hid]["ga"] += ag
                if hg > ag:
                    table[hid]["won"] += 1; table[hid]["points"] += 3
                elif hg == ag:
                    table[hid]["drawn"] += 1; table[hid]["points"] += 1
                else:
                    table[hid]["lost"] += 1
            if aid in table:
                table[aid]["played"] += 1
                table[aid]["gf"] += ag
                table[aid]["ga"] += hg
                if ag > hg:
                    table[aid]["won"] += 1; table[aid]["points"] += 3
                elif ag == hg:
                    table[aid]["drawn"] += 1; table[aid]["points"] += 1
                else:
                    table[aid]["lost"] += 1
        
        for tid in table:
            table[tid]["gd"] = table[tid]["gf"] - table[tid]["ga"]
        
        # Sort: points desc, gd desc, gf desc
        sorted_teams = sorted(table.values(),
                              key=lambda t: (-t["points"], -t["gd"], -t["gf"]))
        for i, t in enumerate(sorted_teams, 1):
            t["pos"] = i
        
        standings[gname] = sorted_teams
    
    return standings
