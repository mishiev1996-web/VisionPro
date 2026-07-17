"""
config.py — Centralized configuration for Football AI Predictor.

All magic numbers, API keys, and tunable parameters live here.
Override via environment variables where applicable.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "football.db"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
API_KEY_DIR = PROJECT_ROOT / "Апи"

# ── Database ─────────────────────────────────────────────────────────────────
DB_TIMEOUT = 30  # seconds
DB_WAL_MODE = True

# ── Data Collection ──────────────────────────────────────────────────────────
SOURCE_TIMEOUT = 30  # seconds per individual source call
MAX_RETRIES = 2
RETRY_DELAY_BASE = 2.0  # exponential backoff base
PARALLEL_WORKERS = 4  # for Understat parallel fetch
UNDERSTAT_PAUSE = 0.3  # seconds between requests

# ── Model Training ───────────────────────────────────────────────────────────
ROLLING_WINDOW = 10
SHORT_WINDOW = 3
H2H_WINDOW = 5
FORM_WINDOW = 5
MIN_PRIOR_FOR_TRAINING = 30
MIN_LEAGUE_TRAIN_ROWS = 3000
TIME_DECAY_HALF_LIFE_DAYS = 365
CV_FOLDS = 5
N_OOF_FOLDS = 3
USE_CLASS_WEIGHTS = True
PER_LEAGUE_PROVES_ITSELF = True

# ── Scheduler ────────────────────────────────────────────────────────────────
UNDERSTAT_REFRESH_HOURS = 6
LIVE_REFRESH_MINUTES = 5
ESPN_REFRESH_MINUTES = 30

# ── API ──────────────────────────────────────────────────────────────────────
MAX_SSE_EVENTS = 10000
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 200
MODEL_STATS_CACHE_SECONDS = 300
SSTATS_CACHE_MAX = 5000

# ── API Keys ─────────────────────────────────────────────────────────────────
def _load_api_key(filename: str, env_var: str = "") -> str:
    """Load API key from env var or file."""
    key = os.environ.get(env_var, "") if env_var else ""
    if not key:
        key_path = API_KEY_DIR / filename
        if key_path.exists():
            with open(key_path, "r") as f:
                key = f.read().strip()
    return key

POLZA_API_KEY = _load_api_key("key.txt", "POLZA_API_KEY")
SSTATS_API_KEY = _load_api_key("sstats_key.txt", "SSTATS_API_KEY")

# ── AI / LLM ────────────────────────────────────────────────────────────────
POLZA_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_AI_MODEL = "deepseek/deepseek-v4-flash"

AI_MODELS = [
    {"id": "google/gemini-3.5-flash", "name": "Gemini 3.5 Flash", "tag": "Быстрая"},
    {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek V4 Flash", "tag": "Быстрая"},
    {"id": "x-ai/grok-4.3", "name": "Grok 4.3", "tag": "Быстрая"},
    {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro", "tag": "Мощная"},
    {"id": "anthropic/claude-sonnet-5", "name": "Claude Sonnet 5", "tag": "Качественная"},
]
AI_MAX_TOKENS = 2000
AI_TEMPERATURE = 0.7
AI_TIMEOUT = 60

# ── Feature Flags ────────────────────────────────────────────────────────────
ENABLE_ESPN_REFRESH = True
ENABLE_FLASHSCORE_REFRESH = True
ENABLE_UNDERSTAT_REFRESH = True
ENABLE_WEATHER = True
ENABLE_INJURIES = True
ENABLE_ODDS = True
ENABLE_SSTATS = True
ENABLE_ESPN_AI = False  # ESPN in AI analysis pipeline (set False to skip)

# ── Supported Sports ────────────────────────────────────────────────────────
SUPPORTED_SPORTS = ["football", "tennis"]
DEFAULT_SPORT = "football"

# ── Tennis ───────────────────────────────────────────────────────────────────
TENNIS_DB_PATH = DATA_DIR / "tennis.db"
TENNIS_API_HOST = "tennisapi1.p.rapidapi.com"
TENNIS_API_KEY = _load_api_key("tennis_key.txt", "TENNIS_API_KEY")

# ── Understat Leagues (verified: only these 6 exist on Understat) ────────────
UNDERSTAT_LEAGUES = {
    "EPL":        {"name": "Premier League",  "country": "England"},
    "La_liga":    {"name": "La Liga",         "country": "Spain"},
    "Bundesliga": {"name": "Bundesliga",      "country": "Germany"},
    "Serie_A":    {"name": "Serie A",         "country": "Italy"},
    "Ligue_1":    {"name": "Ligue 1",         "country": "France"},
    "RFPL":       {"name": "Russian Premier", "country": "Russia"},
}

# ── League Tiers ────────────────────────────────────────────────────────────
# Tier 1: Full xG + Elo + odds + form + weather (Understat leagues)
# Tier 2: Elo + form + odds (no xG, but still trainable with goals-based features)
# Tier 3: ESPN coverage + odds only (limited features, used for live predictions)
#
# Trainable leagues = Tier 1 + Tier 2. Tier 3 is for display/live only.
LEAGUE_TIERS = {
    # ── Tier 1: Full features (Understat xG) ──
    "EPL":           {"tier": 1, "name": "Premier League",    "country": "England"},
    "La_liga":       {"tier": 1, "name": "La Liga",           "country": "Spain"},
    "Bundesliga":    {"tier": 1, "name": "Bundesliga",        "country": "Germany"},
    "Serie_A":       {"tier": 1, "name": "Serie A",           "country": "Italy"},
    "Ligue_1":       {"tier": 1, "name": "Ligue 1",           "country": "France"},
    "RFPL":          {"tier": 1, "name": "Russian Premier",   "country": "Russia"},

    # ── Tier 2: Elo + form + odds (no xG) ──
    "Eredivisie":    {"tier": 2, "name": "Eredivisie",        "country": "Netherlands"},
    "Primeira_Liga": {"tier": 2, "name": "Primeira Liga",     "country": "Portugal"},
    "Super_Lig":     {"tier": 2, "name": "Süper Lig",         "country": "Turkey"},
    "Championship":  {"tier": 2, "name": "Championship",      "country": "England"},
    "Belgian_First": {"tier": 2, "name": "Belgian Pro League","country": "Belgium"},
    "Greek_Super":   {"tier": 2, "name": "Super League",      "country": "Greece"},

    # ── Tier 3: Coverage only (live predictions, no training) ──
    "MLS":           {"tier": 3, "name": "MLS",               "country": "USA"},
    "Brasileirao":   {"tier": 3, "name": "Brasileirão",       "country": "Brazil"},
    "Argentine_Primary": {"tier": 3, "name": "Primera División", "country": "Argentina"},
    "Liga_MX":       {"tier": 3, "name": "Liga MX",           "country": "Mexico"},
    "Saudi_Pro":     {"tier": 3, "name": "Saudi Pro League",  "country": "Saudi Arabia"},
    "J-League":      {"tier": 3, "name": "J.League",          "country": "Japan"},
    "Belarus_PL":    {"tier": 3, "name": "Belarus Premier",   "country": "Belarus"},
    "Belarus_1D":    {"tier": 3, "name": "Belarus 1. Division","country": "Belarus"},
    "Kazakhstan_PL": {"tier": 3, "name": "Kazakhstan Premier","country": "Kazakhstan"},
    "Kazakhstan_1D": {"tier": 3, "name": "Kazakhstan 1. Division","country": "Kazakhstan"},
    "Iceland_Urvalsdeild": {"tier": 3, "name": "Úrvalsdeild", "country": "Iceland"},
}

# Leagues eligible for model training (Tier 1 + Tier 2)
TRAINABLE_LEAGUES = {k: v for k, v in LEAGUE_TIERS.items() if v["tier"] <= 2}

# ESPN league slugs → our league slugs (used by collect_espn_all)
ESPN_TO_OURS = {
    "eng.1": "EPL", "esp.1": "La_liga", "ger.1": "Bundesliga",
    "ita.1": "Serie_A", "fra.1": "Ligue_1", "rus.1": "RFPL",
    "ned.1": "Eredivisie", "por.1": "Primeira_Liga",
    "tur.1": "Super_Lig", "eng.2": "Championship",
    "usa.1": "MLS", "bra.1": "Brasileirao",
    "arg.1": "Argentine_Primary", "bel.1": "Belgian_First",
    "gre.1": "Greek_Super",
    "mex.1": "Liga_MX", "ksa.1": "Saudi_Pro",
    "jpn.1": "J-League",
}

# football-data.co.uk league codes for odds collection
FDCOOKIE_CODES = {
    "EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1",
    "Serie_A": "I1", "Ligue_1": "F1", "Championship": "E1",
    "La_liga_2": "SP2", "Serie_B": "I2", "Ligue_2": "F2",
    "2_Bundesliga": "D2", "Eredivisie": "N1", "Primeira_Liga": "P1",
    "Belgian_First": "B1", "Super_Lig": "T1", "Greek_Super": "G1",
}
