"""
scrapers/ — collection of Botasaurus-powered football data scrapers.

Each module exposes high-level functions used by data_collector.py.
Re-exports preserve backward compatibility with previous `import scrapers` calls.
"""
from .understat import (
    LEAGUES,
    fetch_understat_league,
    fetch_all_leagues,
    UNDERSTAT_BASE,
)

# New sources
from . import clubelo
from . import openmeteo
from . import transfermarkt
from . import flashscore     # legacy, kept for the @browser code; not used in pipelines
from . import thesportsdb    # backup live source
from . import espn           # primary live/worldwide source — comprehensive coverage
from . import historical_odds  # bookmaker odds (football-data.co.uk)
from . import sstats            # sstats.net Football API (Glicko, multi-bookmaker odds, text summaries)
from . import web             # Jina Reader + DuckDuckGo universal web scraper
from . import fbref           # FBref xG data for 25+ leagues

__all__ = [
    "LEAGUES", "fetch_understat_league", "fetch_all_leagues", "UNDERSTAT_BASE",
    "clubelo", "openmeteo", "transfermarkt", "flashscore", "thesportsdb", "espn",
    "historical_odds", "web", "fbref",
]
