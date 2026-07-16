"""
conftest.py — Shared fixtures for pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database for isolated tests."""
    import db as _db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    old_db_path = _db.DB_PATH
    _db.DB_PATH = db_path
    _db.init_db()
    yield db_path
    _db.DB_PATH = old_db_path
    os.unlink(db_path)


@pytest.fixture
def sample_match_data():
    """Sample match dict for testing feature building and prediction."""
    return {
        "home_name": "Test Home",
        "away_name": "Test Away",
        "league_slug": "EPL",
        "season": 2025,
        "date": "2025-01-15",
    }


@pytest.fixture
def sample_features():
    """Sample feature dict (19 features for batch model)."""
    return [
        1.5, 1.2, 0.4, 10, 2, 0.2,   # home stats
        1.0, 1.5, 0.3, 5, -1, 0.3,    # away stats
        100, 3,                          # elo diff, h2h wins
        0.45, 0.25, 0.30,              # implied odds
        7, 7,                           # rest days
    ]
