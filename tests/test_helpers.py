"""
test_helpers.py — Tests for helpers.py: consensus odds, league maps.
"""
from __future__ import annotations

import pytest


class TestConsensusFromOdds:
    """Tests for _consensus_from_odds (imported from routers.football)."""

    def test_empty_list(self):
        from routers.football import _consensus_from_odds
        assert _consensus_from_odds([]) is None

    def test_valid_odds(self):
        from routers.football import _consensus_from_odds
        odds = [
            {"marketName": "Match Winner", "odds": [
                {"name": "Home", "value": "2.0"},
                {"name": "Draw", "value": "3.5"},
                {"name": "Away", "value": "3.8"},
            ]},
        ]
        result = _consensus_from_odds(odds)
        assert result is not None
        assert result["bookmaker_count"] == 1
        assert result["home_odds"] == 2.0
        assert result["draw_odds"] == 3.5
        assert result["away_odds"] == 3.8
        # Implied probs should sum to ~1
        total = result["implied_home"] + result["implied_draw"] + result["implied_away"]
        assert abs(total - 1.0) < 0.01

    def test_no_match_winner_market(self):
        from routers.football import _consensus_from_odds
        odds = [{"marketName": "Over/Under", "odds": []}]
        assert _consensus_from_odds(odds) is None

    def test_multiple_bookmakers(self):
        from routers.football import _consensus_from_odds
        odds = [
            {"marketName": "Match Winner", "odds": [
                {"name": "Home", "value": "2.0"},
                {"name": "Draw", "value": "3.4"},
                {"name": "Away", "value": "3.6"},
            ]},
            {"marketName": "Match Winner", "odds": [
                {"name": "Home", "value": "2.1"},
                {"name": "Draw", "value": "3.3"},
                {"name": "Away", "value": "3.5"},
            ]},
        ]
        result = _consensus_from_odds(odds)
        assert result is not None
        assert result["bookmaker_count"] == 2
        assert 2.0 <= result["home_odds"] <= 2.1


class TestLeagueNameMap:
    """Verify league name map has expected entries."""

    def test_top_leagues_present(self):
        from helpers import LEAGUE_NAME_MAP
        assert "Premier League" in LEAGUE_NAME_MAP
        assert "La Liga" in LEAGUE_NAME_MAP
        assert "Bundesliga" in LEAGUE_NAME_MAP
        assert "Serie A" in LEAGUE_NAME_MAP
        assert "Ligue 1" in LEAGUE_NAME_MAP

    def test_values_are_russian(self):
        from helpers import LEAGUE_NAME_MAP
        for key in ["Premier League", "La Liga", "Champions League"]:
            val = LEAGUE_NAME_MAP[key]
            assert any(ord(c) > 127 for c in val), f"{key} → {val} should contain Cyrillic"


class TestTeamNameMap:
    """Verify team name map has expected entries."""

    def test_top_clubs_present(self):
        from helpers import TEAM_NAME_MAP
        assert "Barcelona" in TEAM_NAME_MAP
        assert "Real Madrid" in TEAM_NAME_MAP
        assert "Bayern Munich" in TEAM_NAME_MAP

    def test_international_teams(self):
        from helpers import TEAM_NAME_MAP
        assert "Brazil" in TEAM_NAME_MAP
        assert "Argentina" in TEAM_NAME_MAP
        assert "Germany" in TEAM_NAME_MAP
