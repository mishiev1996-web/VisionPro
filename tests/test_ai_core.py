"""
test_ai_core.py — Tests for ai_core.py: PROB parser and chat wrapper.
"""
from __future__ import annotations

import pytest
from ai_core import parse_prob_line


class TestParseProbLine:
    """Tests for the unified PROB line parser."""

    def test_football_basic(self):
        text = "PROB:home=0.45:draw=0.25:away=0.30"
        result = parse_prob_line(text)
        assert result is not None
        assert result["home_win"] == 45.0
        assert result["draw"] == 25.0
        assert result["away_win"] == 30.0

    def test_football_with_bet_and_confidence(self):
        text = "PROB:home=0.50:draw=0.25:away=0.25:bet=Победа хозяев:confidence=Высокая"
        result = parse_prob_line(text)
        assert result is not None
        assert result["home_win"] == 50.0
        assert result["main_bet"] == "Победа хозяев"
        assert result["confidence"] == "Высокая"

    def test_football_with_totals_and_btts(self):
        text = "PROB:home=0.40:draw=0.20:away=0.40:total_over=0.60:total_under=0.40:btts_yes=0.55:btts_no=0.45"
        result = parse_prob_line(text)
        assert result is not None
        assert result["total_over_2_5"] == 60.0
        assert result["total_under_2_5"] == 40.0
        assert result["btts_yes"] == 55.0
        assert result["btts_no"] == 45.0

    def test_tennis_basic(self):
        text = "PROB:p1=0.60:p2=0.40"
        result = parse_prob_line(text)
        assert result is not None
        assert result["player1_win"] == 60.0
        assert result["player2_win"] == 40.0

    def test_tennis_with_bet(self):
        text = "PROB:p1=0.55:p2=0.45:bet=Победа игрока 1:confidence=Средняя"
        result = parse_prob_line(text)
        assert result is not None
        assert result["main_bet"] == "Победа игрока 1"
        assert result["confidence"] == "Средняя"

    def test_tennis_odds_conversion(self):
        """Values > 1 should be treated as odds and converted to probabilities."""
        text = "PROB:p1=1.50:p2=2.50"
        result = parse_prob_line(text)
        assert result is not None
        # 1/1.50 = 0.667, 1/2.50 = 0.400, total = 1.067
        # p1 = 0.667/1.067 ≈ 62.5%, p2 = 0.400/1.067 ≈ 37.5%
        assert 62.0 <= result["player1_win"] <= 63.0
        assert 37.0 <= result["player2_win"] <= 38.0

    def test_no_prob_line(self):
        text = "Just some analysis text without PROB"
        result = parse_prob_line(text)
        assert result is None

    def test_none_input(self):
        result = parse_prob_line(None)
        assert result is None

    def test_empty_string(self):
        result = parse_prob_line("")
        assert result is None

    def test_football_in_long_text(self):
        """PROB line embedded in longer analysis text."""
        text = """Here is the analysis of the match.
Arsenal are in great form with 5 wins in a row.
The away team has been struggling recently.
PROB:home=0.55:draw=0.22:away=0.23:bet=Победа хозяев:confidence=Высокая
This is a strong recommendation."""
        result = parse_prob_line(text)
        assert result is not None
        assert result["home_win"] == 55.0
        assert result["main_bet"] == "Победа хозяев"
