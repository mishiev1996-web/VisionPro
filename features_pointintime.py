"""
features_pointintime.py — Point-in-time фичи, рассчитанные по истории матчей в БД.

ВАЖНО: Все функции принимают обязательный параметр `before_date` для
предотвращения утечки будущих данных. Это аналог существующих
rolling-window фичей в train.py (ROLLING_WINDOW=10 в config.py).

Используются ТОЛЬКО исторические данные из таблиц matches + match_odds.
fetch_season_table ИЗ API НЕ вызывается (пункт C из ТЗ).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# Cache for preloaded data
_team_matches_cache: Dict[int, List[Dict]] = {}


def _preload_team_matches(all_matches: List[Dict]) -> None:
    """Build team_id → [(date, home_id, away_id, home_goals, away_goals)] mapping."""
    global _team_matches_cache
    _team_matches_cache.clear()
    for m in all_matches:
        hid = m.get("home_id")
        aid = m.get("away_id")
        date_str = (m.get("date") or "")[:10]
        hg = m.get("home_goals")
        ag = m.get("away_goals")
        if not date_str or hg is None or ag is None:
            continue
        entry = {"date": date_str, "home_id": hid, "away_id": aid,
                 "home_goals": hg, "away_goals": ag}
        _team_matches_cache.setdefault(hid, []).append(entry)
        _team_matches_cache.setdefault(aid, []).append(entry)
    # Sort each team's matches by date descending
    for tid in _team_matches_cache:
        _team_matches_cache[tid].sort(key=lambda x: x["date"], reverse=True)


def _get_team_matches_before(team_id: int, before_date: str, limit: int = 20) -> List[Dict]:
    """Get team's matches before a specific date (point-in-time safe)."""
    games = _team_matches_cache.get(team_id, [])
    return [m for m in games if m["date"] < before_date][:limit]


def compute_over25_rate(team_id: int, before_date: str, window: int = 20) -> float:
    """Compute % of matches with total goals > 2.5 for a team.

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date (YYYY-MM-DD)
        window: Number of recent matches to consider

    Returns:
        Rate between 0.0 and 1.0, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 5:
        return None
    over25_count = 0
    for m in matches:
        total = m["home_goals"] + m["away_goals"]
        if total > 2.5:
            over25_count += 1
    return over25_count / len(matches)


def compute_btts_rate(team_id: int, before_date: str, window: int = 20) -> float:
    """Compute % of matches where both teams scored (BTTS).

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date
        window: Number of recent matches to consider

    Returns:
        Rate between 0.0 and 1.0, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 5:
        return None
    btts_count = 0
    for m in matches:
        if m["home_goals"] > 0 and m["away_goals"] > 0:
            btts_count += 1
    return btts_count / len(matches)


def compute_team_form_score(team_id: int, before_date: str, window: int = 10) -> float:
    """Compute weighted form score (recent matches weighted more).

    Points: Win=3, Draw=1, Loss=0. Weighted by recency (most recent = window, oldest = 1).

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date
        window: Number of recent matches to consider

    Returns:
        Weighted form score, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 3:
        return None
    total_weight = 0.0
    weighted_points = 0.0
    for i, m in enumerate(matches):
        weight = window - i  # most recent gets highest weight
        if m["home_id"] == team_id:
            gf, ga = m["home_goals"], m["away_goals"]
        else:
            gf, ga = m["away_goals"], m["home_goals"]
        if gf > ga:
            points = 3
        elif gf == ga:
            points = 1
        else:
            points = 0
        weighted_points += points * weight
        total_weight += weight
    return weighted_points / total_weight if total_weight > 0 else None


def compute_attack_strength(team_id: int, before_date: str, window: int = 20) -> float:
    """Compute average goals scored per match.

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date
        window: Number of recent matches to consider

    Returns:
        Average goals scored, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 5:
        return None
    goals = []
    for m in matches:
        if m["home_id"] == team_id:
            goals.append(m["home_goals"])
        else:
            goals.append(m["away_goals"])
    return float(np.mean(goals)) if goals else None


def compute_defense_strength(team_id: int, before_date: str, window: int = 20) -> float:
    """Compute average goals conceded per match.

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date
        window: Number of recent matches to consider

    Returns:
        Average goals conceded, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 5:
        return None
    conceded = []
    for m in matches:
        if m["home_id"] == team_id:
            conceded.append(m["away_goals"])
        else:
            conceded.append(m["home_goals"])
    return float(np.mean(conceded)) if conceded else None


def compute_home_advantage(team_id: int, before_date: str, window: int = 30) -> float:
    """Compute home win rate minus away win rate (positive = strong home advantage).

    Args:
        team_id: Team ID
        before_date: Only use matches strictly before this date
        window: Number of recent matches to consider

    Returns:
        Difference between home and away win rates, or NaN if insufficient data
    """
    matches = _get_team_matches_before(team_id, before_date, limit=window)
    if len(matches) < 10:
        return None

    home_wins = home_total = 0
    away_wins = away_total = 0
    for m in matches:
        is_home = m["home_id"] == team_id
        gf = m["home_goals"] if is_home else m["away_goals"]
        ga = m["away_goals"] if is_home else m["home_goals"]
        if is_home:
            home_total += 1
            if gf > ga:
                home_wins += 1
        else:
            away_total += 1
            if gf > ga:
                away_wins += 1

    home_rate = home_wins / home_total if home_total > 0 else 0.0
    away_rate = away_wins / away_total if away_total > 0 else 0.0
    return home_rate - away_rate
