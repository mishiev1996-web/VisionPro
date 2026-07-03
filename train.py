"""
train.py — Train the prediction model on real xG-enriched data from SQLite.

What changed in Stage A:
    • 23 features (was 15): rest days, current streak, table position,
      xG over/under-performance, time-aware Elo
    • Model: XGBoost wrapped in CalibratedClassifierCV (was GradientBoosting)
    • Time-aware train/test split (was random)
    • 8 seasons of data (was 3)

Run:  python train.py
"""
from __future__ import annotations

import datetime as dt
import math
import os
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import db
from models.dixon_coles import DixonColes
from models.ensemble import (
    Ensemble, _make_xgb, _make_lgbm,
    _sample_weights, _time_decay_weights,
    DC_BLEND_WEIGHT,
)


MODEL_PATH = "model.pkl"
ROLLING_WINDOW = 10
SHORT_WINDOW = 3    # For momentum detection
H2H_WINDOW = 5
FORM_WINDOW = 5
MIN_PRIOR_FOR_TRAINING = 30
MIN_LEAGUE_TRAIN_ROWS = 3000  # Increased to prevent overfitting
USE_CLASS_WEIGHTS = True  # Enable to balance under-represented draws
PER_LEAGUE_PROVES_ITSELF = True
TIME_DECAY_HALF_LIFE_DAYS = 365   # XGB/LGB sample weights: matches 1 year old weigh 0.5

# Performance caches (cleared between training runs)
_sstats_cache: Optional[Dict[str, List[float]]] = None
_elo_cache: Dict[int, Optional[float]] = {}
_match_odds_cache: Dict[int, Optional[Dict]] = {}
_sstats_stats_cache: Dict[int, Dict[str, float]] = {}


FEATURE_NAMES = [
    # Goals (rolling)
    "home_avg_goals_for", "home_avg_goals_against",
    "away_avg_goals_for", "away_avg_goals_against",
    # xG (rolling)
    "home_avg_xg_for",    "home_avg_xg_against",
    "away_avg_xg_for",    "away_avg_xg_against",
    # Win rates
    "home_win_rate",      "home_home_win_rate",
    "away_win_rate",      "away_away_win_rate",
    # Recent form & H2H
    "form_diff",          "h2h_home_wins",
    # External rating
    "elo_diff",
    # Stage A
    "home_rest_days",     "away_rest_days",
    "home_streak",        "away_streak",
    "home_table_pos",     "away_table_pos",
    "home_xg_overperform", "away_xg_overperform",
    # Stage C — trend features
    "home_goal_diff_trend", "away_goal_diff_trend",
    # Stage D — scoring patterns
    "home_clean_sheet_pct", "away_clean_sheet_pct",
    "home_btts_pct",       "away_btts_pct",
    # Stage E — Elo trend
    "elo_trend",
    # Stage B — bookmaker market-implied probabilities (NaN if no odds)
    "market_implied_h",   "market_implied_d",   "market_implied_a",
    # Stage H — momentum features
    "home_goals_momentum", "away_goals_momentum",
    "home_conceded_momentum", "away_conceded_momentum",
    # Stage I — home advantage & draw tendency
    "home_advantage_strength",
    "home_draw_tendency", "away_draw_tendency",
    # Stage J — H2H dominance
    "h2h_dominance_score",
    # Stage K — xG quality
    "home_xg_quality", "away_xg_quality",
    # Stage L — fatigue
    "home_matches_30d", "away_matches_30d",
    "home_fatigue_diff",
    # Stage M — tournament context
    "position_diff",
    "home_in_top3", "home_in_bottom3",
    "away_in_top3", "away_in_bottom3",
    # Stage N — interaction features
    "xg_attack_ratio",
    "form_momentum_product",
    "elo_position_interaction",
    "rest_fatigue_ratio",
    # Stage O — advanced interactions
    "home_attack_defense_ratio",
    "away_attack_defense_ratio",
    "form_elo_synergy",
    "scoring_conceding_balance",
    "home_league_strength",
    "away_league_strength",
    # Stage P — volatility & dominance
    "home_goals_volatility",  "away_goals_volatility",
    "home_max_goals",         "away_max_goals",
    "home_big_win_pct",       "away_big_win_pct",
    "home_heavy_loss_pct",    "away_heavy_loss_pct",
    # Stage Q — form trend (improving vs declining)
    "home_form_trend",        "away_form_trend",
    "home_xg_trend",          "away_xg_trend",
    "home_conceding_trend",   "away_conceding_trend",
    # SStats — match statistics (rolling averages)
    "home_avg_shots_on_target", "away_avg_shots_on_target",
    "home_avg_total_shots",     "away_avg_total_shots",
    "home_avg_corners",         "away_avg_corners",
    "home_avg_possession",      "away_avg_possession",
    "home_avg_xg_stat",         "away_avg_xg_stat",
    # SStats odds — market-implied features
    "home_avg_over25_odds",      "away_avg_over25_odds",
    "home_avg_asian_hcap",       "away_avg_asian_hcap",
    "home_avg_corners_line",     "away_avg_corners_line",
    # SStats events — match patterns
    "home_avg_goals_1h",         "away_avg_goals_1h",
    "home_avg_yellows",          "away_avg_yellows",
]


# ── Per-team stats from prior matches ─────────────────────────────────────────

def _compute_trend(values):
    """Linear regression slope over values — positive = improving, negative = declining."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _team_history_stats(matches: List[Dict], team_id: int) -> Dict[str, float]:
    if not matches:
        return {
            "avg_goals_for": 1.2, "avg_goals_against": 1.2,
            "avg_xg_for":   1.2, "avg_xg_against":   1.2,
            "win_rate":     0.33, "home_win_rate":  0.4, "away_win_rate":  0.25,
            "form":         5.0, "streak": 0, "xg_overperform": 0.0,
            "goal_diff_trend": 0.0, "clean_sheet_pct": 0.3, "btts_pct": 0.5,
            "goals_momentum": 0.0, "conceded_momentum": 0.0,
            "draw_tendency": 0.33, "xg_quality": 0.0,
            "goals_volatility": 0.5, "max_goals_scored": 2,
            "big_win_pct": 0.1, "heavy_loss_pct": 0.1,
            "form_trend": 0.0, "xg_trend": 0.0, "conceding_trend": 0.0,
        }

    gf, ga, xf, xa, wins = [], [], [], [], 0
    home_played = home_wins = away_played = away_wins = 0
    form_pts = 0.0
    over = 0.0
    n_over = 0
    goal_diffs = []
    clean_sheets = 0
    btts_count = 0
    scored_matches = 0
    draws = 0
    big_wins = 0
    heavy_losses = 0

    # Short-term vs long-term for momentum
    gf_short, ga_short = [], []
    gf_long, ga_long = [], []

    for i, m in enumerate(matches[:FORM_WINDOW]):
        if m["home_id"] == team_id:
            r = (m["home_goals"], m["away_goals"])
        else:
            r = (m["away_goals"], m["home_goals"])
        if r[0] is None or r[1] is None:
            continue
        weight = FORM_WINDOW - i
        if r[0] > r[1]:    form_pts += 3 * weight
        elif r[0] == r[1]: form_pts += 1 * weight
        goal_diffs.append((r[0] or 0) - (r[1] or 0))

    # Streak — walk newest-first until result changes from initial sign
    streak = 0
    for m in matches:
        if m["home_id"] == team_id:
            gs, gc = m["home_goals"], m["away_goals"]
        else:
            gs, gc = m["away_goals"], m["home_goals"]
        if gs is None or gc is None:
            break
        if gs > gc:
            if streak >= 0:  streak += 1
            else:            break
        elif gs < gc:
            if streak <= 0:  streak -= 1
            else:            break
        else:
            break

    for i, m in enumerate(matches):
        if m["home_id"] == team_id:
            home_played += 1
            hg = m["home_goals"] or 0
            ag = m["away_goals"] or 0
            hx = m["home_xg"]
            ax = m["away_xg"]
            # Estimate xG from goals if not available (ratio ~0.9xG per goal)
            if hx is None: hx = hg * 0.9
            if ax is None: ax = ag * 0.9
            gf.append(hg); ga.append(ag); xf.append(hx); xa.append(ax)
            if hg > ag: wins += 1; home_wins += 1
            elif hg == ag: draws += 1
            over += (hg - hx); n_over += 1
            scored_matches += 1
            if ag == 0: clean_sheets += 1
            if hg > 0 and ag > 0: btts_count += 1
            if hg - ag >= 2: big_wins += 1
            if ag - hg >= 2: heavy_losses += 1
        else:
            away_played += 1
            hg = m["home_goals"] or 0
            ag = m["away_goals"] or 0
            hx = m["home_xg"]
            ax = m["away_xg"]
            # Estimate xG from goals if not available (ratio ~0.9xG per goal)
            if hx is None: hx = hg * 0.9
            if ax is None: ax = ag * 0.9
            gf.append(ag); ga.append(hg); xf.append(ax); xa.append(hx)
            if ag > hg: wins += 1; away_wins += 1
            elif ag == hg: draws += 1
            over += (ag - ax); n_over += 1
            scored_matches += 1
            if hg == 0: clean_sheets += 1
            if hg > 0 and ag > 0: btts_count += 1
            if ag - hg >= 2: big_wins += 1
            if hg - ag >= 2: heavy_losses += 1

        # Momentum tracking
        if i < SHORT_WINDOW:
            gf_short.append(gf[-1])
            ga_short.append(ga[-1])
        gf_long.append(gf[-1])
        ga_long.append(ga[-1])

    total = len(matches)
    
    # Momentum: short-term avg minus long-term avg
    goals_momentum = (np.mean(gf_short) - np.mean(gf_long)) if (gf_short and gf_long) else 0.0
    conceded_momentum = (np.mean(ga_short) - np.mean(ga_long)) if (ga_short and ga_long) else 0.0
    
    # xG quality: how well team converts chances (goals vs xG)
    xg_quality = (np.mean(gf) / max(np.mean(xf), 0.1)) if xf else 1.0
    
    return {
        "avg_goals_for":   float(np.mean(gf)) if gf else 1.2,
        "avg_goals_against": float(np.mean(ga)) if ga else 1.2,
        "avg_xg_for":      float(np.mean(xf)) if xf else 1.2,
        "avg_xg_against":  float(np.mean(xa)) if xa else 1.2,
        "win_rate":        wins / total if total else 0.33,
        "home_win_rate":   home_wins / home_played if home_played else 0.4,
        "away_win_rate":   away_wins / away_played if away_played else 0.25,
        "form":            form_pts,
        "streak":          int(streak),
        "xg_overperform":  (over / n_over) if n_over else 0.0,
        "goal_diff_trend": float(np.mean(goal_diffs)) if goal_diffs else 0.0,
        "clean_sheet_pct": clean_sheets / scored_matches if scored_matches else 0.3,
        "btts_pct":        btts_count / scored_matches if scored_matches else 0.5,
        "goals_momentum":  float(goals_momentum),
        "conceded_momentum": float(conceded_momentum),
        "draw_tendency":   draws / total if total else 0.33,
        "xg_quality":      float(xg_quality),
        "goals_volatility": float(np.std(gf)) if len(gf) > 1 else 0.5,
        "max_goals_scored": max(gf) if gf else 2,
        "big_win_pct":     big_wins / total if total else 0.1,
        "heavy_loss_pct":  heavy_losses / total if total else 0.1,
        "form_trend": _compute_trend([3 if r[0]>r[1] else 1 if r[0]==r[1] else 0
                                       for r in [(m["home_goals"], m["away_goals"]) if m["home_id"]==team_id
                                                  else (m["away_goals"], m["home_goals"])
                                                  for m in matches[:5]]]),
        "xg_trend": _compute_trend(xf[:5]) if xf else 0.0,
        "conceding_trend": _compute_trend(ga[:5]) if ga else 0.0,
    }


def _rest_days(team_id: int, match_date: str, all_prior_matches: List[Dict]) -> int:
    """Days since the team's previous match. 7 if not found (~normal week)."""
    md = _parse_date(match_date)
    for m in all_prior_matches:
        if m["home_id"] == team_id or m["away_id"] == team_id:
            pd_ = _parse_date(m["date"])
            if pd_:
                return max(0, (md - pd_).days)
    return 7


def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.datetime.fromisoformat(s.replace(" ", "T")).date()
    except Exception:
        return None


def _table_position(team_id: int, league_slug: str, season: int,
                    season_matches: List[Dict]) -> int:
    """Compute team's position in league standings from matches played so far this season."""
    points: Dict[int, int] = {}
    gf: Dict[int, int] = {}
    ga: Dict[int, int] = {}
    for m in season_matches:
        if m["league_slug"] != league_slug or m["season"] != season:
            continue
        if m["home_goals"] is None or m["away_goals"] is None:
            continue
        h, a = m["home_id"], m["away_id"]
        hg, ag = m["home_goals"], m["away_goals"]
        gf[h] = gf.get(h, 0) + hg; ga[h] = ga.get(h, 0) + ag
        gf[a] = gf.get(a, 0) + ag; ga[a] = ga.get(a, 0) + hg
        if hg > ag:
            points[h] = points.get(h, 0) + 3; points.setdefault(a, 0)
        elif hg < ag:
            points[a] = points.get(a, 0) + 3; points.setdefault(h, 0)
        else:
            points[h] = points.get(h, 0) + 1
            points[a] = points.get(a, 0) + 1
    if team_id not in points:
        return 0
    sortable = [(tid, points.get(tid, 0), gf.get(tid, 0) - ga.get(tid, 0))
                for tid in points.keys()]
    sortable.sort(key=lambda x: (-x[1], -x[2]))
    for i, (tid, _, _) in enumerate(sortable, 1):
        if tid == team_id:
            return i
    return 0


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(home_id: int, away_id: int,
                   all_prior_matches: List[Dict],
                   match_date: Optional[str] = None,
                   league_slug: Optional[str] = None,
                   season: Optional[int] = None,
                   match_id: Optional[int] = None,
                   forecast: Optional[Dict[str, float]] = None) -> List[float]:
    """Compute the full feature vector. `all_prior_matches` is DESC by date."""
    home_hist = [m for m in all_prior_matches
                 if m["home_id"] == home_id or m["away_id"] == home_id][:ROLLING_WINDOW]
    away_hist = [m for m in all_prior_matches
                 if m["home_id"] == away_id or m["away_id"] == away_id][:ROLLING_WINDOW]

    h2h_hist = [m for m in all_prior_matches
                if {m["home_id"], m["away_id"]} == {home_id, away_id}][:H2H_WINDOW]
    h2h_home_wins = sum(
        1 for m in h2h_hist
        if (m["home_id"] == home_id and (m["home_goals"] or 0) > (m["away_goals"] or 0))
        or (m["away_id"] == home_id and (m["away_goals"] or 0) > (m["home_goals"] or 0))
    )

    h = _team_history_stats(home_hist, home_id)
    a = _team_history_stats(away_hist, away_id)

    # Time-aware Elo: use rating as of match_date (no leakage).
    # Uses cache to avoid repeated DB lookups.
    if match_date:
        d_iso = match_date[:10]
        elo_key_h = (home_id, d_iso)
        elo_key_a = (away_id, d_iso)
        if elo_key_h not in _elo_cache:
            _elo_cache[elo_key_h] = db.get_elo_at_date(home_id, d_iso) or db.get_team_elo(home_id) or 0.0
        if elo_key_a not in _elo_cache:
            _elo_cache[elo_key_a] = db.get_elo_at_date(away_id, d_iso) or db.get_team_elo(away_id) or 0.0
        home_elo = _elo_cache[elo_key_h]
        away_elo = _elo_cache[elo_key_a]
    else:
        home_elo = db.get_team_elo(home_id) or 0.0
        away_elo = db.get_team_elo(away_id) or 0.0
    elo_diff = (home_elo - away_elo) if (home_elo and away_elo) else 0.0

    # Elo trend: current Elo minus Elo from ~5 matches ago
    elo_trend = 0.0
    if home_hist and away_hist:
        prev_match = home_hist[min(4, len(home_hist) - 1)]
        prev_date = prev_match.get("date", "")[:10]
        if prev_date and match_date and prev_date < match_date:
            home_elo_prev = db.get_elo_at_date(home_id, prev_date) or home_elo
            away_elo_prev = db.get_elo_at_date(away_id, prev_date) or away_elo
            elo_trend = ((home_elo - home_elo_prev) - (away_elo - away_elo_prev)) if (home_elo and away_elo) else 0.0

    home_rest = _rest_days(home_id, match_date or dt.date.today().isoformat(),
                           all_prior_matches)
    away_rest = _rest_days(away_id, match_date or dt.date.today().isoformat(),
                           all_prior_matches)

    if league_slug and season:
        home_pos = _table_position(home_id, league_slug, season, all_prior_matches)
        away_pos = _table_position(away_id, league_slug, season, all_prior_matches)
    else:
        home_pos = away_pos = 0

    # Market-implied probabilities — bookmaker odds first, Understat forecast as fallback.
    implied_h = implied_d = implied_a = float("nan")
    if match_id is not None:
        if match_id not in _match_odds_cache:
            _match_odds_cache[match_id] = db.get_match_odds(match_id)
        odds_row = _match_odds_cache[match_id]
        if odds_row:
            implied_h = float(odds_row["implied_h"])
            implied_d = float(odds_row["implied_d"])
            implied_a = float(odds_row["implied_a"])
    # Fallback to Understat's own forecast if no bookmaker odds
    if (implied_h != implied_h) and forecast:   # NaN check
        try:
            implied_h = float(forecast.get("forecast_w") or 0)
            implied_d = float(forecast.get("forecast_d") or 0)
            implied_a = float(forecast.get("forecast_l") or 0)
            s = implied_h + implied_d + implied_a
            if s > 0:
                implied_h, implied_d, implied_a = implied_h/s, implied_d/s, implied_a/s
            else:
                implied_h = implied_d = implied_a = float("nan")
        except (TypeError, ValueError):
            implied_h = implied_d = implied_a = float("nan")

    # Cards lines from sstats odds (if match has sstats data)
    sstats_odds = _get_all_sstats_odds() if match_id else {}
    cards_over_line = _extract_sstats_value(sstats_odds, "Cards Over/Under", "Over")
    home_cards_line = _extract_sstats_value(sstats_odds, "Home Team Total Cards", "Over")
    away_cards_line = _extract_sstats_value(sstats_odds, "Away Team Total Cards", "Over")
    yellow_over_line = _extract_sstats_value(sstats_odds, "Yellow Over/Under", "Over")
    home_yellow_line = _extract_sstats_value(sstats_odds, "Home Team Yellow Cards", "Over")
    away_yellow_line = _extract_sstats_value(sstats_odds, "Away Team Yellow Cards", "Over")

    # Goal scorer odds from sstats
    home_top_scorer_odds = _extract_sstats_value(sstats_odds, "Anytime Goal Scorer", None)
    away_top_scorer_odds = _extract_sstats_value(sstats_odds, "Away Anytime Goal Scorer", None)

    # BTTS and exact score lines
    btts_line = _extract_sstats_value(sstats_odds, "Both Teams Score", "Yes")
    exact_score_line = _extract_sstats_value(sstats_odds, "Exact Score", None)

    # Home advantage strength: how much better team plays at home vs away
    home_advantage = h["home_win_rate"] - h["away_win_rate"]
    # Away team's vulnerability at home (lower = more vulnerable)
    away_home_vulnerability = 1.0 - a["away_win_rate"]

    # H2H dominance: weighted wins in head-to-head
    h2h_dominance = 0.0
    if h2h_hist:
        for i, m in enumerate(h2h_hist):
            weight = H2H_WINDOW - i
            if m["home_id"] == home_id:
                if (m["home_goals"] or 0) > (m["away_goals"] or 0):
                    h2h_dominance += weight
                elif (m["home_goals"] or 0) < (m["away_goals"] or 0):
                    h2h_dominance -= weight
            else:
                if (m["away_goals"] or 0) > (m["home_goals"] or 0):
                    h2h_dominance += weight
                elif (m["away_goals"] or 0) < (m["home_goals"] or 0):
                    h2h_dominance -= weight
        h2h_dominance /= H2H_WINDOW

    # Fatigue: matches in last 30 days
    match_date_dt = _parse_date(match_date or dt.date.today().isoformat())
    def _matches_in_window(team_id, window_days=30):
        if not match_date_dt:
            return 0
        count = 0
        for m in all_prior_matches:
            if m["home_id"] != team_id and m["away_id"] != team_id:
                continue
            md = _parse_date(m["date"])
            if md and 0 <= (match_date_dt - md).days <= window_days:
                count += 1
        return count

    home_matches_30d = _matches_in_window(home_id)
    away_matches_30d = _matches_in_window(away_id)
    home_fatigue_diff = home_matches_30d - away_matches_30d

    # Tournament context
    position_diff = (home_pos - away_pos) if (home_pos and away_pos) else 0
    home_in_top3 = 1.0 if (home_pos and home_pos <= 3) else 0.0
    home_in_bottom3 = 1.0 if (home_pos and home_pos >= 18) else 0.0
    away_in_top3 = 1.0 if (away_pos and away_pos <= 3) else 0.0
    away_in_bottom3 = 1.0 if (away_pos and away_pos >= 18) else 0.0

    # Interaction features
    xg_attack_ratio = (h["avg_xg_for"] / max(a["avg_xg_against"], 0.1))
    form_momentum_product = (h["form"] - a["form"]) * (h["goals_momentum"] - a["goals_momentum"])
    elo_position_interaction = elo_diff * position_diff
    rest_fatigue_ratio = (home_rest + 1) / max(away_matches_30d + 1, 1)

    # Stage O — advanced interactions
    home_attack_defense_ratio = (h["avg_xg_for"] / max(h["avg_xg_against"], 0.1))
    away_attack_defense_ratio = (a["avg_xg_for"] / max(a["avg_xg_against"], 0.1))
    form_elo_synergy = (h["form"] - a["form"]) * elo_diff / 100.0
    scoring_conceding_balance = (h["avg_goals_for"] - h["avg_goals_against"]) - (a["avg_goals_for"] - a["avg_goals_against"])
    home_league_strength = home_pos if home_pos else 10
    away_league_strength = away_pos if away_pos else 10

    # SStats — rolling match statistics (shots, corners, possession, xG from sstats)
    h_stats = _get_team_sstats_stats(home_id, match_date or dt.date.today().isoformat())
    a_stats = _get_team_sstats_stats(away_id, match_date or dt.date.today().isoformat())

    # SStats — odds-based features (totals, handicaps, corners)
    h_odds = _get_team_sstats_odds_features(home_id, match_date or dt.date.today().isoformat())
    a_odds = _get_team_sstats_odds_features(away_id, match_date or dt.date.today().isoformat())

    # SStats — event-based features (goals timing, cards)
    h_events = _get_team_sstats_event_features(home_id, match_date or dt.date.today().isoformat())
    a_events = _get_team_sstats_event_features(away_id, match_date or dt.date.today().isoformat())

    return [
        h["avg_goals_for"], h["avg_goals_against"],
        a["avg_goals_for"], a["avg_goals_against"],
        h["avg_xg_for"],    h["avg_xg_against"],
        a["avg_xg_for"],    a["avg_xg_against"],
        h["win_rate"],      h["home_win_rate"],
        a["win_rate"],      a["away_win_rate"],
        h["form"] - a["form"],
        h2h_home_wins,
        elo_diff,
        home_rest,          away_rest,
        h["streak"],        a["streak"],
        home_pos,           away_pos,
        h["xg_overperform"], a["xg_overperform"],
        h["goal_diff_trend"], a["goal_diff_trend"],
        h["clean_sheet_pct"], a["clean_sheet_pct"],
        h["btts_pct"],        a["btts_pct"],
        elo_trend,
        implied_h,           implied_d,           implied_a,
        # Stage H — momentum features
        h["goals_momentum"],   a["goals_momentum"],
        h["conceded_momentum"], a["conceded_momentum"],
        # Stage I — home advantage & draw tendency
        home_advantage,
        h["draw_tendency"],    a["draw_tendency"],
        # Stage J — H2H dominance
        h2h_dominance,
        # Stage K — xG quality
        h["xg_quality"],       a["xg_quality"],
        # Stage L — fatigue
        home_matches_30d,     away_matches_30d,
        home_fatigue_diff,
        # Stage M — tournament context
        position_diff,
        home_in_top3,         home_in_bottom3,
        away_in_top3,         away_in_bottom3,
        # Stage N — interaction features
        xg_attack_ratio,      form_momentum_product,
        elo_position_interaction, rest_fatigue_ratio,
        # Stage O — advanced interactions
        home_attack_defense_ratio, away_attack_defense_ratio,
        form_elo_synergy,     scoring_conceding_balance,
        home_league_strength, away_league_strength,
        # Stage P — volatility & dominance
        h["goals_volatility"], a["goals_volatility"],
        h["max_goals_scored"], a["max_goals_scored"],
        h["big_win_pct"],     a["big_win_pct"],
        h["heavy_loss_pct"],  a["heavy_loss_pct"],
        # Stage Q — form trend
        h["form_trend"],     a["form_trend"],
        h["xg_trend"],       a["xg_trend"],
        h["conceding_trend"], a["conceding_trend"],
        # SStats — match statistics (rolling averages)
        h_stats["shots_on_target"], a_stats["shots_on_target"],
        h_stats["total_shots"],     a_stats["total_shots"],
        h_stats["corners"],         a_stats["corners"],
        h_stats["possession"],      a_stats["possession"],
        h_stats["xg_stat"],         a_stats["xg_stat"],
        # SStats odds — market-implied features
        h_odds["avg_over25"],        a_odds["avg_over25"],
        h_odds["avg_asian_hcap"],    a_odds["avg_asian_hcap"],
        h_odds["avg_corners_line"],  a_odds["avg_corners_line"],
        # SStats events — match patterns
        h_events["avg_goals_1h"],    a_events["avg_goals_1h"],
        h_events["avg_yellows"],     a_events["avg_yellows"],
    ]


def _get_all_sstats_odds() -> Dict[str, List[float]]:
    """Fetch ALL sstats odds in a single query, grouped by (market, name).
    
    Returns dict like {"Cards Over/Under|Over": [2.1, 2.2, ...], ...}
    Much faster than calling _get_sstats_line() 11 times.
    Uses module-level cache to avoid repeated DB queries.
    """
    global _sstats_cache
    if _sstats_cache is not None:
        return _sstats_cache
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT market, name, value FROM sstats_odds "
                "WHERE value IS NOT NULL AND value > 1.0"
            ).fetchall()
        grouped: Dict[str, List[float]] = {}
        for market, name, value in rows:
            key = f"{market}|{name}" if name else market
            grouped.setdefault(key, []).append(float(value))
        _sstats_cache = grouped
        return grouped
    except Exception:
        return {}


def _clear_train_caches():
    """Clear all module-level caches between training runs."""
    global _sstats_cache
    _sstats_cache = None
    _elo_cache.clear()
    _match_odds_cache.clear()
    _sstats_stats_cache.clear()


def _extract_sstats_value(odds: Dict[str, List[float]], market: str, name: str = None) -> float:
    """Extract average odds from pre-fetched sstats data. Returns NaN if not found."""
    key = f"{market}|{name}" if name else market
    vals = odds.get(key, [])
    return sum(vals) / len(vals) if vals else float("nan")


def _get_team_sstats_stats(team_id: int, match_date: str, limit: int = 10) -> Dict[str, float]:
    """Get rolling average of sstats statistics for a team's recent matches.

    Returns dict with keys like shots_on_target, total_shots, corners, possession, xg_stat.
    Values are averages of home/away values depending on which side the team played.
    """
    cache_key = (team_id, match_date[:10], limit)
    if cache_key in _sstats_stats_cache:
        return _sstats_stats_cache[cache_key]

    defaults = {
        "shots_on_target": 3.0, "total_shots": 10.0, "corners": 4.5,
        "possession": 50.0, "xg_stat": 1.2,
    }

    try:
        with db.connect() as conn:
            # Find sstats game_ids that match this team around this date
            rows = conn.execute(
                "SELECT ss.game_id, ss.stat_name, ss.home_value, ss.away_value, sm.home_id, sm.away_id "
                "FROM sstats_statistics ss "
                "JOIN sstats_matches sm ON sm.game_id = ss.game_id "
                "WHERE (sm.home_id = ? OR sm.away_id = ?) "
                "AND sm.date < ? "
                "ORDER BY sm.date DESC LIMIT ?",
                (team_id, team_id, match_date, limit * 20),
            ).fetchall()

            if not rows:
                _sstats_stats_cache[cache_key] = defaults
                return defaults

            # Group by game_id, compute per-game stats
            from collections import defaultdict
            game_stats: Dict[int, Dict[str, float]] = defaultdict(dict)
            game_sides: Dict[int, str] = {}  # 'home' or 'away'

            for r in rows:
                gid = r[0]
                stat_name = r[1]
                home_val = r[2]
                away_val = r[3]
                home_id = r[4]
                away_id = r[5]

                if gid not in game_sides:
                    game_sides[gid] = "home" if home_id == team_id else "away"

                side = game_sides[gid]
                val = home_val if side == "home" else away_val
                try:
                    game_stats[gid][stat_name] = float(val) if val else 0.0
                except (TypeError, ValueError):
                    pass

            if not game_stats:
                _sstats_stats_cache[cache_key] = defaults
                return defaults

            # Average across games
            result = {}
            for stat, default in defaults.items():
                vals = [gs.get(stat, 0.0) for gs in game_stats.values() if stat in gs]
                result[stat] = sum(vals) / len(vals) if vals else default

            _sstats_stats_cache[cache_key] = result
            return result

    except Exception:
        _sstats_stats_cache[cache_key] = defaults
        return defaults


def _get_team_sstats_odds_features(team_id: int, match_date: str, limit: int = 10) -> Dict[str, float]:
    """Get average odds for totals, handicaps, corners from sstats_odds."""
    cache_key = ("odds", team_id, match_date[:10], limit)
    if cache_key in _sstats_stats_cache:
        return _sstats_stats_cache[cache_key]

    defaults = {"avg_over25": 2.5, "avg_asian_hcap": 0.0, "avg_corners_line": 9.5}

    try:
        with db.connect() as conn:
            # Get recent sstats games for this team
            game_ids = conn.execute(
                "SELECT sm.game_id FROM sstats_matches sm "
                "WHERE (sm.home_id = ? OR sm.away_id = ?) AND sm.date < ? "
                "ORDER BY sm.date DESC LIMIT ?",
                (team_id, team_id, match_date, limit),
            ).fetchall()

            if not game_ids:
                return defaults

            gids = [r[0] for r in game_ids]

            # Get Goals Over/Under odds
            over25_vals = conn.execute(
                "SELECT so.value FROM sstats_odds so "
                "WHERE so.game_id IN ({}) AND so.market = 'Goals Over/Under' "
                "AND so.name = 'Over' AND so.value > 1.0".format(",".join("?" * len(gids))),
                gids,
            ).fetchall()

            # Get Asian Handicap odds
            ah_vals = conn.execute(
                "SELECT so.value FROM sstats_odds so "
                "WHERE so.game_id IN ({}) AND so.market = 'Asian Handicap' "
                "AND so.name LIKE 'Home%' AND so.value > 1.0".format(",".join("?" * len(gids))),
                gids,
            ).fetchall()

            # Get Corners Over/Under odds
            corners_vals = conn.execute(
                "SELECT so.value FROM sstats_odds so "
                "WHERE so.game_id IN ({}) AND so.market = 'Corners Over Under' "
                "AND so.name = 'Over' AND so.value > 1.0".format(",".join("?" * len(gids))),
                gids,
            ).fetchall()

            result = {
                "avg_over25": sum(r[0] for r in over25_vals) / len(over25_vals) if over25_vals else defaults["avg_over25"],
                "avg_asian_hcap": sum(r[0] for r in ah_vals) / len(ah_vals) if ah_vals else defaults["avg_asian_hcap"],
                "avg_corners_line": sum(r[0] for r in corners_vals) / len(corners_vals) if corners_vals else defaults["avg_corners_line"],
            }

            _sstats_stats_cache[cache_key] = result
            return result

    except Exception:
        return defaults


def _get_team_sstats_event_features(team_id: int, match_date: str, limit: int = 10) -> Dict[str, float]:
    """Get average goals in 1st half and yellow cards from sstats_events."""
    cache_key = ("events", team_id, match_date[:10], limit)
    if cache_key in _sstats_stats_cache:
        return _sstats_stats_cache[cache_key]

    defaults = {"avg_goals_1h": 1.0, "avg_yellows": 2.0}

    try:
        with db.connect() as conn:
            # Get recent sstats games for this team
            game_ids = conn.execute(
                "SELECT sm.game_id FROM sstats_matches sm "
                "WHERE (sm.home_id = ? OR sm.away_id = ?) AND sm.date < ? "
                "ORDER BY sm.date DESC LIMIT ?",
                (team_id, team_id, match_date, limit),
            ).fetchall()

            if not game_ids:
                return defaults

            gids = [r[0] for r in game_ids]

            # Count goals in first half (minute < 45)
            goals_1h = conn.execute(
                "SELECT COUNT(*) FROM sstats_events se "
                "WHERE se.game_id IN ({}) AND se.event_type = '1' "
                "AND se.minute < 45".format(",".join("?" * len(gids))),
                gids,
            ).fetchone()[0]

            # Count yellow cards
            yellows = conn.execute(
                "SELECT COUNT(*) FROM sstats_events se "
                "WHERE se.game_id IN ({}) AND se.event_type = '3'".format(",".join("?" * len(gids))),
                gids,
            ).fetchone()[0]

            n_games = len(gids)
            result = {
                "avg_goals_1h": goals_1h / n_games if n_games else defaults["avg_goals_1h"],
                "avg_yellows": yellows / n_games if n_games else defaults["avg_yellows"],
            }

            _sstats_stats_cache[cache_key] = result
            return result

    except Exception:
        return defaults


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Returns X, y, dates, league_slugs, and meta (home/away ids+names) for DC training.
    
    Trains on ALL leagues in the database. xG features are NaN for leagues
    without Understat data — the model handles this via median imputation.
    """
    _clear_train_caches()

    all_matches = db.all_matches_for_training()
    print(f"  Loaded {len(all_matches)} finished matches from all leagues in DB")

    # Resolve team names once
    name_cache: Dict[int, str] = {}
    def name_of(tid: int) -> str:
        if tid not in name_cache:
            t = db.get_team(tid)
            name_cache[tid] = t["name"] if t else f"team_{tid}"
        return name_cache[tid]

    X_rows, y_rows, date_rows, league_rows = [], [], [], []
    meta_rows = []
    past_desc: List[Dict] = []

    for m in all_matches:
        prior_same_league = [p for p in past_desc if p["league_slug"] == m["league_slug"]]
        if len(prior_same_league) < MIN_PRIOR_FOR_TRAINING:
            past_desc.insert(0, m); continue

        features = build_features(
            m["home_id"], m["away_id"],
            prior_same_league,
            match_date=m["date"],
            league_slug=m["league_slug"],
            season=m["season"],
            match_id=m.get("id"),
            forecast={
                "forecast_w": m.get("forecast_w"),
                "forecast_d": m.get("forecast_d"),
                "forecast_l": m.get("forecast_l"),
            },
        )

        hg, ag = m["home_goals"] or 0, m["away_goals"] or 0
        if hg > ag:     result = 2
        elif hg == ag:  result = 1
        else:           result = 0

        X_rows.append(features)
        y_rows.append(result)
        date_rows.append(m["date"])
        league_rows.append(m["league_slug"])
        meta_rows.append({
            "match_id":   m.get("id"),
            "home_id":    m["home_id"],
            "away_id":    m["away_id"],
            "home_name":  name_of(m["home_id"]),
            "away_name":  name_of(m["away_id"]),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        })
        past_desc.insert(0, m)

    X = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    y = pd.Series(y_rows, name="result")
    dates = pd.Series(date_rows, name="date")
    leagues = pd.Series(league_rows, name="league_slug")
    meta = pd.DataFrame(meta_rows)

    # Fill NaN with column medians for robustness
    for col in X.columns:
        X[col] = X[col].astype(float)
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)

    return X, y, dates, leagues, meta



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(">>> Building feature dataset...")
    X, y, dates, leagues, meta = build_dataset()
    if len(X) < 100:
        raise SystemExit(
            f"Only {len(X)} usable training rows. "
            "Run 'python data_collector.py' first to populate the DB."
        )
    print(f"  Dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"  Class distribution: away={sum(y==0)} draw={sum(y==1)} home={sum(y==2)}")

    # Time-aware split
    order = dates.argsort().values
    X       = X.iloc[order].reset_index(drop=True)
    y       = y.iloc[order].reset_index(drop=True)
    dates   = dates.iloc[order].reset_index(drop=True)
    leagues = leagues.iloc[order].reset_index(drop=True)
    meta    = meta.iloc[order].reset_index(drop=True)

    # ── League normalization: z-score xG and goals within each league ─────────
    norm_cols = [
        "home_avg_goals_for", "home_avg_goals_against",
        "away_avg_goals_for", "away_avg_goals_against",
        "home_avg_xg_for", "home_avg_xg_against",
        "away_avg_xg_for", "away_avg_xg_against",
        "home_goals_volatility", "away_goals_volatility",
        "home_max_goals", "away_max_goals",
        "home_league_strength", "away_league_strength",
    ]
    print("  Normalizing features by league (z-score)...")
    for slug in leagues.unique():
        mask = leagues == slug
        if mask.sum() < 50:
            continue
        for col in norm_cols:
            if col not in X.columns:
                continue
            X[col] = X[col].astype(float)
            vals = X.loc[mask, col]
            mean_val = vals.mean()
            std_val = vals.std()
            if std_val > 0.01:
                X.loc[mask, col] = (vals - mean_val) / std_val

    # ── Expanding window cross-validation ────────────────────────────────────
    N_FOLDS = 5
    fold_size = len(X) // (N_FOLDS + 1)
    print(f"  Expanding window CV: {N_FOLDS} folds, {len(X)} rows total")

    all_fold_accs = []
    all_fold_lls = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, len(X))

        if test_end <= test_start:
            break

        X_tr, X_te = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_tr, y_te = y.iloc[:train_end], y.iloc[test_start:test_end]
        lg_tr, lg_te = leagues.iloc[:train_end], leagues.iloc[test_start:test_end]
        dt_tr = dates.iloc[:train_end]
        mt_tr, mt_te = meta.iloc[:train_end], meta.iloc[test_start:test_end]

        print(f"\n  Fold {fold+1}/{N_FOLDS}: train={len(X_tr)} test={len(X_te)} "
              f"(up to {str(dt_tr.iloc[-1])[:10]})")

        fold_ens = Ensemble()
        fold_ens.fit(X_tr, y_tr, lg_tr, dt_tr, mt_tr)

        probas_f = np.zeros((len(X_te), 3))
        for i in range(len(X_te)):
            row = mt_te.iloc[i]
            probas_f[i] = fold_ens.predict_proba(
                X_te.iloc[[i]], league_slug=lg_te.iloc[i],
                home_name=row["home_name"], away_name=row["away_name"],
            )[0]

        preds_f = np.argmax(probas_f, axis=1)
        acc_f = accuracy_score(y_te, preds_f)
        ll_f = log_loss(y_te, probas_f, labels=[0, 1, 2])
        all_fold_accs.append(acc_f)
        all_fold_lls.append(ll_f)
        print(f"    Acc: {acc_f:.2%}  Log-loss: {ll_f:.3f}")

    print(f"\n=== CV Results ({N_FOLDS} folds) ===")
    print(f"  Mean accuracy:  {np.mean(all_fold_accs):.2%} ± {np.std(all_fold_accs):.2%}")
    print(f"  Mean log-loss:  {np.mean(all_fold_lls):.3f} ± {np.std(all_fold_lls):.3f}")

    # ── Final model on all data (except last 20% for held-out eval) ──────────
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    leagues_train, leagues_test = leagues.iloc[:split], leagues.iloc[split:]
    dates_train,   dates_test   = dates.iloc[:split],   dates.iloc[split:]
    meta_train,    meta_test    = meta.iloc[:split],    meta.iloc[split:]
    print(f"\n  Final train: {len(X_train)} rows (up to {dates.iloc[split-1]})")
    print(f"  Final test:  {len(X_test)} rows (after {dates.iloc[split]})")

    print(">>> Training ensemble (XGB + LightGBM + Dixon-Coles, time-decayed)...")
    ens = Ensemble()
    ens.fit(X_train, y_train, leagues_train, dates_train, meta_train)

    print("\n>>> Evaluating on held-out future matches...")
    probas = np.zeros((len(X_test), 3))
    for i in range(len(X_test)):
        slug = leagues_test.iloc[i]
        row  = meta_test.iloc[i]
        p = ens.predict_proba(X_test.iloc[[i]], league_slug=slug,
                              home_name=row["home_name"], away_name=row["away_name"])
        probas[i] = p[0]
    preds = np.argmax(probas, axis=1)

    acc = accuracy_score(y_test, preds)
    ll  = log_loss(y_test, probas, labels=[0, 1, 2])

    print(f"\n=== Test set (held-out future matches) ===")
    print(f"Accuracy: {acc:.2%}    Log-loss: {ll:.3f}")
    print(classification_report(y_test, preds,
                                target_names=["Away Win", "Draw", "Home Win"]))

    # Per-league breakdown
    print("Per-league accuracy:")
    for slug in sorted(set(leagues_test)):
        mask = (leagues_test.values == slug)
        if mask.sum() < 30:
            continue
        l_acc = accuracy_score(y_test[mask], preds[mask])
        sub = "league-specific" if slug in ens.leagues else "global fallback"
        print(f"   {slug:14s}  n={int(mask.sum()):4d}  acc={l_acc:6.2%}  [{sub}]")

    # Feature importances from the global XGB
    try:
        base_xgb = ens.global_model["xgb"].calibrated_classifiers_[0].estimator
        importances = sorted(zip(FEATURE_NAMES, base_xgb.feature_importances_),
                             key=lambda x: -x[1])
        print("\nTop global features (XGB):")
        for name, imp in importances[:10]:
            print(f"   {name:30s} {imp:.3f}")
    except Exception as e:
        print(f"(could not extract feature importances: {e})")

    # ── ROI simulation: does our model beat the bookmaker? ────────────────────
    print("\n>>> ROI simulation against bookmaker odds (held-out test set)...")
    _simulate_roi(probas, y_test.values, meta_test)

    joblib.dump({"ensemble": ens, "features": FEATURE_NAMES, "format": "ensemble_v3"},
                MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def _simulate_roi(probas: np.ndarray, y_true: np.ndarray, meta: pd.DataFrame) -> None:
    """For each test match where we have bookmaker odds, place a flat-stake bet
    on the outcome whose model probability exceeds the bookmaker implied prob
    by EDGE_THRESHOLD. Track cumulative ROI.

    Only uses odds from historical sources (pinnacle_close, bet365, market_avg)
    to avoid circular dependency with sstats consensus odds used as features.
    """
    EDGE_THRESHOLDS = [0.02, 0.05, 0.10]
    OUTCOMES = ["away", "draw", "home"]
    odds_cache: Dict[int, Dict] = {}

    # Use a dedicated query that only returns historical odds (not sstats_consensus)
    with db.connect() as conn:
        for i in range(len(meta)):
            mid = int(meta.iloc[i]["match_id"]) if meta.iloc[i]["match_id"] is not None else None
            if mid is None: continue
            row = conn.execute(
                "SELECT * FROM match_odds WHERE match_id=? AND source != 'sstats_consensus' LIMIT 1",
                (mid,),
            ).fetchone()
            if row:
                odds_cache[i] = dict(row)

    print(f"  Matches with bookmaker odds: {len(odds_cache)}/{len(meta)}")
    if not odds_cache: return

    for thr in EDGE_THRESHOLDS:
        stake_total = 0.0; pnl = 0.0; bets = 0; wins = 0
        for i, o in odds_cache.items():
            # Our predicted probabilities — class order [away, draw, home]
            my = probas[i]
            # Bookmaker implied probabilities (already normalized in DB)
            bk = np.array([o["implied_a"], o["implied_d"], o["implied_h"]])
            edge = my - bk
            best = int(np.argmax(edge))
            if edge[best] < thr: continue   # no value
            stake = 1.0
            # Decimal odds for the picked outcome
            odds_dec = {"away": o["away_odds"], "draw": o["draw_odds"],
                        "home": o["home_odds"]}[OUTCOMES[best]]
            bets += 1; stake_total += stake
            if int(y_true[i]) == best:
                pnl += stake * (odds_dec - 1.0)  # net winnings
                wins += 1
            else:
                pnl -= stake                     # lose stake
        if bets == 0:
            print(f"  edge ≥ {thr*100:>4.0f}%: no value bets found"); continue
        roi = pnl / stake_total * 100
        wr  = wins / bets * 100
        verdict = "PROFITABLE" if roi > 0 else "loses to market"
        print(f"  edge >= {int(thr*100):>2d}%:  bets={bets:>4d}  win_rate={wr:>5.1f}%  "
              f"ROI={roi:+6.2f}%  ({verdict})")


if __name__ == "__main__":
    main()
