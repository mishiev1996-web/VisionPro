"""
train.py — Train the prediction model on real data from SQLite.
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
import config
import features_pointintime as fpit
from models.dixon_coles import DixonColes
from models.ensemble import (
    Ensemble, _make_xgb, _make_lgbm,
    _sample_weights, _time_decay_weights,
    DC_BLEND_WEIGHT,
)


MODEL_PATH = "model.pkl"
ROLLING_WINDOW = 10
SHORT_WINDOW = 3
H2H_WINDOW = 5
FORM_WINDOW = 5
MIN_PRIOR_FOR_TRAINING = 30
MIN_LEAGUE_TRAIN_ROWS = 3000
USE_CLASS_WEIGHTS = True
TIME_DECAY_HALF_LIFE_DAYS = 365

# Performance caches
_elo_cache: Dict[int, Optional[float]] = {}
_match_odds_cache: Dict[int, Optional[Dict]] = {}
_sstats_stats_cache: Dict[int, Dict[str, float]] = {}


FEATURE_NAMES = [
    # Goals (rolling)
    "home_avg_goals_for", "home_avg_goals_against",
    "away_avg_goals_for", "away_avg_goals_against",
    # Win rates
    "home_win_rate",      "home_home_win_rate",
    "away_win_rate",      "away_away_win_rate",
    # Recent form & H2H
    "form_diff",          "h2h_home_dominance",  "h2h_avg_goals",
    # External rating
    "elo_diff",
    # League position
    "home_table_pos",     "away_table_pos",      "position_diff",
    # Rest days
    "home_rest_days",     "away_rest_days",
    # Streaks
    "home_streak",        "away_streak",
    # SStats: xG (individual + diff)
    "home_xg",            "away_xg",             "xg_diff",
    # SStats: possession, corners, fouls, yellow_cards, shots_on_target, big_chances
    "home_possession",    "away_possession",     "possession_diff",
    "home_corners",       "away_corners",        "corners_diff",
    "home_fouls",         "away_fouls",          "fouls_diff",
    "home_yellow_cards",  "away_yellow_cards",   "yellow_cards_diff",
    "home_shots_on_target", "away_shots_on_target", "shots_on_target_diff",
    "home_big_chances",   "away_big_chances",    "big_chances_diff",
    # SStats: cards & substitutions per game
    "home_cards_per_game", "away_cards_per_game", "cards_per_game_diff",
    "home_subs_per_game", "away_subs_per_game",  "subs_per_game_diff",
    # Injuries
    "home_injuries",      "away_injuries",
    # Point-in-time (from match history)
    "home_over25_rate",   "away_over25_rate",   "over25_rate_diff",
    "home_btts_rate",     "away_btts_rate",     "btts_rate_diff",
    "home_form_score",    "away_form_score",    "form_score_diff",
    "home_attack_str",    "away_attack_str",    "attack_str_diff",
    "home_defense_str",   "away_defense_str",   "defense_str_diff",
    "home_home_advantage","away_home_advantage", "home_advantage_diff",
    # Market odds + uncertainty
    "market_implied_h",   "market_implied_d",   "market_implied_a",
    "market_entropy",
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
            "win_rate": 0.33, "home_win_rate": 0.4, "away_win_rate": 0.25,
            "form": 5.0, "streak": 0,
        }

    gf, ga, wins = [], [], 0
    home_played = home_wins = away_played = away_wins = 0
    form_pts = 0.0
    n = 0

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

    for m in matches:
        if m["home_id"] == team_id:
            home_played += 1
            hg = m["home_goals"] or 0
            ag = m["away_goals"] or 0
            gf.append(hg); ga.append(ag)
            if hg > ag: wins += 1; home_wins += 1
        else:
            away_played += 1
            hg = m["home_goals"] or 0
            ag = m["away_goals"] or 0
            gf.append(ag); ga.append(hg)
            if ag > hg: wins += 1; away_wins += 1

    total = len(matches)
    
    return {
        "avg_goals_for":   float(np.mean(gf)) if gf else 1.2,
        "avg_goals_against": float(np.mean(ga)) if ga else 1.2,
        "win_rate":        wins / total if total else 0.33,
        "home_win_rate":   home_wins / home_played if home_played else 0.4,
        "away_win_rate":   away_wins / away_played if away_played else 0.25,
        "form":            form_pts,
        "streak":          int(streak),
    }


def _rest_days(team_id: int, match_date: str, all_prior_matches: List[Dict]) -> Optional[int]:
    """Days since the team's previous match."""
    md = _parse_date(match_date)
    for m in all_prior_matches:
        if m["home_id"] == team_id or m["away_id"] == team_id:
            pd_ = _parse_date(m["date"])
            if pd_:
                return max(0, (md - pd_).days)
    return None


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
    h2h_total = len(h2h_hist)
    h2h_home_wins = sum(
        1 for m in h2h_hist
        if (m["home_id"] == home_id and (m["home_goals"] or 0) > (m["away_goals"] or 0))
        or (m["away_id"] == home_id and (m["away_goals"] or 0) > (m["home_goals"] or 0))
    )
    h2h_home_dominance = h2h_home_wins / h2h_total if h2h_total > 0 else 0.5
    h2h_goals_list = [(m["home_goals"] or 0) + (m["away_goals"] or 0) for m in h2h_hist]
    h2h_avg_goals = float(np.mean(h2h_goals_list)) if h2h_goals_list else 2.5

    h = _team_history_stats(home_hist, home_id)
    a = _team_history_stats(away_hist, away_id)

    # Time-aware Elo
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

    home_rest = _rest_days(home_id, match_date or dt.date.today().isoformat(), all_prior_matches)
    away_rest = _rest_days(away_id, match_date or dt.date.today().isoformat(), all_prior_matches)
    home_rest_val = float(home_rest) if home_rest is not None else None
    away_rest_val = float(away_rest) if away_rest is not None else None

    # League table position
    home_pos = _table_position(home_id, league_slug or "", season or 0,
                               [m for m in all_prior_matches
                                if m.get("league_slug") == league_slug and m.get("season") == season])
    away_pos = _table_position(away_id, league_slug or "", season or 0,
                               [m for m in all_prior_matches
                                if m.get("league_slug") == league_slug and m.get("season") == season])
    position_diff = float(home_pos - away_pos) if home_pos and away_pos else 0.0

    # Market-implied probabilities
    implied_h = implied_d = implied_a = None
    if match_id is not None:
        if match_id not in _match_odds_cache:
            _match_odds_cache[match_id] = db.get_match_odds(match_id)
        odds_row = _match_odds_cache[match_id]
        if odds_row:
            implied_h = float(odds_row["implied_h"])
            implied_d = float(odds_row["implied_d"])
            implied_a = float(odds_row["implied_a"])
    if (implied_h != implied_h) and forecast:
        try:
            implied_h = float(forecast.get("forecast_w") or 0)
            implied_d = float(forecast.get("forecast_d") or 0)
            implied_a = float(forecast.get("forecast_l") or 0)
            s = implied_h + implied_d + implied_a
            if s > 0:
                implied_h, implied_d, implied_a = implied_h/s, implied_d/s, implied_a/s
            else:
                implied_h = implied_d = implied_a = None
        except (TypeError, ValueError):
            implied_h = implied_d = implied_a = None
    # Fallback: league-average odds if still NaN
    if implied_h != implied_h and league_slug and league_slug in _league_avg_odds:
        implied_h, implied_d, implied_a = _league_avg_odds[league_slug]

    # Market entropy (uncertainty)
    _eps = 1e-10
    if implied_h is not None and implied_d is not None and implied_a is not None:
        p_vals = [max(implied_h, _eps), max(implied_d, _eps), max(implied_a, _eps)]
        s_p = sum(p_vals)
        p_norm = [p / s_p for p in p_vals]
        market_entropy = -sum(p * math.log(p) for p in p_norm if p > 0)
    else:
        market_entropy = None

    # SStats: xG, shots on target, goals by half + extended statistics
    # Early exit: skip sstats lookups for teams without any sstats data
    md10 = match_date[:10] if match_date else ""
    _home_has_sstats = home_id in _team_sstats_games
    _away_has_sstats = away_id in _team_sstats_games
    home_sstats = _get_team_sstats_stats(home_id, md10) if _home_has_sstats else {"shots_on_target": None, "total_shots": None, "xg_stat": None, "possession": None, "corners": None, "fouls": None, "yellow_cards": None, "red_cards": None, "offsides": None, "passes_accurate": None, "big_chances": None}
    away_sstats = _get_team_sstats_stats(away_id, md10) if _away_has_sstats else {"shots_on_target": None, "total_shots": None, "xg_stat": None, "possession": None, "corners": None, "fouls": None, "yellow_cards": None, "red_cards": None, "offsides": None, "passes_accurate": None, "big_chances": None}
    home_events = _get_team_sstats_event_features(home_id, md10) if _home_has_sstats else {"cards_per_game": None, "subs_per_game": None}
    away_events = _get_team_sstats_event_features(away_id, md10) if _away_has_sstats else {"cards_per_game": None, "subs_per_game": None}

    # Weather (dead code — table has 0 rows, feature not in FEATURE_NAMES)
    # weather = _get_match_weather(match_id) if match_id else {"temp_c": 15.0, "rain_mm": 0.0, "wind_ms": 2.0}

    # Injuries
    home_inj = _get_team_injury_count(home_id, match_date[:10] if match_date else "")
    away_inj = _get_team_injury_count(away_id, match_date[:10] if match_date else "")

    _h_poss = home_sstats["possession"]
    _a_poss = away_sstats["possession"]
    _h_corr = home_sstats["corners"]
    _a_corr = away_sstats["corners"]
    _h_foul = home_sstats["fouls"]
    _a_foul = away_sstats["fouls"]
    _h_yc = home_sstats["yellow_cards"]
    _a_yc = away_sstats["yellow_cards"]
    _h_sot = home_sstats["shots_on_target"]
    _a_sot = away_sstats["shots_on_target"]
    _h_bc = home_sstats["big_chances"]
    _a_bc = away_sstats["big_chances"]

    # Point-in-time features: computed from match history (NaN if <5 matches)
    _h_o25 = fpit.compute_over25_rate(home_id, match_date[:10] if match_date else "")
    _a_o25 = fpit.compute_over25_rate(away_id, match_date[:10] if match_date else "")
    _h_btts = fpit.compute_btts_rate(home_id, match_date[:10] if match_date else "")
    _a_btts = fpit.compute_btts_rate(away_id, match_date[:10] if match_date else "")
    _h_form = fpit.compute_team_form_score(home_id, match_date[:10] if match_date else "")
    _a_form = fpit.compute_team_form_score(away_id, match_date[:10] if match_date else "")
    _h_atk = fpit.compute_attack_strength(home_id, match_date[:10] if match_date else "")
    _a_atk = fpit.compute_attack_strength(away_id, match_date[:10] if match_date else "")
    _h_def = fpit.compute_defense_strength(home_id, match_date[:10] if match_date else "")
    _a_def = fpit.compute_defense_strength(away_id, match_date[:10] if match_date else "")
    _h_ha = fpit.compute_home_advantage(home_id, match_date[:10] if match_date else "")
    _a_ha = fpit.compute_home_advantage(away_id, match_date[:10] if match_date else "")

    return [
        h["avg_goals_for"], h["avg_goals_against"],
        a["avg_goals_for"], a["avg_goals_against"],
        h["win_rate"],      h["home_win_rate"],
        a["win_rate"],      a["away_win_rate"],
        h["form"] - a["form"],
        h2h_home_dominance, h2h_avg_goals,
        elo_diff,
        float(home_pos),    float(away_pos),    position_diff,
        home_rest_val,      away_rest_val,
        h["streak"],        a["streak"],
        home_sstats["xg_stat"], away_sstats["xg_stat"],
        (home_sstats["xg_stat"] - away_sstats["xg_stat"]) if home_sstats["xg_stat"] is not None and away_sstats["xg_stat"] is not None else None,
        # sstats_statistics: possession, corners, fouls, yellow_cards, shots_on_target, big_chances
        home_sstats["possession"], away_sstats["possession"],
        (home_sstats["possession"] - away_sstats["possession"]) if home_sstats["possession"] is not None and away_sstats["possession"] is not None else None,
        home_sstats["corners"], away_sstats["corners"],
        (home_sstats["corners"] - away_sstats["corners"]) if home_sstats["corners"] is not None and away_sstats["corners"] is not None else None,
        home_sstats["fouls"], away_sstats["fouls"],
        (home_sstats["fouls"] - away_sstats["fouls"]) if home_sstats["fouls"] is not None and away_sstats["fouls"] is not None else None,
        home_sstats["yellow_cards"], away_sstats["yellow_cards"],
        (home_sstats["yellow_cards"] - away_sstats["yellow_cards"]) if home_sstats["yellow_cards"] is not None and away_sstats["yellow_cards"] is not None else None,
        home_sstats["shots_on_target"], away_sstats["shots_on_target"],
        (home_sstats["shots_on_target"] - away_sstats["shots_on_target"]) if home_sstats["shots_on_target"] is not None and away_sstats["shots_on_target"] is not None else None,
        home_sstats["big_chances"], away_sstats["big_chances"],
        (home_sstats["big_chances"] - away_sstats["big_chances"]) if home_sstats["big_chances"] is not None and away_sstats["big_chances"] is not None else None,
        # sstats_events: cards_per_game, subs_per_game
        home_events["cards_per_game"], away_events["cards_per_game"],
        (home_events["cards_per_game"] - away_events["cards_per_game"]) if home_events["cards_per_game"] is not None and away_events["cards_per_game"] is not None else None,
        home_events["subs_per_game"], away_events["subs_per_game"],
        (home_events["subs_per_game"] - away_events["subs_per_game"]) if home_events["subs_per_game"] is not None and away_events["subs_per_game"] is not None else None,
        float(home_inj),    float(away_inj),
        # Point-in-time features
        _h_o25,             _a_o25,
        (_h_o25 - _a_o25) if _h_o25 is not None and _a_o25 is not None else None,
        _h_btts,            _a_btts,
        (_h_btts - _a_btts) if _h_btts is not None and _a_btts is not None else None,
        _h_form,            _a_form,
        (_h_form - _a_form) if _h_form is not None and _a_form is not None else None,
        _h_atk,             _a_atk,
        (_h_atk - _a_atk) if _h_atk is not None and _a_atk is not None else None,
        _h_def,             _a_def,
        (_h_def - _a_def) if _h_def is not None and _a_def is not None else None,
        _h_ha,              _a_ha,
        (_h_ha - _a_ha) if _h_ha is not None and _a_ha is not None else None,
        # Market odds + uncertainty
        implied_h,           implied_d,           implied_a,
        market_entropy,
    ]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Returns X, y, dates, league_slugs, and meta (home/away ids+names) for DC training."""
    _clear_train_caches()

    all_matches = db.all_matches_for_training()
    print(f"  Loaded {len(all_matches)} finished matches from all leagues in DB")

    # Filter to last 3 seasons for best balance of data volume vs recency
    current_year = dt.date.today().year
    min_year = current_year - 3
    filtered = []
    for m in all_matches:
        date_str = m.get("date", "")
        if date_str:
            year = int(date_str[:4]) if len(date_str) >= 4 else 0
            if year >= min_year:
                filtered.append(m)
    
    if len(filtered) < len(all_matches):
        all_matches = filtered
        print(f"  Filtered to last 3 seasons: {len(all_matches)} matches")

    # Pre-load supplementary data for O(1) lookup
    _preload_training_data(all_matches)

    # Resolve team names once
    name_cache: Dict[int, str] = {}
    def name_of(tid: int) -> str:
        if tid not in name_cache:
            t = db.get_team(tid)
            name_cache[tid] = t["name"] if t else f"team_{tid}"
        return name_cache[tid]

    X_rows, y_rows, date_rows, league_rows = [], [], [], []
    meta_rows = []

    # O(N) incremental indexes instead of O(N²) filtering
    from collections import defaultdict, deque
    _by_league: Dict[str, List[Dict]] = defaultdict(list)
    _by_team: Dict[int, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    _h2h: Dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    total_matches = len(all_matches)
    last_pct = -1

    for i, m in enumerate(all_matches):
        pct = int((i + 1) / total_matches * 100)
        if pct != last_pct:
            filled = pct // 5
            bar = "=" * filled + ">" + " " * (20 - filled)
            print(f"\r  [{bar}] {pct}% ({i+1}/{total_matches})", end="", flush=True)
            last_pct = pct

        lg = m["league_slug"]
        prior_same_league = _by_league[lg]
        if len(prior_same_league) < MIN_PRIOR_FOR_TRAINING:
            _by_league[lg].append(m)
            _by_team[m["home_id"]].appendleft(m)
            _by_team[m["away_id"]].appendleft(m)
            _h2h[frozenset((m["home_id"], m["away_id"]))].appendleft(m)
            continue

        # Pass pre-filtered lists to build_features (already O(window), not O(N))
        features = build_features(
            m["home_id"], m["away_id"],
            prior_same_league,
            match_date=m["date"],
            league_slug=lg,
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
        league_rows.append(lg)
        meta_rows.append({
            "match_id":   m.get("id"),
            "home_id":    m["home_id"],
            "away_id":    m["away_id"],
            "home_name":  name_of(m["home_id"]),
            "away_name":  name_of(m["away_id"]),
            "home_goals": hg,
            "away_goals": ag,
        })

        # Incremental index updates — O(1) amortized
        _by_league[lg].append(m)
        _by_team[m["home_id"]].appendleft(m)
        _by_team[m["away_id"]].appendleft(m)
        _h2h[frozenset((m["home_id"], m["away_id"]))].appendleft(m)

    X = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    y = pd.Series(y_rows, name="result")
    dates = pd.Series(date_rows, name="date")
    leagues = pd.Series(league_rows, name="league_slug")
    meta = pd.DataFrame(meta_rows)

    # Fill NaN with column medians
    for col in X.columns:
        X[col] = X[col].astype(float)
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)

    print()
    return X, y, dates, leagues, meta


def _clear_train_caches():
    _elo_cache.clear()
    _match_odds_cache.clear()
    _sstats_stats_cache.clear()
    _sstats_events_by_game.clear()
    _sstats_stats_by_game.clear()
    _weather_cache.clear()
    _injuries_cache.clear()
    _team_sstats_games.clear()
    _league_avg_odds.clear()


# ── Bulk pre-loaded caches for sstats/weather/injuries ──────────────────────
_sstats_events_by_game: Dict[int, List[Dict]] = {}
_sstats_stats_by_game: Dict[int, List[Dict]] = {}
_weather_cache: Dict[int, Dict[str, float]] = {}
_injuries_cache: Dict[int, List[Dict]] = {}
_team_sstats_games: Dict[int, List[Tuple[int, str]]] = {}  # team_id → [(game_id, date)]
_league_avg_odds: Dict[str, Tuple[float, float, float]] = {}  # league → (avg_h, avg_d, avg_a)


def _preload_training_data(all_matches: List[Dict]) -> None:
    """Bulk-load sstats, weather, injuries into dicts for O(1) lookup per match."""
    print("  Pre-loading supplementary data (sstats/weather/injuries)...")

    # ── 1. Weather: all at once ──
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT match_id, temp_c, rain_mm, wind_ms FROM weather"
            ).fetchall()
        for r in rows:
            _weather_cache[r["match_id"]] = {
                "temp_c": float(r["temp_c"]) if r["temp_c"] is not None else 15.0,
                "rain_mm": float(r["rain_mm"]) if r["rain_mm"] is not None else 0.0,
                "wind_ms": float(r["wind_ms"]) if r["wind_ms"] is not None else 2.0,
            }
        print(f"    weather: {len(_weather_cache)} matches loaded")
    except Exception as e:
        print(f"    weather: load failed ({e})")

    # ── 2. Injuries: all at once, indexed by team_id ──
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT team_id, player_name, since, until FROM injuries"
            ).fetchall()
        for r in rows:
            tid = r["team_id"]
            _injuries_cache.setdefault(tid, []).append({
                "player": r["player_name"],
                "since": r["since"] or "",
                "until": r["until"] or "2099-12-31",
            })
        print(f"    injuries: {sum(len(v) for v in _injuries_cache.values())} records for {len(_injuries_cache)} teams")
    except Exception as e:
        print(f"    injuries: load failed ({e})")

    # ── 3. SStats matches → team_id → [(game_id, date, is_home)] mapping ──
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT game_id, home_id, away_id, date FROM sstats_matches"
            ).fetchall()
        for r in rows:
            gid = r["game_id"]
            date_str = (r["date"] or "")[:10]
            if r["home_id"]:
                _team_sstats_games.setdefault(r["home_id"], []).append((gid, date_str, True))
            if r["away_id"]:
                _team_sstats_games.setdefault(r["away_id"], []).append((gid, date_str, False))
        # Sort each team's games by date descending (most recent first)
        for tid in _team_sstats_games:
            _team_sstats_games[tid].sort(key=lambda x: x[1], reverse=True)
        print(f"    sstats matches: {len(rows)} games for {len(_team_sstats_games)} teams")
    except Exception as e:
        print(f"    sstats matches: load failed ({e})")

    # ── 4. SStats statistics: all at once, indexed by game_id ──
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT game_id, stat_name, home_value, away_value FROM sstats_statistics"
            ).fetchall()
        for r in rows:
            gid = r["game_id"]
            _sstats_stats_by_game.setdefault(gid, []).append({
                "stat": r["stat_name"],
                "home": r["home_value"],
                "away": r["away_value"],
            })
        print(f"    sstats statistics: {len(rows)} records for {len(_sstats_stats_by_game)} games")
    except Exception as e:
        print(f"    sstats statistics: load failed ({e})")

    # ── 5. SStats events: all at once, indexed by game_id ──
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT game_id, minute, event_type, team FROM sstats_events"
            ).fetchall()
        for r in rows:
            gid = r["game_id"]
            _sstats_events_by_game.setdefault(gid, []).append({
                "minute": r["minute"],
                "type": r["event_type"] or "",
                "team": r["team"] or "",
            })
        print(f"    sstats events: {len(rows)} records for {len(_sstats_events_by_game)} games")
    except Exception as e:
        print(f"    sstats events: load failed ({e})")

    # ── 6. League-average odds (fallback for matches without odds) ──
    try:
        league_odds: Dict[str, List[Tuple[float, float, float]]] = {}
        for m in all_matches:
            mid = m.get("id")
            if mid and mid in _match_odds_cache:
                row = _match_odds_cache[mid]
                if row:
                    league_odds.setdefault(m["league_slug"], []).append(
                        (float(row["implied_h"]), float(row["implied_d"]), float(row["implied_a"]))
                    )
        for slug, odds_list in league_odds.items():
            if odds_list:
                n = len(odds_list)
                _league_avg_odds[slug] = (
                    sum(o[0] for o in odds_list) / n,
                    sum(o[1] for o in odds_list) / n,
                    sum(o[2] for o in odds_list) / n,
                )
        print(f"    league avg odds: {len(_league_avg_odds)} leagues computed")
    except Exception as e:
        print(f"    league avg odds: compute failed ({e})")

    # ── 7. Point-in-time features: preload team match history ──
    try:
        fpit._preload_team_matches(all_matches)
        print(f"    point-in-time: {len(fpit._team_matches_cache)} teams loaded")
    except Exception as e:
        print(f"    point-in-time: load failed ({e})")


def _get_team_sstats_stats(team_id: int, match_date: str, limit: int = 10) -> Dict[str, float]:
    """Get average sstats statistics for a team's recent games.

    Returns NaN for stats not available in the data (not 0 — missing data ≠ 0).
    Coverage: ~2-13% of training rows have sstats data (as of 2026-07).
    """
    _NaN = None
    defaults = {"shots_on_target": _NaN, "total_shots": _NaN, "xg_stat": _NaN,
                "possession": _NaN, "corners": _NaN, "fouls": _NaN,
                "yellow_cards": _NaN, "red_cards": _NaN, "offsides": _NaN,
                "passes_accurate": _NaN, "big_chances": _NaN}
    games = _team_sstats_games.get(team_id, [])
    if not games:
        return defaults
    # Filter to games before match_date, keep is_home flag
    relevant = [(gid, d, is_h) for gid, d, is_h in games if d < match_date][:limit]
    if not relevant:
        return defaults

    # Accumulate per-stat: for each stat, collect team-specific values
    # (home_value if team was home, away_value if team was away)
    acc = {k: [] for k in defaults}
    for gid, _, is_home in relevant:
        for stat in _sstats_stats_by_game.get(gid, []):
            sn = stat["stat"].lower().replace("_", " ")
            raw = stat["home"] if is_home else stat["away"]
            try:
                val = float(raw) if raw not in (None, "") else None
            except (ValueError, TypeError):
                val = None
            if val is None:
                continue
            if "shots on target" in sn or "shot on target" in sn:
                acc["shots_on_target"].append(val)
            elif "total shot" in sn:
                acc["total_shots"].append(val)
            elif "expected goal" in sn or "xg" in sn:
                acc["xg_stat"].append(val)
            elif "possession" in sn:
                acc["possession"].append(val)
            elif "corner" in sn:
                acc["corners"].append(val)
            elif "foul" in sn:
                acc["fouls"].append(val)
            elif "yellow" in sn:
                acc["yellow_cards"].append(val)
            elif "red" in sn:
                acc["red_cards"].append(val)
            elif "offside" in sn:
                acc["offsides"].append(val)
            elif "pass" in sn and "accurate" in sn:
                acc["passes_accurate"].append(val)
            elif "big chance" in sn:
                acc["big_chances"].append(val)

    return {k: float(np.mean(v)) if v else defaults[k] for k, v in acc.items()}


def _get_team_sstats_event_features(team_id: int, match_date: str, limit: int = 10) -> Dict[str, float]:
    """Get card/substitution stats for a team's recent games.

    Returns None if no sstats data available (not float('nan')).
    Coverage: ~2-13% of training rows have sstats data (as of 2026-07).

    NOTE: avg_goals_1h/2h removed — sstats API does not return event minute data
    for finished matches, making goals-by-half split impossible from this source.
    """
    _NaN = None
    defaults = {"cards_per_game": _NaN, "subs_per_game": _NaN}
    games = _team_sstats_games.get(team_id, [])
    relevant = [(gid, d, is_h) for gid, d, is_h in games if d < match_date][:limit]
    if not relevant:
        return defaults
    cards_list, subs_list = [], []
    for gid, _, _ in relevant:
        events = _sstats_events_by_game.get(gid, [])
        # type='2' is Cards, type='1' is Substitutions
        cards = sum(1 for e in events if e["type"] == "2")
        subs = sum(1 for e in events if e["type"] == "1")
        cards_list.append(cards)
        subs_list.append(subs)
    return {
        "cards_per_game": float(np.mean(cards_list)) if cards_list else defaults["cards_per_game"],
        "subs_per_game": float(np.mean(subs_list)) if subs_list else defaults["subs_per_game"],
    }


def _get_team_injury_count(team_id: int, match_date: str = "") -> int:
    """Count active injuries for a team at match_date."""
    injuries = _injuries_cache.get(team_id, [])
    if not injuries:
        return 0
    if not match_date:
        return len(injuries)
    return sum(1 for inj in injuries if inj["since"] <= match_date <= inj["until"])


def _get_match_weather(match_id: int) -> Dict[str, float]:
    """Get weather data for a match."""
    return _weather_cache.get(match_id, {"temp_c": 15.0, "rain_mm": 0.0, "wind_ms": 2.0})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    import time as _time

    # Line-buffered stdout so live logs appear immediately when piped to a file
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    def _ts(msg: str) -> None:
        print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)

    run_t0 = _time.time()
    _ts("=== TRAIN START (parallel DC + heartbeat logs) ===")
    _ts(">>> Building feature dataset...")
    X, y, dates, leagues, meta = build_dataset()
    if len(X) < 100:
        raise SystemExit(
            f"Only {len(X)} usable training rows. "
            "Run 'python data_collector.py' first to populate the DB."
        )
    _ts(f"  Dataset: {len(X)} rows, {len(X.columns)} features "
        f"(build took {_time.time() - run_t0:.0f}s)")
    _ts(f"  Class distribution: away={sum(y==0)} draw={sum(y==1)} home={sum(y==2)}")

    # Time-aware split
    order = dates.argsort().values
    X       = X.iloc[order].reset_index(drop=True)
    y       = y.iloc[order].reset_index(drop=True)
    dates   = dates.iloc[order].reset_index(drop=True)
    leagues = leagues.iloc[order].reset_index(drop=True)
    meta    = meta.iloc[order].reset_index(drop=True)

    # ── Expanding window cross-validation ────────────────────────────────────
    N_FOLDS = 3
    fold_size = len(X) // (N_FOLDS + 1)
    _ts(f"  Expanding window CV: {N_FOLDS} folds, {len(X)} rows total")

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

        fold_t0 = _time.time()
        _ts(f"  Fold {fold+1}/{N_FOLDS}: train={len(X_tr)} test={len(X_te)} "
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
        _ts(f"    Fold {fold+1} Acc: {acc_f:.2%}  Log-loss: {ll_f:.3f} "
            f"(fold wall {_time.time() - fold_t0:.0f}s)")

    _ts(f"=== CV Results ({N_FOLDS} folds) ===")
    print(f"  Mean accuracy:  {np.mean(all_fold_accs):.2%} +/- {np.std(all_fold_accs):.2%}", flush=True)
    print(f"  Mean log-loss:  {np.mean(all_fold_lls):.3f} +/- {np.std(all_fold_lls):.3f}", flush=True)

    # ── Final model on all data (except last 20% for held-out eval) ──────────
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    leagues_train, leagues_test = leagues.iloc[:split], leagues.iloc[split:]
    dates_train,   dates_test   = dates.iloc[:split],   dates.iloc[split:]
    meta_train,    meta_test    = meta.iloc[:split],    meta.iloc[split:]
    _ts(f"  Final train: {len(X_train)} rows (up to {dates.iloc[split-1]})")
    _ts(f"  Final test:  {len(X_test)} rows (after {dates.iloc[split]})")

    _ts(">>> Training FINAL ensemble (XGB + LightGBM + Dixon-Coles)...")
    ens = Ensemble()
    ens.fit(X_train, y_train, leagues_train, dates_train, meta_train)

    _ts(">>> Evaluating on held-out future matches...")
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

    joblib.dump({"ensemble": ens, "features": FEATURE_NAMES, "format": "ensemble_v4",
                 "temperature": ens.temperature},
                MODEL_PATH)
    _ts(f"Model saved to {MODEL_PATH}")
    _ts(f"=== TRAIN FINISHED in {(_time.time() - run_t0)/60:.1f} min ===")


def _simulate_roi(probas: np.ndarray, y_true: np.ndarray, meta: pd.DataFrame) -> None:
    """For each test match where we have bookmaker odds, place a flat-stake bet
    on the outcome whose model probability exceeds the bookmaker implied prob
    by EDGE_THRESHOLD. Track cumulative ROI."""
    EDGE_THRESHOLDS = [0.02, 0.05, 0.10]
    OUTCOMES = ["away", "draw", "home"]
    odds_cache: Dict[int, Dict] = {}

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
            my = probas[i]
            bk = np.array([o["implied_a"], o["implied_d"], o["implied_h"]])
            edge = my - bk
            best = int(np.argmax(edge))
            if edge[best] < thr: continue
            stake = 1.0
            odds_dec = {"away": o["away_odds"], "draw": o["draw_odds"],
                        "home": o["home_odds"]}[OUTCOMES[best]]
            bets += 1; stake_total += stake
            if int(y_true[i]) == best:
                pnl += stake * (odds_dec - 1.0)
                wins += 1
            else:
                pnl -= stake
        if bets == 0:
            print(f"  edge >= {thr*100:>4.0f}%: no value bets found")
            continue
        roi = pnl / stake_total * 100
        wr  = wins / bets * 100
        verdict = "PROFITABLE" if roi > 0 else "loses to market"
        print(f"  edge >= {int(thr*100):>2d}%:  bets={bets:>4d}  win_rate={wr:>5.1f}%  "
              f"ROI={roi:+6.2f}%  ({verdict})")


if __name__ == "__main__":
    main()
