"""
tennis_trainer.py — Fast tennis prediction model trainer.

Optimized for large datasets (100K+ matches).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

import tennis.tennis_db as tennis_db


MODEL_PATH = str(Path(__file__).parent / "tennis_model.pkl")
MIN_MATCHES_FOR_TRAINING = 1000
SAMPLE_SIZE = 50000  # Use subset for speed


def _parse_date(s: str):
    try:
        return dt.datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _to_float(v):
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _to_int(v):
    try:
        return int(v) if v is not None and str(v).strip() else None
    except (TypeError, ValueError):
        return None


# ── Fast feature computation using pre-built lookup tables ───────────────────

def _build_player_stats(matches: list) -> Dict[int, dict]:
    """Pre-compute per-player stats from all matches (single pass)."""
    # Track per-player: matches played, wins, serve stats
    stats = {}

    for m in matches:
        p1 = m.get("player1_id")
        p2 = m.get("player2_id")
        winner = m.get("winner_id")

        if not p1 or not p2:
            continue

        for pid in [p1, p2]:
            if pid not in stats:
                stats[pid] = {
                    "total": 0, "wins": 0,
                    "aces": 0, "dfs": 0, "svpts": 0,
                    "first_in": 0, "first_won": 0, "second_won": 0,
                    "bp_saved": 0, "bp_faced": 0,
                    "surfaces": {},
                }
            s = stats[pid]
            s["total"] += 1
            if winner == pid:
                s["wins"] += 1

            # Serve stats
            is_p1 = (pid == p1)
            prefix = "w_" if (winner == pid) else "l_"
            s["aces"] += _to_int(m.get(f"{prefix}ace")) or 0
            s["dfs"] += _to_int(m.get(f"{prefix}df")) or 0
            s["svpts"] += _to_int(m.get(f"{prefix}svpt")) or 0
            s["first_in"] += _to_int(m.get(f"{prefix}1stIn")) or 0
            s["first_won"] += _to_int(m.get(f"{prefix}1stWon")) or 0
            s["second_won"] += _to_int(m.get(f"{prefix}2ndWon")) or 0
            s["bp_saved"] += _to_int(m.get(f"{prefix}bpSaved")) or 0
            s["bp_faced"] += _to_int(m.get(f"{prefix}bpFaced")) or 0

            # Surface tracking
            surface = m.get("surface", "")
            if surface:
                if surface not in s["surfaces"]:
                    s["surfaces"][surface] = {"total": 0, "wins": 0}
                s["surfaces"][surface]["total"] += 1
                if winner == pid:
                    s["surfaces"][surface]["wins"] += 1

    return stats


def _build_ranking_lookup(matches: list) -> Dict[int, Tuple[int, int]]:
    """Get latest ranking for each player: {player_id: (rank, points)}."""
    rankings = {}
    for m in matches:
        for prefix in ["player1", "player2"]:
            pid = m.get(f"{prefix}_id")
            rank = _to_int(m.get(f"{prefix}_ranking"))
            pts = _to_int(m.get(f"{prefix}_rank_points"))
            if pid and rank:
                rankings[pid] = (rank, pts or 0)
    return rankings


def _build_h2h(matches: list) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Pre-compute H2H: {(p1, p2): (p1_wins, p2_wins)}."""
    h2h = {}
    for m in matches:
        p1 = m.get("player1_id")
        p2 = m.get("player2_id")
        winner = m.get("winner_id")
        if not p1 or not p2 or not winner:
            continue

        key = (min(p1, p2), max(p1, p2))
        if key not in h2h:
            h2h[key] = [0, 0]

        if winner == p1:
            h2h[key][0] += 1
        else:
            h2h[key][1] += 1

    return {k: tuple(v) for k, v in h2h.items()}


def _build_form(matches: list, window: int = 20) -> Dict[int, List[int]]:
    """Build rolling form for each player: list of wins (1) / losses (0)."""
    form = {}
    for m in matches:
        p1 = m.get("player1_id")
        p2 = m.get("player2_id")
        winner = m.get("winner_id")
        if not p1 or not p2:
            continue

        for pid in [p1, p2]:
            if pid not in form:
                form[pid] = []
            form[pid].append(1 if winner == pid else 0)
            if len(form[pid]) > window * 2:
                form[pid] = form[pid][-window:]

    return form


# ── Main feature builder ─────────────────────────────────────────────────────

TOURNEY_LEVEL_MAP = {"G": 5, "A": 4, "M": 3, "B": 2, "D": 1, "O": 1, "": 1}

FEATURE_NAMES = [
    "p1_ranking", "p2_ranking", "ranking_diff",
    "p1_rank_pts", "p2_rank_pts", "rank_pts_diff",
    "p1_win_rate", "p2_win_rate", "form_diff",
    "p1_surface_wr", "p2_surface_wr", "surface_form_diff",
    "p1_ace_pct", "p1_df_pct", "p1_1st_serve_pct", "p1_1st_won_pct",
    "p2_ace_pct", "p2_df_pct", "p2_1st_serve_pct", "p2_1st_won_pct",
    "ace_diff", "1st_won_diff",
    "h2h_p1", "h2h_p2", "h2h_diff",
    "tourney_level", "draw_size",
    "p1_matches_30d", "p2_matches_30d", "fatigue_diff",
]


def build_features_fast(m: dict, player_stats: dict, rankings: dict,
                        h2h: dict, form: dict, all_matches_by_date: list,
                        p1_id: int = None, p2_id: int = None) -> list:
    """Build feature vector for one match using pre-computed lookups.
    
    Can be called with either:
    - A match dict (m) containing player1_id, player2_id
    - Explicit p1_id, p2_id parameters
    """
    if p1_id is None:
        p1_id = m.get("player1_id")
    if p2_id is None:
        p2_id = m.get("player2_id")
    
    if not p1_id or not p2_id:
        # Return default features
        return [0.0] * len(FEATURE_NAMES)
    
    surface = m.get("surface", "")

    # Rankings
    r1 = rankings.get(p1_id, (100, 0))
    r2 = rankings.get(p2_id, (100, 0))

    # Win rates
    s1 = player_stats.get(p1_id, {"total": 0, "wins": 0})
    s2 = player_stats.get(p2_id, {"total": 0, "wins": 0})
    wr1 = s1["wins"] / s1["total"] if s1["total"] > 0 else 0.5
    wr2 = s2["wins"] / s2["total"] if s2["total"] > 0 else 0.5

    # Surface form
    sf1 = s1.get("surfaces", {}).get(surface, {"total": 0, "wins": 0})
    sf2 = s2.get("surfaces", {}).get(surface, {"total": 0, "wins": 0})
    swr1 = sf1["wins"] / sf1["total"] if sf1["total"] > 3 else 0.5
    swr2 = sf2["wins"] / sf2["total"] if sf2["total"] > 3 else 0.5

    # Serve stats
    def _serve_pct(stats, key_num, key_den):
        num = stats.get(key_num, 0)
        den = stats.get(key_den, 1)
        return num / den if den > 0 else 0

    p1_ace = _serve_pct(s1, "aces", "svpts")
    p1_df = _serve_pct(s1, "dfs", "svpts")
    p1_1st_in = _serve_pct(s1, "first_in", "svpts")
    p1_1st_won = _serve_pct(s1, "first_won", "first_in")
    p2_ace = _serve_pct(s2, "aces", "svpts")
    p2_df = _serve_pct(s2, "dfs", "svpts")
    p2_1st_in = _serve_pct(s2, "first_in", "svpts")
    p2_1st_won = _serve_pct(s2, "first_won", "first_in")

    # H2H
    h2h_key = (min(p1_id, p2_id), max(p1_id, p2_id))
    h = h2h.get(h2h_key, (0, 0))
    h2h_p1 = h[0] if p1_id < p2_id else h[1]
    h2h_p2 = h[1] if p1_id < p2_id else h[0]

    # Tournament level
    tlev = TOURNEY_LEVEL_MAP.get(m.get("tourney_level", ""), 1)
    draw = _to_int(m.get("draw_size")) or 32

    # Fatigue (matches in last 30 days)
    match_date = _parse_date(m.get("date", ""))
    p1_fatigue = 0
    p2_fatigue = 0
    if match_date:
        for prev in all_matches_by_date:
            prev_date = _parse_date(prev.get("date", ""))
            if not prev_date or prev_date >= match_date:
                continue
            if (match_date - prev_date).days > 30:
                break
            if prev.get("player1_id") == p1_id or prev.get("player2_id") == p1_id:
                p1_fatigue += 1
            if prev.get("player1_id") == p2_id or prev.get("player2_id") == p2_id:
                p2_fatigue += 1

    return [
        r1[0], r2[0], r1[0] - r2[0],
        r1[1], r2[1], r1[1] - r2[1],
        wr1, wr2, wr1 - wr2,
        swr1, swr2, swr1 - swr2,
        p1_ace, p1_df, p1_1st_in, p1_1st_won,
        p2_ace, p2_df, p2_1st_in, p2_1st_won,
        p1_ace - p2_ace, p1_1st_won - p2_1st_won,
        h2h_p1, h2h_p2, h2h_p1 - h2h_p2,
        tlev, draw,
        p1_fatigue, p2_fatigue, p1_fatigue - p2_fatigue,
    ]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset():
    """Build training dataset."""
    print("  Loading matches from DB...")
    all_matches = tennis_db.all_finished_matches()
    print(f"  Total matches: {len(all_matches)}")

    if len(all_matches) < MIN_MATCHES_FOR_TRAINING:
        raise SystemExit(f"Too few matches ({len(all_matches)}).")

    # Pre-compute lookups (single pass)
    print("  Building player stats...")
    player_stats = _build_player_stats(all_matches)

    print("  Building rankings...")
    rankings = _build_ranking_lookup(all_matches)

    print("  Building H2H...")
    h2h = _build_h2h(all_matches)

    print("  Building form...")
    form = _build_form(all_matches, 20)

    # Sample matches for speed (use most recent)
    if len(all_matches) > SAMPLE_SIZE:
        sampled = all_matches[-SAMPLE_SIZE:]
        print(f"  Using {SAMPLE_SIZE} most recent matches for training")
    else:
        sampled = all_matches

    # Build features
    print("  Computing features...")
    X_rows = []
    y_rows = []
    date_rows = []
    meta_rows = []

    # Use expanding window: need enough history
    min_idx = min(MIN_MATCHES_FOR_TRAINING, len(all_matches) // 4)
    matches_for_lookup = all_matches[:min_idx]

    for i in range(min_idx, len(sampled)):
        m = sampled[i]
        p1 = m.get("player1_id")
        p2 = m.get("player2_id")
        winner = m.get("winner_id")

        if not p1 or not p2 or not winner:
            continue

        # Update lookups incrementally
        matches_for_lookup.append(m)

        features = build_features_fast(
            m, player_stats, rankings, h2h, form, matches_for_lookup
        )

        label = 1 if winner == p1 else 0

        X_rows.append(features)
        y_rows.append(label)
        date_rows.append(m.get("date", ""))
        meta_rows.append({
            "p1_name": m.get("player1_name", ""),
            "p2_name": m.get("player2_name", ""),
            "surface": m.get("surface", ""),
        })

        if len(X_rows) % 10000 == 0:
            print(f"    {len(X_rows)} rows computed...")

    X = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    y = pd.Series(y_rows, name="result")
    dates = pd.Series(date_rows, name="date")
    meta = pd.DataFrame(meta_rows)

    # Fill NaN
    for col in X.columns:
        X[col] = X[col].astype(float)
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0.0)

    print(f"  Dataset: {len(X)} rows, {len(X.columns)} features")
    print(f"  Class: p1_win={sum(y==1)}, p2_win={sum(y==0)}")

    return X, y, dates, meta


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    print("=== Tennis Model Training ===\n")

    X, y, dates, meta = build_dataset()

    # Time-aware split
    order = dates.argsort().values
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    # CV
    N_FOLDS = 5
    fold_size = len(X) // (N_FOLDS + 1)
    print(f"\n  Expanding window CV: {N_FOLDS} folds")

    all_accs = []
    all_lls = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, len(X))
        if test_end <= test_start:
            break

        X_tr, X_te = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_tr, y_te = y.iloc[:train_end], y.iloc[test_start:test_end]

        # Skip if test set too small or has only one class
        if len(X_te) < 100 or len(y_te.unique()) < 2:
            print(f"\n  Fold {fold+1}/{N_FOLDS}: skipped (too small)")
            continue

        print(f"\n  Fold {fold+1}/{N_FOLDS}: train={len(X_tr)} test={len(X_te)}")

        xgb = XGBClassifier(
            objective="binary:logistic", n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, min_child_weight=5,
            tree_method="hist", random_state=42, verbosity=0,
        )
        xgb.fit(X_tr, y_tr)

        lgbm = LGBMClassifier(
            objective="binary", n_estimators=200, learning_rate=0.05,
            num_leaves=16, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, min_child_samples=20,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_tr, y_tr)

        p = 0.5 * xgb.predict_proba(X_te)[:, 1] + 0.5 * lgbm.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, (p > 0.5).astype(int))
        ll = log_loss(y_te, p)

        all_accs.append(acc)
        all_lls.append(ll)
        print(f"    Acc: {acc:.2%}  Log-loss: {ll:.3f}")

    print(f"\n=== CV Results ===")
    print(f"  Accuracy: {np.mean(all_accs):.2%} ± {np.std(all_accs):.2%}")
    print(f"  Log-loss: {np.mean(all_lls):.3f} ± {np.std(all_lls):.3f}")

    # Final 80/20
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    meta_te = meta.iloc[split:]

    print(f"\n>>> Training final model...")

    xgb_f = XGBClassifier(
        objective="binary:logistic", n_estimators=300, learning_rate=0.03,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, min_child_weight=5,
        tree_method="hist", random_state=42, verbosity=0,
    )
    xgb_f.fit(X_tr, y_tr)

    lgbm_f = LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03,
        num_leaves=16, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    lgbm_f.fit(X_tr, y_tr)

    p = 0.5 * xgb_f.predict_proba(X_te)[:, 1] + 0.5 * lgbm_f.predict_proba(X_te)[:, 1]
    preds = (p > 0.5).astype(int)
    acc = accuracy_score(y_te, preds)

    print(f"\n=== Test Set ===")
    print(f"Accuracy: {acc:.2%}")
    if len(y_te.unique()) >= 2:
        ll = log_loss(y_te, p)
        print(f"Log-loss: {ll:.3f}")
        print(classification_report(y_te, preds, target_names=["P2 Win", "P1 Win"]))
    else:
        print("(Log-loss skipped: single class in test set)")

    # Features
    imps = sorted(zip(FEATURE_NAMES, xgb_f.feature_importances_), key=lambda x: -x[1])
    print("\nTop features:")
    for name, imp in imps[:10]:
        print(f"   {name:25s} {imp:.3f}")

    # Save
    joblib.dump({
        "xgb": xgb_f, "lgbm": lgbm_f,
        "features": FEATURE_NAMES, "format": "tennis_v1",
    }, MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
