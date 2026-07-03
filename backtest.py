"""
backtest.py — Walk-forward backtest with NO data leakage.

Fast approach: pre-build ALL features once, then walk-forward by
splitting the dataset at each cutoff date. No per-match DB queries.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

import db
import train
from models.ensemble import Ensemble


def _build_full_dataset_fast() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Build full dataset once (reuses train.build_dataset logic)."""
    train._clear_train_caches()
    return train.build_dataset()


def run_walk_forward_backtest(
    n_folds: int = 3,
    test_window_days: int = 180,
    max_matches_per_league: int = 200,
    min_confidence: float = 0.5,
    progress_cb=None,
) -> Dict[str, Any]:
    """Walk-forward: train on data BEFORE cutoff, test AFTER.

    All features are pre-built once. Each fold trains a fresh ensemble
    on the expanding training window and evaluates on the test window.
    """
    def _emit(msg, **kw):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg, **kw})
        else:
            print(msg)

    _emit("=== Walk-Forward Backtest (no data leakage) ===")
    start_time = time.time()

    # Step 1: build full dataset once
    _emit("  Pre-building full feature dataset...")
    X, y, dates, leagues, meta = _build_full_dataset_fast()
    n = len(X)
    _emit(f"  Dataset: {n} rows, {len(X.columns)} features")

    # Sort by date
    order = dates.argsort().values
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)
    leagues = leagues.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    date_strs = [str(d)[:10] for d in dates]
    unique_dates = sorted(set(date_strs))
    first_d = unique_dates[0]
    last_d = unique_dates[-1]
    total_days = (dt.date.fromisoformat(last_d) - dt.date.fromisoformat(first_d)).days
    fold_step = total_days // (n_folds + 1)

    all_fold_results = []

    for fold in range(n_folds):
        cutoff_date = dt.date.fromisoformat(first_d) + dt.timedelta(days=fold_step * (fold + 1))
        test_end = cutoff_date + dt.timedelta(days=test_window_days)
        cutoff_str = cutoff_date.isoformat()
        test_end_str = test_end.isoformat()

        # Find row indices for train/test splits
        train_mask = np.array([d <= cutoff_str for d in date_strs])
        test_mask = np.array([cutoff_str < d and d <= test_end_str for d in date_strs])

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        _emit(f"\n--- Fold {fold+1}/{n_folds}: cutoff={cutoff_str}, test={cutoff_str}..{test_end_str} ---")
        _emit(f"  Train: {n_train} rows, Test: {n_test} rows")

        if n_train < 500 or n_test < 30:
            print("  SKIPPED: insufficient data")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        dt_tr = dates[train_mask]
        lg_tr = leagues[train_mask]
        mt_tr = meta[train_mask]

        X_te, y_te = X[test_mask], y[test_mask]
        lg_te = leagues[test_mask]
        mt_te = meta[test_mask]

        # Train ensemble from scratch
        ens = Ensemble()
        ens.fit(X_tr, y_tr, lg_tr, dt_tr, mt_tr)

        # Predict on test set
        probas = np.zeros((n_test, 3))
        test_indices = np.where(test_mask)[0]
        for j in range(n_test):
            probas[j] = ens.predict_proba(
                X_te.iloc[[j]], league_slug=lg_te.iloc[j],
                home_name=mt_te.iloc[j]["home_name"],
                away_name=mt_te.iloc[j]["away_name"],
            )[0]

        preds = np.argmax(probas, axis=1)
        correct = (preds == y_te.values).sum()

        eps = 1e-10
        ll_per = [-math.log(max(proba[int(actual)], eps)) for proba, actual in zip(probas, y_te.values)]
        brier_per = [sum((proba[k] - (1.0 if k == int(actual) else 0.0)) ** 2 for k in range(3))
                     for proba, actual in zip(probas, y_te.values)]

        confidence = np.max(probas, axis=1)
        picks = [(float(c), bool(pred == actual))
                 for c, pred, actual in zip(confidence, preds, y_te.values)
                 if c >= min_confidence]
        proba_data = [(float(c), bool(pred == actual))
                      for c, pred, actual in zip(confidence, preds, y_te.values)]

        acc = correct / n_test * 100
        avg_ll = np.mean(ll_per)
        avg_brier = np.mean(brier_per)

        fold_result = {
            "fold": fold + 1,
            "correct": int(correct), "total": n_test,
            "accuracy": round(acc, 1),
            "log_loss_sum": float(np.sum(ll_per)),
            "avg_log_loss": round(float(avg_ll), 4),
            "brier_score": round(float(avg_brier), 4),
            "picks": picks, "proba_data": proba_data,
            "train_rows": n_train, "test_rows": n_test,
        }
        all_fold_results.append(fold_result)

        _emit(f"  => Acc={acc:.1f}%  LL={avg_ll:.4f}  Brier={avg_brier:.4f}")

    if not all_fold_results:
        return {"error": "All folds skipped"}

    total_correct = sum(r["correct"] for r in all_fold_results)
    total_n = sum(r["total"] for r in all_fold_results)
    avg_ll = sum(r["log_loss_sum"] for r in all_fold_results) / max(total_n, 1)
    avg_brier = np.mean([r["brier_score"] for r in all_fold_results])

    all_picks = []
    all_proba = []
    for r in all_fold_results:
        all_picks.extend(r.get("picks", []))
        all_proba.extend(r.get("proba_data", []))

    overall = {
        "accuracy": round(total_correct / total_n * 100, 1) if total_n else 0,
        "correct": total_correct, "total": total_n,
        "avg_log_loss": round(avg_ll, 4),
        "brier_score": round(float(avg_brier), 4),
        "n_folds": len(all_fold_results),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }

    if all_picks:
        ps = sorted(all_picks, key=lambda x: x[0], reverse=True)
        top = ps[:max(1, len(ps) // 5)]
        overall["top_20pct_accuracy"] = round(sum(1 for _, c in top if c) / len(top) * 100, 1)
        overall["top_20pct_count"] = len(top)

    if all_proba:
        overall["calibration"] = _calibration_report(all_proba)

    overall["folds"] = all_fold_results
    return overall


def _calibration_report(proba_data: list) -> dict:
    brier_sum = sum((1 - p) ** 2 if c else p ** 2 for p, c in proba_data)
    brier = brier_sum / len(proba_data) if proba_data else 0

    bins = np.linspace(0, 1, 11)
    calibration = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [(p, c) for p, c in proba_data if lo <= p < hi]
        if in_bin:
            mean_pred = float(np.mean([p for p, _ in in_bin]))
            mean_actual = float(np.mean([1.0 if c else 0.0 for _, c in in_bin]))
            calibration.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "mean_predicted": round(mean_pred, 3),
                "mean_actual": round(mean_actual, 3),
                "count": len(in_bin),
                "gap": round(abs(mean_pred - mean_actual), 3),
            })
    total = sum(c["count"] for c in calibration)
    ece = sum(c["gap"] * c["count"] / total for c in calibration) if total else 0
    return {"brier_score": round(float(brier), 4), "ece": round(float(ece), 4), "curve": calibration}


def run_backtest(seasons=3, min_confidence=0.5, max_matches_per_league=200, progress_cb=None):
    return run_walk_forward_backtest(n_folds=3, test_window_days=180,
                                     max_matches_per_league=max_matches_per_league,
                                     min_confidence=min_confidence,
                                     progress_cb=progress_cb)


if __name__ == "__main__":
    results = run_walk_forward_backtest(n_folds=3, test_window_days=180)
    print("\n=== RESULTS ===")
    for k, v in results.items():
        if k not in ("folds", "calibration"):
            print(f"  {k}: {v}")
    if "calibration" in results:
        print("\n  Calibration:")
        for b in results["calibration"].get("curve", []):
            print(f"    {b['bin']}: pred={b['mean_predicted']:.3f} "
                  f"actual={b['mean_actual']:.3f} n={b['count']}")
