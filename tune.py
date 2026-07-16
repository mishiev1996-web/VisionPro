"""
tune.py — Optuna hyperparameter tuning for XGB + LGBM + CatBoost.

Uses log_loss as the optimization metric (proper for probabilistic forecasting).
Uses walk-forward expanding-window CV (no data leakage).

Run: python tune.py [--trials 50] [--years 5]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import db
from train import build_dataset, FEATURE_NAMES


optuna.logging.set_verbosity(optuna.logging.WARNING)

_CACHED = None
N_FOLDS = 5
_YEARS = 5  # default: last 5 seasons


def _get_dataset(years: int = 5):
    global _CACHED
    if _CACHED is None:
        print(f"  Building dataset (last {years} years)...", flush=True)
        X, y, dates, leagues, meta = build_dataset()
        # Filter to recent N years
        cutoff = (dt.date.today() - dt.timedelta(days=years * 365)).isoformat()
        mask = dates >= cutoff
        X, y, dates, leagues = (X[mask].reset_index(drop=True),
                                y[mask].reset_index(drop=True),
                                dates[mask].reset_index(drop=True),
                                leagues[mask].reset_index(drop=True))
        meta = meta[mask].reset_index(drop=True)
        print(f"  Done. Filtered to {len(X)} rows (last {years} years)", flush=True)
        _CACHED = X, y, dates, leagues, meta
    return _CACHED


def _time_decay_weights(dates: pd.Series, half_life: int = 365,
                         reference: str = None) -> np.ndarray:
    ref = dt.date.fromisoformat(reference) if reference else dt.date.today()
    out = np.empty(len(dates), dtype=float)
    decay = math.log(2) / half_life
    for i, d in enumerate(dates):
        try:
            md = dt.date.fromisoformat(str(d)[:10])
            days = max(0, (ref - md).days)
            out[i] = max(0.05, math.exp(-decay * days))
        except Exception:
            out[i] = 1.0
    return out


def objective(trial: optuna.Trial) -> float:
    X, y, dates, leagues, meta = _get_dataset(_YEARS)
    order = dates.argsort().values
    X, y, dates, leagues = (X.iloc[order].reset_index(drop=True),
                            y.iloc[order].reset_index(drop=True),
                            dates.iloc[order].reset_index(drop=True),
                            leagues.iloc[order].reset_index(drop=True))

    # XGB params
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("xgb_lambda", 0.1, 5.0, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
    }

    # LGBM params
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 63),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 12),
        "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("lgbm_lambda", 0.1, 5.0, log=True),
        "min_child_samples": trial.suggest_int("lgbm_min_child", 5, 30),
    }

    # CatBoost params
    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 300, 1500),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("cat_depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 1.0, 5.0, log=True),
    }

    # Walk-forward expanding window CV
    n = len(X)
    fold_size = n // (N_FOLDS + 1)
    all_fold_lls = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break

        X_tr, X_val = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_tr, y_val = y.iloc[:train_end], y.iloc[test_start:test_end]
        dt_tr = dates.iloc[:train_end]

        decay_w = _time_decay_weights(dt_tr, reference=str(dt_tr.iloc[-1])[:10])

        xgb = XGBClassifier(
            objective="multi:softprob", num_class=3,
            tree_method="hist", eval_metric="mlogloss",
            random_state=42, verbosity=0, **xgb_params)
        xgb_cal = CalibratedClassifierCV(xgb, cv=3, method="isotonic")
        xgb_cal.fit(X_tr, y_tr, sample_weight=decay_w)

        lgbm = LGBMClassifier(
            objective="multiclass", num_class=3,
            random_state=42, verbosity=-1, force_col_wise=True, **lgbm_params)
        lgbm_cal = CalibratedClassifierCV(lgbm, cv=3, method="isotonic")
        lgbm_cal.fit(X_tr, y_tr, sample_weight=decay_w)

        cat = CatBoostClassifier(
            random_seed=42, verbose=0,
            loss_function="MultiClass", eval_metric="MultiClass", **cat_params)
        cat_cal = CalibratedClassifierCV(cat, cv=3, method="isotonic")
        cat_cal.fit(X_tr, y_tr, sample_weight=decay_w)

        avg = (xgb_cal.predict_proba(X_val) + lgbm_cal.predict_proba(X_val) +
               cat_cal.predict_proba(X_val)) / 3.0

        ll = log_loss(y_val, avg, labels=[0, 1, 2])
        all_fold_lls.append(ll)

    if not all_fold_lls:
        return 1e6

    mean_ll = np.mean(all_fold_lls)
    return -mean_ll


def main():
    global _YEARS
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--years", type=int, default=5, help="Use last N years of data (default: 5)")
    args = parser.parse_args()
    _YEARS = args.years

    print(f"=== Optuna Tuning ({args.trials} trials, last {args.years} years, log_loss objective) ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\n=== Best trial ===")
    print(f"  Log-loss: {-study.best_trial.value:.4f}")

    print(f"\n  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\n  Saved to best_params.json")


if __name__ == "__main__":
    main()
