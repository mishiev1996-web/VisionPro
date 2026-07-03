"""
ablation.py — Quick feature ablation (single split, fast).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import train


MARKET_COLS = ["market_implied_h", "market_implied_d", "market_implied_a"]


def run_ablation():
    print("=== Quick Feature Ablation (single split) ===")
    X, y, dates, leagues, meta = train.build_dataset()
    order = dates.argsort().values
    X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    dates, leagues = dates.iloc[order].reset_index(drop=True), leagues.iloc[order].reset_index(drop=True)

    n = len(X)
    split = int(n * 0.7)  # 70/30 for speed
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    print(f"  Train: {split}, Test: {n-split}, Features: {X.shape[1]}")

    # NaN check
    mkt_present = (~X[MARKET_COLS].isna()).all(axis=1).sum()
    print(f"  Rows with market odds: {mkt_present}/{n} ({mkt_present/n*100:.1f}%)")

    configs = {
        "all_features": X.columns.tolist(),
        "no_market": [c for c in X.columns if c not in MARKET_COLS],
        "market_only": MARKET_COLS,
    }

    for name, cols in configs.items():
        Xtr = X_tr[cols].fillna(0)
        Xte = X_te[cols].fillna(0)

        xgb = CalibratedClassifierCV(
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                          tree_method="hist", eval_metric="mlogloss",
                          random_state=42, verbosity=0),
            cv=3, method="isotonic")
        lgbm = CalibratedClassifierCV(
            LGBMClassifier(n_estimators=300, num_leaves=16, learning_rate=0.03,
                           random_state=42, verbosity=-1, force_col_wise=True),
            cv=3, method="isotonic")

        xgb.fit(Xtr, y_tr)
        lgbm.fit(Xtr, y_tr)

        avg = (xgb.predict_proba(Xte) + lgbm.predict_proba(Xte)) / 2
        ll = log_loss(y_te, avg, labels=[0, 1, 2])
        acc = accuracy_score(y_te, np.argmax(avg, axis=1))
        print(f"  {name:20s} ({len(cols):2d} feats): LL={ll:.4f}  Acc={acc:.2%}")

    # Market prediction accuracy (is bookmaker itself good?)
    mkt_test = X_te[MARKET_COLS].fillna(0).values
    mkt_preds = np.argmax(mkt_test, axis=1)
    mkt_acc = accuracy_score(y_te, mkt_preds)
    mkt_ll = log_loss(y_te, mkt_test, labels=[0, 1, 2])
    print(f"  {'market_purely':20s} ({len(MARKET_COLS):2d} feats): LL={mkt_ll:.4f}  Acc={mkt_acc:.2%}")
    print(f"\n  => Market baseline accuracy: {mkt_acc:.2%}")

    # Correlation between model and market predictions
    model_preds = np.argmax(avg, axis=1)
    agree = (model_preds == mkt_preds).mean() * 100
    print(f"  => Model-market agreement: {agree:.1f}%")


if __name__ == "__main__":
    run_ablation()
