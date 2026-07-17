"""
train_v2.py — Train VisionPro model with expanded data sources.

Uses xgabora dataset (230k matches, 25 years) with Elo, Form, Odds.
16 features total (Pi-rating removed after diagnostic: 0.00% accuracy gain).
"""
from __future__ import annotations

import os
import warnings
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "база для обучения")
MODEL_PATH = "model_v2.pkl"

# Feature columns from xgabora dataset (16 features)
# REMOVED: Pi-rating (5 features) — 0.00% accuracy gain, within noise
# REMOVED: match-final stats (shots, corners, etc.) — data leakage
FEATURE_COLS = [
    'HomeElo', 'AwayElo',
    'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'OddHome', 'OddDraw', 'OddAway',
    'Over25', 'Under25',
]

TARGET = 'FTResult'  # H=Home, D=Draw, A=Away


def load_data() -> pd.DataFrame:
    """Load xgabora Matches.csv."""
    csv_path = os.path.join(DATA_DIR, "Matches.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"  Loaded {len(df)} matches")

    # Clean data
    df = df.dropna(subset=['FTHome', 'FTAway', 'FTResult'])
    df = df[df['FTResult'].isin(['H', 'D', 'A'])]

    # Convert numeric columns
    for col in FEATURE_COLS + ['FTHome', 'FTAway']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with too many NaN features
    df = df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) - 5)

    # Encode target
    df['target'] = df['FTResult'].map({'H': 0, 'D': 1, 'A': 2})

    # Derived features
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']
    df['form_diff'] = df['Form5Home'] - df['Form5Away']

    # Implied probabilities from odds
    df['implied_home'] = 1.0 / df['OddHome']
    df['implied_draw'] = 1.0 / df['OddDraw']
    df['implied_away'] = 1.0 / df['OddAway']
    total = df['implied_home'] + df['implied_draw'] + df['implied_away']
    df['implied_home'] /= total
    df['implied_draw'] /= total
    df['implied_away'] /= total

    # Drop rows with NaN targets
    df = df.dropna(subset=['target'])

    print(f"  Final dataset: {len(df)} matches")
    return df


def get_feature_names():
    """Return list of all feature names."""
    return FEATURE_COLS + ['elo_diff', 'form_diff',
                           'implied_home', 'implied_draw', 'implied_away']


def train_model(df: pd.DataFrame) -> Dict:
    """Train XGBoost + LightGBM ensemble."""
    feature_cols = get_feature_names()
    X = df[feature_cols].values
    y = df['target'].values

    # Chronological split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='mlogloss', random_state=42, use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_probs = lgbm.predict_proba(X_test)

    # Ensemble (simple average)
    ensemble_probs = (xgb_probs + lgbm_probs) / 2

    # Metrics
    xgb_pred = np.argmax(xgb_probs, axis=1)
    lgbm_pred = np.argmax(lgbm_probs, axis=1)
    ens_pred = np.argmax(ensemble_probs, axis=1)

    results = {
        'xgb_accuracy': accuracy_score(y_test, xgb_pred),
        'lgbm_accuracy': accuracy_score(y_test, lgbm_pred),
        'ensemble_accuracy': accuracy_score(y_test, ens_pred),
        'xgb_logloss': log_loss(y_test, xgb_probs),
        'lgbm_logloss': log_loss(y_test, lgbm_probs),
        'ensemble_logloss': log_loss(y_test, ensemble_probs),
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    # Retrain on full data for production
    xgb_full = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='mlogloss', random_state=42, use_label_encoder=False
    )
    xgb_full.fit(X, y)

    lgbm_full = LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    lgbm_full.fit(X, y)

    # Save model
    model_data = {
        'xgb': xgb_full,
        'lgbm': lgbm_full,
        'feature_names': feature_cols,
        'target_names': ['Home', 'Draw', 'Away'],
        'results': results,
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return results


def main():
    print("=" * 60)
    print("  VisionPro Model Training v2")
    print("=" * 60)

    # Load data
    df = load_data()

    # Train
    results = train_model(df)

    # Print results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  XGBoost Accuracy:     {results['xgb_accuracy']:.4f}")
    print(f"  LightGBM Accuracy:    {results['lgbm_accuracy']:.4f}")
    print(f"  Ensemble Accuracy:    {results['ensemble_accuracy']:.4f}")
    print(f"  XGBoost Log Loss:     {results['xgb_logloss']:.4f}")
    print(f"  LightGBM Log Loss:    {results['lgbm_logloss']:.4f}")
    print(f"  Ensemble Log Loss:    {results['ensemble_logloss']:.4f}")
    print(f"  Features: {results['n_features']}")
    print(f"  Train: {results['n_train']}, Test: {results['n_test']}")


if __name__ == "__main__":
    main()
