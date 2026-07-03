"""models/ensemble.py — ensemble classifier with proper OOF stacking,
per-league model selection via log-loss, and per-league calibration.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from models.dixon_coles import DixonColes


USE_CLASS_WEIGHTS = True
PER_LEAGUE_PROVES_ITSELF = True
MIN_LEAGUE_TRAIN_ROWS = 3000
TIME_DECAY_HALF_LIFE_DAYS = 365
N_OOF_FOLDS = 5


def _load_tuned_params() -> Dict:
    path = os.path.join(os.path.dirname(__file__), "..", "best_params.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


TUNED = _load_tuned_params()
DC_BLEND_WEIGHT = TUNED.get("dc_weight", 0.20)


def _make_xgb() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=TUNED.get("xgb_n_estimators", 400),
        learning_rate=TUNED.get("xgb_lr", 0.03),
        max_depth=TUNED.get("xgb_max_depth", 3),
        subsample=TUNED.get("xgb_subsample", 0.87),
        colsample_bytree=TUNED.get("xgb_colsample", 0.63),
        reg_lambda=TUNED.get("xgb_lambda", 4.0),
        min_child_weight=TUNED.get("xgb_min_child", 4),
        tree_method="hist", eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )


def _make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=TUNED.get("lgbm_n_estimators", 350),
        learning_rate=TUNED.get("lgbm_lr", 0.03),
        num_leaves=TUNED.get("lgbm_num_leaves", 16),
        max_depth=TUNED.get("lgbm_max_depth", 6),
        subsample=TUNED.get("lgbm_subsample", 0.72),
        colsample_bytree=TUNED.get("lgbm_colsample", 0.93),
        reg_lambda=TUNED.get("lgbm_lambda", 1.33),
        min_child_samples=TUNED.get("lgbm_min_child", 20),
        random_state=42, verbosity=-1,
        force_col_wise=True,
    )


def _make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
    )


def _sample_weights(y: pd.Series) -> Optional[np.ndarray]:
    if not USE_CLASS_WEIGHTS:
        return None
    return compute_sample_weight("balanced", y)


def _time_decay_weights(dates: pd.Series, reference: Optional[str] = None) -> np.ndarray:
    ref = (reference or dt.date.today().isoformat())[:10]
    try:
        ref_date = dt.date.fromisoformat(ref)
    except Exception:
        return np.ones(len(dates))
    out = np.empty(len(dates), dtype=float)
    decay = math.log(2) / TIME_DECAY_HALF_LIFE_DAYS
    for i, d in enumerate(dates):
        try:
            d10 = str(d)[:10]
            md = dt.date.fromisoformat(d10)
            days = max(0, (ref_date - md).days)
            out[i] = math.exp(-decay * days)
        except Exception:
            out[i] = 1.0
    return out


def _select_calibration_method(X: pd.DataFrame, y: pd.Series) -> str:
    """Choose isotonic vs sigmoid based on sample size.
    Isotonic needs large samples; sigmoid is more robust for small data."""
    n = len(X)
    if n < 500:
        return "sigmoid"
    return "isotonic"


def _fit_calibrated_pair(X: pd.DataFrame, y: pd.Series,
                          decay_w: Optional[np.ndarray] = None) -> Dict:
    """Fit XGB + LGBM + CatBoost with calibrated probabilities.
    Calibration method is chosen based on sample size."""
    cw = _sample_weights(y)
    if cw is not None and decay_w is not None:
        w = cw * decay_w
    elif cw is not None:
        w = cw
    elif decay_w is not None:
        w = decay_w
    else:
        w = None

    method = _select_calibration_method(X, y)
    cv_folds = min(3, len(X) // 100) if len(X) > 300 else 2

    xgb_cal = CalibratedClassifierCV(_make_xgb(), cv=cv_folds, method=method)
    lgb_cal = CalibratedClassifierCV(_make_lgbm(), cv=cv_folds, method=method)
    cat_cal = CalibratedClassifierCV(_make_catboost(), cv=cv_folds, method=method)
    if w is not None:
        xgb_cal.fit(X, y, sample_weight=w)
        lgb_cal.fit(X, y, sample_weight=w)
        cat_cal.fit(X, y, sample_weight=w)
    else:
        xgb_cal.fit(X, y)
        lgb_cal.fit(X, y)
        cat_cal.fit(X, y)
    return {"xgb": xgb_cal, "lgbm": lgb_cal, "cat": cat_cal, "cal_method": method}


class Ensemble:
    """Soft-voting ensemble: calibrated XGB + LightGBM + Dixon-Coles (per-league)
    with proper k-fold OOF stacking meta-learner.

    Class label convention: 0=Away Win, 1=Draw, 2=Home Win.
    """
    def __init__(self):
        self.global_model: Dict = {}
        self.leagues: Dict[str, Dict] = {}
        self.dc: Dict[str, DixonColes] = {}
        self.meta_learner = None

    def fit(self, X: pd.DataFrame, y: pd.Series, leagues: pd.Series,
            dates: pd.Series, meta: pd.DataFrame) -> None:
        decay_w = _time_decay_weights(dates)
        print(f"    time-decay: oldest weight={decay_w.min():.3f}, "
              f"newest={decay_w.max():.3f}")

        # Global XGB+LGB+CatBoost
        print("    [global] fit XGB + LightGBM + CatBoost (calibrated, time-decayed)...")
        self.global_model = _fit_calibrated_pair(X, y, decay_w)

        # Dixon-Coles per league with per-league decay_xi
        for slug in sorted(set(leagues)):
            mask = (leagues == slug)
            sub = meta[mask].copy()
            sub_dates = dates[mask].reset_index(drop=True)
            matches = [
                {"home": r["home_name"], "away": r["away_name"],
                 "home_goals": int(r["home_goals"] or 0),
                 "away_goals": int(r["away_goals"] or 0),
                 "date": d}
                for (_, r), d in zip(sub.iterrows(), sub_dates)
            ]
            if len(matches) < 200:
                continue
            print(f"    [{slug}] Dixon-Coles on {len(matches)} matches...")
            try:
                ref = max(d[:10] for d in sub_dates if d)
                # Per-league decay_xi from tuned params, fallback to default
                dc_xi = TUNED.get(f"dc_xi_{slug}", TUNED.get("dc_xi", 0.0065))
                dc = DixonColes(decay_xi=dc_xi).fit(matches, reference_date=ref)
                self.dc[slug] = dc
            except Exception as e:
                print(f"      DC failed for {slug}: {e}")

        if not PER_LEAGUE_PROVES_ITSELF:
            self._fit_meta_learner(X, y, leagues, dates, meta)
            return

        # Per-league booster sub-models (kept only if they beat global on log_loss)
        for slug in sorted(set(leagues)):
            mask = (leagues == slug)
            X_lg, y_lg = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
            d_lg = dates[mask].reset_index(drop=True)
            if len(X_lg) < MIN_LEAGUE_TRAIN_ROWS:
                continue
            sp = int(len(X_lg) * 0.85)
            X_in, X_val = X_lg.iloc[:sp], X_lg.iloc[sp:]
            y_in, y_val = y_lg.iloc[:sp], y_lg.iloc[sp:]
            w_in = _time_decay_weights(d_lg.iloc[:sp])
            candidate = _fit_calibrated_pair(X_in, y_in, w_in)
            cand_proba = (candidate["xgb"].predict_proba(X_val)
                          + candidate["lgbm"].predict_proba(X_val)
                          + candidate["cat"].predict_proba(X_val)) / 3
            glob_proba = (self.global_model["xgb"].predict_proba(X_val)
                          + self.global_model["lgbm"].predict_proba(X_val)
                          + self.global_model["cat"].predict_proba(X_val)) / 3
            # Use log_loss (lower is better) instead of accuracy
            try:
                cand_ll = log_loss(y_val, cand_proba, labels=[0, 1, 2])
                glob_ll = log_loss(y_val, glob_proba, labels=[0, 1, 2])
            except Exception:
                continue
            if cand_ll < glob_ll - 0.001:  # cand is better (lower log_loss)
                w_full = _time_decay_weights(d_lg)
                self.leagues[slug] = _fit_calibrated_pair(X_lg, y_lg, w_full)
                print(f"    [{slug}] booster kept (cand LL={cand_ll:.4f} vs glob LL={glob_ll:.4f})")

        # Fit stacking meta-learner with proper k-fold OOF
        self._fit_meta_learner(X, y, leagues, dates, meta)

    def _fit_meta_learner(self, X: pd.DataFrame, y: pd.Series,
                          leagues: pd.Series, dates: pd.Series,
                          meta: pd.DataFrame) -> None:
        """Fit stacking meta-learner on k-fold OOF predictions from base models."""
        print("    [meta] fitting stacking meta-learner (k-fold OOF)...")
        n = len(X)
        oof = np.zeros((n, 12))  # 4 models × 3 classes

        # Time-aware expanding window OOF
        fold_size = n // N_OOF_FOLDS
        for fold in range(N_OOF_FOLDS):
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                continue

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            dt_tr = dates.iloc[:train_end]
            X_val = X.iloc[val_start:val_end]
            y_val = y.iloc[val_start:val_end]
            lg_val = leagues.iloc[val_start:val_end]
            mt_val = meta.iloc[val_start:val_end]

            # Fit base models on expanding training window
            pair = _fit_calibrated_pair(X_tr, y_tr, _time_decay_weights(dt_tr))

            # Get OOF predictions on validation fold
            p_xgb = pair["xgb"].predict_proba(X_val)
            p_lgb = pair["lgbm"].predict_proba(X_val)
            p_cat = pair["cat"].predict_proba(X_val)
            oof[val_start:val_end, 0:3] = p_xgb
            oof[val_start:val_end, 3:6] = p_lgb
            oof[val_start:val_end, 6:9] = p_cat

            # DC probs on validation
            for i in range(len(X_val)):
                slug = lg_val.iloc[i]
                row = mt_val.iloc[i]
                if slug in self.dc:
                    p_h, p_d, p_a = self.dc[slug].predict_proba(
                        row["home_name"], row["away_name"])
                    oof[val_start + i, 9] = p_a
                    oof[val_start + i, 10] = p_d
                    oof[val_start + i, 11] = p_h
                else:
                    avg = (p_xgb[i] + p_lgb[i] + p_cat[i]) / 3
                    oof[val_start + i, 9:12] = avg

        # Only use rows that were in validation folds (non-zero OOF)
        used = oof.sum(axis=1) > 0
        if used.sum() < 50:
            print("    [meta] too few OOF samples, skipping meta-learner")
            return

        self.meta_learner = LogisticRegression(
            C=0.5, max_iter=2000, random_state=42,
            class_weight="balanced", solver="lbfgs")
        self.meta_learner.fit(oof[used], y.values[used])
        print(f"    [meta] fitted on {used.sum()} OOF samples")

    def predict_proba(self, X: pd.DataFrame,
                      league_slug: Optional[str] = None,
                      home_name: Optional[str] = None,
                      away_name: Optional[str] = None) -> np.ndarray:
        pair = self.leagues.get(league_slug) if league_slug else None
        if not pair:
            pair = self.global_model
        p_xgb = pair["xgb"].predict_proba(X)
        p_lgb = pair["lgbm"].predict_proba(X)
        p_cat = pair["cat"].predict_proba(X)

        meta_features = np.zeros((len(X), 12))
        meta_features[:, 0:3] = p_xgb
        meta_features[:, 3:6] = p_lgb
        meta_features[:, 6:9] = p_cat

        if league_slug and home_name and away_name and league_slug in self.dc:
            dc = self.dc[league_slug]
            p_h, p_d, p_a = dc.predict_proba(home_name, away_name)
            meta_features[:, 9] = p_a
            meta_features[:, 10] = p_d
            meta_features[:, 11] = p_h
        else:
            avg = (p_xgb + p_lgb + p_cat) / 3.0
            meta_features[:, 9:12] = avg

        if self.meta_learner is not None:
            return self.meta_learner.predict_proba(meta_features)

        # Fallback: simple average
        avg = (p_xgb + p_lgb + p_cat) / 3.0
        if league_slug and home_name and away_name and league_slug in self.dc:
            dc = self.dc[league_slug]
            p_h, p_d, p_a = dc.predict_proba(home_name, away_name)
            dc_arr = np.array([p_a, p_d, p_h])
            return (1.0 - DC_BLEND_WEIGHT) * avg + DC_BLEND_WEIGHT * dc_arr
        return avg

    def predict(self, X: pd.DataFrame, league_slug: Optional[str] = None,
                home_name: Optional[str] = None,
                away_name: Optional[str] = None) -> np.ndarray:
        return np.argmax(
            self.predict_proba(X, league_slug, home_name, away_name), axis=1)
