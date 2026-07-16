"""models/ensemble.py — ensemble classifier with proper OOF stacking,
per-league model selection via log-loss, and per-league calibration.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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
import config

# Re-export from config for backward compatibility
USE_CLASS_WEIGHTS = config.USE_CLASS_WEIGHTS
PER_LEAGUE_PROVES_ITSELF = config.PER_LEAGUE_PROVES_ITSELF
MIN_LEAGUE_TRAIN_ROWS = config.MIN_LEAGUE_TRAIN_ROWS
TIME_DECAY_HALF_LIFE_DAYS = config.TIME_DECAY_HALF_LIFE_DAYS
N_OOF_FOLDS = config.N_OOF_FOLDS


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
# Parallel Dixon-Coles: threads ok (numpy/scipy release GIL). Cap workers
# so we leave headroom for the OS; biggest leagues dominate wall time anyway.
DC_MAX_WORKERS = max(1, min(6, (os.cpu_count() or 4) - 1))


def _fit_dc_league(
    slug: str,
    matches: List[dict],
    ref: str,
    dc_xi: float,
) -> Tuple[str, DixonColes, float]:
    """Worker for parallel per-league Dixon-Coles fit. Returns (slug, model, secs)."""
    t0 = time.time()
    print(
        f"    [{slug}] Dixon-Coles START — {len(matches)} matches "
        f"(xi={dc_xi:.4f})",
        flush=True,
    )
    dc = DixonColes(decay_xi=dc_xi).fit(
        matches, reference_date=ref, progress_label=slug
    )
    elapsed = time.time() - t0
    status = "ok" if dc.fitted else "FAILED"
    print(
        f"    [{slug}] Dixon-Coles DONE [{status}] in {elapsed:.1f}s",
        flush=True,
    )
    return slug, dc, elapsed


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
        random_state=42, verbosity=0, n_jobs=-1,
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
        force_col_wise=True, n_jobs=-1,
    )


def _make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=max(TUNED.get("cat_iterations", 500), 300),
        learning_rate=TUNED.get("cat_lr", 0.03),
        depth=TUNED.get("cat_depth", 4),
        l2_leaf_reg=TUNED.get("cat_l2", 3.0),
        random_seed=42,
        verbose=0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        thread_count=-1,
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
            out[i] = max(0.05, math.exp(-decay * days))
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
    with proper k-fold OOF stacking meta-learner and temperature scaling.

    Class label convention: 0=Away Win, 1=Draw, 2=Home Win.
    """
    def __init__(self):
        self.global_model: Dict = {}
        self.leagues: Dict[str, Dict] = {}
        self.dc: Dict[str, DixonColes] = {}
        self.meta_learner = None
        self.temperature = 1.0  # learned temperature for calibration
        self.calibrator = None  # isotonic calibrator on OOF predictions

    def fit(self, X: pd.DataFrame, y: pd.Series, leagues: pd.Series,
            dates: pd.Series, meta: pd.DataFrame) -> None:
        fit_t0 = time.time()
        # Use last training date as reference, not "today" — otherwise time decay
        # is meaningless for CV folds where training data is historical.
        last_date = str(dates.iloc[-1])[:10]
        decay_w = _time_decay_weights(dates, reference=last_date)
        print(
            f"    time-decay (ref={last_date}): oldest weight={decay_w.min():.3f}, "
            f"newest={decay_w.max():.3f}",
            flush=True,
        )

        # Global XGB+LGB+CatBoost
        print(
            "    [global] fit XGB + LightGBM + CatBoost (calibrated, time-decayed)...",
            flush=True,
        )
        t_trees = time.time()
        self.global_model = _fit_calibrated_pair(X, y, decay_w)
        print(
            f"    [global] trees done in {time.time() - t_trees:.1f}s",
            flush=True,
        )

        # Dixon-Coles per league (parallel) with per-league decay_xi
        dc_jobs: List[Tuple[str, List[dict], str, float]] = []
        for slug in sorted(set(leagues)):
            mask = (leagues == slug)
            sub = meta[mask]
            sub_dates = dates[mask].reset_index(drop=True)
            matches = [
                {
                    "home": r["home_name"],
                    "away": r["away_name"],
                    "home_goals": int(r["home_goals"] or 0),
                    "away_goals": int(r["away_goals"] or 0),
                    "date": d,
                }
                for (_, r), d in zip(sub.iterrows(), sub_dates)
            ]
            if len(matches) < 200:
                continue
            try:
                ref = max(str(d)[:10] for d in sub_dates if d)
            except ValueError:
                continue
            dc_xi = TUNED.get(f"dc_xi_{slug}", TUNED.get("dc_xi", 0.0065))
            dc_jobs.append((slug, matches, ref, float(dc_xi)))

        # Heaviest leagues first so workers stay busy longer
        dc_jobs.sort(key=lambda j: len(j[1]), reverse=True)
        n_workers = min(DC_MAX_WORKERS, max(1, len(dc_jobs)))
        print(
            f"    [DC] fitting {len(dc_jobs)} leagues in parallel "
            f"(workers={n_workers}, cpu={os.cpu_count()})...",
            flush=True,
        )
        for slug, matches, _, _ in dc_jobs:
            print(f"      queue: {slug} ({len(matches)} matches)", flush=True)

        t_dc = time.time()
        done = 0
        if dc_jobs:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_fit_dc_league, slug, matches, ref, xi): slug
                    for slug, matches, ref, xi in dc_jobs
                }
                for fut in as_completed(futures):
                    slug = futures[fut]
                    try:
                        slug_out, dc, _elapsed = fut.result()
                        if dc.fitted:
                            self.dc[slug_out] = dc
                        done += 1
                        print(
                            f"    [DC] progress {done}/{len(dc_jobs)} leagues "
                            f"(last={slug_out}, fitted_ok={dc.fitted})",
                            flush=True,
                        )
                    except Exception as e:
                        done += 1
                        print(f"    [DC] {slug} failed: {e}", flush=True)
        print(
            f"    [DC] all leagues finished in {time.time() - t_dc:.1f}s "
            f"({len(self.dc)} models kept)",
            flush=True,
        )

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
            w_in = _time_decay_weights(d_lg.iloc[:sp], reference=str(d_lg.iloc[sp-1])[:10])
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
                w_full = _time_decay_weights(d_lg, reference=str(d_lg.iloc[-1])[:10])
                self.leagues[slug] = _fit_calibrated_pair(X_lg, y_lg, w_full)
                print(f"    [{slug}] booster kept (cand LL={cand_ll:.4f} vs glob LL={glob_ll:.4f})")

        # Fit stacking meta-learner with proper k-fold OOF
        print("    [meta] starting OOF stacking...", flush=True)
        oof_probas = self._fit_meta_learner(X, y, leagues, dates, meta)

        # Learn temperature for calibration on OOF predictions (not in-sample!)
        self._fit_temperature_from_oof(oof_probas, y)

        # Fit isotonic calibrator on OOF predictions
        self._fit_isotonic_from_oof(oof_probas, y)
        print(
            f"    [Ensemble.fit] complete in {time.time() - fit_t0:.1f}s",
            flush=True,
        )

    def _fit_meta_learner(self, X: pd.DataFrame, y: pd.Series,
                          leagues: pd.Series, dates: pd.Series,
                          meta: pd.DataFrame) -> np.ndarray:
        """Fit stacking meta-learner on k-fold OOF predictions from base models.
        Returns OOF predictions for temperature calibration."""
        print(
            f"    [meta] fitting stacking meta-learner "
            f"({N_OOF_FOLDS}-fold OOF)...",
            flush=True,
        )
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

            print(
                f"    [meta] fold {fold + 1}/{N_OOF_FOLDS}: "
                f"train={len(X_tr)}, val={len(X_val)} — fitting trees...",
                flush=True,
            )
            t_fold = time.time()
            # Fit base models on expanding training window
            ref_tr = str(dt_tr.iloc[-1])[:10]
            pair = _fit_calibrated_pair(
                X_tr, y_tr, _time_decay_weights(dt_tr, reference=ref_tr)
            )
            print(
                f"    [meta] fold {fold + 1}/{N_OOF_FOLDS}: trees done in "
                f"{time.time() - t_fold:.1f}s",
                flush=True,
            )

            # Get OOF predictions on validation fold
            p_xgb = pair["xgb"].predict_proba(X_val)
            p_lgb = pair["lgbm"].predict_proba(X_val)
            p_cat = pair["cat"].predict_proba(X_val)
            oof[val_start:val_end, 0:3] = p_xgb
            oof[val_start:val_end, 3:6] = p_lgb
            oof[val_start:val_end, 6:9] = p_cat

            # DC probs on validation — using pre-fitted self.dc
            # (minor in-sample overlap: DC was fit on all training data, but it's
            # only 1/12 meta-features and the meta-learner re-weights accordingly)
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

        # Save OOF for external calibration (not in-sample!)
        self.oof_probas_ = oof[used].copy()
        self.oof_y_ = y.values[used].copy()

        return oof

    def _fit_temperature_from_oof(self, oof: np.ndarray, y: pd.Series) -> None:
        """Learn optimal temperature for probability calibration using OOF predictions.
        
        Temperature scaling: p_i = softmax(logit_i / T)
        T > 1 → softer (less confident), T < 1 → sharper (more confident).
        Learned on OOF predictions (not in-sample) to avoid overfitting.
        """
        # Only use rows that have OOF predictions (non-zero)
        used = oof.sum(axis=1) > 0
        if used.sum() < 100:
            return  # not enough data

        oof_used = oof[used]
        y_used = y.values[used]

        # Get meta-learner probabilities on OOF
        probas = self.meta_learner.predict_proba(oof_used)

        # Vectorized temperature optimization
        logits = np.log(np.maximum(probas, 1e-10))
        y_onehot = np.zeros_like(probas)
        y_onehot[np.arange(len(y_used)), y_used.astype(int)] = 1.0

        best_T = 1.0
        best_nll = float("inf")
        for T in np.arange(0.5, 2.01, 0.05):
            scaled = np.exp(logits / T)
            scaled = scaled / scaled.sum(axis=1, keepdims=True)
            nll = -np.mean(np.sum(y_onehot * np.log(np.maximum(scaled, 1e-10)), axis=1))
            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        print(f"    [calibration] temperature T={best_T:.2f} (NLL={best_nll:.4f})")

    def _fit_isotonic_from_oof(self, oof: np.ndarray, y: pd.Series) -> None:
        """Fit isotonic calibrator on OOF predictions from base models.

        Uses the raw OOF outputs (before meta-learner) — per-class isotonic
        regression to correct systematic over/under-confidence.
        """
        from calibration import OofCalibrator

        used = oof.sum(axis=1) > 0
        if used.sum() < 200:
            print("    [calibration] Isotonic skipped: too few OOF samples")
            return

        # Use average of base model OOF predictions (columns 0-8: xgb, lgbm, cat)
        base_oof = (oof[used, 0:3] + oof[used, 3:6] + oof[used, 6:9]) / 3.0

        self.calibrator = OofCalibrator()
        self.calibrator.fit(base_oof, y.values[used])

    def predict_proba(self, X: pd.DataFrame,
                      league_slug: Optional[str] = None,
                      home_name: Optional[str] = None,
                      away_name: Optional[str] = None,
                      apply_calibration: bool = True) -> np.ndarray:
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
            raw = self.meta_learner.predict_proba(meta_features)
        else:
            # Fallback: simple average
            avg = (p_xgb + p_lgb + p_cat) / 3.0
            if league_slug and home_name and away_name and league_slug in self.dc:
                dc = self.dc[league_slug]
                p_h, p_d, p_a = dc.predict_proba(home_name, away_name)
                dc_arr = np.array([p_a, p_d, p_h])
                raw = (1.0 - DC_BLEND_WEIGHT) * avg + DC_BLEND_WEIGHT * dc_arr
            else:
                raw = avg

        # Apply temperature scaling for calibration
        if apply_calibration and self.temperature != 1.0 and self.temperature > 0:
            logits = np.log(np.maximum(raw, 1e-10))
            scaled = np.exp(logits / self.temperature)
            raw = scaled / scaled.sum(axis=1, keepdims=True)

        # Apply isotonic calibration (per-class, then renormalize)
        if apply_calibration and getattr(self, 'calibrator', None) is not None and self.calibrator.is_fitted:
            raw = self.calibrator.transform(raw)

        return raw

    def predict(self, X: pd.DataFrame, league_slug: Optional[str] = None,
                home_name: Optional[str] = None,
                away_name: Optional[str] = None) -> np.ndarray:
        return np.argmax(
            self.predict_proba(X, league_slug, home_name, away_name), axis=1)
