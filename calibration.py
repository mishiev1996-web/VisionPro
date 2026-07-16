"""
calibration.py — Isotonic regression calibration on OOF predictions.

Fits per-class isotonic calibrators on out-of-fold base model predictions,
then applies them at inference time with renormalization.

Usage:
    calibrator = OofCalibrator()
    calibrator.fit(oof_probas, y_true)   # on OOF predictions from training
    calibrated = calibrator.transform(raw_probas)  # at inference
    calibrator.save("calibrator.pkl")
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class OofCalibrator:
    """Per-class isotonic regression calibrator for 3-class problems.

    Fits one isotonic regressor per class (one-vs-rest), then renormalizes
    so probabilities sum to 1.
    """

    def __init__(self):
        self.calibrators = [None, None, None]  # one per class
        self.n_classes = 3
        self.is_fitted = False

    def fit(self, oof_probas: np.ndarray, y_true: np.ndarray,
            min_samples: int = 200) -> bool:
        """Fit isotonic calibrators on OOF predictions.

        Args:
            oof_probas: (N, 3) array of OOF predicted probabilities
            y_true: (N,) array of true class labels (0, 1, 2)
            min_samples: minimum samples per class to fit isotonic

        Returns:
            True if calibration was fitted, False if skipped (too few data)
        """
        n = len(y_true)
        if n < min_samples:
            print(f"  [calibration] Skipped: only {n} OOF samples (need {min_samples})")
            return False

        y_onehot = np.zeros((n, self.n_classes))
        y_onehot[np.arange(n), y_true.astype(int)] = 1.0

        fitted = 0
        for c in range(self.n_classes):
            n_pos = int(y_onehot[:, c].sum())
            if n_pos < 50:
                print(f"  [calibration] Class {c}: skipped ({n_pos} positives, need 50)")
                continue

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(oof_probas[:, c], y_onehot[:, c])
            self.calibrators[c] = iso
            fitted += 1

        self.is_fitted = fitted > 0
        if self.is_fitted:
            print(f"  [calibration] Isotonic fitted on {n} OOF samples ({fitted}/{self.n_classes} classes)")
        return self.is_fitted

    def transform(self, probas: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration and renormalize.

        Args:
            probas: (N, 3) or (1, 3) raw probabilities

        Returns:
            (N, 3) calibrated probabilities summing to 1
        """
        if not self.is_fitted:
            return probas

        out = np.copy(probas)
        for c in range(self.n_classes):
            if self.calibrators[c] is not None:
                out[:, c] = self.calibrators[c].predict(probas[:, c])

        # Renormalize to sum to 1
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        out = out / row_sums

        return out

    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        import joblib
        joblib.dump(self, path)
        print(f"  [calibration] Saved to {path}")

    @staticmethod
    def load(path: str) -> "OofCalibrator":
        """Load calibrator from disk."""
        import joblib
        return joblib.load(path)


def compare_before_after(raw_probas: np.ndarray, calibrated: np.ndarray,
                         y_true: np.ndarray) -> dict:
    """Compare metrics before and after calibration."""
    from sklearn.metrics import log_loss, brier_score_loss

    n_classes = raw_probas.shape[1]
    eps = 1e-10

    # Log loss
    ll_before = log_loss(y_true, np.clip(raw_probas, eps, 1), labels=list(range(n_classes)))
    ll_after = log_loss(y_true, np.clip(calibrated, eps, 1), labels=list(range(n_classes)))

    # Brier score (macro-averaged over classes)
    brier_before = 0
    brier_after = 0
    for c in range(n_classes):
        y_binary = (y_true == c).astype(float)
        brier_before += brier_score_loss(y_binary, raw_probas[:, c])
        brier_after += brier_score_loss(y_binary, calibrated[:, c])
    brier_before /= n_classes
    brier_after /= n_classes

    # ECE (Expected Calibration Error)
    ece_before = _compute_ece(raw_probas, y_true)
    ece_after = _compute_ece(calibrated, y_true)

    # Accuracy (should barely change)
    acc_before = (np.argmax(raw_probas, axis=1) == y_true).mean() * 100
    acc_after = (np.argmax(calibrated, axis=1) == y_true).mean() * 100

    return {
        "log_loss": {"before": round(ll_before, 4), "after": round(ll_after, 4)},
        "brier": {"before": round(brier_before, 4), "after": round(brier_after, 4)},
        "ece": {"before": round(ece_before, 4), "after": round(ece_after, 4)},
        "accuracy": {"before": round(acc_before, 1), "after": round(acc_after, 1)},
    }


def _compute_ece(probas: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    max_probs = np.max(probas, axis=1)
    preds = np.argmax(probas, axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(max_probs)
    for i in range(n_bins):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_table(probas: np.ndarray, y_true: np.ndarray,
                      n_bins: int = 10) -> list:
    """Build calibration table (same format as backtest)."""
    max_probs = np.max(probas, axis=1)
    preds = np.argmax(probas, axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    for i in range(n_bins):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        table.append({
            "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "mean_predicted": round(float(max_probs[mask].mean()), 3),
            "mean_actual": round(float(correct[mask].mean()), 3),
            "count": int(mask.sum()),
            "gap": round(abs(float(max_probs[mask].mean()) - float(correct[mask].mean())), 3),
        })
    return table
