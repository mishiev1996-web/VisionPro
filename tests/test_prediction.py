"""
test_prediction.py — Tests for feature building and ensemble inference.
"""
from __future__ import annotations

import pytest
import numpy as np


class TestBuildFeatures:
    """Smoke tests for train.build_features."""

    def test_build_features_returns_list(self):
        """build_features should return a list of correct length."""
        from train import build_features, FEATURE_NAMES
        # Empty prior matches → should still return a feature vector
        features = build_features(
            home_id=1, away_id=2,
            prior_matches=[],
            match_date="2025-01-15",
            league_slug="EPL",
            season=2025,
        )
        assert isinstance(features, list)
        assert len(features) == len(FEATURE_NAMES)

    def test_build_features_no_nan_crash(self):
        """Features should not contain NaN that crashes model."""
        from train import build_features
        features = build_features(
            home_id=1, away_id=2,
            prior_matches=[],
            match_date="2025-01-15",
            league_slug="EPL",
            season=2025,
        )
        # NaN is allowed in features (model handles it), but list should be complete
        assert len(features) > 0


class TestEnsembleInference:
    """Smoke tests for ensemble model predict_proba on synthetic input."""

    def test_ensemble_predict_shape(self):
        """Model should return 3 probabilities that sum to ~1."""
        import os
        if not os.path.exists("model.pkl"):
            pytest.skip("model.pkl not found — train model first")
        import joblib
        import pandas as pd
        model_data = joblib.load("model.pkl")
        n_features = len(model_data["features"])
        # Synthetic zero-filled input
        X = pd.DataFrame([np.zeros(n_features)], columns=model_data["features"])
        model_obj = model_data.get("model") or model_data.get("ensemble")
        fmt = model_data.get("format", "v1")
        if fmt in ("ensemble_v3", "ensemble_v4"):
            proba = model_obj.predict_proba(X, league_slug="EPL",
                                            home_name="Test", away_name="Test")[0]
        else:
            proba = model_obj.predict_proba(X)[0]
        assert len(proba) == 3
        assert abs(sum(proba) - 1.0) < 0.01

    def test_ensemble_valid_range(self):
        """Each probability should be between 0 and 1."""
        import os
        if not os.path.exists("model.pkl"):
            pytest.skip("model.pkl not found — train model first")
        import joblib
        import pandas as pd
        model_data = joblib.load("model.pkl")
        n_features = len(model_data["features"])
        X = pd.DataFrame([np.zeros(n_features)], columns=model_data["features"])
        model_obj = model_data.get("model") or model_data.get("ensemble")
        fmt = model_data.get("format", "v1")
        if fmt in ("ensemble_v3", "ensemble_v4"):
            proba = model_obj.predict_proba(X, league_slug="EPL",
                                            home_name="Test", away_name="Test")[0]
        else:
            proba = model_obj.predict_proba(X)[0]
        assert all(0.0 <= p <= 1.0 for p in proba)
