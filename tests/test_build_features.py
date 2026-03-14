"""Tests for feature engineering utilities."""

import numpy as np
import pytest

from src.features.build_features import compute_class_weights


class TestComputeClassWeights:
    def test_balanced_classes(self):
        y = np.array([0, 0, 1, 1])
        weights = compute_class_weights(y)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(1.0)

    def test_imbalanced_classes(self):
        y = np.array([0, 0, 0, 1])
        weights = compute_class_weights(y)
        # Minority class should have higher weight
        assert weights[1] > weights[0]
        # Formula: n_samples / (n_classes * n_samples_for_class)
        assert weights[0] == pytest.approx(4.0 / (2 * 3))
        assert weights[1] == pytest.approx(4.0 / (2 * 1))

    def test_single_class(self):
        y = np.array([0, 0, 0])
        weights = compute_class_weights(y)
        assert weights[0] == pytest.approx(1.0)

    def test_returns_int_keys(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y)
        for key in weights:
            assert isinstance(key, int)
