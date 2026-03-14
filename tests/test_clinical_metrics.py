"""Tests for clinical evaluation metrics."""

import numpy as np
import pytest

from src.models.train_Carmine400 import compute_clinical_metrics


class TestComputeClinicalMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_clinical_metrics(y_true, y_pred)
        assert m['sensitivity'] == pytest.approx(1.0)
        assert m['specificity'] == pytest.approx(1.0)
        assert m['ppv'] == pytest.approx(1.0)
        assert m['npv'] == pytest.approx(1.0)

    def test_all_false_negatives(self):
        """Model misses all disease cases — sensitivity should be 0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        m = compute_clinical_metrics(y_true, y_pred)
        assert m['sensitivity'] == pytest.approx(0.0)
        assert m['specificity'] == pytest.approx(1.0)
        assert m['npv'] == pytest.approx(0.5)

    def test_all_false_positives(self):
        """Model flags everything as disease — specificity should be 0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        m = compute_clinical_metrics(y_true, y_pred)
        assert m['sensitivity'] == pytest.approx(1.0)
        assert m['specificity'] == pytest.approx(0.0)
        assert m['ppv'] == pytest.approx(0.5)

    def test_known_confusion_matrix(self):
        """TP=3, FP=1, FN=2, TN=4."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        m = compute_clinical_metrics(y_true, y_pred)
        assert m['sensitivity'] == pytest.approx(3 / 5)
        assert m['specificity'] == pytest.approx(4 / 5)
        assert m['ppv'] == pytest.approx(3 / 4)
        assert m['npv'] == pytest.approx(4 / 6)

    def test_returns_float_values(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_clinical_metrics(y_true, y_pred)
        for key in ('sensitivity', 'specificity', 'ppv', 'npv'):
            assert isinstance(m[key], float)
