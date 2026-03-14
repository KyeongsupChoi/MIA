"""Tests for Grad-CAM explainability utilities."""

import numpy as np
import pytest

from src.visualization.gradcam import overlay_heatmap


class TestOverlayHeatmap:
    def test_output_shape_matches_input(self):
        image = np.random.rand(224, 224, 3).astype(np.float32)
        heatmap = np.random.rand(7, 7).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.shape == (224, 224, 3)

    def test_output_is_uint8(self):
        image = np.random.rand(224, 224, 3).astype(np.float32)
        heatmap = np.random.rand(7, 7).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.dtype == np.uint8

    def test_zero_heatmap_preserves_image(self):
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        heatmap = np.zeros((7, 7), dtype=np.float32)
        result = overlay_heatmap(image, heatmap, alpha=0.4)
        # With zero heatmap, result should be close to the original image
        expected = np.uint8(255 * 0.5 * 0.6)  # (1-alpha) * image
        assert np.allclose(result, expected, atol=5)
