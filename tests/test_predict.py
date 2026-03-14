"""Tests for prediction utilities."""

import pytest
from pathlib import Path

from src.models.predict_model import load_and_preprocess_image


class TestLoadAndPreprocessImage:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match='not found'):
            load_and_preprocess_image('/nonexistent/image.png')

    def test_non_image_raises(self, tmp_path):
        bad_file = tmp_path / 'not_an_image.txt'
        bad_file.write_text('hello')
        with pytest.raises(ValueError, match='cannot decode'):
            load_and_preprocess_image(str(bad_file))
