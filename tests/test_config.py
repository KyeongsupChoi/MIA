"""Tests for shared configuration."""

from src.config import LABEL_MAPPING, LABEL_NAMES, NUM_CLASSES


def test_label_names_match_mapping():
    """LABEL_NAMES order must match LABEL_MAPPING values."""
    for label_str, idx in LABEL_MAPPING.items():
        expected_name = label_str.strip("[]'")
        assert LABEL_NAMES[idx] == expected_name


def test_num_classes_consistent():
    assert NUM_CLASSES == len(LABEL_MAPPING)
    assert NUM_CLASSES == len(LABEL_NAMES)
