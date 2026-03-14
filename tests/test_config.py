"""Tests for shared configuration."""

import json
import logging

from src.config import LABEL_MAPPING, LABEL_NAMES, NUM_CLASSES, JSONFormatter


def test_label_names_match_mapping():
    """LABEL_NAMES order must match LABEL_MAPPING values."""
    for label_str, idx in LABEL_MAPPING.items():
        expected_name = label_str.strip("[]'")
        assert LABEL_NAMES[idx] == expected_name


def test_num_classes_consistent():
    assert NUM_CLASSES == len(LABEL_MAPPING)
    assert NUM_CLASSES == len(LABEL_NAMES)


def test_json_formatter_produces_valid_json():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name='test', level=logging.INFO, pathname='', lineno=0,
        msg='hello %s', args=('world',), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed['message'] == 'hello world'
    assert parsed['level'] == 'INFO'
    assert 'timestamp' in parsed
