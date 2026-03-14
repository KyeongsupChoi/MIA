"""Shared constants and configuration for the MIA project."""

import json
import logging
from pathlib import Path
from typing import Dict, List

PROJECT_DIR: Path = Path(__file__).resolve().parents[1]

# Label encoding — single source of truth
LABEL_MAPPING: Dict[str, int] = {
    "['normal']": 0,
    "['pneumonia']": 1,
}

LABEL_NAMES: List[str] = ['normal', 'pneumonia']

NUM_CLASSES: int = len(LABEL_MAPPING)

# Image preprocessing defaults
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_TARGET_SIZE = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

# Required columns in the raw dataset
REQUIRED_COLUMNS = {'ImageID', 'Labels', 'Projection', 'Pediatric'}

LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def configure_logging(json_logs: bool = False, level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        json_logs: If True, use structured JSON logging (production).
                   If False, use human-readable format (development).
        level: Logging level.
    """
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
