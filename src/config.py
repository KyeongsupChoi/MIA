"""Shared constants and configuration for the MIA project."""

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
