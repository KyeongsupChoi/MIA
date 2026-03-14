"""Image loading and feature engineering for chest X-ray classification."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.config import DEFAULT_TARGET_SIZE, LABEL_MAPPING

logger = logging.getLogger(__name__)


def load_images_and_labels(
    csv_path: str | Path,
    image_dir: str | Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images from disk and pair with encoded labels.

    Images that are missing or corrupt are skipped with a warning rather
    than crashing the pipeline.

    Args:
        csv_path: Path to CSV with ImageID and Labels columns.
        image_dir: Directory containing the image files.
        target_size: Resize dimensions (height, width).

    Returns:
        X: numpy array of shape (N, H, W, 3), normalized to [0, 1].
        y: numpy array of integer labels, shape (N,).
        skipped: list of image paths that could not be loaded.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If no valid images could be loaded.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    if not image_dir.is_dir():
        raise FileNotFoundError(f'image directory not found: {image_dir}')

    df = pd.read_csv(csv_path)

    for col in ('ImageID', 'Labels'):
        if col not in df.columns:
            raise ValueError(f'CSV missing required column: {col}')

    X: List[np.ndarray] = []
    y: List[int] = []
    skipped: List[str] = []

    for _, row in df.iterrows():
        img_path = image_dir / row['ImageID']

        if not img_path.exists():
            skipped.append(f'{img_path} (missing)')
            continue

        label_str = row['Labels']
        if label_str not in LABEL_MAPPING:
            skipped.append(f'{img_path} (unknown label: {label_str})')
            continue

        try:
            img = load_img(str(img_path), target_size=target_size)
            img_array = img_to_array(img) / 255.0
        except Exception as exc:
            skipped.append(f'{img_path} (corrupt: {exc})')
            continue

        X.append(img_array)
        y.append(LABEL_MAPPING[label_str])

    if skipped:
        logger.warning('skipped %d images:', len(skipped))
        for s in skipped[:10]:
            logger.warning('  %s', s)
        if len(skipped) > 10:
            logger.warning('  ... and %d more', len(skipped) - 10)

    if not X:
        raise ValueError(
            f'no valid images loaded from {csv_path} + {image_dir}. '
            f'All {len(skipped)} entries were skipped.'
        )

    logger.info('loaded %d images, skipped %d', len(X), len(skipped))
    return np.array(X), np.array(y), skipped


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced datasets.

    Uses the sklearn "balanced" formula: n_samples / (n_classes * count).
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        int(cls): total / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
