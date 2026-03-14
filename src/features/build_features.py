import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img


PROJECT_DIR = Path(__file__).resolve().parents[2]

LABEL_MAPPING = {
    "['normal']": 0,
    "['pneumonia']": 1,
}


def load_images_and_labels(csv_path, image_dir, target_size=(224, 224)):
    """Load images from disk and pair with encoded labels.

    Args:
        csv_path: Path to CSV with ImageID and Labels columns.
        image_dir: Directory containing the image files.
        target_size: Resize dimensions (height, width).

    Returns:
        X: numpy array of shape (N, H, W, 3), normalized to [0, 1].
        y: numpy array of integer labels, shape (N,).
        skipped: list of image paths that could not be loaded.
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)

    X, y, skipped = [], [], []

    for _, row in df.iterrows():
        img_path = image_dir / row['ImageID']

        if not img_path.exists():
            skipped.append(str(img_path))
            continue

        label_str = row['Labels']
        if label_str not in LABEL_MAPPING:
            skipped.append(str(img_path))
            continue

        img = load_img(str(img_path), target_size=target_size)
        img_array = img_to_array(img) / 255.0
        X.append(img_array)
        y.append(LABEL_MAPPING[label_str])

    if skipped:
        logger.warning(f'skipped {len(skipped)} images (missing or bad label)')

    return np.array(X), np.array(y), skipped


def compute_class_weights(y):
    """Compute balanced class weights for imbalanced datasets.

    Returns a dict mapping class index to weight, where
    under-represented classes get higher weight.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {
        int(cls): total / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
    return weights
