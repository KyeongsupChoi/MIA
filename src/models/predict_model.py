"""Inference module for chest X-ray classification."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.config import DEFAULT_TARGET_SIZE, LABEL_NAMES, LOG_FORMAT

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_and_preprocess_image(
    image_path: str | Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Load and normalize a single image for model input.

    Returns:
        Array of shape (1, H, W, 3) with values in [0, 1].

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image cannot be decoded.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'image not found: {image_path}')

    try:
        img = load_img(str(image_path), target_size=target_size)
    except Exception as exc:
        raise ValueError(f'cannot decode image {image_path}: {exc}') from exc

    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(
    model,
    image_path: str | Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Dict[str, Any]:
    """Predict label and confidence for a single image.

    Returns:
        Dict with 'label', 'confidence', and per-class 'probabilities'.
    """
    img_array = load_and_preprocess_image(image_path, target_size)
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))

    return {
        'label': LABEL_NAMES[predicted_idx],
        'confidence': float(probabilities[predicted_idx]),
        'probabilities': {
            name: float(prob)
            for name, prob in zip(LABEL_NAMES, probabilities)
        },
    }


def predict_batch(
    model,
    image_paths: List[str | Path],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> List[Dict[str, Any]]:
    """Predict labels for multiple images in a single forward pass."""
    images: List[np.ndarray] = []
    valid_paths: List[str] = []

    for path in image_paths:
        try:
            img = load_and_preprocess_image(path, target_size)
            images.append(img[0])
            valid_paths.append(str(path))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning('skipping %s: %s', path, exc)

    if not images:
        return []

    batch = np.array(images)
    all_probs = model.predict(batch, verbose=0)

    results = []
    for path, probs in zip(valid_paths, all_probs):
        idx = int(np.argmax(probs))
        results.append({
            'image': path,
            'label': LABEL_NAMES[idx],
            'confidence': float(probs[idx]),
            'probabilities': {
                name: float(p) for name, p in zip(LABEL_NAMES, probs)
            },
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict pneumonia on chest X-ray images',
    )
    parser.add_argument(
        'image_paths', nargs='+', type=str,
        help='Path(s) to image file(s) to classify',
    )
    parser.add_argument(
        '--model-path',
        default=str(PROJECT_DIR / 'models' / 'final_model.keras'),
    )
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'model not found at {model_path}. Run training first.'
        )

    model = load_model(str(model_path))
    target_size = (args.image_size, args.image_size)

    if len(args.image_paths) == 1:
        result = predict(model, args.image_paths[0], target_size)
        print(f"Prediction: {result['label']} "
              f"(confidence: {result['confidence']:.1%})")
        for name, prob in result['probabilities'].items():
            print(f"  {name}: {prob:.1%}")
    else:
        results = predict_batch(model, args.image_paths, target_size)
        for r in results:
            print(f"{r['image']}: {r['label']} ({r['confidence']:.1%})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    main()
