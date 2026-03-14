import argparse
import logging
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


PROJECT_DIR = Path(__file__).resolve().parents[2]

LABEL_NAMES = ['normal', 'pneumonia']


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load a single image and preprocess for model input."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, image_path, target_size=(224, 224)):
    """Predict label and confidence for a single image.

    Returns:
        dict with 'label', 'confidence', and per-class 'probabilities'.
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


def predict_batch(model, image_paths, target_size=(224, 224)):
    """Predict labels for multiple images at once."""
    images = []
    valid_paths = []

    for path in image_paths:
        if not Path(path).exists():
            logging.warning(f'image not found: {path}')
            continue
        img = load_and_preprocess_image(path, target_size)
        images.append(img[0])
        valid_paths.append(path)

    if not images:
        return []

    batch = np.array(images)
    all_probs = model.predict(batch, verbose=0)

    results = []
    for path, probs in zip(valid_paths, all_probs):
        idx = int(np.argmax(probs))
        results.append({
            'image': str(path),
            'label': LABEL_NAMES[idx],
            'confidence': float(probs[idx]),
            'probabilities': {
                name: float(p) for name, p in zip(LABEL_NAMES, probs)
            },
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict pneumonia on chest X-ray images'
    )
    parser.add_argument(
        'image_paths', nargs='+', type=str,
        help='Path(s) to image file(s) to classify',
    )
    parser.add_argument(
        '--model-path',
        default=str(PROJECT_DIR / 'models' / 'final_model.keras'),
        help='Path to trained model file',
    )
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    model = load_model(args.model_path)
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
    main()
