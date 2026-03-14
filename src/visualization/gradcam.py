"""Grad-CAM explainability for chest X-ray classification.

Generates class activation maps that highlight which regions of an image
most influenced the model's prediction. Critical for medical imaging where
clinicians need to verify that the model is attending to diagnostically
relevant anatomy rather than spurious artifacts.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import DEFAULT_TARGET_SIZE, LABEL_NAMES, LOG_FORMAT
from src.models.predict_model import load_and_preprocess_image

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last convolutional layer in the model.

    For a Sequential model wrapping ResNet50, looks inside the base model.
    """
    base = model
    # Unwrap Sequential to get the ResNet50 base
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'layers'):
            base = first_layer

    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError('no convolutional layer found in model')


def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: Optional[int] = None,
    conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the given image.

    Args:
        model: Trained Keras model.
        image: Preprocessed image array of shape (1, H, W, 3).
        class_idx: Target class index. If None, uses the predicted class.
        conv_layer_name: Name of the convolutional layer to use. If None,
            automatically finds the last conv layer.

    Returns:
        Heatmap array of shape (H, W) with values in [0, 1].
    """
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)
        logger.info('using conv layer: %s', conv_layer_name)

    # Build a model that outputs both the conv layer activations and
    # the final predictions
    base_model = model.layers[0]
    conv_output = base_model.get_layer(conv_layer_name).output
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_output, base_model.output],
    )

    # Build the full pipeline: base model → dense head
    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(image, training=False)
        # Pass through the dense head layers (after the base model)
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        predictions = x

        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError(
            f'no gradients for layer {conv_layer_name}. '
            'Ensure the layer is part of the computation graph.'
        )

    # Global average pooling of gradients → channel weights
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)

    # ReLU and normalize
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())

    return cam.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image array of shape (H, W, 3) with values in [0, 1].
        heatmap: Heatmap array of shape (h, w) with values in [0, 1].
        alpha: Opacity of the heatmap overlay.
        colormap: OpenCV colormap to apply.

    Returns:
        Blended image of shape (H, W, 3) as uint8.
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(255 * image)
    blended = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return blended


def explain(
    model_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    class_idx: Optional[int] = None,
) -> None:
    """Generate and save a Grad-CAM explanation for a single image.

    Args:
        model_path: Path to the trained .keras model.
        image_path: Path to the input chest X-ray image.
        output_path: Path to save the overlay image.
        target_size: Resize dimensions for the input image.
        class_idx: Target class. None uses the predicted class.
    """
    model = load_model(str(model_path))
    image = load_and_preprocess_image(image_path, target_size)

    heatmap = generate_gradcam(model, image, class_idx=class_idx)
    overlay = overlay_heatmap(image[0], heatmap)

    # Predict for logging
    probs = model.predict(image, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)

    logger.info(
        'Grad-CAM saved to %s (prediction: %s, confidence: %.1f%%)',
        output_path, pred_label, confidence * 100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM explanations for chest X-ray predictions',
    )
    parser.add_argument('image_path', type=str, help='Path to chest X-ray image')
    parser.add_argument(
        '--model-path',
        default=str(PROJECT_DIR / 'models' / 'final_model.keras'),
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for the heatmap overlay (default: <image>_gradcam.png)',
    )
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument(
        '--class-idx', type=int, default=None,
        help='Target class index (default: predicted class)',
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        p = Path(args.image_path)
        output = str(p.parent / f'{p.stem}_gradcam{p.suffix}')

    explain(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=output,
        target_size=(args.image_size, args.image_size),
        class_idx=args.class_idx,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    main()
