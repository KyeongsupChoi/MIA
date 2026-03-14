"""Two-phase transfer-learning trainer for chest X-ray classification.

Phase 1: Train a classification head on top of a frozen ResNet50 backbone.
Phase 2: Unfreeze the top layers of the backbone and fine-tune end-to-end
          with a reduced learning rate.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from src.config import LABEL_MAPPING, LABEL_NAMES, LOG_FORMAT, NUM_CLASSES
from src.features.build_features import compute_class_weights, load_images_and_labels

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_augmenter() -> ImageDataGenerator:
    """Build augmentations suited for chest X-rays.

    Uses geometric transforms only — no color jitter, because grayscale
    radiographs have diagnostically meaningful pixel intensities.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
    )


def create_model(
    input_shape: tuple,
    num_classes: int,
    fine_tune_from: Optional[int] = None,
) -> Sequential:
    """Create a ResNet50 transfer-learning model.

    Args:
        input_shape: (H, W, C).
        num_classes: Number of output classes.
        fine_tune_from: Layer index in the base model from which to unfreeze.
            None keeps the entire base frozen.
    """
    base_model = ResNet50(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet',
    )

    if fine_tune_from is not None:
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_from:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    return model


def build_callbacks(checkpoint_path: Path) -> List[Callback]:
    """Build training callbacks."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
    ]


def evaluate_model(
    model: Sequential,
    X: np.ndarray,
    y_true_onehot: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """Evaluate a model and return structured metrics.

    Prints a human-readable report and returns a JSON-serializable dict.
    """
    loss, accuracy, auc = model.evaluate(X, y_true_onehot, verbose=0)

    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nLoss: {loss:.4f}  Accuracy: {accuracy:.4f}  '
          f'AUC: {auc:.4f}  ROC-AUC: {roc_auc:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f'Confusion Matrix:\n{cm}')

    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'auc': float(auc),
        'roc_auc': float(roc_auc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }


def save_training_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Persist the full training configuration for reproducibility."""
    config = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'args': vars(args),
        'label_mapping': LABEL_MAPPING,
        'tensorflow_version': tf.__version__,
    }
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info('training config saved to %s', config_path)


def train(args: argparse.Namespace) -> None:
    """Run the full two-phase training pipeline."""
    set_seeds(args.seed)

    input_shape = (args.image_size, args.image_size, 3)
    target_size = (args.image_size, args.image_size)

    # --- Load data via shared feature module ---
    X, y, skipped = load_images_and_labels(
        args.data_csv, args.image_dir, target_size=target_size,
    )
    logger.info('loaded %d images (%d skipped)', len(X), len(skipped))

    class_weight = compute_class_weights(y)
    logger.info('class weights: %s', class_weight)

    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # --- Three-way stratified split ---
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y_onehot,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    y_tv_labels = np.argmax(y_tv, axis=1)
    val_fraction = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=y_tv_labels,
    )
    logger.info(
        'splits — train: %d, val: %d, test: %d',
        len(X_train), len(X_val), len(X_test),
    )

    # --- Output setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_config(args, output_dir)

    # --- Data augmentation ---
    augmenter = build_augmenter()
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    train_gen = augmenter.flow(X_train, y_train, batch_size=args.batch_size)

    # === Phase 1: Frozen backbone ===
    logger.info('Phase 1: training with frozen ResNet50 base')
    model = create_model(input_shape, NUM_CLASSES, fine_tune_from=None)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    cp_phase1 = output_dir / 'best_model_phase1.keras'
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_phase1,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(cp_phase1),
        verbose=1,
    )

    # === Phase 2: Unfreeze top ResNet layers on the same model ===
    logger.info('Phase 2: fine-tuning top ResNet50 layers')
    base_model = model.layers[0]
    for layer in base_model.layers[140:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    cp_phase2 = output_dir / 'best_model_finetuned.keras'
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_phase2,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(cp_phase2),
        verbose=1,
    )

    # === Evaluate on held-out test set ===
    logger.info('evaluating on held-out test set')
    print('\n' + '=' * 50)
    print('TEST SET EVALUATION')
    print('=' * 50)
    metrics = evaluate_model(model, X_test, y_test, LABEL_NAMES)

    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info('metrics saved to %s', metrics_path)

    final_path = output_dir / 'final_model.keras'
    model.save(str(final_path))
    logger.info('final model saved to %s', final_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train chest X-ray classifier (normal vs pneumonia)',
    )
    parser.add_argument(
        '--data-csv',
        default=str(PROJECT_DIR / 'data' / 'processed' / 'balanced.csv'),
    )
    parser.add_argument(
        '--image-dir',
        default=str(PROJECT_DIR / 'data' / 'raw' / 'img'),
    )
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_DIR / 'models'),
    )
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-phase1', type=int, default=20)
    parser.add_argument('--epochs-phase2', type=int, default=15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train(parse_args())
