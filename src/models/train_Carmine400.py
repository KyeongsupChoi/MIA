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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Compute clinically relevant metrics for binary classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_label: Index of the positive (disease) class.

    Returns:
        Dict with sensitivity, specificity, PPV, and NPV.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
    }


def evaluate_model(
    model: Sequential,
    X: np.ndarray,
    y_true_onehot: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """Evaluate a model and return structured metrics.

    Computes standard ML metrics (accuracy, AUC) plus clinically relevant
    metrics (sensitivity, specificity, PPV, NPV) critical for medical imaging.
    """
    loss, accuracy, auc = model.evaluate(X, y_true_onehot, verbose=0)

    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    clinical = compute_clinical_metrics(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nLoss: {loss:.4f}  Accuracy: {accuracy:.4f}  '
          f'AUC: {auc:.4f}  ROC-AUC: {roc_auc:.4f}')
    print(f"Sensitivity: {clinical['sensitivity']:.4f}  "
          f"Specificity: {clinical['specificity']:.4f}")
    print(f"PPV: {clinical['ppv']:.4f}  NPV: {clinical['npv']:.4f}")
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f'Confusion Matrix:\n{cm}')

    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'auc': float(auc),
        'roc_auc': float(roc_auc),
        'sensitivity': clinical['sensitivity'],
        'specificity': clinical['specificity'],
        'ppv': clinical['ppv'],
        'npv': clinical['npv'],
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


def _train_single_split(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: Dict[int, float],
    output_dir: Path,
    fold_label: str = '',
) -> Sequential:
    """Train a single model through both phases. Returns the trained model."""
    input_shape = (args.image_size, args.image_size, 3)

    augmenter = build_augmenter()
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    train_gen = augmenter.flow(X_train, y_train, batch_size=args.batch_size)

    prefix = f'fold{fold_label}_' if fold_label else ''

    # === Phase 1: Frozen backbone ===
    logger.info('%sPhase 1: training with frozen ResNet50 base', prefix)
    model = create_model(input_shape, NUM_CLASSES, fine_tune_from=None)
    model.compile(
        optimizer=Adam(learning_rate=args.lr_phase1),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    cp_phase1 = output_dir / f'{prefix}best_model_phase1.keras'
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_phase1,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(cp_phase1),
        verbose=1,
    )

    # === Phase 2: Unfreeze top ResNet layers ===
    logger.info('%sPhase 2: fine-tuning top ResNet50 layers', prefix)
    base_model = model.layers[0]
    for layer in base_model.layers[args.fine_tune_from:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=args.lr_phase2),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    cp_phase2 = output_dir / f'{prefix}best_model_finetuned.keras'
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_phase2,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(cp_phase2),
        verbose=1,
    )

    return model


def cross_validate(args: argparse.Namespace) -> None:
    """Run stratified k-fold cross-validation to estimate model performance.

    Useful when the dataset is small (e.g., 400 samples per class) and a
    single train/test split may not give reliable performance estimates.
    """
    set_seeds(args.seed)
    target_size = (args.image_size, args.image_size)

    X, y, skipped = load_images_and_labels(
        args.data_csv, args.image_dir, target_size=target_size,
    )
    logger.info('loaded %d images (%d skipped)', len(X), len(skipped))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_config(args, output_dir)

    skf = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed,
    )

    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info('=== Fold %d/%d ===', fold_idx, args.cv_folds)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]

        class_weight = compute_class_weights(y_train_raw)
        y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
        y_val = to_categorical(y_val_raw, num_classes=NUM_CLASSES)

        model = _train_single_split(
            args, X_train, y_train, X_val, y_val,
            class_weight, output_dir, fold_label=str(fold_idx),
        )

        metrics = evaluate_model(model, X_val, y_val, LABEL_NAMES)
        metrics['fold'] = fold_idx
        fold_metrics.append(metrics)

    # --- Aggregate results ---
    agg_keys = [
        'accuracy', 'roc_auc', 'sensitivity', 'specificity', 'ppv', 'npv',
    ]
    summary: Dict[str, Any] = {'folds': fold_metrics}
    for key in agg_keys:
        values = [m[key] for m in fold_metrics]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    print('\n' + '=' * 50)
    print(f'CROSS-VALIDATION SUMMARY ({args.cv_folds} folds)')
    print('=' * 50)
    for key in agg_keys:
        print(f"  {key}: {summary[f'{key}_mean']:.4f} "
              f"± {summary[f'{key}_std']:.4f}")

    cv_path = output_dir / 'cv_metrics.json'
    with open(cv_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('cross-validation metrics saved to %s', cv_path)


def train(args: argparse.Namespace) -> None:
    """Run the full two-phase training pipeline."""
    if args.cv_folds > 1:
        cross_validate(args)
        return

    set_seeds(args.seed)

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

    model = _train_single_split(
        args, X_train, y_train, X_val, y_val,
        class_weight, output_dir,
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
    # Data / IO
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
    # Architecture / training
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-phase1', type=int, default=20)
    parser.add_argument('--epochs-phase2', type=int, default=15)
    parser.add_argument('--lr-phase1', type=float, default=1e-3,
                        help='Learning rate for Phase 1 (frozen backbone)')
    parser.add_argument('--lr-phase2', type=float, default=1e-5,
                        help='Learning rate for Phase 2 (fine-tuning)')
    parser.add_argument('--fine-tune-from', type=int, default=140,
                        help='ResNet50 layer index to unfreeze from')
    # Split / validation
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--cv-folds', type=int, default=1,
                        help='Number of CV folds (1 = no CV, >1 = k-fold)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    train(parse_args())


if __name__ == '__main__':
    main()
