import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Resolve project root relative to this file
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Labels
LABEL_MAPPING = {
    "['normal']": 0,
    "['pneumonia']": 1,
}
LABEL_NAMES = ['normal', 'pneumonia']


def set_seeds(seed=42):
    """Set seeds for reproducibility across numpy, tf, and python."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_augmenter():
    """Build an ImageDataGenerator with augmentations suited for chest X-rays.

    Medical images benefit from geometric transforms but NOT color jitter
    (grayscale radiographs have meaningful intensity values).
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
    )


def create_model(input_shape, num_classes, fine_tune_from=None):
    """Create a ResNet50 transfer-learning model.

    Args:
        input_shape: Tuple (H, W, C).
        num_classes: Number of output classes.
        fine_tune_from: If set, unfreeze all layers from this index onward
            in the base model. None keeps everything frozen.

    Returns:
        Compiled Keras Sequential model.
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


def build_callbacks(checkpoint_path):
    """Build training callbacks: early stopping, LR reduction, checkpointing."""
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


def evaluate_model(model, X, y_true_onehot, label_names):
    """Run full evaluation: loss, accuracy, AUC, classification report,
    and confusion matrix.

    Returns a dict of metrics.
    """
    loss, accuracy, auc = model.evaluate(X, y_true_onehot, verbose=0)

    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nLoss: {loss:.4f}  Accuracy: {accuracy:.4f}  AUC: {auc:.4f}  '
          f'ROC-AUC: {roc_auc:.4f}')
    print(f'\nClassification Report:')
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


def load_data(csv_path, image_dir, target_size=(224, 224)):
    """Load images and labels from a CSV + image directory."""
    import pandas as pd
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)

    X, y = [], []
    for _, row in df.iterrows():
        img_path = image_dir / row['ImageID']
        if not img_path.exists():
            continue

        label_str = row['Labels']
        if label_str not in LABEL_MAPPING:
            continue

        img = load_img(str(img_path), target_size=target_size)
        img_array = img_to_array(img) / 255.0
        X.append(img_array)
        y.append(LABEL_MAPPING[label_str])

    return np.array(X), np.array(y)


def train(args):
    """Two-phase training: frozen base, then fine-tuned top layers."""
    logger = logging.getLogger(__name__)
    set_seeds(args.seed)

    input_shape = (args.image_size, args.image_size, 3)
    num_classes = len(LABEL_MAPPING)

    # --- Load data ---
    csv_path = Path(args.data_csv)
    image_dir = Path(args.image_dir)
    X, y = load_data(csv_path, image_dir, target_size=input_shape[:2])

    logger.info(f'loaded {len(X)} images')

    # Compute class weights before one-hot encoding
    classes, counts = np.unique(y, return_counts=True)
    class_weight = {
        int(cls): len(y) / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
    logger.info(f'class weights: {class_weight}')

    y_onehot = to_categorical(y, num_classes=num_classes)

    # --- Three-way split: train / val / test ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_onehot, test_size=0.15, random_state=args.seed, stratify=y
    )
    # Stratify on argmax of one-hot for the second split
    y_tv_labels = np.argmax(y_train_val, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.176,  # ~15% of total
        random_state=args.seed,
        stratify=y_tv_labels,
    )
    logger.info(
        f'splits — train: {len(X_train)}, val: {len(X_val)}, '
        f'test: {len(X_test)}'
    )

    # --- Data augmentation ---
    augmenter = build_augmenter()
    train_gen = augmenter.flow(X_train, y_train, batch_size=args.batch_size)

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Train with frozen base ===
    logger.info('Phase 1: training with frozen ResNet50 base')
    model = create_model(input_shape, num_classes, fine_tune_from=None)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    checkpoint_path = output_dir / 'best_model_phase1.keras'
    history1 = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs_phase1,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(checkpoint_path),
        verbose=1,
    )

    # === Phase 2: Fine-tune top layers of ResNet50 ===
    logger.info('Phase 2: fine-tuning top ResNet50 layers')
    model = create_model(input_shape, num_classes, fine_tune_from=140)
    # Load phase 1 weights
    model.load_weights(str(checkpoint_path))
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    checkpoint_path_ft = output_dir / 'best_model_finetuned.keras'
    history2 = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs_phase2,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=build_callbacks(checkpoint_path_ft),
        verbose=1,
    )

    # === Evaluation on held-out test set ===
    logger.info('Evaluating on held-out test set')
    print('\n' + '=' * 50)
    print('TEST SET EVALUATION')
    print('=' * 50)
    metrics = evaluate_model(model, X_test, y_test, LABEL_NAMES)

    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'metrics saved to {metrics_path}')

    # Save final model
    final_path = output_dir / 'final_model.keras'
    model.save(str(final_path))
    logger.info(f'final model saved to {final_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train chest X-ray classifier (normal vs pneumonia)'
    )
    parser.add_argument(
        '--data-csv',
        default=str(PROJECT_DIR / 'data' / 'processed' / 'balanced.csv'),
        help='Path to CSV with ImageID and Labels columns',
    )
    parser.add_argument(
        '--image-dir',
        default=str(PROJECT_DIR / 'data' / 'raw' / 'img'),
        help='Directory containing the X-ray images',
    )
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_DIR / 'models'),
        help='Directory to save trained models and metrics',
    )
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-phase1', type=int, default=20)
    parser.add_argument('--epochs-phase2', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parse_args()
    train(args)
