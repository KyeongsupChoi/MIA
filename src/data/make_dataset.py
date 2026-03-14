# -*- coding: utf-8 -*-
"""Process raw TSV data into balanced, split CSVs ready for modeling."""

import logging
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from src.config import LOG_FORMAT, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def load_raw_data(input_filepath: str | Path) -> pd.DataFrame:
    """Load a raw TSV or CSV dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(input_filepath)
    if not filepath.exists():
        raise FileNotFoundError(f'input file not found: {filepath}')

    sep = '\t' if filepath.suffix == '.tsv' else ','
    df = pd.read_csv(filepath, delimiter=sep)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f'dataset is missing required columns: {missing}. '
            f'Found: {list(df.columns)}'
        )

    logger.info('loaded %d records from %s', len(df), filepath.name)
    return df


def filter_dataset(df: pd.DataFrame, n_per_class: int = 400) -> pd.DataFrame:
    """Filter to PA-projection, non-pediatric, normal vs pneumonia only.

    Uses exact string matching to avoid false positives (e.g. "abnormal"
    matching a contains('normal') check).

    Raises:
        ValueError: If fewer samples than requested are available for a class.
    """
    df = df[df['Projection'].str.contains('PA', na=False)].copy()
    logger.info('%d records after PA projection filter', len(df))

    df = df[df['Pediatric'].str.contains('No', na=False)].copy()
    logger.info('%d records after pediatric filter', len(df))

    normal = df[df['Labels'] == "['normal']"].head(n_per_class)
    pneumonia = df[df['Labels'] == "['pneumonia']"].head(n_per_class)

    if len(normal) == 0 or len(pneumonia) == 0:
        raise ValueError(
            f'insufficient data after filtering — '
            f'normal: {len(normal)}, pneumonia: {len(pneumonia)}'
        )

    if len(normal) < n_per_class:
        logger.warning(
            'only %d normal samples available (requested %d)',
            len(normal), n_per_class,
        )
    if len(pneumonia) < n_per_class:
        logger.warning(
            'only %d pneumonia samples available (requested %d)',
            len(pneumonia), n_per_class,
        )

    balanced = pd.concat([normal, pneumonia], axis=0).reset_index(drop=True)
    logger.info('balanced dataset: %d normal, %d pneumonia',
                len(normal), len(pneumonia))
    return balanced


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits.

    The val_size is relative to the total dataset, not to the train set.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError('split sizes must be between 0 and 1')
    if test_size + val_size >= 1:
        raise ValueError('test_size + val_size must be less than 1')

    train_val, test = train_test_split(
        df, test_size=test_size,
        stratify=df['Labels'], random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size,
        stratify=train_val['Labels'], random_state=random_state,
    )

    return train, val, test


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n-per-class', default=400, help='Samples per class')
@click.option('--test-size', default=0.15, help='Test split fraction')
@click.option('--val-size', default=0.15, help='Validation split fraction')
@click.option('--seed', default=42, help='Random seed')
def main(input_filepath, output_filepath, n_per_class, test_size, val_size,
         seed):
    """Process raw data into train/val/test splits for modeling."""
    output_dir = Path(output_filepath)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(input_filepath)
    balanced = filter_dataset(df, n_per_class=n_per_class)

    balanced.to_csv(output_dir / 'balanced.csv', index=False)
    logger.info('saved balanced dataset: %d records', len(balanced))

    train, val, test = create_splits(
        balanced, test_size=test_size, val_size=val_size, random_state=seed,
    )

    train.to_csv(output_dir / 'train.csv', index=False)
    val.to_csv(output_dir / 'val.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)

    logger.info(
        'splits — train: %d, val: %d, test: %d',
        len(train), len(val), len(test),
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    load_dotenv(find_dotenv())
    main()
