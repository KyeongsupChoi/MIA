# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import find_dotenv, load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_raw_data(input_filepath):
    """Load the raw TSV dataset and return a DataFrame."""
    logger = logging.getLogger(__name__)
    filepath = Path(input_filepath)

    if filepath.suffix == '.tsv':
        df = pd.read_csv(filepath, delimiter='\t')
    else:
        df = pd.read_csv(filepath)

    logger.info(f'loaded {len(df)} records from {filepath.name}')
    return df


def filter_dataset(df, n_per_class=400):
    """Filter to PA-projection, non-pediatric, normal vs pneumonia only.

    Returns a balanced dataset with n_per_class samples per label.
    """
    logger = logging.getLogger(__name__)

    # Keep only PA projection (most diagnostically reliable)
    df = df[df['Projection'].str.contains('PA', na=False)]
    logger.info(f'{len(df)} records after PA projection filter')

    # Exclude pediatric cases
    df = df[df['Pediatric'].str.contains('No', na=False)]
    logger.info(f'{len(df)} records after pediatric filter')

    # Split into normal and pneumonia subsets
    normal = df[df['Labels'].str.contains('normal', na=False)].head(n_per_class)
    pneumonia = df[
        df['Labels'].apply(
            lambda x: isinstance(x, str)
            and 'pneumonia' in x
            and x.strip() == "['pneumonia']"
        )
    ].head(n_per_class)

    logger.info(f'selected {len(normal)} normal, {len(pneumonia)} pneumonia')

    balanced = pd.concat([normal, pneumonia], axis=0).reset_index(drop=True)
    return balanced


def create_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """Create train/val/test splits with stratification.

    With defaults: ~70% train, ~15% val, ~15% test.
    """
    # First split off the test set
    train_val, test = train_test_split(
        df, test_size=test_size,
        stratify=df['Labels'], random_state=random_state
    )

    # Then split train_val into train and val
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size,
        stratify=train_val['Labels'], random_state=random_state
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
    logger = logging.getLogger(__name__)

    output_dir = Path(output_filepath)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter
    df = load_raw_data(input_filepath)
    balanced = filter_dataset(df, n_per_class=n_per_class)

    # Save the full balanced dataset
    balanced.to_csv(output_dir / 'balanced.csv', index=False)
    logger.info(f'saved balanced dataset: {len(balanced)} records')

    # Create and save splits
    train, val, test = create_splits(
        balanced, test_size=test_size, val_size=val_size, random_state=seed
    )

    train.to_csv(output_dir / 'train.csv', index=False)
    val.to_csv(output_dir / 'val.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)

    logger.info(
        f'splits — train: {len(train)}, val: {len(val)}, test: {len(test)}'
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
