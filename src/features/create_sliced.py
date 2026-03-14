"""Create stratified train/test splits from a labeled CSV."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LOG_FORMAT

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]


def create_splits(
    input_csv: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labeled CSV into stratified train/test sets.

    Raises:
        FileNotFoundError: If input_csv does not exist.
        KeyError: If 'Labels' column is missing.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f'input CSV not found: {input_csv}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if 'Labels' not in df.columns:
        raise KeyError("CSV missing required 'Labels' column")

    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df['Labels'], random_state=random_state,
    )

    train_df.to_csv(output_dir / 'train_dataset.csv', index=False)
    test_df.to_csv(output_dir / 'test_dataset.csv', index=False)

    logger.info('train: %d, test: %d', len(train_df), len(test_df))
    return train_df, test_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description='Create train/test splits')
    parser.add_argument(
        '--input', default=str(PROJECT_DIR / 'data' / 'raw' / 'sliced.csv'),
    )
    parser.add_argument(
        '--output-dir', default=str(PROJECT_DIR / 'data' / 'processed'),
    )
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    create_splits(args.input, args.output_dir, args.test_size, args.seed)
