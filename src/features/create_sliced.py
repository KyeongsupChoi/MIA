import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[2]


def create_splits(input_csv, output_dir, test_size=0.2, random_state=42):
    """Split a labeled CSV into stratified train/test sets."""
    logger = logging.getLogger(__name__)

    df = pd.read_csv(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df['Labels'], random_state=random_state,
    )

    train_path = output_dir / 'train_dataset.csv'
    test_path = output_dir / 'test_dataset.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f'train: {len(train_df)}, test: {len(test_df)}')
    return train_df, test_df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description='Create train/test splits')
    parser.add_argument(
        '--input', default=str(PROJECT_DIR / 'data' / 'raw' / 'sliced.csv'),
        help='Input CSV file path',
    )
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_DIR / 'data' / 'processed'),
        help='Output directory for split CSVs',
    )
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    create_splits(args.input, args.output_dir, args.test_size, args.seed)
