"""Exploratory data analysis for the chest X-ray dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]


def run_eda(tsv_path: str | Path) -> pd.DataFrame:
    """Load the raw dataset and print summary statistics.

    Returns the loaded DataFrame for further analysis.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f'dataset not found: {tsv_path}')

    df = pd.read_csv(tsv_path, sep='\t')

    print(f'Shape: {df.shape}')
    print(f'\nColumns: {list(df.columns)}')
    print(f'\nHead:\n{df.head().to_string()}')
    print(f'\nDescribe:\n{df.describe().to_string()}')
    print(f'\nInfo:')
    df.info()

    print('\n--- Value counts per column ---')
    for col in df.columns:
        vc = df[col].value_counts()
        print(f'\n{col} ({df[col].nunique()} unique):')
        print(vc.head(10).to_string())

    # Label distribution
    cols = ['ImageID', 'PatientID', 'PatientBirth', 'Projection',
            'Pediatric', 'Modality_DICOM', 'Manufacturer_DICOM',
            'Labels', 'group']
    available = [c for c in cols if c in df.columns]
    subset = df[available]

    if 'Labels' in subset.columns:
        exploded = subset.explode('Labels')
        label_counts = exploded['Labels'].value_counts()
        print(f'\n--- Top 10 labels ---\n{label_counts.head(10).to_string()}')

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description='Run EDA on raw dataset')
    parser.add_argument(
        '--input',
        default=str(PROJECT_DIR / 'data' / 'raw' / 'neumo_dataset.tsv'),
    )
    args = parser.parse_args()
    run_eda(args.input)
