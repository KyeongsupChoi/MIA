"""Tests for the data pipeline."""

import pandas as pd
import pytest

from src.data.make_dataset import create_splits, filter_dataset


@pytest.fixture
def raw_df():
    """Minimal DataFrame mimicking the raw TSV structure."""
    rows = []
    for i in range(50):
        rows.append({
            'ImageID': f'normal_{i}.png',
            'Labels': "['normal']",
            'Projection': 'PA',
            'Pediatric': 'No',
        })
    for i in range(50):
        rows.append({
            'ImageID': f'pneumonia_{i}.png',
            'Labels': "['pneumonia']",
            'Projection': 'PA',
            'Pediatric': 'No',
        })
    # These should be filtered out
    rows.append({
        'ImageID': 'lateral.png',
        'Labels': "['normal']",
        'Projection': 'L',
        'Pediatric': 'No',
    })
    rows.append({
        'ImageID': 'peds.png',
        'Labels': "['normal']",
        'Projection': 'PA',
        'Pediatric': 'Yes',
    })
    return pd.DataFrame(rows)


class TestFilterDataset:
    def test_filters_to_balanced_classes(self, raw_df):
        result = filter_dataset(raw_df, n_per_class=10)
        labels = result['Labels'].value_counts()
        assert labels["['normal']"] == 10
        assert labels["['pneumonia']"] == 10

    def test_excludes_non_pa(self, raw_df):
        result = filter_dataset(raw_df, n_per_class=50)
        assert 'lateral.png' not in result['ImageID'].values

    def test_excludes_pediatric(self, raw_df):
        result = filter_dataset(raw_df, n_per_class=50)
        assert 'peds.png' not in result['ImageID'].values

    def test_warns_on_insufficient_samples(self, raw_df, caplog):
        # Ask for more than available
        result = filter_dataset(raw_df, n_per_class=999)
        assert len(result) == 100  # 50 + 50
        assert 'only' in caplog.text.lower()

    def test_raises_on_empty_result(self):
        empty = pd.DataFrame({
            'ImageID': ['x.png'],
            'Labels': ["['infiltrates']"],
            'Projection': ['PA'],
            'Pediatric': ['No'],
        })
        with pytest.raises(ValueError, match='insufficient data'):
            filter_dataset(empty)

    def test_exact_match_not_contains(self, raw_df):
        """'normal' filter should not match 'abnormal'."""
        df = raw_df.copy()
        df = pd.concat([df, pd.DataFrame([{
            'ImageID': 'abnormal.png',
            'Labels': "['abnormal']",
            'Projection': 'PA',
            'Pediatric': 'No',
        }])], ignore_index=True)
        result = filter_dataset(df, n_per_class=50)
        assert 'abnormal.png' not in result['ImageID'].values


class TestCreateSplits:
    def test_split_sizes(self, raw_df):
        balanced = filter_dataset(raw_df, n_per_class=50)
        train, val, test = create_splits(balanced, test_size=0.2, val_size=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(balanced)

    def test_stratification_preserved(self, raw_df):
        balanced = filter_dataset(raw_df, n_per_class=50)
        train, val, test = create_splits(balanced)
        for split in (train, val, test):
            labels = split['Labels'].value_counts()
            # Both classes should be present in every split
            assert len(labels) == 2

    def test_deterministic(self, raw_df):
        balanced = filter_dataset(raw_df, n_per_class=50)
        t1, v1, te1 = create_splits(balanced, random_state=42)
        t2, v2, te2 = create_splits(balanced, random_state=42)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True),
                                      t2.reset_index(drop=True))

    def test_invalid_split_sizes(self, raw_df):
        balanced = filter_dataset(raw_df, n_per_class=50)
        with pytest.raises(ValueError):
            create_splits(balanced, test_size=0.6, val_size=0.5)
