"""
Unit tests for src/data_loader.py
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_data_loader_initialization():
    loader = DataLoader(raw_data_dir="data/raw")
    assert loader.raw_data_dir == Path("data/raw")
    status = loader.check_dataset_exists()
    assert "train_transaction" in status
    assert "train_identity" in status


def test_download_instructions():
    loader = DataLoader()
    instructions = loader.get_download_instructions()
    assert "kaggle.com" in instructions
    assert "train_transaction.csv" in instructions


def test_data_summary():
    df = pd.DataFrame({
        "TransactionID": [101, 102, 103, 104, 105],
        "isFraud": [0, 0, 1, 0, 0],
        "TransactionAmt": [50.0, 100.0, None, 25.0, 500.0],
        "card4": ["visa", "mastercard", "visa", None, "discover"],
    })

    summary = DataLoader.get_data_summary(df, target_col="isFraud")
    assert summary["total_rows"] == 5
    assert summary["total_columns"] == 4
    assert summary["duplicates"] == 0
    assert summary["columns_with_missing"] == 2
    assert summary["target_distribution"]["fraud_count"] == 1
    assert summary["target_distribution"]["legitimate_count"] == 4
    assert summary["target_distribution"]["fraud_percentage"] == 20.0


def test_missing_file_raises_filenotfound(tmp_path):
    loader = DataLoader(raw_data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_train_data()
    assert "Missing required file" in str(exc_info.value)
    assert "KAGGLE DATASET SETUP INSTRUCTIONS" in str(exc_info.value)


def test_load_train_data_with_mock_files(tmp_path):
    # Create mock CSV files in temporary directory
    tx_df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionAmt": [10.5, 20.0, 30.5],
    })
    id_df = pd.DataFrame({
        "TransactionID": [1, 2],
        "DeviceType": ["desktop", "mobile"],
    })

    tx_df.to_csv(tmp_path / "train_transaction.csv", index=False)
    id_df.to_csv(tmp_path / "train_identity.csv", index=False)

    loader = DataLoader(raw_data_dir=str(tmp_path))
    merged = loader.load_train_data(reduce_memory=True)

    assert len(merged) == 3
    assert "DeviceType" in merged.columns
    assert merged.loc[merged["TransactionID"] == 1, "DeviceType"].iloc[0] == "desktop"
    assert pd.isna(merged.loc[merged["TransactionID"] == 3, "DeviceType"].iloc[0])
