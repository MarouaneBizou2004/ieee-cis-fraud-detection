"""
Data Loader Module for IEEE-CIS Fraud Detection.
Handles loading raw transaction and identity tables, merging on TransactionID,
memory optimization, and dataset profiling without unnecessary complexity.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils import reduce_mem_usage, setup_logger

logger = setup_logger("data_loader")


class DataLoader:
    """
    Robust data loader for IEEE-CIS Fraud Detection dataset.
    Loads and joins transaction and identity files with automatic memory reduction.
    """

    def __init__(self, raw_data_dir: str = "data/raw", config: Optional[dict] = None):
        self.raw_data_dir = Path(raw_data_dir)
        self.config = config or {}

    def check_dataset_exists(self) -> Dict[str, bool]:
        """
        Verifies if expected raw CSV files are present.
        """
        expected_files = {
            "train_transaction": self.raw_data_dir / "train_transaction.csv",
            "train_identity": self.raw_data_dir / "train_identity.csv",
            "test_transaction": self.raw_data_dir / "test_transaction.csv",
            "test_identity": self.raw_data_dir / "test_identity.csv",
        }
        status = {name: path.exists() for name, path in expected_files.items()}
        return status

    def get_download_instructions(self) -> str:
        """
        Returns clear instructions for downloading the official Kaggle dataset.
        """
        return """
================================================================================
KAGGLE DATASET SETUP INSTRUCTIONS:
================================================================================
1. Visit the official Kaggle competition page:
   https://www.kaggle.com/c/ieee-fraud-detection/data

2. Download the following required files:
   - train_transaction.csv.zip
   - train_identity.csv.zip
   - test_transaction.csv.zip
   - test_identity.csv.zip

3. Unzip and place the files directly into:
   data/raw/
   |-- train_transaction.csv
   |-- train_identity.csv
   |-- test_transaction.csv
   +-- test_identity.csv

Alternatively, using the official Kaggle CLI:
   kaggle competitions download -c ieee-fraud-detection -p data/raw/
   cd data/raw && unzip -o "*.zip"
================================================================================
"""

    def load_train_data(
        self,
        nrows: Optional[int] = None,
        usecols_transaction: Optional[List[str]] = None,
        usecols_identity: Optional[List[str]] = None,
        reduce_memory: bool = True,
    ) -> pd.DataFrame:
        """
        Loads and merges train_transaction.csv and train_identity.csv on TransactionID.
        """
        train_tx_path = self.raw_data_dir / "train_transaction.csv"
        train_id_path = self.raw_data_dir / "train_identity.csv"

        if not train_tx_path.exists():
            msg = f"Missing required file: {train_tx_path}\n" + self.get_download_instructions()
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading transaction data from {train_tx_path} (nrows={nrows})...")
        df_tx = pd.read_csv(train_tx_path, nrows=nrows, usecols=usecols_transaction)

        if reduce_memory:
            df_tx = reduce_mem_usage(df_tx, verbose=False)

        if train_id_path.exists():
            logger.info(f"Loading identity data from {train_id_path} (nrows={nrows})...")
            df_id = pd.read_csv(train_id_path, nrows=nrows, usecols=usecols_identity)
            if reduce_memory:
                df_id = reduce_mem_usage(df_id, verbose=False)

            logger.info("Merging transaction and identity on TransactionID...")
            df = df_tx.merge(df_id, on="TransactionID", how="left")
            del df_tx, df_id
        else:
            logger.warning(
                f"Identity file {train_id_path} not found. Continuing with transaction table only."
            )
            df = df_tx

        if reduce_memory:
            df = reduce_mem_usage(df, verbose=True)

        logger.info(f"Loaded train dataset with shape: {df.shape}")
        return df

    def load_test_data(
        self,
        nrows: Optional[int] = None,
        reduce_memory: bool = True,
    ) -> pd.DataFrame:
        """
        Loads and merges test_transaction.csv and test_identity.csv on TransactionID.
        """
        test_tx_path = self.raw_data_dir / "test_transaction.csv"
        test_id_path = self.raw_data_dir / "test_identity.csv"

        if not test_tx_path.exists():
            msg = f"Missing required file: {test_tx_path}\n" + self.get_download_instructions()
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading test transaction data from {test_tx_path} (nrows={nrows})...")
        df_tx = pd.read_csv(test_tx_path, nrows=nrows)

        if test_id_path.exists():
            logger.info(f"Loading test identity data from {test_id_path} (nrows={nrows})...")
            df_id = pd.read_csv(test_id_path, nrows=nrows)
            df = df_tx.merge(df_id, on="TransactionID", how="left")
            del df_tx, df_id
        else:
            df = df_tx

        if reduce_memory:
            df = reduce_mem_usage(df, verbose=True)

        logger.info(f"Loaded test dataset with shape: {df.shape}")
        return df

    @staticmethod
    def get_data_summary(df: pd.DataFrame, target_col: str = "isFraud") -> Dict[str, Union[int, float, dict]]:
        """
        Generates a comprehensive summary of the dataset for Data Understanding.
        """
        total_rows, total_cols = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

        # Missing values
        missing_series = df.isnull().sum()
        total_missing = int(missing_series.sum())
        cols_with_missing = int((missing_series > 0).sum())
        max_missing_col = missing_series.idxmax() if not missing_series.empty else "N/A"
        max_missing_pct = (
            float(missing_series.max() / total_rows * 100) if total_rows > 0 else 0.0
        )

        # Duplicate check
        duplicates = int(df.duplicated(subset=["TransactionID"]).sum()) if "TransactionID" in df.columns else 0

        # Target distribution
        target_info = {}
        if target_col in df.columns:
            target_counts = df[target_col].value_counts().to_dict()
            fraud_count = int(target_counts.get(1, 0))
            legit_count = int(target_counts.get(0, 0))
            fraud_rate = float(fraud_count / total_rows * 100) if total_rows > 0 else 0.0
            target_info = {
                "legitimate_count": legit_count,
                "fraud_count": fraud_count,
                "fraud_percentage": round(fraud_rate, 3),
            }

        # Data types
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_info = {str(k): int(v) for k, v in dtype_counts.items()}

        summary = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_mb": round(mem_mb, 2),
            "duplicates": duplicates,
            "columns_with_missing": cols_with_missing,
            "total_missing_values": total_missing,
            "max_missing_column": max_missing_col,
            "max_missing_percentage": round(max_missing_pct, 2),
            "data_types": dtype_info,
            "target_distribution": target_info,
        }
        return summary
