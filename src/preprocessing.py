"""
Data Preprocessing Pipeline for IEEE-CIS Fraud Detection.
Handles missing values, categorical encoding, scaling, and infinite values
with zero data leakage and simple, maintainable Python code.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logger

logger = setup_logger("preprocessing")


class Preprocessor:
    """
    Robust preprocessor for tabular fraud detection data.
    Fitted exclusively on training data to prevent any data leakage.
    """

    def __init__(
        self,
        drop_high_missing_thresh: float = 0.85,
        target_col: str = "isFraud",
        id_col: str = "TransactionID",
    ):
        self.drop_high_missing_thresh = drop_high_missing_thresh
        self.target_col = target_col
        self.id_col = id_col

        # Learned state on training split
        self.columns_to_drop_: List[str] = []
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.freq_encodings_: Dict[str, Dict[str, float]] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.fitted_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Learns imputation values and frequency mappings strictly from training data.
        """
        logger.info("Fitting preprocessor on training data...")
        df = X.copy()

        # Target variable must NEVER be part of input features
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        if self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])

        n_samples = len(df)

        # 1. Identify columns with excessive missingness (> threshold)
        missing_ratios = df.isnull().sum() / n_samples
        self.columns_to_drop_ = missing_ratios[
            missing_ratios > self.drop_high_missing_thresh
        ].index.tolist()
        logger.info(
            f"Identified {len(self.columns_to_drop_)} columns exceeding missing threshold "
            f"({self.drop_high_missing_thresh * 100:.0f}%)"
        )
        df = df.drop(columns=self.columns_to_drop_, errors="ignore")

        # 2. Identify numerical and categorical columns
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 3. Learn medians for numerical columns
        for col in self.num_cols_:
            # Replace inf with nan
            valid_series = df[col].replace([np.inf, -np.inf], np.nan)
            median_val = float(valid_series.median())
            # Fallback if entirely NaN
            self.medians_[col] = 0.0 if np.isnan(median_val) else median_val

        # 4. Learn frequency encoding for categorical columns
        for col in self.cat_cols_:
            freq = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .value_counts(normalize=True)
                .to_dict()
            )
            self.freq_encodings_[col] = freq

        # 5. Fit StandardScaler for models requiring normalization
        clean_num_df = df[self.num_cols_].replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(clean_num_df)

        self.fitted_feature_names_ = self.num_cols_ + self.cat_cols_
        logger.info(
            f"Preprocessor fitted successfully. Final features: {len(self.fitted_feature_names_)} "
            f"({len(self.num_cols_)} numerical, {len(self.cat_cols_)} categorical)"
        )
        return self

    def transform(self, X: pd.DataFrame, scale_numeric: bool = False) -> pd.DataFrame:
        """
        Applies learned transformations to input data without data leakage.
        """
        if not self.fitted_feature_names_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        df = X.copy()

        # Remove target and ID if present
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        if self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])

        # Drop excessive missing columns identified during fit
        df = df.drop(columns=self.columns_to_drop_, errors="ignore")

        # Process numerical features
        for col in self.num_cols_:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(self.medians_.get(col, 0.0))
            else:
                df[col] = self.medians_.get(col, 0.0)

        # Process categorical features using learned frequency maps
        for col in self.cat_cols_:
            if col in df.columns:
                series = df[col].fillna("Unknown").astype(str)
                mapping = self.freq_encodings_.get(col, {})
                # Unseen categories default to 0.0 frequency
                df[col] = series.map(mapping).fillna(0.0)
            else:
                df[col] = 0.0

        # Ensure exact column ordering
        df_out = df[self.fitted_feature_names_].copy()

        if scale_numeric and self.scaler_ is not None:
            df_out[self.num_cols_] = self.scaler_.transform(df_out[self.num_cols_])

        return df_out

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, scale_numeric: bool = False
    ) -> pd.DataFrame:
        """
        Fits on training data and transforms it in one step.
        """
        return self.fit(X, y).transform(X, scale_numeric=scale_numeric)
