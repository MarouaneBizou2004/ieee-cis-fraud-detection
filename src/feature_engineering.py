"""
Feature Engineering Pipeline for IEEE-CIS Fraud Detection.
Creates domain-specific fraud detection features:
- Time-based cyclics (hours, days)
- Amount representations (log, decimal/cents)
- Email provider clustering and matching
- Aggregation features (amount relative to card average)
- Device/browser categorization
- Missingness density signals
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """
    Builds domain-specific fraud features while strictly preventing data leakage.
    Group statistics (e.g. card1 amount mean/std) are computed on training data
    and reused during inference.
    """

    def __init__(self, time_col: str = "TransactionDT", amt_col: str = "TransactionAmt"):
        self.time_col = time_col
        self.amt_col = amt_col
        self.card1_amt_stats_: Dict[str, Dict[str, float]] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Computes group statistics on training data.
        """
        logger.info("Fitting FeatureEngineer aggregations on training data...")
        df = X

        if "card1" in df.columns and self.amt_col in df.columns:
            # Group stats for card1
            stats = df.groupby("card1")[self.amt_col].agg(["mean", "std"]).to_dict("index")
            self.card1_amt_stats_ = stats

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature transformations to the DataFrame.
        """
        df = X.copy()

        # 1. Null count indicator
        df["null_count"] = df.isnull().sum(axis=1)

        # 2. Time features
        if self.time_col in df.columns:
            # Time in seconds -> hour of day and day of week
            df["Transaction_hour"] = (df[self.time_col] // 3600) % 24
            df["Transaction_day"] = (df[self.time_col] // (3600 * 24)) % 7

        # 3. Transaction Amount features
        if self.amt_col in df.columns:
            df["TransactionAmt_log"] = np.log1p(df[self.amt_col].clip(lower=0))
            df["TransactionAmt_decimal"] = df[self.amt_col] - np.floor(df[self.amt_col])

        # 4. Email Domain features
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            # Clean domains
            p_domain = df["P_emaildomain"].fillna("missing").astype(str).str.split(".").str[0]
            r_domain = df["R_emaildomain"].fillna("missing").astype(str).str.split(".").str[0]
            df["P_email_prefix"] = p_domain
            df["email_domain_match"] = (
                (p_domain == r_domain) & (p_domain != "missing")
            ).astype(int)

        # 5. Device and Browser Normalization (from identity features if present)
        if "id_30" in df.columns:
            # Operating system
            df["os_simplified"] = (
                df["id_30"]
                .fillna("Unknown")
                .astype(str)
                .str.lower()
                .apply(
                    lambda x: "windows"
                    if "windows" in x
                    else "ios"
                    if "ios" in x
                    else "android"
                    if "android" in x
                    else "mac"
                    if "mac" in x
                    else "linux"
                    if "linux" in x
                    else "other"
                )
            )

        if "id_31" in df.columns:
            # Browser
            df["browser_simplified"] = (
                df["id_31"]
                .fillna("Unknown")
                .astype(str)
                .str.lower()
                .apply(
                    lambda x: "chrome"
                    if "chrome" in x
                    else "safari"
                    if "safari" in x
                    else "firefox"
                    if "firefox" in x
                    else "edge"
                    if "edge" in x or "ie" in x
                    else "other"
                )
            )

        # 6. Group aggregations (card1 amount relative to card average)
        if "card1" in df.columns and self.amt_col in df.columns and self.card1_amt_stats_:
            card1_means = {k: v.get("mean", np.nan) for k, v in self.card1_amt_stats_.items()}
            card1_stds = {k: v.get("std", np.nan) for k, v in self.card1_amt_stats_.items()}

            mean_s = df["card1"].map(card1_means)
            std_s = df["card1"].map(card1_stds)

            global_amt_mean = df[self.amt_col].mean()
            mean_s = mean_s.fillna(global_amt_mean)

            df["TransactionAmt_to_mean_card1"] = df[self.amt_col] / (mean_s + 1e-5)
            df["TransactionAmt_to_std_card1"] = df[self.amt_col] / (std_s.fillna(1.0) + 1e-5)

        logger.info(f"Feature engineering applied. New dataframe shape: {df.shape}")
        return df

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fits on training data and transforms it.
        """
        return self.fit(X, y).transform(X)
