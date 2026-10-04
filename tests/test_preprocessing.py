"""
Unit tests for src/preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import Preprocessor


def test_preprocessor_fit_transform():
    train_df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionAmt": [10.0, 20.0, np.nan, 40.0, 50.0],
        "card4": ["visa", "mastercard", "visa", "visa", "discover"],
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],  # >85% null
        "isFraud": [0, 0, 1, 0, 1],
    })

    preprocessor = Preprocessor(drop_high_missing_thresh=0.85)
    X_proc = preprocessor.fit_transform(train_df)

    # 1. Target and ID should never be in output features
    assert "isFraud" not in X_proc.columns
    assert "TransactionID" not in X_proc.columns

    # 2. Column with 100% missing should be dropped
    assert "all_null" not in X_proc.columns

    # 3. Missing TransactionAmt should be filled with median (30.0)
    assert not X_proc["TransactionAmt"].isnull().any()
    assert preprocessor.medians_["TransactionAmt"] == 30.0

    # 4. Categorical frequency encoding
    assert "card4" in X_proc.columns
    # visa occurs 3 times out of 5 -> 0.6
    assert preprocessor.freq_encodings_["card4"]["visa"] == 0.6


def test_preprocessor_unseen_categories():
    train_df = pd.DataFrame({
        "TransactionAmt": [100.0, 200.0],
        "card4": ["visa", "mastercard"],
        "isFraud": [0, 1],
    })
    test_df = pd.DataFrame({
        "TransactionAmt": [300.0, 400.0],
        "card4": ["american express", "discover"],  # Unseen categories
    })

    preprocessor = Preprocessor()
    preprocessor.fit(train_df)
    X_test_proc = preprocessor.transform(test_df)

    # Unseen categories must be filled with 0.0 frequency without raising error
    assert (X_test_proc["card4"] == 0.0).all()


def test_preprocessor_scaling():
    train_df = pd.DataFrame({
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "card4": ["visa", "visa", "visa", "visa", "visa"],
    })

    preprocessor = Preprocessor()
    X_scaled = preprocessor.fit_transform(train_df, scale_numeric=True)

    # Standard scaled numeric column should have approximately 0 mean and 1 variance
    assert np.isclose(X_scaled["TransactionAmt"].mean(), 0.0, atol=1e-5)
    assert np.isclose(X_scaled["TransactionAmt"].std(ddof=0), 1.0, atol=1e-5)
