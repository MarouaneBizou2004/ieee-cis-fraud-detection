"""
Unit tests for src/feature_engineering.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


def test_feature_engineering_transforms():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [86400 * 2 + 3600 * 5, 86400 * 4 + 3600 * 15],  # 5:00 AM day 2, 15:00 day 4
        "TransactionAmt": [100.25, 45.00],
        "card1": [10020, 10020],
        "P_emaildomain": ["gmail.com", "anonymous.com"],
        "R_emaildomain": ["gmail.com", "yahoo.com"],
        "id_30": ["Windows 10", "iOS 12.1.0"],
        "id_31": ["chrome 66.0", "safari 11.0"],
    })

    fe = FeatureEngineer()
    df_out = fe.fit_transform(df)

    # 1. Hour and Day features
    assert "Transaction_hour" in df_out.columns
    assert df_out["Transaction_hour"].iloc[0] == 5
    assert df_out["Transaction_hour"].iloc[1] == 15

    # 2. Amount features
    assert "TransactionAmt_log" in df_out.columns
    assert np.isclose(df_out["TransactionAmt_log"].iloc[0], np.log1p(100.25))
    assert np.isclose(df_out["TransactionAmt_decimal"].iloc[0], 0.25)
    assert np.isclose(df_out["TransactionAmt_decimal"].iloc[1], 0.00)

    # 3. Email features
    assert "email_domain_match" in df_out.columns
    assert df_out["email_domain_match"].iloc[0] == 1  # gmail == gmail
    assert df_out["email_domain_match"].iloc[1] == 0  # anonymous != yahoo

    # 4. Device and OS
    assert "os_simplified" in df_out.columns
    assert df_out["os_simplified"].iloc[0] == "windows"
    assert df_out["os_simplified"].iloc[1] == "ios"

    # 5. Null count
    assert "null_count" in df_out.columns
    assert (df_out["null_count"] == 0).all()
