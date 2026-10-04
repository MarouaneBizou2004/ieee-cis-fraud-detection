"""
Unit tests for src/predict.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import FeatureEngineer
from src.predict import FraudPredictor
from src.preprocessing import Preprocessor
from src.utils import save_artifact, save_json


def test_predictor_missing_artifacts(tmp_path):
    # Empty directory with no artifacts
    predictor = FraudPredictor(
        model_path=str(tmp_path / "model.pkl"),
        preprocessor_path=str(tmp_path / "preprocessor.pkl"),
        fe_path=str(tmp_path / "fe.pkl"),
        metadata_path=str(tmp_path / "meta.json"),
    )
    assert not predictor.is_ready

    with pytest.raises(RuntimeError):
        predictor.predict_single({"TransactionAmt": 100})


def test_predictor_end_to_end_mock(tmp_path):
    # Train and serialize mock pipeline
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [86400, 86400 * 2, 86400 * 3, 86400 * 4],
        "TransactionAmt": [50.0, 1500.0, 30.0, 2000.0],
        "card1": [10020, 10020, 15885, 15885],
        "P_emaildomain": ["gmail.com", "anonymous.com", "yahoo.com", "protonmail.com"],
        "R_emaildomain": ["gmail.com", "protonmail.com", "yahoo.com", "anonymous.com"],
        "isFraud": [0, 1, 0, 1],
    })

    fe = FeatureEngineer()
    df_fe = fe.fit_transform(df)

    prep = Preprocessor()
    X_proc = prep.fit_transform(df_fe)

    model = LogisticRegression()
    model.fit(X_proc, df["isFraud"])

    # Save artifacts to temp path
    model_path = tmp_path / "final_model.pkl"
    prep_path = tmp_path / "preprocessor.pkl"
    fe_path = tmp_path / "feature_engineer.pkl"
    meta_path = tmp_path / "model_metadata.json"

    save_artifact(model, str(model_path))
    save_artifact(prep, str(prep_path))
    save_artifact(fe, str(fe_path))
    save_json({"model_name": "Logistic Regression", "optimal_threshold": 0.40}, str(meta_path))

    # Initialize predictor
    predictor = FraudPredictor(
        model_path=str(model_path),
        preprocessor_path=str(prep_path),
        fe_path=str(fe_path),
        metadata_path=str(meta_path),
    )
    assert predictor.is_ready
    assert predictor.threshold == 0.40

    # Test single prediction
    tx = {
        "TransactionID": 10,
        "TransactionDT": 86400 * 5 + 3600 * 3,
        "TransactionAmt": 1800.0,
        "card1": 15885,
        "P_emaildomain": "anonymous.com",
        "R_emaildomain": "gmail.com",
    }
    result = predictor.predict_single(tx)

    assert "fraud_probability" in result
    assert "prediction" in result
    assert result["prediction"] in ["LEGITIMATE", "FRAUD"]
    assert "confidence" in result
    assert "top_risk_factors" in result
    assert isinstance(result["top_risk_factors"], list)
