"""
Unit tests for Streamlit application components.
"""

import pytest
from app.components.about_model import render_about_model
from app.components.dashboard import render_dashboard
from app.components.data_insights import render_data_insights
from app.components.feature_importance import render_feature_importance
from app.components.fraud_prediction import render_fraud_prediction
from app.components.model_performance import render_model_performance, get_default_benchmarks


def test_default_benchmarks_structure():
    df = get_default_benchmarks()
    assert len(df) >= 8
    expected_cols = ["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for col in expected_cols:
        assert col in df.columns


def test_components_are_callable():
    assert callable(render_dashboard)
    assert callable(render_model_performance)
    assert callable(render_fraud_prediction)
    assert callable(render_feature_importance)
    assert callable(render_data_insights)
    assert callable(render_about_model)
