"""
Unit tests for src/evaluate.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calculate_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    save_model_comparison_results,
)


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_find_optimal_threshold():
    y_true = np.array([0]*90 + [1]*10)
    # Fraud cases have higher probabilities
    y_prob = np.array([0.05]*90 + [0.75]*10)

    res = find_optimal_threshold(y_true, y_prob)

    assert "optimal_f1" in res
    assert "optimal_cost" in res
    assert "sweep_data" in res
    assert isinstance(res["sweep_data"], pd.DataFrame)
    assert 0.05 < res["optimal_f1"]["threshold"] < 0.75


def test_plot_curves_and_confusion_matrix(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)

    curve_path = tmp_path / "test_curve.png"
    cm_path = tmp_path / "test_cm.png"

    plot_roc_pr_curves(y_true, y_prob, model_name="TestModel", save_path=str(curve_path))
    plot_confusion_matrix(y_true, y_pred, model_name="TestModel", save_path=str(cm_path))

    assert curve_path.exists()
    assert cm_path.exists()


def test_save_model_comparison_results(tmp_path):
    records = [
        {"model": "ModelA", "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.9, "pr_auc": 0.85},
        {"model": "ModelB", "precision": 0.9, "recall": 0.8, "f1": 0.85, "roc_auc": 0.95, "pr_auc": 0.92},
    ]
    out_file = tmp_path / "results.csv"
    df = save_model_comparison_results(records, output_path=str(out_file))

    assert out_file.exists()
    assert len(df) == 2
    assert df.iloc[0]["model"] == "ModelB"  # Sorted descending by PR-AUC
