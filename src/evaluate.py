"""
Model Evaluation and Threshold Optimization Module for Fraud Detection.
Evaluates models using PR-AUC, Recall, Precision, F1, ROC-AUC, and financial cost.
Generates evaluation curves, confusion matrices, and model comparison artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import setup_logger

logger = setup_logger("evaluate")


def calculate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes standard and fraud-specific evaluation metrics for binary classification.
    Prioritizes PR-AUC and Recall for imbalanced fraud datasets.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Core fraud detection metrics
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": round(threshold, 3),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(acc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 15.0,
    cost_tp: float = 15.0,
) -> Dict[str, Any]:
    """
    Sweeps probability thresholds to identify optimal decision boundaries:
    1. Best F1-Score (balanced harmonic mean)
    2. Best F2-Score (recall-oriented fraud priority)
    3. Minimum Financial Cost (business operational loss)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    records = []

    for t in thresholds:
        m = calculate_metrics(y_true, y_prob, threshold=t)
        # Financial cost calculation:
        # Undetected fraud (FN) costs direct chargeback losses (~$500 avg)
        # False alarm (FP) costs customer friction & investigation (~$15)
        # Confirmed fraud (TP) costs verification workflow (~$15)
        total_cost = (m["fn"] * cost_fn) + (m["fp"] * cost_fp) + (m["tp"] * cost_tp)
        m["financial_cost"] = total_cost
        records.append(m)

    df_sweep = pd.DataFrame(records)

    best_f1_idx = df_sweep["f1"].idxmax()
    best_f2_idx = df_sweep["f2"].idxmax()
    best_cost_idx = df_sweep["financial_cost"].idxmin()

    best_f1 = df_sweep.loc[best_f1_idx].to_dict()
    best_f2 = df_sweep.loc[best_f2_idx].to_dict()
    best_cost = df_sweep.loc[best_cost_idx].to_dict()

    return {
        "optimal_f1": best_f1,
        "optimal_f2": best_f2,
        "optimal_cost": best_cost,
        "sweep_data": df_sweep,
    }


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots professional ROC and Precision-Recall curves side by side.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title(f"{model_name} - ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR Curve
    axes[1].plot(recall, precision, color="#d62728", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    baseline = np.mean(y_true)
    axes[1].axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Fraud Detected)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{model_name} - Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC/PR curves to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a clean, professional confusion matrix with counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = [
        f"{count}\n({pct:.1%})"
        for count, pct in zip(cm.flatten(), cm_pct.flatten())
    ]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    plt.close()


def save_model_comparison_results(
    results_list: List[Dict[str, Any]], output_path: str = "reports/model_results.csv"
) -> pd.DataFrame:
    """
    Saves and formats model benchmark comparison table.
    """
    df_results = pd.DataFrame(results_list)
    cols = ["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for c in cols:
        if c not in df_results.columns:
            df_results[c] = np.nan
    df_ordered = df_results[cols].sort_values(by="pr_auc", ascending=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_ordered.to_csv(output_path, index=False)
    logger.info(f"Saved model comparison table to {output_path}")
    return df_ordered
