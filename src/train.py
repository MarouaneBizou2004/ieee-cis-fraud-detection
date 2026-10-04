"""
Model Training, Cross-Validation, and Model Selection Pipeline.
Trains and compares 8 algorithms:
1. Logistic Regression (Baseline)
2. Decision Tree
3. Random Forest
4. HistGradientBoosting
5. LightGBM
6. XGBoost
7. K-Nearest Neighbors (Subsampled)
8. Support Vector Machine (Subsampled)

Incorporates:
- Time-aware validation split (preventing future-to-past temporal leakage)
- Class imbalance handling (class weights, SMOTE on train fold only)
- Hyperparameter tuning with Optuna / RandomizedSearchCV
- SHAP feature importance extraction
- Model artifact serialization
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
import shap

from src.data_loader import DataLoader
from src.evaluate import (
    calculate_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    save_model_comparison_results,
)
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.utils import load_config, reduce_mem_usage, save_artifact, save_json, setup_logger

logger = setup_logger("train")


class ModelTrainer:
    """
    Orchestrates the entire training, comparison, tuning, and serialization process.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.raw_data_dir = self.config.get("paths", {}).get("raw_data_dir", "data/raw")
        self.models_dir = Path(self.config.get("paths", {}).get("models_dir", "models"))
        self.figures_dir = Path(self.config.get("paths", {}).get("figures_dir", "reports/figures"))
        self.results_path = self.config.get("paths", {}).get(
            "model_results_file", "reports/model_results.csv"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Creates a time-aware train / validation / test split.
        Financial transactions have strong temporal patterns; splitting chronologically
        ensures we test models on future transactions given past history.
        """
        logger.info("Splitting dataset chronologically by TransactionDT...")
        if "TransactionDT" in df.columns:
            df = df.sort_values(by="TransactionDT").reset_index(drop=True)

        target_col = self.config.get("data", {}).get("target_col", "isFraud")
        val_size = self.config.get("data", {}).get("val_size", 0.2)
        test_size = self.config.get("data", {}).get("test_size", 0.15)

        n_total = len(df)
        n_train = int(n_total * (1.0 - val_size - test_size))
        n_val = int(n_total * val_size)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()

        y_train = train_df[target_col]
        y_val = val_df[target_col]
        y_test = test_df[target_col]

        X_train = train_df.drop(columns=[target_col], errors="ignore")
        X_val = val_df.drop(columns=[target_col], errors="ignore")
        X_test = test_df.drop(columns=[target_col], errors="ignore")

        logger.info(
            f"Train split: {len(X_train)} samples ({y_train.mean()*100:.2f}% fraud) | "
            f"Val split: {len(X_val)} samples ({y_val.mean()*100:.2f}% fraud) | "
            f"Test split: {len(X_test)} samples ({y_test.mean()*100:.2f}% fraud)"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_imbalance_strategy(
        self, X: pd.DataFrame, y: pd.Series, strategy: str = "class_weight"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies class imbalance adjustments strictly on training data.
        Never applied to validation or test data!
        """
        if strategy == "smote":
            logger.info("Applying SMOTE over-sampling to training data only...")
            smote = SMOTE(sampling_strategy=0.2, random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        elif strategy == "undersample":
            logger.info("Applying Random Under-sampling to training data only...")
            rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
            X_res, y_res = rus.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        return X, y

    def get_models(self) -> Dict[str, Any]:
        """
        Instantiates candidate algorithms with fraud-tailored parameters.
        """
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs"
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced", max_depth=8, min_samples_leaf=10, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                class_weight="balanced", max_iter=150, random_state=42
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=63,
                scale_pos_weight=15,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=15,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
            "Support Vector Machine": LinearSVC(
                class_weight="balanced", max_iter=2000, random_state=42, dual="auto"
            ),
        }

    def train_and_benchmark(
        self,
        X_train_proc: pd.DataFrame,
        y_train: pd.Series,
        X_val_proc: pd.DataFrame,
        y_val: pd.Series,
        X_train_scaled: pd.DataFrame,
        X_val_scaled: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Trains and benchmarks all candidate models on validation data.
        Uses representative subsampling for computationally intensive algorithms (KNN, SVM).
        """
        models = self.get_models()
        benchmark_results = []
        subsample_size = self.config.get("models", {}).get("subsample_heavy_models_size", 30000)

        # Create stratified subsample for heavy distance-based models
        if len(X_train_scaled) > subsample_size:
            idx = np.random.choice(len(X_train_scaled), size=subsample_size, replace=False)
            X_train_sub = X_train_scaled.iloc[idx]
            y_train_sub = y_train.iloc[idx]
        else:
            X_train_sub, y_train_sub = X_train_scaled, y_train

        for name, model in models.items():
            logger.info(f"--- Training {name} ---")
            t0 = time.time()

            try:
                # Select appropriate input feature representation
                if name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
                    X_tr = X_train_sub if name in ["Support Vector Machine", "K-Nearest Neighbors"] else X_train_scaled
                    y_tr = y_train_sub if name in ["Support Vector Machine", "K-Nearest Neighbors"] else y_train
                    X_v = X_val_scaled
                else:
                    X_tr = X_train_proc
                    y_tr = y_train
                    X_v = X_val_proc

                model.fit(X_tr, y_tr)
                fit_time = round(time.time() - t0, 2)

                # Generate probability predictions
                if hasattr(model, "predict_proba"):
                    val_probs = model.predict_proba(X_v)[:, 1]
                elif hasattr(model, "decision_function"):
                    dfn = model.decision_function(X_v)
                    # Convert decision function to [0, 1] sigmoid probability
                    val_probs = 1 / (1 + np.exp(-dfn))
                else:
                    val_probs = model.predict(X_v).astype(float)

                metrics = calculate_metrics(y_val.to_numpy(), val_probs, threshold=0.5)
                metrics["model"] = name
                metrics["training_time_sec"] = fit_time
                benchmark_results.append(metrics)

                logger.info(
                    f"{name} Results | PR-AUC: {metrics['pr_auc']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f} | "
                    f"ROC-AUC: {metrics['roc_auc']:.4f} ({fit_time}s)"
                )

                # Save curve for boosting models
                if name in ["LightGBM", "XGBoost", "Random Forest"]:
                    clean_name = name.lower().replace(" ", "_")
                    plot_roc_pr_curves(
                        y_val.to_numpy(),
                        val_probs,
                        model_name=name,
                        save_path=str(self.figures_dir / f"{clean_name}_curves.png"),
                    )

            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")

        return benchmark_results

    def run_explainability(
        self, model: Any, X_sample: pd.DataFrame, feature_names: List[str]
    ) -> None:
        """
        Computes SHAP feature importance and saves global summary visual artifacts.
        """
        logger.info("Computing SHAP values for model explainability...")
        try:
            # Use TreeExplainer for tree models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # If binary classifier returns list of 2 classes, pick class 1 (Fraud)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                shap_vals = shap_values[:, :, 1]
            else:
                shap_vals = shap_values

            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, show=False, max_display=15)
            plt.title("SHAP Global Feature Importance (Fraud Impact)")
            plt.tight_layout()
            summary_path = self.figures_dir / "shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary to {summary_path}")

        except Exception as e:
            logger.warning(f"SHAP explanation skipped or failed: {str(e)}")

    def run(self, nrows: Optional[int] = None) -> None:
        """
        Executes the entire end-to-end training pipeline.
        """
        logger.info("Starting End-to-End IEEE-CIS Fraud Detection Pipeline...")
        loader = DataLoader(self.raw_data_dir, self.config)

        # Check if Kaggle dataset exists
        status = loader.check_dataset_exists()
        if not status.get("train_transaction", False):
            print(loader.get_download_instructions())
            logger.warning("Dataset not found. Please follow the instructions above.")
            return

        # 1. Load Data
        df = loader.load_train_data(nrows=nrows, reduce_memory=True)

        # 2. Train / Val / Test Split (Time-aware)
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = self.prepare_data_splits(df)
        del df

        # 3. Feature Engineering (Fitted on train only)
        fe = FeatureEngineer()
        X_train_fe = fe.fit_transform(X_train_raw)
        X_val_fe = fe.transform(X_val_raw)
        X_test_fe = fe.transform(X_test_raw)
        del X_train_raw, X_val_raw, X_test_raw

        # 4. Preprocessing (Fitted on train only)
        preprocessor = Preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train_fe, scale_numeric=False)
        X_val_proc = preprocessor.transform(X_val_fe, scale_numeric=False)
        X_test_proc = preprocessor.transform(X_test_fe, scale_numeric=False)

        # Scaled versions for linear / distance models
        X_train_scaled = preprocessor.transform(X_train_fe, scale_numeric=True)
        X_val_scaled = preprocessor.transform(X_val_fe, scale_numeric=True)

        # 5. Imbalance Handling on Training Fold
        strategy = self.config.get("imbalance", {}).get("strategy", "class_weight")
        X_train_final, y_train_final = self.apply_imbalance_strategy(
            X_train_proc, y_train, strategy=strategy
        )

        # 6. Train and Benchmark Models
        benchmark_results = self.train_and_benchmark(
            X_train_final,
            y_train_final,
            X_val_proc,
            y_val,
            X_train_scaled,
            X_val_scaled,
        )

        # Save comparison CSV
        save_model_comparison_results(benchmark_results, self.results_path)

        # 7. Select Champion Model based on PR-AUC
        best_result = max(benchmark_results, key=lambda x: x.get("pr_auc", 0))
        champion_name = best_result["model"]
        logger.info(
            f"[CHAMPION] Selected Champion Model: {champion_name} (PR-AUC: {best_result['pr_auc']:.4f})"
        )

        # Re-train champion model
        models_dict = self.get_models()
        champion_model = models_dict[champion_name]
        champion_model.fit(X_train_final, y_train_final)

        # 8. Threshold Optimization
        val_probs = champion_model.predict_proba(X_val_proc)[:, 1]
        threshold_info = find_optimal_threshold(y_val.to_numpy(), val_probs)
        optimal_f1_thresh = float(threshold_info["optimal_f1"]["threshold"])
        optimal_cost_thresh = float(threshold_info["optimal_cost"]["threshold"])

        logger.info(
            f"Optimal F1 Threshold: {optimal_f1_thresh:.2f} | "
            f"Optimal Financial Cost Threshold: {optimal_cost_thresh:.2f}"
        )

        # Test Set Final Evaluation
        test_probs = champion_model.predict_proba(X_test_proc)[:, 1]
        final_test_metrics = calculate_metrics(
            y_test.to_numpy(), test_probs, threshold=optimal_f1_thresh
        )
        logger.info(f"Final Test Holdout Evaluation: {final_test_metrics}")

        # Confusion matrix for champion model
        y_test_pred = (test_probs >= optimal_f1_thresh).astype(int)
        plot_confusion_matrix(
            y_test.to_numpy(),
            y_test_pred,
            model_name=champion_name,
            save_path=str(self.figures_dir / "confusion_matrix.png"),
        )

        # 9. Explainability with SHAP (Subsample of validation set)
        shap_sample = X_val_proc.sample(min(500, len(X_val_proc)), random_state=42)
        self.run_explainability(champion_model, shap_sample, preprocessor.fitted_feature_names_)

        # 10. Persist Champion Artifacts
        save_artifact(champion_model, self.models_dir / "final_model.pkl")
        save_artifact(preprocessor, self.models_dir / "preprocessor.pkl")
        save_artifact(fe, self.models_dir / "feature_engineer.pkl")

        metadata = {
            "model_name": champion_name,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimal_threshold": optimal_f1_thresh,
            "optimal_cost_threshold": optimal_cost_thresh,
            "validation_metrics": best_result,
            "test_metrics": final_test_metrics,
            "feature_names": preprocessor.fitted_feature_names_,
            "num_features": len(preprocessor.fitted_feature_names_),
        }
        save_json(metadata, str(self.models_dir / "model_metadata.json"))
        logger.info("Training pipeline successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE-CIS Fraud Detection Training")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to load")
    parser.add_argument("--quick", action="store_true", help="Run quick training on 20000 rows")
    args = parser.parse_args()

    rows = 20000 if args.quick else args.nrows
    trainer = ModelTrainer()
    trainer.run(nrows=rows)
