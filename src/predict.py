"""
Inference and Prediction Pipeline for IEEE-CIS Fraud Detection.
Handles raw transaction input, executes feature engineering, preprocessing,
and yields fraud probability, classification decision, and confidence scores.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils import load_artifact, load_json, setup_logger

logger = setup_logger("predict")


class FraudPredictor:
    """
    Production inference engine. Loads serialized model artifacts and performs
    end-to-end inference on single transactions or batches.
    """

    def __init__(
        self,
        model_path: str = "models/final_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
        fe_path: str = "models/feature_engineer.pkl",
        metadata_path: str = "models/model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.fe_path = Path(fe_path)
        self.metadata_path = Path(metadata_path)

        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.metadata = {}
        self.threshold = 0.35  # Default threshold optimized for fraud detection

        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """
        Loads all required serialized artifacts.
        """
        if self.model_path.exists():
            self.model = load_artifact(str(self.model_path))
            logger.info("Loaded final model artifact.")
        else:
            logger.warning(f"Model artifact not found at {self.model_path}")

        if self.preprocessor_path.exists():
            self.preprocessor = load_artifact(str(self.preprocessor_path))
            logger.info("Loaded preprocessor artifact.")
        else:
            logger.warning(f"Preprocessor artifact not found at {self.preprocessor_path}")

        if self.fe_path.exists():
            self.feature_engineer = load_artifact(str(self.fe_path))
            logger.info("Loaded feature engineering artifact.")
        else:
            logger.warning(f"Feature engineering artifact not found at {self.fe_path}")

        if self.metadata_path.exists():
            self.metadata = load_json(str(self.metadata_path))
            self.threshold = self.metadata.get("optimal_threshold", 0.35)
            logger.info(f"Loaded metadata. Active decision threshold: {self.threshold:.2f}")

    @property
    def is_ready(self) -> bool:
        """
        Returns True if all components are loaded and ready for inference.
        """
        return (
            self.model is not None
            and self.preprocessor is not None
            and self.feature_engineer is not None
        )

    def predict_single(
        self,
        transaction: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on a single transaction dictionary.
        """
        df = pd.DataFrame([transaction])
        results = self.predict_batch(df, threshold=threshold)
        return results[0]

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs batch prediction on a pandas DataFrame of transactions.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Prediction pipeline is not fully initialized. "
                "Ensure models are trained and saved in models/ directory."
            )

        active_thresh = threshold if threshold is not None else self.threshold

        # Step 1: Feature Engineering
        df_fe = self.feature_engineer.transform(df)

        # Step 2: Preprocessing
        df_proc = self.preprocessor.transform(df_fe, scale_numeric=False)

        # Step 3: Model Prediction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_proc)[:, 1]
        elif hasattr(self.model, "decision_function"):
            dfn = self.model.decision_function(df_proc)
            probs = 1 / (1 + np.exp(-dfn))
        else:
            probs = self.model.predict(df_proc).astype(float)

        output = []
        for i, prob in enumerate(probs):
            prob = float(prob)
            is_fraud = prob >= active_thresh
            prediction_label = "FRAUD" if is_fraud else "LEGITIMATE"

            # Determine confidence level
            if is_fraud:
                confidence = "HIGH" if prob >= 0.75 else "MEDIUM"
            else:
                confidence = "HIGH" if prob <= 0.20 else "MEDIUM"

            # Compute top risk signals
            top_factors = self._extract_risk_factors(df_proc.iloc[i], is_fraud)

            record = {
                "fraud_probability": round(prob, 4),
                "fraud_probability_pct": f"{prob * 100:.1f}%",
                "prediction": prediction_label,
                "confidence": confidence,
                "threshold_used": round(active_thresh, 3),
                "top_risk_factors": top_factors,
            }
            output.append(record)

        return output

    def _extract_risk_factors(
        self, feature_row: pd.Series, is_fraud: bool, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Provides intuitive risk signals for a prediction.
        """
        factors = []
        # Key domain feature checks
        if "TransactionAmt" in feature_row.index and feature_row["TransactionAmt"] > 1000:
            factors.append({
                "factor": "High Transaction Amount",
                "detail": f"${feature_row['TransactionAmt']:,.2f} is significantly above median",
                "impact": "HIGH RISK"
            })
        if "Transaction_hour" in feature_row.index and feature_row["Transaction_hour"] in [1, 2, 3, 4, 5]:
            factors.append({
                "factor": "Late Night / Off-Hours Activity",
                "detail": f"Transaction executed at {int(feature_row['Transaction_hour'])}:00 UTC",
                "impact": "MEDIUM RISK"
            })
        if "email_domain_match" in feature_row.index and feature_row["email_domain_match"] == 0:
            factors.append({
                "factor": "P_email / R_email Mismatch",
                "detail": "Purchaser and recipient domains differ",
                "impact": "MEDIUM RISK"
            })
        if "null_count" in feature_row.index and feature_row["null_count"] > 10:
            factors.append({
                "factor": "Sparse Identity Profile",
                "detail": f"{int(feature_row['null_count'])} unverified identity attributes",
                "impact": "ELEVATED RISK"
            })

        if not factors:
            factors.append({
                "factor": "Standard Behavioral Baseline",
                "detail": "Transaction parameters align with verified customer history",
                "impact": "LOW RISK" if not is_fraud else "NEUTRAL"
            })

        return factors[:top_k]


def predict_cli(sample_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    CLI helper for testing predictions.
    """
    predictor = FraudPredictor()
    if not predictor.is_ready:
        print("[!] Model artifacts not found. Please train models first using: python -m src.train")
        return

    sample = sample_dict or {
        "TransactionID": 3999999,
        "TransactionDT": 86400 * 12 + 3600 * 3,  # 3:00 AM on day 12
        "TransactionAmt": 1250.00,
        "ProductCD": "W",
        "card1": 10020,
        "card2": 555.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "P_emaildomain": "anonymous.com",
        "R_emaildomain": "gmail.com",
    }

    result = predictor.predict_single(sample)
    print("\n================ PREDICTION RESULT ================")
    print(f"Fraud Probability : {result['fraud_probability_pct']} ({result['fraud_probability']})")
    print(f"Prediction        : {result['prediction']}")
    print(f"Confidence        : {result['confidence']}")
    print(f"Decision Threshold: {result['threshold_used']}")
    print("Risk Factors:")
    for f in result["top_risk_factors"]:
        print(f"  - [{f['impact']}] {f['factor']}: {f['detail']}")
    print("===================================================\n")


if __name__ == "__main__":
    predict_cli()
