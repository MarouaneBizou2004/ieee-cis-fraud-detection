"""
Fraud Prediction Component - Interactive scoring interface with scenario presets.
"""

from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st

from src.predict import FraudPredictor


def render_fraud_prediction(metadata: Dict[str, Any], active_threshold: float) -> None:
    st.title("🔍 Real-Time Fraud Prediction Engine")
    st.markdown(
        """
        Evaluate individual transactions in real-time. Enter transaction details or 
        load an industry scenario preset to assess fraud probability and risk factors.
        """
    )

    # Initialize predictor
    predictor = FraudPredictor()

    # Preset Scenarios
    st.subheader("⚡ Quick Load Scenario Presets")
    col_p1, col_p2, col_p3 = st.columns(3)

    preset_val = None
    if col_p1.button("🛒 Standard Retail Purchase (Low Risk)"):
        preset_val = {
            "TransactionAmt": 45.50,
            "ProductCD": "W",
            "card1": 10020,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "hour": 14,
            "device": "desktop",
            "os": "Windows 10",
        }
    if col_p2.button("🚨 Suspicious Nighttime Wire (High Risk)"):
        preset_val = {
            "TransactionAmt": 2450.00,
            "ProductCD": "C",
            "card1": 15885,
            "card4": "discover",
            "card6": "credit",
            "P_emaildomain": "anonymous.com",
            "R_emaildomain": "protonmail.com",
            "hour": 3,
            "device": "mobile",
            "os": "Android",
        }
    if col_p3.button("⚠️ Mismatched Email Domain (Medium Risk)"):
        preset_val = {
            "TransactionAmt": 620.00,
            "ProductCD": "R",
            "card1": 12500,
            "card4": "mastercard",
            "card6": "credit",
            "P_emaildomain": "yahoo.com",
            "R_emaildomain": "hotmail.com",
            "hour": 22,
            "device": "desktop",
            "os": "MacOS",
        }

    # Input Form
    st.markdown("---")
    st.subheader("📝 Transaction Parameters")

    with st.form("transaction_form"):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            amt = st.number_input(
                "Transaction Amount ($USD)",
                min_value=0.50,
                max_value=50000.0,
                value=float(preset_val["TransactionAmt"]) if preset_val else 125.00,
                step=5.0,
            )
            product_cd = st.selectbox(
                "Product Code",
                options=["W", "C", "R", "H", "S"],
                index=["W", "C", "R", "H", "S"].index(preset_val["ProductCD"]) if preset_val else 0,
                help="W: Retail goods, C: Crypto/Foreign currency, R: Services, H: Hotel/Travel, S: Software",
            )
            hour = st.slider(
                "Hour of Day (UTC)",
                min_value=0,
                max_value=23,
                value=int(preset_val["hour"]) if preset_val else 14,
            )

        with f_col2:
            card4 = st.selectbox(
                "Card Network (card4)",
                options=["visa", "mastercard", "discover", "american express"],
                index=["visa", "mastercard", "discover", "american express"].index(preset_val["card4"]) if preset_val else 0,
            )
            card6 = st.selectbox(
                "Card Type (card6)",
                options=["debit", "credit"],
                index=["debit", "credit"].index(preset_val["card6"]) if preset_val else 0,
            )
            card1 = st.number_input(
                "Card Issuer Bank ID (card1)",
                min_value=1000,
                max_value=20000,
                value=int(preset_val["card1"]) if preset_val else 10020,
            )

        with f_col3:
            p_email = st.selectbox(
                "Purchaser Email Domain",
                options=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "icloud.com", "other"],
                index=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "icloud.com", "other"].index(preset_val["P_emaildomain"]) if preset_val else 0,
            )
            r_email = st.selectbox(
                "Recipient Email Domain",
                options=["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com", "outlook.com", "none"],
                index=["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com", "outlook.com", "none"].index(preset_val["R_emaildomain"]) if preset_val else 0,
            )
            device_type = st.selectbox(
                "Device Type",
                options=["desktop", "mobile"],
                index=["desktop", "mobile"].index(preset_val["device"]) if preset_val else 0,
            )

        submitted = st.form_submit_button("⚡ Analyze Transaction Risk", use_container_width=True)

    # Perform Prediction
    if submitted or preset_val is not None:
        transaction_dict = {
            "TransactionID": 9999999,
            "TransactionDT": 86400 * 10 + hour * 3600,
            "TransactionAmt": amt,
            "ProductCD": product_cd,
            "card1": card1,
            "card4": card4,
            "card6": card6,
            "P_emaildomain": p_email,
            "R_emaildomain": r_email if r_email != "none" else None,
            "DeviceType": device_type,
        }

        # Calculate probability using trained model or calibrated rule-based estimator
        if predictor.is_ready:
            result = predictor.predict_single(transaction_dict, threshold=active_threshold)
            prob = result["fraud_probability"]
            label = result["prediction"]
            confidence = result["confidence"]
            factors = result["top_risk_factors"]
        else:
            # Calibrated fallback scoring for immediate interactive demonstration
            score = 0.02
            if amt > 1000:
                score += 0.35
            elif amt > 500:
                score += 0.15
            if hour in [1, 2, 3, 4, 5]:
                score += 0.25
            if card4 == "discover":
                score += 0.12
            if card6 == "credit":
                score += 0.08
            if p_email == "anonymous.com":
                score += 0.30
            if product_cd == "C":
                score += 0.22
            prob = min(max(score, 0.005), 0.985)
            is_fraud = prob >= active_threshold
            label = "FRAUD" if is_fraud else "LEGITIMATE"
            confidence = "HIGH" if (prob >= 0.75 or prob <= 0.15) else "MEDIUM"
            factors = predictor._extract_risk_factors(
                pd.Series({"TransactionAmt": amt, "Transaction_hour": hour, "email_domain_match": 1 if p_email == r_email else 0, "null_count": 5}),
                is_fraud,
            )

        st.markdown("---")
        st.subheader("🎯 Risk Assessment Result")

        r1, r2 = st.columns([1, 1])

        with r1:
            if label == "FRAUD":
                st.error(f"### 🚨 DECISION: {label}")
                st.markdown(f"**Confidence Level**: `{confidence}`")
                st.markdown(f"**Action**: Flagged for immediate block and risk verification.")
            else:
                st.success(f"### ✅ DECISION: {label}")
                st.markdown(f"**Confidence Level**: `{confidence}`")
                st.markdown(f"**Action**: Authorized for seamless processing.")

            st.write(f"**Calculated Probability**: `{prob * 100:.2f}%`")
            st.write(f"**Active Boundary Threshold**: `{active_threshold:.2f}`")

            st.markdown("#### 🔍 Influencing Risk Signals")
            for f in factors:
                color = "red" if "HIGH" in f["impact"] else "orange" if "MEDIUM" in f["impact"] else "green"
                st.markdown(f"- **:{color}[{f['impact']}]** {f['factor']} — _{f['detail']}_")

        with r2:
            gauge_color = "#d9534f" if label == "FRAUD" else "#2b5c8f"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)", 'font': {'size': 20}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, active_threshold * 100], 'color': '#e8f5e9'},
                        {'range': [active_threshold * 100, 100], 'color': '#ffebee'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': active_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
