"""
Fraud Detection Intelligence - Production Streamlit Web Application.
Main entry point coordinating modular UI components.
"""

from pathlib import Path
import streamlit as st

from app.components.about_model import render_about_model
from app.components.dashboard import render_dashboard
from app.components.data_insights import render_data_insights
from app.components.feature_importance import render_feature_importance
from app.components.fraud_prediction import render_fraud_prediction
from app.components.model_performance import render_model_performance
from src.utils import load_config, load_json, setup_logger

logger = setup_logger("app")

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_application_state():
    """
    Loads saved metadata and configuration safely.
    """
    config = load_config("config/config.yaml")
    metadata_path = Path("models/model_metadata.json")
    metadata = load_json(str(metadata_path)) if metadata_path.exists() else {}
    return config, metadata


def main():
    config, metadata = load_application_state()

    # Sidebar Navigation & Global Controls
    st.sidebar.title("🛡️ Fraud Intelligence")
    st.sidebar.caption("Production ML Risk Engine")

    # Global Decision Boundary Threshold
    default_thresh = float(metadata.get("optimal_threshold", 0.35))
    if "decision_threshold" not in st.session_state:
        st.session_state["decision_threshold"] = default_thresh

    threshold = st.sidebar.slider(
        "🎯 Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=st.session_state["decision_threshold"],
        step=0.01,
        help="Probability threshold for classifying a transaction as fraudulent.",
    )
    st.session_state["decision_threshold"] = threshold

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation Menu",
        options=[
            "🏠 Dashboard",
            "📈 Model Performance",
            "🔍 Fraud Prediction",
            "🧠 Feature Importance",
            "📊 Data Insights",
            "ℹ️ About Model",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    raw_tx_path = Path("data/raw/train_transaction.csv")
    if raw_tx_path.exists():
        st.sidebar.success("🟢 Kaggle Dataset: Connected")
    else:
        st.sidebar.warning("🟡 Kaggle Dataset: Not in data/raw")

    model_path = Path("models/final_model.pkl")
    if model_path.exists():
        st.sidebar.success("🟢 Model Engine: Loaded")
    else:
        st.sidebar.info("🔵 Model Engine: Demo Mode")

    st.sidebar.markdown("---")
    st.sidebar.caption("IEEE-CIS Fraud Detection • v1.0.0")

    # Page Routing
    if page == "🏠 Dashboard":
        render_dashboard(metadata, None, threshold)
    elif page == "📈 Model Performance":
        render_model_performance(metadata)
    elif page == "🔍 Fraud Prediction":
        render_fraud_prediction(metadata, threshold)
    elif page == "🧠 Feature Importance":
        render_feature_importance(metadata)
    elif page == "📊 Data Insights":
        render_data_insights()
    elif page == "ℹ️ About Model":
        render_about_model()


if __name__ == "__main__":
    main()
