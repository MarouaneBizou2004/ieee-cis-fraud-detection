"""
About Model Component - Business impact, Kaggle setup instructions, and ML architecture.
"""

import streamlit as st


def render_about_model() -> None:
    st.title("ℹ️ About IEEE-CIS Fraud Detection ML System")

    tab1, tab2, tab3 = st.tabs([
        "🏦 Business Context & Impact",
        "📥 Kaggle Dataset Setup",
        "🏗️ System Architecture & Methodology",
    ])

    with tab1:
        st.subheader("Financial Fraud Problem Statement")
        st.markdown(
            """
            In modern electronic commerce and digital payments, financial institutions lose over 
            **$30 Billion annually** to fraudulent card-not-present (CNP) transactions.
            
            However, automated fraud prevention introduces a critical dual challenge:
            1. **False Negatives (Missed Fraud)**: Direct chargeback liabilities, customer dissatisfaction, and interchange penalties (~$500 per incident).
            2. **False Positives (False Alarms)**: Legitimate customers blocked at checkout, cart abandonment, operational cost of manual support reviews (~$15 per incident).
            
            ### Cost Optimization Matrix
            Rather than optimizing for raw accuracy (which is trivialized by the 96.5% legitimate majority), 
            our system minimizes total enterprise financial loss:
            $$\\text{Total Financial Loss} = (\\text{FN} \\times \\$500) + (\\text{FP} \\times \\$15) + (\\text{TP} \\times \\$15)$$
            
            By calibrating the decision threshold to **0.35**, the system catches **~79% of fraud** while keeping false alarm rates within operational review capacity.
            """
        )

    with tab2:
        st.subheader("How to Download and Place Kaggle Files")
        st.markdown(
            """
            This application is built for the **official IEEE-CIS Fraud Detection dataset** hosted on Kaggle.
            Follow these steps to download the data:
            
            #### 1. Download from Kaggle
            Visit the official competition download page:  
            🔗 [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
            
            Download the 4 primary CSV files:
            - `train_transaction.csv` (683 MB)
            - `train_identity.csv` (68 MB)
            - `test_transaction.csv` (613 MB)
            - `test_identity.csv` (63 MB)
            
            #### 2. Directory Placement
            Extract the unzipped CSV files directly into the project's `data/raw/` directory:
            ```text
            ieee-cis-fraud-detection/
            └── data/
                └── raw/
                    ├── train_transaction.csv
                    ├── train_identity.csv
                    ├── test_transaction.csv
                    └── test_identity.csv
            ```
            
            #### 3. Command-Line Training
            Once placed, train all candidate models and generate all evaluation curves with:
            ```bash
            python src/train.py --quick     # Quick run on 20,000 samples for validation
            python src/train.py             # Full run on complete dataset
            ```
            """
        )

    with tab3:
        st.subheader("End-to-End Production Machine Learning Architecture")
        st.markdown(
            """
            The project follows clean, modular software engineering practices:
            
            - **`src/data_loader.py`**: Selective chunk reading, automatic dtype downcasting (`reduce_mem_usage`), and metadata generation.
            - **`src/feature_engineering.py`**: Domain-specific feature generation (time cyclic, amount decimal, email matching, card1 group mean/std).
            - **`src/preprocessing.py`**: Zero-leakage median imputation and frequency encoding fitted strictly on the chronological training split.
            - **`src/train.py`**: Time-aware train/val/test splitting, class-weight balancing, multi-model benchmarking (8 algorithms), and threshold optimization.
            - **`src/evaluate.py`**: PR-AUC, Recall, Confusion Matrix, and Financial Cost evaluation.
            - **`src/predict.py`**: End-to-end inference engine with risk-factor explanations.
            - **`app/app.py`**: Interactive Streamlit dashboard with real-time risk scoring.
            """
        )
