import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.build_features import add_features, CATEGORICAL_COLS, NUMERIC_BASE_COLS, ENGINEERED_COLS  # noqa: E402

NUMERIC_COLS = NUMERIC_BASE_COLS + ENGINEERED_COLS

EMI_SCENARIOS = {
    "E-commerce Shopping EMI": {"amount": (10_000, 200_000), "tenure": (3, 24)},
    "Home Appliances EMI": {"amount": (20_000, 300_000), "tenure": (6, 36)},
    "Vehicle EMI": {"amount": (80_000, 1_500_000), "tenure": (12, 84)},
    "Personal Loan EMI": {"amount": (50_000, 1_000_000), "tenure": (12, 60)},
    "Education EMI": {"amount": (50_000, 500_000), "tenure": (6, 48)},
}


@st.cache_resource
def load_classifier():
    path = ROOT / "models" / "best_classifier.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_regressor():
    path = ROOT / "models" / "best_regressor.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_raw_data():
    path = ROOT / "data" / "raw" / "EMI_dataset.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def applicant_to_frame(applicant: dict) -> pd.DataFrame:
    """Turn a single applicant dict (raw fields) into the engineered-feature row the models expect."""
    df = pd.DataFrame([applicant])
    df = add_features(df)
    return df[NUMERIC_COLS + CATEGORICAL_COLS]
