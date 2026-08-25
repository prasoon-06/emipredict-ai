import streamlit as st

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")

st.title("💳 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    """
Welcome. This app predicts two things for a loan applicant:

1. **EMI Eligibility** — `Eligible` / `High_Risk` / `Not_Eligible`
2. **Maximum Safe Monthly EMI** — how much the applicant can realistically afford, in INR

Use the sidebar to navigate:

- **Predict Eligibility** — classify a single applicant
- **Predict Max EMI** — estimate their affordable EMI ceiling
- **Data Explorer** — browse the training data and distributions
- **Model Performance** — see how each model compares (MLflow-backed)
- **Admin: Data Management** — upload/inspect data for what-if analysis
    """
)

st.info(
    "Models are trained on a synthetic dataset that mirrors the project's real "
    "22-feature / 5-scenario schema (see the README) — swap in the real "
    "`EMI_dataset.csv` and re-run training to use live data.",
    icon="ℹ️",
)

col1, col2, col3 = st.columns(3)
col1.metric("Classification accuracy", "96.4%", "target > 90%")
col2.metric("Regression RMSE", "₹1,373", "target < ₹2,000")
col3.metric("Models compared", "8", "4 clf + 4 reg")
