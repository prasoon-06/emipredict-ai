import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance & MLflow Tracking")

results_dir = ROOT / "mlflow_results"
clf_files = sorted(results_dir.glob("clf_*.json"))
reg_files = sorted(results_dir.glob("reg_*.json"))

if not clf_files or not reg_files:
    st.error("No training results found yet. Run the training scripts in src/models/ first.")
    st.stop()

clf_df = pd.DataFrame([json.loads(p.read_text()) for p in clf_files]).sort_values("accuracy", ascending=False)
reg_df = pd.DataFrame([json.loads(p.read_text()) for p in reg_files]).sort_values("rmse")

st.subheader("Classification models (emi_eligibility)")
st.dataframe(clf_df.set_index("model").style.highlight_max(subset=["accuracy", "f1", "roc_auc"], color="#d4f7d4"),
             use_container_width=True)
st.bar_chart(clf_df.set_index("model")["accuracy"])
st.success(f"Champion: **{clf_df.iloc[0]['model']}** — accuracy {clf_df.iloc[0]['accuracy']:.4f}")

st.subheader("Regression models (max_monthly_emi)")
st.dataframe(reg_df.set_index("model").style.highlight_min(subset=["rmse", "mae"], color="#d4f7d4"),
             use_container_width=True)
st.bar_chart(reg_df.set_index("model")["rmse"])
st.success(f"Champion: **{reg_df.iloc[0]['model']}** — RMSE {reg_df.iloc[0]['rmse']:.1f}")

st.divider()
st.subheader("Full MLflow tracking")
st.markdown(
    """
Every run above (params, metrics, and the serialized pipeline artifact) is logged in MLflow
using a local SQLite backend. To browse the full experiment UI (parallel-coordinates plots,
per-run artifacts, model registry):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Then open the printed local URL in a browser.
    """
)
