import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.model_utils import load_raw_data

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")

df = load_raw_data()
if df is None:
    st.error("No dataset found at data/raw/EMI_dataset.csv.")
    st.stop()

st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

c1, c2 = st.columns(2)
with c1:
    st.subheader("EMI eligibility distribution")
    st.bar_chart(df["emi_eligibility"].value_counts())
with c2:
    st.subheader("EMI scenario distribution")
    st.bar_chart(df["emi_scenario"].value_counts())

c3, c4 = st.columns(2)
with c3:
    st.subheader("Max monthly EMI distribution")
    st.bar_chart(df["max_monthly_emi"].value_counts(bins=20).sort_index())
with c4:
    st.subheader("Credit score distribution")
    st.bar_chart(df["credit_score"].value_counts(bins=20).sort_index())

st.subheader("Correlation with max_monthly_emi (numeric features)")
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr(numeric_only=True)["max_monthly_emi"].drop("max_monthly_emi").sort_values()
st.bar_chart(corr)

st.subheader("Raw data sample")
st.dataframe(df.sample(min(200, len(df)), random_state=1), use_container_width=True)
