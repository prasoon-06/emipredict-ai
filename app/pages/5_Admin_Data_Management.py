import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.model_utils import load_raw_data

st.set_page_config(page_title="Admin: Data Management", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin: Data Management")
st.caption("CRUD-style tools for the working dataset (session-only — changes aren't written back to disk).")

if "admin_df" not in st.session_state:
    base = load_raw_data()
    st.session_state.admin_df = base.copy() if base is not None else pd.DataFrame()

df = st.session_state.admin_df

tab_view, tab_add, tab_update, tab_delete, tab_upload = st.tabs(
    ["View", "Add record", "Update record", "Delete record", "Bulk upload"]
)

with tab_view:
    st.dataframe(df, use_container_width=True, height=400)
    st.caption(f"{len(df):,} records in the working set")

with tab_add:
    st.write("Add a single record via CSV row (comma-separated, matching the dataset's column order).")
    new_row = st.text_input("New row (CSV)")
    if st.button("Add") and new_row:
        try:
            values = new_row.split(",")
            if len(values) == len(df.columns):
                st.session_state.admin_df.loc[len(df)] = values
                st.success("Record added.")
            else:
                st.error(f"Expected {len(df.columns)} values, got {len(values)}.")
        except Exception as e:
            st.error(f"Could not add record: {e}")

with tab_update:
    if len(df) > 0:
        idx = st.number_input("Row index to update", 0, len(df) - 1, 0)
        col = st.selectbox("Column", df.columns)
        new_val = st.text_input("New value")
        if st.button("Update"):
            st.session_state.admin_df.loc[idx, col] = new_val
            st.success(f"Updated row {idx}, column '{col}'.")

with tab_delete:
    if len(df) > 0:
        idx = st.number_input("Row index to delete", 0, len(df) - 1, 0, key="del_idx")
        if st.button("Delete", type="secondary"):
            st.session_state.admin_df = st.session_state.admin_df.drop(index=idx).reset_index(drop=True)
            st.success(f"Deleted row {idx}.")

with tab_upload:
    uploaded = st.file_uploader("Upload a CSV to replace the working set", type="csv")
    if uploaded is not None:
        st.session_state.admin_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(st.session_state.admin_df):,} records.")
