import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.model_utils import load_classifier, applicant_to_frame, EMI_SCENARIOS

st.set_page_config(page_title="Predict Eligibility", page_icon="✅", layout="wide")
st.title("✅ Predict EMI Eligibility")

bundle = load_classifier()
if bundle is None:
    st.error("No trained classifier found. Run `python src/models/train_one.py clf ...` "
              "for each model, then `python src/models/finalize.py`.")
    st.stop()

pipeline, label_encoder, model_name = bundle["pipeline"], bundle["label_encoder"], bundle["model_name"]
st.caption(f"Serving model: **{model_name}**")

with st.form("applicant_form"):
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 25, 60, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type = st.selectbox("Employment type", ["Private", "Government", "Self-employed"])
        company_type = st.selectbox("Company type", ["Startup", "SME", "MNC", "Government"])
        years_of_employment = st.number_input("Years of employment", 0.0, 35.0, 5.0)
    with c2:
        monthly_salary = st.number_input("Monthly salary (INR)", 15_000, 200_000, 45_000, step=1000)
        house_type = st.selectbox("House type", ["Rented", "Own", "Family"])
        monthly_rent = st.number_input("Monthly rent (INR)", 0, 100_000, 8_000, step=500)
        family_size = st.number_input("Family size", 1, 10, 3)
        dependents = st.number_input("Dependents", 0, 6, 1)
        existing_loans = st.selectbox("Existing loans?", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI amount (INR)", 0, 100_000, 0, step=500)
    with c3:
        credit_score = st.slider("Credit score", 300, 850, 700)
        bank_balance = st.number_input("Bank balance (INR)", 0, 5_000_000, 60_000, step=1000)
        emergency_fund = st.number_input("Emergency fund (INR)", 0, 2_000_000, 20_000, step=1000)
        school_fees = st.number_input("School fees (INR)", 0, 50_000, 0, step=500)
        college_fees = st.number_input("College fees (INR)", 0, 50_000, 0, step=500)
        travel_expenses = st.number_input("Travel expenses (INR)", 0, 50_000, 3_000, step=500)
        groceries_utilities = st.number_input("Groceries & utilities (INR)", 0, 100_000, 8_000, step=500)
        other_monthly_expenses = st.number_input("Other monthly expenses (INR)", 0, 50_000, 2_000, step=500)

    st.subheader("Requested loan")
    c4, c5, c6 = st.columns(3)
    emi_scenario = c4.selectbox("EMI scenario", list(EMI_SCENARIOS.keys()))
    amt_lo, amt_hi = EMI_SCENARIOS[emi_scenario]["amount"]
    ten_lo, ten_hi = EMI_SCENARIOS[emi_scenario]["tenure"]
    requested_amount = c5.number_input("Requested amount (INR)", amt_lo, amt_hi, amt_lo, step=1000)
    requested_tenure = c6.number_input("Requested tenure (months)", ten_lo, ten_hi, ten_lo)

    submitted = st.form_submit_button("Predict eligibility", type="primary")

if submitted:
    applicant = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=monthly_salary, employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
        dependents=dependents, school_fees=school_fees, college_fees=college_fees,
        travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
        other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
        current_emi_amount=current_emi_amount, credit_score=credit_score,
        bank_balance=bank_balance, emergency_fund=emergency_fund, emi_scenario=emi_scenario,
        requested_amount=requested_amount, requested_tenure=requested_tenure,
    )
    X = applicant_to_frame(applicant)
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    label = label_encoder.inverse_transform([pred])[0]

    colors = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}
    st.markdown(f"## Result: :{colors.get(label,'blue')}[{label}]")

    prob_df = {label_encoder.inverse_transform([i])[0]: p for i, p in enumerate(proba)}
    st.bar_chart(prob_df)
