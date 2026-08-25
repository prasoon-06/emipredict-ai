import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.build_features import add_features, FEATURE_COLS  # noqa: E402


def _sample_row():
    return {
        "age": 30, "gender": "Male", "marital_status": "Single", "education": "Graduate",
        "monthly_salary": 50000, "employment_type": "Private", "years_of_employment": 4.0,
        "company_type": "MNC", "house_type": "Rented", "monthly_rent": 10000, "family_size": 2,
        "dependents": 0, "school_fees": 0, "college_fees": 0, "travel_expenses": 2000,
        "groceries_utilities": 6000, "other_monthly_expenses": 1500, "existing_loans": "No",
        "current_emi_amount": 0, "credit_score": 720, "bank_balance": 80000, "emergency_fund": 30000,
        "emi_scenario": "Personal Loan EMI", "requested_amount": 100000, "requested_tenure": 24,
    }


def test_add_features_creates_expected_columns():
    df = pd.DataFrame([_sample_row()])
    out = add_features(df)
    for col in FEATURE_COLS:
        assert col in out.columns, f"missing engineered/base column: {col}"


def test_affordability_ratio_bounded():
    df = pd.DataFrame([_sample_row()])
    out = add_features(df)
    assert 0 <= out.loc[0, "affordability_ratio"] <= 1


def test_zero_salary_does_not_crash():
    row = _sample_row()
    row["monthly_salary"] = 0
    df = pd.DataFrame([row])
    out = add_features(df)  # should not raise / produce inf
    assert out.loc[0, "debt_to_income_ratio"] == 0
