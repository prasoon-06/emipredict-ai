import sys
from pathlib import Path
import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

MODELS_MISSING = not (ROOT / "models" / "best_classifier.pkl").exists()


@pytest.mark.skipif(MODELS_MISSING, reason="run training first: see README Quickstart")
def test_classifier_predicts_known_label():
    from components.model_utils import applicant_to_frame

    bundle = joblib.load(ROOT / "models" / "best_classifier.pkl")
    pipe, le = bundle["pipeline"], bundle["label_encoder"]

    applicant = {
        "age": 30, "gender": "Male", "marital_status": "Single", "education": "Graduate",
        "monthly_salary": 50000, "employment_type": "Private", "years_of_employment": 4.0,
        "company_type": "MNC", "house_type": "Rented", "monthly_rent": 10000, "family_size": 2,
        "dependents": 0, "school_fees": 0, "college_fees": 0, "travel_expenses": 2000,
        "groceries_utilities": 6000, "other_monthly_expenses": 1500, "existing_loans": "No",
        "current_emi_amount": 0, "credit_score": 720, "bank_balance": 80000, "emergency_fund": 30000,
        "emi_scenario": "Personal Loan EMI", "requested_amount": 100000, "requested_tenure": 24,
    }
    X = applicant_to_frame(applicant)
    pred = pipe.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    assert label in {"Eligible", "High_Risk", "Not_Eligible"}


@pytest.mark.skipif(MODELS_MISSING, reason="run training first: see README Quickstart")
def test_regressor_predicts_within_target_bounds():
    from components.model_utils import applicant_to_frame

    bundle = joblib.load(ROOT / "models" / "best_regressor.pkl")
    pipe = bundle["pipeline"]

    applicant = {
        "age": 30, "gender": "Male", "marital_status": "Single", "education": "Graduate",
        "monthly_salary": 50000, "employment_type": "Private", "years_of_employment": 4.0,
        "company_type": "MNC", "house_type": "Rented", "monthly_rent": 10000, "family_size": 2,
        "dependents": 0, "school_fees": 0, "college_fees": 0, "travel_expenses": 2000,
        "groceries_utilities": 6000, "other_monthly_expenses": 1500, "existing_loans": "No",
        "current_emi_amount": 0, "credit_score": 720, "bank_balance": 80000, "emergency_fund": 30000,
        "emi_scenario": "Personal Loan EMI", "requested_amount": 100000, "requested_tenure": 24,
    }
    X = applicant_to_frame(applicant)
    pred = pipe.predict(X)[0]
    assert 500 <= pred <= 50000
