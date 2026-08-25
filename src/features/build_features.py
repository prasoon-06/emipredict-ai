"""Derived financial features used across both the training pipeline and the app."""
import pandas as pd
import numpy as np


CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]

NUMERIC_BASE_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "requested_amount", "requested_tenure",
]

ENGINEERED_COLS = [
    "total_monthly_expenses", "debt_to_income_ratio", "expense_to_income_ratio",
    "affordability_ratio", "risk_score", "proposed_emi",
]

FEATURE_COLS = NUMERIC_BASE_COLS + ENGINEERED_COLS + CATEGORICAL_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    salary = df["monthly_salary"].replace(0, np.nan)

    df["total_monthly_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] + df["travel_expenses"]
        + df["groceries_utilities"] + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    df["debt_to_income_ratio"] = (df["current_emi_amount"] / salary).fillna(0)
    df["expense_to_income_ratio"] = (df["total_monthly_expenses"] / salary).fillna(0)
    df["affordability_ratio"] = ((salary - df["total_monthly_expenses"]) / salary).fillna(0)

    # simple composite risk score (0-100, higher = safer) from credit score + stability + affordability
    credit_component = (df["credit_score"] - 300) / 550 * 50
    stability_component = np.clip(df["years_of_employment"] / 15, 0, 1) * 25
    affordability_component = np.clip(df["affordability_ratio"], 0, 1) * 25
    df["risk_score"] = (credit_component + stability_component + affordability_component).round(1)

    df["proposed_emi"] = (df["requested_amount"] / df["requested_tenure"]).round(1)

    return df
