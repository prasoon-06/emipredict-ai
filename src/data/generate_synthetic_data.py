"""
Generates a synthetic EMI dataset that mirrors the schema in the project brief:
22 input features, 5 EMI scenarios, 2 targets (emi_eligibility, max_monthly_emi).

This stands in for the real 400K-record dataset (distributed via a Google Drive
link the environment can't fetch). Swap this out for the real CSV by dropping it
at data/raw/EMI_dataset.csv and skipping this script.

Usage: python generate_synthetic_data.py [n_rows] [out_path]
"""
import sys
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SCENARIOS = {
    "E-commerce Shopping EMI": {"amount": (10_000, 200_000), "tenure": (3, 24), "share": 0.2},
    "Home Appliances EMI":     {"amount": (20_000, 300_000), "tenure": (6, 36), "share": 0.2},
    "Vehicle EMI":             {"amount": (80_000, 1_500_000), "tenure": (12, 84), "share": 0.2},
    "Personal Loan EMI":       {"amount": (50_000, 1_000_000), "tenure": (12, 60), "share": 0.2},
    "Education EMI":          {"amount": (50_000, 500_000), "tenure": (6, 48), "share": 0.2},
}

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]
EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
COMPANY_TYPES = ["Startup", "SME", "MNC", "Government"]
HOUSE_TYPES = ["Rented", "Own", "Family"]


def generate(n: int) -> pd.DataFrame:
    age = RNG.integers(25, 61, n)
    gender = RNG.choice(["Male", "Female"], n)
    marital_status = RNG.choice(["Single", "Married"], n, p=[0.4, 0.6])
    education = RNG.choice(EDUCATION_LEVELS, n, p=[0.15, 0.45, 0.30, 0.10])

    edu_bump = pd.Series(education).map(
        {"High School": 0, "Graduate": 0.15, "Post Graduate": 0.3, "Professional": 0.45}
    ).values
    monthly_salary = np.clip(
        (15_000 + np.exp(RNG.normal(10.3, 0.45, n)) * (1 + edu_bump) * (1 + (age - 25) / 200)),
        15_000, 200_000,
    ).round(-2)

    employment_type = RNG.choice(EMPLOYMENT_TYPES, n, p=[0.55, 0.20, 0.25])
    years_of_employment = np.clip((age - 22) * RNG.uniform(0.2, 0.9, n), 0, 35).round(1)
    company_type = RNG.choice(COMPANY_TYPES, n)

    house_type = RNG.choice(HOUSE_TYPES, n, p=[0.45, 0.35, 0.20])
    monthly_rent = np.where(
        house_type == "Rented", (monthly_salary * RNG.uniform(0.1, 0.3, n)).round(-2), 0
    )
    family_size = RNG.integers(1, 9, n)
    dependents = np.clip((family_size - RNG.integers(1, 3, n)), 0, 6)

    school_fees = np.where(dependents > 0, RNG.uniform(0, 8_000, n) * (dependents > 0), 0).round(-1)
    college_fees = np.where(dependents > 1, RNG.uniform(0, 15_000, n), 0).round(-1)
    travel_expenses = (monthly_salary * RNG.uniform(0.02, 0.08, n)).round(-1)
    groceries_utilities = (monthly_salary * RNG.uniform(0.08, 0.18, n) + family_size * 800).round(-1)
    other_monthly_expenses = (monthly_salary * RNG.uniform(0.02, 0.10, n)).round(-1)

    existing_loans = RNG.choice(["Yes", "No"], n, p=[0.35, 0.65])
    current_emi_amount = np.where(
        existing_loans == "Yes", (monthly_salary * RNG.uniform(0.05, 0.25, n)).round(-2), 0
    )

    credit_score = np.clip(RNG.normal(650, 90, n), 300, 850).round(0)
    bank_balance = np.clip(monthly_salary * RNG.uniform(0.5, 6, n), 1_000, None).round(-2)
    emergency_fund = np.clip(monthly_salary * RNG.uniform(0, 3, n), 0, None).round(-2)

    scenario_names = list(SCENARIOS.keys())
    scenario_probs = [SCENARIOS[s]["share"] for s in scenario_names]
    emi_scenario = RNG.choice(scenario_names, n, p=scenario_probs)

    requested_amount = np.zeros(n)
    requested_tenure = np.zeros(n, dtype=int)
    for s in scenario_names:
        mask = emi_scenario == s
        lo, hi = SCENARIOS[s]["amount"]
        tlo, thi = SCENARIOS[s]["tenure"]
        requested_amount[mask] = RNG.uniform(lo, hi, mask.sum()).round(-2)
        requested_tenure[mask] = RNG.integers(tlo, thi + 1, mask.sum())

    df = pd.DataFrame({
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type,
        "house_type": house_type, "monthly_rent": monthly_rent, "family_size": family_size,
        "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses, "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount, "credit_score": credit_score,
        "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    })

    total_expenses = (
        df.monthly_rent + df.school_fees + df.college_fees + df.travel_expenses
        + df.groceries_utilities + df.other_monthly_expenses + df.current_emi_amount
    )
    disposable_income = (df.monthly_salary - total_expenses).clip(lower=0)
    credit_factor = (df.credit_score - 300) / 550
    stability_factor = np.clip(df.years_of_employment / 15, 0, 1)

    affordability = disposable_income * (0.25 + 0.35 * credit_factor + 0.15 * stability_factor)
    noise = RNG.normal(1.0, 0.08, n)
    max_monthly_emi = np.clip(affordability * noise, 500, 50_000).round(-1)
    df["max_monthly_emi"] = max_monthly_emi

    proposed_emi = df.requested_amount / df.requested_tenure
    ratio = proposed_emi / df.max_monthly_emi.replace(0, np.nan)

    eligibility = np.where(
        (ratio <= 0.85) & (df.credit_score >= 650), "Eligible",
        np.where((ratio <= 1.15) & (df.credit_score >= 550), "High_Risk", "Not_Eligible"),
    )
    df["emi_eligibility"] = eligibility

    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/emipredict-ai/data/raw/EMI_dataset.csv"
    df = generate(n)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(df["emi_eligibility"].value_counts(normalize=True))
    print(df["max_monthly_emi"].describe())
