"""
Trains >=3 classification models (emi_eligibility) and >=3 regression models
(max_monthly_emi), logs every run to MLflow, picks the best of each, and saves
the winning pipelines (preprocessing + model bundled together) to models/.
"""
import sys
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import add_features, CATEGORICAL_COLS, NUMERIC_BASE_COLS, ENGINEERED_COLS

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/EMI_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}"

NUMERIC_COLS = NUMERIC_BASE_COLS + ENGINEERED_COLS


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)


def run():
    mlflow.set_tracking_uri(MLFLOW_URI)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y_clf = df["emi_eligibility"]
    y_reg = df["max_monthly_emi"]

    X_train, X_test, yclf_train, yclf_test, yreg_train, yreg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    le = LabelEncoder()
    yclf_train_enc = le.fit_transform(yclf_train)
    yclf_test_enc = le.transform(yclf_test)

    # ---------------- Classification ----------------
    mlflow.set_experiment("emipredict-classification")
    clf_candidates = {
        "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "random_forest_clf": RandomForestClassifier(n_estimators=120, max_depth=12, n_jobs=-1, random_state=42),
        "decision_tree_clf": DecisionTreeClassifier(max_depth=12, random_state=42),
        "xgboost_clf": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.15, tree_method="hist",
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        ),
    }

    clf_results = []
    best_clf_name, best_clf_pipeline, best_clf_score = None, None, -1

    for name, model in clf_candidates.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, yclf_train_enc)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)

            acc = accuracy_score(yclf_test_enc, preds)
            prec = precision_score(yclf_test_enc, preds, average="macro")
            rec = recall_score(yclf_test_enc, preds, average="macro")
            f1 = f1_score(yclf_test_enc, preds, average="macro")
            try:
                auc = roc_auc_score(yclf_test_enc, proba, multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")

            mlflow.log_params({"model": name})
            mlflow.log_metrics({
                "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
                "f1_macro": f1, "roc_auc_macro": auc,
            })
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")

            elapsed = time.time() - t0
            print(f"[clf] {name:20s} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} ({elapsed:.1f}s)")
            clf_results.append({"model": name, "accuracy": acc, "precision": prec,
                                 "recall": rec, "f1": f1, "roc_auc": auc})

            if acc > best_clf_score:
                best_clf_score, best_clf_name, best_clf_pipeline = acc, name, pipe

    # ---------------- Regression ----------------
    mlflow.set_experiment("emipredict-regression")
    reg_candidates = {
        "linear_regression": LinearRegression(),
        "random_forest_reg": RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=42),
        "decision_tree_reg": DecisionTreeRegressor(max_depth=12, random_state=42),
        "xgboost_reg": XGBRegressor(
            n_estimators=180, max_depth=5, learning_rate=0.12, tree_method="hist", random_state=42, n_jobs=-1,
        ),
    }

    reg_results = []
    best_reg_name, best_reg_pipeline, best_reg_score = None, None, float("inf")

    for name, model in reg_candidates.items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, yreg_train)
            preds = pipe.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(yreg_test, preds)))
            mae = mean_absolute_error(yreg_test, preds)
            r2 = r2_score(yreg_test, preds)
            mp = mape(yreg_test, preds)

            mlflow.log_params({"model": name})
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "mape": mp})
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")

            elapsed = time.time() - t0
            print(f"[reg] {name:20s} rmse={rmse:.1f} mae={mae:.1f} r2={r2:.4f} mape={mp:.2f}% ({elapsed:.1f}s)")
            reg_results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mp})

            if rmse < best_reg_score:
                best_reg_score, best_reg_name, best_reg_pipeline = rmse, name, pipe

    # ---------------- Persist champions ----------------
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_clf_pipeline, "label_encoder": le, "model_name": best_clf_name},
                MODELS_DIR / "best_classifier.pkl")
    joblib.dump({"pipeline": best_reg_pipeline, "model_name": best_reg_name},
                MODELS_DIR / "best_regressor.pkl")

    print(f"\nBest classifier: {best_clf_name} (accuracy={best_clf_score:.4f})")
    print(f"Best regressor:  {best_reg_name} (rmse={best_reg_score:.1f})")

    # ---------------- Report ----------------
    REPORTS_DIR.mkdir(exist_ok=True)
    clf_df = pd.DataFrame(clf_results).sort_values("accuracy", ascending=False)
    reg_df = pd.DataFrame(reg_results).sort_values("rmse")

    report = ["# Model Comparison Report\n",
              "## Classification (emi_eligibility)\n",
              clf_df.to_markdown(index=False),
              f"\n\n**Selected: `{best_clf_name}`** (highest accuracy)\n",
              "\n## Regression (max_monthly_emi)\n",
              reg_df.to_markdown(index=False),
              f"\n\n**Selected: `{best_reg_name}`** (lowest RMSE)\n"]
    (REPORTS_DIR / "model_comparison_report.md").write_text("\n".join(report))
    print(f"\nReport written to {REPORTS_DIR / 'model_comparison_report.md'}")


if __name__ == "__main__":
    run()
