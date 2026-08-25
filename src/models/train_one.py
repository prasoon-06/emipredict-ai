"""
Trains exactly one model (classification or regression) and appends its
metrics + saved pipeline to disk. Designed to be called repeatedly (one
model per invocation) so each run finishes well inside a short time budget.

Usage: python train_one.py <clf|reg> <model_key>
"""
import sys
import json
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
RESULTS_DIR = ROOT / "mlflow_results"
RESULTS_DIR.mkdir(exist_ok=True)
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}"
NUMERIC_COLS = NUMERIC_BASE_COLS + ENGINEERED_COLS
SPLIT_CACHE = RESULTS_DIR / "split.pkl"


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)


def get_split():
    if SPLIT_CACHE.exists():
        return joblib.load(SPLIT_CACHE)
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
    split = dict(X_train=X_train, X_test=X_test, yclf_train=yclf_train_enc, yclf_test=yclf_test_enc,
                 yreg_train=yreg_train, yreg_test=yreg_test, label_encoder=le)
    joblib.dump(split, SPLIT_CACHE)
    return split


CLF_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=500),
    "random_forest_clf": lambda: RandomForestClassifier(n_estimators=120, max_depth=12, n_jobs=-1, random_state=42),
    "decision_tree_clf": lambda: DecisionTreeClassifier(max_depth=12, random_state=42),
    "xgboost_clf": lambda: XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.15,
                                          tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=-1),
}

REG_MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest_reg": lambda: RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=42),
    "decision_tree_reg": lambda: DecisionTreeRegressor(max_depth=12, random_state=42),
    "xgboost_reg": lambda: XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.12,
                                         tree_method="hist", random_state=42, n_jobs=-1),
}


def main():
    task, key = sys.argv[1], sys.argv[2]
    mlflow.set_tracking_uri(MLFLOW_URI)
    split = get_split()
    t0 = time.time()

    if task == "clf":
        mlflow.set_experiment("emipredict-classification")
        model = CLF_MODELS[key]()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=key):
            pipe.fit(split["X_train"], split["yclf_train"])
            preds = pipe.predict(split["X_test"])
            proba = pipe.predict_proba(split["X_test"])
            acc = accuracy_score(split["yclf_test"], preds)
            prec = precision_score(split["yclf_test"], preds, average="macro")
            rec = recall_score(split["yclf_test"], preds, average="macro")
            f1 = f1_score(split["yclf_test"], preds, average="macro")
            try:
                auc = roc_auc_score(split["yclf_test"], proba, multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")
            metrics = {"model": key, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
            mlflow.log_params({"model": key})
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")
        joblib.dump({"pipeline": pipe, "label_encoder": split["label_encoder"], "model_name": key},
                     RESULTS_DIR / f"clf_{key}.pkl")
        (RESULTS_DIR / f"clf_{key}.json").write_text(json.dumps(metrics))
        print(f"[clf] {key:20s} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} ({time.time()-t0:.1f}s)")

    elif task == "reg":
        mlflow.set_experiment("emipredict-regression")
        model = REG_MODELS[key]()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=key):
            pipe.fit(split["X_train"], split["yreg_train"])
            preds = pipe.predict(split["X_test"])
            rmse = float(np.sqrt(mean_squared_error(split["yreg_test"], preds)))
            mae = mean_absolute_error(split["yreg_test"], preds)
            r2 = r2_score(split["yreg_test"], preds)
            mp = mape(split["yreg_test"], preds)
            metrics = {"model": key, "rmse": rmse, "mae": mae, "r2": r2, "mape": mp}
            mlflow.log_params({"model": key})
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")
        joblib.dump({"pipeline": pipe, "model_name": key}, RESULTS_DIR / f"reg_{key}.pkl")
        (RESULTS_DIR / f"reg_{key}.json").write_text(json.dumps(metrics))
        print(f"[reg] {key:20s} rmse={rmse:.1f} mae={mae:.1f} r2={r2:.4f} mape={mp:.2f}% ({time.time()-t0:.1f}s)")

    else:
        raise ValueError(task)


if __name__ == "__main__":
    main()
