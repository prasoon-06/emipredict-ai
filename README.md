# EMIPredict AI — Intelligent Financial Risk Assessment Platform

A financial risk assessment platform that predicts:
1. **EMI eligibility** (`Eligible` / `High_Risk` / `Not_Eligible`) — classification
2. **Maximum safe monthly EMI** (INR) — regression

Built with scikit-learn + XGBoost, tracked with MLflow, served through a multi-page Streamlit app.

## ⚠️ About the data

The real 400K-record dataset is distributed via a Google Drive link (see the original project
brief) that this environment can't fetch. Everything here runs on a **synthetic dataset
generated to match the exact same schema** — 22 input features, 5 EMI scenarios, the same two
targets — so the whole pipeline is real and functional, but the actual numbers (accuracy, RMSE)
will shift once you swap in the real data.

To use the real dataset: download `EMI_dataset.csv` from the Drive link, drop it at
`data/raw/EMI_dataset.csv`, delete `mlflow_results/split.pkl` (it's a cache), and re-run training.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt

# 1. Get data (synthetic, or drop in the real EMI_dataset.csv — see above)
python src/data/generate_synthetic_data.py 100000 data/raw/EMI_dataset.csv

# 2. Train every model (each call trains ONE model — run all 8)
python src/models/train_one.py clf logistic_regression
python src/models/train_one.py clf random_forest_clf
python src/models/train_one.py clf decision_tree_clf
python src/models/train_one.py clf xgboost_clf
python src/models/train_one.py reg linear_regression
python src/models/train_one.py reg random_forest_reg
python src/models/train_one.py reg decision_tree_reg
python src/models/train_one.py reg xgboost_reg

# 3. Pick champions + write the comparison report
python src/models/finalize.py

# 4. Run the app
streamlit run app/app.py
```

(`src/models/train.py` is a single-script version that runs all 8 in one go — use it if your
machine can handle ~2-3 minutes of uninterrupted training; `train_one.py` + `finalize.py` is the
same pipeline split into steps, useful in constrained/sandboxed environments.)

## View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

## Results (synthetic data, 60K rows, 80/20 split)

| | Champion | Metric |
|---|---|---|
| Classification | `xgboost_clf` | 96.4% accuracy (target: >90%) |
| Regression | `xgboost_reg` | RMSE ₹1,373 (target: <₹2,000) |

Full comparison across all 8 models: [`reports/model_comparison_report.md`](reports/model_comparison_report.md).

## Project structure

See [`root_structure.md`](../root_structure.md) from the planning docs, or just browse the tree —
it matches that spec: `data/`, `src/{data,features,models}/`, `mlflow/`, `models/`, `app/`,
`reports/`, `tests/`.

## Deploying

This is built to deploy on **Streamlit Community Cloud** (the platform named in the original
brief):

1. Push this repo to GitHub. Note `.gitignore` excludes the raw CSV, the MLflow SQLite db, and
   `mlflow_results/` — but **keeps** `models/best_classifier.pkl` and `models/best_regressor.pkl`,
   since those are the only artifacts the deployed app actually needs.
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect GitHub → point it at
   `app/app.py`.
3. It installs from `requirements.txt` and gives you a public URL.

## Notes / limitations of this build

- Dataset is synthetic (see above) — re-run training on the real data before treating metrics as final.
- The "Admin: Data Management" page edits an in-memory session copy of the data only; it does not
  write back to `data/raw/EMI_dataset.csv`.
- MLflow uses a local SQLite file store here for portability; point `MLFLOW_URI` at a remote
  tracking server if you want shared/team experiment tracking.
