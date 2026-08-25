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
# Root Structure — EMIPredict AI

```
emipredict-ai/
│
├── data/
│   ├── raw/                       # original EMI_dataset (400K records) — gitignored, keep local/DVC
│   ├── interim/                   # cleaned but not yet feature-engineered
│   └── processed/                 # final train/val/test splits used for modeling
│
├── notebooks/
│   ├── 01_eda.ipynb                # exploratory data analysis
│   ├── 02_preprocessing.ipynb      # cleaning / missing values / dedup walkthrough
│   ├── 03_feature_engineering.ipynb
│   ├── 04_classification_models.ipynb
│   └── 05_regression_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # paths, constants, random seed
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   └── split_data.py
│   ├── features/
│   │   ├── build_features.py       # ratios, risk score, interaction features
│   │   └── encoders.py             # categorical encoding / scaling
│   ├── models/
│   │   ├── train_classification.py # LogReg, RF, XGBoost, etc.
│   │   ├── train_regression.py     # LinReg, RF, XGBoost, etc.
│   │   ├── evaluate.py             # accuracy/precision/recall/F1/ROC-AUC, RMSE/MAE/R²/MAPE
│   │   └── select_best_model.py
│   └── utils/
│       └── logging_utils.py
│
├── mlflow/
│   ├── mlruns/                     # local MLflow tracking store (or remote URI in config)
│   └── register_models.py          # promote best models to the registry
│
├── models/
│   ├── best_classifier.pkl
│   └── best_regressor.pkl
│
├── app/                             # Streamlit application
│   ├── app.py                       # entry point / home page
│   ├── pages/
│   │   ├── 1_Predict_Eligibility.py
│   │   ├── 2_Predict_Max_EMI.py
│   │   ├── 3_Data_Explorer.py
│   │   ├── 4_Model_Performance.py   # MLflow dashboard view
│   │   └── 5_Admin_Data_Management.py
│   └── components/
│       └── ui_helpers.py
│
├── reports/
│   ├── eda_report.md
│   ├── model_comparison_report.md
│   └── business_impact.md
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── .streamlit/
│   └── config.toml                  # theme + server settings for deployment
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Notes

- **`data/raw/`** should be gitignored — 400K-record CSVs don't belong in Git history. Use Git LFS, DVC, or just keep it out of the repo and document the download link in `README.md`.
- **`app/pages/`** uses Streamlit's automatic multi-page routing (files prefixed with a number control sidebar order).
- **`mlflow/mlruns/`** works fine locally for development; for the deployed app, either bundle only the exported `models/*.pkl` (recommended — keeps the deployed app lightweight) or point MLflow to a remote tracking URI if you want the deployed app to query MLflow live.
- **`models/`** holds only the two final serialized "champion" models the Streamlit app actually loads — not every experiment run.
- Keep `requirements.txt` pinned (exact versions) since Streamlit Cloud rebuilds the environment from it on every deploy.



## Deploying

This is built to deploy on **Streamlit Community Cloud** 
LINK: https://emipredict-ai-6izvv9qzmezhzsamrptueu.streamlit.app/

## Notes / limitations of this build

- Dataset is synthetic (see above) — re-run training on the real data before treating metrics as final.
- The "Admin: Data Management" page edits an in-memory session copy of the data only; it does not
  write back to `data/raw/EMI_dataset.csv`.
- MLflow uses a local SQLite file store here for portability; point `MLFLOW_URI` at a remote
  tracking server if you want shared/team experiment tracking.
