"""Reads mlflow_results/*.json + *.pkl, picks champions, saves them to models/, writes report."""
import json
import shutil
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "mlflow_results"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

clf_results = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("clf_*.json")]
reg_results = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("reg_*.json")]

clf_df = pd.DataFrame(clf_results).sort_values("accuracy", ascending=False).reset_index(drop=True)
reg_df = pd.DataFrame(reg_results).sort_values("rmse").reset_index(drop=True)

best_clf = clf_df.iloc[0]["model"]
best_reg = reg_df.iloc[0]["model"]

shutil.copy(RESULTS_DIR / f"clf_{best_clf}.pkl", MODELS_DIR / "best_classifier.pkl")
shutil.copy(RESULTS_DIR / f"reg_{best_reg}.pkl", MODELS_DIR / "best_regressor.pkl")

print(f"Best classifier: {best_clf} (accuracy={clf_df.iloc[0]['accuracy']:.4f})")
print(f"Best regressor:  {best_reg} (rmse={reg_df.iloc[0]['rmse']:.1f})")

report = [
    "# Model Comparison Report\n",
    "## Classification (emi_eligibility)\n",
    clf_df.round(4).to_markdown(index=False),
    f"\n\n**Selected: `{best_clf}`** — highest accuracy on held-out test set.\n",
    "\n## Regression (max_monthly_emi)\n",
    reg_df.round(2).to_markdown(index=False),
    f"\n\n**Selected: `{best_reg}`** — lowest RMSE on held-out test set.\n",
    "\n## Notes\n",
    "- All 8 runs (4 classification + 4 regression) are logged in MLflow "
    "(`mlflow/mlflow.db`, sqlite backend) with full params, metrics, and the "
    "serialized pipeline as an artifact.\n",
    "- View them with: `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`\n",
    "- Champion pipelines (preprocessing + model bundled) are saved to "
    "`models/best_classifier.pkl` and `models/best_regressor.pkl` and are what "
    "the Streamlit app loads.\n",
]
(REPORTS_DIR / "model_comparison_report.md").write_text("\n".join(report))
print(f"\nReport written to {REPORTS_DIR / 'model_comparison_report.md'}")
