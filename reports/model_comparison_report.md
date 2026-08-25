# Model Comparison Report

## Classification (emi_eligibility)

| model               |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| xgboost_clf         |     0.9658 |      0.9616 |   0.9598 | 0.9607 |    0.9977 |
| decision_tree_clf   |     0.9342 |      0.9256 |   0.926  | 0.9258 |    0.9569 |
| random_forest_clf   |     0.9322 |      0.9361 |   0.9128 | 0.9197 |    0.9893 |
| logistic_regression |     0.7988 |      0.7685 |   0.7656 | 0.7667 |    0.9327 |


**Selected: `xgboost_clf`** — highest accuracy on held-out test set.


## Regression (max_monthly_emi)

| model             |    rmse |     mae |   r2 |   mape |
|:------------------|--------:|--------:|-----:|-------:|
| xgboost_reg       | 1365.27 |  916.66 | 0.98 |   7.84 |
| random_forest_reg | 1375.59 |  910.97 | 0.98 |   6.6  |
| decision_tree_reg | 1825.97 | 1162.37 | 0.97 |   8.32 |
| linear_regression | 1944.87 | 1337.29 | 0.96 |  36.04 |


**Selected: `xgboost_reg`** — lowest RMSE on held-out test set.


## Notes

- All 8 runs (4 classification + 4 regression) are logged in MLflow (`mlflow/mlflow.db`, sqlite backend) with full params, metrics, and the serialized pipeline as an artifact.

- View them with: `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`

- Champion pipelines (preprocessing + model bundled) are saved to `models/best_classifier.pkl` and `models/best_regressor.pkl` and are what the Streamlit app loads.
