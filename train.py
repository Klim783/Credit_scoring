# train.py
import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.transformers import FinancialRatiosTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

console.print(Panel.fit("[bold green]Home Credit Machine Learning Training Pipeline[/bold green]"))

table = Table(title="Model Cross-Validation Results")
table.add_column("Model Architecture", style="cyan", no_wrap=True)
table.add_column("ROC-AUC", style="magenta")
table.add_column("Gini Score", style="green")

table.add_row("LightGBM Classifier", f"{lgbm_auc:.4f}", f"{2 * lgbm_auc - 1:.4f}")
table.add_row("Logistic Regression", f"{lr_auc:.4f}", f"{2 * lr_auc - 1:.4f}")

console.print(table)

if __name__ == "__main__":
    # 1. Load Data
    df = pd.read_csv("data/application_train.csv")

    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df["TARGET"]

    # 2. Columns setup
    categorical_cols = X.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Ratios generated inside transformer
    new_ratio_cols = [
        "CREDIT_INCOME_PERCENT",
        "ANNUITY_INCOME_PERCENT",
        "CREDIT_TERM",
        "GOODS_PRICE_RATIO",
        "DAYS_EMPLOYED_PERCENT",
        "AGE_YEARS",
    ]
    numerical_cols.extend(new_ratio_cols)

    # 3. Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",  # Drops any extra dataset features not explicitly passed in
    )

    lgbm_pipeline = Pipeline([
        ("ratios", FinancialRatiosTransformer()),
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=80,
            learning_rate=0.03,
            num_leaves=12,
            max_depth=4,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])

    # 4. Cross-Validation Evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Evaluating LightGBM Classifier...")
    lgbm_oof_preds = cross_val_predict(lgbm_pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    lgbm_auc = roc_auc_score(y, lgbm_oof_preds)
    print(f"LightGBM OOF ROC-AUC: {lgbm_auc:.4f} | Gini: {2 * lgbm_auc - 1:.4f}")

    # 5. Fit on Full Data and Save Artifact
    print("\nTraining model on full dataset...")
    lgbm_pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "home_credit_pipeline.pkl")
    joblib.dump(lgbm_pipeline, model_path)
    print(f"Pipeline successfully saved to {model_path}")