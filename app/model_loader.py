import os
import joblib
from app.transformers import FinancialRatiosTransformer

MODEL_PATH = os.path.join("models", "home_credit_pipeline.pkl")


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена по пути: {os.path.abspath(MODEL_PATH)}"
        )

    return joblib.load(MODEL_PATH)