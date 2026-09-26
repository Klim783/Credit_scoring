import os
import joblib
import logging
from functools import lru_cache

logger = logging.getLogger("uvicorn.error")
model_path = os.path.join("models", "home_credit_pipeline.pkl")


@lru_cache(maxsize=1)
def load_pipeline():
    if not os.path.exists(model_path):
        logger.error(f"Model artifact missing at path: {os.path.abspath(model_path)}")
        raise FileNotFoundError(
            f"Model file not found at: {os.path.abspath(model_path)}"
        )
    logger.info(f"Loading model from {os.path.abspath(model_path)}")
    pipeline = joblib.load(model_path)
    logger.info(f"Pipeline loaded successfully")
    return pipeline