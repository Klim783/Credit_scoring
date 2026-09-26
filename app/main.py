import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException,status
import logging
from app.model_loader import load_pipeline
from app.schemas import CustomerData, PredictionResponse

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app : FastAPI):
    try:
        load_pipeline()
    except Exception as e:
        logger.warning(f"Startup warning: Pipeline cound not be loaded: {e}")
    yield


app = FastAPI(
    title="💳 Home Credit Scoring Engine",
    description="""
    ### End-to-End Machine Learning Credit Assessment Service

    This API calculates the **probability of default** for credit applicants using a tuned **LightGBM** model.

    * **`/`** - Service health status check
    * **`/predict`** - Submit client attributes and retrieve default probability + risk decision
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
def health_check():
    try:
        pipeline = load_pipeline()
        is_loaded = pipeline is not None
    except Exception:
        is_loaded = False
    return{
        "status":"healthy" if is_loaded else "degraded",
        "model_loaded":is_loaded
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_credit_risk(data: CustomerData, threshold: float = 0.35):
    try:
        pipeline = load_pipeline()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Model service currently not available"
        )
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prob = float(pipeline.predict_proba(input_df)[:1,0][0])
        decision = "rejected" if prob >= threshold else "accepted"

        return PredictionResponse(
            default_probability = round(prob,4),
            decision = decision,
            threshold = threshold,
        )
    except Exception as err:
        logger.warning(f"Inference error: {str(err)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail = f"Inference execution error: {str(err)}"
        )