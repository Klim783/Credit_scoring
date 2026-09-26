# app/main.py
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_pipeline
from app.schemas import CustomerData, PredictionResponse

app = FastAPI(
    title="Home credit scoring service",
    description="API for scoring credit clients' queries",
)

try:
    pipeline = load_pipeline()
except Exception as e:
    pipeline = None
    print(f"Failed to load pipeline: {e}")


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(data: CustomerData, threshold: float = 0.35):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="No pipeline loaded")

    input_df = pd.DataFrame([data.model_dump()])
    prob = float(pipeline.predict_proba(input_df)[:, 1][0])
    decision = "rejected" if prob >= threshold else "accepted"

    return PredictionResponse(
        default_probability=round(prob, 4),
        decision=decision,
        threshold=threshold,
    )