import os
import joblib

model_path = os.path.join("models", "home_credit_pipeline.joblib")

def load_pipeline():
	if not os.path.exists(model_path):
		raise FileNotFoundError(
			f"Model not found at {model_path}"
		)
	pipeline = joblib.load(model_path)
	return pipeline
