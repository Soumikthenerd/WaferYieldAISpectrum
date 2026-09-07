from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model artifact
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "model_a.pkl"))

class WaferPayload(BaseModel):
    x: int
    y: int
    signal_2000: list[float]


@app.get("/")
def health_check():
    return {"status": "Model API is live"}


@app.post("/predict")
def predict(data: WaferPayload):
    # Combine spatial coordinates and signal array into single row
    features = np.hstack([[data.x, data.y], data.signal_2000]).reshape(1, -1)

    prob = float(model.predict_proba(features)[0][1])
    label = int(prob >= 0.5)

    return {"label": label, "fail_prob": round(prob, 4)}