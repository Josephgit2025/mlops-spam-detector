import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

APP_TITLE = os.getenv("APP_TITLE", "Spam Detector API")
APP_ENV = os.getenv("APP_ENV", "development")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")

# Chargement lazy — pas au démarrage
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None:
        model_file = BASE_DIR / MODEL_PATH
        vectorizer_file = BASE_DIR / VECTORIZER_PATH
        if not model_file.exists() or not vectorizer_file.exists():
            raise HTTPException(
                status_code=503,
                detail="Modèle non disponible. Lancez d'abord l'entraînement."
            )
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer

app = FastAPI(title=APP_TITLE)

class SMSRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: SMSRequest):
    m, v = load_model()
    text_vec = v.transform([request.text])
    prediction = m.predict(text_vec)[0]
    probability = m.predict_proba(text_vec)[0]
    return {
        "text": request.text,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": round(float(max(probability)), 4),
        "env": APP_ENV
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": APP_ENV}