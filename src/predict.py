from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

# Chemin absolu vers la racine du projet
BASE_DIR = Path(__file__).parent.parent

# Chargement du modèle et du vectorizer
model = joblib.load(BASE_DIR / "models" / "model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "vectorizer.pkl")

# Initialisation de l'API
app = FastAPI(title="Spam Detector API")

# Schéma de la requête
class SMSRequest(BaseModel):
    text: str

# Endpoint de prédiction
@app.post("/predict")
def predict(request: SMSRequest):
    text_vec = vectorizer.transform([request.text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    return {
        "text": request.text,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": round(float(max(probability)), 4)
    }

# Endpoint de santé
@app.get("/health")
def health():
    return {"status": "ok"}