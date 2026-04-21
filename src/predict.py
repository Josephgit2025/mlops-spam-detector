from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Chargement du modèle et du vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

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