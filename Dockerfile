# Image de base légère
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code, du modèle et du dataset
COPY src/ ./src/
COPY models/ ./models/
COPY sms.tsv .

# Port exposé
EXPOSE 8000

# Par défaut, lance l'API
# Pour entraîner le modèle : docker run spam-detector python src/train.py
# Pour lancer l'API : docker run -p 8000:8000 spam-detector
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]