# Image de base légère
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY src/ ./src/
COPY models/ ./models/

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]