 [![pipeline status](https://gitlab.com/Josephgit2025/mlops-spam-detector/badges/main/pipeline.svg)](https://gitlab.com/Josephgit2025/mlops-spam-detector/-/commits/main)


# MLOps Spam Detector

Pipeline MLOps complet de détection de spam par SMS.

## 📑 Table des matières
- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Architecture](#architecture)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Lancer le projet](#lancer-le-projet)
  - [Installation](#installation-en-local)
  - [Entraîner le modèle](#entraîner-le-modèle)
  - [Lancer l'API](#lancer-lapi)
  - [Docker](#avec-docker)
  - [Tester l'API](#tester-lapi)
  - [MLflow UI](#mlflow-ui-suivi-du-training)
- [Troubleshooting](#troubleshooting)
- [Ce que fait le modèle IA](#ce-que-fait-le-modèle-ia)
- [Licence](#licence)
- [Auteur](#auteur)

## Stack technique
- Python + Scikit-learn — modèle de classification (Naive Bayes)
- FastAPI — API REST de prédiction
- Docker — conteneurisation de l'application
- GitLab CI — pipeline automatisé entraînement → tests → build
- MLflow — tracking des performances du modèle

## Prérequis
- Python 3.11+
- Docker (optionnel, pour conteneurisation)
- pip (gestionnaire de paquets Python)
- Git (pour cloner le repo)

## Architecture
SMS texte → API FastAPI → Vectorisation TF-IDF → Modèle Naive Bayes → spam/ham

## Résultats
- Précision globale : **97%**
- Précision spam : **100%**
- Recall spam : **75%**

## Structure du projet
```
mlops-spam-detector/
├── src/
│   ├── predict.py      # API FastAPI de prédiction
│   └── train.py        # Script d'entraînement du modèle
├── models/
│   ├── model.pkl       # Modèle Naive Bayes (généré après training)
│   └── vectorizer.pkl  # Vectorizer TF-IDF (généré après training)
├── tests/
│   └── test_model.py   # Tests unitaires
├── sms.tsv             # Dataset (5574 SMS labellisés)
├── requirements.txt    # Dépendances Python
├── requirements-dev.txt # Dépendances de développement
├── Dockerfile          # Configuration Docker
└── README.md
```

## Lancer le projet

### Installation (En local)
```bash
# Cloner le repository
git clone <repository-url>
cd mlops-spam-detector

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Linux/Mac:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Entraîner le modèle
```bash
python src/train.py
```
Ceci va :
- Charger et préparer le dataset `sms.tsv`
- Vectoriser les textes avec TF-IDF
- Entraîner le modèle Naive Bayes
- Sauvegarder le modèle et vectorizer dans `models/`
- Logger les métriques dans MLflow

### Lancer l'API
```bash
uvicorn src.predict:app --reload
```
L'API sera accessible sur `http://127.0.0.1:8000`

### Avec Docker

#### Build l'image
```bash
docker build -t spam-detector .
```

#### Lancer l'API
```bash
docker run -p 8000:8000 spam-detector
```

#### Entraîner le modèle dans Docker
```bash
docker run spam-detector python src/train.py
```

#### Entraîner et récupérer les modèles en local
```bash
docker run -v $(pwd)/models:/app/models spam-detector python src/train.py
```

### Tester l'API
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Free prize call now!"}'
```

Réponse attendue :
```json
{
  "text": "WINNER! Free prize call now!",
  "prediction": "spam",
  "confidence": 0.9234
}
```

#### Vérifier la santé de l'API
```bash
curl http://127.0.0.1:8000/health
```

### MLflow UI (Suivi du training)
Pour visualiser les résultats des entraînements :
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Accédez à `http://127.0.0.1:5000` pour voir :
- Les métriques d'accuracy
- L'historique des runs
- Les paramètres du modèle

## Troubleshooting

### Erreur "No module named"
```
FileNotFoundError: No module named 'fastapi'
```
**Solution :** Vérifiez que l'environnement virtuel est activé et les dépendances installées :
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Modèle non trouvé
```
FileNotFoundError: models/model.pkl
```
**Solution :** Vous devez d'abord entraîner le modèle :
```bash
python src/train.py
```

### Port 8000 déjà utilisé
```
Address already in use
```
**Solution :** Trouvez le processus et changez le port :
```bash
lsof -i :8000  # Voir quel processus utilise le port
uvicorn src.predict:app --port 8001  # Utiliser un autre port
```

### Erreur Docker
Assurez-vous que Docker est installé et en cours d'exécution :
```bash
docker --version
docker ps
```

## Ce que fait le modèle IA

Le modèle utilise le **traitement du langage naturel (NLP)** pour analyser 
le contenu textuel d'un SMS et prédire s'il est spam ou non.

### Comment fonctionne l'IA
1. **TF-IDF** (Term Frequency - Inverse Document Frequency) : 
   transforme chaque SMS en vecteur numérique en mesurant 
   l'importance de chaque mot dans le message par rapport 
   à l'ensemble du dataset
2. **Naive Bayes** : algorithme probabiliste qui calcule 
   la probabilité qu'un SMS soit spam en se basant 
   sur les mots qu'il contient
3. **MLflow** : enregistre automatiquement les métriques 
   à chaque entraînement pour suivre l'évolution du modèle

### Performances du modèle
| Métrique | Ham (normal) | Spam |
|----------|-------------|------|
| Précision | 96% | 100% |
| Recall | 100% | 75% |
| F1-score | 98% | 86% |
| **Accuracy globale** | **97%** | |

### Exemple de prédiction
```json
{
  "text": "WINNER! You have been selected. Call now to claim prize!",
  "prediction": "spam",
  "confidence": 0.9064
}
```

## Licence
MIT

## Auteur
Joseph - 2026# mlops-spam-detector
# mlops-spam-detector
# mlops-spam-detector
# mlops-spam-detector
# mlops-spam-detector
