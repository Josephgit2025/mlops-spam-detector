 [![pipeline status](https://gitlab.com/Josephgit2025/mlops-spam-detector/badges/main/pipeline.svg)](https://gitlab.com/Josephgit2025/mlops-spam-detector/-/commits/main)


# MLOps Spam Detector

Pipeline MLOps complet de détection de spam par SMS — entraînement automatisé, API REST, conteneurisation et CI/CD sur deux plateformes.

---

## Ce que fait le modèle IA

Le modèle utilise le traitement du langage naturel (NLP) pour analyser le contenu textuel d'un SMS et prédire s'il est spam ou non.

**Fonctionnement :**

1. **TF-IDF** — transforme chaque SMS en vecteur numérique en mesurant l'importance de chaque mot par rapport à l'ensemble du dataset
2. **Naive Bayes** — algorithme probabiliste qui calcule la probabilité qu'un SMS soit spam en se basant sur les mots qu'il contient
3. **MLflow** — enregistre automatiquement les métriques à chaque entraînement pour suivre l'évolution du modèle

Stack : Python · Scikit-learn · FastAPI · Docker · GitLab CI · Drone CI · Gitea · MLflow

---

## 📑 Table des matières

- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Architecture](#architecture)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Lancer le projet](#lancer-le-projet)
  - [Installation](#installation)
  - [Entraîner le modèle](#entraîner-le-modèle)
  - [Lancer l'API](#lancer-lapi)
  - [Docker](#docker)
  - [Tester l'API](#tester-lapi)
  - [MLflow UI](#mlflow-ui)
- [CI/CD Pipeline](#cicd-pipeline)
  - [GitLab CI](#gitlab-ci)
  - [Drone CI (Homelab)](#drone-ci-homelab)
- [Troubleshooting](#troubleshooting)

---

## Stack technique

| Outil | Rôle |
|---|---|
| Python + Scikit-learn | Modèle de classification (Naive Bayes) |
| FastAPI | API REST de prédiction |
| Docker | Conteneurisation de l'application |
| GitLab CI | Pipeline CI/CD public (entraînement → tests → build) |
| Drone CI | Pipeline CI/CD sur homelab (Gitea local) |
| MLflow | Tracking des performances du modèle |

---

## Prérequis

- Python 3.11+
- Docker (optionnel, pour conteneurisation)
- pip (gestionnaire de paquets Python)
- Git (pour cloner le repo)

---

## Architecture

```
SMS texte → API FastAPI → Vectorisation TF-IDF → Modèle Naive Bayes → spam / ham
```

---

## Résultats

| Métrique | Ham (normal) | Spam |
|---|---|---|
| Précision | 96% | 100% |
| Recall | 100% | 75% |
| F1-score | 98% | 86% |
| **Accuracy globale** | **97%** | |

---

## Structure du projet

```
mlops-spam-detector/
├── src/
│   ├── predict.py          # API FastAPI de prédiction
│   └── train.py            # Script d'entraînement du modèle
├── models/
│   ├── model.pkl           # Modèle Naive Bayes (généré après training)
│   └── vectorizer.pkl      # Vectorizer TF-IDF (généré après training)
├── tests/
│   └── test_model.py       # Tests unitaires
├── sms.tsv                 # Dataset (5574 SMS labellisés)
├── requirements.txt        # Dépendances Python
├── requirements-dev.txt    # Dépendances de développement
├── Dockerfile              # Configuration Docker
├── .gitlab-ci.yml          # Pipeline GitLab CI
├── .drone.yml              # Pipeline Drone CI (homelab)
└── README.md
```

---

## Lancer le projet

### Installation

```bash
# Cloner le repository
git clone https://gitlab.com/Josephgit2025/mlops-spam-detector.git
cd mlops-spam-detector

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

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

### Docker

```bash
# Build l'image
docker build -t spam-detector .

# Lancer l'API
docker run -p 8000:8000 spam-detector

# Entraîner le modèle dans Docker
docker run spam-detector python src/train.py

# Entraîner et récupérer les modèles en local
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

Vérifier la santé de l'API :

```bash
curl http://127.0.0.1:8000/health
```

### MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Accédez à `http://127.0.0.1:5000` pour visualiser :
- Les métriques d'accuracy
- L'historique des runs
- Les paramètres du modèle

---

## CI/CD Pipeline

Le projet dispose de deux pipelines CI/CD indépendants selon l'environnement cible.

### GitLab CI

Pipeline public hébergé sur [gitlab.com](https://gitlab.com/Josephgit2025/mlops-spam-detector).

**Stages :**

```
train → test → build → deploy
```

| Stage | Description |
|---|---|
| `train_model` | Installe les dépendances, entraîne le modèle, sauvegarde `models/` via artifacts |
| `test_api` | Installe les dépendances de dev, exécute les tests pytest |
| `build_image` | Build l'image Docker via Docker-in-Docker (dind), push vers GitLab Container Registry |
| `deploy` | Déclare l'image comme déployée sur l'environment `production` GitLab |

**Fichier :** `.gitlab-ci.yml`

```yaml
stages:
  - train
  - test
  - build
  - deploy

train_model:
  stage: train
  image: python:3.11-slim
  script:
    - pip install -r requirements.txt
    - python src/train.py
  artifacts:
    paths:
      - models/
    when: always

test_api:
  stage: test
  image: python:3.11-slim
  needs: [train_model]
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - python -m pytest tests/ -v

build_image:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  needs: [test_api]
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:latest .
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: alpine:latest
  needs: [build_image]
  environment:
    name: production
    url: https://registry.gitlab.com/josephgit2025/mlops-spam-detector
  script:
    - echo "Image déployée sur le registry GitLab"
    - echo "Image disponible à $CI_REGISTRY_IMAGE:latest"
  only:
    - main
```

> Les variables `$CI_REGISTRY_*` sont injectées automatiquement par GitLab — aucune configuration manuelle requise.

**Accéder à l'image depuis le registry GitLab :**

L'image est disponible après chaque push sur `main` :

```bash
# Pull l'image depuis le registry GitLab
docker pull registry.gitlab.com/josephgit2025/mlops-spam-detector:latest

# Lancer l'API directement depuis le registry
docker run -p 8000:8000 registry.gitlab.com/josephgit2025/mlops-spam-detector:latest
```

> GitLab → **Deploy** → **Container Registry** → `spam-detector:latest`

---

### Drone CI (Homelab)

Pipeline hébergé sur un serveur local (Gitea + Drone CI auto-hébergés).

**Différences clés avec GitLab CI :**

| GitLab CI | Drone CI |
|---|---|
| `artifacts` pour passer les fichiers entre jobs | `volumes` partagés sur le host |
| `needs` pour les dépendances | `depends_on` |
| `docker:24-dind` pour builder | `plugins/docker` (pas besoin de dind) |
| `only: - main` | `trigger: branch: - main` |

**Fichier :** `.drone.yml`

```yaml
kind: pipeline
type: docker
name: spam-detector

trigger:
  branch:
    - main

volumes:
  - name: models
    host:
      path: /tmp/drone-models

steps:
  - name: train_model
    image: python:3.11-slim
    volumes:
      - name: models
        path: /drone/src/models
    commands:
      - pip install -r requirements.txt
      - python src/train.py

  - name: test_api
    image: python:3.11-slim
    depends_on: [train_model]
    volumes:
      - name: models
        path: /drone/src/models
    commands:
      - pip install -r requirements.txt
      - pip install -r requirements-dev.txt
      - python -m pytest tests/ -v

  - name: build_image
    image: plugins/docker
    depends_on: [test_api]
    settings:
      repo: <gitea-host>/admin/spam-detector
      registry: <gitea-host>
      username:
        from_secret: GITEA_USERNAME
      password:
        from_secret: GITEA_PASSWORD
      tags: latest
      insecure: true
```

> Les credentials sont stockés comme secrets Drone CI — jamais en clair dans le fichier.

---

## Troubleshooting

**Erreur "No module named"**
```
No module named 'fastapi'
```
Vérifiez que l'environnement virtuel est activé :
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Modèle non trouvé**
```
FileNotFoundError: models/model.pkl
```
Entraînez d'abord le modèle :
```bash
python src/train.py
```

**Port 8000 déjà utilisé**
```
Address already in use
```
```bash
lsof -i :8000
uvicorn src.predict:app --port 8001
```

**Erreur Docker**
```bash
docker --version
docker ps
```

---

**Exemple de prédiction :**

```json
{
  "text": "WINNER! You have been selected. Call now to claim prize!",
  "prediction": "spam",
  "confidence": 0.9064
}
```