# MLOps Spam Detector

![CI/CD](https://github.com/Josephgit2025/mlops-spam-detector/actions/workflows/ci.yaml/badge.svg)

Pipeline MLOps complet de détection de spam par SMS — entraînement automatisé, API REST, déploiement cloud sur DigitalOcean avec Terraform, Ansible, ArgoCD et GitHub Actions.

---

## Ce que fait le modèle IA

Le modèle utilise le traitement du langage naturel (NLP) pour analyser le contenu textuel d'un SMS et prédire s'il est spam ou non.

**Fonctionnement :**

1. **TF-IDF** — transforme chaque SMS en vecteur numérique en mesurant l'importance de chaque mot par rapport à l'ensemble du dataset
2. **Naive Bayes** — algorithme probabiliste qui calcule la probabilité qu'un SMS soit spam en se basant sur les mots qu'il contient
3. **MLflow** — enregistre automatiquement les métriques à chaque entraînement pour suivre l'évolution du modèle

Stack : Python · Scikit-learn · FastAPI · Docker · Terraform · Ansible · ArgoCD · GitHub Actions · DigitalOcean

---

## Résultats

| Métrique | Ham (normal) | Spam |
|---|---|---|
| Précision | 96% | 100% |
| Recall | 100% | 75% |
| F1-score | 98% | 86% |
| **Accuracy globale** | **97%** | |

---

## Architecture Cloud

```
GitHub (code)
      │
      ▼
GitHub Actions (CI)
  ├── train → test → build → push Docker Hub
  └── update image tag → push GitHub
      │
      ▼
ArgoCD (GitOps) — cluster DOKS DigitalOcean
  ├── spam-detector-api    → API FastAPI (LoadBalancer)
  └── spam-detector-training → Job Kubernetes (entraînement)
      │
      ▼
MLflow Server — Droplet DigitalOcean
  └── PostgreSQL — DigitalOcean Managed Database
```

---

## Architecture détaillée

GitHub (source of truth)
    │
    ▼
GitHub Actions (CI)
    ├── train   → python src/train.py → logs métriques sur MLflow (DO Droplet)
    ├── test    → pytest avec modèle artifact entre jobs
    ├── build   → docker build + push Docker Hub (josephmariebile/spam-detector)
    └── deploy  → sed sur deployment.yaml (tag = $GITHUB_SHA) + git push
                        │
                        ▼
                  ArgoCD détecte le diff
                        │
                        ▼
              DOKS (DigitalOcean Kubernetes)
                  ├── spam-detector-api (Deployment + LoadBalancer)
                  └── spam-detector-training (Job)
                        │
                        ▼
              MLflow Server (Droplet Ubuntu 22.04)
                  └── Backend PostgreSQL (DO Managed Database)

## Infrastructure as Code

L'infrastructure complète est provisionnée avec **Terraform** et configurée avec **Ansible**.

### Terraform — Provisionnement DO

```
terraform/
├── provider.tf      # Provider DigitalOcean
├── main.tf          # Droplet MLflow + DOKS + Firewall + PostgreSQL DO
├── variables.tf     # Variables (token, région, SSH key)
└── outputs.tf       # IPs, cluster ID, DB URI
```

Ressources créées :

| Ressource | Type | Rôle |
|---|---|---|
| `mlflow-server` | Droplet (1vCPU / 2GB) | Serveur MLflow |
| `spam-detector-cluster` | DOKS (2 nodes) | Cluster Kubernetes |
| `mlflow-postgres` | DO Managed Database | Backend PostgreSQL MLflow |
| `mlflow-firewall` | Firewall | Ports 22 + 5000 |

```bash
# Provisionner toute l'infrastructure
terraform init
terraform plan
terraform apply
```

### Ansible — Configuration du Droplet

```
ansible/
├── inventory.ini    # IP du Droplet (non commité)
├── playbook.yaml    # Installation Docker + PostgreSQL + MLflow
└── secrets.yaml     # Credentials chiffrés (Ansible Vault)
```

```bash
# Configurer le serveur MLflow
ansible-playbook -i inventory.ini playbook.yaml --ask-vault-pass
```

---

## Stack technique

| Outil | Rôle |
|---|---|
| Python + Scikit-learn | Modèle NLP (TF-IDF + Naive Bayes) |
| FastAPI | API REST de prédiction |
| MLflow | Tracking des performances du modèle |
| PostgreSQL (DO Managed) | Backend de stockage MLflow |
| Docker | Conteneurisation |
| Terraform | Provisionnement infrastructure DO |
| Ansible + Vault | Configuration serveur + gestion secrets |
| Kubernetes (DOKS) | Orchestration des workloads |
| ArgoCD | GitOps — déploiement automatique |
| GitHub Actions | CI/CD — build, test, push, deploy |

---

## Structure du projet

```
mlops-spam-detector/
├── src/
│   ├── predict.py              # API FastAPI de prédiction
│   └── train.py                # Script d'entraînement
├── k8s/
│   ├── api/
│   │   ├── deployment.yaml     # Déploiement API FastAPI
│   │   ├── service.yaml        # LoadBalancer DO
│   │   └── configmap.yaml      # Variables d'environnement
│   └── training/
│       ├── job.yaml            # Job Kubernetes d'entraînement
│       └── configmap.yaml      # URI MLflow (non commité)
├── argocd/
│   ├── root-app.yaml           # App of Apps ArgoCD
│   ├── argocd.app.yaml         # App API FastAPI
│   └── training-app.yaml       # App Job entraînement
├── terraform/
│   ├── main.tf                 # Ressources DO
│   ├── variables.tf
│   ├── outputs.tf
│   └── provider.tf
├── ansible/
│   ├── playbook.yaml           # Config serveur MLflow
│   └── secrets.yaml            # Secrets chiffrés (Ansible Vault)
├── .github/
│   └── workflows/
│       └── ci.yaml             # Pipeline GitHub Actions
├── tests/
│   └── test_model.py
├── Dockerfile                  # Image API FastAPI
├── Dockerfile.mlflow           # Image MLflow custom (+ psycopg2)
├── sms.tsv                     # Dataset (5574 SMS labellisés)
└── requirements.txt
```

---

## CI/CD Pipeline

### GitHub Actions

```
push sur main
      │
      ├── train    → python src/train.py (logs vers MLflow)
      ├── test     → pytest tests/
      ├── build    → docker build + push Docker Hub
      └── deploy   → update image tag → ArgoCD sync automatique
```

**Secrets configurés :**

```
DOCKERHUB_USERNAME    → username Docker Hub
DOCKERHUB_TOKEN       → token Docker Hub
MLFLOW_TRACKING_URI   → URI du serveur MLflow sur DO
```

### ArgoCD (GitOps)

ArgoCD surveille le repo GitHub et déploie automatiquement sur DOKS à chaque changement.

```
root-app                → surveille argocd/
  ├── spam-detector-api      → déploie k8s/api/
  └── spam-detector-training → déploie k8s/training/
```

---

## Lancer le projet en local

### Installation

```bash
git clone https://github.com/Josephgit2025/mlops-spam-detector.git
cd mlops-spam-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Entraîner le modèle

```bash
export MLFLOW_TRACKING_URI="http://<mlflow-server-ip>:5000"
python src/train.py
```

### Lancer l'API

```bash
uvicorn src.predict:app --reload
```

### Tester l'API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Free prize call now!"}'
```

Réponse :

```json
{
  "text": "WINNER! Free prize call now!",
  "prediction": "spam",
  "confidence": 0.9234
}
```

---

## Sécurité

- Secrets gérés avec **Ansible Vault** (chiffrement AES256)
- Token DO et credentials DB via **variables d'environnement** (`TF_VAR_*`)
- Fichiers sensibles exclus du repo via `.gitignore`
- Firewall DO restreint aux ports 22 et 5000 uniquement