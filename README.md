# MLOps Spam Detector

Pipeline MLOps complet de détection de spam par SMS.

## Stack technique
- Python + Scikit-learn — modèle de classification (Naive Bayes)
- FastAPI — API REST de prédiction
- Docker — conteneurisation de l'application
- GitLab CI — pipeline automatisé entraînement → tests → build
- MLflow — tracking des performances du modèle

## Architecture
SMS texte → API FastAPI → Vectorisation TF-IDF → Modèle Naive Bayes → spam/ham

## Résultats
- Précision globale : **97%**
- Précision spam : **100%**
- Recall spam : **75%**

## Lancer le projet

### En local
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
uvicorn src.predict:app --reload
```

### Avec Docker
```bash
docker build -t spam-detector .
docker run -p 8000:8000 spam-detector
```

### Tester l'API
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Free prize call now!"}'
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