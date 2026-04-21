import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

# Chargement du dataset
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"])

# Encodage : spam = 1, ham = 0 car le modèle ne comprends pas les mots, raison de conversion
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorisation du texte
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entraînement avec tracking MLflow
mlflow.set_experiment("spam-detector")

with mlflow.start_run():
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Évaluation
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Précision : {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Sauvegarde dans MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

# Sauvegarde locale du modèle et du vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Modèle sauvegardé dans models/")