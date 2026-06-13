import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

@pytest.fixture
def trained_model():
    df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def test_model_loads(trained_model):
    model, vectorizer = trained_model
    assert model is not None

def test_spam_prediction(trained_model):
    model, vectorizer = trained_model
    text_vec = vectorizer.transform(["WINNER! Free prize call now!"])
    prediction = model.predict(text_vec)[0]
    assert prediction == 1

def test_ham_prediction(trained_model):
    model, vectorizer = trained_model
    text_vec = vectorizer.transform(["Hey, are you coming to the meeting?"])
    prediction = model.predict(text_vec)[0]
    assert prediction == 0