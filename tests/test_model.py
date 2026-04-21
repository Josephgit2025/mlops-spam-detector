import joblib

def test_model_loads():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    assert model is not None
    assert vectorizer is not None

def test_spam_prediction():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    text = ["WINNER! Free prize call now 0906123456"]
    vec = vectorizer.transform(text)
    prediction = model.predict(vec)[0]
    assert prediction == 1  # doit être spam

def test_ham_prediction():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    text = ["Hey, are we still meeting tomorrow for lunch?"]
    vec = vectorizer.transform(text)
    prediction = model.predict(vec)[0]
    assert prediction == 0  # doit être ham