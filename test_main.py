from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_predict_returns_boolean():
    r = client.post("/predict", json={
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70, "BMI": 28.5, "Age": 45
    })
    assert r.status_code == 200
    assert isinstance(r.json()["diabetic"], bool)


def test_rejects_invalid_input():
    r = client.post("/predict", json={
        "Pregnancies": 2, "Glucose": "abc", "BloodPressure": 70, "BMI": 28.5, "Age": 45
    })
    assert r.status_code == 422