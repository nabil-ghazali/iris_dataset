import pytest
from fastapi.testclient import TestClient
from app.api import app

"""
    On utilise le TestClient de FastAPI pour appeler l’API localement sans lancer le serveur.

"""

client = TestClient(app)


# Test route GET /
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Bienvenue" in data["message"]


# Test la route /predict
def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "variety" in data
    assert data["prediction"] in [
        0,
        1,
        2,
    ]
