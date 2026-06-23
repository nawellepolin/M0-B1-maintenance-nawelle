# tests/test_predict.py
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Fixture partagée : un seul TestClient pour tout le module."""
    with TestClient(app) as c:
        yield c

def test_predict_cas_valide(client):
    payload = {
        "type_machine": "compresseur",
        "age_machine_jours": 1500,
        "derniere_maintenance_jours": 45,
        "temperature_moyenne": 68.5,
        "vibration_moyenne": 3.2,
        "pression_moyenne": 7.8,
        "nb_incidents_3_mois": 2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["criticite"] in {"basse", "moyenne", "haute"}
    assert set(body["probabilites"].keys()) == {"basse", "moyenne", "haute"}
    # somme des probas ≈ 1
    assert abs(sum(body["probabilites"].values()) - 1.0) < 1e-6

def test_predict_type_machine_invalide(client):
    payload = {
        "type_machine": "INCONNU",
        "age_machine_jours": 1000,
        "derniere_maintenance_jours": 30,
        "temperature_moyenne": 65.0,
        "vibration_moyenne": 3.0,
        "pression_moyenne": 8.0,
        "nb_incidents_3_mois": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

@pytest.mark.parametrize("type_m", ["pompe", "convoyeur", "presse", "four"])
def test_predict_tous_types_machine(client, type_m):
    payload = {
        "type_machine": type_m,
        "age_machine_jours": 2000,
        "derniere_maintenance_jours": 60,
        "temperature_moyenne": 70.0,
        "vibration_moyenne": 3.5,
        "pression_moyenne": 8.5,
        "nb_incidents_3_mois": 2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["criticite"] in {"basse", "moyenne", "haute"}
