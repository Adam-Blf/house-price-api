"""Tests du service de prédiction.

Le modèle doit exister avant de lancer ces tests : `python train_model.py`.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HOUSE = {"size": 120.0, "nb_rooms": 3, "garden": 1}


def test_health_repond_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rend_un_prix():
    response = client.post("/predict", json=HOUSE)
    assert response.status_code == 200
    assert response.json()["y_pred"] > 0


def test_predict_en_get_est_refuse():
    """L'endpoint GET a été retiré : il doit répondre 405, pas 200."""
    response = client.get("/predict", params=HOUSE)
    assert response.status_code == 405


def test_une_surface_negative_est_refusee():
    response = client.post("/predict", json={**HOUSE, "size": -10})
    assert response.status_code == 422


def test_un_jardin_hors_bornes_est_refuse():
    """Le modèle a été entraîné sur garden dans {0, 1} : 2 n'a aucun sens."""
    response = client.post("/predict", json={**HOUSE, "garden": 2})
    assert response.status_code == 422


def test_un_champ_manquant_est_refuse():
    response = client.post("/predict", json={"size": 120})
    assert response.status_code == 422


def test_une_grande_maison_vaut_plus_qu_une_petite():
    """Contrôle de cohérence métier, pas seulement de plomberie HTTP.

    Le prix doit croître avec la surface. Un test qui vérifie seulement le code
    HTTP resterait vert avec un modèle chargé à l'envers.
    """
    small = client.post("/predict", json={**HOUSE, "size": 60}).json()["y_pred"]
    large = client.post("/predict", json={**HOUSE, "size": 240}).json()["y_pred"]
    assert large > small
