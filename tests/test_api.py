"""Tests du service de prédiction.

Le modèle doit exister avant de lancer ces tests : `python train_model.py`.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HOUSE = {"size": 120.0, "nb_rooms": 3, "garden": 1}


def test_health_repond_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_get_rend_un_prix():
    response = client.get("/predict", params=HOUSE)
    assert response.status_code == 200
    assert response.json()["y_pred"] > 0


def test_predict_post_rend_un_prix():
    response = client.post("/predict", json=HOUSE)
    assert response.status_code == 200
    assert response.json()["y_pred"] > 0


def test_les_deux_verbes_donnent_la_meme_valeur():
    """GET et POST servent le même modèle : ils ne peuvent pas diverger."""
    by_get = client.get("/predict", params=HOUSE).json()["y_pred"]
    by_post = client.post("/predict", json=HOUSE).json()["y_pred"]
    assert by_get == pytest.approx(by_post)


def test_une_surface_negative_est_refusee():
    response = client.get("/predict", params={**HOUSE, "size": -10})
    assert response.status_code == 422


def test_un_jardin_hors_bornes_est_refuse():
    """Le modèle a été entraîné sur garden dans {0, 1} : 2 n'a aucun sens."""
    response = client.post("/predict", json={**HOUSE, "garden": 2})
    assert response.status_code == 422


def test_une_grande_maison_vaut_plus_qu_une_petite():
    """Contrôle de cohérence métier, pas seulement de plomberie HTTP.

    Le prix doit croître avec la surface. Un test qui vérifie seulement le code
    HTTP resterait vert avec un modèle chargé à l'envers.
    """
    small = client.post("/predict", json={**HOUSE, "size": 60}).json()["y_pred"]
    large = client.post("/predict", json={**HOUSE, "size": 240}).json()["y_pred"]
    assert large > small
