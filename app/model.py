"""Chargement du modèle de régression et calcul des prédictions.

Le modèle est produit par `train_model.py`, qui écrit `regression.joblib` à la
racine du projet. Il est chargé une seule fois puis gardé en mémoire : le
recharger à chaque requête coûterait une lecture disque pour rien.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "regression.joblib"

# L'ordre des colonnes doit être celui de l'entraînement, sinon le modèle reçoit
# la surface là où il attend le nombre de chambres, et rend un prix absurde sans
# lever la moindre erreur.
COLUMNS = ["size", "nb_rooms", "garden"]


class ModelUnavailable(RuntimeError):
    """Le fichier du modèle est absent : il faut lancer `train_model.py`."""


@lru_cache(maxsize=1)
def load_model():
    """Charge le modèle depuis le disque, une fois pour toutes."""
    if not MODEL_PATH.exists():
        raise ModelUnavailable(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Lancer `python train_model.py` avant de démarrer le service."
        )
    return joblib.load(MODEL_PATH)


def predict(size: float, nb_rooms: int, garden: int) -> float:
    """Rend le prix estimé pour une maison.

    On passe un DataFrame et non un tableau brut : le modèle a été entraîné sur
    des colonnes nommées, et scikit-learn se contente d'un avertissement quand
    les noms manquent, ce qui laisserait passer une inversion de colonnes.
    """
    features = pd.DataFrame([[size, nb_rooms, garden]], columns=COLUMNS)
    return float(load_model().predict(features)[0])
