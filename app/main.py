"""Service web de prédiction du prix d'une maison.

Expose `/predict` en GET et en POST. Les deux verbes rendent la même réponse :
la valeur prédite par le modèle de régression linéaire entraîné sur houses.csv.
"""

from fastapi import FastAPI, HTTPException, Query

from app.model import ModelUnavailable, predict
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Prédiction du prix des maisons",
    description=(
        "Mini projet du module Machine learning in production. "
        "Le modèle est chargé depuis regression.joblib au premier appel."
    ),
    version="1.0.0",
)


def _predict_or_503(size: float, nb_rooms: int, garden: int) -> PredictionResponse:
    """Applique le modèle et traduit son absence en 503 plutôt qu'en 500.

    Un modèle manquant est un problème de déploiement, pas une requête fautive :
    le service n'est pas prêt, il n'est pas cassé.
    """
    try:
        return PredictionResponse(y_pred=predict(size, nb_rooms, garden))
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health", summary="Vérifier que le service répond")
def health() -> dict[str, str]:
    """Sonde de vie. Ne charge pas le modèle : elle teste le serveur, pas le modèle."""
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse, summary="Prédire depuis l'URL")
def predict_get(
    size: float = Query(..., gt=0, description="Surface habitable, en mètres carrés"),
    nb_rooms: int = Query(..., ge=0, description="Nombre de chambres"),
    garden: int = Query(0, ge=0, le=1, description="Présence d'un jardin : 0 ou 1"),
) -> PredictionResponse:
    """Prédit le prix à partir des paramètres passés dans l'URL.

    Utile depuis un navigateur, qui ne sait pas envoyer de corps de requête.
    """
    return _predict_or_503(size, nb_rooms, garden)


@app.post(
    "/predict", response_model=PredictionResponse, summary="Prédire depuis un corps JSON"
)
def predict_post(request: PredictionRequest) -> PredictionResponse:
    """Prédit le prix à partir d'un corps JSON."""
    return _predict_or_503(request.size, request.nb_rooms, request.garden)
