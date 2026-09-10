"""Service web de prédiction du prix d'une maison.

Expose `/predict` en POST. Les caractéristiques de la maison voyagent dans le
corps de la requête, ce qui est leur place : ce sont des données d'entrée d'un
calcul, pas l'identification d'une ressource qu'on irait chercher.
"""

from fastapi import FastAPI, HTTPException

from app.model import ModelUnavailable, predict
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Prédiction du prix des maisons",
    description=(
        "Mini projet du module Machine learning in production. "
        "Le modèle est chargé depuis regression.joblib au premier appel."
    ),
    version="2.0.0",
)


@app.get("/health", summary="Vérifier que le service répond")
def health() -> dict[str, str]:
    """Sonde de vie. Ne charge pas le modèle : elle teste le serveur, pas le modèle."""
    return {"status": "ok"}


@app.post(
    "/predict", response_model=PredictionResponse, summary="Prédire depuis un corps JSON"
)
def predict_post(request: PredictionRequest) -> PredictionResponse:
    """Prédit le prix d'une maison à partir d'un corps JSON.

    Un modèle manquant devient un 503 plutôt qu'un 500 : c'est un problème de
    déploiement, pas une requête fautive. Le service n'est pas prêt, il n'est
    pas cassé.
    """
    try:
        return PredictionResponse(
            y_pred=predict(request.size, request.nb_rooms, request.garden)
        )
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
