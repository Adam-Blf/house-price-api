"""Schémas d'entrée et de sortie du service de prédiction."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Caractéristiques d'une maison, dans l'ordre attendu par le modèle."""

    size: float = Field(..., gt=0, description="Surface habitable, en mètres carrés")
    nb_rooms: int = Field(..., ge=0, description="Nombre de chambres")
    garden: int = Field(..., ge=0, le=1, description="Présence d'un jardin : 0 ou 1")

    model_config = {
        "json_schema_extra": {"examples": [{"size": 120.0, "nb_rooms": 3, "garden": 1}]}
    }


class PredictionResponse(BaseModel):
    """Prix prédit pour la maison décrite en entrée."""

    y_pred: float = Field(
        ..., description="Prix estimé, dans l'unité du jeu d'entraînement"
    )
