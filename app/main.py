"""API FastAPI — service de classification de criticité (M0-B1).

Expose un modèle scikit-learn pré-entraîné (cf. `model/train_baseline.py`) via
deux routes :

- `GET /health`  : santé du service (déjà fonctionnel)
- `POST /predict` : prédiction de criticité (🎯 à compléter par l'apprenant)

Le modèle est chargé une seule fois au démarrage via le `lifespan` FastAPI puis
réutilisé pour chaque requête.

Lancement local :
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.schemas import HealthResponse, MachineInput, PredictionResponse

import pandas as pd
import time
from loguru import logger

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.joblib"

# Mémoire d'application — peuplée par le lifespan
state: dict[str, Any] = {}

logger.add(
    "logs/api.log",
    rotation="5 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère à l'arrêt.

    Args:
        app: instance FastAPI.
    """
    if not MODEL_PATH.is_file():
        logger.error(
            f"Modèle introuvable : {MODEL_PATH}. "
            f"Lance d'abord : python model/train_baseline.py"
        )
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

    logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé.")

    yield

    state.clear()
    logger.info("Service arrêté, état libéré.")


app = FastAPI(
    title="FastIA — Service de criticité maintenance prédictive",
    description=(
        "API d'exposition d'un modèle scikit-learn de classification de criticité "
        "d'incidents machine (3 classes : basse, moyenne, haute)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Retourne le statut du service et du modèle.

    Returns:
        HealthResponse — `status="ok"` si le modèle est chargé, `degraded` sinon.
    """
    is_loaded = "model" in state
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(item: MachineInput) -> PredictionResponse:
    """Prédit la criticité d'une machine à partir de ses caractéristiques.

    🎯 **À COMPLÉTER PAR L'APPRENANT.**

    Indices d'implémentation :

    1. Construire un DataFrame pandas à 1 ligne à partir de `item.model_dump()`.
       Le pipeline scikit-learn attend les colonnes dans le même ordre qu'à
       l'entraînement (cf. `model/train_baseline.py`, `NUM_FEATURES` + `CAT_FEATURES`).
    2. Récupérer le modèle via `state["model"]`.
    3. Appeler `model.predict(df)[0]` pour obtenir la classe prédite (str).
    4. Appeler `model.predict_proba(df)[0]` pour obtenir les probabilités.
       Les classes correspondantes sont dans `model.classes_`.
    5. Construire et retourner un `PredictionResponse`.
    6. Logger l'entrée + la classe prédite + le temps de réponse via Loguru.

    Args:
        item: caractéristiques de la machine (cf. `schemas.MachineInput`).

    Returns:
        PredictionResponse avec la classe prédite et les probabilités.
    """

    logger.info(f"Requête /predict reçue : {item.model_dump()}")
    t0 = time.perf_counter()

    # 1. Charger le modèle depuis state
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    # 2. Construire un DataFrame à partir de l'item
    df = pd.DataFrame([item.model_dump()])

    # 3. Prédire la classe et les probabilités
    classe = str(model.predict(df)[0])
    probas = model.predict_proba(df)[0]
    classes = model.classes_

    # 4. Construire le dict {classe: proba}
    proba_dict = {str(c): float(p) for c, p in zip(classes, probas)}

    # 5. Logger l'événement
    logger.info(f"Prédiction : {classe} (entrée={item.model_dump()})")

    duree_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Prédiction = {classe} (durée {duree_ms:.1f} ms)")

    # 6. Retourner la réponse typée
    return PredictionResponse(criticite=classe, probabilites=proba_dict)

