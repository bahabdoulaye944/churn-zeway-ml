"""
API de scoring churn.

Local :  uvicorn app.main:app --reload
Lambda : ce même fichier est packagé dans le conteneur Docker, et 'handler'
         (via Mangum) sert d'entrypoint AWS Lambda.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"

app = FastAPI(title="Churn ZeWay — API de scoring")

_model = None
_explainer = None


def get_model():
    """
    Charge le modèle et crée l'explainer SHAP à la première utilisation
    seulement (pas au démarrage de la fonction) — import de shap différé
    ici, car son chargement est lent (compilation JIT via numba) et ne
    doit pas bloquer des routes qui n'en ont pas besoin, comme /health.
    """
    global _model, _explainer
    if _model is None:
        import shap

        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


class ClientFeatures(BaseModel):
    usage_frequency_30d: float
    usage_trend_pct: float
    incidents_recents: int
    retard_paiement: int
    tickets_sav_30d: int
    exposition_marketing: int
    eu_panne_recente: int
    panne_resolue_lentement: int


class PredictionResponse(BaseModel):
    risk_score: float
    top_factors: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures):
    model, explainer = get_model()
    df = pd.DataFrame([features.dict()])

    score = float(model.predict_proba(df)[0][1])
    shap_values = explainer.shap_values(df)
    contributions = dict(zip(df.columns, shap_values[0].tolist()))
    top_factors = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    )

    return PredictionResponse(risk_score=round(score, 3), top_factors=top_factors)


handler = Mangum(app)