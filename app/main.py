"""
API de scoring churn.

Local :  uvicorn app.main:app --reload
Lambda : ce même fichier est packagé dans le conteneur Docker, et 'handler'
         (via Mangum) sert d'entrypoint AWS Lambda.
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from mangum import Mangum
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"

app = FastAPI(title="Churn ZeWay — API de scoring")

_model = None
_explainer = None


def get_model():
    global _model, _explainer
    if _model is None:
        import shap

        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def verify_api_key(x_api_key: str = Header(None)):
    """
    Vérification au niveau applicatif (les clés natives d'API Gateway ne
    sont pas disponibles sur une HTTP API, seulement sur REST API).
    Le secret attendu est stocké en variable d'environnement Lambda,
    jamais en dur dans le code.
    """
    expected = os.environ.get("API_SECRET_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=403, detail="Clé API manquante ou invalide")


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
def predict(features: ClientFeatures, _: None = Depends(verify_api_key)):
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