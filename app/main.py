"""
API de scoring churn.

Local :  uvicorn app.main:app --reload
Lambda : ce même fichier est packagé dans le conteneur Docker, et 'handler'
         (via Mangum) sert d'entrypoint AWS Lambda.
"""
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from mangum import Mangum
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "churn-predictions-log")

app = FastAPI(title="Churn ZeWay — API de scoring")

_model = None
_explainer = None
_dynamo_table = None


def get_model():
    global _model, _explainer
    if _model is None:
        import shap

        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def get_dynamo_table():
    global _dynamo_table
    if _dynamo_table is None:
        _dynamo_table = boto3.resource("dynamodb").Table(DYNAMODB_TABLE)
    return _dynamo_table


def log_prediction(features: dict, risk_score: float):
    """
    Enregistre chaque prédiction réelle dans DynamoDB, pour que le
    monitoring puisse comparer les vraies requêtes entrantes aux données
    de référence (pas seulement des données factices).
    Échoue silencieusement si l'écriture rate — ne doit jamais bloquer
    la réponse à l'utilisateur pour un problème de journalisation.
    """
    try:
        table = get_dynamo_table()
        item = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": str(risk_score),
        }
        item.update({k: str(v) for k, v in features.items()})
        table.put_item(Item=item)
    except Exception as e:
        print(f"Avertissement : échec de journalisation DynamoDB — {e}")


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

    log_prediction(features.dict(), score)

    return PredictionResponse(risk_score=round(score, 3), top_factors=top_factors)


handler = Mangum(app)