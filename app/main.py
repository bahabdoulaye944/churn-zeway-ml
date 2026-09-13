from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

#MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"

app = FastAPI(title="Churn ZeWay - API de scoring")

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)


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
    df = pd.DataFrame([features.dict()])

    score = float(model.predict_proba(df)[0][1])
    shap_values = explainer.shap_values(df)
    contributions = dict(zip(df.columns, shap_values[0].tolist()))
    top_factors = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    )

    return PredictionResponse(risk_score=round(score, 3), top_factors=top_factors)


