"""
Rapport de monitoring du drift : compare les données de référence
(entraînement) avec de nouvelles données (production réelle ou simulées),
via Evidently AI.

Lancer : python monitoring.py
"""
import boto3
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.features import FEATURE_COLUMNS, load_dataset

DYNAMODB_TABLE = "churn-predictions-log"
DYNAMODB_REGION = "eu-north-1"

# Données de référence : celles utilisées à l'entraînement
reference_data = load_dataset()[FEATURE_COLUMNS]


def generate_report(current_data: pd.DataFrame, output_name: str):
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_data, current_data=current_data)
    result.save_html(f"{output_name}.html")
    print(f"Rapport généré : {output_name}.html")


def load_real_predictions() -> pd.DataFrame:
    """
    Récupère les vraies requêtes reçues par l'API, journalisées dans
    DynamoDB à chaque appel de /predict.
    """
    table = boto3.resource("dynamodb", region_name=DYNAMODB_REGION).Table(
        DYNAMODB_TABLE
    )
    items = table.scan().get("Items", [])
    if not items:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = pd.DataFrame(items)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FEATURE_COLUMNS].dropna()


def generate_fake_drifted_data(n=500):
    """
    Génère des clients factices avec un profil délibérément différent
    des données d'entraînement, pour vérifier que le monitoring
    détecte bien un vrai changement quand il y en a un.
    """
    np.random.seed(42)

    fake = pd.DataFrame({
        "usage_frequency_30d": np.random.randint(50, 100, n),
        "usage_trend_pct": np.random.uniform(1.5, 3.0, n),
        "incidents_recents": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "retard_paiement": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "tickets_sav_30d": np.random.randint(0, 2, n),
        "exposition_marketing": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "eu_panne_recente": np.random.choice([0, 1], n, p=[0.1, 0.9]),
        "panne_resolue_lentement": np.random.choice([0, 1], n, p=[0.1, 0.9]),
    })
    return fake


if __name__ == "__main__":
    generate_report(reference_data, "monitoring_report_reference")

    fake_data = generate_fake_drifted_data()
    generate_report(fake_data, "monitoring_report_fake_drift")

    real_data = load_real_predictions()
    if len(real_data) >= 10:
        generate_report(real_data, "monitoring_report_real")
        print(f"{len(real_data)} vraies requêtes analysées")
    else:
        print(
            f"Seulement {len(real_data)} requête(s) réelle(s) enregistrée(s) — "
            "au moins 10 recommandées pour un rapport significatif"
        )