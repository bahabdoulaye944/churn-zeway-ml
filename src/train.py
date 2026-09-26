"""
Entraînement du modèle churn, avec tracking MLflow systématique.

Lancer : python -m src.train
Voir les runs : mlflow ui
"""
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATASET_URL = (
    "https://raw.githubusercontent.com/AlaaNabil98/"
    "CodeClause_Customer_Churn_Rate_Analysis/main/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MIN_AUC = 0.75  # Garde-fou : on ne sauvegarde/déploie jamais un modèle sous ce seuil


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset public "Telco Customer Churn" (IBM, via GitHub raw)
    et mappe ses colonnes vers le schéma métier ZeWay (voir README étape 2
    pour la justification de chaque mapping).

    NOTE METHODOLOGIQUE : ce dataset est une photo à un instant T, sans
    date par client. Un split temporel (comme on ferait sur les vraies
    données ZeWay avec un historique daté) n'est donc pas possible ici.
    On utilise un split stratifié classique à la place (voir main()),
    documenté explicitement comme une limite du dataset de substitution.
    """
    raw = pd.read_csv(DATASET_URL)

    # Nettoyage : 11 clients avec tenure=0 ont TotalCharges vide (chaîne,
    # pas un vrai NaN — non détecté par isnull()). Ce sont des clients
    # tout juste arrivés : on met TotalCharges à 0, cohérent avec tenure=0.
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)

    df = pd.DataFrame()
    df["usage_frequency_30d"] = raw["tenure"]
    df["usage_trend_pct"] = (
        raw["MonthlyCharges"] - raw["MonthlyCharges"].mean()
    ) / raw["MonthlyCharges"].mean()
    df["incidents_recents"] = (raw["TechSupport"] == "Yes").astype(int)
    df["retard_paiement"] = (
        (raw["PaperlessBilling"] == "No")
        & (raw["PaymentMethod"] == "Mailed check")
    ).astype(int)
    optional_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["tickets_sav_30d"] = (raw[optional_services] == "No").sum(axis=1)
    df["exposition_marketing"] = (raw["Contract"] == "Month-to-month").astype(int)

    # Proxy pour "panne récente" et "vitesse de résolution" (voir README
    # étape 2 : le dataset public n'a pas de vrai historique de tickets
    # daté, contrairement aux vraies données ZeWay via Jira).
    df["eu_panne_recente"] = (
        (raw["InternetService"] != "No") & (raw["DeviceProtection"] == "No")
    ).astype(int)
    df["panne_resolue_lentement"] = (
        (df["eu_panne_recente"] == 1) & (raw["TechSupport"] == "No")
    ).astype(int)

    df[TARGET_COLUMN] = (raw["Churn"] == "Yes").astype(int)

    return df


def main():
    mlflow.set_experiment("churn-zeway")

    df = load_dataset()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    # Split stratifié 80/20 (voir note méthodologique dans load_dataset
    # sur le split temporel non applicable à ce dataset de substitution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Compensation du déséquilibre de classes (26.5% de résiliations)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)

        # Évaluation UNIQUEMENT sur le jeu de test, jamais vu à l'entraînement
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, preds_proba)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        # Garde-fou : on ne sauvegarde/déploie jamais un modèle sous ce seuil
        if auc < MIN_AUC:
            print(f"ÉCHEC : AUC={auc:.3f} sous le seuil minimum de {MIN_AUC}")
            raise SystemExit(1)

        mlflow.log_params(model.get_params())
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metrics(
            {"test_auc": auc, "test_precision": precision, "test_recall": recall}
        )
        mlflow.xgboost.log_model(model, "model")

        joblib.dump(model, MODELS_DIR / "churn_model.joblib")

        print(f"Test AUC={auc:.3f} | precision={precision:.3f} | recall={recall:.3f}")
        print()
        print("Matrice de confusion (test) :")
        print(confusion_matrix(y_test, preds))
        print()
        print(classification_report(y_test, preds, target_names=["Pas de churn", "Churn"]))


if __name__ == "__main__":
    main()