from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    df = load_dataset()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    # Split stratifié 80/20 (voir note : split temporel impossible sur ce
    # dataset public, contrairement aux vraies données ZeWay avec historique daté)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Compensation du déséquilibre de classes (26.5% de résiliations)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, preds_proba)
    print(f"Test AUC={auc:.3f}")
    print()
    print(classification_report(y_test, preds, target_names=["Pas de churn", "Churn"]))

    joblib.dump(model, MODELS_DIR / "churn_model.joblib")
    print("Modèle sauvegardé dans models/churn_model.joblib")


if __name__ == "__main__":
    main()