# Churn ZeWay — Prédiction de résiliation client

Projet ML de bout en bout : EDA → feature engineering → modélisation (comparaison régression logistique / XGBoost) → explicabilité SHAP → API de scoring.

Inspiré d'un cas réel (opérateur de trottinettes/scooters électriques), reconstruit sur le dataset public "Telco Customer Churn" (IBM), faute d'accès aux données réelles.

## Problème business

Identifier les clients à risque de résiliation pour déclencher une action de rétention ciblée, avant que le client parte plutôt qu'après.

## Démarche

1. **EDA** : déséquilibre de classes (26,5% de résiliations), 11 valeurs invalides dans `TotalCharges` corrigées, aucun doublon
2. **Feature engineering** : 8 variables reconstruisant le contexte ZeWay à partir des colonnes Telco (voir `src/features.py` pour le détail de chaque mapping)
3. **Modélisation** : comparaison régression logistique (baseline) vs XGBoost — écart de performance négligeable sur ce dataset (AUC 0.831 vs 0.835), documenté honnêtement plutôt que masqué
4. **Explicabilité** : SHAP au niveau global (quelles features comptent) et individuel (pourquoi CE client a ce score)
5. **API** : service FastAPI qui expose le scoring avec justification SHAP

## Limites connues (assumées)

- Split stratifié plutôt que temporel : ce dataset public est une photo à un instant T, sans date par client — un vrai split temporel serait utilisé sur les données ZeWay réelles
- Les features de panne (`eu_panne_recente`, `panne_resolue_lentement`) sont des proxys construits sur des colonnes d'abonnement, faute de vrai historique de tickets daté — signal plus fort attendu sur données réelles

## Structure du repo
app/ API FastAPI (/predict)
src/ Chargement des données, feature engineering, entraînement
notebooks/ Exploration initiale (EDA, tests de features)
models/ Modèle entraîné (non versionné, voir .gitignore)


## Lancer en local

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

python -m src.train          # Entraîne et sauvegarde le modèle
uvicorn app.main:app --reload   # Lance l'API sur http://localhost:8000/docs
```

## Résultats

| Modèle | AUC (test) | Precision (Churn) | Recall (Churn) |
|---|---|---|---|
| Régression logistique | 0.831 | 0.50 | 0.79 |
| XGBoost | 0.835 | 0.51 | 0.79 |