# %%
import pandas as pd

# %%
URL = "https://raw.githubusercontent.com/AlaaNabil98/CodeClause_Customer_Churn_Rate_Analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(URL)

# %%
df.shape

# %%
df.head()

# %%
df["Churn"].value_counts()

# %%
df.isnull().sum()

# %%
df["TotalCharges"].apply(lambda x: not str(x).strip().replace(".", "", 1).isdigit()).sum()
# %%

# %%
df[df["TotalCharges"].apply(lambda x: not str(x).strip().replace(".", "", 1).isdigit())][["tenure", "MonthlyCharges", "TotalCharges"]]
# %%

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["TotalCharges"].dtype

# %%
df["TotalCharges"].isnull().sum()
# %%

# %%
df["customerID"].duplicated().sum()

# %%
df.groupby("Contract")["Churn"].value_counts(normalize=True)

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# %%
df["exposition_marketing"] = (df["Contract"] == "Month-to-month").astype(int)
df[["Contract", "exposition_marketing"]].head()

# %%
df["usage_frequency_30d"] = df["tenure"]

# %%
df["usage_trend_pct"] = (df["MonthlyCharges"] - df["MonthlyCharges"].mean()) / df["MonthlyCharges"].mean()
df[["MonthlyCharges", "usage_trend_pct"]].head()

# %%
df["incidents_recents"] = (df["TechSupport"] == "Yes").astype(int)

# %%
df["retard_paiement"] = ((df["PaperlessBilling"] == "No") & (df["PaymentMethod"] == "Mailed check")).astype(int)

# %%
optional_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
df["tickets_sav_30d"] = (df[optional_services] == "No").sum(axis=1)
df[["incidents_recents", "retard_paiement", "tickets_sav_30d"]].head()

# %%
df["eu_panne_recente"] = ((df["InternetService"] != "No") & (df["DeviceProtection"] == "No")).astype(int)

# %%
df["panne_resolue_lentement"] = ((df["eu_panne_recente"] == 1) & (df["TechSupport"] == "No")).astype(int)
df[["InternetService", "DeviceProtection", "eu_panne_recente", "TechSupport", "panne_resolue_lentement"]].head()

# %%
df["churned"] = (df["Churn"] == "Yes").astype(int)

# %%
FEATURE_COLUMNS = [
    "usage_frequency_30d",
    "usage_trend_pct",
    "incidents_recents",
    "retard_paiement",
    "tickets_sav_30d",
    "exposition_marketing",
    "eu_panne_recente",
    "panne_resolue_lentement",
]

df_final = df[FEATURE_COLUMNS + ["churned"]]
df_final.head()

# %%
df_final.shape

# %%
from sklearn.model_selection import train_test_split

X = df_final[FEATURE_COLUMNS]
y = df_final["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train.shape, X_test.shape

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# La régression logistique est sensible à l'échelle des variables
# (contrairement à XGBoost) : usage_frequency_30d va de 0 à ~70,
# tickets_sav_30d de 0 à 5 -> sans mise à l'échelle, le modèle
# surpondérerait les variables aux valeurs les plus grandes.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
baseline_model = LogisticRegression(class_weight="balanced", random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# %%
from sklearn.metrics import roc_auc_score, classification_report

baseline_preds_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
baseline_preds = baseline_model.predict(X_test_scaled)

print("AUC baseline (régression logistique):", roc_auc_score(y_test, baseline_preds_proba))
print()
print(classification_report(y_test, baseline_preds, target_names=["Pas de churn", "Churn"]))

# %%
from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
scale_pos_weight

# %%
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
)
model.fit(X_train, y_train)


# %%
xgb_preds_proba = model.predict_proba(X_test)[:, 1]
xgb_preds = model.predict(X_test)

print("AUC XGBoost:", roc_auc_score(y_test, xgb_preds_proba))
print()
print(classification_report(y_test, xgb_preds, target_names=["Pas de churn", "Churn"]))


# %%
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_values.shape

# %%
import numpy as np
import pandas as pd

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance_moyenne": np.abs(shap_values).mean(axis=0)
}).sort_values("importance_moyenne", ascending=False)

importance

# %%
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS)

# %%
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
risk_scores = model.predict_proba(X_test)[:, 1]

results = X_test_reset.copy()
results["risk_score"] = risk_scores
results["churned_reel"] = y_test_reset

results.sort_values("risk_score", ascending=False).head(5)

# %%
idx = results.sort_values("risk_score", ascending=False).index[0]

shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[idx],
    X_test_reset.iloc[idx],
    feature_names=FEATURE_COLUMNS,
)

# %%
import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.joblib")
print("Modèle sauvegardé")

# %%
