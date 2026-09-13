import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/AlaaNabil98/"
    "CodeClause_Customer_Churn_Rate_Analysis/main/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

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

TARGET_COLUMN = "churned"


def load_dataset() -> pd.DataFrame:
    raw = pd.read_csv(DATASET_URL)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)

    df = pd.DataFrame()
    df["usage_frequency_30d"] = raw["tenure"]
    df["usage_trend_pct"] = (
        raw["MonthlyCharges"] - raw["MonthlyCharges"].mean()
    ) / raw["MonthlyCharges"].mean()
    df["incidents_recents"] = (raw["TechSupport"] == "Yes").astype(int)
    df["retard_paiement"] = (
        (raw["PaperlessBilling"] == "No") & (raw["PaymentMethod"] == "Mailed check")
    ).astype(int)
    optional_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "StreamingTV", "StreamingMovies",
    ]
    df["tickets_sav_30d"] = (raw[optional_services] == "No").sum(axis=1)
    df["exposition_marketing"] = (raw["Contract"] == "Month-to-month").astype(int)
    df["eu_panne_recente"] = (
        (raw["InternetService"] != "No") & (raw["DeviceProtection"] == "No")
    ).astype(int)
    df["panne_resolue_lentement"] = (
        (df["eu_panne_recente"] == 1) & (raw["TechSupport"] == "No")
    ).astype(int)
    df[TARGET_COLUMN] = (raw["Churn"] == "Yes").astype(int)

    return df