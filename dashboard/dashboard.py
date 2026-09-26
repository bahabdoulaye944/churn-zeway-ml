import requests
import streamlit as st

API_URL = "https://ba8n0asdl8.execute-api.eu-north-1.amazonaws.com/predict"
API_KEY = st.secrets["API_KEY"]

st.set_page_config(page_title="Suivi résiliation clients", layout="wide")
st.title("📊 Suivi des clients à risque de résiliation")
st.caption("Prédiction basée sur le modèle churn — scores mis à jour en temps réel")

# Quelques clients types pour la démo (à remplacer plus tard par une vraie liste)
clients_demo = [
    {"nom": "Client A", "usage_frequency_30d": 1, "usage_trend_pct": 0.3, "incidents_recents": 0,
     "retard_paiement": 0, "tickets_sav_30d": 4, "exposition_marketing": 1,
     "eu_panne_recente": 1, "panne_resolue_lentement": 1},
    {"nom": "Client B", "usage_frequency_30d": 45, "usage_trend_pct": -0.1, "incidents_recents": 0,
     "retard_paiement": 0, "tickets_sav_30d": 1, "exposition_marketing": 0,
     "eu_panne_recente": 0, "panne_resolue_lentement": 0},
    {"nom": "Client C", "usage_frequency_30d": 3, "usage_trend_pct": 0.5, "incidents_recents": 1,
     "retard_paiement": 1, "tickets_sav_30d": 5, "exposition_marketing": 1,
     "eu_panne_recente": 1, "panne_resolue_lentement": 0},
]

FACTOR_LABELS = {
    "usage_frequency_30d": "Ancienneté du client",
    "usage_trend_pct": "Écart de facturation vs moyenne",
    "incidents_recents": "Support technique récent",
    "retard_paiement": "Retard de paiement",
    "tickets_sav_30d": "Services non souscrits",
    "exposition_marketing": "Contrat sans engagement",
    "eu_panne_recente": "Panne récente",
    "panne_resolue_lentement": "Panne résolue lentement",
}

results = []
errors = []

with st.spinner("Calcul des scores en cours..."):
    for client in clients_demo:
        payload = {k: v for k, v in client.items() if k != "nom"}
        try:
            response = requests.post(
                API_URL, json=payload, headers={"X-API-Key": API_KEY}, timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "Client": client["nom"],
                    "Score de risque": data["risk_score"],
                    "Facteurs principaux": ", ".join(
                        FACTOR_LABELS.get(f, f) for f in data["top_factors"].keys()
                    ),
                })
            else:
                errors.append(f"{client['nom']} : erreur HTTP {response.status_code} — {response.text}")
        except Exception as e:
            errors.append(f"{client['nom']} : exception — {e}")

for err in errors:
    st.error(err)

results.sort(key=lambda x: x["Score de risque"], reverse=True)

for r in results:
    score = r["Score de risque"]
    color = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(r["Client"], f"{color} {score:.0%}")
        with col2:
            st.write(f"**Facteurs principaux :** {r['Facteurs principaux']}")