# ui/app.py
import os, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Project Copilot", page_icon="🤖", layout="centered")
st.title("🤖 Project Copilot — RAG local (Ollama/Mistral)")
st.caption("Pose une question (/ask) sur la doc indexée + génère un digest des tickets.")

# --- /ask ---
st.subheader("🔎 Question /ask")
q = st.text_input("Ta question", value="Quels sont les objectifs principaux du projet ?")
if st.button("Envoyer"):
    with st.spinner("Je réfléchis..."):
        try:
            r = requests.post(f"{API_BASE}/ask", json={"question": q}, timeout=120)
            r.raise_for_status()
            data = r.json()
            st.markdown("### Réponse")
            st.write(data.get("answer", "(pas de réponse)"))
            st.markdown("**Sources utilisées**")
            srcs = data.get("sources", [])
            st.write(", ".join(sorted({s or '?' for s in srcs})) or "(aucune)")
            st.markdown("**Meta**")
            st.json(data.get("meta", {}))
        except Exception as e:
            st.error(f"Erreur /ask : {e}")

st.divider()

# --- /digest ---
st.subheader("📊 Digest tickets /digest")
st.caption("Lit data/tickets.csv, calcule des KPIs simples et produit une synthèse.")
if st.button("Générer le digest"):
    with st.spinner("Je compile les infos..."):
        try:
            r = requests.get(f"{API_BASE}/digest", timeout=120)
            r.raise_for_status()
            data = r.json()
            st.markdown("### KPIs")
            st.json(data.get("kpis", {}))
            st.markdown("### Synthèse")
            st.write(data.get("summary", "(Résumé indisponible)"))
        except Exception as e:
            st.error(f"Erreur /digest : {e}")