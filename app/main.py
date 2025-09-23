# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd

# Vector store
from langchain_community.vectorstores import Chroma

# Ollama (LangChain)
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()

app = FastAPI(
    title="Project Copilot API",
    version="0.1.0",
    description="RAG local (Ollama) : /ask répond à partir des docs indexés + sources."
)

# === Config (.env) ===
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
EMBED_MODEL     = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "mistral:latest")
DATA_DIR        = os.path.join(os.getcwd(), "data")  # pour /digest

# === Chargement du store + retriever ===
# IMPORTANT : la collection doit matcher celle de l’ingestion ("copilot")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vs = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings,
    collection_name="copilot"
)

# Retriever MMR (diversité des passages)
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5}
)

# LLM paramétré (réponses courtes, factuelles)
llm = ChatOllama(
    model=LLM_MODEL,
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
    num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
)

class AskPayload(BaseModel):
    question: str

SYSTEM = (
    "Tu es un assistant projet. Réponds UNIQUEMENT à partir des extraits fournis. "
    "Si l'information n'est pas dans les extraits, réponds : 'Je ne sais pas.' "
    "Style: concis, factuel. Ajoute les sources (nom de fichier) en fin de réponse."
)

# Seuil de confiance — si trop peu de sources pertinentes trouvées
CONFIDENCE_MIN_SOURCES = 1  # passe à 2 quand tu auras plus de docs

@app.post("/ask")
def ask(payload: AskPayload):
    question = payload.question.strip()

    # 1) Retrieve (MMR)
    docs = retriever.get_relevant_documents(question)

    # 2) Contrôle de confiance (nb de sources distinctes)
    sources = list({(d.metadata.get("source") or "?") for d in docs})
    if len(sources) < CONFIDENCE_MIN_SOURCES:
        return {
            "answer": "Je ne sais pas. Je n'ai pas assez de contexte fiable dans la documentation indexée.",
            "sources": sources,
            "meta": {"confidence": "low", "retrieved": len(docs)}
        }

    # 3) Construit le contexte + prompt
    context = "\n\n".join(
        f"[{i+1}] {d.page_content}\n(SOURCE: {d.metadata.get('source','?')})"
        for i, d in enumerate(docs)
    )
    prompt = f"{SYSTEM}\n\nContexte:\n{context}\n\nQuestion: {question}\nRéponse:"

    # 4) Appel LLM (Ollama local)
    msg = llm.invoke(prompt)
    answer_text = getattr(msg, "content", str(msg))

    return {
        "answer": answer_text,
        "sources": sources,
        "meta": {"confidence": "normal", "retrieved": len(docs), "model": LLM_MODEL}
    }

@app.get("/digest")
def digest():
    """
    KPIs simples à partir de data/tickets.csv + synthèse LLM.
    """
    tickets_csv = os.path.join(DATA_DIR, "tickets.csv")
    if not os.path.isfile(tickets_csv):
        return {"kpis": {}, "summary": "Aucun data/tickets.csv trouvé."}

    df = pd.read_csv(tickets_csv)

    total = len(df)
    open_count = int((df["status"] == "open").sum()) if "status" in df.columns else None
    closed_count = int((df["status"] == "closed").sum()) if "status" in df.columns else None
    critical_open = []
    if all(c in df.columns for c in ["status","severity","id"]):
        critical_open = df[(df["status"]=="open") & (df["severity"]=="critical")]["id"].astype(str).tolist()

    # Durée moyenne de résolution (si dates dispo)
    avg_resolution = None
    if all(c in df.columns for c in ["created_at","closed_at","status"]):
        closed_df = df[df["status"]=="closed"].copy()
        if not closed_df.empty:
            closed_df["created_at"] = pd.to_datetime(closed_df["created_at"], errors="coerce")
            closed_df["closed_at"] = pd.to_datetime(closed_df["closed_at"], errors="coerce")
            closed_df["resolution_days"] = (closed_df["closed_at"] - closed_df["created_at"]).dt.days
            try:
                avg_resolution = float(closed_df["resolution_days"].dropna().mean())
            except Exception:
                avg_resolution = None

    kpis = {
        "total_tickets": int(total),
        "open": open_count,
        "closed": closed_count,
        "critical_open_ids": critical_open,
        "avg_resolution_days": avg_resolution
    }

    # Synthèse LLM
    summary_prompt = (
        "Synthétise en 5 bullets clairs l'état des tickets ci-dessous : "
        "points de vigilance, et propose 3 priorités d'action.\n\n"
        f"KPIs: {kpis}"
    )
    msg = llm.invoke(summary_prompt)
    summary_text = getattr(msg, "content", str(msg))

    return {"kpis": kpis, "summary": summary_text}

@app.get("/")
def root():
    return {"ok": True, "message": "Project Copilot API (RAG local via Ollama) prête."}