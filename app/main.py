from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Vector store
from langchain_community.vectorstores import Chroma

# ✅ Quick win 1: imports mis à jour (lib dédiée Ollama)
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()
app = FastAPI(
    title="Project Copilot API",
    version="0.1.0",
    description="RAG local (Ollama) : /ask répond à partir des docs indexés + sources."
)

# === Config ===
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
EMBED_MODEL     = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3:8b")

# === Chargement du store + retriever ===
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vs = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

# ✅ Quick win 2: retriever MMR (diversité des passages)
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5}
)

# ✅ Quick win 2: LLM paramétré pour réponses factuelles et concises
llm = ChatOllama(
    model=LLM_MODEL,
    num_ctx=8192,      # plus grand contexte = moins de coupures
    temperature=0.2,   # plus factuel
    num_predict=256,   # réponses concises
)

class AskPayload(BaseModel):
    question: str

SYSTEM = (
    "Tu es un assistant projet. Réponds UNIQUEMENT à partir des extraits fournis. "
    "Si l'information n'est pas dans les extraits, réponds : 'Je ne sais pas.' "
    "Style: concis, factuel. Ajoute les sources (nom de fichier) en fin de réponse."
)

# ✅ Quick win 3: seuil de confiance — si trop peu de sources pertinentes trouvées
CONFIDENCE_MIN_SOURCES = 1  # Mets 2 quand tu auras plus de documents

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
    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": sources,
        "meta": {"confidence": "normal", "retrieved": len(docs), "model": LLM_MODEL}
    }

@app.get("/")
def root():
    return {"ok": True, "message": "Project Copilot API (RAG local via Ollama) prête."}
