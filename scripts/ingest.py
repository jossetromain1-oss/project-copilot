import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

DATA_DIR = "data/docs"
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") in ("1", "true", "True")
EMB_MODEL = os.getenv("EMBEDDINGS_MODEL") or ("nomic-embed-text" if USE_OLLAMA else "text-embedding-3-small")

def load_docs():
    docs = []
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Dossier {DATA_DIR} introuvable.")
        return docs
    files = os.listdir(DATA_DIR)
    print(f"📁 Fichiers trouvés : {files}")
    for file in files:
        path = os.path.join(DATA_DIR, file)
        if file.lower().endswith((".txt", ".md")):
            print(f"→ Chargement texte : {file}")
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif file.lower().endswith(".pdf"):
            print(f"→ Chargement PDF : {file}")
            docs.extend(PyPDFLoader(path).load())
        else:
            print(f"↷ Ignoré : {file}")
    print(f"📄 Documents chargés : {len(docs)}")
    return docs

def chunk_and_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    print(f"🧩 Chunks générés : {len(splits)}")
    if not splits:
        print("⚠️ Aucun chunk à indexer (fichiers vides ?).")
        return

    if USE_OLLAMA:
        print(f"🧠 Embeddings (Ollama) : {EMB_MODEL}")
        embeddings = OllamaEmbeddings(model=EMB_MODEL)
    else:
        print(f"🧠 Embeddings (OpenAI) : {EMB_MODEL}")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.endswith("XXXX"):
            print("❌ OPENAI_API_KEY manquante. Passe en USE_OLLAMA=1 ou ajoute une clé.")
            return
        embeddings = OpenAIEmbeddings(model=EMB_MODEL)

    db = Chroma.from_documents(splits, embeddings, persist_directory=VECTORSTORE_DIR)
    db.persist()
    print(f"✅ {len(splits)} chunks stockés dans {VECTORSTORE_DIR}")

if __name__ == "__main__":
    docs = load_docs()
    if docs:
        chunk_and_store(docs)
    else:
        print("⚠️ Aucun document trouvé.")
