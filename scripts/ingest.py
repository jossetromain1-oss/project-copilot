# scripts/ingest.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
COLLECTION_NAME = "copilot"

def load_all_docs():
    docs = []

    # 1) .txt à la racine de data/ (facultatif)
    if os.path.isdir(DATA_DIR):
        loader_txt = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader, show_progress=True)
        docs += loader_txt.load()

    # 2) data/docs/ : .md, .pdf, .txt
    if os.path.isdir(DOCS_DIR):
        loader_md  = DirectoryLoader(DOCS_DIR, glob="**/*.md",  loader_cls=UnstructuredMarkdownLoader, show_progress=True)
        loader_pdf = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader,                 show_progress=True)
        loader_tx2 = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader,                  show_progress=True)
        docs += loader_md.load() + loader_pdf.load() + loader_tx2.load()

    return docs

def main():
    print(f"📁 DATA_DIR: {DATA_DIR}")
    print(f"📁 DOCS_DIR: {DOCS_DIR}")
    print(f"💾 VECTORSTORE_DIR: {VECTORSTORE_DIR}")
    print(f"🧠 EMBEDDINGS_MODEL (Ollama): {EMBED_MODEL}")

    # Embeddings 100% locaux via Ollama
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    raw_docs = load_all_docs()
    print(f"📄 Documents chargés : {len(raw_docs)}")

    if not raw_docs:
        print("⚠️ Aucun document trouvé. Ajoute des fichiers dans data/ ou data/docs/")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    print(f"🧩 Chunks générés : {len(chunks)}")

    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME
    )
    vs.persist()
    print(f"✅ Ingestion OK : {len(chunks)} chunks indexés dans '{VECTORSTORE_DIR}' (collection '{COLLECTION_NAME}').")

if __name__ == "__main__":
    main()