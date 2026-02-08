import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure project root is on Python path (needed for Streamlit)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import numpy as np

from configs.config_loader import load_config
from ingestion.loader import DocumentLoader
from ingestion.cleaner import TextCleaner
from chunking.recursive import RecursiveChunker
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FaissStore
from retrieval.retriever import Retriever
from llm.llm_client import LLMClient
from rag.pipeline import RAGPipeline


@st.cache_resource
def load_rag_pipeline():
    """
    Loads and initializes the full RAG pipeline.
    This function runs ONCE per Streamlit session.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config = load_config()

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    loader = DocumentLoader("data/raw")
    documents = loader.load()

    cleaner = TextCleaner()
    chunker = RecursiveChunker()

    chunks = []
    for doc in documents:
        clean_text = cleaner.clean(doc["text"])
        chunks.extend(
            chunker.chunk(
                clean_text,
                metadata={"source": doc["source"]},
            )
        )

    # ------------------------------------------------------------------
    # Embeddings + FAISS (persistent)
    # ------------------------------------------------------------------
    ARTIFACT_DIR = Path("artifacts")
    EMB_PATH = ARTIFACT_DIR / "embeddings" / "embeddings.npy"
    FAISS_PATH = ARTIFACT_DIR / "faiss" / "index.faiss"

    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "embeddings").mkdir(exist_ok=True)
    (ARTIFACT_DIR / "faiss").mkdir(exist_ok=True)

    embedder = Embedder(
        model_name=config["embeddings"]["model_name"]
    )

    if EMB_PATH.exists() and FAISS_PATH.exists():
        embeddings = np.load(EMB_PATH)
        store = FaissStore(dimension=embeddings.shape[1])
        store.load(str(FAISS_PATH))
    else:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        np.save(EMB_PATH, embeddings)

        store = FaissStore(dimension=embeddings.shape[1])
        store.add(embeddings)
        store.save(str(FAISS_PATH))

    # ------------------------------------------------------------------
    # Retriever (config-driven)
    # ------------------------------------------------------------------
    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        chunks=chunks,
        top_k=config["retrieval"]["top_k"],
    )

    # ------------------------------------------------------------------
    # LLM (config-driven)
    # ------------------------------------------------------------------
    llm_cfg = config["llm"]
    llm = LLMClient(
        model_path=llm_cfg["model_path"],
        n_ctx=llm_cfg["context_window"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
    )

    return RAGPipeline(retriever, llm)


def main():
    st.set_page_config(
        page_title="Intelligent Knowledge Assistant",
        layout="wide",
    )

    st.title("Intelligent Knowledge Assistant (RAG)🔎")
    st.write(
        "🤔Ask questions based on internal documents. "
        "Answers are generated strictly using retrieved evidence."
    )

    rag = load_rag_pipeline()

    question = st.text_input(
        "Enter your question:",
        placeholder="Does AquilaAI support fine-tuning?",
    )

    if question:
        with st.spinner("Retrieving and generating answer..."):
            result = rag.answer(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for src in result["sources"]:
            st.markdown(f"- `{src}`")


if __name__ == "__main__":
    main()
