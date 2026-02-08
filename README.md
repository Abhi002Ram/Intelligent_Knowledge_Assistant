# Intelligent Knowledge Assistant using Retrieval-Augmented Generation (RAG)

## 📌 Overview
The Intelligent Knowledge Assistant is a fully local, open-source Retrieval-Augmented Generation (RAG) system that answers questions from domain-specific documents with strong factual grounding and explicit source citations.

The system combines semantic retrieval with local large language model inference to ensure that generated answers are based strictly on retrieved evidence, minimizing hallucinations and unsupported claims.


## 🧩 Models Used

### Embedding Model
- **Name**: `intfloat/e5-base-v2`
- **Purpose**: Semantic retrieval
- **Characteristics**:
  - Strong retrieval performance
  - Open-source
  - Efficient CPU inference

### Language Model (LLM)
- **Name**: Mistral 7B Instruct
- **Format**: GGUF
- **Quantization**: Q4_K_M
- **Runtime**: `llama.cpp` via `llama-cpp-python`
- **Characteristics**:
  - Instruction-tuned
  - Optimized for local inference
  - No external dependencies


## Tech Stack
- Programming Language: Python 3.10+
- Embeddings: Sentence Transformers
- Vector Database: FAISS
- LLM Runtime: llama.cpp (llama-cpp-python)
- Configuration: YAML
- UI: Streamlit
- Persistence: NumPy (.npy), FAISS index serialization


## 🏗️ Architecture
The system follows a modular, pipeline-based architecture:
- **Document ingestion and cleaning**  
  Loads raw domain documents and performs basic normalization.

- **Recursive text chunking**  
  Splits documents into semantically meaningful chunks suitable for retrieval.

- **Embeddings generation**  
  Uses `intfloat/e5-base-v2` to convert text chunks and queries into dense vectors.

- **Vector search (FAISS)**  
  Performs efficient similarity search to retrieve the top-K relevant chunks.

- **Local LLM inference**  
  Uses Mistral 7B Instruct (GGUF format) via `llama.cpp` for grounded answer generation.

- **Streamlit UI**  
  Provides an interactive interface for querying the knowledge base and viewing sources.


## 🎯 Application Flow

    User Question
    ↓
    Query Embedding (e5-base-v2)
    ↓
    FAISS Vector Search
    ↓
    Top-K Relevant Chunks
    ↓
    Grounded Prompt Construction
    ↓
    Local LLM Inference (Mistral 7B)
    ↓
    Answer + Source Citations



## 🎯 Design Decisions
- **No fine-tuning**  
  The system intentionally avoids fine-tuning to test and enforce hallucination control purely through retrieval grounding.

- **Strict separation of retrieval and generation**  
  Retrieval determines *what information is available*; generation only summarizes retrieved content.

- **Local inference**  
  All models run locally to ensure privacy, cost control, and independence from external APIs.

- **Configuration-driven system**  
  Model, embedding, and retrieval parameters are externalized in YAML configuration files.

- **Persistent embeddings and indexes**  
  Embeddings and FAISS indexes are stored on disk to avoid recomputation and improve startup performance.

---

## 📂 Project Structure

        intelligent-knowledge-assistant/
        │
        ├── data/
        │   └── raw/                  # Domain documents (product, policy, FAQ, technical)
        │
        ├── ingestion/                # Document loading & cleaning
        ├── chunking/                 # Recursive text chunking
        ├── embeddings/               # Embedding model wrapper
        ├── vectorstore/              # FAISS index logic
        ├── retrieval/                # Top-K retrieval logic
        ├── llm/                      # Local LLM wrapper (llama.cpp)
        ├── rag/                      # RAG orchestration
        │
        ├── configs/
        │   ├── model.yaml            # All runtime configuration
        │   └── config_loader.py
        │
        ├── evaluation/
        │   ├── eval_questions.json   # Evaluation question set
        │   └── evaluator.py
        │
        ├── ui/
        │   └── app.py                # Streamlit UI
        │
        ├── artifacts/                # Auto-generated (ignored in Git)
        │   ├── embeddings/
        │   │   └── embeddings.npy
        │   └── faiss/
        │       └── index.faiss
        │
        ├── main.py                   # CLI / non-UI entry point
        ├── requirements.txt
        ├── README.md


## 🚀 Running the System

- Install Dependencies
   > pip install -r requirements.txt

- **Run with Streamlit UI**
   >streamlit run ui/app.py

 - Interactive question answering
 - Displays grounded answers and source files
 - Pipeline is cached per session for performance

- **Run via CLI**
  >python main.py

 - Useful for testing, batch execution, or non-UI usage
 - Uses the same configuration and pipeline as the UI


## 💾 Artifacts and Persistence

 - The system persists heavy computations in the artifacts/ directory:

    artifacts/
    ├── embeddings/embeddings.npy
    └── faiss/index.faiss

 - Created automatically on first run
 - Reused on subsequent runs
 - Not committed to version control
 - Rebuilt only when source documents change


## 🧪 Evaluation

 - Evaluation is performed using predefined question sets located in:
    evaluation/eval_questions.json

 - The evaluation process focuses on:
    - Answer grounding
    - Hallucination avoidance
    - Correct refusal behavior
    - Source citation correctness

 - Evaluation is currently performed through controlled queries and manual inspection.
 - Evaluation is run explicitly via a separate script and is not executed as part of the UI or main runtime.


## 🛡️ Hallucination Control

The system enforces strict grounding by:
- Passing only retrieved chunks to the LLM
- Explicitly instructing the model to refuse when information is absent
- Using low-temperature generation
- Avoiding fine-tuning entirely

If evidence is missing, the system responds with:
> “The information is not available in the provided documents.”



## ⚠️ Limitations

 - Single-user local deployment

 - CPU-based inference only

 - Manual evaluation (no automated metrics)

 - No authentication or access control

 - Designed for local usage and experimentation


## 🧠 Key Concepts Demonstrated

 - Retrieval-Augmented Generation (RAG)

 - Dense vector embeddings

 - Semantic similarity search

 - Local large language model inference

 - Prompt grounding and citation

 - Configuration-driven ML systems

 - Persistent vector stores

 - Interactive ML applications with Streamlit






