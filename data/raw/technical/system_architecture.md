# System Architecture

AquilaAI follows a modular architecture designed for scalability, transparency, and fault isolation.

## High-Level Components
- Document Ingestion Service
- Text Chunking and Preprocessing Layer
- Embedding Generation Module
- Vector Store for Semantic Retrieval
- Language Model for Answer Generation
- Response Formatter with Source Attribution

## Architecture Flow
1. Documents are ingested and converted into clean text.
2. Text is split into overlapping semantic chunks.
3. Each chunk is converted into an embedding vector.
4. Embeddings are stored in a vector database.
5. User queries are embedded and matched against stored vectors.
6. Retrieved chunks are passed to the language model for answer generation.

## Design Principles
- Loose coupling between components
- Deterministic retrieval before generation
- No direct access from the language model to raw documents
