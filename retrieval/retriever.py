class Retriever:
    """
    Retrieves top-k relevant chunks using embeddings + FAISS.
    """

    def __init__(self, embedder, vector_store, chunks, top_k: int = 3):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks
        self.top_k = top_k

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed([query])
        _, indices = self.vector_store.search(query_embedding, self.top_k)

        return [self.chunks[i] for i in indices[0]]
