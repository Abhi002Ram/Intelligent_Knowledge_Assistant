import faiss
import numpy as np
from pathlib import Path


class FaissStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings) # type: ignore

    def search(self, query_embedding: np.ndarray, top_k: int):
        return self.index.search(query_embedding, top_k) # type: ignore

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
