from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Converts text chunks into dense vector embeddings.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        """
        return self.model.encode(texts, show_progress_bar=True)
