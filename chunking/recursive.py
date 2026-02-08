from typing import List, Dict
from chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Splits text into overlapping chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

            start = end - self.overlap

        return chunks
