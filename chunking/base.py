from abc import ABC, abstractmethod
from typing import List, Dict


class BaseChunker(ABC):
    """
    Abstract base class for text chunking strategies.
    """

    @abstractmethod
    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Splits text into chunks.

        Returns:
            List of dictionaries with:
            - text
            - metadata
        """
        pass
