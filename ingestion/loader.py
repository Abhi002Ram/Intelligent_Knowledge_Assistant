from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


class DocumentLoader:
    """
    Loads raw documents from disk and converts them into plain text.

    Each document is returned with minimal metadata so downstream
    components can trace answers back to the source.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def load(self) -> List[Dict]:
        """
        Walks through the base_path and loads all supported documents.

        Returns:
            List of dictionaries with keys:
            - text: extracted text
            - source: file name
        """
        documents = []

        for file_path in self.base_path.rglob("*"):
            if file_path.suffix.lower() == ".pdf":
                text = self._load_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = self._load_text(file_path)
            else:
                continue

            documents.append(
                {
                    "text": text,
                    "source": str(file_path),
                }
            )

        return documents

    def _load_pdf(self, path: Path) -> str:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")
