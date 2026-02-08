import re


class TextCleaner:
    """
    Performs minimal, safe text normalization.
    """

    @staticmethod
    def clean(text: str) -> str:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive line breaks
        text = text.replace("\n", " ").strip()

        return text
