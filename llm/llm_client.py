from llama_cpp import Llama
from typing import Any, Dict, cast


class LLMClient:
    """
    Thin wrapper over llama.cpp for local LLM inference.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        temperature: float,
        max_tokens: int,
    ):
        self.max_tokens = max_tokens

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            temperature=temperature,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the local LLM (non-streaming).
        """

        raw_output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            stop=["\n\n"],
        )

        output = cast(Dict[str, Any], raw_output)

        return output["choices"][0]["text"].strip()
