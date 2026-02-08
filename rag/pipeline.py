class RAGPipeline:
    """
    Orchestrates retrieval + prompt construction + generation.
    """

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client

    def _build_prompt(self, question: str, chunks):
        context_blocks = []

        for i, chunk in enumerate(chunks):
            block = f"[Source {i+1}]\n{chunk['text']}"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are an internal enterprise knowledge assistant.

Answer the question strictly using the information provided in the sources below.
If the answer is not present in the sources, say:
"The information is not available in the provided documents."

Sources:
{context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    def answer(self, question: str):
        retrieved_chunks = self.retriever.retrieve(question)

        prompt = self._build_prompt(question, retrieved_chunks)
        answer = self.llm.generate(prompt)

        sources = list(
            {chunk["metadata"]["source"] for chunk in retrieved_chunks}
        )

        return {
            "answer": answer,
            "sources": sources,
        }
