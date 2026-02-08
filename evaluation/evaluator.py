import json


class SimpleEvaluator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline

    def run(self, eval_file: str):
        with open(eval_file, "r") as f:
            cases = json.load(f)

        for case in cases:
            result = self.rag.answer(case["question"])

            print("\nQUESTION:", case["question"])
            print("ANSWER:", result["answer"])
            print("SOURCES:", result["sources"])
            print("EXPECTED:", case["expected_behavior"])
