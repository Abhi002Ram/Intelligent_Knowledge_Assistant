# Model Usage

AquilaAI uses open-source language models exclusively for all inference tasks.

## Language Model Role
The language model is used only after relevant document chunks have been retrieved.
It does not operate independently of the retrieval layer.

## Constraints
- The language model does not have internet access.
- The model cannot access documents outside the indexed knowledge base.
- Model outputs are constrained by system prompts to rely only on retrieved content.

## Prompting Strategy
- Retrieved document chunks are injected into the prompt context.
- The model is instructed to cite sources when answering.
- If relevant information is missing, the model must state that explicitly.
