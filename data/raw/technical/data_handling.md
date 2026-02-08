# Data Handling and Storage

AquilaAI is designed with data isolation and minimal retention in mind.

## Document Processing
- Uploaded documents are parsed into text-only format.
- No raw documents are exposed directly to the language model.
- All document chunks are stored with metadata for traceability.

## Storage
- Vector embeddings are stored in an isolated vector database.
- Metadata includes document name and section identifiers.
- Temporary processing files are deleted after indexing.

## Retention
- Indexed data follows the retention rules defined in the data privacy policy.
- Data is not reused across different customer environments.
