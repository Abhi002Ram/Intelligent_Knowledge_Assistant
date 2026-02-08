# API Constraints and Limitations

AquilaAI enforces constraints to ensure system stability and predictable behavior.

## Query Constraints
- Maximum query length is enforced.
- Queries outside the indexed domain may result in empty responses.

## Rate Limiting
- Rate limits depend on the pricing tier.
- Excessive requests may be throttled or rejected.

## Error Handling
- If retrieval returns no relevant documents, the system responds accordingly.
- The system does not fabricate answers when confidence is low.

## Unsupported Scenarios
- Real-time data querying
- External API calls during inference
- Autonomous decision-making without human oversight
