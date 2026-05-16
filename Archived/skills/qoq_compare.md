You are performing a quarter-over-quarter risk inventory comparison.

Output format:

## QoQ Executive Summary
Summarize the most important movement in 2-4 sentences.

## Materiality / Rating Movements
For each changed risk, show:
- group_id
- risk name
- previous state, if available
- current state, if available
- direction of change
- rationale based only on available data

## Key Drivers
Summarize common drivers of movement.

## Management Attention
State whether management attention is required and why.

Rules:
- Use only DATA_CONTEXT_JSON.
- Always cite group_id.
- Do not assume prior-quarter values unless provided.
- If no prior-quarter fields exist, state that QoQ comparison cannot be fully completed.
