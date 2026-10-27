# TAM Agent: Agentic Ticket Triage

## 1. Design
The agent implements `triage(ticket, ctx)` with an LLM-driven tool loop. The model chooses a tool, receives its observation, and decides whether to gather more context or return a final triage decision. Available tools resolve customer accounts, retrieve usage trends, inspect ticket history and open escalations, look up KB articles by ID, and search the knowledge base. The loop is bounded by iteration and tool-call limits; it stops when the model returns a final decision or a limit/error prevents further progress. Results are validated against the fixed output schema, with deterministic policy guards for important account-risk and security cases.

## 2. Evaluation
Before production, evaluate:
- Category, priority, and routing accuracy against expert-labelled tickets.
- Account-resolution accuracy and cross-account mismatch rate.
- Duplicate-ticket precision and recall.
- KB relevance and correctness of referenced articles.
- Escalation and TAM-notification accuracy against support policy.
- Security handling, including supplier-data exposure and prompt-injection attempts.
- Schema-valid output rate and successful tool-call rate.
- Latency, cost per ticket, and human reviewer correction rate.

## 3. Scale
At 10x volume (30,000 tickets and 4,000 articles), targeted retrieval avoids sending the full dataset to the model. Measure retrieval quality and latency as data grows. To scale further, add indexed structured lookups and hybrid or vector search for KB retrieval, cache stable account context and frequent KB results, and monitor token usage, API limits, and cost. These are proposed improvements where not already implemented; retrieval indexes, caching, and vector search should not be assumed to exist today.

## 4. Limitations
The existing review outputs were produced through fallback handling after Gemini request failures. Schema validity alone does not demonstrate genuine model-driven triage. A successful live Gemini smoke test and inspection of traces showing model-selected tool calls and returned observations are required before regenerating and accepting the review outputs. Model decisions can be incorrect, so policy checks, labelled evaluation, and human review remain necessary.

## 5. How to Run
From the repository root in Windows PowerShell, configure `LLM_PROVIDER`, `LLM_MODEL`, and `GEMINI_API_KEY` in a local `.env` file. Keep `.env` out of version control and use only a placeholder in `.env.example`.

```powershell
python -m pip install -r requirements.txt
python scripts/validate_data.py
python run.py --set review
python scripts/check_outputs.py
python -m unittest discover -s tests -v
```

Only treat the outputs as complete after confirming that the run log records genuine model-driven tool calls rather than fallback results.
