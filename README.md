# TAM Intern Task: Agentic Ticket Triage

**Expected effort:** 6–8 hours
**Deadline:** set by the hiring team in your invitation email

## Context
Our support and Technical Account Management (TAM) teams handle thousands of procurement-software tickets a year. A good support engineer does not triage a ticket from its text alone. They check who the customer is, look at account health and ticket history, find the right knowledge-base (KB) article, apply support policy, and then decide.

Build an agent that does this for one ticket at a time.

## What you build
Implement `triage(ticket, ctx)` in `tam_agent/agent.py`. For each ticket, the agent must:

1. **Resolve the account.** `account_id` is sometimes blank.
2. **Gather the account context that matters.** Tier, renewal date, usage trend, open escalations, related tickets.
3. **Retrieve the relevant KB articles,** including support policy.
4. **Decide.** Category, priority, routing, escalation, TAM notification, duplicate status, and whether clarification is needed.
5. **Draft** a customer reply and an internal note.
6. **Record every step** in `reasoning_trace`.

The output format is fixed. See `tam_agent/schema.py`.

## Must-have: the agent decides what to do next
Expose your lookups as tools. The LLM chooses which tool to call, reads the result, decides whether it needs more information, and decides when to stop.

A fixed sequence of LLM calls that runs the same steps for every ticket does **not** meet this bar. Some tickets need one lookup. Others need three or four, including following one article to another.

## Scale
The data is production-sized:

| Data | Volume |
|---|---|
| Tickets | ~3,000 (12 months) |
| Accounts | 200 |
| Usage metrics | 12 months per account |
| KB articles | ~390 (~100k tokens) |

- Design for the full dataset. Do not paste the whole KB or ticket table into a prompt.
- You only need to **run** the 10 tickets in `data/review_set.txt`. Running a larger sample is optional.
- The data is realistic: blank fields, inconsistent labels, similar company names, outdated articles. Foreign keys are valid. See `data/README.md`.
- Review tickets reflect real situations. Some are not what they first appear.

## What is provided
| File | Purpose |
|---|---|
| `tam_agent/data.py`, `tam_agent/kb.py` | Loaders only. No joins or search. |
| `tam_agent/llm.py` | Minimal client for Ollama (free, local), Anthropic or OpenAI |
| `tam_agent/schema.py` | Required output format and validator |
| `tam_agent/agent.py` | **Your code goes here.** Add any modules you need. |
| `run.py` | Runs the agent and writes `outputs/` |
| `scripts/check_outputs.py` | Confirms your review-set outputs are valid |
| `scripts/validate_data.py` | Confirms dataset integrity |

You can use any framework (LangGraph, LlamaIndex, etc.) or none. You can replace any provided file except `schema.py`.

## Write SOLUTION.md (1 page max)
1. **Design.** How the agent works, which tools it has, how it decides to stop. A diagram is optional.
2. **Evaluation.** 5–8 bullets: how you would measure triage quality before putting this in production.
3. **Scale.** How does your approach behave at 10x the data (30,000 tickets, 4,000 articles)? Cover cost, latency and retrieval quality. What would you change?
4. **Limitations.** What does not work yet.
5. **How to run.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # choose provider and model
python scripts/validate_data.py
python run.py --set review        # after implementing agent.py
python scripts/check_outputs.py
```

## Submission
- Push to a GitHub repository shared with the hiring team.
- Commit `outputs/` for the 10 review tickets, plus `outputs/run_log.jsonl`.
- `python scripts/check_outputs.py` must show **10/10 valid**. Submissions that fail this check are not reviewed.
- Commit as you work. Do not squash into a single commit.
- Never commit API keys or `.env`.

## Rules
- Any LLM provider is fine, including a free local model. We evaluate design and reasoning, not model size.
- AI coding assistants are allowed. You must be able to explain every part of your code.

## How you will be evaluated
| Area | Weight |
|---|---|
| Working build | 30% |
| Agentic depth | 30% |
| Retrieval and decision accuracy (review set) | 20% |
| Written reasoning (SOLUTION.md) | 20% |

Shortlisted candidates are invited to a 15-minute code walkthrough.

## Local setup (Windows PowerShell)

Run these commands from the repository root. The project currently requires pandas; provider SDKs and agent frameworks are optional and are not installed by the starter requirements.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally to select a provider and model. Add a provider credential only if that provider requires one. Never put real credentials in `.env.example`, source files, commits, or shared logs; `.env` is ignored by Git. The example is configured for Ollama, which must already be installed and running with the selected model. This setup does not install Ollama or download a model. If PowerShell blocks activation, use a process-scoped execution-policy bypass for that activation only; do not change the machine-wide policy.

## Validation and run commands

Validate the supplied data without calling an LLM:

```powershell
python scripts/validate_data.py
```

After `tam_agent/agent.py` is implemented and the configured provider is available, run the required review tickets and validate their outputs:

```powershell
python run.py --set review
python scripts/check_outputs.py
```

The starter's `triage()` function is currently unimplemented. Therefore the run and output-check commands are documented targets, not verified passing checks. Do not treat generated outputs as present unless the commands have been run and checked.

## Repository map

- `data/`: supplied accounts, tickets, usage metrics, review set, and data dictionary.
- `kb/`: supplied knowledge-base articles.
- `tam_agent/`: loaders, LLM client, required output schema, and the agent implementation location.
- `scripts/`: dataset integrity and output validation scripts.
- `outputs/`: destination for per-ticket JSON results and the run log; generated results are not part of this setup phase.
- `run.py`: assignment runner.
- `requirements.txt`: declared Python dependencies.
- `.env.example`: non-secret provider configuration template.

## Development and Git safety

Implement and verify one cohesive change at a time. Review the exact diff before staging, stage only explicitly approved file paths (never stage the whole workspace blindly), run relevant checks, and make a separate descriptive commit for each milestone. Keep generated outputs, local environments, credentials, and `.env` out of commits unless the assignment specifically requires the generated review outputs; never commit secrets. Preserve existing branches, remotes, and history, and do not publish or push without approval.

The supplied dataset and `tam_agent/schema.py` are assignment inputs and must remain unchanged. The data validator can be used during setup; agent behavior, review-set results, and evaluation outcomes remain unverified until the implementation phase.
