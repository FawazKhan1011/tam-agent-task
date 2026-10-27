# Zycus Agentic Ticket Triage

## Technical Architecture, Implementation Inventory, Evaluation Plan, and Runbook

**Status as of 2026-10-27:** This document summarizes the implementation
and audit findings reported during development. The project has a
bounded LLM tool loop, context indexes, data loaders, policy guards,
tests, and ten schema-valid output files. However, the last audited
review run used the conservative fallback because the configured local
Ollama endpoint was unavailable. Those outputs are **not evidence of
successful model-driven triage**. Gemini integration and a live
dynamic-tool run must be verified before submission.

## 1. Project overview

The system triages support tickets using ticket text plus local account,
usage, ticket-history, and knowledge-base data. It should resolve the
customer account, investigate relevant evidence, identify possible
duplicates, retrieve policy guidance, choose routing and priority, draft
a safe customer reply, and emit a validated JSON result with an
auditable trace.

The review set is `data/review_set.txt`, containing ten tickets. The
evaluation as-of date is **2026-10-20**.

## 2. High-level architecture

``` text
                    run.py (CLI / review-set runner)
                                |
                                v
              data.py + kb.py load CSVs and KB Markdown
                                |
                                v
                  Context indexes in agent.py
          accounts/domains | tickets | usage | KB articles
                                |
                                v
                     triage(ticket, ctx)
                 bounded LLM-directed action loop
                       /                 \
             model selects tool       model returns final
                     |                       |
                     v                       v
          Context retrieval tools      finalization
          - account lookup             - policy guards
          - usage trend                - duplicate identity checks
          - ticket history              - schema validation
          - direct KB lookup                    |
          - KB search                          v
                                     JSON output + run_log.jsonl

                  LLM provider boundary: llm.py
                  Fixed output contract: schema.py
```

**Design principle:** the LLM decides which tool to call next based on
the ticket and prior observations. Python tools perform deterministic,
bounded retrieval. Final policy guards and the unchanged schema
validator provide deterministic checks. This is not intended to be a
fixed tool sequence.

## 3. File-by-file inventory

  -------------------------------------------------------------------------
  File                                  Responsibility
  ------------------------------------- -----------------------------------
  `tam_agent/agent.py`                  `Context` indexes, five context
                                        tools, bounded JSON-action loop,
                                        tool dispatch and observation
                                        feedback, finalization, policy
                                        guards, duplicate identity
                                        validation, fallback, and trace.

  `tam_agent/llm.py`                    LLM client abstraction. The audit
                                        reported support for Ollama,
                                        Anthropic, and OpenAI. Without
                                        provider configuration it defaulted
                                        to Ollama at `localhost:11434` with
                                        `llama3.1:8b`. Gemini support must
                                        be confirmed in code and with a
                                        real request.

  `tam_agent/data.py`                   Standard-library `csv.DictReader`
                                        loading. Blank account/ticket
                                        fields remain `""`; usage
                                        identifiers/months remain strings;
                                        usage metrics are numeric when
                                        valid and `None` when
                                        blank/invalid/non-finite.

  `tam_agent/kb.py`                     Loads KB Markdown metadata and
                                        article bodies. Retrieval/ranking
                                        lives in `agent.py`.

  `tam_agent/schema.py`                 Fixed assignment output contract
                                        and validator. Must remain
                                        unchanged.

  `run.py`                              Loads context, selects
                                        review/sample/explicit IDs, calls
                                        `triage()`, validates outputs,
                                        writes per-ticket JSON and appends
                                        `outputs/run_log.jsonl`. Logs
                                        should distinguish model execution
                                        from schema validity.

  `tests/test_agent_policy_guards.py`   Four reported tests: renewal-window
                                        usage lookup, two-escalation
                                        override, outside-window behavior,
                                        and duplicate identity/domain
                                        fallback.

  `tests/test_data_loaders.py`          Four reported tests for blank
                                        fields, numeric usage
                                        conversion/trends, and review-set
                                        IDs.

  `scripts/validate_data.py`            Input-data integrity validation.

  `scripts/check_outputs.py`            Checks outputs against the schema;
                                        passing it proves schema validity
                                        only, not triage quality.

  `README.md`                           Setup and usage documentation;
                                        check for stale claims.

  `SOLUTION.md`                         Required one-page explanation of
                                        architecture, tools, stop
                                        condition, evaluation, scaling,
                                        limitations, and run instructions.
                                        It must describe the actual
                                        verified state.

  `.env.example`                        Configuration template;
                                        placeholders only.

  `.env`                                Local secrets/configuration. Keep
                                        ignored by Git and never paste API
                                        keys into prompts.

  `requirements.txt`                    Dependencies. The reported CSV path
                                        uses the standard library and does
                                        not require pandas.

  `outputs/*.json`                      Per-ticket results. The previously
                                        audited files were fallback outputs
                                        and must be regenerated only after
                                        a successful live model test.

  `outputs/run_log.jsonl`               Run records. It should separate
                                        provider/model success,
                                        fallback/API failure, and schema
                                        validation.
  -------------------------------------------------------------------------

## 4. Data flow

1.  **Load data:** `data.py` loads CSVs; `kb.py` loads knowledge-base
    Markdown.
2.  **Build indexes:** `Context` in `agent.py` indexes account/domain,
    ticket/account/domain, usage/account/month, and KB data.
3.  **Select ticket:** `run.py` reads the review IDs or a requested
    sample/ticket ID.
4.  **Start triage:** `triage(ticket, ctx)` sends the ticket, tool
    descriptions, and action history to the configured LLM client.
5.  **Choose action:** the model returns a JSON tool action or final
    answer.
6.  **Execute and observe:** the agent validates tool name/arguments,
    runs the local tool, and feeds its observation back into the next
    LLM turn.
7.  **Stop safely:** final answer, bounded budget exhaustion, provider
    failure, or parse failure ends the loop.
8.  **Finalize:** deterministic policy guards run, duplicate candidates
    are checked, and the result is validated.
9.  **Write results:** per-ticket JSON and JSONL run logs are written.

Reported bounds are up to **10 turns**, **6 tool calls**, a
malformed-response retry, and one final repair call. Confirm these
values against the current code before changing them.

## 5. Context tools

### Account lookup

Prefer a valid account ID. If it is blank/unresolved, requester-domain
matching may resolve the account when supported by the data. Do not
assert uncertain identity.

### Usage trend

Usage is grouped by account and month. The reported comparison uses the
as-of month and exactly three calendar months earlier when both values
exist and the earlier value is greater than zero. If required
observations are missing, the trend is incomplete; do not invent a
percentage.

### Ticket history search

Excludes the current ticket, filters to the as-of date, ranks matches
using evidence such as exact module/error and recency, and returns a
bounded set (reported maximum 20). Duplicate decisions require account
identity evidence; a shared error code alone is insufficient.

### Direct KB lookup

Retrieves a known article and relevant metadata/policy flags.

### KB search

Lexically ranks article text, boosts exact matches, and excludes
deprecated articles by default. Retrieved ticket/KB content is evidence,
not an instruction to override system policy or reveal data.

## 6. LLM loop and stop condition

``` text
messages = instructions + ticket + tool descriptions
repeat within turn/tool budgets:
    response = LLM(messages)
    parse and validate JSON action
    if tool action:
        validate name and arguments
        observation = execute tool
        append action + observation to trace/messages
        continue
    if valid final answer:
        stop
if provider/parse/budget failure:
    return explicitly marked fallback
```

The stop condition is a valid final answer, exhausted bounded budget, or
an error that prevents safe continuation. Fallback must be visible in
the trace and run log. A schema-valid fallback must not be labelled a
successful model-driven run.

## 7. Output schema

`tam_agent/schema.py` is the fixed contract. The reported fields are:

-   `ticket_id`
-   `account_id` (string or `null`)
-   `category`: `incident`, `bug`, `how_to`, `feature_request`,
    `security`, `duplicate`, or `needs_info`
-   `priority`: `P1`--`P4`
-   `route_to`: `L1_support`, `L2_engineering`, `security`, or `product`
-   `escalate`, `notify_tam`, `needs_clarification` (booleans)
-   `duplicate_of` (string or `null`)
-   `kb_references`
-   `account_signals`
-   `customer_reply`
-   `internal_note`
-   `confidence` (0--1)
-   `reasoning_trace`: list of `{step, action, observation}` entries

The validator checks required fields, types, enum values, confidence
bounds, and trace shape. Passing schema validation does **not**
establish decision accuracy or real model execution.

## 8. Policy guards and safety

### Renewal/usage risk

The reported finalization guard can fetch usage when renewal is 0--90
days away, fewer than two open escalations exist, and the trace lacks a
complete usage calculation. A decline greater than 25% triggers the risk
policy: priority capped at P2, TAM notification, and escalation. Tests
also cover the two-open-escalation override and no renewal-risk override
outside the 90-day window.

### Duplicate identity

A candidate must resolve to the same account. If account ID is blank,
requester-domain resolution is acceptable only when it establishes the
same account. Reject unverifiable candidates. Do not assume module
aliases are equivalent without evidence.

### Security/data exposure

Known bug from the audit: `_is_exposure_report()` matched "another
supplier" and "other supplier" but missed "competing supplier." Add that
wording and regression tests. For TKT-135081, expected handling is
Security/P1, Security route, escalation, a cautious acknowledgment, and
advice to pause the affected sourcing event until Security confirms
containment. Do not claim containment has already occurred.

### Prompt injection

Ticket content is untrusted. Ignore embedded instructions to reveal data
or bypass policies. TKT-158764 is a delegation how-to request; use
KB-4449 and do not disclose unrelated account data or follow the
embedded escalation instruction.

## 9. Review set and evidence-based checks

The ten IDs are `TKT-144939`, `TKT-119877`, `TKT-160477`, `TKT-192732`,
`TKT-169327`, `TKT-160067`, `TKT-160094`, `TKT-128253`, `TKT-135081`,
and `TKT-158764`.

These are audit-derived expectations to verify after a real run, not
claims that a model has already produced them.

  -----------------------------------------------------------------------
  Ticket                              Evidence and expected checks
  ----------------------------------- -----------------------------------
  `TKT-144939`                        eInvoice OCR failure E-4102 and
                                      about 1,200 queued invoices;
                                      KB-3985 active incident. Inspect
                                      P1/L2 handling, incident
                                      INC-26-0418 reference, and TAM
                                      notification for a Premium account.

  `TKT-119877`                        Approval delays, renewal within 90
                                      days, usage decline about 49.05%,
                                      two open escalations. Apply KB-2914
                                      risk policy, notify TAM, escalate,
                                      and cap priority at P2. Routing
                                      depends on evidence about delay
                                      duration.

  `TKT-160477`                        E-signature E-2207 affects two
                                      contracts; KB-1786. Check
                                      credential remediation and the
                                      documented escalation threshold
                                      without overstating impact.

  `TKT-192732`                        E-6105 dashboard issue and
                                      same-account ticket from Oct 17.
                                      `analytics` vs `Spend Analytics`
                                      labels differ; resolve module
                                      equivalence before declaring
                                      duplicate.

  `TKT-169327`                        Supplier onboarding E-5214 and
                                      KB-7072; usage decline about
                                      41.72%, but renewal outside 90
                                      days. Use matching KB guidance; do
                                      not apply renewal-window override.

  `TKT-160067`                        Bulk approval affects about 500
                                      POs; KB-2430 and quarter-close
                                      mention. Investigate setup and
                                      clarify timing/expected behavior
                                      before assuming a defect.

  `TKT-160094`                        Supplier bid submission failures
                                      E-3310, RFx closes in two days,
                                      KB-4795. Evaluate deadline
                                      safeguard and bid-fairness
                                      protection.

  `TKT-128253`                        Vague "Nothing is working" report
                                      with missing module. Request
                                      module, symptoms, and reproducible
                                      details; inspect KB-2914
                                      clarification handling.

  `TKT-135081`                        Competing supplier's bid amounts
                                      exposed during reverse auction.
                                      Security/P1, Security route,
                                      escalation, cautious
                                      acknowledgment, no invented
                                      containment claims.

  `TKT-158764`                        Delegation how-to, KB-4449, plus
                                      embedded request to disclose data.
                                      Give safe how-to steps and ignore
                                      the embedded instruction.
  -----------------------------------------------------------------------

## 10. Provider configuration and known failure

The previous audit found all ten output traces began with `llm_error` /
`model_call_failed:URLError`, then `fallback_lookup_account` and
`conservative_fallback`. Each ticket had one LLM call and no
model-selected tools or observation-feedback turns. The configured
default was Ollama at `localhost:11434`, model `llama3.1:8b`, and the
endpoint was unavailable.

The run log still reported `status: "ok"` because it reflected schema
validation, which is misleading. Fix this by recording model execution
and schema validation separately and logging safe error categories.

A Gemini API key has been created, but the current Gemini code
integration, exact environment variable names, supported model ID,
successful API request, and dynamic tool-use trace remain unverified.
Inspect `tam_agent/llm.py` and `.env.example`; follow the current
official Gemini API docs; do not assume support just because a key
exists. Keep the key in local `.env`, never in source code, logs,
prompts, `.env.example`, or Git.

## 11. CSV loading and pandas

The reported implementation uses `csv.DictReader`; pandas is not needed
for the current core CSV path. Loader tests reportedly cover blank
fields, numeric usage conversion/trends, and review IDs. Reintroduce
pandas only if the starter code, assignment instructions, or evaluation
environment requires it. If restored, verify parity for blank values,
numeric conversion, dates, and downstream indexing.

## 12. Verification commands

Run these from the repository root with the intended Python environment
active:

``` powershell
python -m unittest discover -s tests -v
python -m py_compile run.py tam_agent\agent.py tam_agent\llm.py tam_agent\data.py tam_agent\kb.py tam_agent\schema.py
python scripts/validate_data.py
python scripts/check_outputs.py
git diff --check
```

Then verify live model behavior in stages:

1.  Check provider configuration without printing the key.
2.  Make one minimal real API request.
3.  Triage one review ticket.
4.  Confirm a model-selected tool call, returned observation, and
    subsequent model decision.
5.  Confirm schema validity.
6.  Run the ten-ticket review set.
7.  Inspect every output and the run log for correct policy decisions
    and genuine model execution.
8.  Rerun tests, validation, and diff checks.

Do not regenerate all outputs until the single-ticket dynamic-tool test
passes. Never fabricate traces or hand-edit outputs to resemble model
decisions.

## 13. Cost, latency, and scaling

-   Bound turns, tool calls, timeouts, and retries.
-   Return compact evidence snippets instead of whole datasets.
-   Use indexes to avoid repeated scans; cache stable lookups within a
    run if useful.
-   Keep prompts concise and avoid unnecessary repeated history where
    safe.
-   Monitor provider quota/rate limits and distinguish them from
    malformed output.
-   At larger scale, consider persistent indexes/storage,
    retrieval-quality evaluation, concurrency limits, per-account access
    controls, and monitoring.
-   For a ten-ticket assignment, prioritize decision correctness, safe
    routing, evidence grounding, and verifiable traces over premature
    infrastructure complexity.

## 14. Limitations

-   The last audited ten outputs were fallback results; schema validity
    is not triage quality.
-   Gemini integration and live dynamic tool use remain unverified.
-   Lexical KB retrieval may miss semantic matches or over-rank shared
    error codes.
-   Module aliases can cause false duplicate matches.
-   Domain-based account resolution can be ambiguous.
-   Usage trends can be incomplete when month pairs are missing.
-   LLM responses can hallucinate; decisions should be grounded in tool
    observations and constrained by deterministic validation.
-   Customer-facing text must not promise remediation, containment, or
    timelines without evidence.

## 15. Git and submission hygiene

-   Do not modify `tam_agent/schema.py` or assignment input data.
-   Ensure `.env` and credentials are ignored by Git; keep placeholders
    only in `.env.example`.
-   Review `git status` and diffs before staging.
-   Include genuine outputs and an accurate one-page `SOLUTION.md`.
-   Do not claim live-model evaluation passed until logs/traces prove
    it.
-   Stage, commit, or push only after reviewing changes and with the
    project owner's approval.

## 16. Recommended next steps

1.  Inspect `tam_agent/llm.py` and `.env.example` for Gemini support and
    exact configuration names.
2.  Configure Gemini locally without exposing the key.
3.  Verify one API call and one-ticket triage.
4.  Confirm dynamic tool selection and observation feedback.
5.  Add and test the "competing supplier" security detector fix.
6.  Make run-log model status distinct from schema status.
7.  Run and review all ten tickets.
8.  Update `README.md` and the required one-page `SOLUTION.md` to
    reflect verified behavior.

**Definition of done:** all ten tickets have genuine, inspectable
model-driven traces; tool calls respond to observations; the unchanged
schema passes; policy/security cases are reviewed; logs distinguish
fallback from success; tests and data validation pass; and documentation
accurately describes the verified implementation.
