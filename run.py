"""Run the triage agent.

  python run.py --set review            # the 10 review tickets (required)
  python run.py --set sample --n 50     # random sample (optional)
  python run.py --ids TKT-123456 TKT-234567

Writes one JSON per ticket to outputs/ plus outputs/run_log.jsonl.
"""
import argparse, json, random, time
from pathlib import Path

from tam_agent.data import load_accounts, load_tickets, load_usage, load_review_set
from tam_agent.kb import load_kb
from tam_agent.agent import Context, triage
from tam_agent.schema import validate

OUT = Path(__file__).resolve().parent / "outputs"


def _execution_log_metadata(result: dict, errors: list[str]) -> dict:
    """Classify model success versus fallback separately from schema validity."""
    trace = result.get("reasoning_trace", [])
    actions = [step.get("action") for step in trace if isinstance(step, dict)]
    fallback_actions = {"llm_error", "conservative_fallback", "final_validation_failed",
                        "fallback_validation_failed"}
    is_fallback = any(action in fallback_actions for action in actions)
    if errors:
        status = "schema_error"
    elif is_fallback:
        status = "fallback"
    elif "model_final_decision" in actions:
        status = "ok"
    else:
        status = "no_model_decision"
    policy_events = [step.get("observation", "") for step in trace
                     if isinstance(step, dict) and step.get("action") == "policy_override"]
    model_failures = [step.get("observation", "") for step in trace
                      if isinstance(step, dict) and step.get("action") == "llm_error"]
    return {
        "status": status,
        "schema_valid": not errors,
        "execution": "fallback" if is_fallback else (
            "model" if "model_final_decision" in actions else "no_model_decision"),
        "policy_overrides": len(policy_events),
        "policy_override_events": policy_events,
        "model_failures": model_failures,
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["review", "sample"], default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    tickets = load_tickets()
    ctx = Context(load_accounts(), tickets, load_usage(), load_kb())
    if args.ids:
        ids = args.ids
    elif args.set == "sample":
        ids = random.Random(args.seed).sample([row["ticket_id"] for row in tickets], args.n)
    else:
        ids = load_review_set()

    OUT.mkdir(exist_ok=True)
    by_id = {row["ticket_id"]: row for row in tickets}
    ok = 0
    with open(OUT / "run_log.jsonl", "a") as log:
        for tid in ids:
            t0, calls0 = time.time(), ctx.llm.calls
            rec = {"ticket_id": tid}
            try:
                ticket = by_id[tid]
                result = triage(ticket, ctx)
                errs = validate(result)
                (OUT / f"{tid}.json").write_text(json.dumps(result, indent=2))
                rec |= _execution_log_metadata(result, errs)
                ok += not errs
            except Exception as e:
                # Provider exception text can be implementation-specific; keep
                # logs useful without serializing request data or credentials.
                from tam_agent.llm import LLMProviderError
                diagnostic = (str(e) if isinstance(e, LLMProviderError)
                          else f"Unhandled exception ({type(e).__name__}).")
                rec |= {"status": "exception", "schema_valid": False,
                    "execution": "exception", "errors": [diagnostic]}
            rec |= {"seconds": round(time.time() - t0, 2), "llm_calls": ctx.llm.calls - calls0}
            log.write(json.dumps(rec) + "\n")
            print(f"{tid}: {rec['status']} ({rec['seconds']}s, {rec['llm_calls']} LLM calls)")
    print(f"\n{ok}/{len(ids)} tickets produced valid output")


if __name__ == "__main__":
    main()
