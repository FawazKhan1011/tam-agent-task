import json
import unittest

from tam_agent.agent import Context, _is_exposure_report, _normalize_final_answer, triage
from tam_agent.schema import validate


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return json.dumps(next(self.responses))


def _account(account_id="A1", domain="a.example", renewal="2026-12-01", escalations="0"):
    return {
        "account_id": account_id,
        "domain": domain,
        "renewal_date": renewal,
        "open_escalations": escalations,
        "tier": "Premium",
    }


def _ticket(**overrides):
    ticket = {
        "ticket_id": "TKT-000001",
        "account_id": "A1",
        "requester_domain": "a.example",
        "created_at": "2026-10-20T12:00:00Z",
        "module": "CLM",
        "subject": "Unable to submit contract",
        "description": "Submission fails for one contract.",
        "error_code": "ERR-1",
        "status": "open",
        "customer_priority": "P4",
    }
    ticket.update(overrides)
    return ticket


def _kb():
    return [{
        "id": "KB-2914",
        "meta": {"title": "Support priority and escalation policy", "type": "policy",
                 "module": "All", "status": "current", "error_codes": []},
        "body": "Renewal risk policy evidence.",
    }]


def _final_answer(**overrides):
    answer = {
        "ticket_id": "TKT-000001",
        "account_id": None,
        "category": "bug",
        "priority": "P3",
        "route_to": "L1_support",
        "escalate": False,
        "notify_tam": False,
        "duplicate_of": None,
        "needs_clarification": False,
        "kb_references": [],
        "account_signals": [],
        "customer_reply": "Thank you; support is reviewing the issue.",
        "internal_note": "Investigate the reported failure.",
        "confidence": 0.8,
    }
    answer.update(overrides)
    return {"type": "final", "answer": answer}


def _context(account, usage=None, tickets=None):
    return Context([account], tickets or [], usage or [], _kb())


class AccountRiskPolicyTests(unittest.TestCase):
    def test_finalization_fetches_usage_when_model_skips_it(self):
        ctx = _context(
            _account(),
            usage=[
                {"account_id": "A1", "month": "2026-07", "active_users": "100"},
                {"account_id": "A1", "month": "2026-10", "active_users": "70"},
            ],
        )
        ctx.llm = SequenceLLM([_final_answer()])

        result = triage(_ticket(), ctx)

        self.assertEqual(ctx.llm.calls, 1, "the deterministic check must not need another model turn")
        self.assertEqual(result["priority"], "P2")
        self.assertTrue(result["notify_tam"])
        self.assertTrue(result["escalate"])
        usage_steps = [step for step in result["reasoning_trace"] if step["action"] == "get_usage_trend"]
        self.assertEqual(len(usage_steps), 1)
        self.assertIn('"decline_percent": 30.0', usage_steps[0]["observation"])
        self.assertEqual(validate(result), [])

    def test_two_escalations_triggers_override_without_usage_lookup(self):
        ctx = _context(_account(escalations="2"))
        ctx.llm = SequenceLLM([_final_answer()])

        result = triage(_ticket(), ctx)

        self.assertEqual(result["priority"], "P2")
        self.assertTrue(result["notify_tam"])
        self.assertTrue(result["escalate"])
        self.assertFalse(any(step["action"] == "get_usage_trend" for step in result["reasoning_trace"]))
        self.assertIn("2 open escalations", result["account_signals"])
        self.assertEqual(validate(result), [])

    def test_renewal_outside_90_days_does_not_apply_override(self):
        ctx = _context(
            _account(renewal="2027-02-01"),
            usage=[
                {"account_id": "A1", "month": "2026-07", "active_users": "100"},
                {"account_id": "A1", "month": "2026-10", "active_users": "70"},
            ],
        )
        ctx.llm = SequenceLLM([
            {"type": "tool_call", "tool": "lookup_account", "arguments": {}},
            {"type": "tool_call", "tool": "get_usage_trend", "arguments": {}},
            _final_answer(),
        ])

        result = triage(_ticket(), ctx)

        self.assertEqual(result["priority"], "P3")
        self.assertFalse(result["notify_tam"])
        self.assertFalse(result["escalate"])
        self.assertEqual(validate(result), [])


class DuplicateIdentityTests(unittest.TestCase):
    def test_rejects_cross_account_candidate_and_allows_verified_domain_fallback(self):
        ctx = Context(
            [_account(), _account(account_id="A2", domain="b.example")], [], [], _kb())
        ticket = _ticket()
        base_candidate = {
            "ticket_id": "TKT-000099", "created_at": "2026-10-19T12:00:00Z",
            "status": "open", "module": "CLM", "error_code": "ERR-1",
            "subject": "Unable to submit contract",
        }
        answer = _final_answer(category="duplicate", duplicate_of="TKT-000099")

        cross_account = dict(base_candidate, account_id="A2", requester_domain="b.example")
        rejected = _normalize_final_answer(
            answer["answer"], ticket, "A1", [], ctx, {"TKT-000099": cross_account})
        self.assertIsNone(rejected["duplicate_of"])
        self.assertNotEqual(rejected["category"], "duplicate")

        blank_id_same_domain = dict(base_candidate, account_id="", requester_domain="A.EXAMPLE")
        accepted = _normalize_final_answer(
            answer["answer"], ticket, "A1", [], ctx, {"TKT-000099": blank_id_same_domain})
        self.assertEqual(accepted["duplicate_of"], "TKT-000099")
        self.assertEqual(accepted["category"], "duplicate")

        blank_id_unknown_domain = dict(base_candidate, account_id="", requester_domain="unknown.example")
        rejected_unknown = _normalize_final_answer(
            answer["answer"], ticket, "A1", [], ctx, {"TKT-000099": blank_id_unknown_domain})
        self.assertIsNone(rejected_unknown["duplicate_of"])
        self.assertNotEqual(rejected_unknown["category"], "duplicate")


class SecurityExposureTests(unittest.TestCase):
    def test_competing_supplier_bid_exposure_forces_security_response(self):
        ticket = _ticket(
            subject="FYI - odd thing in auction",
            description=("A supplier could see the bid amounts of a competing supplier "
                         "during our reverse auction."),
        )
        ctx = _context(_account())
        ctx.kb_by_id["KB-6401"] = {
            "id": "KB-6401",
            "meta": {"title": "Tenant data exposure", "type": "policy",
                     "module": "All", "status": "current", "error_codes": []},
            "body": "Security handling policy.",
        }

        self.assertTrue(_is_exposure_report(ticket))
        result = _normalize_final_answer(
            _final_answer(category="how_to", priority="P4", route_to="L1_support")["answer"],
            ticket, "A1", [], ctx, {},
        )

        self.assertEqual(result["category"], "security")
        self.assertEqual(result["priority"], "P1")
        self.assertEqual(result["route_to"], "security")
        self.assertTrue(result["escalate"])
        self.assertIn("Security team has been engaged", result["customer_reply"])
        self.assertNotIn("cause", result["customer_reply"].casefold())
        self.assertNotIn("scope", result["customer_reply"].casefold())
        self.assertNotIn("ACC-", result["customer_reply"])
        self.assertEqual(validate(result), [])

    def test_ordinary_supplier_question_is_not_exposure(self):
        ticket = _ticket(
            subject="Supplier asks about their own bid",
            description="Our supplier wants to see its own submitted bid status.",
        )
        self.assertFalse(_is_exposure_report(ticket))


if __name__ == "__main__":
    unittest.main()
