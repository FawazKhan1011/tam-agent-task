import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tam_agent import data
from tam_agent.agent import Context


class StandardLibraryCsvLoaderTests(unittest.TestCase):
    def test_account_and_ticket_loaders_preserve_blank_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "accounts.csv").write_text(
                "account_id,domain,nps_last\nA1,example.test,\n", encoding="utf-8"
            )
            (root / "tickets.csv").write_text(
                "ticket_id,account_id,requester_domain,module,error_code\n"
                "T1,,example.test,,\n",
                encoding="utf-8",
            )
            with patch.object(data, "DATA", root):
                accounts = data.load_accounts()
                tickets = data.load_tickets()

        self.assertEqual(accounts, [{"account_id": "A1", "domain": "example.test", "nps_last": ""}])
        self.assertEqual(tickets, [{
            "ticket_id": "T1", "account_id": "", "requester_domain": "example.test",
            "module": "", "error_code": "",
        }])

    def test_usage_loader_keeps_identifiers_and_converts_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "usage_metrics.csv").write_text(
                "account_id,month,active_users,documents_processed,login_rate_pct,tickets_opened\n"
                "A1,2026-07,100,2400,81.5,2\n"
                "A1,2026-10,70,,70,0\n",
                encoding="utf-8",
            )
            with patch.object(data, "DATA", root):
                usage = data.load_usage()

        self.assertEqual(usage[0], {
            "account_id": "A1", "month": "2026-07", "active_users": 100,
            "documents_processed": 2400, "login_rate_pct": 81.5, "tickets_opened": 2,
        })
        self.assertIsNone(usage[1]["documents_processed"])
        self.assertIsInstance(usage[1]["active_users"], int)
        self.assertIsInstance(usage[0]["login_rate_pct"], float)

    def test_numeric_loaded_usage_drives_three_month_trend(self):
        usage = [
            {"account_id": "A1", "month": "2025-11", "active_users": 100},
            {"account_id": "A1", "month": "2026-02", "active_users": 70},
        ]
        context = Context(
            [{"account_id": "A1", "domain": "example.test"}], [], usage, []
        )

        trend = context.get_usage_trend("A1")

        self.assertEqual(trend["decline_calculation"]["status"], "calculated")
        self.assertEqual(trend["decline_calculation"]["comparison_month"], "2025-11")
        self.assertEqual(trend["decline_calculation"]["decline_percent"], 30.0)

    def test_review_set_remains_exactly_ten_existing_ticket_ids(self):
        review_ids = data.load_review_set()
        tickets = data.load_tickets()
        ticket_ids = {row["ticket_id"] for row in tickets}

        self.assertEqual(len(review_ids), 10)
        self.assertEqual(len(set(review_ids)), 10)
        self.assertTrue(set(review_ids).issubset(ticket_ids))
        with (data.DATA / "review_set.txt").open(encoding="utf-8") as source:
            expected_ids = [line.strip() for line in source if line.strip()]
        self.assertEqual(review_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
