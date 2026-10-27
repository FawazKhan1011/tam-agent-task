import unittest

from run import _execution_log_metadata


class RunLoggingTests(unittest.TestCase):
    def test_api_fallback_is_not_reported_as_success(self):
        result = {"reasoning_trace": [
            {"step": 1, "action": "llm_error", "observation": "Gemini request timed out."},
            {"step": 2, "action": "conservative_fallback", "observation": "Used safe fallback."},
        ]}
        metadata = _execution_log_metadata(result, [])
        self.assertEqual(metadata["status"], "fallback")
        self.assertEqual(metadata["execution"], "fallback")
        self.assertTrue(metadata["schema_valid"])
        self.assertEqual(len(metadata["model_failures"]), 1)

    def test_model_decision_and_policy_overrides_are_distinguished(self):
        result = {"reasoning_trace": [
            {"step": 1, "action": "lookup_account", "observation": "resolved"},
            {"step": 2, "action": "model_final_decision", "observation": "accepted"},
            {"step": 3, "action": "policy_override", "observation": "KB-6401 applied"},
        ]}
        metadata = _execution_log_metadata(result, [])
        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["execution"], "model")
        self.assertEqual(metadata["policy_overrides"], 1)
        self.assertEqual(metadata["policy_override_events"], ["KB-6401 applied"])

    def test_schema_error_takes_precedence(self):
        result = {"reasoning_trace": [{"step": 1, "action": "model_final_decision", "observation": "accepted"}]}
        metadata = _execution_log_metadata(result, ["missing field"])
        self.assertEqual(metadata["status"], "schema_error")
        self.assertFalse(metadata["schema_valid"])


if __name__ == "__main__":
    unittest.main()
