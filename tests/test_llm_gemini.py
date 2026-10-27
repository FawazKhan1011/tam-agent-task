import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tam_agent.llm import LLMClient, LLMProviderError
import tam_agent.llm as llm_module


class GeminiClientTests(unittest.TestCase):
    def test_default_provider_is_documented_free_flash_model(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient()
        self.assertEqual(client.provider, "gemini")
        self.assertEqual(client.model, "gemini-3.8-flash")

    def setUp(self):
        self.env = patch.dict(os.environ, {
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-3.8-flash",
        })
        self.local_env = patch.dict(llm_module._LOCAL_ENV_VALUES, {
            "GEMINI_API_KEY": "unit-test-key-not-a-credential",
        })
        self.env.start()
        self.local_env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.local_env.stop)

    def test_generate_content_translates_chat_and_requests_json(self):
        response = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"type":"final"}'}]},
            }]
        }
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as open_url:
            client = LLMClient()
            result = client.chat([
                {"role": "user", "content": "ticket"},
                {"role": "assistant", "content": "prior tool call"},
            ], system="policy", max_tokens=1234)

        self.assertEqual(result, '{"type":"final"}')
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
        )
        self.assertEqual(request.get_header("X-goog-api-key"), "unit-test-key-not-a-credential")
        body = json.loads(request.data)
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["contents"][1]["role"], "model")
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "policy")
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 1234)

    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "shell-value-is-not-used"}), \
             patch.dict(llm_module._LOCAL_ENV_VALUES, {"GEMINI_API_KEY": ""}), \
             patch("urllib.request.urlopen") as open_url:
            with self.assertRaisesRegex(LLMProviderError, "GEMINI_API_KEY"):
                LLMClient().chat([{"role": "user", "content": "hello"}])
        open_url.assert_not_called()

    def test_quota_error_is_safe_and_does_not_retry(self):
        fake_body = io.BytesIO(b'{"error":{"message":"quota"}}')
        error = HTTPError("https://example.invalid", 429, "quota", {}, fake_body)
        with patch("urllib.request.urlopen", side_effect=error) as open_url:
            with self.assertRaisesRegex(LLMProviderError, "quota or rate limit") as raised:
                LLMClient().chat([{"role": "user", "content": "hello"}])
        self.assertNotIn("unit-test-key-not-a-credential", str(raised.exception))
        self.assertEqual(open_url.call_count, 1)

    def test_invalid_key_error_is_safe(self):
        body = json.dumps({"error": {"status": "API_KEY_INVALID", "message": "redacted"}}).encode()
        error = HTTPError("https://example.invalid", 400, "bad request", {}, io.BytesIO(body))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(LLMProviderError, "authentication rejected") as raised:
                LLMClient().chat([{"role": "user", "content": "hello"}])
        self.assertNotIn("unit-test-key-not-a-credential", str(raised.exception))

    def test_transient_server_error_retries_once(self):
        response = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}
        failure = HTTPError("https://example.invalid", 503, "unavailable", {}, io.BytesIO())
        with patch("urllib.request.urlopen", side_effect=[failure, io.BytesIO(json.dumps(response).encode())]) as open_url, \
             patch("tam_agent.llm.time.sleep") as sleep:
            result = LLMClient().chat([{"role": "user", "content": "hello"}])
        self.assertEqual(result, "{}")
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_blocked_or_incomplete_responses_raise_safe_errors(self):
        payloads = [
            [],
            {"promptFeedback": {"blockReason": "SAFETY"}},
            {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]},
            {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())
            ):
                with self.assertRaises(LLMProviderError):
                    LLMClient().chat([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
