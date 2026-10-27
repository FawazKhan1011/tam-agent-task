"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`
    LLM_PROVIDER=gemini     needs GEMINI_API_KEY; uses Google's REST API

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import re
import socket
import time
import urllib.request
import urllib.error
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
_LOCAL_ENV_VALUES = {}
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip()
            if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in {"'", '"'}:
                _v = _v[1:-1]
            _LOCAL_ENV_VALUES[_k] = _v
            os.environ.setdefault(_k, _v)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini", "gemini": "gemini-3.8-flash",
        }.get(self.provider, ""))
        if not self.model:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
        self.calls = 0  # simple counter you can report for cost/latency

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text."""
        self.calls += 1
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                          system=system or "", messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                  temperature=temperature)
            return resp.choices[0].message.content or ""
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                               "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        if self.provider == "gemini":
            return self._chat_gemini(messages, system, max_tokens)
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _chat_gemini(self, messages: list[dict], system: str | None,
                     max_tokens: int) -> str:
        """Call Gemini GenerateContent using the local API key and JSON output mode."""
        # Per assignment configuration, Gemini credentials are read from the
        # ignored project .env file only; do not fall back to shell variables.
        api_key = _LOCAL_ENV_VALUES.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key in {"YOUR_GEMINI_API_KEY", "AQ.Ab8RN6JndD9ESP6jmx8pKvDstJWNvXlWosApOwKXe4CjHZvqOw"}:
            raise LLMProviderError(
                "Gemini API key is missing. Set GEMINI_API_KEY in the ignored project .env file."
            )
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            raise LLMProviderError("Gemini model ID contains unsupported characters.")

        contents = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant", "model"}:
                raise LLMProviderError("Gemini chat message has an unsupported role.")
            contents.append({
                "role": "model" if role in {"assistant", "model"} else "user",
                "parts": [{"text": str(message.get("content", ""))}],
            })

        body = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": max(1, min(int(max_tokens), 4096)),
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        # One retry is reserved for transient server failures only. Auth,
        # invalid-request, and quota errors fail immediately to avoid wasting
        # free-tier calls or concealing configuration problems.
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read())
                return _gemini_response_text(payload)
            except urllib.error.HTTPError as exc:
                if exc.code in {500, 502, 503, 504} and attempt == 0:
                    time.sleep(0.5)
                    continue
                raise _gemini_http_error(exc.code, _gemini_error_status(exc)) from None
            except (TimeoutError, socket.timeout):
                raise LLMProviderError("Gemini request timed out after 30 seconds.") from None
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    raise LLMProviderError("Gemini request timed out after 30 seconds.") from None
                raise LLMProviderError(
                    "Gemini could not be reached. Check the network connection and retry."
                ) from None
            except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                raise LLMProviderError(
                    "Gemini returned an empty or malformed GenerateContent response."
                ) from None
        raise LLMProviderError("Gemini request failed after one transient retry.")


class LLMProviderError(RuntimeError):
    """Safe provider diagnostic that never includes request headers or secrets."""


def _gemini_http_error(status: int, api_status: str = "") -> LLMProviderError:
    if status in {401, 403} or api_status in {"API_KEY_INVALID", "UNAUTHENTICATED", "PERMISSION_DENIED"}:
        message = "authentication rejected; verify GEMINI_API_KEY and Gemini API restrictions"
    elif status == 400 or api_status == "INVALID_ARGUMENT":
        message = "request rejected; check the model ID and GenerateContent request format"
    elif status == 404 or api_status == "NOT_FOUND":
        message = "model or endpoint not found; verify the Gemini model ID"
    elif status == 429 or api_status == "RESOURCE_EXHAUSTED":
        message = "quota or rate limit reached; check Google AI Studio usage and limits"
    elif status >= 500:
        message = "Gemini service error after a transient retry"
    else:
        message = "request rejected by Gemini"
    return LLMProviderError(f"Gemini {message} (HTTP {status}).")


def _gemini_error_status(error: urllib.error.HTTPError) -> str:
    """Extract only Google's short machine status; never surface response text."""
    try:
        payload = json.loads(error.read(8192))
        status = payload.get("error", {}).get("status", "")
    except (OSError, ValueError, TypeError, AttributeError):
        return ""
    return status if isinstance(status, str) and re.fullmatch(r"[A-Z_]{1,48}", status) else ""


def _gemini_response_text(payload: dict) -> str:
    """Extract candidate text and turn API safety/length stops into safe errors."""
    if not isinstance(payload, dict):
        raise LLMProviderError("Gemini returned a malformed GenerateContent response object.")
    prompt_feedback = payload.get("promptFeedback") or {}
    if not isinstance(prompt_feedback, dict):
        raise LLMProviderError("Gemini returned malformed prompt feedback.")
    if prompt_feedback.get("blockReason"):
        raise LLMProviderError("Gemini blocked the prompt; review policy and input content.")
    candidates = payload.get("candidates") or []
    if not candidates:
        raise LLMProviderError("Gemini returned no response candidate.")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise LLMProviderError("Gemini returned a malformed response candidate.")
    finish = candidate.get("finishReason")
    if finish == "MAX_TOKENS":
        raise LLMProviderError("Gemini response reached the output-token limit before completion.")
    if finish not in {None, "STOP"}:
        raise LLMProviderError(f"Gemini stopped generation ({_safe_reason_code(finish)}).")
    content = candidate.get("content") or {}
    if not isinstance(content, dict):
        raise LLMProviderError("Gemini returned malformed candidate content.")
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise LLMProviderError("Gemini returned malformed candidate parts.")
    answer_parts = [part for part in parts if isinstance(part, dict) and not part.get("thought")]
    if answer_parts:
        text = "".join(part.get("text", "") for part in answer_parts)
    else:
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise LLMProviderError("Gemini returned no text for the JSON action response.")
    return text


def _safe_reason_code(value) -> str:
    """Allow only short enum-like response codes in diagnostics."""
    value = str(value or "UNKNOWN")
    return value if re.fullmatch(r"[A-Z_]{1,40}", value) else "UNKNOWN"
