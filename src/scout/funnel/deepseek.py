"""DeepSeek client for funnel stages 4-5 (OpenAI-compatible chat completions).

Verified against DeepSeek's docs (not yet against a live key):

* POST {base}/chat/completions, Bearer auth. Models: `deepseek-flash`,
  `deepseek-v4-pro` (the older deepseek-chat / deepseek-reasoner names are gone).
* THINKING IS ON BY DEFAULT, and forcing a named tool_choice returns HTTP 400 in
  thinking mode. We always send thinking={"type": "disabled"}; without it every
  forced tool call would fail.
* Tool arguments come back as a JSON *string* that "does not always" parse, so we
  parse and validate defensively. We do not use `strict` (Beta).

Failure policy:
* 400/401/402/403/404/422 are FATAL (LLMFatalError): bad key, no balance, wrong
  model name, or a request the API rejects. Retrying cannot help, and swallowing
  them would silently queue every candidate forever. The runner disables the LLM
  and says so loudly.
* 429 / 5xx / network errors are retried with backoff, then raise LLMUnavailable,
  which the stages treat as "leave the candidate for later".
* A truncated (finish_reason=length) or malformed answer returns None.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .llm import LLMFatalError, LLMUnavailable

log = logging.getLogger(__name__)

BASE_URL = "https://api.deepseek.com"
FATAL_STATUSES = {400, 401, 402, 403, 404, 422}


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise LLMFatalError("DEEPSEEK_API_KEY is not set")
        self._max_retries = max(1, max_retries)
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def call_tool(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tool: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any] | None:
        """Force `tool` and return its parsed arguments, or None if unusable."""
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }],
            "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
            # Required: named tool_choice is rejected (400) while thinking is on.
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._post("/chat/completions", body)

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            log.warning("deepseek: response had no choices")
            return None
        if choice.get("finish_reason") == "length":
            log.warning("deepseek: answer truncated at max_tokens=%d", max_tokens)
            return None

        for call in (choice.get("message") or {}).get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != tool["name"]:
                continue
            try:
                args = json.loads(fn.get("arguments") or "")
            except (TypeError, ValueError):
                log.warning("deepseek: tool arguments were not valid JSON")
                return None
            return args if isinstance(args, dict) else None
        log.warning("deepseek: no matching tool call in response")
        return None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.post(path, json=body)
            except httpx.HTTPError as exc:
                last = exc
                log.warning("deepseek network error (attempt %d): %s", attempt, type(exc).__name__)
                self._backoff(attempt)
                continue

            if resp.status_code in FATAL_STATUSES:
                raise LLMFatalError(_fatal_message(resp))
            if resp.status_code == 429 or resp.status_code >= 500:
                last = LLMUnavailable(f"HTTP {resp.status_code}")
                log.warning("deepseek HTTP %d (attempt %d)", resp.status_code, attempt)
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            if resp.status_code >= 400:
                raise LLMFatalError(_fatal_message(resp))
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMUnavailable("deepseek returned non-JSON") from exc
        raise LLMUnavailable(f"deepseek unavailable after {self._max_retries} attempts: {last}")

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> None:
        delay = 2.0**attempt
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(min(delay, 60.0))

    def close(self) -> None:
        self._client.close()


_HINTS = {
    401: "authentication failed -- check DEEPSEEK_API_KEY",
    402: "insufficient balance -- top up your DeepSeek account",
    403: "access forbidden for this key",
    404: "model or endpoint not found -- check llm.*_model in config/filters.yaml "
         "(current names: deepseek-flash, deepseek-v4-pro)",
}


def _fatal_message(resp: httpx.Response) -> str:
    hint = _HINTS.get(resp.status_code, "request rejected by the API")
    detail = ""
    try:
        err = resp.json().get("error")
        detail = err.get("message", "") if isinstance(err, dict) else str(err or "")
    except (ValueError, AttributeError):
        detail = resp.text[:200]
    return f"DeepSeek HTTP {resp.status_code}: {hint}" + (f" ({detail[:200]})" if detail else "")
