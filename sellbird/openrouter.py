"""OpenRouter brain: any OpenAI-compatible chat endpoint (``brain.provider: openrouter``).

Talks to ``{base_url}/chat/completions`` with a Bearer key: OpenRouter itself
(``https://openrouter.ai/api/v1``, free ``:free`` models included) or a gateway
in front of it that speaks the same API. One user message per call, no tools,
no streaming.

The key is read from the environment variable named by ``brain.api_key_env``
(default ``OPENROUTER_API_KEY``, usually set in ``.env``) at call time and is
never logged, stored or shown. Error messages never contain prompts, replies
or credentials.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
PREFLIGHT_TIMEOUT_SECONDS = 20.0


class OpenRouterError(Exception):
    """A call failed. ``kind`` drives retry and the operator message:
    ``setup`` / ``auth`` repeat until the operator acts, ``usage_limit``
    pauses for a while, anything ``retryable`` is tried again."""

    def __init__(self, message: str, kind: str = "protocol", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class OpenRouterResult:
    text: str
    model: str = ""
    duration_ms: int = 0
    request_id: str = ""
    usage: dict = field(default_factory=dict)


class OpenRouterClient:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = DEFAULT_API_KEY_ENV, transport=None):
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env or DEFAULT_API_KEY_ENV
        self._transport = transport  # tests inject httpx.MockTransport

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise OpenRouterError(
                f"No API key: set {self.api_key_env} in .env for brain.provider openrouter.",
                kind="setup")
        if not self.model:
            raise OpenRouterError(
                "brain.model is required for brain.provider openrouter "
                "(e.g. a ':free' model id, or a gateway alias such as 'free').",
                kind="setup")
        return key

    def fingerprint(self) -> str:
        """Changes when the key, endpoint or model changes, so a paused brain
        re-checks as soon as the operator fixes its settings."""
        raw = "|".join([os.environ.get(self.api_key_env, ""), self.base_url, self.model])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _client(self, key: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {key}",
                "X-Title": "Sellbird",
            },
            timeout=timeout,
            transport=self._transport,
        )

    async def preflight(self):
        """Key and model present and the endpoint answers ``GET /models``.
        No prompt is sent."""
        key = self._key()
        try:
            async with self._client(key, PREFLIGHT_TIMEOUT_SECONDS) as client:
                response = await client.get("/models")
        except httpx.HTTPError as e:
            raise OpenRouterError(
                f"{self.base_url} could not be reached ({type(e).__name__}).",
                kind="network", retryable=True) from None
        _raise_for_status(response)

    async def run(self, prompt: str, timeout: float) -> OpenRouterResult:
        key = self._key()
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "usage": {"include": True},
        }
        started = time.monotonic()
        try:
            async with self._client(key, timeout) as client:
                response = await client.post("/chat/completions", json=body)
        except httpx.TimeoutException:
            raise OpenRouterError(f"No reply within {timeout:.0f}s.",
                                  kind="timeout", retryable=True) from None
        except httpx.HTTPError as e:
            raise OpenRouterError(
                f"{self.base_url} could not be reached ({type(e).__name__}).",
                kind="network", retryable=True) from None
        _raise_for_status(response)

        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise OpenRouterError("The reply was not a chat completion.",
                                  kind="protocol", retryable=True) from None
        if isinstance(text, list):  # content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        if not str(text).strip():
            raise OpenRouterError("The model returned an empty reply.",
                                  kind="empty", retryable=True)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = usage.get("prompt_tokens_details") if isinstance(
            usage.get("prompt_tokens_details"), dict) else {}
        return OpenRouterResult(
            text=str(text),
            model=str(data.get("model") or self.model),
            duration_ms=int((time.monotonic() - started) * 1000),
            request_id=str(data.get("id") or ""),
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "cache_read_tokens": int(details.get("cached_tokens") or 0),
                "cost_usd": float(usage.get("cost") or 0.0),
            },
        )


def _raise_for_status(response: httpx.Response):
    status = response.status_code
    if status < 400:
        return
    if status in (401, 403):
        raise OpenRouterError(
            f"The API key was refused (HTTP {status}). Check the key in .env.",
            kind="auth")
    if status in (402, 429):
        raise OpenRouterError(
            f"Rate or credit limit reached (HTTP {status}).", kind="usage_limit")
    if status == 404:
        raise OpenRouterError(
            "The endpoint or model was not found (HTTP 404). Check brain.base_url "
            "and brain.model.", kind="setup")
    if status in (400, 422):
        raise OpenRouterError(
            f"The request was rejected (HTTP {status}); check brain.model.",
            kind="request", retryable=False)
    raise OpenRouterError(f"Server error (HTTP {status}).", kind="server",
                          retryable=True)


class OpenRouterQuotaClient:
    """OpenRouter has no subscription windows to read, so the daily budget
    falls back to Sellbird's own call counter (``brain.is_within_budget``)."""

    async def get_utilization(self):
        return None
