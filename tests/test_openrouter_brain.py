"""brain.provider: openrouter — an OpenAI-compatible chat endpoint (OpenRouter
or a gateway in front of it), driven through httpx.MockTransport: no network."""

import logging
import os
import tempfile

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

import sellbird.brain as brain_mod
from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.state import StateManager

KEY = "test-key-not-real-0123456789"


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(brain_mod, "RETRY_BASE_DELAY", 0)


def _brain(state, handler, monkeypatch, key=KEY, **config):
    if key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    cfg = BrainConfig(provider="openrouter", model=config.pop("model", "free"),
                      base_url=config.pop("base_url", "https://gateway.example.com/ai/api/v1"))
    brain = Brain(state, cfg)
    brain.openrouter._transport = httpx.MockTransport(handler)
    return brain


def _reply(text="Hello there", status=200):
    def handler(request):
        return httpx.Response(status, json={
            "id": "gen-1", "model": "vendor/model:free",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0},
        })
    return handler


# ── config ──


def test_openrouter_config():
    cfg = BrainConfig(provider=" OpenRouter ", model="meta-llama/llama-3.3-70b-instruct:free",
                      base_url="https://openrouter.ai/api/v1/")
    assert (cfg.provider, cfg.base_url, cfg.api_key_env) == (
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    for bad in ({"base_url": "ftp://x"}, {"base_url": "not a url"}, {"api_key_env": "1BAD"},
                {"api_key_env": "A B"}):
        with pytest.raises(ValidationError):
            BrainConfig(provider="openrouter", model="free", **bad)


# ── calls ──


@pytest.mark.asyncio
async def test_successful_call_returns_text_and_records_usage(state, monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return _reply("Hi")(request)

    brain = _brain(state, handler, monkeypatch)
    assert await brain.think("Say hi", agent="writer", task="draft") == "Hi"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["url"] == "https://gateway.example.com/ai/api/v1/chat/completions"
    assert '"model": "free"' in seen["body"] or '"model":"free"' in seen["body"]

    assert await state.get_usage_today("openrouter") == 1
    assert await state.get_usage_today("claude") == 0
    rows = await state.usage_by_provider(days=1)
    assert [(r["provider"], r["input_tokens"], r["output_tokens"]) for r in rows] == [
        ("openrouter", 12, 3)]


@pytest.mark.asyncio
async def test_think_json_parses_the_reply(state, monkeypatch):
    brain = _brain(state, _reply('```json\n{"score": 7}\n```'), monkeypatch)
    assert await brain.think_json("Score it") == {"score": 7}


@pytest.mark.asyncio
async def test_refused_key_pauses_the_brain(state, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    brain = _brain(state, handler, monkeypatch)
    assert await brain.think("x") == ""
    assert "refused" in brain.paused_reason
    assert await brain.think("x") == ""  # paused: no second request
    assert len(calls) == 1

    # A new key un-pauses it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "another-test-key")
    brain.openrouter._transport = httpx.MockTransport(_reply("back"))
    assert await brain.think("x") == "back"
    assert brain.paused_reason == ""


@pytest.mark.asyncio
async def test_rate_limit_starts_a_usage_pause(state, monkeypatch):
    brain = _brain(state, _reply(status=429), monkeypatch)
    assert await brain.think("x") == ""
    assert "usage limit" in brain.paused_reason
    assert await brain.is_within_budget(max_daily_calls=100, max_percent=60) is False


@pytest.mark.asyncio
async def test_server_errors_are_retried(state, monkeypatch):
    responses = [httpx.Response(502), httpx.Response(503)]

    def handler(request):
        return responses.pop(0) if responses else _reply("ok")(request)

    brain = _brain(state, handler, monkeypatch)
    assert await brain.think("x", max_retries=2) == "ok"


@pytest.mark.asyncio
async def test_retries_are_bounded(state, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    brain = _brain(state, handler, monkeypatch)
    assert await brain.think("x", max_retries=1) == ""
    assert len(calls) == 2
    assert brain.paused_reason == ""  # transient: not a pause


@pytest.mark.asyncio
async def test_missing_key_is_a_setup_pause_without_a_request(state, monkeypatch):
    def handler(request):
        raise AssertionError("no request may be sent without a key")

    brain = _brain(state, handler, monkeypatch, key=None)
    assert await brain.think("x") == ""
    assert "OPENROUTER_API_KEY" in brain.paused_reason
    assert "OPENROUTER_API_KEY" in await brain.preflight()


@pytest.mark.asyncio
async def test_preflight_sends_no_prompt(state, monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": []})

    brain = _brain(state, handler, monkeypatch)
    assert await brain.preflight() == ""
    assert seen == [("GET", "/ai/api/v1/models")]


@pytest.mark.asyncio
async def test_budget_falls_back_to_the_call_counter(state, monkeypatch):
    brain = _brain(state, _reply(), monkeypatch)
    assert await brain.is_within_budget(max_daily_calls=1, max_percent=60) is True
    await brain.think("x")
    assert await brain.is_within_budget(max_daily_calls=1, max_percent=60) is False


@pytest.mark.asyncio
async def test_the_key_never_reaches_the_logs(state, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    brain = _brain(state, _reply(status=401), monkeypatch)
    await brain.think("x")
    brain.openrouter._transport = httpx.MockTransport(_reply(status=500))
    await brain.preflight()
    assert KEY not in caplog.text
    assert KEY not in brain.paused_reason
