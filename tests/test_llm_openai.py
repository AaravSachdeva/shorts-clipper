import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from shorts_generator.local import llm


def _rate_limited():
    req = httpx.Request("POST", "http://router.test/v1/chat/completions")
    return openai.RateLimitError("All models exhausted", response=httpx.Response(429, request=req), body=None)


class _Client:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


@pytest.fixture
def client(monkeypatch):
    holder = {}
    monkeypatch.setattr(llm, "require_openai_key", lambda: "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: holder.setdefault("slept", []).append(s))

    def install(outcomes):
        c = _Client(outcomes)
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: c)
        holder["client"] = c
        return c
    holder["install"] = install
    return holder


def test_schema_also_goes_in_a_system_message(client):
    c = client["install"](['{"a": 1}'])
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert llm.call_openai_llm("hi", schema) == '{"a": 1}'
    msgs = c.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and json.dumps(schema) in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "hi"}
    assert c.calls[0]["response_format"]["type"] == "json_schema"


def test_no_schema_sends_only_the_prompt(client):
    c = client["install"](["text"])
    llm.call_openai_llm("hi")
    assert c.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_rate_limit_waits_and_retries(client):
    c = client["install"]([_rate_limited(), _rate_limited(), '{"a": 1}'])
    assert llm.call_openai_llm("hi", {"type": "object"}) == '{"a": 1}'
    assert len(c.calls) == 3 and len(client["slept"]) == 2


def test_rate_limit_gives_up_after_the_wait_budget(client, monkeypatch):
    monkeypatch.setattr(llm.config, "RATE_LIMIT_WAIT_SECONDS", 60)
    client["install"]([_rate_limited()] * 10)
    with pytest.raises(RuntimeError, match=r"^openai: rate limited"):
        llm.call_openai_llm("hi")
