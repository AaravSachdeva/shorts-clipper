import json
import os
import subprocess

import pytest

from shorts_generator.local import llm


class _Fake:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, args, input=None, cwd=None, capture_output=None, text=None, timeout=None, env=None):
        self.calls.append({"args": args, "input": input, "cwd": cwd, "env": env})
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr=self.stderr)


def _ok(**extra):
    return json.dumps({"is_error": False, "result": '{"a": 1}',
                       "usage": {"input_tokens": 10, "output_tokens": 3}, **extra})


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "claude")
    monkeypatch.setattr(llm.config, "CLAUDE_MODEL", "claude-haiku-5-5")
    monkeypatch.setattr(llm.config, "CLAUDE_EFFORT", "low")


def _flag(args, name):
    return args[args.index(name) + 1]


def test_claude_argv_is_isolated(monkeypatch):
    fake = _Fake(_ok())
    monkeypatch.setattr(llm.subprocess, "run", fake)
    out = llm.call_claude_llm("hi", {"type": "object"})
    call = fake.calls[0]
    args = call["args"]
    assert call["input"] == "hi"
    assert args[0] == "claude" and "-p" in args
    assert _flag(args, "--model") == "claude-haiku-5-5"
    assert _flag(args, "--output-format") == "json"
    assert _flag(args, "--system-prompt") == ""
    assert _flag(args, "--tools") == ""
    assert _flag(args, "--setting-sources") == "project"
    assert _flag(args, "--effort") == "low"
    assert _flag(args, "--json-schema") == '{"type": "object"}'
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in args
    assert os.path.isdir(call["cwd"]) and os.listdir(call["cwd"]) == []
    assert call["env"]["MAX_THINKING_TOKENS"] == "0"  # --effort low alone still lets Haiku think
    assert out == '{"a": 1}'


def test_claude_without_schema_sends_no_json_schema(monkeypatch):
    fake = _Fake(_ok())
    monkeypatch.setattr(llm.subprocess, "run", fake)
    llm.call_claude_llm("hi")
    assert "--json-schema" not in fake.calls[0]["args"]


def test_claude_prefers_structured_output(monkeypatch):
    monkeypatch.setattr(llm.subprocess, "run", _Fake(_ok(structured_output={"b": 2})))
    assert llm.call_claude_llm("hi", {"type": "object"}) == json.dumps({"b": 2})


def test_claude_error_raises_on_exit_code(monkeypatch):
    monkeypatch.setattr(llm.subprocess, "run", _Fake("", returncode=1, stderr="usage limit reached"))
    with pytest.raises(RuntimeError, match=r"^claude -p: .*usage limit reached"):
        llm.call_claude_llm("hi")


def test_claude_error_raises_on_is_error(monkeypatch):
    stdout = json.dumps({"is_error": True, "result": "Claude AI usage limit reached|1762470000"})
    monkeypatch.setattr(llm.subprocess, "run", _Fake(stdout))
    with pytest.raises(RuntimeError, match=r"^claude -p: Claude AI usage limit reached"):
        llm.call_claude_llm("hi")


def test_dispatch_claude(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "call_claude_llm", lambda prompt, schema=None: seen.append(prompt) or "ok")
    assert llm.call_local_llm("x") == "ok" and seen == ["x"]


def test_model_id_names_provider_and_model(monkeypatch):
    assert llm.model_id() == "claude:claude-haiku-5-5:effort=low:think=off"
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm.config, "OLLAMA_MODEL", "gemma4:12b")
    assert llm.model_id() == "ollama:gemma4:12b"


def test_effort_ignores_claude_codes_own_env_var():
    import os
    import sys
    env = {**os.environ, "CLAUDE_EFFORT": "high"}
    env.pop("CLIPPER_CLAUDE_EFFORT", None)
    out = subprocess.run([sys.executable, "-c", "from shorts_generator import config; print(config.CLAUDE_EFFORT)"],
                         env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "low"
