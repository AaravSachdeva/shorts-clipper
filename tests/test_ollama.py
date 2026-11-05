import json

import pytest
import requests

from shorts_generator.local import llm


class _Resp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data or {}
        self.text = text or json.dumps(self._data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm.config, "OLLAMA_HOST", "http://ollama.test:11434")
    monkeypatch.setattr(llm.config, "OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(llm.config, "OLLAMA_NUM_CTX", 16384)
    monkeypatch.setattr(llm, "_ollama_checked", False)


def _tags(*names):
    return _Resp(data={"models": [{"name": n} for n in names]})


def test_request_sends_schema_num_ctx_and_native_endpoint(monkeypatch):
    calls = {}
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("qwen3:8b"))

    def post(url, json=None, timeout=None):
        calls["url"], calls["body"] = url, json
        return _Resp(data={"message": {"content": '{"ok": true}'}, "prompt_eval_count": 120, "eval_count": 9})

    monkeypatch.setattr(requests, "post", post)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    out = llm.call_local_llm("hello", schema)
    assert out == '{"ok": true}'
    assert calls["url"] == "http://ollama.test:11434/api/chat"
    assert calls["body"]["format"] == schema
    assert calls["body"]["options"]["num_ctx"] == 16384
    assert calls["body"]["stream"] is False


def test_latest_tag_matches_bare_name(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("qwen3:8b:latest", "llama3.2:latest"))
    llm.check_ollama(model="llama3.2")  # no raise


def test_missing_model_says_how_to_pull(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("llama3.2:latest"))
    with pytest.raises(RuntimeError, match=r"ollama pull qwen3:8b"):
        llm.call_local_llm("hi")


def test_server_down_message(monkeypatch):
    def boom(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RuntimeError, match=r"ollama serve"):
        llm.call_local_llm("hi")


def test_truncation_warning(monkeypatch, capsys):
    monkeypatch.setattr(llm.config, "OLLAMA_NUM_CTX", 4096)
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("qwen3:8b"))
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, timeout=None: _Resp(data={"message": {"content": "{}"}, "prompt_eval_count": 4096}),
    )
    llm.call_local_llm("x" * 30000)
    err = capsys.readouterr().out
    assert "may not fit" in err and "probably truncated" in err


def test_http_error_surfaces(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("qwen3:8b"))
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(status=500, text="boom"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        llm.call_local_llm("hi")


@pytest.mark.parametrize("num_gpu", [None, 3])
def test_num_gpu_sent_only_when_set(monkeypatch, num_gpu):
    monkeypatch.setattr(llm.config, "OLLAMA_NUM_GPU", num_gpu)
    monkeypatch.setattr(requests, "get", lambda url, timeout: _tags("qwen3:8b"))
    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(json["options"])
        return _Resp(data={"message": {"content": "{}"}, "prompt_eval_count": 10, "eval_count": 1})

    monkeypatch.setattr(requests, "post", post)
    llm.call_local_llm("hi")
    assert sent.get("num_gpu") == num_gpu
