"""Local LLM backends — OpenAI, Gemini, Ollama or Claude, selected by LLM_PROVIDER.

Every backend takes (prompt, schema=None). `schema` is a JSON Schema dict for
the expected response; providers that support constrained/structured output
use it, and the caller still validates the result either way.
"""
import json
import subprocess
import tempfile
import time
from typing import Dict, Optional

from .. import config
from ..config import (
    GEMINI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for LLM_PROVIDER=openai. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    from openai import RateLimitError  # type: ignore

    client = OpenAI(api_key=require_openai_key())
    kwargs = {}
    messages = [{"role": "user", "content": prompt}]
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": False},
        }
        # Routers and many hosted models ignore response_format; the schema in the prompt still holds.
        messages.insert(0, {"role": "system", "content":
                            "Respond with only a JSON object that matches this JSON Schema, and nothing else:\n"
                            + json.dumps(schema)})
    waited = 0.0
    while True:
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL, temperature=0.7, messages=messages, **kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            if waited >= config.RATE_LIMIT_WAIT_SECONDS:
                raise RuntimeError(f"openai: rate limited for {waited:.0f}s: {str(e)[:200]}") from e
            time.sleep(config.RATE_LIMIT_RETRY_SECONDS)
            waited += config.RATE_LIMIT_RETRY_SECONDS


def call_gemini_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    cfg = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
        "max_output_tokens": 8192,
    }
    if schema:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config={**cfg, "response_json_schema": schema}
            )
            return response.text or ""
        except Exception as e:  # older SDKs/models without JSON-schema support
            if "schema" not in str(e).lower():
                raise
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
    return response.text or ""


# --- Ollama -----------------------------------------------------------------

_ollama_checked = False


def _ollama_error(msg: str) -> RuntimeError:
    return RuntimeError(f"Ollama: {msg}")


def check_ollama(host: Optional[str] = None, model: Optional[str] = None) -> None:
    """Fail fast with a clear message if the server is down or the model isn't pulled."""
    import requests

    host = host or config.OLLAMA_HOST
    model = model or config.OLLAMA_MODEL
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise _ollama_error(
            f"no server reachable at {host} ({e.__class__.__name__}). "
            "Start it with `ollama serve` (or open the Ollama app), or set OLLAMA_HOST."
        ) from e
    names = {m.get("name", "") for m in r.json().get("models", [])}
    names |= {n[: -len(":latest")] for n in names if n.endswith(":latest")}
    if model not in names:
        have = ", ".join(sorted(names)) or "none"
        raise _ollama_error(
            f"model {model!r} is not pulled. Run `ollama pull {model}` "
            f"or set OLLAMA_MODEL to one you have (installed: {have})."
        )


def _estimate_tokens(text: str) -> int:
    # ~4 characters per token for English; good enough for a context warning.
    return len(text) // 4


def call_ollama_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Ollama backend via the native /api/chat endpoint.

    The native API (not the OpenAI-compatible one) is used because it is the
    only way to set num_ctx per request. Ollama's default context is small and
    it truncates longer prompts without raising an error, which would silently
    drop most of a transcript chunk.
    """
    import requests

    global _ollama_checked
    if not _ollama_checked:
        check_ollama()
        _ollama_checked = True

    num_ctx = config.OLLAMA_NUM_CTX
    est = _estimate_tokens(prompt)
    if est + 2048 > num_ctx:
        print(
            f"[llm/ollama] WARNING: prompt is ~{est} tokens; with room for the answer it may not fit "
            f"in num_ctx={num_ctx}. Raise OLLAMA_NUM_CTX or lower HIGHLIGHT_CHUNK_SECONDS.",
            flush=True,
        )

    body = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema if schema else "json",
        "options": {"temperature": 0.2, "num_ctx": num_ctx},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "think": config.OLLAMA_THINK,
    }
    if config.OLLAMA_NUM_GPU is not None:
        body["options"]["num_gpu"] = config.OLLAMA_NUM_GPU
    try:
        r = requests.post(
            f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=config.OLLAMA_TIMEOUT_SECONDS
        )
    except requests.RequestException as e:
        raise _ollama_error(f"request failed: {e}") from e
    if r.status_code != 200:
        raise _ollama_error(f"HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    evaluated = data.get("prompt_eval_count")
    # When the prompt fills the whole window, Ollama has cut tokens off the front.
    if isinstance(evaluated, int) and evaluated >= num_ctx - 8:
        print(
            f"[llm/ollama] WARNING: prompt filled the context window ({evaluated}/{num_ctx} tokens) "
            "and was probably truncated. Raise OLLAMA_NUM_CTX or lower HIGHLIGHT_CHUNK_SECONDS.",
            flush=True,
        )
    seconds = (data.get("total_duration") or 0) / 1e9
    print(
        f"[llm/ollama] {config.OLLAMA_MODEL}: {evaluated} prompt + {data.get('eval_count')} output tokens "
        f"in {seconds:.0f}s",
        flush=True,
    )
    return (data.get("message") or {}).get("content", "") or ""


def release_ollama() -> None:
    """Unload the model so its RAM is free for rendering (keep_alive: 0)."""
    import requests

    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": config.OLLAMA_MODEL, "keep_alive": 0},
            timeout=30,
        )
    except requests.RequestException:
        pass


# --- Claude (headless Claude Code) -------------------------------------------

# An empty working directory, so no project CLAUDE.md or settings are found.
_claude_cwd: Optional[str] = None


def _claude_run(prompt: str, schema: Optional[Dict] = None) -> Dict:
    """Run one isolated `claude -p` call and return the CLI's parsed JSON result.

    The model sees only `prompt`: no default system prompt, no tools, no user
    settings (so no hooks or plugins), no MCP servers, no skills or slash commands.
    """
    global _claude_cwd
    if _claude_cwd is None:
        _claude_cwd = tempfile.mkdtemp(prefix="claude-llm-")
    args = [
        "claude", "-p",
        "--model", config.CLAUDE_MODEL,
        "--output-format", "json",
        "--system-prompt", "",
        "--tools", "",
        "--setting-sources", "project",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--effort", config.CLAUDE_EFFORT,
    ]
    if schema:
        args += ["--json-schema", json.dumps(schema)]
    t0 = time.time()
    try:
        proc = subprocess.run(
            args, input=prompt, cwd=_claude_cwd, capture_output=True, text=True,
            timeout=config.CLAUDE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"claude -p: {e}") from e
    try:
        data = json.loads(proc.stdout or "")
    except ValueError:
        data = {}
    if proc.returncode != 0 or not isinstance(data, dict) or data.get("is_error"):
        detail = str(data.get("result") or "") if isinstance(data, dict) else ""
        raise RuntimeError(f"claude -p: {(detail or proc.stderr or proc.stdout or 'no output')[:300]}")
    usage = data.get("usage") or {}
    tokens_in = sum(int(usage.get(k) or 0) for k in
                    ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    print(
        f"[llm/claude] {config.CLAUDE_MODEL}: {tokens_in} input + {usage.get('output_tokens')} output tokens "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    return data


def call_claude_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Claude backend (LLM_PROVIDER=claude): a stand-in for the local model during development."""
    data = _claude_run(prompt, schema)
    if data.get("structured_output") is not None:
        return json.dumps(data["structured_output"])
    return str(data.get("result") or "")


# --- dispatch ---------------------------------------------------------------

def model_id() -> str:
    """Provider and model behind call_local_llm, e.g. "claude:claude-haiku-5-5"."""
    provider = (config.LLM_PROVIDER or "openai").strip().lower()
    model = {
        "openai": config.OPENAI_MODEL,
        "gemini": GEMINI_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "claude": config.CLAUDE_MODEL,
    }.get(provider, "")
    return f"{provider}:{model}"


def call_local_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (config.LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt, schema)
    if provider == "gemini":
        return call_gemini_llm(prompt, schema)
    if provider == "ollama":
        return call_ollama_llm(prompt, schema)
    if provider == "claude":
        return call_claude_llm(prompt, schema)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', 'ollama', or 'claude'."
    )


def release_local_llm() -> None:
    """Free provider resources after the last LLM call (Ollama only)."""
    if (config.LLM_PROVIDER or "").strip().lower() == "ollama":
        release_ollama()
