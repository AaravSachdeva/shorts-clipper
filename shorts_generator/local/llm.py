"""Local LLM backends — OpenAI, Gemini, or Ollama, selected by LLM_PROVIDER.

Every backend takes (prompt, schema=None). `schema` is a JSON Schema dict for
the expected response; providers that support constrained/structured output
use it, and the caller still validates the result either way.
"""
from typing import Dict, Optional

from .. import config
from ..config import (
    GEMINI_MODEL,
    OPENAI_MODEL,
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

    client = OpenAI(api_key=require_openai_key())
    kwargs = {}
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": False},
        }
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content or ""


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
    }
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


# --- dispatch ---------------------------------------------------------------

def call_local_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (config.LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt, schema)
    if provider == "gemini":
        return call_gemini_llm(prompt, schema)
    if provider == "ollama":
        return call_ollama_llm(prompt, schema)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'ollama'."
    )


def release_local_llm() -> None:
    """Free provider resources after the last LLM call (Ollama only)."""
    if (config.LLM_PROVIDER or "").strip().lower() == "ollama":
        release_ollama()
