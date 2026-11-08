"""LLM call plumbing for the deep selector: a hash-keyed disk cache, prompt files, a parallel map.

Every call is keyed by sha256(model + prompt + schema + sample + run) and saved to
work/<episode>/calls/<hash>.json. A rerun reads saved answers instead of calling the
model, so a crash or a usage limit resumes where it stopped.
"""
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import config
from ..highlights import _parse_json_loose
from ..local.llm import model_id

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

LLMFn = Callable[..., str]
# ask(prompt, schema, prompt_id, sample=0) -> parsed answer, or None when unparseable
Ask = Callable[..., Optional[Dict]]
# An unparseable answer is re-asked in the same run: a gap that only fills on a later run would
# change the bible, and so every prompt built on it, which empties the cache for the rest of the episode.
ASK_ATTEMPTS = 3

# Live model calls across every thread (episodes, stages) at once; created on first use from LLM_WORKERS.
_SLOTS: Optional[threading.BoundedSemaphore] = None
_SLOTS_LOCK = threading.Lock()
# Live calls made by this process; LLM_MAX_CALLS caps them so a bad run cannot drain a usage plan.
_live_calls = 0
_BUDGET_LOCK = threading.Lock()


def _spend_one() -> None:
    global _live_calls
    with _BUDGET_LOCK:
        if config.LLM_MAX_CALLS and _live_calls >= config.LLM_MAX_CALLS:
            raise RuntimeError(f"LLM_MAX_CALLS={config.LLM_MAX_CALLS} reached; "
                               "rerun with a higher cap to resume from the cache")
        _live_calls += 1


def _slots() -> threading.BoundedSemaphore:
    global _SLOTS
    with _SLOTS_LOCK:
        if _SLOTS is None:
            _SLOTS = threading.BoundedSemaphore(max(1, config.LLM_WORKERS))
        return _SLOTS


def _cached(llm_fn: LLMFn, calls_dir: Path, prompt: str, schema: Optional[Dict], prompt_id: str,
            sample: int, run: int, stats: Optional[Dict]) -> Tuple[str, Path]:
    model = model_id()
    key = hashlib.sha256(
        (model + prompt + json.dumps(schema, sort_keys=True) + f"{sample}:{run}").encode()
    ).hexdigest()
    path = Path(calls_dir) / f"{key}.json"
    entry = None
    if path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entry = None
    if entry is None:
        _spend_one()
        with _slots():
            t0 = time.time()
            response = llm_fn(prompt, schema=schema)  # an exception propagates and writes nothing
        entry = {"key": key, "prompt_id": prompt_id, "model": model, "sample": sample, "run": run,
                 "seconds": round(time.time() - t0, 2), "prompt": prompt, "response": response}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # a crash mid-write never leaves a half-written answer
        if stats is not None:
            stats["misses"] = stats.get("misses", 0) + 1
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["seconds"] = stats.get("seconds", 0.0) + float(entry.get("seconds") or 0)
    return str(entry.get("response") or ""), path


def cached_llm(llm_fn: LLMFn, calls_dir: Path, prompt_id: str, run: int = 0,
               stats: Optional[Dict] = None) -> LLMFn:
    """A drop-in llm_fn(prompt, schema=None) -> str that goes through the cache.

    The text is returned raw for the caller to validate; only answers that parse as JSON are kept,
    so the caller's own retries reach the model instead of replaying a broken answer.
    """
    def fn(prompt: str, schema: Optional[Dict] = None) -> str:
        text, path = _cached(llm_fn, calls_dir, prompt, schema, prompt_id, 0, run, stats)
        try:
            _parse_json_loose(text)
        except ValueError:
            path.unlink(missing_ok=True)
        return text
    return fn


def make_ask(llm_fn: LLMFn, calls_dir: Path, run: int = 0, stats: Optional[Dict] = None) -> Ask:
    """ask(prompt, schema, prompt_id, sample=0) -> parsed JSON answer, or None.

    An answer that does not parse is dropped from the cache, so a rerun asks again.
    """
    def ask(prompt: str, schema: Dict, prompt_id: str, sample: int = 0) -> Optional[Dict]:
        for _ in range(ASK_ATTEMPTS):
            text, path = _cached(llm_fn, calls_dir, prompt, schema, prompt_id, sample, run, stats)
            try:
                parsed = _parse_json_loose(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            print(f"[deep] unparseable answer to {prompt_id}; not cached: {text[:120]!r}", flush=True)
            path.unlink(missing_ok=True)
        return None
    return ask


def pmap(fn: Callable, items: List, workers: Optional[int] = None) -> List:
    """map(fn, items) on a thread pool, results in input order."""
    with ThreadPoolExecutor(max_workers=max(1, workers or config.LLM_WORKERS)) as pool:
        return list(pool.map(fn, items))


def load_prompt(name: str, root: Path = PROMPTS_DIR) -> Tuple[str, str]:
    """(template, "name@<first 8 hex of its sha256>") — the id changes whenever the wording does."""
    text = (Path(root) / f"{name}.md").read_text(encoding="utf-8")
    return text, f"{name}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def judge_prefix(bible: Dict) -> str:
    """The identical opening of every judging prompt: system text, then the episode bible."""
    system, _ = load_prompt("judge_system")
    body = json.dumps(bible, ensure_ascii=False, separators=(",", ":")) if bible else "(none)"
    return f"{system.strip()}\n\nEpisode bible:\n{body}"
