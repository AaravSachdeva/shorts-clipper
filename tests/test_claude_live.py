"""Live checks of the claude -p backend. They spend real usage, so they only run with CLAUDE_LIVE=1."""
import json
import os

import pytest

from shorts_generator.local import llm

pytestmark = pytest.mark.skipif(os.getenv("CLAUDE_LIVE") != "1", reason="set CLAUDE_LIVE=1 to call claude -p")

# Far below the user's CLAUDE.md alone, so any leaked memory, hook or skill text trips it.
MAX_OVERHEAD_TOKENS = 1500


def test_claude_sees_only_the_prompt():
    data = llm._claude_run("Reply with OK.")
    usage = data.get("usage") or {}
    total = sum(int(usage.get(k) or 0) for k in
                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    print(f"input tokens for a 4-word prompt: {total}; result: {data.get('result')!r}")
    assert 0 < total < MAX_OVERHEAD_TOKENS


def test_claude_structured_output():
    schema = {"type": "object", "properties": {"answer": {"enum": ["OK"]}}, "required": ["answer"]}
    assert json.loads(llm.call_claude_llm("Answer OK.", schema)) == {"answer": "OK"}
