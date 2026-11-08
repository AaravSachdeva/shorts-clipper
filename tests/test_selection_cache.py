import json

import pytest

from shorts_generator.selection import cache

SCHEMA = {"type": "object"}


class _LLM:
    def __init__(self, answer='{"ok": true}'):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt, schema=None):
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def _model(monkeypatch):
    monkeypatch.setattr(cache, "model_id", lambda: "fake:model")


def test_same_call_hits_disk_once(tmp_path):
    llm = _LLM()
    ask = cache.make_ask(llm, tmp_path)
    assert ask("p", SCHEMA, "hook@1") == {"ok": True}
    assert ask("p", SCHEMA, "hook@1") == {"ok": True}
    assert len(llm.prompts) == 1
    [f] = list(tmp_path.glob("*.json"))
    saved = json.loads(f.read_text())
    assert saved["prompt"] == "p" and saved["prompt_id"] == "hook@1" and saved["model"] == "fake:model"


def test_sample_and_run_change_key(tmp_path):
    llm = _LLM()
    cache.make_ask(llm, tmp_path)("p", SCHEMA, "x", 0)
    cache.make_ask(llm, tmp_path)("p", SCHEMA, "x", 1)
    cache.make_ask(llm, tmp_path, run=1)("p", SCHEMA, "x", 0)
    assert len(llm.prompts) == 3


def test_schema_changes_key(tmp_path):
    llm = _LLM()
    ask = cache.make_ask(llm, tmp_path)
    ask("p", {"type": "object"}, "x")
    ask("p", {"type": "object", "required": ["a"]}, "x")
    assert len(llm.prompts) == 2


def test_failed_call_writes_nothing_and_retries(tmp_path):
    llm = _LLM(RuntimeError("claude -p: usage limit reached"))
    ask = cache.make_ask(llm, tmp_path)
    with pytest.raises(RuntimeError, match="usage limit"):
        ask("p", SCHEMA, "x")
    assert list(tmp_path.iterdir()) == []
    llm.answer = '{"ok": 1}'
    assert ask("p", SCHEMA, "x") == {"ok": 1}


def test_corrupt_file_is_a_miss(tmp_path):
    llm = _LLM()
    ask = cache.make_ask(llm, tmp_path)
    ask("p", SCHEMA, "x")
    [f] = list(tmp_path.glob("*.json"))
    f.write_text("{not json")
    assert ask("p", SCHEMA, "x") == {"ok": True}
    assert len(llm.prompts) == 2
    assert json.loads(f.read_text())["response"] == '{"ok": true}'


def test_unparseable_answer_is_not_kept(tmp_path):
    llm = _LLM("sorry, no JSON here")
    ask = cache.make_ask(llm, tmp_path)
    assert ask("p", SCHEMA, "x") is None
    assert list(tmp_path.glob("*.json")) == []
    assert ask("p", SCHEMA, "x") is None
    assert len(llm.prompts) == 2 * cache.ASK_ATTEMPTS


def test_unparseable_answer_is_retried_in_run(tmp_path):
    answers = iter(["not json", '{"ok": 2}'])
    prompts = []

    def llm(prompt, schema=None):
        prompts.append(prompt)
        return next(answers)

    ask = cache.make_ask(llm, tmp_path)
    assert ask("p", SCHEMA, "x") == {"ok": 2}
    assert ask("p", SCHEMA, "x") == {"ok": 2} and len(prompts) == 2


def test_stats_count_hits_and_misses(tmp_path):
    stats = {}
    ask = cache.make_ask(_LLM(), tmp_path, stats=stats)
    ask("a", SCHEMA, "x")
    ask("a", SCHEMA, "x")
    ask("b", SCHEMA, "x")
    assert stats["calls"] == 3 and stats["misses"] == 2 and stats["seconds"] >= 0


def test_cached_llm_returns_raw_text(tmp_path):
    llm = _LLM('```json\n{"x": "raw"}\n```')
    fn = cache.cached_llm(llm, tmp_path, "classic")
    assert fn("p", schema=SCHEMA) == '```json\n{"x": "raw"}\n```'
    assert fn("p", schema=SCHEMA) == '```json\n{"x": "raw"}\n```'
    assert len(llm.prompts) == 1


def test_pmap_keeps_order():
    assert cache.pmap(lambda x: x * 2, [3, 1, 2], workers=3) == [6, 2, 4]


def test_prompt_version_changes_with_text(tmp_path):
    (tmp_path / "hook.md").write_text("Question {x}")
    text, pid = cache.load_prompt("hook", root=tmp_path)
    assert text == "Question {x}" and pid.startswith("hook@") and len(pid) == len("hook@") + 8
    (tmp_path / "hook.md").write_text("Question {x}?")
    assert cache.load_prompt("hook", root=tmp_path)[1] != pid


def test_judge_prefix_includes_bible_or_none():
    assert cache.judge_prefix({}).endswith("Episode bible:\n(none)")
    assert cache.judge_prefix({"episode_summary": "s"}).endswith('Episode bible:\n{"episode_summary":"s"}')


def test_parallel_identical_asks_do_not_collide(tmp_path):
    import time as _time

    def slow(prompt, schema=None):
        _time.sleep(0.05)
        return '{"ok": true}'

    ask = cache.make_ask(slow, tmp_path)
    out = cache.pmap(lambda _: ask("same", SCHEMA, "x"), range(8), workers=8)
    assert out == [{"ok": True}] * 8
    assert len(list(tmp_path.glob("*.json"))) == 1 and not list(tmp_path.glob("*.tmp"))


def test_cached_llm_does_not_keep_non_json_answers(tmp_path):
    answers = iter(["", '{"ok": 1}'])
    prompts = []

    def llm(prompt, schema=None):
        prompts.append(prompt)
        return next(answers)

    fn = cache.cached_llm(llm, tmp_path, "classic")
    assert fn("p", schema=SCHEMA) == ""
    assert list(tmp_path.glob("*.json")) == []
    assert fn("p", schema=SCHEMA) == '{"ok": 1}' and len(prompts) == 2
    assert fn("p", schema=SCHEMA) == '{"ok": 1}' and len(prompts) == 2
