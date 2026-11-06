import json

from shorts_generator import highlights as hl
from shorts_generator.selection import bible as bb


def _transcript(duration=900.0):
    segs = [{"start": float(t), "end": t + 9.5, "text": f"Sentence at {t}."} for t in range(0, int(duration), 10)]
    return {"duration": duration, "segments": segs}


REDUCED = {"episode_summary": "s", "chapters": [{"start": 0, "end": 450, "title": "a", "one_line": "x"}],
           "setups_payoffs": [], "running_jokes": [], "extra": "dropped"}


def _fake_ask():
    seen = {"map": [], "reduce": []}

    def ask(prompt, schema, prompt_id, sample=0):
        if prompt_id.startswith("bible_map@"):
            seen["map"].append(prompt)
            n = len(seen["map"])
            return {"topics": [f"topic{n}"], "stories": [], "notable_lines": []}
        assert prompt_id.startswith("bible_reduce@")
        seen["reduce"].append(prompt)
        return REDUCED
    return ask, seen


def test_map_once_per_chunk_and_one_reduce_with_every_map_output():
    transcript = _transcript()
    ask, seen = _fake_ask()
    bible = bb.build_bible(transcript, ask)
    n_chunks = len(hl.chunk_transcript(transcript, size=300))
    assert len(seen["map"]) == n_chunks and len(seen["reduce"]) == 1
    for n in range(1, n_chunks + 1):
        assert f"topic{n}" in seen["reduce"][0]
    assert set(bible) == {"episode_summary", "chapters", "setups_payoffs", "running_jokes"}


def test_map_prompts_use_absolute_timestamps():
    ask, seen = _fake_ask()
    bb.build_bible(_transcript(), ask)
    assert "[240.0s]" in seen["map"][1]


def test_no_map_answers_gives_empty_bible():
    assert bb.build_bible(_transcript(), lambda *a, **k: None) == {}


def test_chunk_transcript_size_override_keeps_default():
    t = _transcript(3600.0)
    assert len(hl.chunk_transcript(t, size=300)) == len(range(0, 3600, 240))
    assert len(hl.chunk_transcript(t)) == len(hl.chunk_transcript(t, size=hl.chunk_size_seconds()))


def test_chapter_of():
    chapters = [{"start": 0, "end": 100}, {"start": 100, "end": 200}]
    assert bb.chapter_of(50, chapters) == 0
    assert bb.chapter_of(100, chapters) == 0
    assert bb.chapter_of(150, chapters) == 1
    assert bb.chapter_of(250, chapters) is None
    assert bb.chapter_of(5, []) is None


def test_reduce_prompt_is_json_of_map_outputs():
    ask, seen = _fake_ask()
    bb.build_bible(_transcript(600.0), ask)
    assert json.dumps({"topics": ["topic1"], "stories": [], "notable_lines": []}, ensure_ascii=False) in seen["reduce"][0]
