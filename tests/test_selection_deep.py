import json
import re
import sys

import pytest

import main
from shorts_generator import config, pipeline
from shorts_generator import selection
from shorts_generator.selection import bible as bb
from shorts_generator.selection import candidates as cd
from shorts_generator.selection import eval as ev
from shorts_generator.selection import score as sc
from shorts_generator.selection import tournament as tn


def _transcript(n_sentences=150, per=8, step=0.5):
    words = []
    for s in range(n_sentences):
        for k in range(per):
            i = s * per + k
            words.append({"start": i * step, "end": i * step + 0.4, "word": f"s{s}w{k}" + ("." if k == per - 1 else "")})
    duration = n_sentences * per * step
    segs = [{"start": float(t), "end": t + 9.5, "text": " ".join(w["word"] for w in words if t <= w["start"] < t + 10),
             "words": [w for w in words if t <= w["start"] < t + 10]} for t in range(0, int(duration), 10)]
    return {"duration": duration, "segments": segs}


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, schema=None):
        self.calls += 1
        props = (schema or {}).get("properties", {})
        if schema is bb.MAP_SCHEMA:
            return json.dumps({"topics": ["t"], "stories": [], "notable_lines": []})
        if schema is bb.REDUCE_SCHEMA:
            return json.dumps({"episode_summary": "s", "setups_payoffs": [], "running_jokes": [],
                               "chapters": [{"start": 0, "end": 300, "title": "a", "one_line": "x"},
                                            {"start": 300, "end": 600, "title": "b", "one_line": "y"}]})
        if schema is cd.WINDOW_SCHEMA:
            w0 = float(re.search(r"Transcript:\n\[(\d+\.\d)s\]", prompt).group(1))
            return json.dumps({"candidates": [{"start": w0 + 10, "end": w0 + 45, "kind": "quotable", "why": "w"}]})
        if schema is tn.PAIR_SCHEMA:
            return json.dumps({"winner": "A"})
        if schema is tn.TEXT_SCHEMA:
            return json.dumps({"title": "T", "hook_sentence": "h", "on_screen_hook": "o", "virality_reason": "r"})
        if schema is sc.QUESTIONS["payoff"] or schema is sc.QUESTIONS["ending"]:
            return json.dumps({"answer": "yes"})
        if "label" in props:
            return json.dumps({"label": "funny", "score": 4})
        if "opening_line" in props:
            return json.dumps({"opening_line": "o", "score": 4})
        return json.dumps({"score": 3})


@pytest.fixture
def deep_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(selection, "loudness_per_second", lambda src, cache: [-30.0] * 600)
    monkeypatch.setattr(selection.cache, "model_id", lambda: "fake:model")
    return tmp_path


def test_run_deep_end_to_end(deep_env):
    llm = _FakeLLM()
    out = selection.run_deep(_transcript(), str(deep_env / "ep.mp4"), 3, llm_fn=llm)
    hs = out["highlights"]
    assert 1 <= len(hs) <= 3
    for h in hs:
        for key in ("title", "start_time", "end_time", "score", "hook_sentence", "on_screen_hook",
                    "virality_reason", "bt_strength", "combiner_score", "sub_scores", "origin", "kind"):
            assert key in h
        assert 0 <= h["score"] <= 100
    assert out["candidates"] and out["bible"]["episode_summary"] == "s"
    saved = json.loads((deep_env / "work" / "ep" / "deep.json").read_text())
    assert saved["picks"] == hs


def test_run_deep_resumes_from_cache(deep_env):
    llm = _FakeLLM()
    first = selection.run_deep(_transcript(), str(deep_env / "ep.mp4"), 3, llm_fn=llm)
    n = llm.calls
    second = selection.run_deep(_transcript(), str(deep_env / "ep.mp4"), 3, llm_fn=llm)
    assert n > 0 and llm.calls == n
    assert second["highlights"] == first["highlights"]


def test_run_deep_skips(deep_env):
    llm = _FakeLLM()
    out = selection.run_deep(_transcript(), str(deep_env / "ep.mp4"), 3, llm_fn=llm, skip=["bible", "tournament"])
    assert out["bible"] == {} and all(h["bt_strength"] is None for h in out["highlights"])


def test_pipeline_uses_deep_selector(tmp_path, monkeypatch):
    from shorts_generator import metadata
    from shorts_generator.local import clipper, downloader, llm, transcriber

    seen = {}
    monkeypatch.setattr(config, "SELECTOR", "deep")
    monkeypatch.setattr(downloader, "download_youtube_local", lambda url, fmt="720": "/v/ep.mp4")
    monkeypatch.setattr(downloader, "group_dir", lambda root, src: str(tmp_path))
    monkeypatch.setattr(transcriber, "transcribe_local",
                        lambda p, language=None: {"duration": 60.0, "segments": [{"start": 0, "end": 60, "text": "x"}]})

    def fake_deep(transcript, source_path, num_clips, llm_fn=None):
        seen["args"] = (source_path, num_clips)
        return {"highlights": [{"title": "t", "start_time": 0.0, "end_time": 20.0, "score": 90}]}

    def no_classic(*a, **k):
        raise AssertionError("classic selector must not run")

    monkeypatch.setattr(selection, "run_deep", fake_deep)
    monkeypatch.setattr(pipeline, "get_highlights", no_classic)
    monkeypatch.setattr(metadata, "generate_metadata", lambda t, top, fn: top)
    monkeypatch.setattr(metadata, "write_sidecars", lambda shorts: None)
    monkeypatch.setattr(clipper, "crop_highlights_local", lambda *a, **k: [])
    monkeypatch.setattr(llm, "release_local_llm", lambda: None)
    pipeline._run_local("u", 3, "9:16", "720", None)
    assert seen["args"] == ("/v/ep.mp4", 3)


def test_selector_flag_and_profiles(monkeypatch):
    monkeypatch.delenv("SELECTOR", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py", "x", "--selector", "deep"])
    main._apply_settings(main._parse_args())
    import os
    assert os.environ["SELECTOR"] == "deep"
    assert main.PROFILES["claude-deep"]["LLM_PROVIDER"] == "claude"
    assert main.PROFILES["claude-deep"]["SELECTOR"] == "deep"
    assert main.PROFILES["gemma-deep"]["OLLAMA_MODEL"] == "gemma4:12b"


def test_eval_deep_passes_options_and_labels_rows(monkeypatch, tmp_path):
    seen = {}

    def fake_deep(transcript, source_path, num_clips, llm_fn=None, skip=(), samples=None, run=0, stats=None, **k):
        seen.update(num_clips=num_clips, skip=list(skip), samples=samples, run=run)
        return {"highlights": [{"start_time": 0.0, "end_time": 30.0}], "candidates": [{"start_time": 0.0, "end_time": 30.0}]}

    monkeypatch.setattr(ev, "run_deep", fake_deep)
    ep = {"episode": "e", "source": str(tmp_path / "e.mp4"), "gold": [{"start": 0, "end": 30, "rating": 4}]}
    out = ev.run_selector("deep", ep, {"duration": 60.0, "segments": []}, run=1, skip=["bible"], samples=3)
    assert seen == {"num_clips": 10, "skip": ["bible"], "samples": 3, "run": 1}
    assert out["picks"] == [{"start_time": 0.0, "end_time": 30.0}]
    assert ev.selector_label("deep", ["bible", "tournament"], 3) == "deep[-bible,-tournament,samples=3]"
    assert ev.selector_label("deep", [], None) == "deep"
    assert ev.selector_label("classic", ["bible"], 3) == "classic"
