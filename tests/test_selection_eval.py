import json
import math

import pytest

from shorts_generator.selection import eval as ev


def _pick(s, e):
    return {"start_time": s, "end_time": e}


def _gold(s, e, rating=4):
    return {"start": s, "end": e, "rating": rating}


def test_iou():
    assert ev.iou((0, 10), (5, 15)) == pytest.approx(1 / 3)
    assert ev.iou((0, 10), (20, 30)) == 0
    assert ev.iou((3, 9), (3, 9)) == 1


def test_match_is_greedy_and_uses_each_gold_once():
    gold = [_gold(100, 140)]
    picks = [_pick(100, 140), _pick(101, 141)]
    assert ev.match(picks, gold, k=5) == [(0, 0, 1.0)]


def test_match_ignores_low_iou():
    assert ev.match([_pick(0, 10)], [_gold(8, 30)], k=5) == []


def test_precision_divides_by_k():
    gold = [_gold(0, 30), _gold(100, 130)]
    picks = [_pick(0, 30), _pick(100, 130), _pick(200, 230), _pick(300, 330), _pick(400, 430)]
    assert ev.precision_at(picks, gold, 5) == pytest.approx(0.4)
    assert ev.precision_at([_pick(0, 30)], gold, 5) == pytest.approx(0.2)


def test_ndcg_uses_ratings_as_gain():
    gold = [_gold(0, 30, 5), _gold(100, 130, 3)]
    picks = [_pick(0, 30), _pick(500, 530), _pick(100, 130)]
    assert ev.ndcg_at(picks, gold, 10) == pytest.approx(6.5 / (5 + 3 / math.log2(3)))


def test_recall_counts_gold_hit_by_any_candidate():
    gold = [_gold(0, 30), _gold(100, 130)]
    assert ev.recall([_pick(500, 530), _pick(2, 30)], gold) == pytest.approx(0.5)


def test_boundary_error_is_mean_of_start_and_end_offsets():
    gold = [_gold(100, 140)]
    assert ev.boundary_error([_pick(102, 144)], gold, 5) == pytest.approx(3.0)
    assert ev.boundary_error([_pick(500, 540)], gold, 5) is None


def test_jaccard_top():
    a = [_pick(0, 30), _pick(100, 130)]
    b = [_pick(0, 30), _pick(200, 230)]
    assert ev.jaccard_top(a, b, 5) == pytest.approx(1 / 3)
    assert ev.jaccard_top(a, a, 5) == 1


def test_random_spans_are_seeded_and_inside_duration():
    a = ev.random_spans(600.0, n=10, seed="ep1")
    assert a == ev.random_spans(600.0, n=10, seed="ep1")
    assert a != ev.random_spans(600.0, n=10, seed="ep2")
    assert len(a) == 10
    for s in a:
        assert 0 <= s["start_time"] < s["end_time"] <= 600.0
        assert 20 <= s["end_time"] - s["start_time"] <= 60


def test_load_episodes_filters_split_and_skips_empty_gold(tmp_path, capsys):
    def write(name, split, gold):
        d = tmp_path / name
        d.mkdir()
        (d / "gold.json").write_text(json.dumps({"episode": name, "source": f"{name}.mp4", "split": split, "gold": gold}))

    write("a", "tune", [_gold(0, 30)])
    write("b", "test", [_gold(0, 30)])
    write("c", "test", [])
    assert [e["episode"] for e in ev.load_episodes(tmp_path, "test")] == ["b"]
    assert [e["episode"] for e in ev.load_episodes(tmp_path, "all")] == ["a", "b"]
    assert "c" in capsys.readouterr().out


def test_report_has_row_per_selector():
    rows = []
    for sel in ("random", "signal", "classic"):
        for ep in ("e1", "e2"):
            rows.append({"selector": sel, "episode": ep, "recall": 0.5, "p5": 0.2, "ndcg10": 0.3,
                         "boundary": 4.0, "stability": None, "calls": 10, "llm_seconds": 3600.0})
    md = ev.report(rows)
    for sel in ("random", "signal", "classic"):
        assert f"| {sel} | 2 |" in md
    assert "Episodes: 2" in md


def _classic_fake():
    import re as _re
    calls = []

    def fake(prompt, schema=None):
        calls.append(prompt)
        if "classify the content type" in prompt:
            return json.dumps({"content_type": "podcast", "density": "medium"})
        if "candidate clips" in prompt:
            ids = [int(x) for x in _re.findall(r"^\[(\d+)\]", prompt, flags=_re.M)]
            return json.dumps({"rankings": [{"id": i, "score": 50 + i} for i in ids]})
        return json.dumps({"highlights": [{
            "title": "t", "start_time": 10.0, "end_time": 40.0, "hook_sentence": "h", "virality_reason": "r",
            "hook": 7, "standalone": 6, "payoff": 5, "on_screen_hook": "o"}]})
    return fake, calls


def _ep(tmp_path):
    return {"episode": "e1", "source": str(tmp_path / "source_e1.mp4"), "gold": [_gold(10, 40)]}


def _transcript(duration=200.0):
    segs = [{"start": t, "end": t + 9.5, "text": f"Sentence at {t:.0f}."} for t in range(0, int(duration), 10)]
    return {"duration": duration, "segments": segs}


def test_classic_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.config, "WORK_DIR", str(tmp_path / "work"))
    fake, calls = _classic_fake()
    monkeypatch.setattr(ev, "call_local_llm", fake)
    first = ev.run_selector("classic", _ep(tmp_path), _transcript())
    n = len(calls)
    second = ev.run_selector("classic", _ep(tmp_path), _transcript())
    assert n > 0 and len(calls) == n
    assert second["stats"].get("misses", 0) == 0 and second["stats"]["calls"] == first["stats"]["calls"]
    assert [p["start_time"] for p in second["picks"]] == [10.0]


def test_signal_picks_loudest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.config, "WORK_DIR", str(tmp_path / "work"))
    db = [-30.0] * 300
    db[60], db[200] = -20.0, -10.0
    monkeypatch.setattr(ev, "loudness_per_second", lambda src, cache: db)
    words = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": f"w{i}" + ("." if i % 8 == 7 else "")}
             for i in range(600)]
    transcript = {"duration": 300.0, "segments": [{"start": 0, "end": 300, "text": "x", "words": words}]}
    out = ev.run_selector("signal", _ep(tmp_path), transcript)
    assert [p["strength"] for p in out["picks"]] == [-10.0, -20.0]
    assert out["candidates"] == out["picks"]


def test_random_is_seeded_by_episode_and_run(tmp_path):
    a = ev.run_selector("random", _ep(tmp_path), _transcript())["picks"]
    assert a == ev.run_selector("random", _ep(tmp_path), _transcript())["picks"]
    assert a != ev.run_selector("random", _ep(tmp_path), _transcript(), run=1)["picks"]


def test_merge_rows_replaces_same_label_and_episode():
    old = [{"selector": "deep", "episode": "e1", "p5": 0.0}, {"selector": "classic", "episode": "e1", "p5": 0.2}]
    new = [{"selector": "deep", "episode": "e1", "p5": 0.4}, {"selector": "deep[-bible]", "episode": "e1", "p5": 0.2}]
    assert ev.merge_rows(old, new) == [old[1], new[0], new[1]]


def test_evaluate_skips_transcript_without_word_timings(monkeypatch, capsys, tmp_path):
    from shorts_generator.local import transcriber
    monkeypatch.setattr(transcriber, "transcribe_local", lambda src, language=None: _transcript())
    assert ev.evaluate([_ep(tmp_path)], ["random"]) == []
    assert "no word timings" in capsys.readouterr().out


def test_evaluate_runs_episodes_in_parallel_and_keeps_order(monkeypatch, tmp_path):
    import threading
    import time as _time
    from shorts_generator.local import transcriber
    words = [{"start": 0.0, "end": 0.4, "word": "hi"}]
    monkeypatch.setattr(transcriber, "transcribe_local", lambda src, language=None: {
        "duration": 60.0, "segments": [{"start": 0, "end": 60, "text": "hi", "words": words}]})
    lock, live, peak = threading.Lock(), [0], [0]

    def fake_run(name, ep, transcript, run=0, skip=(), samples=None):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        _time.sleep(0.05)
        with lock:
            live[0] -= 1
        return {"picks": [], "candidates": [], "stats": {}}

    monkeypatch.setattr(ev, "run_selector", fake_run)
    eps = [{"episode": f"e{i}", "source": f"/v/e{i}.mp4", "gold": [_gold(0, 30)]} for i in range(4)]
    rows = ev.evaluate(eps, ["random", "classic"])
    assert [(r["episode"], r["selector"]) for r in rows] == [(f"e{i}", s) for i in range(4) for s in ("random", "classic")]
    assert peak[0] > 1


def test_selector_label_names_a_non_default_target_length(monkeypatch):
    from shorts_generator.selection import eval as ev
    monkeypatch.setattr(ev.config, "TARGET_LENGTH", "medium")
    assert ev.selector_label("deep", ["bible"], None) == "deep[-bible]"
    monkeypatch.setattr(ev.config, "TARGET_LENGTH", "long")
    assert ev.selector_label("classic", [], None) == "classic@long"
    assert ev.selector_label("deep", [], 3) == "deep[samples=3]@long"
