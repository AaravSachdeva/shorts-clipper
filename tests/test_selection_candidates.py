import shutil
import subprocess

import numpy as np
import pytest

from shorts_generator.boundaries import START_PAD
from shorts_generator.selection import candidates as cd
from shorts_generator.selection import loudness as ld


def _sentences(n_sentences, per=8, step=0.5):
    """Words of n sentences, `per` words each, one word every `step` seconds."""
    words = []
    for s in range(n_sentences):
        for k in range(per):
            i = s * per + k
            text = f"s{s}w{k}" + ("." if k == per - 1 else "")
            words.append({"start": i * step, "end": i * step + 0.4, "word": text})
    return words


def test_rms_db_per_whole_second():
    rate = 100
    silence = np.zeros(rate, dtype=np.int16)
    square = np.array([32767, -32767] * (rate // 2), dtype=np.int16)
    db = ld.rms_db(np.concatenate([silence, square, square[:10]]), rate)
    assert len(db) == 2
    assert db[0] < -100
    assert db[1] == pytest.approx(0.0, abs=0.01)


def test_loud_peaks_finds_one_run():
    assert ld.loud_peaks([-30.0] * 60 + [-20.0] * 3 + [-30.0] * 60) == [(60, -20.0)]


def test_loud_peaks_merges_runs_closer_than_30s():
    db = [-30.0] * 60 + [-20.0] + [-30.0] * 10 + [-15.0] + [-30.0] * 60
    assert ld.loud_peaks(db) == [(71, -15.0)]


def test_loud_peaks_needs_more_than_6db():
    assert ld.loud_peaks([-30.0] * 60 + [-24.0] * 3 + [-30.0] * 60) == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_loudness_per_second_decodes_and_caches(tmp_path, monkeypatch):
    src = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    str(src)], check=True)
    cache = tmp_path / "loudness.json"
    db = ld.loudness_per_second(str(src), cache)
    assert len(db) == 2 and all(-30 < d < -10 for d in db)
    assert cache.exists()

    def boom(*a, **k):
        raise AssertionError("ffmpeg must not run when the cache is fresh")

    monkeypatch.setattr(ld.subprocess, "Popen", boom)
    assert ld.loudness_per_second(str(src), cache) == db


def test_signal_spans_snap_to_sentences():
    words = _sentences(100)  # 400 s, a sentence every 4 s
    [span] = cd.signal_spans(words, [(200.0, "loudness", -10.0)], duration=400.0)
    starts = {w["start"] for i, w in enumerate(words) if i % 8 == 0}
    assert any(abs(span["start_time"] - (s - START_PAD)) < 0.2 for s in starts)
    assert 20 <= span["end_time"] - span["start_time"] <= 60
    assert span["origin"] == ["loudness"] and span["strength"] == -10.0



def test_signal_spans_cap_per_hour_keeps_strongest():
    words = _sentences(900)  # 3600 s
    times = [(t, "loudness", float(i)) for i, t in enumerate([300.0, 900.0, 1500.0, 2100.0, 2700.0])]
    spans = cd.signal_spans(words, times, duration=3600.0, per_hour=2)
    assert sorted(s["strength"] for s in spans) == [3.0, 4.0]


def _window_transcript(duration=300.0):
    segs = [{"start": float(t), "end": t + 9.5, "text": f"Sentence at {t}."} for t in range(0, int(duration), 10)]
    return {"duration": duration, "segments": segs}


def _window_ask(cands):
    seen = []

    def ask(prompt, schema, prompt_id, sample=0):
        assert prompt_id.startswith("window@")
        seen.append((prompt, sample))
        return {"candidates": cands}
    return ask, seen


def test_each_window_asked_twice_with_bible_prefix():
    ask, seen = _window_ask([])
    cd.window_candidates(_window_transcript(), {"episode_summary": "about cats"}, ask)
    from shorts_generator import highlights as hl
    n = len(hl.chunk_transcript(_window_transcript(), size=300))
    assert sorted(s for _, s in seen) == sorted([0, 1] * n)
    assert all("about cats" in p and "OPINION BOMBS" in p for p, _ in seen)


def test_out_of_window_and_inverted_spans_dropped():
    ask, _ = _window_ask([
        {"start": 10, "end": 40, "kind": "quotable", "why": "w"},
        {"start": 50, "end": 45, "kind": "quotable", "why": "inverted"},
        {"start": 400, "end": 430, "kind": "quotable", "why": "past the end"},
    ])
    out = cd.window_candidates(_window_transcript(), {}, ask)
    assert [(c["start_time"], c["end_time"], c["window_ids"]) for c in out] == [(10.0, 40.0, [0])] * 2
    assert out[0]["origin"] == ["llm"] and out[0]["kind"] == "quotable"


def _c(s, e, origin="llm", w=None):
    return {"start_time": s, "end_time": e, "origin": [origin], "kind": "k", "why": "y", "window_ids": w or []}


def test_merge_unions_overlapping_spans_and_origins():
    [m] = cd.merge([_c(0, 30, "llm", [1]), _c(4.5, 34.5, "loudness")], [], "medium")
    assert (m["start_time"], m["end_time"]) == (0, 34.5)
    assert m["origin"] == ["llm", "loudness"] and m["window_ids"] == [1]


def test_merge_keeps_iou_half_separate():
    assert len(cd.merge([_c(0, 30), _c(10, 40)], [], "medium")) == 2


def test_merge_drops_out_of_range_lengths_and_numbers_by_start():
    out = cd.merge([_c(500, 530), _c(0, 100), _c(200, 210), _c(100, 130)], [], "medium")
    assert [(c["id"], c["start_time"]) for c in out] == [("c000", 100), ("c001", 500)]


def test_generate_candidates_adds_loudness_and_bible_spans():
    words = _sentences(150)  # 600 s
    transcript = {"duration": 600.0, "segments": [{"start": 0, "end": 600, "text": "x", "words": words}]}
    db = [-30.0] * 600
    db[100] = -10.0
    bible = {"setups_payoffs": [{"setup_t": 10, "payoff_t": 400, "description": "d"}], "running_jokes": []}
    ask, _ = _window_ask([])
    out = cd.generate_candidates(transcript, bible, db, ask)
    origins = sorted(o for c in out for o in c["origin"])
    assert origins == ["loudness", "payoff"]


def test_window_prompt_uses_the_target_length():
    seen = []

    def ask(prompt, schema, prompt_id, sample=0):
        seen.append(prompt)
        return {"candidates": []}

    cd.window_candidates(_window_transcript(), {}, ask, target_length="long")
    assert "45-90 seconds" in seen[0] and "150 seconds" in seen[0]
    seen.clear()
    cd.window_candidates(_window_transcript(), {}, ask)
    assert "20-60 seconds" in seen[0] and "90 seconds" in seen[0]
