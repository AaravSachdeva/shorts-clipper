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
