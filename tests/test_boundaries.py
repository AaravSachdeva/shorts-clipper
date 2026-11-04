from shorts_generator.boundaries import snap_clip, snap_highlights, words_in_range


def _words(text_times):
    return [{"word": w, "start": s, "end": e} for (w, s, e) in text_times]


WORDS = _words([
    ("that", 9.0, 9.3), ("was", 9.3, 9.6), ("crazy.", 9.6, 10.0),
    ("So", 10.4, 10.6), ("um", 10.6, 10.8), ("the", 10.9, 11.0), ("secret", 11.0, 11.4),
    ("is", 11.4, 11.6), ("simple.", 11.6, 12.0),
    ("You", 12.3, 12.5), ("just", 12.5, 12.8), ("start", 12.8, 13.2), ("early.", 13.2, 13.7),
] + [(f"w{i}", 14.0 + i * 0.4, 14.3 + i * 0.4) for i in range(30)] + [
    ("done.", 26.2, 26.6), ("Next", 26.9, 27.1), ("topic", 27.1, 27.5),
])


def test_start_snaps_to_sentence_and_skips_fillers():
    out = snap_clip(WORDS, start=11.2, end=26.0)
    # Sentence starts at "So" (10.4); fillers "So", "um" are skipped → "the" at 10.9.
    assert out["start_time"] == 10.82   # "the" at 10.9 minus the 0.08s lead-in pad


def test_end_snaps_after_sentence_end_without_eating_next_word():
    out = snap_clip(WORDS, start=11.0, end=25.5)
    assert 26.6 <= out["end_time"] < 26.9   # after "done." and before "Next"


def test_too_short_returns_none():
    assert snap_clip(WORDS, start=11.0, end=13.0) is None


def test_snap_highlights_noop_without_words():
    t = {"segments": [{"start": 0, "end": 5, "text": "x"}]}
    h = [{"start_time": 1.0, "end_time": 30.0, "score": 50}]
    assert snap_highlights(t, h) == h


def test_words_in_range_relative():
    t = {"segments": [{"start": 9, "end": 13.7, "text": "", "words": WORDS[:13]}]}
    w = words_in_range(t, 10.9, 13.7)
    assert w[0]["word"] == "the" and w[0]["start"] == 0.0
