import re

from shorts_generator.local.captions import _ts, build_ass, group_words


def _w(word, s, e):
    return {"word": word, "start": s, "end": e}


WORDS = [
    _w("Netflix", 0.10, 0.50), _w("is", 0.50, 0.65), _w("now", 0.65, 0.90), _w("free.", 0.90, 1.30),
    _w("No", 2.00, 2.20), _w("ads,", 2.20, 2.50), _w("no", 2.55, 2.70), _w("payment.", 2.70, 3.20),
]


def test_ts_format():
    assert _ts(0) == "0:00:00.00"
    assert _ts(61.257) == "0:01:01.26"


def test_grouping_breaks_on_sentence_end_and_max_words():
    groups = [[x["text"] for x in g] for g in group_words(WORDS)]
    assert groups[0] == ["Netflix", "is", "now"]
    assert groups[1] == ["free"]                    # period stripped, sentence end breaks
    assert groups[2][0] == "No" and "ads" in groups[2]


def _events(doc):
    return [l for l in doc.splitlines() if l.startswith("Dialogue:")]


def test_each_word_highlight_starts_on_its_word():
    doc = build_ass(WORDS, clip_duration=4.0, hook_text=None)
    ev = [e for e in _events(doc) if ",Caption," in e]
    starts = [e.split(",")[1] for e in ev]
    assert starts[0] == _ts(0.10) and _ts(0.50) in starts and _ts(2.00) in starts
    # exactly one highlighted word per event
    assert all(e.count("\\c&H00") == 2 for e in ev)


def test_hook_overlay_first_three_seconds():
    doc = build_ass(WORDS, clip_duration=40.0, hook_text="They gave it away for free")
    hook = [e for e in _events(doc) if ",Hook," in e]
    assert len(hook) == 1 and hook[0].split(",")[1:3] == [_ts(0), _ts(3.0)]


def test_output_resolution_in_header():
    doc = build_ass(WORDS, 4.0, width=1080, height=1920)
    assert "PlayResX: 1080" in doc and "PlayResY: 1920" in doc


def test_braces_are_neutralised():
    doc = build_ass([_w("{\\b1}evil", 0, 1)], 2.0)
    caption = [e for e in _events(doc) if ",Caption," in e][0]
    assert "\\b1" not in caption.split(",,", 1)[1].replace("\\fscx108\\fscy108", "").replace("\\c", "")
