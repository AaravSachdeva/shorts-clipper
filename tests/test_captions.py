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


def test_whisper_split_tokens_are_joined():
    # Whisper emits "U.S." as "u" + ".s" and "10,000" as "10" + ",000".
    words = [_w("the", 0.0, 0.2), _w("u", 0.2, 0.3), _w(".s", 0.3, 0.5), _w("has", 0.5, 0.7),
             _w("10", 1.0, 1.2), _w(",000", 1.2, 1.5), _w("users", 1.5, 1.9)]
    texts = [x["text"] for g in group_words(words) for x in g]
    assert texts == ["the", "u.s", "has", "10,000", "users"]
    joined = [x for g in group_words(words) for x in g if x["text"] == "u.s"][0]
    assert (joined["start"], joined["end"]) == (0.2, 0.5)


def _caption_times(doc):
    out = []
    for e in _events(doc):
        if ",Caption," in e:
            f = e.split(",")
            out.append((f[1], f[2]))
    return out


def _secs(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_caption_events_never_overlap():
    # Whisper word times can overlap, and a short pause bridges into the next line.
    words = [_w("pick", 0.0, 0.4), _w("the", 0.35, 0.6), _w("worst.", 0.55, 1.0),
             _w("know", 1.05, 1.3), _w("how", 1.25, 1.5), _w("to", 1.45, 1.7)]
    times = [(_secs(a), _secs(b)) for a, b in _caption_times(build_ass(words, 3.0))]
    assert times == sorted(times)
    assert all(end <= nxt_start for (_, end), (nxt_start, _) in zip(times, times[1:]))


def test_captions_sit_at_one_fixed_position_on_one_line():
    doc = build_ass(WORDS, 4.0, width=1080, height=1920)
    caps = [e for e in _events(doc) if ",Caption," in e]
    assert caps and all("{\\an5\\pos(540,1344)}" in e for e in caps)
    assert "WrapStyle: 2" in doc


def test_hook_wraps_into_balanced_lines():
    from shorts_generator.local.captions import _wrap
    assert _wrap("Why experts said AI could never work") == r"Why experts said\NAI could never work"
    assert _wrap("Short hook") == "Short hook"
    assert all(len(line) <= 20 for line in _wrap("No equity so why is he still the villain of AI").split(r"\N"))
