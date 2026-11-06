import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parent.parent / "research" / "benchmark.py"
)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_hook_features():
    words = [{"t": 0.4, "w": w} for w in "why nobody tells you the 3 rules of rent".split()]
    f = bm.hook_features(words, "3 rules")
    assert f["first_word_s"] == 0.4
    assert f["hook_is_question"] and f["hook_has_number"] and f["hook_says_you"] and f["hook_contrarian"]


def test_summary_compares_top_decile():
    rows = []
    for i in range(40):
        rows.append({
            "url": f"u{i}", "duration": 25 if i < 4 else 70, "views_per_day": 1000 - i,
            "first_word_s": 0.3, "engagement_rate": 0.05, "hook": f"hook {i}",
            "hook_is_question": i < 4, "hook_has_number": False, "hook_says_you": False, "hook_contrarian": False,
        })
    s = bm.summarize(rows, "views_per_day")
    assert "| median length (s) | 25.0 | 70.0 |" in s
    assert "| hook is a question | 100% | 0% |" in s


def test_summary_needs_enough_rows():
    assert "at least 30" in bm.summarize([{"views_per_day": 1}], "views_per_day")


class _CapYdl:
    def __init__(self, data):
        self.data = data

    def urlopen(self, url):
        import io
        import json as _json
        return io.BytesIO(_json.dumps(self.data).encode())


def test_caption_words_max_seconds():
    data = {"events": [{"tStartMs": 5000, "segs": [{"utf8": "early"}]},
                       {"tStartMs": 30000, "segs": [{"utf8": "late"}]}]}
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": "u"}]}}
    assert [w["w"] for w in bm._caption_words(_CapYdl(data), info)] == ["early"]
    assert [w["w"] for w in bm._caption_words(_CapYdl(data), info, max_seconds=float("inf"))] == ["early", "late"]
