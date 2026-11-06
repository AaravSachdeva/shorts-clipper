import pytest

from shorts_generator import highlights as hl
from shorts_generator.selection import tournament as tn


def test_bradley_terry_orders_a_chain():
    p = tn.bradley_terry(3, wins=[(0, 1), (1, 2), (0, 2)], ties=[])
    assert p[0] > p[1] > p[2]
    assert sum(p) == pytest.approx(1.0)


def test_bradley_terry_all_ties_are_equal():
    p = tn.bradley_terry(3, wins=[], ties=[(0, 1), (1, 2), (0, 2)])
    assert p == pytest.approx([1 / 3] * 3)


def test_bradley_terry_winless_player_is_not_zero():
    assert min(tn.bradley_terry(2, wins=[(0, 1)], ties=[])) > 0


def test_resolve_needs_both_orders_to_agree():
    assert tn.resolve("A", "B") == 0
    assert tn.resolve("B", "A") == 1
    assert tn.resolve("A", "A") is None
    assert tn.resolve(None, "B") is None


def _transcript():
    segs = [{"start": 0.0, "end": 30.0, "text": "strong clip"},
            {"start": 100.0, "end": 130.0, "text": "weak clip"},
            {"start": 200.0, "end": 230.0, "text": "middling clip"}]
    return {"duration": 300.0, "segments": segs}


def _cands():
    return [{"id": "c0", "start_time": 0.0, "end_time": 30.0},
            {"id": "c1", "start_time": 100.0, "end_time": 130.0},
            {"id": "c2", "start_time": 200.0, "end_time": 230.0}]


def test_play_asks_both_orders_and_ranks():
    rank = {"strong": 3, "middling": 2, "weak": 1}
    seen = []

    def ask(prompt, schema, prompt_id, sample=0):
        assert prompt_id.startswith("pair@")
        seen.append(prompt)
        a = prompt.split("Clip A:\n")[1].split("\n")[0].split()[0]
        b = prompt.split("Clip B:\n")[1].split("\n")[0].split()[0]
        return {"winner": "A" if rank[a] > rank[b] else "B"}

    p = tn.play(_cands(), _transcript(), {}, ask)
    assert len(seen) == 6
    assert p[0] > p[2] > p[1]


def test_play_position_bias_gives_ties():
    p = tn.play(_cands(), _transcript(), {}, lambda *a, **k: {"winner": "A"})
    assert p == pytest.approx([1 / 3] * 3)


def test_play_single_candidate_makes_no_calls():
    def ask(*a, **k):
        raise AssertionError("no calls expected")

    assert tn.play(_cands()[:1], _transcript(), {}, ask) == [1.0]


def test_blend_rank_scores():
    assert tn.blend([0.5, 0.3, 0.2], [0.1, 0.9, 0.5]) == pytest.approx([0.6, 0.7, 0.2])
    assert tn.blend([0.4], [0.9]) == [1.0]


def test_mmr_penalises_same_chapter():
    items = [{"final": 1.0, "start_time": 0, "end_time": 20},
             {"final": 0.95, "start_time": 40, "end_time": 60},
             {"final": 0.8, "start_time": 140, "end_time": 160}]
    chapters = [{"start": 0, "end": 100}, {"start": 100, "end": 200}]
    out = tn.mmr(items, k=3, lam=0.3, chapters=chapters)
    assert [items.index(x) for x in out] == [0, 2, 1]
    assert len(tn.mmr(items, k=2, lam=0.3, chapters=chapters)) == 2


def test_text_fields_schema_comes_from_highlight_schema():
    item = hl.HIGHLIGHT_SCHEMA["properties"]["highlights"]["items"]["properties"]
    assert tn.TEXT_SCHEMA["properties"] == {k: item[k] for k in tn.TEXT_FIELDS}


def test_text_fields_answer_and_fallback():
    pick = {"start_time": 0.0, "end_time": 30.0}
    ans = {"title": " T ", "hook_sentence": "h", "on_screen_hook": "o", "virality_reason": "r"}
    assert tn.text_fields(pick, _transcript(), {}, lambda *a, **k: ans) == {
        "title": "T", "hook_sentence": "h", "on_screen_hook": "o", "virality_reason": "r"}
    assert tn.text_fields(pick, _transcript(), {}, lambda *a, **k: None) == {
        "title": "Untitled Highlight", "hook_sentence": "", "on_screen_hook": "", "virality_reason": ""}
