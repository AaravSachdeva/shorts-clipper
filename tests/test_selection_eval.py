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
