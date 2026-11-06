import pytest

from shorts_generator.selection import score as sc


def test_combine_weights():
    a = {"hook": [1.0], "standalone": [0.5], "payoff": [1.0], "emotion": [0.0], "novelty": [1.0]}
    assert sc.combine(a) == pytest.approx(0.75)


def test_combine_renormalises_missing_keys():
    a = {"hook": [1.0], "standalone": [0.5], "emotion": [0.0], "novelty": [1.0]}
    assert sc.combine(a) == pytest.approx(0.55 / 0.8)


def test_combine_penalises_disagreement():
    a = {"hook": [1.0, 0.0], "standalone": [0.5], "payoff": [1.0], "emotion": [0.0], "novelty": [1.0]}
    assert sc.combine(a) == pytest.approx(0.6 - 0.5 * (0.5 / 5))


def test_combine_ignores_ending_and_empty():
    assert sc.combine({"ending": [1.0]}) == 0.0


def test_normalise_clamps_and_maps_answers():
    assert sc.normalise("hook", {"opening_line": "x", "score": 7}) == 1.0
    assert sc.normalise("emotion", {"score": 0}) == 0.0
    assert sc.normalise("standalone", {"score": 3}) == 0.5
    assert sc.normalise("payoff", {"answer": "partial"}) == 0.5
    assert sc.normalise("ending", {"answer": "no"}) == 0.0
    assert sc.normalise("payoff", {"answer": "maybe"}) is None
    assert sc.normalise("emotion", {"score": "high"}) is None
    assert sc.normalise("emotion", None) is None


def _transcript():
    segs = [{"start": 0.0, "end": 30.0, "text": "alpha boring clip"},
            {"start": 100.0, "end": 130.0, "text": "beta great clip"}]
    return {"duration": 200.0, "segments": segs}


def _cands():
    return [{"id": "c000", "start_time": 0.0, "end_time": 30.0},
            {"id": "c001", "start_time": 100.0, "end_time": 130.0}]


def _fake_ask():
    seen = []

    def ask(prompt, schema, prompt_id, sample=0):
        key = prompt_id.split("@")[0]
        clip = "alpha" if "alpha boring" in prompt else "beta"
        seen.append((clip, key, sample))
        s = 1 if clip == "alpha" else 4
        return {"hook": {"opening_line": "o", "score": s}, "standalone": {"score": s},
                "payoff": {"answer": "yes"}, "emotion": {"score": 3},
                "novelty": {"label": "funny", "score": 5}, "ending": {"answer": "yes"}}[key]
    return ask, seen


def test_early_exit_skips_pass_two():
    ask, seen = _fake_ask()
    out = sc.score_candidates(_transcript(), {}, _cands(), ask, samples=1)
    assert [c["id"] for c in out] == ["c001"]
    assert sorted(k for clip, k, _ in seen if clip == "alpha") == ["hook", "standalone"]
    assert sorted(k for clip, k, _ in seen if clip == "beta") == sorted(sc.QUESTIONS)


def test_scored_candidate_has_answers_extras_and_combined():
    ask, _ = _fake_ask()
    [c] = sc.score_candidates(_transcript(), {}, _cands(), ask, samples=1)
    assert c["answers"]["hook"] == [0.75] and c["answers"]["ending"] == [1.0]
    assert c["extras"] == {"hook_quote": "o", "novelty_label": "funny"}
    assert c["combined"] == pytest.approx(sc.combine(c["answers"]))


def test_extra_samples_asked_for_survivors():
    ask, seen = _fake_ask()
    sc.score_candidates(_transcript(), {}, _cands(), ask, samples=2)
    beta = [(k, s) for clip, k, s in seen if clip == "beta"]
    assert sorted(beta) == sorted((k, s) for k in sc.QUESTIONS for s in (0, 1))
