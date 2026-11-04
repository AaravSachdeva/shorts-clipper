import json
import re

from shorts_generator import highlights as hl


def _transcript(duration=3600, step=10.0):
    segs = []
    t = 0.0
    while t < duration:
        segs.append({"start": t, "end": t + step - 0.5, "text": f"Sentence at {t:.0f}."})
        t += step
    return {"duration": duration, "segments": segs}


def test_chunk_prompts_are_relative_and_offsets_round_trip(monkeypatch):
    monkeypatch.setattr(hl.config, "HIGHLIGHT_CHUNK_SECONDS", 1200)
    seen = []

    def fake_llm(prompt, schema=None):
        if "classify the content type" in prompt:
            return json.dumps({"content_type": "podcast", "density": "medium"})
        if "candidate clips" in prompt:
            ids = [int(x) for x in re.findall(r"^\[(\d+)\]", prompt, flags=re.M)]
            return json.dumps({"rankings": [{"id": i, "score": 50 + i} for i in ids]})
        first = re.search(r"Transcript:\n\[(\d+\.\d)s\]", prompt).group(1)
        seen.append(float(first))
        assert schema is hl.HIGHLIGHT_SCHEMA
        # Answer the way a model naturally would: with timestamps as shown.
        return json.dumps({"highlights": [{
            "title": "t", "start_time": 100.0, "end_time": 140.0, "score": 70, "hook_score": 60,
            "shareability_score": 50, "hook_sentence": "h", "on_screen_hook": "o", "virality_reason": "r",
        }]})

    out = hl.get_highlights(_transcript(), num_clips=3, llm_fn=fake_llm)["highlights"]
    # Every chunk prompt starts at (or just after) 0s: chunk-relative.
    assert seen and all(s < 15 for s in seen)
    starts = sorted(h["start_time"] for h in out)
    # Chunk offsets are 0, 1140, 2280, 3420 → clips at offset + 100.
    assert starts == [100.0, 1240.0, 2380.0, 3520.0]


def test_highlight_in_overlap_tail_is_kept(monkeypatch):
    # duration 1250s of a 1200s chunk: highlights in the 60s overlap must survive.
    clean = hl._sanitize_highlights([{"start_time": 1210, "end_time": 1250, "score": 80}], duration=1260)
    assert clean and clean[0]["end_time"] == 1250


def test_parse_json_loose_strips_think_and_fences():
    raw = "<think>reasoning {not json}</think>\n```json\n{\"a\": 1}\n```"
    assert hl._parse_json_loose(raw) == {"a": 1}


def test_invoke_supports_single_arg_llm_fn():
    assert hl._invoke(lambda p: "ok:" + p, "x", {"type": "object"}) == "ok:x"
    assert hl._invoke(lambda p, schema=None: str(bool(schema)), "x", {"t": 1}) == "True"


def test_content_sample_spans_whole_video():
    t = _transcript(duration=3600)
    sample = hl._content_sample(t["segments"])
    assert "Sentence at 0." in sample and "Sentence at 3590." in sample


def test_rerank_rescores_and_keeps_rest_below(monkeypatch):
    t = _transcript(duration=600)
    cands = [
        {"title": f"c{i}", "start_time": i * 40.0, "end_time": i * 40.0 + 30, "score": 90 - i}
        for i in range(14)
    ]

    def fake_llm(prompt, schema=None):
        ids = [int(x) for x in re.findall(r"^\[(\d+)\]", prompt, flags=re.M)]
        assert len(ids) == hl.RERANK_TOP_N
        return json.dumps({"rankings": [{"id": i, "score": 100 - (len(ids) - i) * 5} for i in ids]})

    out = hl.rerank_highlights(t, cands, {"content_type": "podcast"}, fake_llm)
    # Reversed order: the previously lowest of the top 12 now wins.
    assert out[0]["title"] == "c11"
    top_scores = [h["score"] for h in out[:12]]
    assert min(top_scores) > max(h["score"] for h in out[12:])


def test_rerank_failure_keeps_scores():
    t = _transcript(duration=300)
    cands = [{"title": "a", "start_time": 0, "end_time": 30, "score": 80},
             {"title": "b", "start_time": 60, "end_time": 90, "score": 70}]
    out = hl.rerank_highlights(t, cands, {}, lambda p, schema=None: "not json")
    assert [h["score"] for h in out] == [80, 70]


def test_examples_block_in_prompt():
    block = hl._examples_block([{"hook": "Nobody tells you this about rent.", "duration": 31}])
    assert "Nobody tells you this" in block and "(31s clip)" in block
