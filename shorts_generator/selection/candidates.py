"""Stage 2: candidate clips for the deep selector, aiming for recall."""
import math
from typing import Dict, List, Optional, Tuple

from .. import config
from ..boundaries import all_words, snap_clip
from ..highlights import (MIN_CLIP_SECONDS, PLATFORM_SIGNALS, TARGET_LENGTHS, VIRALITY_CRITERIA,
                          build_transcript_text, chunk_transcript)
from .cache import Ask, judge_prefix, load_prompt, pmap
from .loudness import loud_peaks

SIGNALS_PER_HOUR = 20
WINDOW_SECONDS = 300
WINDOW_SAMPLES = 2
WINDOW_SLACK_SECONDS = 5
MERGE_IOU = 0.6
KINDS = ["hook_moment", "emotional_peak", "opinion_bomb", "revelation", "conflict", "quotable",
         "story_peak", "practical_value"]

WINDOW_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "maxItems": 8, "items": {
        "type": "object",
        "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                       "kind": {"type": "string", "enum": KINDS}, "why": {"type": "string"}},
        "required": ["start", "end", "kind", "why"]}}},
    "required": ["candidates"],
}


def _span(d) -> Tuple[float, float]:
    """(start, end) of a pick (start_time/end_time), a gold clip (start/end) or a tuple."""
    if isinstance(d, dict):
        if "start_time" in d:
            return float(d["start_time"]), float(d["end_time"])
        return float(d["start"]), float(d["end"])
    return float(d[0]), float(d[1])


def iou(a, b) -> float:
    """Temporal intersection over union of two spans."""
    (s1, e1), (s2, e2) = _span(a), _span(b)
    inter = min(e1, e2) - max(s1, s2)
    union = max(e1, e2) - min(s1, s2)
    return max(0.0, inter) / union if union > 0 else 0.0


def signal_spans(words: List[Dict], times: List[Tuple[float, str, float]], duration: float,
                 per_hour: int = SIGNALS_PER_HOUR, limit: Optional[int] = None) -> List[Dict]:
    """A 20-60 s sentence-snapped span around each (time, origin, strength).

    Keeps the strongest `per_hour` per hour of audio, or the strongest `limit` when given.
    """
    cap = limit if limit is not None else max(1, math.ceil(per_hour * duration / 3600))
    out = []
    for t, origin, strength in sorted(times, key=lambda x: x[2], reverse=True):
        if len(out) >= cap:
            break
        snapped = snap_clip(words, t - 20, t + 20, window=10.0)
        if snapped:
            out.append({**snapped, "origin": [origin], "strength": strength})
    return out


def window_candidates(transcript: Dict, bible: Dict, ask: Ask, target_length: Optional[str] = None) -> List[Dict]:
    """Ask for up to 8 moments per 5-minute window, twice per window, and keep the answers that fit it."""
    duration = float(transcript.get("duration") or 0)
    min_len, max_len, hard_max = TARGET_LENGTHS.get(target_length or config.TARGET_LENGTH, TARGET_LENGTHS["medium"])
    windows = chunk_transcript(transcript, size=WINDOW_SECONDS)
    template, prompt_id = load_prompt("window")
    prefix = judge_prefix(bible)
    jobs = [(i, w, s) for i, w in enumerate(windows) for s in range(WINDOW_SAMPLES)]

    def one(job):
        i, w, sample = job
        prompt = prefix + "\n\n" + template.format(
            virality_criteria=VIRALITY_CRITERIA, platform_signals=PLATFORM_SIGNALS,
            min_len=min_len, max_len=max_len, hard_max=hard_max, transcript=build_transcript_text(w))
        return ask(prompt, WINDOW_SCHEMA, prompt_id, sample)

    out = []
    for (i, w, _), answer in zip(jobs, pmap(one, jobs)):
        lo = w["_offset"] - WINDOW_SLACK_SECONDS
        hi = w["_offset"] + w["duration"] + WINDOW_SLACK_SECONDS
        if duration > 0:
            hi = min(hi, duration)
        for c in (answer or {}).get("candidates") or []:
            try:
                start, end = float(c["start"]), float(c["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start or start < lo or end > hi:
                continue
            out.append({"start_time": start, "end_time": end, "origin": ["llm"], "kind": c.get("kind"),
                        "why": str(c.get("why") or ""), "window_ids": [i]})
    return out


def merge(cands: List[Dict], words: List[Dict], target_length: Optional[str] = None) -> List[Dict]:
    """Widen spans to the target minimum, snap to sentences, merge spans overlapping at IoU > 0.6 into
    their union, keep the target lengths."""
    min_len, _, hard_max = TARGET_LENGTHS.get(target_length or config.TARGET_LENGTH, TARGET_LENGTHS["medium"])
    snapped = []
    for c in cands:
        start, end = float(c["start_time"]), float(c["end_time"])
        if end - start < min_len:  # models answer shorter than asked (Haiku: median 23 s when asked for 20-60)
            mid = (start + end) / 2
            start, end = max(0.0, mid - min_len / 2), max(0.0, mid - min_len / 2) + min_len
        s = snap_clip(words, start, end) if words else None
        snapped.append({**c, **s} if s else {**c, "start_time": start, "end_time": end})
    merged: List[Dict] = []
    for c in sorted(snapped, key=lambda x: (x["start_time"], x["end_time"])):
        m = next((m for m in merged if iou(m, c) > MERGE_IOU), None)
        if m is None:
            merged.append({**c, "origin": list(c.get("origin") or []), "window_ids": list(c.get("window_ids") or [])})
            continue
        m["start_time"] = min(m["start_time"], c["start_time"])
        m["end_time"] = max(m["end_time"], c["end_time"])
        m["origin"] += [o for o in c.get("origin") or [] if o not in m["origin"]]
        m["window_ids"] = sorted(set(m["window_ids"]) | set(c.get("window_ids") or []))
        if c.get("strength") is not None:
            m["strength"] = max(m.get("strength", c["strength"]), c["strength"])
    kept = [m for m in merged if MIN_CLIP_SECONDS <= m["end_time"] - m["start_time"] <= hard_max]
    kept.sort(key=lambda x: x["start_time"])
    return [{**m, "id": f"c{i:03d}"} for i, m in enumerate(kept)]


def generate_candidates(transcript: Dict, bible: Dict, db: List[float], ask: Ask,
                        target_length: Optional[str] = None) -> List[Dict]:
    """LLM window picks + loudness peaks + the bible's payoffs and running jokes, merged."""
    words = all_words(transcript)
    duration = float(transcript.get("duration") or 0)
    llm = window_candidates(transcript, bible, ask, target_length)
    loud = signal_spans(words, [(float(t), "loudness", d) for t, d in loud_peaks(db)], duration)
    bible_times = [(float(p["payoff_t"]), "payoff", 0.0) for p in (bible or {}).get("setups_payoffs") or []
                   if isinstance(p, dict) and isinstance(p.get("payoff_t"), (int, float))]
    bible_times += [(float(t), "running_joke", 0.0) for j in (bible or {}).get("running_jokes") or []
                    if isinstance(j, dict) for t in j.get("times") or [] if isinstance(t, (int, float))]
    marked = signal_spans(words, bible_times, duration, limit=len(bible_times))
    out = merge(llm + loud + marked, words, target_length)
    print(f"[deep] {len(out)} candidates ({len(llm)} from windows, {len(loud)} loudness, {len(marked)} bible)",
          flush=True)
    return out
