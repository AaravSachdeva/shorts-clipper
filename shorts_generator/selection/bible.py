"""Stage 1: the episode bible, a short summary every judging prompt starts with.

Map-reduce: one call per 5-minute part, then one call that merges the parts.
"""
import json
from typing import Dict, List, Optional

from ..highlights import build_transcript_text, chunk_transcript
from .cache import Ask, load_prompt, pmap

PART_SECONDS = 300

_TIME = {"type": "number"}
_TEXT = {"type": "string"}

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": _TEXT},
        "stories": {"type": "array", "items": {
            "type": "object", "properties": {"start": _TIME, "end": _TIME, "gist": _TEXT},
            "required": ["start", "end", "gist"]}},
        "notable_lines": {"type": "array", "maxItems": 5, "items": {
            "type": "object", "properties": {"t": _TIME, "line": _TEXT}, "required": ["t", "line"]}},
    },
    "required": ["topics", "stories", "notable_lines"],
}

REDUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_summary": _TEXT,
        "chapters": {"type": "array", "maxItems": 15, "items": {
            "type": "object", "properties": {"start": _TIME, "end": _TIME, "title": _TEXT, "one_line": _TEXT},
            "required": ["start", "end", "title", "one_line"]}},
        "setups_payoffs": {"type": "array", "maxItems": 10, "items": {
            "type": "object", "properties": {"setup_t": _TIME, "payoff_t": _TIME, "description": _TEXT},
            "required": ["setup_t", "payoff_t", "description"]}},
        "running_jokes": {"type": "array", "maxItems": 5, "items": {
            "type": "object", "properties": {"label": _TEXT, "times": {"type": "array", "items": _TIME}},
            "required": ["label", "times"]}},
    },
    "required": ["episode_summary", "chapters", "setups_payoffs", "running_jokes"],
}


def build_bible(transcript: Dict, ask: Ask) -> Dict:
    """{episode_summary, chapters, setups_payoffs, running_jokes}, or {} when no part could be read."""
    parts = chunk_transcript(transcript, size=PART_SECONDS)
    map_t, map_id = load_prompt("bible_map")
    maps = pmap(lambda c: ask(map_t.format(transcript=build_transcript_text(c)), MAP_SCHEMA, map_id), parts)
    notes = [
        f"Part {i + 1} ({c['_offset']:.0f}-{c['_offset'] + c['duration']:.0f}s):\n{json.dumps(m, ensure_ascii=False)}"
        for i, (c, m) in enumerate(zip(parts, maps)) if m
    ]
    if not notes:
        return {}
    red_t, red_id = load_prompt("bible_reduce")
    reduced = ask(red_t.format(notes="\n\n".join(notes)), REDUCE_SCHEMA, red_id) or {}
    bible = {k: reduced.get(k) or ([] if k != "episode_summary" else "") for k in REDUCE_SCHEMA["required"]}
    print(f"[deep] bible: {len(json.dumps(bible, ensure_ascii=False))} chars from {len(notes)}/{len(parts)} parts",
          flush=True)
    return bible


def chapter_of(t: float, chapters: List[Dict]) -> Optional[int]:
    """Index of the first chapter whose [start, end] holds t, or None."""
    for i, c in enumerate(chapters or []):
        try:
            if float(c["start"]) <= t <= float(c["end"]):
                return i
        except (KeyError, TypeError, ValueError):
            continue
    return None
