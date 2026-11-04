"""Snap clip boundaries to sentence edges using word timestamps.

LLM timestamps come from segment starts ("[812.4s] ..."), which often land
mid-sentence or a breath early/late. With word-level timings we can:
  * start on the first word of a sentence (after a pause or sentence end),
    skipping filler openers like "so", "um", "yeah";
  * end right after a sentence-final word, with a short tail so the last
    syllable isn't clipped, but before the next word begins.
"""
import re
from typing import Dict, List, Optional

SNAP_WINDOW_SECONDS = 3.0
PAUSE_SECONDS = 0.6           # a gap this long also counts as a sentence break
MIN_CLIP_SECONDS = 8.0
START_PAD = 0.08
END_PAD = 0.35
MAX_FILLER_SKIP = 3

FILLERS = {
    "so", "um", "uh", "uhm", "er", "ah", "yeah", "yes", "okay", "ok", "well",
    "like", "right", "anyway", "and", "but", "oh", "hmm", "mm",
}

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")
_STRIP = re.compile(r"[^\w']+")


def all_words(transcript: Dict) -> List[Dict]:
    words = [w for s in transcript.get("segments", []) for w in (s.get("words") or [])]
    words.sort(key=lambda w: w["start"])
    return words


def _norm(word: str) -> str:
    return _STRIP.sub("", word).lower()


def _ends_sentence(word: Dict) -> bool:
    return bool(_SENTENCE_END.search(word["word"].strip()))


def _is_sentence_start(words: List[Dict], i: int) -> bool:
    if i == 0:
        return True
    prev = words[i - 1]
    return _ends_sentence(prev) or (words[i]["start"] - prev["end"]) >= PAUSE_SECONDS


def _is_sentence_end(words: List[Dict], j: int) -> bool:
    if j == len(words) - 1:
        return True
    return _ends_sentence(words[j]) or (words[j + 1]["start"] - words[j]["end"]) >= PAUSE_SECONDS


def _closest(indices: List[int], key, target: float) -> Optional[int]:
    if not indices:
        return None
    return min(indices, key=lambda i: abs(key(i) - target))


def snap_clip(words: List[Dict], start: float, end: float, window: float = SNAP_WINDOW_SECONDS) -> Optional[Dict]:
    """Return {"start_time", "end_time"} snapped to sentence edges, or None."""
    if not words:
        return None

    near_start = [i for i, w in enumerate(words) if abs(w["start"] - start) <= window]
    i = _closest([k for k in near_start if _is_sentence_start(words, k)], lambda k: words[k]["start"], start)
    if i is None:
        i = _closest(near_start, lambda k: words[k]["start"], start)
    if i is None:
        return None

    # Skip filler openers ("So, um, the thing is" -> "the thing is") and
    # dangling 1-2 word fragments ("us. Yeah. Streaming..." -> "Streaming...").
    skipped = 0
    while skipped < MAX_FILLER_SKIP and i + 1 < len(words):
        if _norm(words[i]["word"]) in FILLERS and words[i + 1]["start"] - words[i]["end"] < 1.0:
            i += 1
        elif _ends_sentence(words[i]) and words[i + 1]["start"] - start < window:
            i += 1
        elif (
            i + 2 < len(words)
            and _ends_sentence(words[i + 1])
            and words[i + 2]["start"] - start < window
        ):
            i += 2
        else:
            break
        skipped += 1

    near_end = [j for j, w in enumerate(words) if abs(w["end"] - end) <= window and j > i]
    j = _closest([k for k in near_end if _is_sentence_end(words, k)], lambda k: words[k]["end"], end)
    if j is None:
        j = _closest(near_end, lambda k: words[k]["end"], end)
    if j is None:
        return None

    new_start = words[i]["start"] - START_PAD
    if i > 0:
        new_start = max(new_start, words[i - 1]["end"])
    new_end = words[j]["end"] + END_PAD
    if j + 1 < len(words):
        new_end = min(new_end, words[j + 1]["start"] - 0.02)
    new_start = max(0.0, new_start)

    if new_end - new_start < MIN_CLIP_SECONDS:
        return None
    return {"start_time": round(new_start, 3), "end_time": round(new_end, 3)}


def snap_highlights(transcript: Dict, highlights: List[Dict]) -> List[Dict]:
    """Snap every highlight in place-safe fashion; leaves clips alone when no words exist."""
    words = all_words(transcript)
    if not words:
        return highlights
    out = []
    moved = 0
    for h in highlights:
        snapped = snap_clip(words, float(h["start_time"]), float(h["end_time"]))
        if snapped:
            moved += 1
            out.append({**h, **snapped, "llm_start_time": h["start_time"], "llm_end_time": h["end_time"]})
        else:
            out.append(h)
    print(f"[highlights] snapped {moved}/{len(highlights)} clips to sentence boundaries", flush=True)
    return out


def words_in_range(transcript: Dict, start: float, end: float) -> List[Dict]:
    """Words fully or mostly inside [start, end], with times relative to `start`."""
    out = []
    for w in all_words(transcript):
        mid = (w["start"] + w["end"]) / 2
        if start <= mid <= end:
            out.append({"start": max(0.0, w["start"] - start), "end": max(0.0, w["end"] - start), "word": w["word"]})
    return out
