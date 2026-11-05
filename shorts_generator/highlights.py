"""Find the most viral-worthy highlights in a transcript.

Pipeline:
  1. classify content type / density from samples across the whole video
  2. ask the LLM for highlights one overlapping window at a time (chunk-relative
     timestamps); it scores each clip on a fixed hook / standalone / payoff
     rubric and the overall score is computed here, not by the model
  3. snap each clip to sentence boundaries using word timestamps
  4. drop overlapping candidates (score-based)
  5. re-rank the top candidates in one comparative call, so scores from
     different chunks become comparable

The LLM call is pluggable via `llm_fn(prompt, schema=None)` so the same prompts
drive MuAPI (--mode api) or a local client (OpenAI / Gemini / Ollama). A
plain `llm_fn(prompt)` without a schema parameter is still accepted.
"""
import inspect
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config, muapi
from .boundaries import all_words, snap_highlights


LLMFn = Callable[..., str]


# --- schemas (structured output) ---------------------------------------------

CONTENT_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {
            "type": "string",
            "enum": ["podcast", "interview", "tutorial", "lecture", "commentary", "debate", "vlog", "other"],
        },
        "density": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["content_type", "density"],
}

HIGHLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Order matters for structured output: quote and reason before scoring.
                    "title": {"type": "string"},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "hook_sentence": {"type": "string"},
                    "virality_reason": {"type": "string"},
                    "hook": {"type": "integer"},
                    "standalone": {"type": "integer"},
                    "payoff": {"type": "integer"},
                    "on_screen_hook": {"type": "string"},
                },
                "required": [
                    "title", "start_time", "end_time", "hook_sentence", "virality_reason",
                    "hook", "standalone", "payoff", "on_screen_hook",
                ],
            },
        }
    },
    "required": ["highlights"],
}

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score"],
            },
        }
    },
    "required": ["rankings"],
}


# --- prompts -------------------------------------------------------------------

CONTENT_TYPE_PROMPT = """Analyze these transcript samples (taken from across one video) and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


PLATFORM_SIGNALS = """
What YouTube Shorts and Instagram Reels reward (judge every clip against this):
- The first 1-3 seconds decide whether viewers swipe away. The clip's very first sentence must be its hook: no greetings, no setup, no "so"/"yeah"/"anyway", no reaction to something the viewer hasn't seen.
- Completion matters more than length: a tight 30-second clip watched to the end beats a 60-second clip abandoned halfway. Choose the shortest version that still lands.
- Instagram weighs sends (DM shares) heavily: favor moments someone would send to a friend because they are relatable, surprising, funny, or quotable.
- Viewers have never seen this show: each clip must make sense on its own.
"""


RUBRIC = """Score every clip 0-10 on this fixed rubric. 5 is an ordinary clip; use the whole range:
- hook: does the first sentence alone stop the scroll? 0 = greeting, filler or setup; 5 = mildly interesting; 10 = instant curiosity or surprise
- standalone: does it make sense to someone who never saw this video? 0 = needs earlier context; 5 = small gaps; 10 = fully self-contained
- payoff: does it end on a punchline, conclusion or reaction? 0 = trails off or cuts mid-thought; 5 = soft ending; 10 = strong landing"""

# Weights for the overall 0-100 score. The hook weighs most: the first seconds decide the swipe.
RUBRIC_WEIGHTS = {"hook": 0.4, "standalone": 0.3, "payoff": 0.3}


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}
{platform_signals}
Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- start_time is the timestamp of the clip's hook sentence: the clip opens on that sentence
- Duration: aim for {min_len}-{max_len} seconds. Go up to {hard_max} seconds only when a story needs the full arc to land. Never shorter than 12 seconds
- end_time lands right after a payoff (punchline, conclusion, reaction); never cut mid-sentence or mid-thought
- NEVER select ads, sponsor reads, trailers for other shows, appeals to support the show (Patreon, merch, "like and subscribe"), intros, or outros
- Clips must not overlap significantly with each other
- hook_sentence: the clip's opening line, quoted from the transcript
- on_screen_hook: 3-8 words of overlay text for the first seconds that creates curiosity without spoiling the payoff (no hashtags, no emojis)
- virality_reason: one sentence on why this clip will perform
{rubric}
- {num_clips_instruction}. If this part of the video is only ads, plugs, or an outro, return {{"highlights": []}}
{examples}
Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"hook_sentence":"string","virality_reason":"string","hook":int,"standalone":int,"payoff":int,"on_screen_hook":"string"}}]}}"""


RERANK_PROMPT = """You are a short-form video editor choosing which clips to post as YouTube Shorts and Instagram Reels.
Below are {n} candidate clips cut from the same {content_type}. They were scored separately, so their scores are not comparable.
Compare them against each other and give each a final score from 0 to 100 based on:
- hook: does the first sentence stop the scroll?
- standalone: tight, no dead air, makes sense without context
- payoff: does it end on a strong landing?
Use the full range: the best clip should score clearly higher than an average one.

{candidates}

Respond ONLY with valid JSON: {{"rankings":[{{"id":int,"score":int,"reason":"string"}}]}}"""


TARGET_LENGTHS = {
    # name: (aim_min, aim_max, hard_max) seconds
    "short": (15, 35, 45),
    "medium": (20, 60, 90),
    "long": (45, 90, 150),
}

CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap MuAPI polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3
RERANK_TOP_N = 12
MIN_CLIP_SECONDS = 12  # matches the prompt's "never shorter than 12 seconds"


def chunk_size_seconds() -> int:
    return max(300, int(config.HIGHLIGHT_CHUNK_SECONDS))


def call_muapi_llm(prompt: str, schema: Optional[Dict] = None) -> str:
    """Default LLM backend: MuAPI gpt-5-mini. `schema` is not supported and ignored."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _invoke(llm_fn: LLMFn, prompt: str, schema: Optional[Dict]) -> str:
    """Call llm_fn with a schema if it accepts one, else with the prompt only."""
    try:
        params = inspect.signature(llm_fn).parameters
        accepts = "schema" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        accepts = False
    return llm_fn(prompt, schema=schema) if accepts else llm_fn(prompt)


def _parse_json_loose(raw: str) -> Dict:
    """Strip reasoning blocks and markdown fences, then parse the JSON object."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp_score(value: object) -> int:
    return max(0, min(100, _coerce_int(value, default=0)))


def _rubric_score(item: Dict) -> Optional[Dict]:
    """The clamped 0-10 rubric, or None if the model left any part of it out."""
    if not all(isinstance(item.get(k), (int, float, str)) and str(item.get(k)).strip() for k in RUBRIC_WEIGHTS):
        return None
    return {k: max(0, min(10, _coerce_int(item.get(k)))) for k in RUBRIC_WEIGHTS}


def _sanitize_highlights(raw_highlights: object, duration: float) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        rubric = _rubric_score(item)
        clip = {
            "title": str(item.get("title") or "Untitled Highlight").strip(),
            "start_time": start,
            "end_time": end,
            # Small models give noisy holistic scores; a weighted fixed rubric is steadier.
            "score": round(sum(rubric[k] * w * 10 for k, w in RUBRIC_WEIGHTS.items())) if rubric
                     else _clamp_score(item.get("score")),
            "hook_sentence": str(item.get("hook_sentence") or "").strip(),
            "on_screen_hook": str(item.get("on_screen_hook") or "").strip(),
            "virality_reason": str(item.get("virality_reason") or "").strip(),
        }
        if rubric:
            clip["rubric"] = rubric
        cleaned.append(clip)

    return cleaned


def _content_sample(segments: List[Dict], windows: int = 5, per_window: int = 6, limit: int = 3000) -> str:
    """Take a few consecutive segments from evenly spaced points in the video.

    Sampling only the opening misclassifies videos that start with an ad,
    trailer, or cold open.
    """
    if not segments:
        return ""
    n = len(segments)
    parts = []
    for k in range(windows):
        i = int(k * max(0, n - per_window) / max(1, windows - 1))
        parts.append(" ".join(s["text"] for s in segments[i:i + per_window]))
    sample = "\n...\n".join(parts)
    return sample[:limit]


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    sample = _content_sample(transcript.get("segments", []))
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript samples:\n{sample}"
    try:
        raw = _invoke(llm_fn, prompt, CONTENT_TYPE_SCHEMA)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict, offset: float = 0.0) -> str:
    """Render segments as "[t s] text". For chunks, pass the chunk offset so the
    timestamps shown to the LLM are chunk-relative — matching the offset that
    get_highlights() adds back to the returned start/end times."""
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start'] - offset:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    size = chunk_size_seconds()
    chunks = []
    start = 0
    while start < duration:
        end = min(start + size, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            # Chunks include up to CHUNK_OVERLAP_SECONDS past `end`; allow
            # highlights in that tail instead of clamping them away.
            chunk["duration"] = min(end + CHUNK_OVERLAP_SECONDS, duration) - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += size - CHUNK_OVERLAP_SECONDS
    return chunks


def load_examples(path: Optional[str]) -> List[Dict]:
    """Read top-performing clip openings (JSONL from research/benchmark.py)."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[highlights] examples file not found: {path}", flush=True)
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _examples_block(examples: List[Dict], limit: int = 15) -> str:
    rows = [e for e in examples if str(e.get("hook") or "").strip()][:limit]
    if not rows:
        return ""
    lines = [
        "",
        "Opening lines of top-performing clips in this niche (match their energy and directness; do not copy them):",
    ]
    for e in rows:
        dur = e.get("duration")
        suffix = f" ({float(dur):.0f}s clip)" if isinstance(dur, (int, float)) else ""
        lines.append(f'- "{str(e["hook"]).strip()[:160]}"{suffix}')
    return "\n".join(lines) + "\n"


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
    target_length: Optional[str] = None,
    examples: Optional[List[Dict]] = None,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload.
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    lo, hi, hard_max = TARGET_LENGTHS.get((target_length or config.TARGET_LENGTH), TARGET_LENGTHS["medium"])
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        platform_signals=PLATFORM_SIGNALS,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        min_len=lo,
        max_len=hi,
        hard_max=hard_max,
        num_clips_instruction=f"Generate at least {min_clips} highlights",
        rubric=RUBRIC,
        examples=_examples_block(examples or []),
    )
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = _invoke(llm_fn, prompt, HIGHLIGHT_SCHEMA)
        try:
            parsed = _parse_json_loose(raw)
            raw_list = parsed.get("highlights")
            highlights = _sanitize_highlights(raw_list, duration=duration)
            if highlights:
                return {"highlights": highlights}
            # A chunk may legitimately have nothing worth clipping (an outro,
            # an ad block); an explicit empty list is a valid answer there.
            if is_chunk and isinstance(raw_list, list) and not raw_list:
                print("[highlights] model found nothing clip-worthy in this chunk", flush=True)
                return {"highlights": []}
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, hook_sentence,"
                + " virality_reason, hook, standalone, payoff (0-10 each), on_screen_hook."
                + " Times are in seconds as shown in the transcript. No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def _clip_excerpt(transcript: Dict, start: float, end: float, head: int = 650, tail: int = 250) -> str:
    """The clip's words exactly as cut (word timings), else overlapping segments."""
    words = [w for w in all_words(transcript) if start <= (w["start"] + w["end"]) / 2 <= end]
    if words:
        text = " ".join(w["word"] for w in words)
    else:
        text = " ".join(
            s["text"].strip() for s in transcript.get("segments", [])
            if s["end"] > start and s["start"] < end
        )
    if len(text) <= head + tail + 5:
        return text
    return text[:head].rstrip() + " … " + text[-tail:].lstrip()


def rerank_highlights(
    transcript: Dict,
    highlights: List[Dict],
    content_info: Dict,
    llm_fn: LLMFn,
    top_n: int = RERANK_TOP_N,
) -> List[Dict]:
    """Re-score the top candidates against each other in a single call.

    Per-chunk scores come from separate LLM calls and aren't comparable; one
    comparative pass gives a consistent ordering. On any failure the original
    scores are kept.
    """
    ranked = sorted(highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    top, rest = ranked[:top_n], ranked[top_n:]
    if len(top) < 2:
        return ranked

    blocks = []
    for i, h in enumerate(top, 1):
        dur = float(h["end_time"]) - float(h["start_time"])
        blocks.append(
            f'[{i}] "{h.get("title", "")}" ({dur:.0f}s)\n'
            f"{_clip_excerpt(transcript, float(h['start_time']), float(h['end_time']))}"
        )
    prompt = RERANK_PROMPT.format(
        n=len(top),
        content_type=content_info.get("content_type", "video"),
        candidates="\n\n".join(blocks),
    )
    try:
        parsed = _parse_json_loose(_invoke(llm_fn, prompt, RERANK_SCHEMA))
        new_scores = {
            _coerce_int(r.get("id"), -1): _clamp_score(r.get("score"))
            for r in parsed.get("rankings", [])
            if isinstance(r, dict)
        }
    except Exception as e:
        print(f"[highlights] re-rank failed ({e}); keeping per-chunk scores", flush=True)
        return ranked
    if not new_scores:
        print("[highlights] re-rank returned no scores; keeping per-chunk scores", flush=True)
        return ranked

    for i, h in enumerate(top, 1):
        if i in new_scores:
            h["chunk_score"] = h.get("score")
            h["score"] = new_scores[i]
    # Anything outside the re-ranked set must not outrank it.
    floor = min(int(h.get("score", 0)) for h in top)
    for h in rest:
        h["chunk_score"] = h.get("score")
        h["score"] = min(int(h.get("score", 0)), max(0, floor - 1))
    print(f"[highlights] re-ranked top {len(top)} candidates", flush=True)
    return sorted(top + rest, key=lambda h: int(h.get("score", 0)), reverse=True)


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
    rerank: Optional[bool] = None,
    target_length: Optional[str] = None,
    examples: Optional[List[Dict]] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    """
    llm_fn = llm_fn or call_muapi_llm
    rerank = config.HIGHLIGHT_RERANK if rerank is None else rerank
    if examples is None:
        examples = load_examples(config.HIGHLIGHT_EXAMPLES_PATH)
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    kwargs = dict(num_clips=num_clips, llm_fn=llm_fn, target_length=target_length, examples=examples)
    all_highlights: List[Dict] = []
    # One pass over a whole video asks too much of a small model: score window by window.
    if duration > chunk_size_seconds():
        chunks = chunk_transcript(transcript)
        print(f"[highlights] splitting into {len(chunks)} overlapping windows of {chunk_size_seconds()}s", flush=True)
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk, offset=offset)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(text, content_info, chunk["duration"], is_chunk=True, **kwargs)
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
    else:
        text = build_transcript_text(transcript)
        all_highlights = call_highlight_api(text, content_info, duration, **kwargs).get("highlights", [])

    all_highlights = snap_highlights(transcript, all_highlights)
    # Models often ignore the length rule; enforce it before ranking.
    hard_max = TARGET_LENGTHS.get((target_length or config.TARGET_LENGTH), TARGET_LENGTHS["medium"])[2]
    in_range = [h for h in all_highlights if MIN_CLIP_SECONDS <= h["end_time"] - h["start_time"] <= hard_max]
    if len(in_range) < len(all_highlights):
        print(f"[highlights] dropped {len(all_highlights) - len(in_range)} clips outside {MIN_CLIP_SECONDS}-{hard_max}s", flush=True)
    highlights = dedupe_highlights(in_range)
    if rerank:
        highlights = rerank_highlights(transcript, highlights, content_info, llm_fn)

    return {"highlights": highlights, "content_info": content_info}
