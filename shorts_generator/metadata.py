"""Post metadata for each rendered short: title, caption, hashtags.

One batched LLM call covers every clip (cheaper on slow local models), then a
sidecar short_XX.json (everything) and short_XX.txt (ready to paste) are
written next to each mp4.
"""
import json
import os
from typing import Dict, List

from .highlights import LLMFn, _clip_excerpt, _invoke, _parse_json_loose

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "caption": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "caption", "hashtags"],
            },
        }
    },
    "required": ["clips"],
}

METADATA_PROMPT = """You write posts for YouTube Shorts and Instagram Reels. For each clip below, write:
- title: under 60 characters, curiosity-driven, no clickbait lies, no emojis
- caption: 1-2 short lines that add context or ask a question viewers will want to answer in the comments
- hashtags: 3-5 relevant hashtags (lowercase, no spaces, include the # sign)

{clips}

Respond ONLY with valid JSON: {{"clips":[{{"id":int,"title":"string","caption":"string","hashtags":["#tag"]}}]}}"""


def generate_metadata(transcript: Dict, highlights: List[Dict], llm_fn: LLMFn) -> List[Dict]:
    """Return highlights with a `post` dict added. Failures leave `post` derived from the highlight."""
    blocks = []
    for i, h in enumerate(highlights, 1):
        blocks.append(
            f'[{i}] working title: "{h.get("title", "")}"\n'
            f"{_clip_excerpt(transcript, float(h['start_time']), float(h['end_time']), head=500, tail=150)}"
        )
    posts: Dict[int, Dict] = {}
    try:
        parsed = _parse_json_loose(_invoke(llm_fn, METADATA_PROMPT.format(clips="\n\n".join(blocks)), METADATA_SCHEMA))
        for c in parsed.get("clips", []):
            if isinstance(c, dict) and "id" in c:
                tags = [str(t).strip() for t in (c.get("hashtags") or []) if str(t).strip()]
                tags = [t if t.startswith("#") else f"#{t}" for t in tags][:5]
                posts[int(c["id"])] = {
                    "title": str(c.get("title") or "").strip()[:100],
                    "caption": str(c.get("caption") or "").strip(),
                    "hashtags": tags,
                }
    except Exception as e:
        print(f"[metadata] generation failed ({e}); using highlight titles", flush=True)

    out = []
    for i, h in enumerate(highlights, 1):
        post = posts.get(i) or {"title": h.get("title", ""), "caption": h.get("hook_sentence", ""), "hashtags": []}
        out.append({**h, "post": post})
    return out


def write_sidecars(shorts: List[Dict]) -> None:
    for s in shorts:
        path = s.get("clip_url")
        if not path:
            continue
        stem = os.path.splitext(path)[0]
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        post = s.get("post") or {}
        with open(stem + ".txt", "w", encoding="utf-8") as f:
            f.write(f"{post.get('title', '')}\n\n{post.get('caption', '')}\n\n{' '.join(post.get('hashtags', []))}\n")
