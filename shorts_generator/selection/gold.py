"""Build eval gold clips from a channel's official Shorts, matched to episodes by transcript text.

    python -m shorts_generator.selection.gold --shorts https://www.youtube.com/@channel/shorts \\
        --episode output/source_abc.mp4 --episode "https://www.youtube.com/watch?v=..."

Writes eval/<episode>/gold.json. Entries with any other origin (hand picks) are kept on reruns.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..boundaries import _norm, all_words

MIN_BLOCK = 3            # matching runs shorter than this are coincidence
NEAR_ANCHOR_SECONDS = 90  # a Short is one moment: ignore matches far from its main run
MIN_RATIO = 0.5
OFFICIAL_RATING = 4


def match_short(ep_words: List[Dict], short_tokens: List[str]) -> Optional[Dict]:
    """Where a Short's caption text sits in an episode: {"start", "end", "match_ratio"}, or None."""
    idx = [i for i, w in enumerate(ep_words) if _norm(w["word"])]
    ep = [_norm(ep_words[i]["word"]) for i in idx]
    sh = [t for t in (_norm(x) for x in short_tokens) if t]
    if not ep or not sh:
        return None
    blocks = [b for b in difflib.SequenceMatcher(None, ep, sh, autojunk=False).get_matching_blocks()
              if b.size >= MIN_BLOCK]
    if not blocks:
        return None
    anchor_t = ep_words[idx[max(blocks, key=lambda b: b.size).a]]["start"]
    keep = [b for b in blocks if abs(ep_words[idx[b.a]]["start"] - anchor_t) <= NEAR_ANCHOR_SECONDS]
    ratio = sum(b.size for b in keep) / len(sh)
    if ratio < MIN_RATIO:
        return None
    start = min(ep_words[idx[b.a]]["start"] for b in keep)
    end = max(ep_words[idx[b.a + b.size - 1]]["end"] for b in keep)
    return {"start": round(start, 3), "end": round(end, 3), "match_ratio": round(ratio, 3)}


def merge_gold(old: List[Dict], official: List[Dict]) -> List[Dict]:
    """New official-Short matches replace old ones; every other entry (hand picks) is kept."""
    return [g for g in old if g.get("origin") != "official_short"] + official


def _split_for(name: str) -> str:
    return "test" if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10 < 3 else "tune"


def episode_gold(name: str, source: str, ep_words: List[Dict], shorts: List[Dict],
                 old: Dict, language: Optional[str]) -> Dict:
    """The gold.json content for one episode."""
    official = []
    for s in shorts:
        m = match_short(ep_words, s["tokens"])
        if m:
            official.append({**m, "rating": OFFICIAL_RATING, "origin": "official_short",
                             "views": s.get("views") or 0, "url": s.get("url")})
    print(f"[gold] {name}: {len(official)} official Shorts matched", flush=True)
    if not official:
        print(f"[gold] WARNING: no official Shorts matched {name}; add hand picks or check the channel", flush=True)
    return {"episode": name, "source": source, "language": language,
            "split": old.get("split") or _split_for(name),
            "gold": merge_gold(old.get("gold") or [], official)}


def _short_entry(ydl, url: str) -> Dict:
    """A Short's spoken words (its audio through the same Whisper the episodes use), views and URL.

    YouTube caption downloads get throttled (HTTP 429) after a few hundred requests; audio downloads do not,
    and Whisper-to-Whisper text matches more tightly than auto-captions.
    """
    from ..local.transcriber import transcribe_local

    info = ydl.extract_info(url, download=True)
    words = all_words(transcribe_local(ydl.prepare_filename(info), language="en"))
    return {"url": info.get("webpage_url") or url, "title": info.get("title"),
            "views": info.get("view_count") or 0, "tokens": [w["word"].strip() for w in words]}


def _fetch_shorts(shorts_url: str, limit: int, cache_dir: Path) -> List[Dict]:
    """Spoken words, views and URL for a channel's Shorts; cached per video so reruns skip the network."""
    import yt_dlp  # type: ignore
    from research.benchmark import _ydl, list_videos

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestaudio",
                              "outtmpl": str(cache_dir / "audio" / "%(id)s.%(ext)s"), "sleep_interval_requests": 1})
    out = []
    for url in list_videos(_ydl(None), shorts_url, limit):
        path = cache_dir / f"{url.rstrip('/').split('/')[-1].split('=')[-1]}.json"
        if path.exists():
            out.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        try:
            entry = _short_entry(audio, url)
        except Exception as e:  # a failed download is skipped, never cached, so a rerun retries it
            print(f"[gold] skip {url}: {e.__class__.__name__}: {str(e)[:120]}", flush=True)
            continue
        path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        out.append(entry)
    print(f"[gold] {len(out)} Shorts read from {shorts_url}", flush=True)
    return [s for s in out if s["tokens"]]


def main() -> None:
    from ..local.downloader import download_youtube_local
    from ..local.transcriber import transcribe_local

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shorts", required=True, help="the channel's Shorts tab URL")
    ap.add_argument("--episode", action="append", required=True, help="episode URL or local path (repeatable)")
    ap.add_argument("--limit", type=int, default=300, help="how many of the channel's Shorts to check")
    ap.add_argument("--language", default=None, help="Whisper language for episodes not yet transcribed")
    ap.add_argument("--eval-dir", default="eval")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    shorts = _fetch_shorts(args.shorts, args.limit, eval_dir / "_shorts_cache")
    for ep in args.episode:
        source = str(Path(download_youtube_local(ep)).resolve())
        transcript = transcribe_local(source, language=args.language)
        name = Path(source).stem
        path = eval_dir / name / "gold.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data = episode_gold(name, source, all_words(transcript), shorts, old,
                            args.language or transcript.get("language"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        for g in data["gold"]:
            print(f"  {g['start']:8.1f}-{g['end']:8.1f}  {g.get('origin')}  {g.get('match_ratio', '')}  {g.get('url', '')}")


if __name__ == "__main__":
    main()
