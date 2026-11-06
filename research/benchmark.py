"""Benchmark real Shorts: what do the top performers in your niche have in common?

Pulls *metadata and auto-captions only* (no video downloads) for Shorts from
channels/playlists you choose, scores each by engagement, and compares the
top decile with the rest: length, how the first line opens, how fast speech
starts. The top openings are written as JSONL you can feed back into clip
selection:

    python research/benchmark.py https://www.youtube.com/@SomeClipChannel/shorts \\
        https://www.youtube.com/@AnotherOne/shorts --max-per-source 60
    python main.py episode.mp4 --mode local --examples research/out/top.jsonl

Run this on your own machine (YouTube blocks most cloud/datacenter IPs).
Use --cookies-from-browser if YouTube asks you to sign in. Keep request
volumes modest and respect YouTube's Terms of Service.
"""
import argparse
import csv
import json
import re
import statistics
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

QUESTION_WORDS = {"what", "why", "how", "who", "when", "where", "which", "do", "does", "did",
                  "is", "are", "can", "could", "would", "should", "have", "has", "will"}
CONTRARIAN = {"nobody", "never", "stop", "wrong", "actually", "truth", "secret", "lie", "lied",
              "myth", "mistake", "worst", "best", "only", "insane", "crazy", "biggest"}
NUMBER = re.compile(r"\b\d[\d,.%$]*\b|\b(one|two|three|four|five|ten|hundred|thousand|million|billion)\b", re.I)


def _ydl(cookies_browser: Optional[str]):
    import yt_dlp  # type: ignore

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "sleep_interval_requests": 1,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    return yt_dlp.YoutubeDL(opts)


def list_videos(ydl, source: str, limit: int) -> List[str]:
    info = ydl.extract_info(source, download=False, process=False)
    urls = []
    for e in info.get("entries") or []:
        if len(urls) >= limit:
            break
        if not isinstance(e, dict):
            continue
        url = e.get("url") or e.get("webpage_url")
        if url and not url.startswith("http"):
            url = f"https://www.youtube.com/shorts/{e.get('id') or url}"
        if url:
            urls.append(url)
    if not urls and info.get("webpage_url"):
        urls.append(info["webpage_url"])   # a single video URL
    return urls


def _caption_words(ydl, info: Dict, max_seconds: float = 20) -> List[Dict]:
    """English auto (or manual) captions up to `max_seconds` as [{t, w}] via json3."""
    tracks = (info.get("subtitles") or {}).get("en") or (info.get("automatic_captions") or {}).get("en") or []
    url = next((t["url"] for t in tracks if t.get("ext") == "json3"), None)
    if not url:
        return []
    try:
        data = json.loads(ydl.urlopen(url).read().decode("utf-8"))
    except Exception:
        return []
    words = []
    for ev in data.get("events", []):
        t0 = ev.get("tStartMs", 0) / 1000
        if t0 > max_seconds:
            break
        for seg in ev.get("segs") or []:
            w = (seg.get("utf8") or "").strip()
            if w and w != "\n":
                words.append({"t": t0 + seg.get("tOffsetMs", 0) / 1000, "w": w})
    return words


def hook_features(words: List[Dict], title: str) -> Dict:
    first = words[:14]
    text = " ".join(w["w"] for w in first)
    tokens = [re.sub(r"[^\w']", "", t).lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    return {
        "hook": text,
        "first_word_s": round(words[0]["t"], 2) if words else None,
        "hook_is_question": bool(tokens and tokens[0] in QUESTION_WORDS) or "?" in text,
        "hook_has_number": bool(NUMBER.search(text)),
        "hook_says_you": "you" in tokens or "your" in tokens,
        "hook_contrarian": any(t in CONTRARIAN for t in tokens),
        "title_has_number": bool(NUMBER.search(title or "")),
    }


def fetch(ydl, url: str) -> Optional[Dict]:
    try:
        info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  skip {url}: {e.__class__.__name__}")
        return None
    views = info.get("view_count") or 0
    likes = info.get("like_count") or 0
    comments = info.get("comment_count") or 0
    up = info.get("upload_date")
    age_days = None
    if up:
        age_days = max(1, (date.today() - datetime.strptime(up, "%Y%m%d").date()).days)
    row = {
        "url": info.get("webpage_url") or url,
        "channel": info.get("channel") or info.get("uploader"),
        "title": info.get("title"),
        "duration": info.get("duration"),
        "upload_date": up,
        "views": views,
        "likes": likes,
        "comments": comments,
        "engagement_rate": round((likes + comments) / views, 4) if views else None,
        "views_per_day": round(views / age_days, 1) if (views and age_days) else None,
    }
    row.update(hook_features(_caption_words(ydl, info), info.get("title") or ""))
    return row


def _share(rows: List[Dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{100 * sum(bool(v) for v in vals) / len(vals):.0f}%" if vals else "n/a"


def _median(rows: List[Dict], key: str) -> str:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return f"{statistics.median(vals):.1f}" if vals else "n/a"


def summarize(rows: List[Dict], rank_key: str) -> str:
    ranked = sorted([r for r in rows if r.get(rank_key) is not None], key=lambda r: r[rank_key], reverse=True)
    if len(ranked) < 10:
        return f"Only {len(ranked)} usable videos; collect at least 30 for a meaningful comparison.\n"
    top = ranked[: max(3, len(ranked) // 10)]
    rest = ranked[len(top):]
    lines = [
        f"# Shorts benchmark ({len(ranked)} videos, ranked by {rank_key})",
        "",
        "| | top 10% | the rest |",
        "|---|---|---|",
        f"| median length (s) | {_median(top, 'duration')} | {_median(rest, 'duration')} |",
        f"| median time to first word (s) | {_median(top, 'first_word_s')} | {_median(rest, 'first_word_s')} |",
        f"| median engagement rate | {_median(top, 'engagement_rate')} | {_median(rest, 'engagement_rate')} |",
        f"| hook is a question | {_share(top, 'hook_is_question')} | {_share(rest, 'hook_is_question')} |",
        f"| hook has a number | {_share(top, 'hook_has_number')} | {_share(rest, 'hook_has_number')} |",
        f"| hook says \"you\" | {_share(top, 'hook_says_you')} | {_share(rest, 'hook_says_you')} |",
        f"| contrarian/superlative hook | {_share(top, 'hook_contrarian')} | {_share(rest, 'hook_contrarian')} |",
        "",
        "## Top openings",
        "",
    ]
    for r in top[:20]:
        lines.append(f"- {r[rank_key]:,} — \"{r.get('hook') or ''}\" ({r.get('duration')}s) {r['url']}")
    lines += ["", "Correlation, not causation: compare within one niche, and re-run as trends change."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="channel /shorts URLs, playlists, or single Shorts")
    ap.add_argument("--max-per-source", type=int, default=50)
    ap.add_argument("--rank-by", choices=["views_per_day", "views", "engagement_rate"], default="views_per_day")
    ap.add_argument("--out", default="research/out")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER", help="e.g. chrome, firefox (if YouTube asks to sign in)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    with _ydl(args.cookies_from_browser) as ydl:
        for src in args.sources:
            urls = list_videos(ydl, src, args.max_per_source)
            print(f"{src}: {len(urls)} videos")
            for u in urls:
                r = fetch(ydl, u)
                if r:
                    rows.append(r)
                time.sleep(0.5)

    if not rows:
        print("No videos fetched.")
        return
    with open(out / "shorts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, args.rank_by)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    ranked = sorted([r for r in rows if r.get(args.rank_by) is not None and r.get("hook")],
                    key=lambda r: r[args.rank_by], reverse=True)
    top = ranked[: max(3, len(ranked) // 10)]
    with open(out / "top.jsonl", "w", encoding="utf-8") as f:
        for r in top:
            f.write(json.dumps({"hook": r["hook"], "duration": r["duration"], "views": r["views"], "url": r["url"]}) + "\n")

    print(summary)
    print(f"Wrote {out / 'shorts.csv'}, {out / 'summary.md'}, {out / 'top.jsonl'}")


if __name__ == "__main__":
    main()
