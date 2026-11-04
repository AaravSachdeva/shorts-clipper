"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \\
        --num-clips 3 --aspect-ratio 9:16

    # fully offline on a modest machine (Ollama + CPU Whisper):
    python main.py /path/to/episode.mp4 --mode local --profile cpu-slow
"""
import argparse
import json
import os
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Settings bundles. Applied as environment defaults *before* the package is
# imported (config is read at import time); explicit env vars and CLI flags win.
PROFILES = {
    # ~2 GB VRAM / 16 GB RAM, time is not a constraint (one long video a day).
    "cpu-slow": {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_MODEL": "qwen3:8b",
        "OLLAMA_NUM_CTX": "16384",
        "HIGHLIGHT_CHUNK_SECONDS": "600",
        "LOCAL_WHISPER_MODEL": "small",
        "LOCAL_WHISPER_DEVICE": "cpu",
    },
    # Same, but a ~14B model for better picks (close other apps; ~10-12 GB RAM).
    "cpu-quality": {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_MODEL": "qwen3:14b",
        "OLLAMA_NUM_CTX": "16384",
        "HIGHLIGHT_CHUNK_SECONDS": "600",
        "LOCAL_WHISPER_MODEL": "medium",
        "LOCAL_WHISPER_DEVICE": "cpu",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (yt-dlp/local file + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="Apply a bundle of local-mode settings (see README)")
    parser.add_argument("--llm-provider", choices=["openai", "gemini", "ollama"], help="Local-mode LLM (overrides LLM_PROVIDER)")
    parser.add_argument("--target-length", choices=["short", "medium", "long"], help="Clip length: short ≈15-35s, medium ≈20-60s (default), long ≈45-90s")
    parser.add_argument("--no-captions", action="store_true", help="Don't burn in word-highlight captions")
    parser.add_argument("--no-hook", action="store_true", help="Don't show the on-screen hook title for the first 3s")
    parser.add_argument("--crop-bottom", type=float, metavar="PCT", help="Cut this %% of frame height off the bottom first (burned-in tickers/chat)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the global comparative re-rank of candidates")
    parser.add_argument("--examples", metavar="JSONL", help="Top-clip openings to show the LLM (from research/benchmark.py)")
    return parser.parse_args()


def _apply_settings(args: argparse.Namespace) -> None:
    if args.profile:
        for k, v in PROFILES[args.profile].items():
            os.environ.setdefault(k, v)
    flag_env = {
        "LLM_PROVIDER": args.llm_provider,
        "TARGET_LENGTH": args.target_length,
        "CAPTIONS": "false" if args.no_captions else None,
        "HOOK_OVERLAY": "false" if args.no_hook else None,
        "CROP_BOTTOM_PCT": None if args.crop_bottom is None else str(args.crop_bottom),
        "HIGHLIGHT_RERANK": "false" if args.no_rerank else None,
        "HIGHLIGHT_EXAMPLES": args.examples,
    }
    for k, v in flag_env.items():
        if v is not None:
            os.environ[k] = v


def main() -> int:
    args = _parse_args()
    _apply_settings(args)

    from shorts_generator import generate_shorts

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
    print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:  {(s.get('post') or {}).get('title') or s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
