"""Stand-in LLM harness for testing the repo's pipeline without an API key.

Each LLM call is keyed by a hash of the exact prompt the repo builds.
- If harness/answers/<hash>.txt exists, it's returned as the model response.
- Otherwise the prompt is saved to harness/prompts/<hash>.txt and the run
  stops (SystemExit, which the repo's `except Exception` won't swallow).
Answer the saved prompt, re-run, repeat until every call is answered.

Usage:
  python harness/run.py select   [--source input/episode.mp4]
  python harness/run.py render N [--source input/episode.mp4]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shorts_generator.highlights import get_highlights  # noqa: E402
from shorts_generator.local.transcriber import transcribe_local  # noqa: E402
from shorts_generator.metadata import generate_metadata, write_sidecars  # noqa: E402

H = ROOT / "harness"
(H / "prompts").mkdir(exist_ok=True)
(H / "answers").mkdir(exist_ok=True)


def replay_llm(prompt: str, schema=None) -> str:
    key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    ans = H / "answers" / f"{key}.txt"
    if ans.exists():
        print(f"[harness] answered {key}", flush=True)
        return ans.read_text()
    (H / "prompts" / f"{key}.txt").write_text(prompt)
    print(f"[harness] NEED ANSWER for prompt {key} ({len(prompt)} chars)", flush=True)
    raise SystemExit(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "render"])
    ap.add_argument("n", nargs="?", type=int, default=3)
    ap.add_argument("--source", default=str(ROOT / "input" / "cordkillers377.mp4"))
    args = ap.parse_args()

    transcript = transcribe_local(args.source, language="en")
    result = get_highlights(transcript, num_clips=args.n, llm_fn=replay_llm)
    hl = result["highlights"]
    (H / "highlights.json").write_text(json.dumps(hl, indent=2))
    for h in hl:
        print(f"{h['score']:3d}  {h['start_time']:7.1f}-{h['end_time']:7.1f}  {h['title']}")
    if args.mode == "render":
        from shorts_generator.local.clipper import crop_highlights_local

        top = sorted(hl, key=lambda h: int(h.get("score", 0)), reverse=True)[: args.n]
        top = generate_metadata(transcript, top, replay_llm)
        shorts = crop_highlights_local(args.source, top, aspect_ratio="9:16", transcript=transcript)
        write_sidecars(shorts)
        (H / "shorts.json").write_text(json.dumps(shorts, indent=2))
        print(json.dumps([{k: s.get(k) for k in ("title", "clip_url", "error")} for s in shorts], indent=2))


if __name__ == "__main__":
    main()
