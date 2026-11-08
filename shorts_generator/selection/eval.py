"""Evaluate clip selectors against labelled episodes (eval/<episode>/gold.json).

    python -m shorts_generator.selection.eval --selectors random,signal,classic --split all

Gold file: {"episode", "source", "language", "split": "tune"|"test",
            "gold": [{"start", "end", "rating", "origin", "views", "url", "match_ratio"}]}
"""
import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config
from ..boundaries import all_words
from ..highlights import get_highlights
from ..local.llm import call_local_llm
from . import run_deep
from .cache import cached_llm
from .candidates import _span, iou, signal_spans
from .loudness import loud_peaks, loudness_per_second

MATCH_IOU = 0.5
TOP_N = 10
MIN_EPISODES_FOR_CONCLUSIONS = 10


def match(picks: Sequence, gold: Sequence, k: int) -> List[Tuple[int, int, float]]:
    """Greedy in pick order: each of the top-k picks takes the unused gold clip with the highest IoU >= 0.5."""
    used = set()
    out = []
    for i, p in enumerate(picks[:k]):
        best = max(((iou(p, g), j) for j, g in enumerate(gold) if j not in used), default=(0.0, -1))
        if best[0] >= MATCH_IOU:
            used.add(best[1])
            out.append((i, best[1], best[0]))
    return out


def precision_at(picks: Sequence, gold: Sequence, k: int = 5) -> float:
    return len(match(picks, gold, k)) / k


def ndcg_at(picks: Sequence, gold: Sequence, k: int = 10) -> float:
    """NDCG with the gold rating as a linear gain."""
    rel = {i: float(gold[j].get("rating") or 0) for i, j, _ in match(picks, gold, k)}
    dcg = sum(r / math.log2(i + 2) for i, r in rel.items())
    ideal = sorted((float(g.get("rating") or 0) for g in gold), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall(cands: Sequence, gold: Sequence) -> float:
    """Share of gold clips that any candidate overlaps at IoU >= 0.5."""
    if not gold:
        return 0.0
    return sum(any(iou(c, g) >= MATCH_IOU for c in cands) for g in gold) / len(gold)


def boundary_error(picks: Sequence, gold: Sequence, k: int = 5) -> Optional[float]:
    """Mean seconds between matched start/end and gold, or None when nothing matched."""
    errs = []
    for i, j, _ in match(picks, gold, k):
        (ps, pe), (gs, ge) = _span(picks[i]), _span(gold[j])
        errs.append((abs(ps - gs) + abs(pe - ge)) / 2)
    return sum(errs) / len(errs) if errs else None


def jaccard_top(a: Sequence, b: Sequence, k: int = 5) -> float:
    """Overlap of two top-k pick lists, where "same clip" means IoU >= 0.5."""
    a, b = list(a[:k]), list(b[:k])
    if not a and not b:
        return 1.0
    m = len(match(a, b, k))
    return m / (len(a) + len(b) - m)


def random_spans(duration: float, n: int = 10, seed: str = "") -> List[Dict]:
    """Baseline: n random spans of 20-60 s inside the episode."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = min(rng.uniform(20, 60), duration)
        start = rng.uniform(0, max(0.0, duration - length))
        out.append({"start_time": round(start, 2), "end_time": round(start + length, 2)})
    return out


def load_episodes(eval_dir: Path, split: str) -> List[Dict]:
    """Labelled episodes for a split (tune, test or all); episodes with no gold are skipped."""
    out = []
    for path in sorted(Path(eval_dir).glob("*/gold.json")):
        ep = json.loads(path.read_text(encoding="utf-8"))
        if split != "all" and ep.get("split") != split:
            continue
        if not ep.get("gold"):
            print(f"[eval] skipping {ep.get('episode') or path.parent.name}: no gold clips", flush=True)
            continue
        out.append(ep)
    return out


# --- running selectors --------------------------------------------------------

def _work(ep: Dict) -> Path:
    return Path(config.WORK_DIR) / Path(ep["source"]).stem


def run_selector(name: str, ep: Dict, transcript: Dict, run: int = 0,
                 skip: Iterable[str] = (), samples: Optional[int] = None) -> Dict:
    """{"picks": top-10 spans, "candidates": every span the selector considered, "stats": LLM call stats}."""
    stats: Dict = {}
    duration = float(transcript.get("duration") or 0)
    if name == "random":
        picks = random_spans(duration, n=TOP_N, seed=f"{ep['episode']}:{run}")
        return {"picks": picks, "candidates": picks, "stats": stats}
    if name == "signal":
        db = loudness_per_second(ep["source"], _work(ep) / "loudness.json")
        spans = signal_spans(all_words(transcript), [(float(t), "loudness", d) for t, d in loud_peaks(db)], duration)
        spans.sort(key=lambda s: s["strength"], reverse=True)
        return {"picks": spans[:TOP_N], "candidates": spans, "stats": stats}
    if name == "classic":
        fn = cached_llm(call_local_llm, _work(ep) / "calls", "classic", run, stats)
        highlights = get_highlights(transcript, num_clips=TOP_N, llm_fn=fn)["highlights"]
        return {"picks": highlights[:TOP_N], "candidates": highlights, "stats": stats}
    if name == "deep":
        res = run_deep(transcript, ep["source"], TOP_N, llm_fn=call_local_llm, skip=skip, samples=samples,
                       run=run, stats=stats)
        return {"picks": res["highlights"], "candidates": res["candidates"], "stats": stats}
    raise ValueError(f"unknown selector {name!r}")


def selector_label(name: str, skip: Iterable[str], samples: Optional[int]) -> str:
    """Report row name, e.g. deep[-bible,samples=3]; only deep takes options."""
    if name != "deep":
        return name
    parts = [f"-{s}" for s in skip] + ([f"samples={samples}"] if samples else [])
    return f"deep[{','.join(parts)}]" if parts else "deep"


def evaluate(episodes: List[Dict], selectors: List[str], runs: int = 1,
             skip: Iterable[str] = (), samples: Optional[int] = None) -> List[Dict]:
    """One metrics row per (selector, episode). Metrics come from run 0; stability needs runs >= 2."""
    from concurrent.futures import ThreadPoolExecutor

    from ..local.transcriber import transcribe_local

    def one_episode(ep: Dict) -> List[Dict]:
        rows: List[Dict] = []
        transcript = transcribe_local(ep["source"], language=ep.get("language"))
        if not all_words(transcript):
            print(f"[eval] skipping {ep['episode']}: transcript has no word timings", flush=True)
            return rows
        gold = ep["gold"]
        for sel in selectors:
            results = [run_selector(sel, ep, transcript, run=r, skip=skip, samples=samples) for r in range(runs)]
            first = results[0]
            pairs = list(itertools.combinations([r["picks"] for r in results], 2))
            rows.append({
                "selector": selector_label(sel, skip, samples), "episode": ep["episode"],
                "recall": recall(first["candidates"], gold),
                "p5": precision_at(first["picks"], gold, 5),
                "ndcg10": ndcg_at(first["picks"], gold, 10),
                "boundary": boundary_error(first["picks"], gold, 5),
                "stability": sum(jaccard_top(a, b, 5) for a, b in pairs) / len(pairs) if pairs else None,
                "calls": sum(r["stats"].get("calls", 0) for r in results),
                "llm_seconds": sum(r["stats"].get("seconds", 0.0) for r in results),
            })
            print(f"[eval] {sel} on {ep['episode']}: P@5 {rows[-1]['p5']:.2f}", flush=True)
        return rows

    # Episodes run side by side (classic asks window by window); the call cache caps live calls at LLM_WORKERS.
    with ThreadPoolExecutor(max_workers=max(1, len(episodes))) as pool:
        return [row for rows in pool.map(one_episode, episodes) for row in rows]


def merge_rows(old: List[Dict], new: List[Dict]) -> List[Dict]:
    """Earlier rows plus this run's; a new row replaces an old one with the same selector label and episode."""
    keys = {(r["selector"], r["episode"]) for r in new}
    return [r for r in old if (r["selector"], r["episode"]) not in keys] + new


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: Optional[float], nd: int = 2) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def report(rows: List[Dict]) -> str:
    """Markdown: one summary row per selector, then one row per selector and episode."""
    n_eps = len({r["episode"] for r in rows})
    out = ["# Clip selector eval", "",
           f"Episodes: {n_eps} (conclusions about a stage need at least {MIN_EPISODES_FOR_CONCLUSIONS})", "",
           "| selector | episodes | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | stability | LLM calls | LLM hours |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for sel in dict.fromkeys(r["selector"] for r in rows):
        rs = [r for r in rows if r["selector"] == sel]
        out.append(
            f"| {sel} | {len(rs)} | {_fmt(_mean(r['recall'] for r in rs))} | {_fmt(_mean(r['p5'] for r in rs))} "
            f"| {_fmt(_mean(r['ndcg10'] for r in rs))} | {_fmt(_mean(r['boundary'] for r in rs), 1)} "
            f"| {_fmt(_mean(r['stability'] for r in rs))} | {sum(r['calls'] for r in rs)} "
            f"| {_fmt(sum(r['llm_seconds'] for r in rs) / 3600)} |"
        )
    out += ["", "## Per episode", "",
            "| selector | episode | Recall@candidates | P@5 | NDCG@10 | boundary error (s) | LLM calls |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['selector']} | {r['episode']} | {_fmt(r['recall'])} | {_fmt(r['p5'])} "
                   f"| {_fmt(r['ndcg10'])} | {_fmt(r['boundary'], 1)} | {r['calls']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score clip selectors on labelled episodes")
    ap.add_argument("--selectors", default="random,signal,classic")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="test")
    ap.add_argument("--runs", type=int, default=1, help=">= 2 also measures stability (each run re-asks the model)")
    ap.add_argument("--skip", default="", help="deep stages to skip, e.g. bible,tournament")
    ap.add_argument("--samples", type=int, default=None, help="answers per deep scoring question")
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--out", default="eval/report.md")
    args = ap.parse_args()

    episodes = load_episodes(Path(args.eval_dir), args.split)
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    # Rows from earlier runs are kept, so deep and its ablations (deep[-bible], ...) share one report.
    rows_path = Path(args.out).with_name("rows.json")
    old = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    rows = merge_rows(old, evaluate(episodes, selectors, runs=args.runs, skip=skip, samples=args.samples))
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = report(rows) if rows else "# Clip selector eval\n\nEpisodes: 0 (no labelled episodes in this split)\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
