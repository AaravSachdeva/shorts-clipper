"""Evaluate clip selectors against labelled episodes (eval/<episode>/gold.json).

Gold file: {"episode", "source", "language", "split": "tune"|"test",
            "gold": [{"start", "end", "rating", "origin", "views", "url", "match_ratio"}]}
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MATCH_IOU = 0.5


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
