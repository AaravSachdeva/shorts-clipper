"""Deep clip selector (--selector deep): bible, candidates, narrow questions, tournament, variety.

run_deep returns the same highlight dicts as highlights.get_highlights, so metadata and rendering
need no changes. Every LLM answer is cached in work/<episode>/calls/; a rerun resumes for free.
"""
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from .. import config
from .bible import build_bible
from .cache import make_ask, pmap
from .candidates import generate_candidates
from .loudness import loudness_per_second
from .score import score_candidates
from .tournament import blend, mmr, play, rank_scores, text_fields


def run_deep(transcript: Dict, source_path: str, num_clips: int, llm_fn: Optional[Callable] = None,
             skip: Iterable[str] = (), samples: Optional[int] = None, run: int = 0,
             target_length: Optional[str] = None, stats: Optional[Dict] = None) -> Dict:
    """{"highlights": picks as highlight dicts, "candidates": every merged candidate, "bible": the bible}."""
    if llm_fn is None:
        from ..local.llm import call_local_llm
        llm_fn = call_local_llm
    skip = set(skip)
    work = Path(config.WORK_DIR) / Path(source_path).stem
    ask = make_ask(llm_fn, work / "calls", run=run, stats=stats)

    bible = {} if "bible" in skip else build_bible(transcript, ask)
    db = loudness_per_second(source_path, work / "loudness.json")
    cands = generate_candidates(transcript, bible, db, ask, target_length)
    scored = score_candidates(transcript, bible, cands, ask, samples or config.SCORE_SAMPLES)
    top = sorted(scored, key=lambda c: c["combined"], reverse=True)[: config.TOURNAMENT_MAX]
    comb = [c["combined"] for c in top]
    if "tournament" in skip:
        bt, finals = [None] * len(top), rank_scores(comb)
    else:
        bt = play(top, transcript, bible, ask)
        finals = blend(bt, comb)
    for c, b, f in zip(top, bt, finals):
        c["bt_strength"], c["final"] = b, f

    picks = mmr(top, num_clips, config.MMR_LAMBDA, bible.get("chapters") or [])
    texts = pmap(lambda p: text_fields(p, transcript, bible, ask), picks)
    highlights = [{
        **t,
        "start_time": p["start_time"], "end_time": p["end_time"],
        "score": round(100 * p["final"]),
        "bt_strength": p["bt_strength"], "combiner_score": round(p["combined"], 4),
        "sub_scores": p["answers"], "origin": p["origin"], "kind": p.get("kind"),
    } for p, t in zip(picks, texts)]

    work.mkdir(parents=True, exist_ok=True)
    (work / "deep.json").write_text(json.dumps(
        {"bible": bible, "candidates": cands, "scored": scored, "picks": highlights}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[deep] picked {len(highlights)} of {len(cands)} candidates; details in {work / 'deep.json'}", flush=True)
    return {"highlights": highlights, "candidates": cands, "bible": bible}
