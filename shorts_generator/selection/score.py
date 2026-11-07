"""Stage 3 (text only): six narrow questions per candidate, and the default combiner (Stage 4 cold start)."""
import statistics
from typing import Dict, List, Optional

from ..boundaries import all_words
from ..highlights import _clip_excerpt
from .cache import Ask, judge_prefix, load_prompt, pmap

_SCORE = {"type": "integer", "minimum": 1, "maximum": 5}

QUESTIONS = {
    "hook": {"type": "object", "properties": {"opening_line": {"type": "string"}, "score": _SCORE},
             "required": ["opening_line", "score"]},
    "payoff": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "partial", "no"]}},
               "required": ["answer"]},
    "emotion": {"type": "object", "properties": {"score": _SCORE}, "required": ["score"]},
    "standalone": {"type": "object", "properties": {"score": _SCORE}, "required": ["score"]},
    "novelty": {"type": "object",
                "properties": {"label": {"type": "string",
                                         "enum": ["funny", "surprising", "contrarian", "useful", "none"]},
                               "score": _SCORE},
                "required": ["label", "score"]},
    "ending": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
               "required": ["answer"]},
}
ANSWER_VALUES = {"payoff": {"yes": 1.0, "partial": 0.5, "no": 0.0}, "ending": {"yes": 1.0, "no": 0.0}}

# Spec Stage 4 cold start; `ending` is stored but not weighted.
DEFAULT_WEIGHTS = {"hook": 0.3, "standalone": 0.2, "payoff": 0.2, "emotion": 0.15, "novelty": 0.15}
STD_PENALTY = 0.5
EXIT_KEYS = ("hook", "standalone")
OPENING_SECONDS = 3.0


def normalise(key: str, answer: Optional[Dict]) -> Optional[float]:
    """An answer as 0-1: scores clamped to 1-5 then (score - 1) / 4; yes/partial/no as 1/0.5/0."""
    if not isinstance(answer, dict):
        return None
    if key in ANSWER_VALUES:
        return ANSWER_VALUES[key].get(str(answer.get("answer") or "").strip().lower())
    try:
        score = int(float(answer.get("score")))
    except (TypeError, ValueError, OverflowError):
        return None
    return (max(1, min(5, score)) - 1) / 4


def combine(answers: Dict[str, List[float]]) -> float:
    """Weighted mean of the answered questions, minus 0.5 x their mean spread across samples."""
    present = {k: v for k, v in answers.items() if k in DEFAULT_WEIGHTS and v}
    if not present:
        return 0.0
    total = sum(DEFAULT_WEIGHTS[k] for k in present)
    mean = sum(DEFAULT_WEIGHTS[k] * statistics.fmean(v) for k, v in present.items()) / total
    spread = statistics.fmean(statistics.pstdev(v) for v in present.values())
    return mean - STD_PENALTY * spread


def score_candidates(transcript: Dict, bible: Dict, cands: List[Dict], ask: Ask, samples: int) -> List[Dict]:
    """Ask every question about every candidate; candidates scoring 1 on both hook and standalone stop early.

    Returns the survivors, each with `answers` (0-1 per sample), `extras` and `combined`.
    """
    prefix = judge_prefix(bible)
    words = all_words(transcript)
    prompts = {k: load_prompt(k) for k in QUESTIONS}
    texts = {}
    for c in cands:
        s, e = float(c["start_time"]), float(c["end_time"])
        opening = " ".join(w["word"].strip() for w in words if s <= w["start"] < s + OPENING_SECONDS)
        texts[c["id"]] = {"clip": _clip_excerpt(transcript, s, e, head=4000, tail=1000), "opening": opening}
    raw: Dict[str, Dict[str, List[Optional[Dict]]]] = {c["id"]: {k: [] for k in QUESTIONS} for c in cands}

    def run(jobs):
        def one(job):
            cid, key, sample = job
            template, prompt_id = prompts[key]
            return ask(prefix + "\n\n" + template.format(**texts[cid]), QUESTIONS[key], prompt_id, sample)
        for (cid, key, _), answer in zip(jobs, pmap(one, jobs)):
            raw[cid][key].append(answer)

    run([(c["id"], k, 0) for c in cands for k in EXIT_KEYS])
    survivors = [c for c in cands
                 if not all(normalise(k, raw[c["id"]][k][0]) == 0.0 for k in EXIT_KEYS)]
    print(f"[deep] early exit dropped {len(cands) - len(survivors)} of {len(cands)} candidates", flush=True)
    run([(c["id"], k, s) for c in survivors for k in QUESTIONS for s in range(samples)
         if not (k in EXIT_KEYS and s == 0)])

    out = []
    for c in survivors:
        r = raw[c["id"]]
        answers = {k: [v for v in (normalise(k, a) for a in r[k]) if v is not None] for k in QUESTIONS}
        hook = next((a for a in r["hook"] if isinstance(a, dict)), {})
        novelty = next((a for a in r["novelty"] if isinstance(a, dict)), {})
        extras = {"hook_quote": str(hook.get("opening_line") or ""), "novelty_label": novelty.get("label")}
        out.append({**c, "answers": answers, "extras": extras, "combined": combine(answers)})
    return out
