"""Stage 6: order the best candidates by head-to-head comparisons, then pick a varied final set."""
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from ..highlights import HIGHLIGHT_SCHEMA, _clip_excerpt
from .bible import chapter_of
from .cache import Ask, judge_prefix, load_prompt, pmap

BT_WEIGHT = 0.6     # final = 0.6 x Bradley-Terry rank + 0.4 x combiner rank
PSEUDO_WINS = 0.1   # per opponent, so a winless clip still gets a strength above 0

PAIR_SCHEMA = {"type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B"]}},
               "required": ["winner"]}
TEXT_FIELDS = ["title", "hook_sentence", "on_screen_hook", "virality_reason"]
_ITEM = HIGHLIGHT_SCHEMA["properties"]["highlights"]["items"]["properties"]
TEXT_SCHEMA = {"type": "object", "properties": {k: _ITEM[k] for k in TEXT_FIELDS}, "required": TEXT_FIELDS}


def bradley_terry(n: int, wins: List[Tuple[int, int]], ties: List[Tuple[int, int]], iters: int = 200) -> List[float]:
    """Strengths (summing to 1) from (winner, loser) pairs and ties, by the MM algorithm. A tie is half a win each."""
    if n == 0:
        return []
    w = [[PSEUDO_WINS if i != j else 0.0 for j in range(n)] for i in range(n)]
    for a, b in wins:
        w[a][b] += 1.0
    for a, b in ties:
        w[a][b] += 0.5
        w[b][a] += 0.5
    total = [sum(row) for row in w]
    p = [1.0 / n] * n
    for _ in range(iters):
        p = [total[i] / sum((w[i][j] + w[j][i]) / (p[i] + p[j]) for j in range(n) if j != i) if n > 1 else 1.0
             for i in range(n)]
        s = sum(p)
        p = [x / s for x in p]
    return p


def resolve(first: Optional[str], second: Optional[str]) -> Optional[int]:
    """Clip i shown as A, then as B: 0 if i won both times, 1 if j did, None (a tie) otherwise."""
    if first == "A" and second == "B":
        return 0
    if first == "B" and second == "A":
        return 1
    return None


def play(cands: List[Dict], transcript: Dict, bible: Dict, ask: Ask) -> List[float]:
    """Bradley-Terry strength of each candidate from every pair, asked in both orders."""
    n = len(cands)
    if n < 2:
        return bradley_terry(n, [], [])
    template, prompt_id = load_prompt("pair")
    prefix = judge_prefix(bible)
    texts = [_clip_excerpt(transcript, float(c["start_time"]), float(c["end_time"]), head=2000, tail=600)
             for c in cands]
    pairs = list(itertools.combinations(range(n), 2))
    jobs = [(a, b) for i, j in pairs for a, b in ((i, j), (j, i))]

    def one(job):
        a, b = job
        answer = ask(prefix + "\n\n" + template.format(a=texts[a], b=texts[b]), PAIR_SCHEMA, prompt_id)
        return (answer or {}).get("winner")

    answers = pmap(one, jobs)
    wins, ties = [], []
    for k, (i, j) in enumerate(pairs):
        r = resolve(answers[2 * k], answers[2 * k + 1])
        if r is None:
            ties.append((i, j))
        else:
            wins.append((i, j) if r == 0 else (j, i))
    print(f"[deep] tournament: {len(pairs)} pairs, {len(ties)} ties", flush=True)
    return bradley_terry(n, wins, ties)


def _rank_scores(scores: Sequence[float]) -> List[float]:
    n = len(scores)
    if n == 1:
        return [1.0]
    out = [0.0] * n
    for rank, i in enumerate(sorted(range(n), key=lambda i: scores[i], reverse=True)):
        out[i] = 1 - rank / (n - 1)
    return out


def blend(bt: Sequence[float], comb: Sequence[float], w_bt: float = BT_WEIGHT) -> List[float]:
    """w_bt x Bradley-Terry rank score + (1 - w_bt) x combiner rank score; rank score 1 = best, 0 = worst."""
    return [w_bt * a + (1 - w_bt) * b for a, b in zip(_rank_scores(bt), _rank_scores(comb))]


def mmr(items: List[Dict], k: int, lam: float, chapters: List[Dict]) -> List[Dict]:
    """Pick k items one at a time, each maximising final - lam x (1 if it shares a bible chapter with a pick)."""
    chap = [chapter_of((float(x["start_time"]) + float(x["end_time"])) / 2, chapters) for x in items]
    left = list(range(len(items)))
    picked: List[int] = []
    while left and len(picked) < k:
        def value(i):
            same = any(chap[i] is not None and chap[i] == chap[j] for j in picked)
            return items[i]["final"] - lam * (1.0 if same else 0.0)
        best = max(left, key=value)
        picked.append(best)
        left.remove(best)
    return [items[i] for i in picked]


def text_fields(pick: Dict, transcript: Dict, bible: Dict, ask: Ask) -> Dict:
    """title, hook_sentence, on_screen_hook and virality_reason for one picked clip."""
    template, prompt_id = load_prompt("text_fields")
    clip = _clip_excerpt(transcript, float(pick["start_time"]), float(pick["end_time"]), head=4000, tail=1000)
    answer = ask(judge_prefix(bible) + "\n\n" + template.format(clip=clip), TEXT_SCHEMA, prompt_id) or {}
    out = {k: str(answer.get(k) or "").strip() for k in TEXT_FIELDS}
    out["title"] = out["title"] or "Untitled Highlight"
    return out
