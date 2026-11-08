# Eval Execution Plan: deep vs classic, run once

## Context

The test-split eval was stopped and restarted three times, and each restart re-asked paid `claude -p` calls:
- deep picked 23 s clips against 70 s gold;
- the eval ran at the wrong target length;
- 16 workers drained usage too fast.

This plan fixes everything known **before** any paid call. It proves the fixes on a small tune-split pilot. The full 9-episode test run then happens once, from frozen code, and only resumes after usage limits; it is never restarted.

**User decisions (binding):**
- No quality cuts: the deep workflow runs exactly as built (spec settings). The only changes are bug fixes and changes that don't affect answers.
- Pilot on 2 **tune** episodes, so the test set stays clean.
- `LLM_WORKERS=2`.
- Thinking off, as already decided in `PLAN.md:11`.

**Facts measured so far:**
- Gold Shorts: median 70 s (test p25 43 s, p75 90 s), which fits `long` (aim 45–90 s, hard max 150 s).
- Haiku window answers: median 23 s even when asked for 20–60 s. The widening fix is already committed (`95fb613`).
- Real usage: about 7.2k input and **1.7k output** tokens per call. Responses are only about 300 tokens, so roughly 1.4k tokens per call are hidden thinking. `--effort low` does not turn thinking off; `PLAN.md:29` flagged this as unverified.
- Deep with spec settings: about 1,300 calls per 2.5 h episode. Six questions go to about 150 candidates, plus 306 tournament pair calls.
- The 9 test episodes total 23.9 h of audio.
- Transcripts and loudness are cached for the test episodes. The pilot episodes have transcripts but no loudness yet.

## Rules for execution

1. No paid call until Phase 1 is committed and the suite is green.
2. Each phase has a gate. If a gate fails: stop, report numbers to the user, and do no ad-hoc rerun.
3. Freeze before the full run: commit and tag `eval-freeze-1`. Any code or prompt change after the freeze means a new pilot of the affected stage on tune only.
4. During the full run, the only allowed action is rerunning the same command after a usage-limit reset. Any other failure: stop and report.
5. A Monitor watches only errors, limits and exit. No frequent polling, because this Opus session shares the same Pro quota.

## Phase 1: code fixes (no paid calls; TDD, one commit each)

**T1. Thinking actually off, and cache keys that can't mix settings**
- `shorts_generator/local/llm.py` `_claude_run`: pass `env={**os.environ, "MAX_THINKING_TOKENS": "0"}` to `subprocess.run`.
- `model_id()`: for claude, return `claude:<model>:effort=<CLAUDE_EFFORT>:think=off`. Cached answers made with thinking on are then never replayed.
- Tests in `tests/test_llm_claude.py`: the fake `run` records `env`, and the test asserts `env["MAX_THINKING_TOKENS"] == "0"`. Update `test_model_id_names_provider_and_model` to expect `"claude:claude-haiku-5-5:effort=low:think=off"`.

**T2. Live-call budget guard (protects usage; no effect on answers)**
- `shorts_generator/config.py`: `LLM_MAX_CALLS = int(os.getenv("LLM_MAX_CALLS", "0"))`, where 0 means no cap.
- `shorts_generator/selection/cache.py` `_cached`: a module counter under a `threading.Lock`, incremented only on a live call. Cache hits are free. Past the cap, raise `RuntimeError(f"LLM_MAX_CALLS={n} reached; rerun with a higher cap to resume from the cache")` and write nothing.
- Test in `tests/test_selection_cache.py`: with cap 2, two live calls pass, the third raises, and replaying the first two cached prompts still works.

**T3. Eval controls for the pilot**
- `shorts_generator/selection/eval.py` `main`:
  - Add `--episodes a,b`, which filters `load_episodes` by episode name.
  - Add `--stop-after candidates`, which applies to deep only.
- `shorts_generator/selection/__init__.py` `run_deep(..., until: Optional[str] = None)`:
  - When `until == "candidates"`, return `{"highlights": [], "candidates": cands, "bible": bible}` right after `generate_candidates`.
  - The full run later reuses these cached calls.
- `selector_label` adds `until=candidates` to the deep options, so these rows never replace full deep rows.
- `report()`: add two columns, "median pick length (s)" and "median candidate length (s)". Each row stores `pick_len` and `cand_len`.
- Tests:
  - `--episodes` filtering in `load_episodes` or `main`;
  - `run_deep(until="candidates")` makes no scoring or pair calls (a fake `ask` that fails on any non-window/bible `prompt_id`);
  - the label is `deep[until=candidates]@long` when `TARGET_LENGTH=long`.

**T4. Baselines use the target length (fair comparison, no LLM)**
- `random_spans`: draw the length from `TARGET_LENGTHS[config.TARGET_LENGTH]` (aim min to aim max) instead of 20–60 s.
- `signal_spans(..., half_width: float = 20.0)`: the signal baseline in `eval.run_selector` passes `max(20, min_len / 2)`.
- Tests: at `long`, random spans are 45–90 s and signal spans are at least about 45 s.

**T5. Merge never drops a candidate by over-merging**
- `candidates.merge`: if the union with an existing span would exceed `hard_max`, keep the new span separate instead of merging and later dropping both.
- Test: two 100 s spans at IoU > 0.6 whose union is 160 s, at `long` (hard max 150 s), come out as 2 candidates, not 0.

Then run `.venv/bin/python -m pytest -q`. Expected: everything passes, with 2 skips as before.

## Phase 2: smoke test (2 paid calls)

Send one real hook-question prompt from the J_W5X-tlmU8 cache through `call_claude_llm` twice:
1. once with `MAX_THINKING_TOKENS=0` (the new default);
2. once without it.

Read `output_tokens` from the `[llm/claude]` log line.

**Gate S:**
- the thinking-off call has under 300 output tokens and gives valid JSON;
- the thinking-on call has clearly more output tokens.

If the two are the same, the env var doesn't control thinking. Stop and report; do not run with thinking silently on.

## Phase 3: pilot on tune (small, paid)

**P1, the cheap stage gate.** Both pilot episodes, candidates only, about 100 calls:
```
TARGET_LENGTH=long LLM_PROVIDER=claude LLM_WORKERS=2 LLM_MAX_CALLS=300 \
  .venv/bin/python -m shorts_generator.selection.eval --split tune \
  --episodes source_J_W5X-tlmU8,source_RBZTLptAMao --selectors deep --stop-after candidates \
  --out eval/pilot/report.md
```
**Gate G1:**
- median candidate length is between 45 and 150 s;
- Recall@candidates is at least 0.5 on each episode (before the fixes it was 0.33–0.52 at medium length).

**P2, the full workflow.** One episode (J_W5X-tlmU8, 72 min, 6 gold):
```
TARGET_LENGTH=long LLM_PROVIDER=claude LLM_WORKERS=2 LLM_MAX_CALLS=2000 HIGHLIGHT_CHUNK_SECONDS=300 \
  .venv/bin/python -m shorts_generator.selection.eval --split tune --episodes source_J_W5X-tlmU8 \
  --selectors random,signal,classic,deep --out eval/pilot/report.md
```
**Gate G2:**
- exit 0, and `work/source_J_W5X-tlmU8/deep.json` is written;
- answer retries stay under 1% of calls;
- median pick length for classic and deep is within 45–150 s;
- every row has metrics.

P@5 is reported but is not a gate, because one episode is too few to judge.

**Projection.** From P2's `[llm/claude]` log lines, compute calls, input tokens and output tokens per audio hour for classic and for deep. Multiply by the 23.9 test hours. Compare with the earlier 5-hour window, which ran out after about 650 calls with thinking on.

**Gate G3:** the user approves the full run after seeing the projected calls, tokens and rough number of 5-hour windows. Then commit and tag `eval-freeze-1`.

## Phase 4: full test run (once)

```
TARGET_LENGTH=long LLM_PROVIDER=claude LLM_WORKERS=2 LLM_MAX_CALLS=<projection x 1.3> HIGHLIGHT_CHUNK_SECONDS=300 \
  .venv/bin/python -m shorts_generator.selection.eval --split test \
  --selectors random,signal,classic,deep --out eval/report.md > work/eval-test-final.log 2>&1
```
- Monitor: `grep -E "P@5|limit|Traceback|Error|LLM_MAX_CALLS|exit [0-9]"`.
- On a usage limit: rerun the same command after the reset. The cache makes completed calls free.
- On any other error: stop and report.
- When done: commit `eval/report.md` and `eval/rows.json`. Report deep vs classic with per-episode results, and note that 9 episodes is under the 10-episode bar for firm conclusions.

## Critical files
- `shorts_generator/local/llm.py` (`_claude_run`, `model_id`)
- `shorts_generator/selection/cache.py` (`_cached`)
- `shorts_generator/selection/eval.py` (`main`, `selector_label`, `report`, `random_spans`, `run_selector`)
- `shorts_generator/selection/__init__.py` (`run_deep`)
- `shorts_generator/selection/candidates.py` (`signal_spans`, `merge`)
- `shorts_generator/config.py`
- Tests: `tests/test_llm_claude.py`, `tests/test_selection_cache.py`, `tests/test_selection_eval.py`, `tests/test_selection_candidates.py`

## Not changed (would alter quality or the workflow)
- Window samples (2), the six questions for every candidate, the tournament size (18), the MMR settings and the prompts other than the length fix.

## Known residual risks
- Usage per 5-hour window with thinking off is unknown until P2 measures it.
- The loudness decode reads stderr only after stdout. A corrupt pilot file could hang it. That happens after the bible stage, and the bible answers are cached, so a hang costs time but no usage.
