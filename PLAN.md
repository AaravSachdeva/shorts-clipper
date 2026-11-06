# Deep Clip Selector, Thin Slice (M0 + M1 + text-only deep path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Context

The spec (Nov 6, 2026, "shorts-clipper: Gemma 4 12B Clip Selection Pipeline Spec") replaces the single-pass `highlights.py` selector with an 8-stage pipeline. Its own rule is "measure first; a stage stays only if the eval shows it helps". The user chose to build only the part that can be measured now:

- **Scope:** M0 (eval set, metrics, baselines) + M1 (plumbing, call cache, `--selector deep`) + a thin text-only deep path (bible, candidates, narrow questions, default combiner, pairwise tournament, MMR). Later stages get their own plans once the eval says they pay.
- **Stand-in model:** during development every LLM call goes to **Claude Haiku 5.5 through headless `claude -p`** on the user's Pro login. Gemma 4 12B replaces it only after the eval shows the deep path adds significant value. Claude gets **only** the prompt Gemma would get: no tools, no settings, no MCP, no CLAUDE.md.
- **Thinking off** by default (gemma4:12b once thought for over an hour on one window; `config.py`, README).
- **Eval seed:** Raj Shamani "Figuring Out" English episodes + the channel's official Shorts, matched by transcript text. Two episodes are already downloaded and transcribed: `output/source_BV7QRw8wtjs.*` (FO578, 10,209 s) and `output/source_TjPPIf-M6KE.*` (FO583, 4,080 s), both `en-US`.

Pro-plan fact (support.claude.com, checked 2026-11-06): `claude -p` on a Pro login draws from the same 5-hour and weekly limits as claude.ai; no API credits on Pro; `--bare` needs an API key (billed separately), so it is not used.

**Goal:** an eval harness that scores selectors on labelled episodes, and a `--selector deep` text-only path that can be compared against random, signal-only and classic, all runnable with Claude Haiku as the model.

**Architecture:** New package `shorts_generator/selection/`. Every LLM call goes through `llm_fn(prompt, schema)` wrapped by a hash-keyed disk cache (`work/<episode>/calls/<hash>.json`), so any crash or Pro usage limit resumes for free. A new `LLM_PROVIDER=claude` backend in `local/llm.py` shells out to `claude -p`. `pipeline._run_local` calls `selection.run_deep` when `SELECTOR=deep`; output dicts match `get_highlights`, so metadata and rendering are unchanged.

**Tech Stack:** Python 3.11 (`.venv`), numpy 2.4, ffmpeg binary, yt-dlp, faster-whisper (cached transcripts), Claude Code CLI 2.1.319, pytest 9.

**Spec:** `docs/superpowers/specs/2026-11-06-gemma-deep-selection-design.md` (copied from the upload in Task 0).

## Global Constraints

- Run everything with `.venv/bin/python`. **No new dependencies** in this plan (no sklearn, pandas, pyarrow, torch, pyannote, choix).
- `SELECTOR=classic` stays the default; `highlights.py` behaviour is unchanged except one optional parameter on `chunk_transcript`.
- Every LLM call in `selection/` goes through `cache.make_ask`; every call passes a JSON schema.
- Claude backend: model `claude-haiku-5-5`, `--effort low` (closest the CLI offers to "thinking off"; **unverified** whether Haiku still thinks at low), empty `--system-prompt`, `--tools ""`, `--setting-sources project` from an empty temp cwd, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`.
- Spec values, verbatim: windows 300 s with 60 s overlap; up to 8 candidates per window; each window asked twice; merge at temporal IoU > 0.6; gold match at IoU ≥ 0.5; loudness peak = more than 6 dB above the episode median; early exit when `hook` = 1 and `standalone` = 1; default weights hook 0.3, standalone 0.2, payoff 0.2, emotion 0.15, novelty 0.15, minus 0.5 × std; final = 0.6 × Bradley–Terry rank + 0.4 × combiner rank; MMR λ = 0.3; `TOURNAMENT_MAX` = 18.
- **Unverified externals** (observation 0002 rule: cite or mark): `claude -p` reading the prompt from stdin, accepting `--system-prompt ""`, `--setting-sources project`, returning `structured_output` with `--json-schema`; the Raj Shamani Shorts tab URL. Task 1 and Task 4 verify them before anything depends on them.
- Conclusions about a stage need ≥ 10 labelled episodes (the spec's combiner threshold); the report prints the episode count.
- Commit messages, docs and code comments in normal English.

## Decisions that deviate from the spec (the planner's calls, for review)

| Spec | This plan | Why |
| --- | --- | --- |
| Stage 0: diarization, PANNs, shots, faces, OCR, `features.parquet` | Loudness per second only (ffmpeg + numpy), JSON | Zero new deps; each extractor must earn its place in a later plan |
| Stage 1: map, who's-who, reduce, callbacks, cross-check, size re-run | Map + one reduce call (summary, chapters, setups_payoffs, running_jokes) | No speaker labels without diarization; size capped by schema `maxItems` and word limits |
| Stage 3: perception 3a, 3 wordings, `SCORE_SAMPLES=3` | Text-only, 1 wording per question, `SCORE_SAMPLES=1` (repeat samples via `--samples N`) | Claude can't hear audio; fast-track; samples ablation still possible |
| Stage 4: YAML defaults, logistic regression | Default weights as a dict constant; training deferred | Needs 10 labelled episodes and sklearn |
| Stage 5 refine + cold test, Stage 7 feedback | Skipped | Spec puts them in M4/M5 |
| Stage 6: Swiss above 18 | Cut to top 18 by combiner score | Same call cap, no scheduler code |
| `mm_server.py` (Transformers) | Dropped | 12B in bf16 needs ~24 GB (estimate) > 16 GB RAM; `ollama show gemma4:12b` reports `audio` and `vision` capabilities, so a later multimodal plan uses Ollama |
| Prompt front matter `{id, version, schema}`; `--fresh <stage>`; stage files per stage | Prompt id = file stem, version = content hash; schemas stay in Python; no `--fresh` (a prompt edit changes the hash); only `loudness.json` + `deep.json` written | The call cache already gives resume and replay |
| `harness/run.py` extension | Unchanged | Call cache replaces it for deep |
| Baseline `classic` with `qwen3:8b` and `gemma4:12b` | `classic` with Claude Haiku, `HIGHLIGHT_CHUNK_SECONDS=300` (as it runs for Ollama) | Same model as deep, so the comparison is fair; Gemma baselines come with the Gemma switch |

---

### Task 0: Save spec and plan into the repo

- [ ] Copy `/home/aarav/.claude/uploads/7ec9f164-3cd6-4ca6-bdb2-792cca7ade94/87eb5a7f-shorts-clipper_Gemma_4_12B_Clip_Selection_Pipeline_Spec.md` to `docs/superpowers/specs/2026-11-06-gemma-deep-selection-design.md`. This plan lives at `PLAN.md` in the repo root (the user asked for that location).
- [ ] Add `work/`, `eval/_shorts_cache/` and `.superpowers/` to `.gitignore` (eval `gold.json` files are committed).
- [ ] Commit: `docs: deep selector spec and thin-slice plan`. Leave the unrelated `README.md` working-tree change unstaged.

### Task 1: `LLM_PROVIDER=claude` backend

**Files:** Modify `shorts_generator/local/llm.py`, `shorts_generator/config.py`, `main.py` (`--llm-provider` choices). Test `tests/test_llm_claude.py`.

**Interfaces:**
- Produces: `_claude_run(prompt: str, schema: Optional[Dict] = None) -> Dict` (the CLI's parsed JSON result); `call_claude_llm(prompt: str, schema: Optional[Dict] = None) -> str`; `model_id() -> str` returning `"claude:<CLAUDE_MODEL>"` or `"ollama:<OLLAMA_MODEL>"` etc.; `call_local_llm` dispatches `"claude"`.
- Config: `CLAUDE_MODEL` (default `claude-haiku-5-5`), `CLAUDE_EFFORT` (`low`), `CLAUDE_TIMEOUT_SECONDS` (env `CLAUDE_TIMEOUT`, `600`), `LLM_WORKERS` (`1`).

- [ ] **Step 1: Failing tests.** Monkeypatch `subprocess.run` in the llm module with a fake that records `args`, `input`, `cwd` and returns `CompletedProcess(args, 0, stdout=json.dumps({"is_error": False, "result": '{"a": 1}', "usage": {"input_tokens": 10, "output_tokens": 3}}), stderr="")`.
  - `test_claude_argv_is_isolated`: after `call_claude_llm("hi", {"type": "object"})`: `input == "hi"`; argv contains `-p`, `--model claude-haiku-5-5`, `--output-format json`, `--system-prompt ""`, `--tools ""`, `--setting-sources project`, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`, `--effort low`, `--json-schema '{"type": "object"}'`; `os.listdir(cwd) == []`; return value `'{"a": 1}'`.
  - `test_claude_prefers_structured_output`: stdout has `"structured_output": {"b": 2}` → returns `json.dumps({"b": 2})`.
  - `test_claude_error_raises`: `returncode=1` or `"is_error": True` → `RuntimeError` whose message starts `claude -p:` and includes the first 300 chars of `result` or stderr.
  - `test_dispatch_claude`: `config.LLM_PROVIDER = "claude"` → `call_local_llm` reaches `call_claude_llm`.
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_llm_claude.py -v` → FAIL (`AttributeError: ... call_claude_llm`).
- [ ] **Step 3:** Implement. Prompt goes on stdin; cwd is one module-level `tempfile.mkdtemp(prefix="claude-llm-")`; `timeout=config.CLAUDE_TIMEOUT_SECONDS`. Log one line like Ollama's: `[llm/claude] <model>: <input+cache> input + <output> output tokens in <s>s`. Add `claude` to `--llm-provider` choices and to the unknown-provider error text.
- [ ] **Step 4:** Tests pass; full suite `.venv/bin/python -m pytest -q` passes.
- [ ] **Step 5: Live isolation probe.** Add `tests/test_claude_live.py`, skipped unless `CLAUDE_LIVE=1`:
  - `test_claude_sees_only_the_prompt`: `_claude_run("Reply with OK.")`; total input tokens (`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`) must be `< 1500` (planner's threshold: the user's CLAUDE.md alone is larger, so any leak trips it).
  - `test_claude_structured_output`: `call_claude_llm("Answer OK.", {"type":"object","properties":{"answer":{"enum":["OK"]}},"required":["answer"]})` parses to `{"answer": "OK"}`.
  - Run `CLAUDE_LIVE=1 .venv/bin/python -m pytest tests/test_claude_live.py -v -s`. **If the token probe fails:** retry with `--system-prompt "You are a helpful assistant."`; if it still fails, stop and report the token count to the user (options: an API key + `--bare`, or a separate `CLAUDE_CONFIG_DIR`). If stdin or `structured_output` don't work, adjust Step 3 to what the CLI does and record it in the commit message.
- [ ] **Step 6:** Commit `feat: claude -p as a local LLM provider for development`.

### Task 2: Call cache, prompt loader, parallel map

**Files:** Create `shorts_generator/selection/__init__.py` (empty for now), `shorts_generator/selection/cache.py`, `shorts_generator/selection/prompts/judge_system.md`. Test `tests/test_selection_cache.py`.

**Interfaces (Produces):**
- `Ask = Callable[[str, Dict, str, int], Optional[Dict]]` — `ask(prompt, schema, prompt_id, sample=0)` returns the parsed answer or `None`.
- `cached_llm(llm_fn, calls_dir: Path, prompt_id: str, run: int = 0, stats: Optional[Dict] = None) -> LLMFn` — a drop-in `fn(prompt, schema=None) -> str` for `get_highlights` (classic baseline). Both it and `make_ask` sit on one internal `_cached(llm_fn, calls_dir, prompt, schema, prompt_id, sample, run, stats) -> Tuple[str, Path]`.
- `make_ask(llm_fn, calls_dir: Path, run: int = 0, stats: Optional[Dict] = None) -> Ask`. Key = `sha256(model_id() + prompt + json.dumps(schema, sort_keys=True) + f"{sample}:{run}")`. File `calls_dir/<key>.json` = `{key, prompt_id, model, sample, run, seconds, prompt, response}`, written to a temp file then `os.replace`. A file that fails to load is a miss. A response `_parse_json_loose` can't parse is logged, its file deleted, and `None` returned. Exceptions from `llm_fn` propagate and write nothing. `stats` accumulates `calls`, `misses`, `seconds` (stored seconds, hit or miss).
- `pmap(fn, items, workers: Optional[int] = None) -> List` — `ThreadPoolExecutor(max_workers=workers or config.LLM_WORKERS)`, order kept.
- `load_prompt(name: str, root: Path = PROMPTS_DIR) -> Tuple[str, str]` → `(template, f"{name}@{sha256(text)[:8]}")`.
- `judge_prefix(bible: Dict) -> str` → `judge_system.md` text + `"\n\nEpisode bible:\n"` + compact JSON of the bible, or `"(none)"` when empty. Every judging prompt starts with this prefix (Ollama prompt cache later).
- Reuses `highlights._parse_json_loose`, `local.llm.model_id`.

- [ ] **Step 1: Failing tests:** `test_same_call_hits_disk_once` (2 asks, 1 llm call, file holds prompt and prompt_id); `test_sample_and_run_change_key`; `test_failed_call_writes_nothing_and_retries`; `test_corrupt_file_is_a_miss`; `test_unparseable_answer_is_not_kept` (returns `None`, no file, next ask calls again); `test_stats_count_hits_and_misses`; `test_pmap_keeps_order` (`pmap(lambda x: x * 2, [3, 1, 2], workers=3) == [6, 2, 4]`); `test_prompt_version_changes_with_text` (tmp root).
- [ ] **Step 2:** Run → FAIL (import error).
- [ ] **Step 3:** Implement `cache.py`; `judge_system.md` is two sentences: the model judges moments of one podcast episode for short-form clips, and answers only the question asked in the required JSON shape.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `feat: hash-keyed LLM call cache for the deep selector`.

### Task 3: Eval metrics and episode loading

**Files:** Create `shorts_generator/selection/eval.py` (metrics part). Test `tests/test_selection_eval.py`.

**Interfaces (Produces):** spans are dicts with `start`/`end` (gold) or `start_time`/`end_time` (picks); one helper `_span(d) -> Tuple[float, float]` accepts both.
- `iou(a, b) -> float` (temporal IoU).
- `match(picks, gold, k: int) -> List[Tuple[int, int, float]]` — greedy in pick order, each gold used once, highest IoU ≥ 0.5.
- `precision_at(picks, gold, k=5)` (divides by k), `ndcg_at(picks, gold, k=10)` (gain = gold `rating`, discount `log2(i + 2)`, linear gain), `recall(cands, gold)` (gold hit by any candidate at IoU ≥ 0.5), `boundary_error(picks, gold, k=5)` (mean of `(|Δstart| + |Δend|) / 2` over matches, `None` if none), `jaccard_top(a, b, k=5)` (matches / (|A| + |B| − matches)).
- `random_spans(duration, n=10, seed: str) -> List[Dict]` — `random.Random(seed)`, length uniform 20–60 s, inside `[0, duration]`.
- `load_episodes(eval_dir: Path, split: str) -> List[Dict]` — reads `eval/*/gold.json`; `split` in `tune|test|all`; skips (with a printed note) episodes whose `gold` is empty.
- Gold file (spec format plus two fields): `{"episode", "source", "language", "split", "gold": [{"start", "end", "rating", "origin", "views", "url", "match_ratio"}]}`.

- [ ] **Step 1: Failing tests** with exact values: `iou((0,10),(5,15)) == pytest.approx(1/3)`, disjoint `0`, identical `1`; precision 2 of 5 → `0.4`, 1 pick matched → `0.2`; NDCG for matched ratings `[5, 0, 3]` vs gold ratings `[5, 3]` → `pytest.approx(6.5 / (5 + 3 / math.log2(3)))`; recall; boundary error; jaccard; `random_spans` same seed → same spans, all inside duration; `load_episodes` split filter and empty-gold skip.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat: eval metrics for clip selection`.

### Task 4: Gold builder from official Shorts

**Files:** Create `shorts_generator/selection/gold.py`. Modify `research/benchmark.py` (`_caption_words(ydl, info, max_seconds: float = 20)`; existing callers unchanged). Test `tests/test_selection_gold.py`, `tests/test_benchmark.py`.

**Interfaces:**
- Consumes: `benchmark._ydl`, `list_videos`, `_caption_words`; `local.downloader.download_youtube_local` (returns local paths unchanged); `local.transcriber.transcribe_local`; `boundaries.all_words`, `boundaries._norm`.
- Produces: `match_short(ep_words: List[Dict], short_tokens: List[str]) -> Optional[Dict]` → `{"start", "end", "match_ratio"}`; `merge_gold(old: List[Dict], official: List[Dict]) -> List[Dict]` (keeps every entry whose `origin != "official_short"`); CLI `python -m shorts_generator.selection.gold --shorts <url> --episode <url-or-path> [--episode ...] [--limit 300] [--language en]`.
- Matching: `difflib.SequenceMatcher(None, ep_tokens, short_tokens, autojunk=False).get_matching_blocks()`, blocks of ≥ 3 tokens; anchor = largest block; keep blocks within ±90 s of the anchor; `match_ratio` = matched tokens / short tokens; `None` below 0.5. Official Shorts get `rating: 4`, `origin: "official_short"`, plus `views` and `url`.
- Short captions are cached in `eval/_shorts_cache/<video_id>.json` so 15 episodes fetch the channel once.
- `split` is set once per episode: `"test"` if `int(sha256(episode).hexdigest(), 16) % 10 < 3` else `"tune"`, kept if the file already has one.

- [ ] **Step 1: Failing tests:** `test_match_short_finds_span` (synthetic 0.5 s words; the short's 30 words sit at 100 s with one word changed and different case/punctuation → start ≈ 100, ratio ≥ 0.8); `test_match_short_rejects_unrelated`; `test_far_common_phrase_ignored` (a shared 3-word phrase 600 s away does not stretch the span); `test_merge_keeps_hand_picks`; `test_caption_words_max_seconds` in `tests/test_benchmark.py`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. Episodes with zero matches still get a `gold.json` with `"gold": []` and a printed warning. **Step 4:** Pass.
- [ ] **Step 5: Verify the Shorts URL (unverified).** Run `.venv/bin/python -m yt_dlp --flat-playlist --print url "https://www.youtube.com/@rajshamani/shorts" | head -5`. Expected: Shorts URLs. If not, ask the user for the right channel URL.
- [ ] **Step 6: Label.** Run the builder on both local episodes (`output/source_BV7QRw8wtjs.mp4`, `output/source_TjPPIf-M6KE.mp4`). Print matches; spot-check 3 by watching the span. Then ask the user for more English episode URLs (target 15, per spec) and run them. Hand picks go into the same `gold.json` with `origin: "hand_pick"` and a 1–5 rating.
- [ ] **Step 7:** Commit `feat: build eval gold from a channel's official Shorts` (include `eval/*/gold.json`).

### Task 5: Loudness signal and signal-peak spans

**Files:** Create `shorts_generator/selection/loudness.py`, `shorts_generator/selection/candidates.py` (signal part). Test `tests/test_selection_candidates.py`.

**Interfaces (Produces):**
- `rms_db(samples: np.ndarray, rate: int) -> List[float]` — dBFS per whole second, `20 * log10(rms + 1e-9)` on int16 scaled to ±1.
- `loudness_per_second(source_path: str, cache_path: Path) -> List[float]` — `ffmpeg -i <src> -ac 1 -ar 16000 -f s16le -` streamed in 1 s blocks; cached as JSON, reused when newer than the source.
- `loud_peaks(db: List[float], above: float = 6.0) -> List[Tuple[int, float]]` — runs of seconds above `median + above`, one `(peak_second, peak_db)` per run, runs closer than 30 s merged.
- `signal_spans(words, times: List[Tuple[float, str, float]], duration: float, per_hour: int = 20) -> List[Dict]` — each `(t, origin, strength)` becomes `snap_clip(words, t - 20, t + 20, window=10.0)`; keeps the top `per_hour × hours` by strength; candidate dict `{"start_time", "end_time", "origin": [origin], "strength"}`.

- [ ] **Step 1: Failing tests:** `rms_db` on 1 s silence + 1 s full-scale square wave → `[< -100, ≈ 0]`; `loud_peaks` on `[-30]*60 + [-20]*3 + [-30]*60` → one peak at second 60–62; two runs 10 s apart merge; `signal_spans` snaps to sentence edges and caps per hour.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat: loudness peaks as clip candidates`.

### Task 6: Eval runner and baseline report

**Files:** Modify `shorts_generator/selection/eval.py` (runner + CLI). Test `tests/test_selection_eval.py`.

**Interfaces:**
- Consumes: Task 2 `make_ask`; Task 3 metrics; Task 5 loudness; `highlights.get_highlights`; `local.llm.call_local_llm`; `transcribe_local(source, language)`.
- Produces: `run_selector(name: str, ep: Dict, transcript: Dict, run: int = 0, skip=(), samples=None) -> Dict` → `{"picks", "candidates", "stats"}` for `random`, `signal` (loudness peaks only, sorted by strength), `classic` (`get_highlights(transcript, num_clips=10, llm_fn=cached_llm(call_local_llm, work/<stem>/calls, "classic", run, stats))`); `report(rows: List[Dict]) -> str` (markdown); CLI `python -m shorts_generator.selection.eval [--selectors random,signal,classic] [--split test] [--runs 1] [--skip ...] [--samples N] [--out eval/report.md]`.
- Report: one row per selector with episodes, Recall@candidates, P@5, NDCG@10, boundary error, stability (only with `--runs ≥ 2`, mean `jaccard_top` over run pairs), LLM calls, LLM hours (from `stats`); then a per-episode table.

- [ ] **Step 1: Failing tests:** `test_report_has_row_per_selector` (stub rows → markdown with the 3 names and the episode count); `test_classic_uses_cache` (fake `call_local_llm`, run twice → second run 0 misses).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5: Baseline report (M0 done).** `LLM_PROVIDER=claude LLM_WORKERS=4 HIGHLIGHT_CHUNK_SECONDS=300 .venv/bin/python -m shorts_generator.selection.eval --selectors random,signal,classic --split all`. Expected: `eval/report.md` with three rows. If the Pro limit stops it, rerun after the reset; the cache resumes.
- [ ] **Step 6:** Commit `feat: eval runner with random, signal and classic baselines` (include `eval/report.md`).

### Task 7: Episode bible

**Files:** Create `shorts_generator/selection/bible.py`, `prompts/bible_map.md`, `prompts/bible_reduce.md`. Modify `highlights.chunk_transcript(transcript, size: Optional[int] = None)` (`size or chunk_size_seconds()`). Test `tests/test_selection_bible.py`.

**Interfaces (Produces):** `build_bible(transcript: Dict, ask: Ask) -> Dict`; `chapter_of(t: float, chapters: List[Dict]) -> Optional[int]`. Map runs over `chunk_transcript(transcript, size=300)`, text = `build_transcript_text(chunk)` (absolute timestamps), via `pmap`.
- Map schema: `{topics: [string], stories: [{start: number, end: number, gist: string}], notable_lines: [{t: number, line: string}]}`.
- Reduce schema (all map outputs in one prompt): `{episode_summary: string, chapters: [{start, end, title, one_line}] (maxItems 15), setups_payoffs: [{setup_t, payoff_t, description}] (maxItems 10), running_jokes: [{label, times: [number]}] (maxItems 5)}`; prompt asks for a summary of at most 150 words. Log the bible's character length.

- [ ] **Step 1: Failing tests:** fake `ask` dispatching on `prompt_id` prefix: map called once per chunk; reduce called once and its prompt contains every map output; result has the 4 keys; `chapter_of` inside, on an edge, and outside all chapters.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass, and `tests/test_highlights.py` still passes.
- [ ] **Step 5:** Commit `feat: map-reduce episode bible`.

### Task 8: LLM window candidates and merge

**Files:** Modify `shorts_generator/selection/candidates.py`; create `prompts/window.md`. Test `tests/test_selection_candidates.py`.

**Interfaces (Produces):** `window_candidates(transcript, bible, ask) -> List[Dict]`; `merge(cands, words, target_length: Optional[str]) -> List[Dict]`; `generate_candidates(transcript, bible, db: List[float], ask, target_length=None) -> List[Dict]` (windows + `signal_spans` from loudness peaks and bible `payoff_t` / running-joke times, then `merge`). Records: `{"id": "c017", "start_time", "end_time", "origin": [...], "kind", "why", "window_ids": [...]}`.
- Window prompt = `judge_prefix(bible)` + window text (absolute timestamps) + `VIRALITY_CRITERIA` + `PLATFORM_SIGNALS` from `highlights.py`. Schema `{candidates: [{start: number, end: number, kind: enum, why: string}] (maxItems 8)}`, `kind` in `hook_moment, emotional_peak, opinion_bomb, revelation, conflict, quotable, story_peak, practical_value`. Asked with `sample=0` and `sample=1`.
- Drop a window answer if `end <= start`, `start < window_start - 5` or `end > window_end + 5`.
- `merge`: snap each with `snap_clip` (keep raw when snapping fails), greedily merge IoU > 0.6 into the union span with origins and window ids unioned, then keep `MIN_CLIP_SECONDS <= duration <= TARGET_LENGTHS[target][2]`.

- [ ] **Step 1: Failing tests:** both samples asked per window; out-of-window and inverted spans dropped; two spans at IoU 0.7 merge with `origin == ["llm", "loudness"]`; IoU 0.5 stays separate; spans over hard max dropped; ids are `c000`, `c001`, … in start order.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat: deep candidates from LLM windows and signal peaks`.

### Task 9: Narrow questions and default combiner

**Files:** Create `shorts_generator/selection/score.py`, `prompts/{hook,payoff,emotion,standalone,novelty,ending}.md`. Test `tests/test_selection_score.py`.

**Interfaces (Produces):** `score_candidates(transcript, bible, cands, ask, samples: int) -> List[Dict]` adds `answers: {key: [float 0–1]}` and `extras: {hook_quote, novelty_label}`; `combine(answers: Dict[str, List[float]]) -> float`; `DEFAULT_WEIGHTS = {"hook": 0.3, "standalone": 0.2, "payoff": 0.2, "emotion": 0.15, "novelty": 0.15}`.
- Each prompt = `judge_prefix(bible)` + clip text (`highlights._clip_excerpt(transcript, s, e, head=4000, tail=1000)`) + the spec's question, verbatim:

| key | question (spec) | schema | normalised |
| --- | --- | --- | --- |
| hook | Would the first 3 seconds stop a stranger scrolling? (prompt also shows the words of the first 3 s) | `{opening_line: string, score: integer 1–5}` | `(score-1)/4` |
| payoff | Does the clip land a punchline, reveal or conclusion before it ends? | `{answer: yes\|partial\|no}` | 1 / 0.5 / 0 |
| emotion | Is there a clear emotional peak? | `{score: 1–5}` | `(score-1)/4` |
| standalone | Can someone who never heard this podcast follow it? | `{score: 1–5}` | `(score-1)/4` |
| novelty | Is it funny, surprising, contrarian or useful? | `{label: funny\|surprising\|contrarian\|useful\|none, score: 1–5}` | `(score-1)/4` |
| ending | Does it end on a complete thought, not mid-argument? | `{answer: yes\|no}` | 1 / 0 (stored, not weighted) |

- Pass 1: `hook` and `standalone` sample 0 for all candidates; drop candidates with both = 1 (score 1, before normalising). Pass 2: the rest, and samples 1..n−1, for survivors. Out-of-range scores are clamped to 1–5; a `None` answer is left out.
- `combine` = Σ wₖ·mean(xₖ) / Σ wₖ over present keys − 0.5 × mean population std over present keys.

- [ ] **Step 1: Failing tests:** early exit makes no pass-2 calls for that candidate; `combine({"hook":[1.0],"standalone":[0.5],"payoff":[1.0],"emotion":[0.0],"novelty":[1.0]}) == pytest.approx(0.75)`; payoff missing → `pytest.approx(0.55 / 0.8)`; hook `[1.0, 0.0]` with others single → penalty `0.5 * (0.5 / 5)`; score 7 clamps to 5.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat: narrow-question scoring with a default combiner`.

### Task 10: Pairwise tournament, MMR, text fields

**Files:** Create `shorts_generator/selection/tournament.py`, `prompts/pair.md`, `prompts/text_fields.md`. Test `tests/test_selection_tournament.py`.

**Interfaces (Produces):**
- `bradley_terry(n: int, wins: List[Tuple[int, int]], ties: List[Tuple[int, int]], iters: int = 200) -> List[float]` — MM update `pᵢ ← Wᵢ / Σⱼ nᵢⱼ / (pᵢ + pⱼ)`, a tie counts half a win to each side, plus 0.1 pseudo-wins per opponent so no strength is 0; normalised to sum 1.
- `play(cands, transcript, bible, ask) -> List[float]` — every pair asked as (A=i, B=j) and (A=j, B=i), schema `{winner: enum A|B}`, question verbatim: "Which clip would more people watch to the end and send to a friend? Answer A or B." Win only when both orders agree; disagreement or a missing answer is a tie. Fewer than 2 candidates → no calls.
- `blend(bt: List[float], comb: List[float], w_bt: float = 0.6) -> List[float]` — rank score `1 - rank / (n - 1)` (1.0 when n = 1), `w_bt·bt_rank + (1 - w_bt)·comb_rank`.
- `mmr(items: List[Dict], k: int, lam: float, chapters: List[Dict]) -> List[Dict]` — greedy `final − lam × max_sim`; sim = 1 when `chapter_of(midpoint)` is the same non-None chapter, else 0.
- `text_fields(pick, transcript, bible, ask) -> Dict` — one call per pick; schema = the `title`, `hook_sentence`, `on_screen_hook`, `virality_reason` properties copied from `HIGHLIGHT_SCHEMA`'s item; rules for `hook_sentence` and `on_screen_hook` copied from `HIGHLIGHT_SYSTEM_PROMPT`. Missing answer → title "Untitled Highlight", other fields "".

- [ ] **Step 1: Failing tests:** BT with A>B, B>C, A>C ranks A > B > C; all ties → equal strengths; agree/disagree/missing pair resolution with a fake `ask`; `blend([0.5, 0.3, 0.2], [0.1, 0.9, 0.5])` → `[0.6, 0.7, 0.2]` (approx); MMR scores `[1.0, 0.95, 0.8]`, first two in one chapter, λ 0.3 → order `[0, 2, 1]`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat: pairwise tournament and varied final picks`.

### Task 11: `run_deep`, CLI wiring, profiles, deep in the eval

**Files:** Modify `shorts_generator/selection/__init__.py`, `shorts_generator/pipeline.py`, `main.py`, `shorts_generator/config.py`, `shorts_generator/selection/eval.py`, `README.md`, `.claude/skills/youtube-shorts-generator/SKILL.md` (lines 10 and 38: upstream URL → `https://github.com/AaravSachdeva/shorts-clipper`). Test `tests/test_selection_deep.py`.

**Interfaces (Produces):**
- `run_deep(transcript: Dict, source_path: str, num_clips: int, llm_fn=None, skip: Iterable[str] = (), samples: Optional[int] = None, run: int = 0, target_length: Optional[str] = None, stats: Optional[Dict] = None) -> Dict` → `{"highlights", "candidates", "bible"}`. `llm_fn` defaults to `call_local_llm`. Work dir `Path(config.WORK_DIR) / Path(source_path).stem`. Flow: bible (`{}` if `"bible"` in skip) → loudness → candidates → score → top `TOURNAMENT_MAX` by combiner → tournament + blend (or combiner rank only if `"tournament"` in skip) → MMR (`MMR_LAMBDA`) → text fields → write `deep.json` (bible, candidates with answers, picks).
- Highlight dict: `title, start_time, end_time, score (round(100 × final)), hook_sentence, on_screen_hook, virality_reason` + `bt_strength, combiner_score, sub_scores, origin, kind`.
- Config: `SELECTOR` (`classic`), `WORK_DIR` (`work`), `SCORE_SAMPLES` (`1`), `TOURNAMENT_MAX` (`18`), `MMR_LAMBDA` (`0.3`).
- `pipeline._run_local`: `run_deep(transcript, source_path, num_clips, llm_fn=call_local_llm)` when `config.SELECTOR == "deep"`, else the existing `get_highlights` call.
- `main.py`: `--selector {classic,deep}` → env `SELECTOR`. Profiles:
  - `claude-deep`: `LLM_PROVIDER=claude`, `SELECTOR=deep`, `CLAUDE_MODEL=claude-haiku-5-5`, `LLM_WORKERS=4`, `HIGHLIGHT_CHUNK_SECONDS=300`, Whisper keys as `cpu-slow`.
  - `gemma-deep` (spec M1; untested until the Gemma switch): `LLM_PROVIDER=ollama`, `SELECTOR=deep`, `OLLAMA_MODEL=gemma4:12b`, `OLLAMA_NUM_CTX=32768` (the bible reduce prompt is ~20k tokens, estimate), `LLM_WORKERS=1`, `HIGHLIGHT_CHUNK_SECONDS=300`, Whisper keys as `cpu-slow`.
- Eval: `deep` selector calls `run_deep(..., num_clips=10, skip, samples, run, stats)`; candidates feed Recall@candidates; `--skip bible,tournament` and `--samples N` label the row (e.g. `deep[-bible]`).
- README: a short "Deep selector (experimental)" section: what it does, the `claude-deep` command, that it uses the Pro plan's limits and resumes from `work/<episode>/calls/`, and the eval command. Fix the install-step clone URL to this fork.

- [ ] **Step 1: Failing tests:** `test_run_deep_end_to_end` (tiny transcript with word timings, loudness monkeypatched to a flat list, fake `llm_fn` answering by schema shape) → highlights have every required key, `deep.json` exists; `test_run_deep_resumes` (second run → 0 `llm_fn` calls); `test_pipeline_uses_deep_selector` (monkeypatch `run_deep`, `SELECTOR=deep`); `test_selector_flag_sets_env`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Full suite passes.
- [ ] **Step 5:** Commit `feat: --selector deep runs the text-only deep pipeline`.

## Review Focus

1. **Pro usage limit hit mid-stage.** `claude -p` exits with an error; the run must stop with a message that starts `claude -p:`, keep every answer already cached, and repeat zero calls on rerun. Tests: Task 1 `test_claude_error_raises`, Task 2 `test_failed_call_writes_nothing_and_retries`, Task 11 `test_run_deep_resumes`.
2. **Context leaking into Claude** (CLAUDE.md, hooks, skills, default system prompt). Claude must see only the pipeline's prompt. Test: Task 1 live probe (`< 1500` input tokens for "Reply with OK.").
3. **Malformed or out-of-range answers** (non-JSON, score 7, missing field). Must be dropped or clamped, never crash a stage, and never get cached as poison. Tests: Task 2 `test_unparseable_answer_is_not_kept`, Task 9 clamp test.
4. **Window timestamps outside the window, inverted, or past the episode end.** These must be dropped before merging. Test: Task 8.
5. **Episodes that can't be evaluated** (no official Shorts matched, transcript without word timings). The gold builder writes empty gold with a warning, and the eval skips the episode with a printed note instead of reporting 0 precision. Tests: Task 3 `load_episodes`, Task 4 zero-match path.

## Verification (end to end)

1. `.venv/bin/python -m pytest -q`: all pass, with no model needed.
2. `CLAUDE_LIVE=1 .venv/bin/python -m pytest tests/test_claude_live.py -v -s`: passes, and the printed input-token count is under 1500.
3. `LLM_PROVIDER=claude LLM_WORKERS=4 HIGHLIGHT_CHUNK_SECONDS=300 .venv/bin/python -m shorts_generator.selection.eval --selectors random,signal,classic,deep --split all`, then `--selectors deep --skip bible`, `--skip tournament` and `--samples 3`. `eval/report.md` should have rows for random, signal, classic, deep and each ablation.
4. Spec targets for deep vs classic: Recall@candidates ≥ 90%, P@5 +15 points, NDCG@10 +0.1, and deep must beat signal-only. Judge them only once the report shows ≥ 10 episodes. Meeting them is the go signal for the Gemma switch plan.
5. `.venv/bin/python main.py output/source_TjPPIf-M6KE.mp4 --mode local --profile claude-deep --num-clips 3` renders 3 shorts in `output/<video title>/`, and `work/source_TjPPIf-M6KE/deep.json` exists.

## Execution

User chose **Subagent-driven** (superpowers:subagent-driven-development): fresh implementer + fresh reviewer per task, whole-branch review at the end. Subagents run on the same account as the eval's `claude -p` calls, so run the baseline (Task 6 Step 5) when the Pro window has room. Work on branch `feat/deep-selector` cut from `feat/rubric-windows` (HEAD `47bffe5`), because that branch is 6 commits ahead of `main` and this plan builds on them: the rubric windows, thinking off by default, and the Whisper/GPU changes.

## Follow-up plans (not in this one)

- Gemma switch: run the eval with `gemma-deep`, fix `_estimate_tokens` to use real token counts, add a `--think` ablation.
- Full Stage 0 signals: diarization, laughter, shots, faces, OCR.
- Multimodal through Ollama (Stage 3a, Stage 5).
- Trained combiner once 10 episodes are labelled.
- Feedback loop and LoRA.
- Rewrite of SKILL.md's MuAPI-centred text.
