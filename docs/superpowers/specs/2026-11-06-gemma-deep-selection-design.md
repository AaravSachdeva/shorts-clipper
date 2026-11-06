# shorts-clipper: Gemma 4 12B Clip Selection Pipeline Spec

Nov 6, 2026 · @Aarav  Sachdeva

This spec replaces the single-pass highlight step in `shorts_generator/highlights.py` with an eight-stage, multimodal selection pipeline that runs entirely on Gemma 4 12B. Download, transcription, boundary snapping and rendering stay as they are.

## Overview

**Goal:** pick better clips from a 1–2 hour podcast, offline, using Gemma 4 12B as the only LLM. Better means higher precision@5 against a labelled eval set, and later, higher retention on posted clips.

**Constraints**

- One model: Gemma 4 12B instruction-tuned, used for text, audio and image input. No second model family to cross-check it.
- Fully local. No API keys in the `--mode local --profile gemma-deep` path.
- Throughput does not matter. A run of several hours per episode on a 16 GB CPU machine is acceptable; days is the outer limit.
- Every stage must be resumable, because a crash 5 hours in should not restart the run.

**Design rules for a 12B judge**

1. Never ask a big question. Split every judgment into narrow yes/no or 1–5 questions, one per call.
2. Sample, don't trust. Each judgment is asked several times with varied prompt wordings, and the results are averaged.
3. Let data set the weights. A small learned combiner turns sub-scores and signal features into one score, instead of the model's overall gut score.
4. Give it the episode, not just the window. A compact episode bible is prepended to every judging prompt.
5. Use its senses. Gemma hears the audio and sees frames of each candidate; tone and reactions are where it beats a text-only model.
6. Compare instead of rating when ranking. Pairwise comparisons in both orders beat absolute scores.

## Current repo and what changes

The repo already has the right skeleton: a pluggable `llm_fn(prompt, schema)`, word-level transcripts, schema-constrained Ollama calls, sentence snapping and a replay harness. The new pipeline reuses all of that and swaps out the selection core.

| Module | Today | In this spec |
| --- | --- | --- |
| `local/transcriber.py` | faster-whisper, word timestamps, JSON cache | Kept. Stage 0 adds diarization on top of its output |
| `highlights.py` | Content-type call, 10–20 min chunks, one highlight call per chunk, dedupe, one re-rank call | Kept as the `classic` selector. New `selection/` package is the `deep` selector |
| `boundaries.py` | Snaps to sentence starts and ends, skips fillers | Kept and reused in Stage 5 |
| `local/llm.py` | Text-only `llm_fn` for OpenAI, Gemini, Ollama | Extended with a multimodal `mm_fn` that accepts audio and image parts |
| `local/clipper.py` | Shot detection, per-shot crop, captions, loudnorm | Kept. Its shot detector is reused by Stage 0 |
| `metadata.py` | Title, caption, hashtags | Kept |
| `harness/run.py` | Prompt-hash replay LLM | Extended to replay `mm_fn` calls and run the eval |
| `research/benchmark.py` | Hook features of top Shorts in a niche | Feeds few-shot examples and combiner features |
| `main.py` | Profiles `cpu-slow`, `cpu-quality` | New profile `gemma-deep` and flag `--selector {classic,deep}` |

Known issues to fix along the way:

- `.claude/skills/youtube-shorts-generator/SKILL.md` and the README install steps still point to the upstream SamurAIGPT repo and MuAPI. Update them when `--selector deep` lands.
- `_estimate_tokens` assumes 4 characters per token. Gemma's 262K-entry vocabulary tokenizes differently; read the real count from Ollama's `prompt_eval_count` and the runtime's tokenizer instead.
- `highlights.py` mixes prompts, schemas and control flow in 580 lines. The new code keeps prompts in `selection/prompts/*.md` files so they can be versioned and A/B tested.

## Architecture

The new code lives in `shorts_generator/selection/` and plugs in where `get_highlights()` is called today, so download, transcription and rendering are untouched.

&#91;embedded content: selection pipeline · stages 0–7 and the feedback loop\]

Each stage reads the previous stage's file and writes its own, so any stage can be rerun alone.

```
shorts_generator/selection/
  __init__.py        run_deep(transcript, source_path, num_clips) -> highlights
  signals/           transcript, diarize, prosody, audio_events, shots, faces, ocr, timeline
  bible.py           Stage 1
  candidates.py      Stage 2
  score.py           Stage 3
  combiner.py        Stage 4
  refine.py          Stage 5
  tournament.py      Stage 6
  feedback.py        Stage 7
  cache.py           hash-keyed call cache (resume + replay)
  mm_server.py       Transformers server for audio/image calls
  eval.py            evaluation runner
  prompts/*.md       one file per prompt, versioned

work/<episode>/
  signals/*.json  timeline.txt  features.parquet  bible.json
  candidates.jsonl  scores.jsonl  combined.jsonl  refined.jsonl  final.json
  calls/<hash>.json
```

`pipeline._run_local` changes in one place: when `SELECTOR=deep`, it calls `selection.run_deep(...)` instead of `get_highlights(...)`. The returned list has the same dict shape, so `generate_metadata` and `crop_highlights_local` need no changes.

## Stage 0: Signals and merged timeline

Stage 0 turns the video into a text timeline that Gemma can read in one pass, plus a feature table per second. No LLM runs here.

**Module:** `selection/signals/`, one file per extractor. Each writes its own JSON to `work/<episode>/signals/` and is skipped if that file is newer than the source.

| Extractor | Tool | Output per item |
| --- | --- | --- |
| `transcript.py` | Existing faster-whisper cache (`large-v3-turbo`) | Words with start, end, probability |
| `diarize.py` | pyannote speaker diarization, aligned to words | `speaker` on every word and segment |
| `prosody.py` | librosa on 16 kHz mono | Per 0.5 s: RMS loudness, pitch (pyin), speech rate (words per second) |
| `audio_events.py` | PANNs CNN14 (AudioSet labels) | Per 1 s: laughter, applause, music, shouting probabilities |
| `shots.py` | Reuse `_analyze_shots` from `local/clipper.py` | Shot boundaries, one keyframe per shot |
| `faces.py` | MediaPipe face detection + TalkNet or LR-ASD active speaker | Per shot: face count, who is speaking on screen |
| `ocr.py` | PaddleOCR on keyframes | On-screen text with timestamps |

**Merged timeline** (`timeline.py` → `timeline.txt`): one line per sentence-level segment.

```
[00:12:34.2–00:12:41.0] S2 (laugh 0.82, loud +6 dB, fast): "No, that's literally what he said on air." | shot 41: close-up, 2 faces
```

Rules: timestamps at segment level only (word timing stays in the JSON); annotations only when a signal crosses its threshold, so quiet stretches cost no extra tokens; speaker labels replaced with names once Stage 1 learns them. Target about 25–30k tokens per hour of audio.

**Feature table** (`features.parquet`): one row per second with every numeric signal. Stage 2 uses it to propose candidates and Stage 4 to compute clip features.

**Acceptance:** on the Cordkillers Ep 377 harness episode, diarization assigns the three hosts correctly on 90% of a hand-checked 5-minute sample, and laughter peaks line up with audible laughs on a spot check of 20.

## Stage 1: Episode bible

Stage 1 produces `bible.json`, a 1.5–3k-token summary of the whole episode that every later prompt starts with. It is built map-reduce style because a 12B model synthesises small inputs more reliably than 30k tokens at once.

**Module:** `selection/bible.py`

1. **Map.** Split `timeline.txt` into 5-minute chunks. For each, one call returns `{topics, stories[{start, end, gist}], notable_lines[{t, speaker, line}], speakers_seen}`.
2. **Who's who.** One call over all chunk outputs maps `S1/S2/S3` to names and roles (host, guest, co-host), using self-introductions and OCR lower-thirds. The names are written back into the timeline.
3. **Reduce.** One call merges the chunk outputs into `chapters[{start, end, title, one_line}]` and an episode summary of at most 150 words.
4. **Callbacks.** One call over the chunk outputs only, asking for `setups_payoffs[{setup_t, payoff_t, description}]` and `running_jokes[{label, times[]}]`. This is the pass that lets a clip at 48:00 know about its setup at 10:00.
5. **Cross-check.** One full-context call over the whole `timeline.txt` (about 30k tokens per hour, inside Gemma's 256K window), asking only: "List any chapters, guests or setup–payoff pairs missing from this bible." Additions are merged in.

**Bible schema (abridged)**

```json
{"episode_summary": "...", "people": [{"label": "S2", "name": "...", "role": "guest"}],
 "chapters": [{"start": 0, "end": 612, "title": "...", "one_line": "..."}],
 "setups_payoffs": [{"setup_t": 604, "payoff_t": 2890, "description": "..."}],
 "running_jokes": [{"label": "...", "times": [410, 1730, 3005]}]}
```

Every field is constrained by a JSON schema, passed through the existing `schema` argument. The same rule applies to every LLM call in this spec.

**Size cap:** if the bible exceeds 3k tokens, the reduce call is re-run with a hard length limit, so it never crowds out the window it is prepended to.

## Stage 2: Candidate generation

Stage 2 aims for recall: 50–100 candidates per hour, from two independent sources, merged into `candidates.jsonl`. Missing a great moment here is unrecoverable; including a weak one costs only scoring time.

**Module:** `selection/candidates.py`

**A. LLM windows.** 5-minute windows with a 1-minute overlap (about 15 per hour). Prompt = bible + window timeline + the existing `VIRALITY_CRITERIA` and `PLATFORM_SIGNALS` text from `highlights.py`. Ask for up to 8 candidates per window, each `{start, end, kind, why}` where `kind` is one of the virality categories. Run each window twice at temperature 0.7 and take the union, so one unlucky sample doesn't drop a moment.

**B. Signal peaks.** From `features.parquet`, propose a 30–60 s span around each:

- laughter probability above 0.6 for at least 1.5 s
- loudness more than 6 dB above the episode median
- 4 or more speaker changes in 20 s (rapid back-and-forth)
- a `payoff_t` or running-joke time from the bible

Spans start at the nearest sentence start up to 30 s before the peak and end at a sentence end after it.

**Merge.** Snap every candidate with `boundaries.snap_clip`, then merge any pair with temporal IoU above 0.6, keeping the union span and both sources in `origin: ["llm", "laughter"]`. Drop spans outside the length range of `--target-length` (existing `TARGET_LENGTHS`).

**Candidate record**

```json
{"id": "c017", "start": 1664.3, "end": 1701.75, "origin": ["llm", "laughter"],
 "kind": "opinion_bomb", "why": "...", "window_ids": [11, 12]}
```

**Acceptance:** recall of at least 90% against the gold clips in the eval set (a gold clip counts as found when any candidate overlaps it with IoU of at least 0.5).

## Stage 3: Decomposed multimodal scoring

Stage 3 asks Gemma six narrow questions about each candidate, each sampled several times, and stores every raw answer in `scores.jsonl`. Nothing here produces a final score; Stage 4 does that.

**Module:** `selection/score.py`

**Step 3a: Perception (audio + frames).** For each candidate, cut audio into 30 s chunks (Gemma's per-clip audio limit) and pick frames at 1 fps, at most 60, at the 70- or 140-token image budget. One `mm_fn` call per 30 s chunk returns a structured description:

```json
{"delivery": "animated|flat|...", "emotion": ["amused", "incredulous"],
 "laughter": true, "reactions_visible": "guest leans back, covers face",
 "energy_1to5": 4, "notes": "..."}
```

Per the model card's guidance, images go before the text and audio after it in each prompt. These descriptions are cached and reused by every question below.

**Step 3b: Narrow questions.** Each is its own call with the bible, the clip transcript and the 3a description. Answers are schema-constrained.

| Key | Question | Answer |
| --- | --- | --- |
| `hook` | Would the first 3 seconds stop a stranger scrolling? | 1–5 + quote of the opening line |
| `payoff` | Does the clip land a punchline, reveal or conclusion before it ends? | yes / partial / no |
| `emotion` | Is there a clear emotional peak? (audio attached) | 1–5 |
| `standalone` | Can someone who never heard this podcast follow it? | 1–5 |
| `novelty` | Is it funny, surprising, contrarian or useful? | one label + 1–5 |
| `ending` | Does it end on a complete thought, not mid-argument? | yes / no |

**Self-consistency.** Each question has 3 prompt wordings in `selection/prompts/<key>_v{1,2,3}.md`. Answers are sampled at temperature 0.7 with thinking mode on, spread evenly across the wordings (`<|think|>` in the system prompt), SCORE\_SAMPLES answers per question (default 3). Store them all; Stage 4 uses the mean and the spread.

**Call budget per hour of podcast.** Default `SCORE_SAMPLES=3` (one sample per wording): about 80 candidates × (2 perception + 6 questions × 3) ≈ 1,600 calls. At roughly 2–3 minutes per call on a 4-core CPU (about 1.5k new prompt tokens after the cached bible prefix, plus a few hundred thinking tokens), that is 2–3 days. `SCORE_SAMPLES=6` doubles it. These are estimates; the run logs real per-call timings, and the early exit below cuts the total substantially.

**Early exit.** After the first wording of `hook` and `standalone`, drop candidates that score 1 on both. This typically halves the remaining calls.

## Stage 4: Learned combiner

Stage 4 turns Stage 3's answers and Stage 0's signals into one score per candidate with a small model trained on your labels. This is the main way a 12B judge gets better without a bigger model.

**Module:** `selection/combiner.py`, model file `models/combiner.joblib`

**Features per candidate (about 25)**

- Stage 3: mean and standard deviation of each of the 6 questions, the perception `energy_1to5`, and whether 3a heard laughter
- Stage 0: peak and mean laughter probability, loudness peak relative to the episode median, pitch range, speech rate, speaker changes per minute, shot changes per minute, faces on screen
- Structure: duration, `origin` flags from Stage 2, whether the clip contains a bible `payoff_t`, position in the episode (fraction)
- Text: the `research/benchmark.py` hook flags on the opening line (question, number, "you", contrarian)

**Model.** Logistic regression with L2 regularisation (scikit-learn), target = "is a gold clip" from the eval set. With fewer than about 150 labelled positives, keep it this simple; switch to gradient boosting (LightGBM) only when the eval shows logistic regression has plateaued.

**Cold start.** Before any labels exist, use a hand-set weighted sum in `config/combiner_default.yaml` (hook 0.3, standalone 0.2, payoff 0.2, emotion 0.15, novelty 0.15, minus 0.5 × the standard deviation as an uncertainty penalty). Replace it as soon as the eval set has 10 labelled episodes.

**Training command:** `python -m selection.combiner train --eval eval/ --out models/combiner.joblib`. It prints cross-validated precision@5 for the default weights and the trained model, and refuses to save a model that does worse than the default.

**Output:** `combined.jsonl`, candidates sorted by score, top 20 passed on. The weights are logged, so you can see what the model learned matters for your niche.

## Stage 5: Boundary refinement and cold viewer test

Stage 5 tunes the start and end of the top 20 candidates and drops any that a viewer without context can't follow.

**Module:** `selection/refine.py`

**5a. Boundary options.** For each candidate, generate up to 6 variants using `boundaries.py` sentence breaks: the original; start 1 and 2 sentences later; start on the line Stage 3 quoted as the best hook; end 1 sentence earlier; end 1 sentence later. Discard variants outside the target length.

**5b. Pick the best variant.** Show Gemma the variants' first and last 2 sentences side by side and ask which gives the strongest opening and cleanest ending, as a pairwise choice in both orders (same method as Stage 6). Keep the winner.

**5c. Cold viewer test.** In a fresh context with no bible, give Gemma only the clip's transcript and its audio, and ask:

1. What is this clip about, in one sentence?
2. Is anything referenced that you can't understand? Quote it.
3. Would you watch to the end? yes / no and why.

Then one more call, with the bible, compares the cold summary from question 1 against the bible's description of that moment and returns `understood: true|false`. If false, or if question 2 quotes something, try extending the start back by up to 2 sentences and re-test once. If it still fails, drop the candidate and log the reason.

**Hook-first cold open (optional, off by default).** When the best hook line sits more than 8 s into the clip, offer a variant that plays that line first, then cuts back to the clip start. The renderer needs a two-segment concat in `local/clipper.py` for this; flag it as `cold_open: {t0, t1}` in the record.

**Acceptance:** mean boundary error against hand-set boundaries in the eval set drops by at least 30% compared with plain `snap_clip`.

## Stage 6: Pairwise tournament and variety

Stage 6 orders the surviving candidates (usually 12–18) by head-to-head comparisons, then picks a varied final set. It replaces `rerank_highlights`, which rates 12 clips in one call.

**Module:** `selection/tournament.py`

**Comparisons.** For every pair, one call with the bible, both transcripts and both Stage 3a descriptions asks: "Which clip would more people watch to the end and send to a friend? Answer A or B." Ask again with A and B swapped. Record a win only when both orders agree; a disagreement counts as a tie. Small models have strong position bias, and this cancels it.

With 16 candidates that is 120 pairs × 2 orders = 240 calls. Above 18 candidates, switch to a Swiss-system schedule (5 rounds) to keep calls roughly linear.

**Ranking.** Fit Bradley–Terry strengths from the win/tie matrix (the `choix` package, or a 30-line implementation). Final score = 0.6 × Bradley–Terry strength rank + 0.4 × Stage 4 combiner rank. Both weights go in config and get tuned on the eval set.

**Variety (MMR).** Pick clips one at a time. Each next pick maximises `score − λ × max_similarity` to clips already picked, where similarity is 1 if two clips share a bible chapter or story and 0.5 if they share a speaker-only topic. Default λ = 0.3.

**Output:** the top `--num-clips` written as the existing highlight dicts (`title`, `start_time`, `end_time`, `score`, `hook_sentence`, `on_screen_hook`, `virality_reason`), so `generate_metadata` and `crop_highlights_local` work unchanged. Extra fields (`bt_strength`, `combiner_score`, `sub_scores`, `cold_test`) ride along for debugging and the eval.

The text fields don't exist yet at this point, so one final call per picked clip writes `title`, `hook_sentence`, `on_screen_hook` and `virality_reason`, reusing the item schema from `HIGHLIGHT_SCHEMA`. `score` is the final score scaled to 0–100.

## Stage 7: Feedback loop and fine-tuning

Stage 7 is what makes the pipeline improve over time: it records how posted clips perform and feeds that back into the few-shot examples, the combiner and eventually Gemma itself.

**Module:** `selection/feedback.py`, data in `feedback/clips.csv`

**7a. Log every posted clip.** One row per clip: episode, clip id, start, end, all Stage 3–6 fields, platform, post date, and after 7 days: views, average view duration, percent viewed, 3-second hold rate, shares. Entry is manual at first (copied from YouTube Studio); later a `yt-dlp`-based fetcher like `research/benchmark.py` can fill views for public Shorts. Your clipping campaign posts count too.

**7b. Performance label.** Within each channel, label a clip `good` if its percent viewed and views are both in the top third for that channel's last 30 posts, `bad` if both are in the bottom third. Compare within a channel only, because channels differ in baseline reach.

**7c. Use the labels**

- Combiner: retrain monthly with performance labels added to the eval-set labels (performance labels weighted 2×, since they measure what you actually care about).
- Few-shot: the 10 best-performing clip openings replace `--examples` in Stage 2 and Stage 3 `hook` prompts. This extends the existing `HIGHLIGHT_EXAMPLES` mechanism.
- Prompt A/B: a new prompt wording runs in shadow mode on the eval set and only replaces the old one if precision@5 improves.

**7d. LoRA fine-tune of Gemma 4 12B (later).** Once there are at least 300 labelled candidates (eval-set gold plus performance labels), train a LoRA adapter on two tasks: the pairwise "A or B" judgment from Stage 6, and the narrow Stage 3 questions, with your labels as targets. Train on a rented GPU or Colab (QLoRA, rank 16, 1–2 epochs); a 16 GB CPU machine can't train a 12B model in reasonable time. Merge or load the adapter for local inference, then keep it only if it beats the base model on the held-out eval episodes.

## Model serving, prompts, config and resumability

Text calls go through the existing Ollama client; audio and image calls need a second backend, because audio input for Gemma 4 12B is officially supported only in Hugging Face Transformers ([Creeta, 2026](https://news.creeta.com/en/gemma-4-12b-ollama-huggingface-llama-cpp-2026/)).

**Two backends behind one interface** (`shorts_generator/local/llm.py`)

```python
def call_local_llm(prompt: str, schema: dict | None = None) -> str: ...      # exists
def call_local_mm(parts: list[Part], schema: dict | None = None) -> str: ...  # new
# Part = {"type": "text"|"image"|"audio", "text"|"path": ..., "budget": 70|140}
```

| Backend | Used for | Model | Notes |
| --- | --- | --- | --- |
| Ollama | All text-only calls (Stages 1, 2, 3b, 4, 5b, 6) | `gemma4:12b`, Q4 (listed at 7.6 GB) | Existing client; set `OLLAMA_NUM_CTX=32768` for the Stage 1 cross-check, 8192 elsewhere |
| `selection/mm_server.py` (Transformers) | Stage 3a perception, Stage 3b `emotion`, Stage 5c | `google/gemma-4-12B-it` | Small FastAPI wrapper so the model loads once per stage |

Only one backend fits in 16 GB at a time, so the runner executes **stage by stage**: all text calls of a stage, unload (`release_ollama`), then all multimodal calls, unload. Never interleave.

**Prefix caching.** Every judging prompt starts with the same bytes: system prompt, then bible, then the task. Keeping that prefix identical lets the runtime's prompt cache skip re-reading it. Log `prompt_eval_count` per call; if it doesn't drop after the first call of a stage, the cache isn't being hit.

**Prompts.** One Markdown file per prompt in `selection/prompts/`, with front matter `{id, version, schema}`. The prompt id and version are stored with every answer, so results stay traceable when wordings change.

**Resumability = the harness idea, generalised.** Every LLM call is keyed by `sha256(backend + model + prompt + attachments)` and its response saved to `work/<episode>/calls/<hash>.json`. A rerun reads cached answers instead of calling the model, so a crash resumes where it stopped, and `harness/run.py` replays any run deterministically. `--fresh <stage>` deletes that stage's cache.

**New config** (`.env` and the `gemma-deep` profile)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SELECTOR` | `classic` | `classic` = current `highlights.py`, `deep` = this pipeline |
| `OLLAMA_MODEL` | `gemma4:12b` | in `gemma-deep` |
| `MM_BACKEND` | `transformers` | multimodal backend; `none` skips audio and image calls |
| `MM_MODEL` | `google/gemma-4-12B-it` |  |
| `WINDOW_SECONDS` / `WINDOW_OVERLAP` | 300 / 60 | Stage 2 windows |
| `SCORE_SAMPLES` | 3 | answers per Stage 3 question |
| `TOURNAMENT_MAX` | 18 | above this, Swiss rounds |
| `MMR_LAMBDA` | 0.3 | variety penalty |
| `WORK_DIR` | `work` | per-episode artifacts and call cache |

`MM_BACKEND=none` must still produce a full run, using Stage 0's prosody and laughter features in place of Gemma's listening. That is the fallback if multimodal inference doesn't fit on the MateBook.

## Evaluation harness

No stage ships unless it improves these numbers on held-out episodes; the eval set is built first, before any stage code.

**Data:** `eval/<episode>/gold.json`, 15–20 episodes, 5–10 gold clips each.

```json
{"episode": "cordkillers-377", "source": "input/cordkillers377.mp4",
 "gold": [{"start": 1664.3, "end": 1701.8, "rating": 5, "origin": "official_short", "views": 41200}]}
```

Two label sources: the podcast's own official Shorts (matched to the episode by transcript text), and your hand picks rated 1–5. Split episodes 70/30 into `tune` and `test`; the combiner and prompt choices only ever see `tune`.

**Runner:** `python -m selection.eval --selector deep --split test`. It runs the full pipeline (with the call cache, so repeats are cheap) and writes `eval/report.md`.

| Metric | Definition | Target vs `classic` |
| --- | --- | --- |
| Recall@candidates | Gold clips matched by any Stage 2 candidate, IoU ≥ 0.5 | ≥ 90% absolute |
| Precision@5 | Top 5 picks that match a gold clip, IoU ≥ 0.5 | +15 points |
| NDCG@10 | Ranking quality using gold ratings as relevance | +0.1 |
| Boundary error | Mean seconds between matched start/end and gold | −30% |
| Cold-test pass rate | Picks a fresh context can summarise correctly | ≥ 80% |
| Stability | Mean Jaccard overlap of the top 5 across 3 runs with different seeds | ≥ 0.6 |
| Cost | Wall-clock hours and LLM calls per audio hour | Logged, no target |

**Baselines in every report:** random spans of the target length; the signal-only ranker (Stage 2B peaks sorted by laughter plus loudness); and the current `classic` selector with Ollama. A stage that can't beat the signal-only ranker is not earning its compute.

**Ablations:** the report can rerun with any stage disabled (`--skip bible`, `--skip mm`, `--skip tournament`, `--samples 1`) so you can see what each stage is worth, and drop the ones that don't pay for their hours.

**Tests:** unit tests next to the existing ones in `tests/` for IoU matching, MMR, Bradley–Terry fitting, window merging and the call cache, all runnable without a model.

## Milestones and tasks

Build in this order. Each milestone ends with an eval report, and a stage stays only if the report shows it helps.

**M0: Measure first**

- [ ] `eval/` format, IoU matching, `selection.eval` runner and report
- [ ] Label 15 episodes (official Shorts + hand picks)
- [ ] Baseline report: random, signal-only, `classic` with `qwen3:8b` and with `gemma4:12b`

**M1: Plumbing**

- [ ] `selection/` package, `--selector deep`, `gemma-deep` profile, `WORK_DIR` layout
- [ ] Call cache keyed by hash; `harness/run.py` replays it
- [ ] Prompt files with front matter

**M2: Signals and bible**

- [ ] Stage 0 extractors and `timeline.txt`, `features.parquet`
- [ ] Stage 1 map-reduce bible
- [ ] Stage 2 windows + signal peaks; check recall ≥ 90%

**M3: Text-only scoring**

- [ ] Stage 3b questions with `MM_BACKEND=none`
- [ ] Stage 4 combiner, default weights, then trained
- [ ] Stage 6 tournament + MMR, Stage 6 text-field call
- [ ] Eval: deep (text-only) vs classic

**M4: Multimodal**

- [ ] `mm_server.py` and `call_local_mm`; smoke test of audio + image input on the MateBook
- [ ] Stage 3a perception and audio-backed `emotion`
- [ ] Stage 5 refinement and cold viewer test
- [ ] Eval with `--skip mm` as the ablation

**M5: Feedback**

- [ ] `feedback/clips.csv` logging and labels
- [ ] Monthly combiner retrain; few-shot refresh
- [ ] LoRA fine-tune once 300 labels exist

**M6: Cleanup**

- [ ] README section for `--selector deep`; point install steps and the bundled skill at this fork, not upstream
- [ ] Unit tests for every new pure function

## Open questions

- Does Gemma 4 12B audio + image inference fit and run acceptably on the 16 GB MateBook through Transformers on CPU? If not: run Stage 3a and 5c on a rented GPU, or ship with `MM_BACKEND=none`.
- Does your installed Ollama version accept audio for `gemma4:12b`? If yes, the Transformers server can be dropped.
- Which language are most of your source podcasts in? Hindi or Hinglish changes Whisper settings, token counts and the prompt wording.
- How many hours per episode are you actually willing to wait in practice? That sets the default for `SCORE_SAMPLES` and the tournament size.
- Which channels' official Shorts should seed the eval set?
