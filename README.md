# AI YouTube Shorts Generator

[![Powered by MuAPI](https://img.shields.io/badge/Powered%20by-MuAPI-6366f1?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNHYtNGgtMnYtMmg0djZoLTJ6bTAtOFY2aDJ2MmgtMnoiLz48L3N2Zz4=)](https://muapi.ai?utm_source=github&utm_medium=badge&utm_campaign=ai-youtube-shorts-generator)


**The open-source alternative to Opus Clip, Vidyo.ai, Klap, SubMagic, 2short.ai, and other AI clipping tools.** Drop in any long-form YouTube video and get back ranked, viral-ready 9:16 shorts — for free, with no per-clip credits, no watermarks, and full control over the highlight algorithm.

Built for creators, agencies, and developers who don't want to pay $20–$300/month or be capped on minutes processed. Uses GPT-class LLM highlight detection and Whisper transcription to extract the most viral-worthy moments and auto-crop them vertically for TikTok, Reels, and Shorts.

<p align="center"><a href="https://www.youtube.com/watch?v=HD47qeTJgtw"><img src="https://i.ytimg.com/vi/HD47qeTJgtw/maxresdefault.jpg" width="720"></a></p>
<p align="center"><a href="https://www.youtube.com/watch?v=HD47qeTJgtw"><b>▶  Watch: Free Unrestricted AI Image & Video Generator (No Filters, No Limits) </b></a></p>

> **Building your own Opus Clip–style SaaS?** Skip the infra and ship on the same APIs that power this repo:
> - [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) — end-to-end clip selection + render
> - [Auto-Crop API](https://muapi.ai/playground/autocrop?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) — vertical reframing only

![longshorts](https://github.com/user-attachments/assets/3f5d1abf-bf3b-475f-8abf-5e253003453a)

<p align="center">
  <a href="https://github.com/Anil-matcha/awesome-generative-ai-apps">
    <img src="https://img.shields.io/badge/Part%20of-Awesome%20Generative%20AI%20Apps-FFD700?style=for-the-badge&logo=github&logoColor=black" alt="Awesome Generative AI Apps">
  </a>
</p>

> 🎨 **[Explore 50+ more open-source AI apps →](https://github.com/Anil-matcha/awesome-generative-ai-apps)**

## Why Use This Instead of Opus Clip / Vidyo.ai / Klap?

| | This repo | Opus Clip / Vidyo.ai / Klap / SubMagic |
|---|---|---|
| **Price** | Free + open source (pay only for API usage) | $20–$300/month subscriptions |
| **Per-clip credits** | None — process unlimited videos | Monthly minute caps, overage fees |
| **Watermarks** | Never | On free tiers |
| **Highlight algorithm** | Fully editable virality framework | Black box |
| **Output format** | Any aspect ratio, any resolution | Locked presets |
| **Batch processing** | `xargs` an entire URL list | Manual upload one-by-one |
| **JSON / API output** | Built-in (`--output-json`) | Limited or paid tier only |
| **Self-hostable** | Yes — runs on your machine or server | SaaS only, your videos sit on their servers |
| **White-label / embeddable** | Yes — MIT licensed, import as Python lib | No |

## Features

- **🎬 YouTube In, Vertical Out**: Hand it any YouTube URL — get back N viral-ready 9:16 mp4s
- **🔀 Two Modes — API (fast) or Local (offline)**: Default `--mode api` uses MuAPI for download/transcription/cropping; `--mode local` runs entirely on your machine with `yt-dlp`, `faster-whisper`, and `ffmpeg`/`opencv`, and lets you pick OpenAI or Gemini for highlight ranking
- **🤖 Virality-Aware Highlight Selection**: Clips ranked on hooks, emotional peaks, opinion bombs, revelation moments, conflict, quotable lines, story peaks, and practical value — not just generic "interesting"
- **📈 Score + Hook + Reason for Every Clip**: Each highlight comes with a viral score, an opening hook line, and a one-sentence explanation of why it works
- **🎤 Whisper Transcription, Your Choice**: Cloud (`/openai-whisper` via MuAPI) or local (`faster-whisper`, CPU or CUDA) — same downstream output shape
- **🧩 Long-Video Aware**: Videos over 30 minutes are auto-chunked with overlap so nothing gets missed
- **♻️ Smart Dedupe**: Overlapping highlights are collapsed by score so you never get two near-duplicate clips
- **🎯 Smart Vertical Crop**: API mode uses MuAPI's auto-crop; local mode runs OpenCV face tracking with motion smoothing
- **📱 Any Aspect Ratio**: 9:16 for TikTok/Reels/Shorts, 1:1 for square, anything else by flag
- **🧰 CLI + Python Library**: Use it from the shell or import `generate_shorts(...)` into your own pipeline
- **📦 JSON Output**: `--output-json` dumps the full result (transcript + every candidate highlight + final clip URLs/paths) for downstream automation

## Quick Start (No Setup)

Don't want to self-host? The [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) gives you the same Opus Clip–style pipeline as a single HTTP call — no Python, no dependencies, pay-per-clip instead of monthly subscriptions.

---

## Installation (Self-Hosted)

### Prerequisites

- Python 3.10+
- For **API mode (default)**: a MuAPI key — powers download, transcription, highlight ranking, and clipping in a single dependency
- For **Local mode** (`--mode local`): `ffmpeg` on your PATH and an LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`; only the LLM step is remote)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AaravSachdeva/shorts-clipper.git
   cd shorts-clipper
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   # Only if you plan to use --mode local:
   pip install -r requirements-local.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   # API mode (default)
   MUAPI_API_KEY=your_muapi_key_here

   # Local mode (--mode local)
   LLM_PROVIDER=openai         # openai or gemini
   OPENAI_API_KEY=your_openai_key_here
   OPENAI_MODEL=gpt-4o-mini          # optional, default gpt-4o-mini
   GEMINI_API_KEY=your_gemini_key_here
   GEMINI_MODEL=gemini-2.5-flash      # optional, default gemini-2.5-flash
   LOCAL_WHISPER_MODEL=base          # tiny / base / small / medium / large-v3
   LOCAL_WHISPER_DEVICE=auto         # auto (GPU if found, CPU fallback) / cpu / cuda
   LOCAL_OUTPUT_DIR=output           # where local mp4s land
   ```

## Usage

### Single video (API mode — default)

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Single video (Local mode — runs offline except for the LLM call)

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local
```

Give it the YouTube link and local mode downloads the video itself with `yt-dlp`. YouTube downloads need Node.js or Deno on the `PATH`: `yt-dlp[default]` from `requirements-local.txt` installs the challenge solver that runs on them.

Local mode writes the rendered shorts to `./output/<video name>/short_01.mp4`, `short_02.mp4`, … (override `output` with `LOCAL_OUTPUT_DIR`). The folder is named after the YouTube title, or after the file name for a local file, so each long video keeps its own shorts. A different video with the same title gets `<title> (2)`, tracked by a `.source` file inside each folder. Each clip is 1080×1920 H.264. It comes with burned-in word-highlight captions, an on-screen hook for the first 3 s, and audio normalized to −14 LUFS. Next to each clip, `short_01.txt` holds a ready-to-paste title, caption and hashtags, and `short_01.json` holds the full metadata.

### Fully offline with Ollama (no API keys)

Clip selection can run on a local model through [Ollama](https://ollama.com). Nothing leaves your machine.

1. Install Ollama, then pull a model:
   ```bash
   ollama pull qwen3:8b      # ~5 GB; or qwen3:14b (~9 GB) for better picks
   ```
2. Run with a profile:
   ```bash
   python main.py /path/to/episode.mp4 --mode local --profile cpu-slow
   ```

| Profile | Model | Whisper | Good for |
|---|---|---|---|
| `cpu-slow` | `qwen3:8b`, 16K context, 10-min chunks | `large-v3-turbo` (CPU) | ~16 GB RAM, little or no GPU. Clips from a 1-hour episode in a few hours |
| `cpu-quality` | `qwen3:14b`, 16K context | `large-v3-turbo` (CPU) | 16 GB RAM with other apps closed. Slower, better picks |

To set things up without a profile, use `LLM_PROVIDER=ollama` with `OLLAMA_MODEL`, `OLLAMA_NUM_CTX` and `OLLAMA_HOST` (see `.env.example`).

Notes:
- **Context size matters.** Ollama's default context window is small, and it **silently truncates** longer prompts. The client always sends `num_ctx` (default 16384) and warns if a prompt filled the window. A bigger `num_ctx` uses more RAM: roughly 2–3 GB extra for an 8B model at 16K.
- **Structured output.** Every call passes a JSON schema through Ollama's `format`, so even small models return valid JSON. The output is still validated and retried.
- **Freeing memory.** The model is unloaded (`keep_alive: 0`) after clip selection, so its RAM is free during rendering.
- **Timings.** The run prints how long each step took: transcribe, LLM, render.
- **Measured on a 4-core CPU, 16 GB RAM, no GPU (59-min episode):**
  - Whisper `small` transcription took 27 min and peaked at 3.7 GB RAM. `large-v3-turbo` is slower and more accurate.
  - Rendering 3 clips (2.5 min of video in total) took about 7 min.
  - Ollama step not measured yet. Expect roughly 1–2 h for an 8B model on CPU; the run prints the real figure.

### Deep selector (experimental)

`--selector deep` replaces the one-pass highlight step with a multi-stage pipeline in `shorts_generator/selection/`. It builds a short summary of the whole episode first (the "bible"), proposes candidates from 5-minute windows and from loudness peaks, asks six narrow questions about each candidate (hook, payoff, emotion, standalone, novelty, ending), combines the answers with fixed weights, ranks the best ones in a pairwise tournament asked in both orders, and spreads the final picks across the episode's chapters. It reads text only for now.

During development the model behind it is Claude Haiku, run through headless `claude -p` (needs the Claude Code CLI, signed in):

```bash
python main.py output/source_<id>.mp4 --mode local --profile claude-deep --num-clips 3
```

Each call is a fresh `claude -p` process with no tools, no settings, no MCP servers and no project files; the model still sees Claude Code's short fixed preamble (about 600 tokens). Calls count against your Claude plan's usage limits, and one 2–3 hour episode is on the order of a thousand calls. Every answer is cached in `work/<episode>/calls/`, so a run that stops on a usage limit or a crash resumes where it left off; `work/<episode>/deep.json` shows the bible, every candidate's answers and the picks. `--profile gemma-deep` runs the same pipeline on Ollama's `gemma4:12b` and is untested.

Measure it before trusting it. Build gold clips for an episode from the channel's official Shorts (matched by caption text), then compare selectors:

```bash
python -m shorts_generator.selection.gold --shorts "https://www.youtube.com/@channel/shorts" --episode output/source_<id>.mp4
LLM_PROVIDER=claude LLM_WORKERS=4 HIGHLIGHT_CHUNK_SECONDS=300 \
  python -m shorts_generator.selection.eval --selectors random,signal,classic,deep --split all
```

The report goes to `eval/report.md` (precision of the top 5, NDCG@10, candidate recall, boundary error, LLM calls). `--skip bible`, `--skip tournament` and `--samples 3` rerun the deep selector without a stage, to see what each one is worth.

### With options

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --mode api \
    --num-clips 5 \
    --aspect-ratio 9:16 \
    --output-json result.json
```

### Local file or path

In `--mode local`, you can pass a `file://` URL or a direct filesystem path and skip YouTube entirely:

```bash
python main.py "/Users/you/Videos/input.mp4" --mode local
python main.py "file:///Users/you/Videos/input.mp4" --mode local
```

The Python API works the same way:

```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "/Users/you/Videos/input.mp4",
    num_clips=5,
    aspect_ratio="9:16",
    mode="local",
)
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

Local transcription is cached in `LOCAL_OUTPUT_DIR` under the video's base name, as `<name>.json` (segments plus word timestamps) and `<name>.srt`. If the cache is newer than the source file, the app reuses it instead of running Whisper again. An older `.srt`-only cache has no word timings, so that video is transcribed again once.

Local downloads are also cached in `LOCAL_OUTPUT_DIR` as
`source_<youtube_id>.mp4` when the input is a YouTube URL. If that file already
exists, the app skips `yt-dlp` and reuses the cached video.

### NVIDIA GPU (Whisper on GPU, Ollama split across GPU and CPU)

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # CUDA 12 libraries for Whisper, ~1 GB
```

With `LOCAL_WHISPER_DEVICE=auto` (the default, and what both profiles use), Whisper runs on the GPU when CTranslate2 finds one. It uses the smallest fast compute type the card supports (`int8` on Pascal cards such as the MX150, `int8_float16` on newer ones). The app loads the pip CUDA libraries itself, so `LD_LIBRARY_PATH` is not needed. If the GPU fails (out of memory, missing libraries), transcription falls back to the CPU and says so. large-v3-turbo peaks at about 1.3 GB of VRAM in `int8`.

Ollama splits layers between the GPU and the CPU itself, but its auto-fit keeps about 1.2 GB of VRAM free, which on a 2 GB card can mean no layers at all. `OLLAMA_NUM_GPU` sets the layer count yourself; too many fails the load with `cudaMalloc failed: out of memory`, so measure. Measured on a 2 GB MX150: gemma4:12b at `OLLAMA_NUM_CTX=65536` got 0 of 49 layers on auto; `OLLAMA_NUM_GPU=2` generated about 10% faster (2.33 vs 2.12 tok/s), and 3 or more ran out of VRAM. With llama3.1:8b at 16k context, auto (2 of 33) was as fast as forcing 6. Whisper finishes and frees the GPU before Ollama starts.

Reasoning models such as gemma4 and qwen3 think before they answer. On a CPU that took about 90% of each window's time, and gemma4:12b once thought for over an hour on one window without answering, so it timed out. Thinking is off by default; set `OLLAMA_THINK=true` to turn it on.

### Review dashboard (approve or reject from your phone)

`review.py` serves a small web page for reviewing local-mode shorts. Shorts are grouped by the long video they came from, and each card plays the clip and shows its title, caption and hashtags. Approve, Reject and Download buttons sit under each clip.

1. Set `REVIEW_TOKEN` in `.env` to a long random string:
   ```bash
   python3 -c 'import secrets;print(secrets.token_urlsafe(24))'
   ```
2. Start the server:
   ```bash
   python review.py              # --host 0.0.0.0 --port 8765 by default
   ```
3. On your phone, connected to the same Wi-Fi, open `http://<this machine's LAN IP>:8765/?token=<REVIEW_TOKEN>` once. The page saves a cookie and then drops the token from the address bar. Bookmark `http://<LAN IP>:8765/`.

To make shorts from a new video, paste its YouTube link into the box at the top of the page and tap **Make shorts**. The server runs `main.py` on it in the background, one video at a time, and the page shows whether it is running, done or failed (with the last lines of `output/.generate.log`). The new shorts then appear under Pending. Only YouTube video links are accepted. Pass pipeline flags with `--pipeline-args`, e.g. `python review.py --pipeline-args "--mode local --profile cpu-slow"` (default `--mode local`).

Approve and Reject only move files, and nothing is ever deleted. A short's files (`.mp4`, `.json`, `.txt`) move into `approved/` or `rejected/` inside the video's folder:

```
output/
  My Podcast Episode 12/
    short_03.mp4  short_03.json  short_03.txt   ← pending
    approved/short_01.mp4 …
    rejected/short_02.mp4 …
```

"Move back" returns a short to pending. If a name is already taken, the moved file gets a suffix (`short_01-2.mp4`) and nothing is overwritten.

Notes:
- Local mode only. API mode returns hosted MuAPI URLs, not files.
- The page is plain HTTP, so anyone on the same network can see the token in transit. Use it on your home Wi-Fi only, never exposed to the internet.
- If your phone can't connect, the server's firewall may be blocking port 8765, for example `sudo ufw allow from 192.168.0.0/16 to any port 8765`.

### Batch processing

Create a `urls.txt` file with one URL per line, then:

```bash
xargs -a urls.txt -I{} python main.py "{}"
```

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `api` | `api` (MuAPI, fast, no setup) or `local` (remote URL, `file://`, or local path + faster-whisper + LLM provider + ffmpeg) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Any ratio; `9:16` for TikTok/Reels, `1:1` for square |
| `--format` | `720` | Source download resolution: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language code (e.g. `en`) |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |
| `--profile` | — | `cpu-slow` or `cpu-quality`: Ollama + CPU Whisper presets (local mode); `claude-deep` or `gemma-deep`: the deep selector |
| `--llm-provider` | `LLM_PROVIDER` | `openai`, `gemini`, `ollama` or `claude` (local mode) |
| `--target-length` | `medium` | `short` ≈15–35 s, `medium` ≈20–60 s, `long` ≈45–90 s |
| `--no-captions` | — | Don't burn in captions |
| `--no-hook` | — | Don't show the on-screen hook title for the first 3 s |
| `--crop-bottom` | `0` | Cut this % of frame height off the bottom first (burned-in tickers or chat bars) |
| `--no-rerank` | — | Skip the comparative re-rank of the top candidates |
| `--examples` | — | JSONL of top-performing clip openings (from `research/benchmark.py`) |
| `--selector` | `classic` | `classic` (one pass per window, then a re-rank) or `deep` (experimental, see above) |

### API mode vs Local mode

| Step | API mode (`--mode api`) | Local mode (`--mode local`) |
|---|---|---|
| Download | MuAPI `/youtube-download` | `yt-dlp` for remote URLs, direct file path for local inputs |
| Transcription | MuAPI `/openai-whisper` | `faster-whisper` (CPU or CUDA) |
| Highlight LLM | MuAPI `gpt-5-mini` | `LLM_PROVIDER=openai` uses OpenAI (`gpt-4o-mini` by default), `gemini` uses Gemini (`gemini-2.5-flash`), `ollama` uses a local model (`qwen3:8b`) |
| Vertical crop | MuAPI `/autocrop` | `ffmpeg` + OpenCV: one steady crop per camera shot, blur-fill when no face is on screen, captions and hook overlay burned in |
| Output | hosted URLs | local mp4 paths |
| Required keys | `MUAPI_API_KEY` | `OPENAI_API_KEY` or `GEMINI_API_KEY`, or none with Ollama (+ `ffmpeg` on PATH) |

## How It Works

1. **Download**: Fetches the source video from YouTube
2. **Transcribe**: MuAPI `/openai-whisper` produces a timestamped transcript (verbose_json segments)
3. **Detect content type**: An LLM classifies the video (podcast, interview, tutorial, vlog, etc.) and density, so the prompt can be tuned per content style
4. **Long-video chunking**: Videos over 30 minutes are split into overlapping chunks (20 min, or 10 min with Ollama). Each chunk's timestamps are relative to that chunk.
5. **Highlight ranking**: An LLM scans the transcript using a virality framework (hook moments, emotional peaks, opinion bombs, revelations, conflict, quotables, story peaks, practical value), plus what Shorts and Reels reward: a hook in the first 1–3 s, completion over length, and moments people send to friends. It skips ads and outros. For each candidate it returns a score, a hook score, a shareability score and a 3–8 word on-screen hook.
6. **Boundary snapping**: Word timestamps move each clip to start on a sentence (skipping "so", "um", "yeah") and end just after one.
7. **Dedupe**: Overlapping candidates are collapsed by score (more than 50% overlap keeps the higher score).
8. **Global re-rank**: The top 12 are re-scored against each other in one call, so scores from different chunks are comparable.
9. **Top-N selection and post metadata**: The top `--num-clips` are kept, and one call writes a title, caption and hashtags for each.
10. **Render**: Each clip is cut, reframed to 9:16 (steady per-shot crop, or blur-fill when there's no face), captioned, given the hook overlay, loudness-normalized, and encoded as 1080×1920 H.264.

**Output**: a list of mp4 URLs plus, for each clip, its title, viral score, hook sentence, and a one-line reason explaining why it should perform.

## Output

Console output looks like:

```
========================================================================
Highlights:    7 candidates → kept top 3
========================================================================

#1  score=92  124.3s → 187.6s
     title:  The one mistake that cost me $50K
     hook:   "Nobody talks about this, but it killed my first startup..."
     clip:   https://.../short_1.mp4

#2  score=88  ...
```

`--output-json result.json` produces:

```json
{
  "source_video_url": "...",
  "transcript": { "duration": 1873.4, "segments": [...] },
  "highlights": [ {...}, {...}, ... ],
  "shorts": [
    {
      "title": "...",
      "start_time": 124.3,
      "end_time": 187.6,
      "score": 92,
      "hook_sentence": "...",
      "virality_reason": "...",
      "clip_url": "https://.../short_1.mp4"
    }
  ]
}
```

## Configuration

### Highlight selection criteria
Edit `shorts_generator/highlights.py`:
- **Virality framework**: `VIRALITY_CRITERIA` — the ranked list of signals the LLM optimizes for
- **System prompt**: `HIGHLIGHT_SYSTEM_PROMPT` — duration sweet spot, hook rules, JSON schema
- **Rubric**: `RUBRIC` and `RUBRIC_WEIGHTS` — the model scores every clip 0-10 on hook, standalone and payoff; the overall 0-100 score is computed from those (hook 40%, standalone 30%, payoff 30%), not taken from the model
- **Window size**: `HIGHLIGHT_CHUNK_SECONDS` in `.env` (default 300 with Ollama, else 1200) — the transcript goes to the model one window at a time, never all at once, and the top candidates are re-ranked against each other in a second pass
- **Window overlap**: `CHUNK_OVERLAP_SECONDS` (default 60) — overlap between windows so cross-boundary clips aren't missed

### Polling / timeout
Edit `shorts_generator/config.py` (or set env vars):
- `MUAPI_POLL_INTERVAL` (default 5s) — seconds between job-status polls
- `MUAPI_POLL_TIMEOUT` (default 1800s) — give up after this long

### Whisper transcription
Audio is transcribed by MuAPI's `/openai-whisper` endpoint (server-side `whisper-1`). Pass `--language <code>` to lock the recognition to a specific language; otherwise it auto-detects.

## Engagement research

`research/` has tools for tuning the defaults against real top-performing clips in your niche:
- `benchmark.py` reads Shorts metadata and auto-captions (no downloads).
- `instagram_reels_template.csv` is a template for logging Reels by hand.

See [research/README.md](research/README.md).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── main.py                       CLI entry point
├── requirements.txt              core deps (api mode)
├── requirements-local.txt        optional deps for --mode local
├── .env.example
└── shorts_generator/
    ├── config.py                 env / settings (MuAPI + local LLM + Whisper)
    ├── muapi.py                  generic submit + poll wrapper
    ├── downloader.py             API mode: YouTube download via MuAPI
    ├── transcriber.py            API mode: MuAPI /openai-whisper client
    ├── highlights.py             shared LLM virality ranking (pluggable backend)
    ├── clipper.py                API mode: MuAPI /autocrop
    ├── pipeline.py               mode dispatcher (api ↔ local)
    └── local/                    --mode local backends (offline)
        ├── downloader.py         yt-dlp download
        ├── transcriber.py        faster-whisper transcription
        ├── llm.py                OpenAI or Gemini client selector
        └── clipper.py            ffmpeg cut + OpenCV vertical crop
```

## Troubleshooting

### Whisper produced no segments
The video may have no detectable speech, or it may be in a language Whisper struggles with. Try passing `--language en` (or the correct ISO-639-1 code) to skip auto-detection.

### Looking for better results?
The [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) uses an improved algorithm that produces higher-quality clips with better highlight detection.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Related Projects

- [awesome-vibecoded-saas](https://github.com/Anil-matcha/awesome-vibecoded-saas) — broader catalog of open-source SaaS alternatives featuring this Shorts workflow.
- [Muapi open-source alternatives](https://muapi.ai/open-source/alternative) — compare the Shorts workflow with the paid creator tools it targets.
- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [Text to Video AI](https://github.com/SamurAIGPT/Text-To-Video-AI)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [No-code YouTube Shorts Generator](https://www.vadoo.tv/clip-youtube-video)
- [ai-creator-academy](https://github.com/Anil-matcha/ai-creator-academy) — free curriculum teaching creators how to monetize AI-generated shorts and video content
