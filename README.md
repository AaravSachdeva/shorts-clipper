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
   git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
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
   LOCAL_WHISPER_DEVICE=auto         # auto / cpu / cuda
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

Local mode writes the rendered shorts to `./output/short_01.mp4`, `short_02.mp4`, … (override with `LOCAL_OUTPUT_DIR`). Each clip is 1080×1920 H.264. It comes with burned-in word-highlight captions, an on-screen hook for the first 3 s, and audio normalized to −14 LUFS. Next to each clip, `short_01.txt` holds a ready-to-paste title, caption and hashtags, and `short_01.json` holds the full metadata.

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
| `--profile` | — | `cpu-slow` or `cpu-quality`: Ollama + CPU Whisper presets (local mode) |
| `--llm-provider` | `LLM_PROVIDER` | `openai`, `gemini` or `ollama` (local mode) |
| `--target-length` | `medium` | `short` ≈15–35 s, `medium` ≈20–60 s, `long` ≈45–90 s |
| `--no-captions` | — | Don't burn in captions |
| `--no-hook` | — | Don't show the on-screen hook title for the first 3 s |
| `--crop-bottom` | `0` | Cut this % of frame height off the bottom first (burned-in tickers or chat bars) |
| `--no-rerank` | — | Skip the comparative re-rank of the top candidates |
| `--examples` | — | JSONL of top-performing clip openings (from `research/benchmark.py`) |

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
- **Chunk size**: `CHUNK_SIZE_SECONDS` (default 1200) — chunk length for long videos
- **Long-video threshold**: `LONG_VIDEO_THRESHOLD` (default 1800) — videos longer than this are chunked
- **Chunk overlap**: `CHUNK_OVERLAP_SECONDS` (default 60) — overlap between chunks so cross-boundary clips aren't missed

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
