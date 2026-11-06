import os

from dotenv import load_dotenv

load_dotenv()

MUAPI_API_KEY = os.getenv("MUAPI_API_KEY", "").strip()
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")

POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# Local-mode (--mode local) settings — only consulted when running offline.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "large-v3-turbo")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto / cpu / cuda
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")

# Ollama (LLM_PROVIDER=ollama) — fully local highlight ranking.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = "http://" + OLLAMA_HOST  # `ollama serve` accepts bare host:port
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
# Ollama's built-in context window is small (a few thousand tokens) and it
# silently truncates longer prompts, so we always send num_ctx explicitly.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
# Layers on the GPU. Unset = Ollama decides; its auto-fit keeps a big VRAM margin on small cards.
OLLAMA_NUM_GPU = int(os.environ["OLLAMA_NUM_GPU"]) if os.getenv("OLLAMA_NUM_GPU", "").strip() else None
# CPU inference on a 14B model can take many minutes per chunk.
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT", "3600"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip()
# Reasoning models (gemma4, qwen3) think before answering. On a CPU that is ~90% of the time,
# and gemma4:12b once thought for over an hour on one window without answering. Off by default.
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").strip().lower() in ("1", "true", "yes")

# Highlight selection.
# Transcript window per highlight call. Small local models do better on ~5 min windows.
HIGHLIGHT_CHUNK_SECONDS = int(
    os.getenv("HIGHLIGHT_CHUNK_SECONDS", "300" if LLM_PROVIDER == "ollama" else "1200")
)
# short ≈ 15-35s, medium ≈ 20-60s (default), long ≈ 45-90s
TARGET_LENGTH = os.getenv("TARGET_LENGTH", "medium").strip().lower()
HIGHLIGHT_RERANK = os.getenv("HIGHLIGHT_RERANK", "true").strip().lower() == "true"
# Optional JSONL of top-performing clip openings (see research/benchmark.py).
HIGHLIGHT_EXAMPLES_PATH = os.getenv("HIGHLIGHT_EXAMPLES", "").strip()

# Rendering.
CAPTIONS_ENABLED = os.getenv("CAPTIONS", "true").strip().lower() == "true"
HOOK_OVERLAY_ENABLED = os.getenv("HOOK_OVERLAY", "true").strip().lower() == "true"
# Percent of frame height to cut from the bottom before reframing (burned-in tickers).
CROP_BOTTOM_PCT = float(os.getenv("CROP_BOTTOM_PCT", "0"))
OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "1080"))
OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "1920"))
TARGET_LUFS = float(os.getenv("TARGET_LUFS", "-14"))

# VAD (Voice Activity Detection) settings for faster-whisper
# Default threshold is 0.5; lower = more sensitive, higher = less sensitive
# Default min_speech_duration_ms is 250ms; increase to avoid tiny false positives
# Default min_silence_duration_ms is 2000ms; increase to avoid splitting mid-sentence
# DISABLED by default because VAD is too aggressive on mixed speech/music content
LOCAL_WHISPER_VAD_FILTER = os.getenv("LOCAL_WHISPER_VAD_FILTER", "false").strip().lower() == "true"
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
if _vad_params_env:
    import json
    LOCAL_WHISPER_VAD_PARAMETERS = json.loads(_vad_params_env)
else:
    # Match faster-whisper defaults when VAD is enabled
    LOCAL_WHISPER_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Local mode needs an OpenAI key for highlight ranking. "
            "Add it to your .env or export it, or switch back to --mode api."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Local mode needs a Gemini key when LLM_PROVIDER=gemini. "
            "Add it to your .env or export it, or switch LLM_PROVIDER back to openai."
        )
    return GEMINI_API_KEY
