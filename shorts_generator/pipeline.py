"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + an LLM + ffmpeg/opencv.
                              Self-hosted; LLM_PROVIDER selects openai, gemini,
                              or ollama (fully offline).
"""
import time
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm, release_local_llm
    from .local.transcriber import transcribe_local
    from .metadata import generate_metadata, write_sidecars

    t0 = time.time()
    timings: Dict[str, float] = {}
    source_path = download_youtube_local(youtube_url, fmt=download_format)
    timings["download"] = time.time() - t0

    t = time.time()
    transcript = transcribe_local(source_path, language=language)
    timings["transcribe"] = time.time() - t
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    t = time.time()
    try:
        highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
        all_highlights: List[Dict] = highlights_result.get("highlights", [])
        if not all_highlights:
            raise RuntimeError("Highlight generator returned zero clips.")
        top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
        top = generate_metadata(transcript, top, call_local_llm)
    finally:
        release_local_llm()  # free the local model's RAM before rendering
    timings["llm"] = time.time() - t

    print(f"[pipeline/local] rendering {len(top)} of {len(all_highlights)} candidates", flush=True)
    t = time.time()
    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio, transcript=transcript)
    write_sidecars(shorts)
    timings["render"] = time.time() - t
    timings["total"] = time.time() - t0
    print(
        "[pipeline/local] timings: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timings.items()),
        flush=True,
    )

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "content_info": highlights_result.get("content_info"),
        "highlights": all_highlights,
        "shorts": shorts,
        "timings": timings,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, download_format, language)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
