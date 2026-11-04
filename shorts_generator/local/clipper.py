"""Local clipping: ffmpeg subclip → steady per-shot 9:16 reframe → captions.

Per highlight:
  1. Cut the source to [start, end] with ffmpeg (input seeking, re-encoded).
  2. Analyse camera shots and faces on a downscaled copy; choose one crop
     centre per shot (podcast cameras are static, so a held crop beats
     chasing per-frame detections). Shots with no face (screen shares,
     B-roll) get a blur-fill layout instead of a blind centre crop.
  3. Compose every frame at the output size (default 1080×1920) and pipe it
     to ffmpeg, which burns in word-highlight captions + the hook overlay
     (ASS via libass) and normalises loudness (two-pass loudnorm).
"""
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

from .. import config
from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _output_size(aspect_ratio: str) -> (int, int):
    r = _ratio(aspect_ratio)
    if abs(r - config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT) < 1e-3:
        return config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    w = config.OUTPUT_WIDTH
    h = int(round(w / r))
    return w - w % 2, h - h % 2


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -t duration → re-encoded mp4 with audio.

    -ss goes before -i (input seeking) so ffmpeg jumps straight to the clip
    instead of decoding the whole source from 0. With re-encoding this is
    frame-accurate in modern ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# Reframing tunables.
_DETECT_EVERY_N_FRAMES = 3      # run the face detector on every Nth frame
_DETECT_WIDTH = 640             # downscale frames to this width for detection
_MIN_FACE_FRAC = 0.08           # ignore "faces" smaller than 8% of frame height
_SHOT_CUT_THRESHOLD = 0.5       # frame-to-frame histogram correlation below this = hard cut
_SHOT_DRIFT_THRESHOLD = 0.3     # correlation vs. the shot's first frame below this = gradual cut
_MIN_SHOT_FRAMES = 6            # merge shorter "shots" (flashes, transitions)
_MIN_FACE_SAMPLES = 3           # fewer face samples than this in a shot = no face


def _is_cut(cv2, prev_hist, ref_hist, hist) -> bool:
    """Hard cuts change the histogram between consecutive frames; animated
    transitions (wipes, zooms, dissolves) only show up against the shot's
    first frame."""
    if prev_hist is None:
        return False
    if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < _SHOT_CUT_THRESHOLD:
        return True
    return ref_hist is not None and cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL) < _SHOT_DRIFT_THRESHOLD


def _analyze_shots(cv2, in_path: str, src_w: int, src_h: int):
    """Pass 1: find camera cuts and face centres.

    `src_h` is the usable height (after --crop-bottom); rows below it are ignored.
    Returns (shot_starts, face_samples, n_frames) where face_samples is
    [(frame_idx, cx, area)].
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scale = min(1.0, _DETECT_WIDTH / float(src_w))
    min_face = max(24, int(src_h * scale * _MIN_FACE_FRAC))

    cap = cv2.VideoCapture(in_path)
    shot_starts = [0]
    samples = []
    prev_hist = None
    ref_hist = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:src_h]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if ref_hist is None:
            ref_hist = hist
        if _is_cut(cv2, prev_hist, ref_hist, hist) and idx - shot_starts[-1] >= _MIN_SHOT_FRAMES:
            shot_starts.append(idx)
            ref_hist = hist
        prev_hist = hist

        if idx % _DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face))
            for (x, y, w, h) in faces:
                samples.append((idx, (x + w / 2.0) / scale, w * h))
        idx += 1
    cap.release()
    return shot_starts, samples, idx


def _shot_centres(shot_starts, samples, n_frames: int, src_w: int):
    """One horizontal centre per shot (or None when the shot has no face).

    Within a shot, faces are clustered by x position; the cluster with the
    most total face area wins (the person on camera), so occasional false
    positives elsewhere in the frame are outvoted instead of yanking the crop.
    """
    import statistics

    bounds = shot_starts + [n_frames]
    centres = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        pts = [(cx, area) for (i, cx, area) in samples if a <= i < b]
        if len(pts) < _MIN_FACE_SAMPLES:
            centres.append(None)
            continue
        bucket_w = src_w / 8.0
        clusters = {}
        for cx, area in pts:
            clusters.setdefault(int(cx // bucket_w), []).append((cx, area))
        # merge each bucket with its neighbours so a face on a boundary isn't split
        best = max(
            clusters,
            key=lambda k: sum(a for kk in (k - 1, k, k + 1) for (_, a) in clusters.get(kk, [])),
        )
        members = [cx for kk in (best - 1, best, best + 1) for (cx, _) in clusters.get(kk, [])]
        centres.append(statistics.median(members))
    return centres


def _blur_fill(cv2, frame, out_w: int, out_h: int):
    """Full frame, scaled to the output width, over a blurred, darkened copy of itself."""
    import numpy as np

    h, w = frame.shape[:2]
    # background: centre-crop to the output ratio, blur at low res (fast), darken
    bg_w = min(w, int(h * out_w / out_h))
    x0 = (w - bg_w) // 2
    bg = cv2.resize(frame[:, x0:x0 + bg_w], (out_w // 10, out_h // 10), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    bg = (bg * 0.55).astype(np.uint8)
    fg_h = int(round(h * out_w / w))
    fg = cv2.resize(frame, (out_w, fg_h), interpolation=cv2.INTER_AREA)
    y = max(0, (out_h - fg_h) // 2 - int(out_h * 0.06))   # sit slightly above centre, clear of captions
    bg[y:y + fg_h] = fg[: max(0, min(fg_h, out_h - y))]
    return bg


def _measure_loudness(in_path: str, target: float) -> Optional[Dict]:
    """First loudnorm pass: measure, return the values for the second pass."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", in_path, "-vn",
        "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if p.returncode != 0 or start == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _loudnorm_filter(in_path: str) -> str:
    target = config.TARGET_LUFS
    m = _measure_loudness(in_path, target)
    if not m:
        return f"loudnorm=I={target}:TP=-1.5:LRA=11"
    return (
        f"loudnorm=I={target}:TP=-1.5:LRA=11:"
        f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
        f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
    )


def _escape_filter_path(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument (handles Windows drives)."""
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _render_short(in_path: str, out_path: str, aspect_ratio: str, ass_path: Optional[str] = None) -> str:
    """Reframe the cut clip, burn in captions, normalise audio, encode H.264."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build has no CascadeClassifier; install opencv-python<5.")

    out_w, out_h = _output_size(aspect_ratio)
    target_ratio = out_w / out_h
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # --crop-bottom: drop burned-in tickers / chat bars before reframing.
    src_h = int(full_h * (1 - max(0.0, min(40.0, config.CROP_BOTTOM_PCT)) / 100.0))
    src_h -= src_h % 2

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    shot_starts, samples, n_frames = _analyze_shots(cv2, in_path, src_w, src_h)
    centres = _shot_centres(shot_starts, samples, n_frames, src_w)
    n_blur = sum(c is None for c in centres)
    print(
        f"[clip/local]   {len(shot_starts)} shot(s), {len(samples)} face samples"
        + (f", {n_blur} without a face → blur-fill" if n_blur else ""),
        flush=True,
    )

    vf = []
    if ass_path:
        vf.append(f"ass='{_escape_filter_path(ass_path)}'")
    af = _loudnorm_filter(in_path)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps}",
        "-i", "-",
        "-i", in_path,
        "-map", "0:v:0", "-map", "1:a:0?",
    ]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += [
        "-af", af, "-ar", "48000",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-shortest",
        out_path,
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(in_path)
    shot = 0
    idx = 0
    y0 = max(0, (src_h - crop_h) // 2)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:src_h]
            while shot + 1 < len(shot_starts) and idx >= shot_starts[shot + 1]:
                shot += 1
            cx = centres[shot]
            if cx is None:
                out = _blur_fill(cv2, frame, out_w, out_h)
            else:
                x0 = max(0, min(src_w - crop_w, int(cx) - crop_w // 2))
                out = cv2.resize(
                    frame[y0:y0 + crop_h, x0:x0 + crop_w], (out_w, out_h), interpolation=cv2.INTER_CUBIC
                )
            ffmpeg.stdin.write(out.tobytes())
            idx += 1
    finally:
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed with code {ffmpeg.returncode}")
    return out_path


def _write_ass(h: Dict, transcript: Optional[Dict], start: float, end: float, aspect_ratio: str) -> Optional[str]:
    from ..boundaries import words_in_range
    from .captions import build_ass

    hook = (h.get("on_screen_hook") or h.get("title") or "").strip() if config.HOOK_OVERLAY_ENABLED else ""
    words = words_in_range(transcript, start, end) if (transcript and config.CAPTIONS_ENABLED) else []
    if not words and not hook:
        return None
    out_w, out_h = _output_size(aspect_ratio)
    doc = build_ass(
        words, end - start, width=out_w, height=out_h, hook_text=hook or None, captions=bool(words)
    )
    fd, path = tempfile.mkstemp(suffix=".ass", prefix="captions_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    highlight: Optional[Dict] = None,
    transcript: Optional[Dict] = None,
) -> str:
    """Cut + reframe (+ captions) one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    ass_path = None
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        ass_path = _write_ass(highlight or {}, transcript, start_time, end_time, aspect_ratio)
        _render_short(cut_path, out_path, aspect_ratio, ass_path=ass_path)
    finally:
        for p in (cut_path, ass_path):
            if p and os.path.exists(p):
                os.remove(p)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript: Optional[Dict] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                highlight=h,
                transcript=transcript,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
