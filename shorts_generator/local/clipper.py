"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


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
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# Reframing tunables.
_DETECT_EVERY_N_FRAMES = 3      # run the face detector on every Nth frame
_DETECT_WIDTH = 640             # downscale frames to this width for detection
_MIN_FACE_FRAC = 0.08           # ignore "faces" smaller than 8% of frame height
_SHOT_CUT_THRESHOLD = 0.5       # histogram correlation below this = camera cut
_MIN_SHOT_FRAMES = 6            # merge shorter "shots" (flashes, transitions)


def _is_cut(cv2, prev_hist, hist) -> bool:
    return prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < _SHOT_CUT_THRESHOLD


def _analyze_shots(cv2, in_path: str, src_w: int, src_h: int):
    """Pass 1: find camera cuts and face centres.

    Returns (shot_starts, face_samples) where shot_starts is a sorted list of
    frame indices that begin a shot and face_samples is [(frame_idx, cx)].
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scale = min(1.0, _DETECT_WIDTH / float(src_w))
    min_face = max(24, int(src_h * scale * _MIN_FACE_FRAC))

    cap = cv2.VideoCapture(in_path)
    shot_starts = [0]
    samples = []
    prev_hist = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if _is_cut(cv2, prev_hist, hist) and idx - shot_starts[-1] >= _MIN_SHOT_FRAMES:
            shot_starts.append(idx)
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
    """One horizontal centre per shot: median of the dominant face's centres.

    Within a shot, faces are clustered by x position; the cluster with the
    most total face area wins (the person on camera), so occasional false
    positives elsewhere in the frame are outvoted instead of yanking the crop.
    """
    import statistics

    bounds = shot_starts + [n_frames]
    centres = []
    prev = src_w / 2.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        pts = [(cx, area) for (i, cx, area) in samples if a <= i < b]
        if not pts:
            centres.append(prev)
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
        prev = statistics.median(members)
        centres.append(prev)
    return centres


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio with a steady, per-shot frame.

    Podcast/interview cameras are almost always static, so instead of chasing
    a face detection every frame (which jitters badly on false positives), we:
      1. detect camera cuts and sample faces on a downscaled copy,
      2. pick one crop position per shot (median of the dominant face),
      3. hold that crop rock-steady for the whole shot, snapping on cuts.
    Frames are piped to ffmpeg as H.264 with the original audio.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build has no CascadeClassifier; install opencv-python<5.")

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

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
    print(f"[clip/local]   {len(shot_starts)} shot(s), {len(samples)} face samples", flush=True)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}",
            "-i", "-",
            "-i", in_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-shortest",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(in_path)
    shot = 0
    idx = 0
    y0 = max(0, (src_h - crop_h) // 2)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            while shot + 1 < len(shot_starts) and idx >= shot_starts[shot + 1]:
                shot += 1
            cx = int(centres[shot])
            x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
            ffmpeg.stdin.write(frame[y0:y0 + crop_h, x0:x0 + crop_w].tobytes())
            idx += 1
    finally:
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed with code {ffmpeg.returncode}")
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
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
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
