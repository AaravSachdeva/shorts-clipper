"""Loudness per second of an episode's audio, and the moments that stand out.

Decoded with the ffmpeg binary the renderer already needs; no audio library.
"""
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

RATE = 16000
MERGE_GAP_SECONDS = 30


def rms_db(samples: np.ndarray, rate: int) -> List[float]:
    """dBFS of each whole second of int16 mono samples."""
    n = len(samples) // rate
    x = samples[: n * rate].astype(np.float64).reshape(n, rate) / 32768.0
    return [float(v) for v in 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-9)]


def loudness_per_second(source_path: str, cache_path: Path) -> List[float]:
    """dBFS per second for the whole file, cached as JSON next to the episode's other work files."""
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(source_path):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", source_path, "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    db: List[float] = []
    block = RATE * 2  # one second of 16-bit samples
    while True:
        chunk = proc.stdout.read(block)
        if len(chunk) < block:
            break
        db += rms_db(np.frombuffer(chunk, dtype=np.int16), RATE)
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg could not decode {source_path}: {err[-300:]}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([round(d, 2) for d in db]), encoding="utf-8")
    return [round(d, 2) for d in db]


def loud_peaks(db: List[float], above: float = 6.0) -> List[Tuple[int, float]]:
    """(second, dB) of the loudest second in each run more than `above` dB over the episode median.

    Runs closer than 30 s count as one moment.
    """
    if not db:
        return []
    threshold = statistics.median(db) + above
    runs: List[List[int]] = []
    for t, d in enumerate(db):
        if d > threshold:
            if runs and t - runs[-1][-1] < MERGE_GAP_SECONDS:
                runs[-1].append(t)
            else:
                runs.append([t])
    peaks = []
    for run in runs:
        t = max(run, key=lambda s: db[s])
        peaks.append((t, db[t]))
    return peaks
