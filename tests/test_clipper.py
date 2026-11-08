import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from shorts_generator.local import clipper


def _write_video(path, frames, fps=25):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def _dark_shot(x, n):
    """A dim figure on a black set: same colours wherever it stands, like a podcast camera."""
    frames = []
    for i in range(n):
        f = np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(f, (x + i % 3, 80), (x + 160 + i % 3, 330), (70, 80, 95), -1)
        frames.append(f)
    return frames


def test_cut_between_dark_shots_is_found(tmp_path):
    # Hue/saturation histograms of a black set barely change across a cut, so a cut
    # between two dark camera angles must still be caught.
    path = tmp_path / "two_shots.mp4"
    _write_video(path, _dark_shot(40, 20) + _dark_shot(420, 20))
    shot_starts, _, n = clipper._analyze_shots(cv2, str(path), 640, 360)
    assert n == 40
    assert shot_starts == [0, 20]


def test_small_movement_is_not_a_cut(tmp_path):
    path = tmp_path / "one_shot.mp4"
    _write_video(path, _dark_shot(40, 40))
    shot_starts, _, _ = clipper._analyze_shots(cv2, str(path), 640, 360)
    assert shot_starts == [0]


def test_encoder_uses_vaapi_only_when_the_probe_passes(monkeypatch):
    from shorts_generator.local import clipper as cl
    calls = []
    monkeypatch.setattr(cl.config, "VIDEO_ENCODER", "auto")
    monkeypatch.setattr(cl, "_vaapi_ok", None)
    monkeypatch.setattr(cl.subprocess, "run", lambda *a, **k: calls.append(a) or type("P", (), {"returncode": 0})())
    pre, video = cl._encode_args(20, ["ass='x.ass'"])
    assert pre == ["-vaapi_device", cl.config.VAAPI_DEVICE]
    assert video[video.index("-vf") + 1] == "ass='x.ass',format=nv12,hwupload"
    assert video[video.index("-c:v") + 1] == "h264_vaapi"
    cl._encode_args(18, [])
    assert len(calls) == 1  # probed once

    monkeypatch.setattr(cl, "_vaapi_ok", False)
    pre, video = cl._encode_args(18, [])
    assert pre == [] and "-vf" not in video and video[video.index("-c:v") + 1] == "libx264"

    monkeypatch.setattr(cl.config, "VIDEO_ENCODER", "x264")
    monkeypatch.setattr(cl, "_vaapi_ok", True)
    assert cl._encode_args(20, [])[1][1] == "libx264"
