import sys
import types

import pytest

from shorts_generator.local import transcriber as tr


def _fake_ct2(monkeypatch, devices=1, types_=("float32", "int8", "int8_float32")):
    mod = types.SimpleNamespace(
        get_cuda_device_count=lambda: devices,
        get_supported_compute_types=lambda device: set(types_),
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", mod)


def test_auto_picks_cuda_when_ctranslate2_sees_a_gpu(monkeypatch):
    monkeypatch.setattr(tr, "LOCAL_WHISPER_DEVICE", "auto")
    _fake_ct2(monkeypatch, devices=1)
    assert tr._resolve_device() == "cuda"
    _fake_ct2(monkeypatch, devices=0)
    assert tr._resolve_device() == "cpu"


@pytest.mark.parametrize("supported,expected", [
    ({"float32", "int8", "int8_float32"}, "int8"),                       # Pascal, e.g. MX150: no fast fp16
    ({"float32", "float16", "int8", "int8_float16", "bfloat16"}, "int8_float16"),  # Turing and newer
    ({"float32"}, "float32"),
])
def test_cuda_compute_type_fits_the_card(monkeypatch, supported, expected):
    _fake_ct2(monkeypatch, types_=supported)
    assert tr._compute_type("cuda") == expected
    assert tr._compute_type("cpu") == "int8"


def test_gpu_failure_falls_back_to_cpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(tr, "LOCAL_WHISPER_DEVICE", "cuda")
    _fake_ct2(monkeypatch)
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"")
    made = []

    class Model:
        def __init__(self, name, device, compute_type):
            made.append((device, compute_type))
            self.device = device

        def transcribe(self, **kw):
            def segs():
                if self.device == "cuda":
                    raise RuntimeError("CUDA failed with error out of memory")
                yield types.SimpleNamespace(start=0.0, end=1.0, text="hi", words=[])
            return segs(), types.SimpleNamespace(duration=1.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    out = tr.transcribe_local(str(media))
    assert made == [("cuda", "int8"), ("cpu", "int8")]
    assert out["segments"][0]["text"] == "hi"
    assert "falling back to CPU" in capsys.readouterr().out


def test_preloads_pip_cuda_libs_in_dependency_order(tmp_path, monkeypatch):
    import ctypes

    for rel in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    monkeypatch.setitem(sys.modules, "nvidia", types.SimpleNamespace(__path__=[str(tmp_path)]))
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(path.rsplit("/", 1)[1]))
    tr._preload_pip_cuda_libs()
    assert loaded == ["libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"]


def test_preload_is_a_no_op_without_the_wheels(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvidia", None)  # import raises ImportError
    tr._preload_pip_cuda_libs()
