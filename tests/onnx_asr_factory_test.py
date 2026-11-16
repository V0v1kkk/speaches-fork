from pathlib import Path

import numpy as np
from onnx_asr.models.gigaam import GigaamV2Rnnt
import pytest


def model_files(tmp_path: Path) -> dict[str, Path]:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a 0\n<blk> 1\n")
    return {"vocab": vocab, **{name: tmp_path / f"{name}.onnx" for name in ("encoder", "decoder", "joint")}}


def test_gigaam_uses_explicit_session_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from onnx_asr.models import gigaam

    monkeypatch.setattr(gigaam.rt, "InferenceSession", lambda *_a, **_kw: pytest.fail("ORT session should not be used"))
    calls = []

    def factory(path: Path, **options: object) -> object:
        calls.append((path.name, options))
        return object()

    model = GigaamV2Rnnt(model_files(tmp_path), lambda _name: lambda *_args: None, {}, session_factory=factory)
    assert [name for name, _ in calls] == ["encoder.onnx", "decoder.onnx", "joint.onnx"]
    assert all(options == {} for _, options in calls)
    assert model._blank_idx == 1  # noqa: SLF001
    assert model._create_state()[0].dtype == np.float32  # noqa: SLF001


def test_gigaam_without_factory_keeps_ort_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from onnx_asr.models import gigaam

    calls = []
    monkeypatch.setattr(gigaam.rt, "InferenceSession", lambda path, **_kw: calls.append(path.name) or object())
    GigaamV2Rnnt(model_files(tmp_path), lambda _name: lambda *_args: None, {})
    assert calls == ["encoder.onnx", "decoder.onnx", "joint.onnx"]
