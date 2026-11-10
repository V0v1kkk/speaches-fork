from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from speaches.audio import Audio
from speaches.executors.duo import MODEL_ID, DuoModelManager
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import SpeechTimestamp, VadOptions


def request(
    samples: int, segments: list[SpeechTimestamp], response_format: Literal["json", "text"] = "json"
) -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=Audio(np.zeros(samples, dtype=np.float32), sample_rate=16000),
        model=MODEL_ID,
        response_format=response_format,
        speech_segments=segments,
        vad_options=VadOptions(),
        timestamp_granularities=["segment"],
    )


class FakeModel:
    def __init__(self) -> None:
        self.lengths = []

    def recognize(self, audio: np.typing.NDArray[np.float32]) -> str:
        self.lengths.append(len(audio))
        return " example "


def test_uses_vad_boundaries_and_resets_each_segment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = FakeModel()
    manager = DuoModelManager(-1, tmp_path, "cpu", 1)
    monkeypatch.setattr(manager, "load_model", lambda _: nullcontext(model))
    result = manager.handle_transcription_request(
        request(1000, [SpeechTimestamp(start=0, end=100), SpeechTimestamp(start=400, end=900)])
    )
    assert result.text == "example example"
    assert model.lengths == [320, 500]


def test_silence_returns_empty_without_recognition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = DuoModelManager(-1, tmp_path, "cpu", 1)
    monkeypatch.setattr(manager, "load_model", lambda _: pytest.fail("silence must not invoke recognition"))
    assert manager.handle_transcription_request(request(16000, [])).text == ""


def test_invalid_segments_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = DuoModelManager(-1, tmp_path, "cpu", 1)
    monkeypatch.setattr(manager, "load_model", lambda _: nullcontext(FakeModel()))
    with pytest.raises(ValueError, match="Invalid VAD segment"):
        manager.handle_transcription_request(request(100, [SpeechTimestamp(start=0, end=101)]))
    n = 31 * 16000 + 1
    with pytest.raises(ValueError, match="exceeds"):
        manager.handle_transcription_request(request(n, [SpeechTimestamp(start=0, end=n)]))
