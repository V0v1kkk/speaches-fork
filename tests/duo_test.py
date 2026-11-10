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


def test_joint_accepts_individual_negative_infinity(monkeypatch: pytest.MonkeyPatch) -> None:
    from onnx_asr.models.gigaam import GigaamV2Rnnt

    from speaches.executors.duo_backend import DuoRnnt

    logits = np.zeros(1025, dtype=np.float32)
    logits[1] = -np.inf
    monkeypatch.setattr(GigaamV2Rnnt, "_decode", lambda _self, _tokens, state, _frame: (logits, -1, state))
    model = object.__new__(DuoRnnt)
    actual, _, _ = model._decode([], [np.zeros(1, dtype=np.float32)], np.zeros(1, dtype=np.float32))  # noqa: SLF001
    assert np.array_equal(actual, logits)


@pytest.mark.parametrize("bad", [[np.nan, 0.0], [np.inf, 0.0], [-np.inf, -np.inf]])
def test_joint_rejects_invalid_logsoftmax(monkeypatch: pytest.MonkeyPatch, bad: list[float]) -> None:
    from onnx_asr.models.gigaam import GigaamV2Rnnt

    from speaches.executors.duo_backend import DuoRnnt

    logits = np.full(1025, bad[1], dtype=np.float32)
    logits[0] = bad[0]
    monkeypatch.setattr(GigaamV2Rnnt, "_decode", lambda _self, _tokens, state, _frame: (logits, -1, state))
    model = object.__new__(DuoRnnt)
    with pytest.raises(ValueError, match="joint LogSoftmax"):
        model._decode([], [np.zeros(1, dtype=np.float32)], np.zeros(1, dtype=np.float32))  # noqa: SLF001


def test_predictor_and_encoder_nonfinite_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    from onnx_asr.models.gigaam import GigaamV2Rnnt

    from speaches.executors.duo_backend import DuoRnnt

    model = object.__new__(DuoRnnt)
    finite = np.zeros(1, dtype=np.float32)
    bad = np.array([np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="predictor or joint input"):
        model._decode([], [bad], finite)  # noqa: SLF001
    monkeypatch.setattr(
        GigaamV2Rnnt, "_decode", lambda _self, _tokens, _state, _frame: (np.zeros(1025, dtype=np.float32), -1, [bad])
    )
    with pytest.raises(ValueError, match="predictor output"):
        model._decode([], [finite], finite)  # noqa: SLF001
    with pytest.raises(ValueError, match="encoder input"):
        model._encode(bad, np.ones(1, dtype=np.int64))  # noqa: SLF001
    monkeypatch.setattr(GigaamV2Rnnt, "_encode", lambda _self, _features, lengths: (bad, lengths))
    with pytest.raises(ValueError, match="encoder output"):
        model._encode(finite, np.ones(1, dtype=np.int64))  # noqa: SLF001


def test_predictor_validation_precedes_joint_and_blocks_bad_input() -> None:
    from speaches.executors.duo_backend import CheckedPredictorSession

    class FakeSession:
        calls = 0
        bad_output = False

        def run(self, _names: list[str], _feed: dict[str, object]) -> list[np.typing.NDArray[np.float32]]:
            self.calls += 1
            result = [np.zeros((1, 1, 320), dtype=np.float32) for _ in range(3)]
            if self.bad_output:
                result[0][0, 0, 0] = -np.inf
            return result

    fake = FakeSession()
    session = CheckedPredictorSession(fake)  # type: ignore[arg-type]
    state = np.zeros((1, 1, 320), dtype=np.float32)
    feed = {"x": [[1024]], "h.1": state, "c.1": state.copy()}
    assert len(session.run(["dec", "h", "c"], feed)) == 3
    bad_state = state.copy()
    bad_state[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="state input"):
        session.run(["dec", "h", "c"], {**feed, "h.1": bad_state})
    assert fake.calls == 1
    fake.bad_output = True
    with pytest.raises(ValueError, match="predictor output"):
        session.run(["dec", "h", "c"], feed)
