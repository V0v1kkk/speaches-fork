"""GigaAM Duo adapter using the existing Speaches VAD and onnx-asr decoder."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
from onnx_asr.adapters import TextResultsAsrAdapter
from openai.types.audio import Transcription

from speaches.api_types import Model
from speaches.executors.duo_backend import load_duo_model
from speaches.executors.duo_model_id import MODEL_ID
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import NonStreamingTranscriptionResponse, TranscriptionRequest
from speaches.hf_utils import HfModelFilter
from speaches.model_registry import ModelRegistry


class DuoModelRegistry(ModelRegistry[Model, Path]):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(HfModelFilter(model_name=MODEL_ID, task="automatic-speech-recognition"))
        self.model_dir = model_dir

    def list_remote_models(self) -> Generator[Model]:
        return
        yield

    def list_local_models(self) -> Generator[Model]:
        if self.model_dir.is_dir():
            yield Model(id=MODEL_ID, owned_by="local", task="automatic-speech-recognition")

    def get_model_files(self, model_id: str) -> Path:
        if model_id != MODEL_ID or not self.model_dir.is_dir():
            raise FileNotFoundError(model_id)
        return self.model_dir

    def download_model_files(self, model_id: str) -> None:
        raise ValueError(f"{model_id} is installed from a local prepared model pack")


class DuoModelManager(BaseModelManager[TextResultsAsrAdapter]):
    def __init__(self, ttl: int, model_dir: Path, device: str, threads: int) -> None:
        super().__init__(ttl)
        self.model_dir = model_dir
        self.device = device
        self.threads = threads

    def _load_fn(self, model_id: str) -> TextResultsAsrAdapter:
        if model_id != MODEL_ID:
            raise ValueError(f"Unsupported model {model_id}")
        return load_duo_model(self.model_dir, device=self.device, threads=self.threads)

    def handle_transcription_request(
        self, request: TranscriptionRequest, **_kwargs
    ) -> NonStreamingTranscriptionResponse:
        if request.stream or request.response_format not in ("text", "json"):
            raise ValueError("Duo currently supports non-streaming text and json responses")
        if request.audio.sample_rate != 16000:
            raise ValueError("Expected 16 kHz decoded audio")
        texts = []
        if request.speech_segments:
            with self.load_model(request.model) as model:
                for segment in request.speech_segments:
                    if not (0 <= segment.start < segment.end <= len(request.audio.data)):
                        raise ValueError("Invalid VAD segment")
                    # Router VAD uses 30-second bounds. Reject unexpected giant
                    # segments rather than silently sending an unbounded encoder input.
                    if segment.end - segment.start > 31 * 16000:
                        raise ValueError("VAD segment exceeds 31 seconds")
                    audio = request.audio.data[segment.start : segment.end]
                    if len(audio) < 320:
                        audio = np.pad(audio, (0, 320 - len(audio)))
                    text = model.recognize(audio)
                    if text:
                        texts.append(text.strip())
        text = " ".join(texts)
        if request.response_format == "text":
            return text, "text/plain"
        return Transcription(text=text)
