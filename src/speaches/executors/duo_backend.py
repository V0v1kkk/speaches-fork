"""Local GigaAM Duo 1.3 model pack support.

The pack contains the original graphs, converted indexed vocabulary, config and
exact frontend coefficients. Its preparation and provenance are documented by
its producer. No weights are downloaded by this loader.
"""

import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.gigaam import GigaamV2Rnnt
from onnx_asr.preprocessors.numpy_preprocessor import GigaamPreprocessorNumpy
from onnx_asr.preprocessors.resampler import Resampler
import onnxruntime as rt
from scipy.fft import rfft

from speaches.executors.shared.openvino_session import OpenVinoSessionFactory


class DuoPreprocessor(GigaamPreprocessorNumpy):
    """GigaAM v3 NumPy frontend with Duo's full-precision coefficients and FFT."""

    def __init__(self, window: NDArray[np.float32], filters: NDArray[np.float32]) -> None:
        super().__init__("gigaam_v3")
        self._window = window
        self._melscale_fbanks = filters

    def __call__(
        self, waveforms: NDArray[np.float32], waveforms_lens: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        if not np.isfinite(waveforms).all():
            raise ValueError("Non-finite Duo waveform input")
        frames = np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[:, :: self._hop_length]
        spectrum = rfft(frames * self._window, self._n_fft, axis=-1, workers=1)
        energy = (spectrum.real**2 + spectrum.imag**2) @ self._melscale_fbanks
        np.maximum(energy, np.float32(1e-9), out=energy)
        np.log(energy, out=energy)
        if not np.isfinite(energy).all():
            raise ValueError("Non-finite Duo frontend output")
        return energy.transpose(0, 2, 1), (waveforms_lens - self._win_length) // self._hop_length + 1


class CheckedPredictorSession:
    """Validate predictor tensors before they can be fed into the joint graph."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def run(self, output_names: list[str], feed: dict[str, object]) -> list[NDArray[np.float32]]:
        tokens = np.asarray(feed["x"])
        if tokens.shape != (1, 1) or tokens.dtype != np.int64:
            raise ValueError("Invalid Duo predictor token input")
        for name in ("h.1", "c.1"):
            value = np.asarray(feed[name])
            if value.shape != (1, 1, 320) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid Duo predictor state input")
        outputs = self.session.run(output_names, feed)
        if len(outputs) != 3 or any(
            not isinstance(value, np.ndarray)
            or value.shape != (1, 1, 320)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            for value in outputs
        ):
            raise ValueError("Invalid Duo predictor output")
        return outputs


class DuoRnnt(GigaamV2Rnnt):
    def _encode(
        self, features: NDArray[np.float32], features_lens: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        if not np.isfinite(features).all():
            raise ValueError("Non-finite Duo encoder input")
        output, lengths = super()._encode(features, features_lens)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite Duo encoder output")
        return output, lengths

    def _decode(
        self, tokens: list[int], state: list[NDArray[np.float32]], frame: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], int, list[NDArray[np.float32]]]:
        if not np.isfinite(frame).all() or any(not np.isfinite(value).all() for value in state):
            raise ValueError("Non-finite Duo predictor or joint input")
        logits, step, new_state = super()._decode(tokens, state, frame)
        # The predictor may cache its output in state; all predictor data must
        # be finite. Only individual -Inf values in final LogSoftmax are valid.
        if any(not np.isfinite(value).all() for value in (*state, *new_state)):
            raise ValueError("Non-finite Duo predictor output")
        if logits.shape != (1025,) or logits.dtype != np.float32:
            raise ValueError("Invalid Duo joint output shape or dtype")
        if np.isnan(logits).any() or np.isposinf(logits).any() or np.isneginf(logits).all():
            raise ValueError("Invalid Duo joint LogSoftmax output")
        return logits, step, new_state


_PACK_HASHES = {
    "encoder.onnx": "7615585129715b406d63c1f7317ce932c15f1b5d83f49bcac481d2acdb6e2912",
    "encoder.weights.bin": "b2bf3c23288810f7ef930488efeb7b49b137e4bb573a2ad601f06ca9fb712780",
    "decoder.onnx": "72cb006e66552ba022dc584f51d3cf848f741b68c48e203b05fd2ff5d83e42cd",
    "joint.onnx": "a9d82276bb4a1005382c7d6cb143fcdba40b0a354c799b05f9f5b672fc12b3b9",
    "vocab.indexed.txt": "39abae20e692998290c574e606f11a9edef2902a1995463fcff63d1490cf22b7",
    "config.json": "d8884186e97baa482d83b7aa9133fc032f70a44a4a4fdf854d1b153595cd0d3d",
    "frontend.npz": "c340c8daf9690110cb1abb0dfb0571fa8de47b4b16069a1885a71b1a7415c77d",
}


def load_duo_model(
    path: str | Path, *, device: str = "cpu", threads: int = 2, profiling: bool = False
) -> TextResultsAsrAdapter:
    """Load a checked pack on CPU, CUDA:N or one explicit OpenVINO GPU.

    Frontend and resampling run on CPU. CUDA requires an initialized provider
    for each graph; shape operations may stay on CPU. OpenVINO uses one
    explicit GPU device. Neither path selects AUTO.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    directory = Path(path)
    names = {
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "joint": "joint.onnx",
        "vocab": "vocab.indexed.txt",
        "config": "config.json",
    }
    model_files = {key: directory / value for key, value in names.items()}
    for name, expected in _PACK_HASHES.items():
        file = directory / name
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Invalid Duo pack file: {name}")
    with np.load(directory / "frontend.npz", allow_pickle=False) as constants:
        window = constants["window"]
        filters = constants["filters"]
    if (
        window.shape != (320,)
        or filters.shape != (161, 64)
        or window.dtype != np.float32
        or filters.dtype != np.float32
        or not np.isfinite(window).all()
        or not np.isfinite(filters).all()
    ):
        raise ValueError("Invalid Duo frontend constants")

    if device == "cpu":
        options = rt.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        graph_options = {"sess_options": options, "providers": ["CPUExecutionProvider"]}
        session_factory = None
    elif re.fullmatch(r"cuda:[0-9]+", device):
        if "CUDAExecutionProvider" not in rt.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is required for Duo CUDA")
        options = rt.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = profiling
        graph_options = {
            "sess_options": options,
            "providers": [
                ("CUDAExecutionProvider", {
                    "device_id": int(device.split(":")[1]),
                    "use_tf32": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                }),
                "CPUExecutionProvider",
            ],
        }
        session_factory = None
    elif re.fullmatch(r"GPU\.[0-9]+", device):
        graph_options = {}
        session_factory = OpenVinoSessionFactory(device, profiling=profiling)
    else:
        raise ValueError("device must be cpu, cuda:N or an explicit OpenVINO GPU name")

    def frontend(name: str) -> GigaamPreprocessorNumpy:
        if name != "gigaam_v3":
            raise ValueError(f"Unexpected Duo frontend: {name}")
        return DuoPreprocessor(window, filters)

    if session_factory is None:
        model = DuoRnnt(model_files, frontend, graph_options)
    else:
        model = DuoRnnt(model_files, frontend, graph_options, session_factory=session_factory)
    if device.startswith("cuda:"):
        for session in (model._encoder, model._decoder, model._joiner):  # noqa: SLF001
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("Duo graph failed to initialize CUDAExecutionProvider")
    if model._blank_idx != 1024 or model._vocab_size != 1025 or model._max_tokens_per_step != 10:  # noqa: SLF001
        raise ValueError("Invalid Duo vocabulary or model configuration")
    # The original RNNT implementation caches predictor outputs then sends
    # them to joint; validate inside run so invalid tensors never reach joint.
    model._decoder = CheckedPredictorSession(model._decoder)  # noqa: SLF001
    return TextResultsAsrAdapter(model, Resampler(16_000, {"providers": ["CPUExecutionProvider"]}))
