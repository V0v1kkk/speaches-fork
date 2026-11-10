"""Local GigaAM Duo 1.3 model pack support.

The pack contains the original graphs, converted indexed vocabulary, config and
exact frontend coefficients. Its preparation and provenance are documented by
its producer. No weights are downloaded by this loader.
"""
from pathlib import Path
import hashlib
import re

import numpy as np
import onnxruntime as rt

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.gigaam import GigaamV2Rnnt
from onnx_asr.preprocessors.numpy_preprocessor import GigaamPreprocessorNumpy
from onnx_asr.preprocessors.resampler import Resampler
from scipy.fft import rfft


class DuoPreprocessor(GigaamPreprocessorNumpy):
    """GigaAM v3 NumPy frontend with Duo's full-precision coefficients and FFT."""

    def __init__(self, window, filters):
        super().__init__("gigaam_v3")
        self._window = window
        self._melscale_fbanks = filters

    def __call__(self, waveforms, waveforms_lens):
        frames = np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[:, ::self._hop_length]
        spectrum = rfft(frames * self._window, self._n_fft, axis=-1, workers=1)
        energy = (spectrum.real ** 2 + spectrum.imag ** 2) @ self._melscale_fbanks
        np.maximum(energy, np.float32(1e-9), out=energy)
        np.log(energy, out=energy)
        if not np.isfinite(energy).all():
            raise ValueError("Non-finite Duo frontend output")
        return energy.transpose(0, 2, 1), (waveforms_lens - self._win_length) // self._hop_length + 1


class DuoRnnt(GigaamV2Rnnt):
    def _encode(self, features, features_lens):
        output, lengths = super()._encode(features, features_lens)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite Duo encoder output")
        return output, lengths

    def _decode(self, tokens, state, frame):
        logits, step, new_state = super()._decode(tokens, state, frame)
        if not np.isfinite(logits).all() or any(not np.isfinite(value).all() for value in new_state):
            raise ValueError("Non-finite Duo predictor or joint output")
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


def load_duo_model(path: str | Path, *, device: str = "cpu", threads: int = 2) -> TextResultsAsrAdapter:
    """Load a prepared pack on CPU or a single, explicit OpenVINO GPU device.

    Frontend and resampling run on CPU. The GPU option fails closed when a
    graph cannot be assigned to the selected device; it never selects AUTO.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    directory = Path(path)
    names = {"encoder": "encoder.onnx", "decoder": "decoder.onnx", "joint": "joint.onnx",
             "vocab": "vocab.indexed.txt", "config": "config.json"}
    model_files = {key: directory / value for key, value in names.items()}
    for name, expected in _PACK_HASHES.items():
        file = directory / name
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Invalid Duo pack file: {name}")
    with np.load(directory / "frontend.npz", allow_pickle=False) as constants:
        window = constants["window"]
        filters = constants["filters"]
    if (window.shape != (320,) or filters.shape != (161, 64) or
        window.dtype != np.float32 or filters.dtype != np.float32 or
        not np.isfinite(window).all() or not np.isfinite(filters).all()):
        raise ValueError("Invalid Duo frontend constants")

    options = rt.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    if device == "cpu":
        graph_options = {"sess_options": options, "providers": ["CPUExecutionProvider"]}
    elif re.fullmatch(r"GPU\.[0-9]+", device):
        import onnxruntime_ep_openvino as ov_ep

        # Register once per process. A missing or ambiguous device fails closed.
        if not any(d.ep_name == ov_ep.get_ep_name() for d in rt.get_ep_devices()):
            rt.register_execution_provider_library("openvino_ep", ov_ep.get_library_path())
        devices = [d for d in rt.get_ep_devices()
                   if d.ep_name == ov_ep.get_ep_name() and d.ep_metadata.get("ov_device") == device]
        if len(devices) != 1:
            raise RuntimeError(f"Expected exactly one OpenVINO device {device!r}, found {len(devices)}")
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(devices, {"load_config": '{"GPU":{"INFERENCE_PRECISION_HINT":"f32"}}'})
        graph_options = {"sess_options": options}
    else:
        raise ValueError("device must be cpu or an explicit OpenVINO GPU name")

    def frontend(name: str) -> GigaamPreprocessorNumpy:
        if name != "gigaam_v3":
            raise ValueError(f"Unexpected Duo frontend: {name}")
        return DuoPreprocessor(window, filters)

    model = DuoRnnt(model_files, frontend, graph_options)
    if model._blank_idx != 1024 or model._vocab_size != 1025 or model._max_tokens_per_step != 10:
        raise ValueError("Invalid Duo vocabulary or model configuration")
    if device != "cpu":
        for session in (model._encoder, model._decoder, model._joiner):
            session.disable_fallback()
            if "OpenVINOExecutionProvider" not in session.get_providers():
                raise RuntimeError("Duo graph did not attach to OpenVINO")
    return TextResultsAsrAdapter(model, Resampler(16_000, {"providers": ["CPUExecutionProvider"]}))
