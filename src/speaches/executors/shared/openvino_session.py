"""Optional OpenVINO sessions for callers that request named outputs."""

from collections import Counter
from pathlib import Path
import re
from threading import Lock
from typing import Any

import numpy as np


class OpenVinoSession:
    """One compiled graph and an infer request whose outputs are copied."""

    def __init__(self, core: Any, path: Path, device: str, *, profiling: bool = False) -> None:
        model = core.read_model(str(path))
        supported = core.query_model(model, device)
        missing = [op.get_friendly_name() for op in model.get_ops() if op.get_friendly_name() not in supported]
        if missing:
            raise RuntimeError(f"OpenVINO device does not support {len(missing)} graph operations")
        self.compiled = core.compile_model(
            model,
            device,
            {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f32",
                "PERF_COUNT": profiling,
            },
        )
        self.execution_devices = [str(value) for value in self.compiled.get_property("EXECUTION_DEVICES")]
        if self.execution_devices != [device]:
            raise RuntimeError(f"Expected execution on {device}; got {self.execution_devices}")
        self.request = self.compiled.create_infer_request()
        self._lock = Lock()

    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[np.ndarray]:
        # Reuse one request safely, including copying results before another call.
        inputs = {name: np.ascontiguousarray(value) for name, value in input_feed.items()}
        with self._lock:
            result = self.request.infer(inputs)
            return [np.array(result[self.compiled.output(name)], copy=True) for name in output_names]

    def profile(self) -> dict[str, Any]:
        with self._lock:
            counts = Counter(
                (str(item.status), item.node_type, item.exec_type) for item in self.request.get_profiling_info()
            )
        return {
            "execution_devices": self.execution_devices,
            "operations": [
                {"status": status, "type": kind, "executor": executor, "count": count}
                for (status, kind, executor), count in counts.items()
            ],
        }


class OpenVinoSessionFactory:
    """Compile ONNX models on one explicit Intel GPU; never retry on CPU."""

    def __init__(self, device: str, *, profiling: bool = False) -> None:
        if not re.fullmatch(r"GPU\.[0-9]+", device):
            raise ValueError("device must be an explicit OpenVINO GPU name")
        import openvino as ov

        self.core = ov.Core()
        self.device = device
        self.profiling = profiling
        self.device_name = self.core.get_property(device, "FULL_DEVICE_NAME")

    def __call__(self, path: Path, **options: Any) -> OpenVinoSession:
        if options:
            raise ValueError("ONNX Runtime options cannot be used with native OpenVINO")
        return OpenVinoSession(self.core, path, self.device, profiling=self.profiling)
