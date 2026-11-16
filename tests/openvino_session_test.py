from concurrent.futures import ThreadPoolExecutor
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from speaches.executors.shared.openvino_session import OpenVinoSession, OpenVinoSessionFactory


class FakeRequest:
    def __init__(self) -> None:
        self.buffer = np.zeros(1, dtype=np.float32)
        self.active = 0
        self.max_active = 0

    def infer(self, inputs: dict) -> dict:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        value = int(inputs["x"][0, 0])
        assert inputs["x"].dtype == np.int64
        assert inputs["x"].flags.c_contiguous
        self.buffer[0] = value
        time.sleep(0.002)
        self.active -= 1
        return {"first": self.buffer, "second": self.buffer + 100}


class FakeCompiled:
    def __init__(self, devices: list[str]) -> None:
        self.devices = devices
        self.request = FakeRequest()

    def get_property(self, name: str) -> list[str]:
        assert name == "EXECUTION_DEVICES"
        return self.devices

    def create_infer_request(self) -> FakeRequest:
        return self.request

    def output(self, name: str) -> str:
        return name


class FakeCore:
    def __init__(self, devices: list[str] | None = None, supported: bool = True) -> None:
        self.compiled = FakeCompiled(devices if devices is not None else ["GPU.0"])
        self.supported = supported
        self.compile_args = None

    def read_model(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(get_ops=lambda: [SimpleNamespace(get_friendly_name=lambda: "node")])

    def query_model(self, _model: object, device: str) -> dict:
        return {"node": device} if self.supported else {}

    def compile_model(self, _model: object, device: str, options: dict) -> FakeCompiled:
        self.compile_args = (device, options)
        return self.compiled


def test_run_orders_and_copies_outputs() -> None:
    core = FakeCore()
    session = OpenVinoSession(core, "model.onnx", "GPU.0")  # type: ignore[arg-type]
    first = session.run(["second", "first"], {"x": [[7]]})
    session.run(["first"], {"x": [[9]]})
    assert [v.item() for v in first] == [107, 7]
    assert core.compile_args == (
        "GPU.0",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": False},
    )


def test_infer_request_is_serialized() -> None:
    core = FakeCore()
    session = OpenVinoSession(core, "model.onnx", "GPU.0")  # type: ignore[arg-type]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda i: session.run(["first"], {"x": [[i]]})[0].item(), range(8)))
    assert outputs == list(range(8))
    assert core.compiled.request.max_active == 1


@pytest.mark.parametrize("devices", [["CPU"], ["GPU.1"], ["GPU.0", "CPU"], []])
def test_wrong_execution_device_is_rejected(devices: list[str]) -> None:
    with pytest.raises(RuntimeError, match="Expected execution"):
        OpenVinoSession(FakeCore(devices), "model.onnx", "GPU.0")  # type: ignore[arg-type]


def test_unsupported_graph_is_rejected() -> None:
    core = FakeCore(supported=False)
    with pytest.raises(RuntimeError, match="does not support"):
        OpenVinoSession(core, "model.onnx", "GPU.0")  # type: ignore[arg-type]
    assert core.compile_args is None


@pytest.mark.parametrize("device", ["GPU", "AUTO", "CPU", "HETERO:GPU,CPU"])
def test_requires_explicit_gpu(device: str) -> None:
    with pytest.raises(ValueError, match="explicit"):
        OpenVinoSessionFactory(device)


def test_unavailable_gpu_fails_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnavailableCore:
        def get_property(self, _device: str, _name: str) -> None:
            raise RuntimeError("missing GPU")

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=UnavailableCore))
    with pytest.raises(RuntimeError, match="missing GPU"):
        OpenVinoSessionFactory("GPU.0")
