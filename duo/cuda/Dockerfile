ARG DUO_APP_IMAGE=git.v-san.dev/homeai/speaches-duo@sha256:15336e1c988f28dc58fae7381b7a1e253efb319d3cca6954157dcab61bb8209f
FROM ${DUO_APP_IMAGE} AS app
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:14d94b039cb94bbd5da559f303b46bc4b0d5d6c24ab1a9d7b186e566ed3400dc
RUN apt-get update && apt-get install -y --no-install-recommends python3 ffmpeg libportaudio2 libopus0 libglib2.0-0 curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=app /opt/speaches-venv /opt/speaches-venv
COPY --from=app /usr/local/bin/uv /usr/local/bin/uv
COPY --from=app /app /app
COPY src/speaches/executors/duo_backend.py /app/src/speaches/executors/duo_backend.py
COPY pyproject.toml /app/pyproject.toml
ENV PATH=/opt/speaches-venv/bin:${PATH} PYTHONDONTWRITEBYTECODE=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy DO_NOT_TRACK=1 DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 PYANNOTE_METRICS_ENABLED=0 UVICORN_HOST=0.0.0.0 UVICORN_PORT=8000
RUN uv pip uninstall --python /opt/speaches-venv/bin/python onnxruntime speaches && uv pip install --python /opt/speaches-venv/bin/python 'onnxruntime-gpu==1.31.1' && uv pip install --python /opt/speaches-venv/bin/python --no-deps /app && python -c 'import onnxruntime as r; assert r.__version__=="1.31.1"; assert "CUDAExecutionProvider" in r.get_available_providers(); print(r.get_available_providers())'
RUN useradd --create-home --uid 10001 speaches
USER 10001:10001
WORKDIR /app
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
LABEL dev.homelab.profile=speaches-duo-cuda-ort1311
