# Isolated Intel profile. This base already contains the Intel OpenCL ICD.
FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS uv
FROM docker.io/openvino/ubuntu24_runtime:2026.4.0@sha256:b326dcae9b91fa25f794a391d9953e65a65096597237e6de431116b184a01718
USER root
COPY --from=uv /uv /usr/local/bin/uv
RUN rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
 && printf '%s\n' 'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20261110T000000Z noble main universe' 'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20261110T000000Z noble-updates main universe' 'deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20261110T000000Z noble-security main universe' > /etc/apt/sources.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/speaches-venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    PATH=/opt/speaches-venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    DO_NOT_TRACK=1 DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_METRICS_ENABLED=0 UVICORN_HOST=0.0.0.0 UVICORN_PORT=8000
WORKDIR /app
COPY duo/intel/pyproject.toml /app/pyproject.toml
COPY duo/intel/uv.lock /app/uv.lock
COPY duo/build-requirements.txt /app/duo/build-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv uv venv --python /usr/bin/python3.12 /opt/speaches-venv \
 && uv pip install --no-config --python /opt/speaches-venv/bin/python --require-hashes -r duo/build-requirements.txt \
 && uv sync --inexact --frozen --no-dev --no-build-isolation --no-install-project
COPY src /app/src
COPY model_aliases.json /app/model_aliases.json
COPY realtime-console /app/realtime-console
COPY duo /app/duo
RUN --mount=type=cache,target=/root/.cache/uv uv sync --inexact --frozen --no-dev --no-build-isolation
RUN useradd --create-home --uid 10001 speaches
USER speaches
EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
