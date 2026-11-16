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
# Build the pinned upstream onnx-asr with a reviewed session-factory seam. The
# resulting wheel has its own RECORD; neither the source nor the patch is hidden.
RUN git clone --quiet https://github.com/istupakov/onnx-asr.git /tmp/onnx-asr-source \
 && git -C /tmp/onnx-asr-source checkout --quiet --detach a2fd7530c01a78727422e592bc67ab98d96a9697 \
 && echo '15879d724aab4fc71bd65bd14a38a5070ddc5788160a7a628d0766d53f51ba33  /tmp/onnx-asr-source/src/onnx_asr/models/gigaam.py' | sha256sum -c - \
 && echo '9c01cabe03d136e40354b9680f96927c6199496228e4304e0a73e5baf6df6a54  /app/duo/onnx-asr-session-factory.patch' | sha256sum -c - \
 && git -C /tmp/onnx-asr-source apply --check /app/duo/onnx-asr-session-factory.patch \
 && git -C /tmp/onnx-asr-source apply /app/duo/onnx-asr-session-factory.patch \
 && echo 'e38877eab8f793c5d96d4f739b774871bf2dee3b3aebf05faa16aa8587d47fc5  /tmp/onnx-asr-source/src/onnx_asr/models/gigaam.py' | sha256sum -c - \
 && uv venv --python /usr/bin/python3.12 /tmp/onnx-asr-build \
 && uv pip install --no-config --python /tmp/onnx-asr-build/bin/python --require-hashes -r /app/duo/build-requirements.txt \
 && SOURCE_DATE_EPOCH=1792587371 VIRTUAL_ENV=/tmp/onnx-asr-build UV_PROJECT_ENVIRONMENT=/tmp/onnx-asr-build uv build --python /tmp/onnx-asr-build/bin/python --no-build-isolation --wheel --out-dir /tmp/onnx-asr-wheel /tmp/onnx-asr-source \
 && uv pip install --python /opt/speaches-venv/bin/python --no-deps --reinstall-package onnx-asr /tmp/onnx-asr-wheel/*.whl \
 && uv pip uninstall --python /opt/speaches-venv/bin/python hatchling hatch-vcs setuptools-scm vcs-versioning onnx onnxscript onnx-ir \
 && python /app/duo/check_dependency_baseline.py \
 && echo 'e38877eab8f793c5d96d4f739b774871bf2dee3b3aebf05faa16aa8587d47fc5  /opt/speaches-venv/lib/python3.12/site-packages/onnx_asr/models/gigaam.py' | sha256sum -c - \
 && rm -rf /tmp/onnx-asr-source /tmp/onnx-asr-build /tmp/onnx-asr-wheel
RUN useradd --create-home --uid 10001 speaches
USER speaches
EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
