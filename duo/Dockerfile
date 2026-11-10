# Rebuild from immutable base images and a dated Debian package snapshot.
FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS uv
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
COPY --from=uv /uv /usr/local/bin/uv
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261110T000000Z trixie main' > /etc/apt/sources.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    DO_NOT_TRACK=1 DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_METRICS_ENABLED=0 UVICORN_HOST=0.0.0.0 UVICORN_PORT=8000
WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY duo/build-requirements.txt /app/duo/build-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv uv venv --python /usr/local/bin/python /opt/venv \
 && uv pip install --no-config --python /opt/venv/bin/python --require-hashes -r duo/build-requirements.txt \
 && uv sync --inexact --frozen --no-dev --no-build-isolation --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --inexact --frozen --no-dev --no-build-isolation
RUN useradd --create-home --uid 1000 speaches
USER speaches
EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
