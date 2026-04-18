# syntax=docker/dockerfile:1.7

FROM python:3.13-slim-bookworm

ARG APT_MIRROR=https://mirrors.aliyun.com
ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TIMEOUT_SECONDS=300

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=${PIP_TIMEOUT_SECONDS} \
    PIP_INDEX_URL=${PYPI_MIRROR} \
    UV_DEFAULT_INDEX=${PYPI_MIRROR} \
    UV_INDEX_URL=${PYPI_MIRROR} \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN sed -i \
        -e "s|http://deb.debian.org/debian|${APT_MIRROR}/debian|g" \
        -e "s|https://deb.debian.org/debian|${APT_MIRROR}/debian|g" \
        -e "s|http://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g" \
        -e "s|https://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fonts-noto-cjk \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Keep dependency installation stable when only application sources change.
COPY pyproject.toml uv.lock ./
RUN python -m venv .venv

# Export the lockfile first so pip can download from the configured mirror.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv export \
        --frozen \
        --no-dev \
        --no-editable \
        --no-emit-project \
        --format requirements.txt \
        --output-file requirements.lock.txt \
    && .venv/bin/pip install --cache-dir /root/.cache/pip -r requirements.lock.txt

COPY README.md ./
COPY src ./src
COPY data/domain_dictionary.txt ./data/domain_dictionary.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/pip install --cache-dir /root/.cache/pip --no-deps .

RUN mkdir -p /app/data/uploads /app/data/tmp/converted

EXPOSE 8000

CMD ["uvicorn", "baozhi_rag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
