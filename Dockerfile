FROM python:3.13-slim-bookworm

ARG APT_MIRROR=https://mirrors.aliyun.com

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
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

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN mkdir -p /app/data/uploads /app/data/tmp/converted

EXPOSE 8000

CMD ["uvicorn", "baozhi_rag.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
