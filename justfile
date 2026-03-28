set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

help:
    @just --list

sync:
    uv sync --all-groups

env:
    if (!(Test-Path .env)) { Copy-Item .env.example .env }

hooks:
    uv run pre-commit install --hook-type pre-commit --hook-type pre-push

bootstrap: sync env hooks

dev:
    uv run uvicorn baozhi_rag.app.main:app --reload

run +args:
    uv run {{args}}

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run mypy

test:
    uv run pytest

test-api:
    uv run pytest tests/api

check:
    uv run ruff check .
    uv run mypy
    uv run pytest

lock:
    uv lock

build:
    uv build

clean:
    $targets = @('build', 'dist', '.coverage', 'htmlcov', '.pytest_cache', '.mypy_cache', '.ruff_cache')
    foreach ($target in $targets) { if (Test-Path $target) { Remove-Item -LiteralPath $target -Recurse -Force } }
    Get-ChildItem -Path src,tests -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
