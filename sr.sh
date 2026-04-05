#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.server.yml"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:6888/health/live}"
export APP_ENV="${APP_ENV:-production}"
ENV_FILE_ARGS=(--env-file "${PROJECT_DIR}/.env")

if [[ -f "${PROJECT_DIR}/.env.${APP_ENV}" ]]; then
  ENV_FILE_ARGS+=(--env-file "${PROJECT_DIR}/.env.${APP_ENV}")
fi

usage() {
  cat <<'EOF'
用法:
  ./sr.sh <command> [service]

命令:
  start              启动项目（不重建镜像）
  stop               停止并移除容器（不删卷）
  restart            重启已运行容器
  rebuild            重建镜像并启动项目
  logs [service]     实时查看日志，可选指定服务名
  ps                 查看容器状态
  pull               仅拉取最新代码
  deploy             拉取代码并重建启动
  health             健康检查
  help               显示帮助
EOF
}

require_prerequisites() {
  command -v git >/dev/null
  command -v docker >/dev/null
  command -v curl >/dev/null
  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "未找到 docker-compose.server.yml，停止执行。"
    exit 1
  fi
}

cmd="${1:-help}"
service="${2:-}"

cd "${PROJECT_DIR}"
require_prerequisites

docker_compose() {
  docker compose "${ENV_FILE_ARGS[@]}" -f "${COMPOSE_FILE}" "$@"
}

case "${cmd}" in
  start)
    docker_compose up -d
    docker_compose ps
    ;;
  stop)
    docker_compose down
    ;;
  restart)
    docker_compose restart
    docker_compose ps
    ;;
  rebuild)
    docker_compose build --progress=plain
    docker_compose up -d
    docker_compose ps
    ;;
  logs)
    if [[ -n "${service}" ]]; then
      docker_compose logs -f "${service}"
    else
      docker_compose logs -f
    fi
    ;;
  ps)
    docker_compose ps
    ;;
  pull)
    git pull --ff-only
    ;;
  deploy)
    git pull --ff-only
    docker_compose build --progress=plain
    docker_compose up -d
    docker_compose ps
    ;;
  health)
    curl -i "${HEALTH_URL}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "未知命令: ${cmd}"
    usage
    exit 1
    ;;
esac
