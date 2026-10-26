#!/bin/bash
# 停止并删除 AREX-2 vLLM 服务容器（释放容器占用的 GPU）
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-arexvllm}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "停止并删除容器 ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}"
    echo "已停止。GPU 已释放。"
    echo "（编译缓存卷 arex2-compile-cache 已保留，下次启动加载更快）"
else
    echo "容器 ${CONTAINER_NAME} 不存在，无需操作。"
fi
