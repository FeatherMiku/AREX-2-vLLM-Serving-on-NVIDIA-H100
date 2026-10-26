#!/bin/bash
# 容器健康检查：vLLM /health 就绪探活（由 docker HEALTHCHECK 调用）
# 模型加载完成前 /health 返回非 2xx，容器状态保持 starting；
# 加载完成后返回 200，容器转为 healthy。
curl -sf --max-time 8 http://127.0.0.1:8000/health >/dev/null 2>&1
