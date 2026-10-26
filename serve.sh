#!/bin/bash
# =============================================================
# 容器内入口：启动 vLLM OpenAI 兼容服务（由容器启动命令调用）
# 模型：AREX-2（Qwen3_5ForConditionalGeneration，27B 稠密多模态，bf16 原生精度）
# 硬件：2 x NVIDIA GPU（如 H100 80GB；由 start_container.sh 指定物理卡，容器内可见为 0、1），TP=2
# 框架：vLLM v0.31.0（镜像 vllm/vllm-openai:v0.31.0）
# 定位：生产级部署，标准 OpenAI Chat Completions 协议，
#       文本对话 + 图像理解 + 视频理解 + 工具调用 + 思考/非思考模式
# =============================================================
set -euo pipefail

MODEL_DIR="/models/AREX-2"
PORT=8000
TP_SIZE=2

# 对外服务的模型名（下游智能体 / OpenAI 客户端填这个名字）
SERVED_NAME="arex-2"

# 可选：API 鉴权（由 start_container.sh 透传 AREX_API_KEY）
API_KEY="${AREX_API_KEY:-}"

# -------------------------------------------------------------
# 推理特性开关（由 start_container.sh 经 -e 透传，可随时按需切换）
# -------------------------------------------------------------
# 重要：本 AREX-2 checkpoint 不含 MTP 模块（权重索引中无 mtp.* 键），
#   因此【不可】启用 MTP 投机解码（--speculative-config mtp 会因缺权重启动失败）。
#   本方案速度优化依赖：前缀缓存（默认开）+ fp8 KV Cache + CUDA Graph + TP=2。
# ENABLE_PREFIX_CACHING：前缀缓存（APC）。
#   默认 1（开）—— agent 多轮对话 / 共享 system prompt 场景下重复前缀直接命中，
#   TTFT 可降 50%~90%，对 deepseek harness 类调用方收益最大。
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# KV_CACHE_DTYPE：KV Cache 精度。默认 fp8（H100 原生支持 FP8，
#   KV 池容量约翻倍，质量影响可忽略；如需最保守可设 auto=bf16）。
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# REASONING_EFFORT：服务端默认思考深度（xhigh / medium / low）。
#   留空 = 使用模型默认（xhigh，思考最充分但输出更长）。
#   单个请求仍可用 extra_body chat_template_kwargs.reasoning_effort 覆盖。
REASONING_EFFORT="${REASONING_EFFORT:-}"
case "${REASONING_EFFORT}" in
    ""|xhigh|medium|low) : ;;
    *) echo "WARN: REASONING_EFFORT=${REASONING_EFFORT} 非法（xhigh/medium/low/空），忽略"; REASONING_EFFORT="" ;;
esac

# GPU_MEMORY_UTILIZATION：显存利用率上限。默认 0.90（H100 80GB 留 10% 余量防 OOM）。
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# MAX_MODEL_LEN：上下文长度。默认 262144（模型原生上限）。
#   若显存紧张（KV 池不足告警）可降至 131072。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

echo "===== AREX-2 vLLM Server (2xGPU / TP=2) ====="
echo "model dir      : ${MODEL_DIR}"
echo "served name    : ${SERVED_NAME}"
echo "tensor parallel: ${TP_SIZE}"
echo "port           : ${PORT}"
echo "prefix caching : ${ENABLE_PREFIX_CACHING}"
echo "kv cache dtype : ${KV_CACHE_DTYPE}"
echo "reasoning effort(server default): ${REASONING_EFFORT:-<model default xhigh>}"
echo "max model len  : ${MAX_MODEL_LEN}"
echo "gpu mem util   : ${GPU_MEMORY_UTILIZATION}"
vllm --version || true
date

# 参数说明（简）：
# - bf16 原生 checkpoint，禁止追加 --quantization。
# - TP=2 双卡张量并行：权重切半（约 27GB/卡），KV 池两卡合计约 80GB（fp8）。
# - reasoning-parser qwen3：思考内容拆到 reasoning_content，正文在 content。
# - tool-call-parser qwen3_xml：官方模板为 XML 工具格式，
#   解析后按标准 message.tool_calls 数组返回（OpenAI 协议）。
# - v0.31.0 已移除 --disable-log-requests；调试需要时加 --enable-log-requests。
ARGS=(
    "${MODEL_DIR}"
    --served-model-name "${SERVED_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --host 0.0.0.0
    --port "${PORT}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs 32
    --max-num-batched-tokens 16384
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --trust-remote-code
)

# KV Cache 精度（fp8 / auto）
if [ "${KV_CACHE_DTYPE}" != "auto" ] && [ -n "${KV_CACHE_DTYPE}" ]; then
    ARGS+=( --kv-cache-dtype "${KV_CACHE_DTYPE}" )
fi

# 前缀缓存
if [ "${ENABLE_PREFIX_CACHING}" = "1" ]; then
    ARGS+=( --enable-prefix-caching )
    echo "mode           : prefix caching ON"
else
    echo "mode           : prefix caching OFF"
fi

# 服务端默认思考深度（可选）
if [ -n "${REASONING_EFFORT}" ]; then
    ARGS+=( --default-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}" )
fi

# API 鉴权（可选）
if [ -n "${API_KEY}" ]; then
    ARGS+=( --api-key "${API_KEY}" )
    echo "auth           : API key enabled"
else
    echo "auth           : disabled (内网模式)"
fi

# -------------------------------------------------------------
# 本地媒体文件访问（企业级：agent 调用方可能传 file:// URL 引用容器内文件）
# 指向允许被 file:// 引用的目录（默认 /data，即挂载 cat.jpg / demo.mp4 的目录）。
# vLLM 默认出于安全禁止读取本地文件；留空则仅支持 data URL（base64）。
# -------------------------------------------------------------
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-/data}"
if [ -n "${ALLOWED_MEDIA_PATH}" ]; then
    ARGS+=( --allowed-local-media-path "${ALLOWED_MEDIA_PATH}" )
    echo "local media    : ${ALLOWED_MEDIA_PATH} (file:// URL 已放行)"
else
    echo "local media    : disabled (仅 data URL)"
fi

# 可选：限制单请求多模态输入数量（防单请求吃爆显存）。默认不限制。
# ARGS+=( --limit-mm-per-prompt image=4,video=1 )

# DRY_RUN=1：仅打印最终启动参数并退出（供预检参数兼容性）
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN_CMD: ${ARGS[*]}"
    exit 0
fi

echo "启动命令：vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
