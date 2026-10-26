#!/bin/bash
# =============================================================
# AREX-2 vLLM 推理服务容器启动脚本（宿主机执行）
# 镜像：vllm/vllm-openai:v0.31.0（容器名默认 arexvllm）
# 硬件：2 x NVIDIA GPU 80GB（如 H100 HBM3），TP=2 张量并行
# 用法：MODEL_PATH=/path/to/AREX-2 GPU_DEVICES=0,1 HOST_PORT=8020 bash start_container.sh
# 幂等：若同名容器已存在，先停止并删除再重建
# 所有配置项均可通过环境变量覆盖（见下方配置区）
# =============================================================
set -euo pipefail

# ---------------- 可按环境修改的配置（全部可用环境变量覆盖） ----------------
MODEL_PATH="${MODEL_PATH:-/opt/arex2/AREX-2}"                   # 宿主机完整权重目录（必须设置）
GPU_DEVICES="${GPU_DEVICES:-0,1}"                               # 本容器独占的物理 GPU 编号（逗号分隔，两个）
HOST_PORT="${HOST_PORT:-8020}"                                  # 宿主机暴露端口
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"                      # CUDA 13.0 构建，驱动需 >= 580
IMAGE_FALLBACK="${IMAGE_FALLBACK:-vllm/vllm-openai:v0.31.0-cu129}"  # 驱动 < 580 时的同版本 CUDA 12.9 构建
CONTAINER_NAME="${CONTAINER_NAME:-arexvllm}"
DOCKER_NET="${DOCKER_NET:-arex2-net}"
TP_SIZE=2                                                       # 固定双卡张量并行
# 多模态验收测试文件目录（cat.jpg / demo.mp4，可选；仅跑 smoke_test.py 多模态项时需要）
TEST_DATA_DIR="${TEST_DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
AREX_API_KEY="${AREX_API_KEY:-}"                                # 可选：留空则不鉴权（仅限内网调试）
AUTO_IMAGE_FALLBACK="${AUTO_IMAGE_FALLBACK:-1}"                 # 驱动过旧时是否自动切换 cu129 镜像

# 推理特性开关（透传给容器内 serve.sh，详见 README 参数表）
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"             # 前缀缓存：1=开（默认）
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"                         # KV Cache 精度：fp8（默认）/ auto
REASONING_EFFORT="${REASONING_EFFORT:-}"                        # 服务端默认思考深度：留空=模型默认 xhigh
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"        # 显存利用率上限
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"                        # 上下文长度
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-/data}"               # file:// URL 可引用的容器内目录（留空=禁止）
# ---------------------------------------------------

if [ -n "${AREX_API_KEY}" ]; then
    AUTH_DESC="enabled"
else
    AUTH_DESC="disabled (内网模式)"
fi

# GPU_DEVICES 校验（逗号分隔，应为恰好 2 个编号）
IFS=',' read -r -a GPU_LIST <<< "${GPU_DEVICES}"
if [ "${#GPU_LIST[@]}" -ne 2 ]; then
    echo "ERROR: GPU_DEVICES 必须为恰好 2 个编号（如 0,1），当前：${GPU_DEVICES}"
    exit 1
fi

echo "[1/7] 前置检查（权重完整性）..."
if [ ! -f "${MODEL_PATH}/config.json" ] || [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    echo "ERROR: 模型目录缺少 config.json 或 model.safetensors.index.json，请确认权重完整：${MODEL_PATH}"
    echo "       下载：modelscope download --model BAAI/AREX-2 --local_dir ${MODEL_PATH}"
    exit 1
fi
# AREX-2 权重结构：model-000XX-of-00011.safetensors（11 个分片，共约 51GB）
SHARDS=$(ls "${MODEL_PATH}"/model-*-of-*.safetensors 2>/dev/null | wc -l)
if [ "${SHARDS}" -ne 11 ]; then
    echo "ERROR: 权重分片数量=${SHARDS}（应=11），模型不完整，禁止启动。"
    exit 1
fi
for f in config.json tokenizer.json preprocessor_config.json processor_config.json chat_template.jinja model.safetensors.index.json; do
    if [ ! -f "${MODEL_PATH}/${f}" ]; then
        echo "ERROR: 缺少 ${f}，模型不完整。"
        exit 1
    fi
done
echo "  权重分片=${SHARDS}（11 分片，约 51GB bf16），OK"

echo "[2/7] 前置检查（GPU / 驱动）..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: 未找到 nvidia-smi，请确认 NVIDIA 驱动已安装。"
    exit 1
fi
GPU_INDEXES=$(nvidia-smi --query-gpu=index --format=csv,noheader)
for g in "${GPU_LIST[@]}"; do
    if ! echo "${GPU_INDEXES}" | grep -qx "${g}"; then
        echo "ERROR: 未找到 ${g} 号 GPU（GPU_DEVICES=${GPU_DEVICES}）。"
        exit 1
    fi
done
echo "  目标 GPU（${GPU_DEVICES}）当前显存占用："
for g in "${GPU_LIST[@]}"; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' -v g="${g}" '$1==g {print "    GPU "$1": "$2" MiB"}'
done
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v a="${GPU_LIST[0]}" -v b="${GPU_LIST[1]}" '($1==a || $1==b) && $2+0 > 1000 {print $1}' | tr '\n' ' ')
if [ -n "${BUSY}" ]; then
    echo "  WARN: GPU ${BUSY}显存占用较高（>1GB），可能被其他进程占用；"
    echo "        若启动报 Free memory less than desired utilization，请先释放该卡或下调 GPU_MEMORY_UTILIZATION。"
fi

# 驱动版本与镜像 CUDA 匹配（v0.31.0 默认构建为 CUDA 13.0，需驱动 >= 580；cu129 构建需 >= 525）
# 注意：不能用 head（head 提前关管道会在 set -o pipefail 下触发 SIGPIPE=141），sed 读满全量输出
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
DRIVER_MAJOR=$(echo "${DRIVER_VER}" | cut -d. -f1)
echo "  NVIDIA 驱动版本：${DRIVER_VER}"
if [ -n "${DRIVER_MAJOR}" ] && [ "${DRIVER_MAJOR}" -lt 580 ] && [[ "${IMAGE}" != *cu* ]]; then
    if [ "${AUTO_IMAGE_FALLBACK}" = "1" ]; then
        echo "  WARN: 驱动 ${DRIVER_VER} < 580，不满足 ${IMAGE}（CUDA 13.0）要求。"
        echo "        已自动切换镜像为 ${IMAGE_FALLBACK}（同版本 vLLM，CUDA 12.9 构建）。"
        IMAGE="${IMAGE_FALLBACK}"
    else
        echo "ERROR: 驱动 ${DRIVER_VER} < 580，${IMAGE} 无法运行。"
        echo "       请改用 ${IMAGE_FALLBACK}，或升级 NVIDIA 驱动后重试（可设 AUTO_IMAGE_FALLBACK=1 自动切换）。"
        exit 1
    fi
fi

echo "[3/7] 检查 NVIDIA Container Toolkit..."
if ! docker info 2>/dev/null | grep -qi "nvidia"; then
    echo "  WARN: docker info 未检测到 nvidia runtime；若 --gpus 报 'could not select device driver'，"
    echo "        请安装 nvidia-container-toolkit 并执行 sudo systemctl restart docker。"
fi

echo "[4/7] 检查镜像 ${IMAGE} ..."
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "  本地无镜像，开始拉取（约数 GB，视网速而定）..."
    docker pull "${IMAGE}"
fi

echo "[5/7] 准备 docker 网络 ${DOCKER_NET}..."
if ! docker network inspect "${DOCKER_NET}" >/dev/null 2>&1; then
    docker network create "${DOCKER_NET}"
fi

echo "[6/7] 清理旧容器..."
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" || true
fi

# 多模态测试文件挂载（可选：存在才挂载；缺失不影响服务启动，仅影响 smoke_test.py 多模态项）
MM_MOUNTS=()
if [ -f "${TEST_DATA_DIR}/cat.jpg" ]; then
    MM_MOUNTS+=( -v "${TEST_DATA_DIR}/cat.jpg":/data/cat.jpg:ro )
else
    echo "  NOTE: 未找到 ${TEST_DATA_DIR}/cat.jpg，跳过挂载（多模态验收前请准备测试图）"
fi
if [ -f "${TEST_DATA_DIR}/demo.mp4" ]; then
    MM_MOUNTS+=( -v "${TEST_DATA_DIR}/demo.mp4":/data/demo.mp4:ro )
else
    echo "  NOTE: 未找到 ${TEST_DATA_DIR}/demo.mp4，跳过挂载（多模态验收前请准备测试视频）"
fi

echo "[7/7] 启动容器 ${CONTAINER_NAME}（GPU ${GPU_DEVICES}，TP=${TP_SIZE}，前缀缓存：${ENABLE_PREFIX_CACHING}，鉴权：${AUTH_DESC}）..."
# 多卡 device 列表必须内嵌双引号：部分 docker 版本 + nvidia-container-toolkit 组合下，
# 无引号的逗号列表会报 "cannot set both Count and DeviceIDs on device request"。
# 外层用双引号以便展开 ${GPU_DEVICES}，内层引号转义为字面量。
# 注意：docker run 续行命令中间不能插注释行（# 会吞掉后续所有参数）。
docker run -d \
    --name "${CONTAINER_NAME}" \
    --hostname "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --network "${DOCKER_NET}" \
    --gpus "\"device=${GPU_DEVICES}\"" \
    --ipc=host \
    -p "${HOST_PORT}:8000" \
    -v "${MODEL_PATH}":/models/AREX-2:ro \
    -v "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)":/opt/arex-deploy:ro \
    ${MM_MOUNTS[@]+"${MM_MOUNTS[@]}"} \
    -v arex2-compile-cache:/root/.cache/vllm \
    -e AREX_API_KEY="${AREX_API_KEY}" \
    -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
    -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
    -e REASONING_EFFORT="${REASONING_EFFORT}" \
    -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    -e ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH}" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=1 \
    --log-driver json-file \
    --log-opt max-size=100m \
    --log-opt max-file=5 \
    --health-cmd "bash /opt/arex-deploy/healthcheck.sh" \
    --health-interval 30s \
    --health-timeout 10s \
    --health-retries 5 \
    --health-start-period 1800s \
    --entrypoint bash \
    "${IMAGE}" \
    /opt/arex-deploy/serve.sh

echo ""
echo "容器已启动。查看实时日志：docker logs -f ${CONTAINER_NAME}"
echo "模型加载约需 3-10 分钟（首次含编译，编译缓存卷生效后更快），健康检查通过（healthy）后服务可用："
echo "  docker inspect --format='{{.State.Health.Status}}' ${CONTAINER_NAME}"
echo ""
echo "服务地址："
echo "  宿主机 / 本机     : http://127.0.0.1:${HOST_PORT}/v1"
echo "  同 docker 网络容器: http://${CONTAINER_NAME}:8000/v1（先 docker network connect ${DOCKER_NET} <容器名>）"
echo "  跨主机            : http://<服务器IP>:${HOST_PORT}/v1"
echo ""
echo "模型名：arex-2    上下文：${MAX_MODEL_LEN} tokens    多模态：image / video"
echo "注意：本服务独占 GPU ${GPU_DEVICES}；其他服务如需使用这些卡，先 bash stop_container.sh。"
