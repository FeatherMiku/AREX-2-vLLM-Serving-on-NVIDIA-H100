<div align="center">

# AREX-2 vLLM Serving on NVIDIA H100

**基于 vLLM 的 AREX-2 27B 多模态大模型生产级部署方案**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-v0.31.0-blue.svg)](https://github.com/vllm-project/vllm)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]
[![GPU](https://img.shields.io/badge/GPU-2x80GB%20%7C%20TP%3D2-green.svg)]

</div>

面向 **BAAI AREX-2**（27B 稠密多模态大模型）的容器化生产部署与运维方案，基于
**vLLM v0.31.0** 官方镜像，双卡张量并行，提供严格符合 **OpenAI Chat Completions**
协议的推理服务，可直接接入各类 agent / harness / 业务系统，客户端零改造。

## 项目简介

| | |
|---|---|
| 模型 | [BAAI/AREX-2](https://modelscope.cn/models/BAAI/AREX-2)（27B 稠密，bf16 原生，多模态：文本 + 图像 + 视频） |
| 推理引擎 | vLLM v0.31.0（官方镜像 `vllm/vllm-openai:v0.31.0`） |
| 并行方式 | 2 卡 TP=2 张量并行（显存充足时单卡亦可参考修改） |
| 接口协议 | OpenAI Chat Completions（流式 SSE + usage、tool_calls、reasoning_content 全支持） |
| 部署形态 | Docker 容器：HEALTHCHECK 探活 / 自动重启 / 日志轮转 / 启动预检 / 可选 API Key 鉴权 |

> 模型权重约 51GB（11 个 safetensors 分片），获取方式见下文[快速开始](#快速开始)。

## 功能特性

- ✅ **标准 OpenAI 接口**：`/v1/chat/completions`、`/v1/models`、`/health`、`/metrics`，任意 OpenAI SDK / HTTP 客户端零改造接入
- ✅ **多模态理解**：图像（JPG/PNG）与视频（MP4）；同时支持 base64 data URL 与 `file://` 本地文件引用（内网 agent 场景）
- ✅ **思考模式**：深度 xhigh（默认）/ medium / low 三档，支持服务端默认或单请求级切换；思考内容经 `reasoning-parser` 分离至 `reasoning_content`，不污染 `content`
- ✅ **工具调用（Tool Calls）**：标准 `tools` 参数 + `message.tool_calls` 输出，开箱即用于 agent 工具编排
- ✅ **256K 长上下文**（可配置）：`--max-model-len 262144`，适配 agent 长 system prompt / 多轮对话
- ✅ **推理优化**：fp8 KV Cache（容量约翻倍）、前缀缓存（共享前缀 TTFT 实测降约 11 倍）、CUDA Graph、FlashInfer 融合
- ✅ **企业级运维**：Docker HEALTHCHECK、`--restart unless-stopped`、日志轮转（100MB×5）、启动预检（权重/GPU/驱动/镜像/工具链）、可选 `API_KEY` 鉴权、编译缓存卷加速二次启动
- ✅ **验收工具内置**：`smoke_test.py`（9 项协议合规验收）+ `bench_speed.py`（速度基线验收）

## 硬件与依赖

| 依赖 | 要求 | 说明 |
|---|---|---|
| GPU | 2 × 80GB（推荐 H100 HBM3） | TP=2；bf16 权重约 27GB/卡 + KV 池约 43GB/卡。48GB 卡可尝试下调 `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` |
| NVIDIA 驱动 | ≥ 580（默认镜像为 CUDA 13.0 构建） | 驱动 525~579 时脚本自动切换 cu129 镜像（CUDA 12.9 构建） |
| Docker | ≥ 24 | 需安装 `nvidia-container-toolkit` |
| 模型权重 | BAAI/AREX-2 | 约 51GB，从 ModelScope 下载（见快速开始） |
| 客户端 | 任意 OpenAI SDK | Python 示例基于 `openai` 包（1.x / 3.x 均验证） |

## 目录结构

```
├── README.md               # 本文档
├── LICENSE                 # MIT 许可证
├── serve.sh                # 容器内入口：vLLM 启动参数（全部推理开关在此配置）
├── start_container.sh      # 宿主机启动脚本：预检 + 构建容器（幂等，可重复执行）
├── stop_container.sh       # 停止并删除容器（释放 GPU）
├── healthcheck.sh          # 容器健康检查（/health 探活）
├── smoke_test.py           # 9 项 OpenAI 协议合规验收测试
└── bench_speed.py          # 速度基线验收（单流 decode 速度 / 长生成 / 长上下文 TTFT）
```

## 快速开始

### 1. 下载模型权重

```bash
# 安装 modelscope CLI（也可在 ModelScope 页面手动下载）
pip install modelscope

# 下载到 /opt/arex2/AREX-2（路径任意，与下文 MODEL_PATH 保持一致即可）
modelscope download --model BAAI/AREX-2 --local_dir /opt/arex2/AREX-2

# 校验完整性：应有 11 个 model-000XX-of-00011.safetensors 分片 + config.json 等文件
ls /opt/arex2/AREX-2/
```

### 2. 配置环境变量

全部参数支持环境变量覆盖，默认值开箱即用（要求 0、1 号 GPU 与 8020 端口空闲）：

```bash
export MODEL_PATH=/opt/arex2/AREX-2   # 宿主机权重完整目录（必填）
export GPU_DEVICES=0,1                # 要独占的两张物理 GPU 编号（逗号分隔）
export HOST_PORT=8020                 # 宿主机暴露端口
```

可选：将多模态验收素材 `cat.jpg` / `demo.mp4` 放在脚本目录（或用 `TEST_DATA_DIR` 指定）。
素材缺失不影响服务启动，仅影响 `smoke_test.py` 的多模态验收项。

### 3. 启动服务

```bash
bash start_container.sh
```

脚本自动完成：权重完整性预检（11 分片 + 配置清单）→ GPU / 驱动检查（旧驱动自动切换
cu129 镜像）→ nvidia-container-toolkit 检查 → 镜像拉取（首次）→ 容器网络准备 →
幂等重建 → 启动容器（含 HEALTHCHECK / 日志轮转 / 自动重启 / 编译缓存卷）。

> 模型加载约需 3-10 分钟；首次启动含算子编译，编译缓存卷 `arex2-compile-cache` 生效后，
> 后续重建启动显著更快。

### 4. 等待服务就绪

```bash
docker logs -f arexvllm        # 跟踪启动日志
docker inspect --format='{{.State.Health.Status}}' arexvllm
# starting → healthy 即为就绪
```

### 5. 验证

```bash
# 快速冒烟
curl http://127.0.0.1:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"arex-2","messages":[{"role":"user","content":"你好，一句话介绍你自己"}],"max_tokens":100}'

# 完整验收（9 项：文本 / 流式 / 思考 / 非思考 / 工具 / 图像 / 视频 / 本地文件 / 错误处理）
pip install openai
python3 smoke_test.py --base-url http://127.0.0.1:8020/v1

# 速度基线（单流 decode 速度，验收目标 ≥ 60 tok/s）
python3 bench_speed.py --base-url http://127.0.0.1:8020/v1
```

## 配置参数

### 容器级（start_container.sh 环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `/opt/arex2/AREX-2` | 宿主机权重完整目录（必填） |
| `GPU_DEVICES` | `0,1` | 独占的物理 GPU 编号（逗号分隔，恰好 2 个） |
| `HOST_PORT` | `8020` | 宿主机暴露端口（容器内端口固定 8000） |
| `IMAGE` | `vllm/vllm-openai:v0.31.0` | vLLM 官方镜像（CUDA 13.0 构建） |
| `IMAGE_FALLBACK` | `vllm/vllm-openai:v0.31.0-cu129` | 旧驱动回退镜像（CUDA 12.9 构建） |
| `CONTAINER_NAME` | `arexvllm` | 容器名 |
| `DOCKER_NET` | `arex2-net` | Docker 网络名（用于容器间互调） |
| `TEST_DATA_DIR` | 脚本所在目录 | 多模态验收素材 `cat.jpg` / `demo.mp4` 所在目录（可选，存在才挂载） |
| `AREX_API_KEY` | 空 | 非空时启用 `--api-key` 鉴权（对外暴露强烈建议开启） |
| `ENABLE_PREFIX_CACHING` | `1` | 前缀缓存开关 |
| `KV_CACHE_DTYPE` | `fp8` | KV Cache 精度：`fp8`（默认，容量约翻倍）/ `auto`（bf16） |
| `REASONING_EFFORT` | 空（模型默认 xhigh） | 服务端默认思考深度：`xhigh` / `medium` / `low` |
| `GPU_MEMORY_UTILIZATION` | `0.90` | 显存利用率上限（留 10% 余量防 OOM） |
| `MAX_MODEL_LEN` | `262144` | 上下文长度（显存紧张可降为 `131072`） |
| `ALLOWED_MEDIA_PATH` | `/data` | `file://` URL 可引用的容器内目录（留空 = 禁止本地文件引用） |
| `AUTO_IMAGE_FALLBACK` | `1` | 驱动 < 580 时自动切换 cu129 镜像 |

### 引擎级（serve.sh 开关，容器内生效）

| 参数 | 值 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | `2` | 双卡张量并行 |
| `--dtype` | `bfloat16` | 与 checkpoint 原生精度一致（禁止追加 `--quantization`） |
| `--max-model-len` | `262144` | 模型原生上下文上限 |
| `--gpu-memory-utilization` | `0.90` | 显存余量 10% |
| `--kv-cache-dtype` | `fp8` | KV 池容量约翻倍，质量影响可忽略 |
| `--max-num-seqs` | `32` | 并发请求上限，超出排队 |
| `--max-num-batched-tokens` | `16384` | 混合长短请求推荐值；调低 8192 可优化单流 ITL |
| `--reasoning-parser` | `qwen3` | 思考内容 → `reasoning_content`，正文 → `content` |
| `--enable-auto-tool-choice` + `--tool-call-parser` | `qwen3_xml` | 官方模板 XML 工具格式，输出标准 `message.tool_calls` |
| `--allowed-local-media-path` | `/data` | 放行 `file:///data/...` 本地文件引用 |
| `--api-key` | 环境变量 `AREX_API_KEY` | 鉴权（空 = 不鉴权，仅限内网） |
| `--chat-template` | 镜像内置（官方模板） | 支持 `enable_thinking` / `reasoning_effort` / tools / 多模态 |

## 接口调用

Base URL：`http://<服务器IP>:8020/v1`，模型名：`arex-2`。
启用鉴权后，所有请求需附加 `Authorization: Bearer <AREX_API_KEY>`。

### 文本对话（含思考模式）

```bash
curl http://127.0.0.1:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arex-2",
    "messages": [{"role": "user", "content": "用一句话解释什么是 KV Cache"}],
    "max_tokens": 512,
    "temperature": 1.0, "top_p": 0.95
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8020/v1", api_key="EMPTY")

r = client.chat.completions.create(
    model="arex-2",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache"}],
    max_tokens=512,
    temperature=1.0, top_p=0.95,   # thinking 场景官方推荐采样参数
)
m = r.choices[0].message
print(m.reasoning_content)   # 思考段（reasoning-parser 分离，可能为 None）
print(m.content)             # 正文
print(r.usage.completion_tokens_details.reasoning_tokens)  # 思考 token 数
```

> 思考模式默认开启（深度 xhigh）。`reasoning_content` 与 `content` 严格分离，
> agent 侧可只渲染 `content`、折叠展示 `reasoning_content`。

### 控制思考模式 / 深度

```python
# 单请求级覆盖（vLLM 0.31 原生支持，无需重启）
extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}     # low / medium / xhigh
extra_body={"chat_template_kwargs": {"enable_thinking": False}}      # 关闭思考（首 token 最快）
```

服务端级默认深度：启动时 `REASONING_EFFORT=medium bash start_container.sh`（重建生效）。

### 流式（SSE + usage）

```python
stream = client.chat.completions.create(
    model="arex-2",
    messages=[{"role": "user", "content": "写一首关于秋天的短诗"}],
    max_tokens=512,
    stream=True,
    stream_options={"include_usage": True},   # 最后一个 chunk 携带 usage
)
for chunk in stream:
    if not chunk.choices:
        continue
    d = chunk.choices[0].delta
    if d.content:
        print(d.content, end="", flush=True)
```

> 注：vLLM 0.31 流式增量中，思考段字段的实际键名为 `reasoning`（extra 字段），
> 正文为 `content`；客户端解析时两者都识别即可获得完整思考流。

### 工具调用（agent 工具编排）

```bash
curl http://127.0.0.1:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arex-2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "城市名"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 1024
  }'
```

模型返回标准 `choices[0].message.tool_calls`（含 `function.name` 与 JSON 字符串
`function.arguments`），执行后将 `role: "tool"` 结果回传即可完成多轮工具循环。

### 图像理解

```python
import base64
from pathlib import Path

b64 = base64.b64encode(Path("cat.jpg").read_bytes()).decode()
r = client.chat.completions.create(
    model="arex-2",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "描述一下图片里有什么内容"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}],
    max_tokens=1024,
)
print(r.choices[0].message.content)
```

内网 agent 场景可直接引用容器内文件（需挂载到 `/data/` 且 `ALLOWED_MEDIA_PATH` 非空）：

```python
{"type": "image_url", "image_url": {"url": "file:///data/cat.jpg"}}
```

### 视频理解

```python
b64 = base64.b64encode(Path("demo.mp4").read_bytes()).decode()
r = client.chat.completions.create(
    model="arex-2",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "视频的内容是什么"},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
    ]}],
    max_tokens=1024,
)
print(r.choices[0].message.content)
```

### 其他端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/models` | GET | 模型列表（返回 `arex-2` 及 `max_model_len`） |
| `/health` | GET | 就绪探活（200 = 就绪） |
| `/metrics` | GET | Prometheus 指标（KV 使用率、排队数、token 吞吐等） |

错误响应为标准 OpenAI 格式：`{"error": {"message": ..., "type": ..., "code": ...}}`。

## 运维

| 操作 | 命令 |
|---|---|
| 实时日志 | `docker logs -f arexvllm` |
| 健康状态 | `docker inspect --format='{{.State.Health.Status}}' arexvllm` |
| 服务探活 | `curl -s http://127.0.0.1:8020/health`（200 = 就绪） |
| Prometheus 指标 | `curl -s http://127.0.0.1:8020/metrics`（KV 使用率、排队数、token 吞吐等） |
| 停止服务 | `bash stop_container.sh`（释放 GPU，保留编译缓存卷） |
| 修改配置重启 | 修改环境变量后重跑 `bash start_container.sh`（幂等重建） |
| 日志轮转 | json-file 驱动，单文件 100MB × 5 份自动轮转（已内置） |
| 自动重启 | `--restart unless-stopped`（宿主重启后自动拉起，手动 stop 除外） |

容器内关键路径：权重 `/models/AREX-2`、部署脚本 `/opt/arex-deploy`、多模态素材 `/data`。

容器间互调：`docker network connect arex2-net <你的容器>` 后使用
`http://arexvllm:8000/v1`（容器网络内端口 8000）。

## 测试与验收

### smoke_test.py — 9 项协议合规验收

| # | 测试项 | 关键断言 |
|---|---|---|
| 1 | 模型列表 | `/v1/models` 返回 `arex-2` |
| 2 | 非流式对话 | `choices[0].message.content` 非空，usage 正常 |
| 3 | 思考模式 | `reasoning_content` 与 `content` 正确分离 |
| 4 | 非思考模式 | `enable_thinking=false` 生效，无 reasoning 字段 |
| 5 | 工具调用 | 标准 `tool_calls` 结构，参数 JSON 可解析 |
| 6 | 图像理解 | 图像描述输出合理（需 `cat.jpg`） |
| 7 | 视频理解 | 视频描述输出合理（需 `demo.mp4`） |
| 8 | 本地文件引用 | `file:///data/...` 模式（需素材挂载 + `ALLOWED_MEDIA_PATH`） |
| 9 | 错误处理 | 不存在模型名返回标准 OpenAI 错误格式 |

```bash
python3 smoke_test.py --base-url http://127.0.0.1:8020/v1 \
    [--api-key <KEY>] [--image cat.jpg] [--video demo.mp4]
```

### bench_speed.py — 速度基线验收

- **单流 decode 速度**（非思考模式，5 轮取中位数）：验收目标默认 ≥ 60 tok/s（`--target` 可调）
- **长生成稳态速度**（2048 token 预算）
- **长上下文 TTFT**（约 16K token 提示词：首轮 vs 前缀缓存命中后）

```bash
python3 bench_speed.py --base-url http://127.0.0.1:8020/v1 [--runs 5] [--target 60]
```

decode 速度口径 = (completion_tokens − 1) / (末 token 时间 − 首 token 时间)，
即排除 prefill（TTFT）后的纯增量解码速度。

### 参考实测数据（2 × H100 80GB HBM3，bf16，TP=2）

> 以下为本方案在 2×H100 80GB 环境下的实测值，**仅作参考**；其他 GPU / 驱动 /
> 镜像环境请以 `bench_speed.py` 本地实测为准。

| 场景 | 结果 |
|---|---|
| 单流 decode 速度（≤512 token 输出，非思考，5 轮） | **中位数 79.9 tok/s，最高 80.4 tok/s** |
| 长生成稳态（2048 token 预算，实出 1287~1362 token） | 78.8 tok/s |
| 思考模式单流 decode（xhigh，多轮实测） | 79.2 ~ 79.5 tok/s（与思考深度无关，速率恒定） |
| 短提示词 TTFT | 0.06 ~ 0.11 s |
| 12,009 字长提示词 TTFT | 首轮 0.92 s；**前缀缓存命中后 0.08 s（约 11 倍）** |
| 多模态（图像 360 / 视频 2,721 prompt tokens）TTFT | 0.11 s / 0.48 s |
| 显存占用 | 权重约 27GB/卡 + KV 池约 43GB/卡（合计约 74GB/卡） |
| 单流吞吐 | 80 tok/s ≈ 4,800 tokens/分钟，约为人类阅读速度的 4~5 倍 |
| 256K 上下文理论并发 | ≈ 10.36 路（KV 池 2,714,847 tokens） |

decode 为显存带宽瓶颈：每 token 需读取约 27GB/卡 权重，H100 HBM3 3.35TB/s
理论上限约 124 tok/s，实测约 80 tok/s 达理论带宽 65%，属 vLLM bf16 稠密模型的
正常效率区间。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `cannot set both Count and DeviceIDs on device request` | 部分 docker + nvidia-container-toolkit 版本组合要求多卡 `--gpus` 的内嵌引号写法；本脚本已按此处理，请勿手动去掉引号 |
| `Free memory less than desired utilization` | 目标 GPU 被其他进程占用；释放进程或下调 `GPU_MEMORY_UTILIZATION` |
| `could not select device driver "nvidia"` | 未安装 nvidia-container-toolkit，安装后需 `sudo systemctl restart docker` |
| 驱动 < 580 时默认镜像无法启动 | 脚本自动切换 cu129 镜像（`AUTO_IMAGE_FALLBACK=1` 默认开启）；或升级驱动 |
| `file:///data/...` 请求报 400 | ① 文件必须已挂载到容器 `/data/` 目录；② `ALLOWED_MEDIA_PATH` 不能留空 |
| 长请求 `finish_reason=length` 且 `content` 为空 | 思考模式下思考块占满 `max_tokens` 预算被截断；调大 `max_tokens` 或降低思考深度（见已知限制） |
| 客户端 `ModuleNotFoundError: No module named 'openai'` | `pip install openai`（仅客户端需要，与服务无关） |
| 二次重建启动变慢 / 变快 | 首次启动含算子编译；编译缓存卷 `arex2-compile-cache` 生效后二次启动显著加快 |

## 已知限制

1. **无法启用投机解码**：AREX-2 官方 checkpoint 不含 MTP（multi-token prediction）
   权重，vLLM `--speculative-config mtp` 不可用，启动参数中已显式关闭相关路径。
2. **思考块截断行为**：思考模式下思考块占用 `max_tokens` 预算，
   `finish_reason=length` 时若思考块被截断，`content` 可能为空（思考内容也可能被丢弃）。
   调用方应处理：调大 `max_tokens`、降低 `reasoning_effort` 或关闭思考。
3. **bf16 原生精度**：checkpoint 为 bf16，禁止追加 `--quantization`。
   如需更高单流速度，可等待官方 FP8 checkpoint 或引入独立 draft 模型（复杂度高）。
4. **`file://` 本地文件引用**：面向内网 agent 场景设计。对外暴露服务时务必启用
   `AREX_API_KEY`，并将 `ALLOWED_MEDIA_PATH` 收敛到最小必要目录（或留空禁止）。

## License

MIT License，详见 [LICENSE](./LICENSE)。

模型权重以 [BAAI/AREX-2](https://modelscope.cn/models/BAAI/AREX-2) 发布页面
的许可证条款为准；本仓库不包含任何模型权重。

---

*如果这个项目对你有帮助，欢迎点个 ⭐ Star。*
