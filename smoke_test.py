#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AREX-2 vLLM 服务冒烟测试（OpenAI Chat Completions 合规性验收）—— 双卡 TP=2 部署
覆盖：模型列表 / 非流式（思考模式，reasoning_content 与 content 分离）/ 流式（SSE + usage）/
      非思考模式 / 工具调用（Tool Calls，agent 场景核心）/
      图像理解（cat.jpg）/ 视频理解（demo.mp4）/ 错误处理（标准 OpenAI 错误格式）。

用法：
    pip install openai
    python3 smoke_test.py --base-url http://127.0.0.1:8020/v1
    python3 smoke_test.py --base-url http://宿主机IP:8020/v1 --api-key 你的key
    容器内互调：python3 smoke_test.py --base-url http://arexvllm:8000/v1

多模态测试文件默认在当前目录查找 cat.jpg / demo.mp4，可用 --image / --video 覆盖为任意本地路径
（以 base64 data URL 方式提交，为标准 OpenAI 协议用法；另附容器内文件路径模式验证）
"""
import argparse
import base64
import json
import mimetypes
import sys
import time

PASS = 0
FAIL = 0


def record(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{tag}] {name}" + (f" —— {detail}" if detail else ""))
    return ok


def b64_data_url(path, mime_hint=None):
    mime = mimetypes.guess_type(path)[0] or mime_hint or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--api-key", default="EMPTY", help="服务端开了 --api-key 时必填")
    parser.add_argument("--model", default="arex-2")
    parser.add_argument("--image", default="cat.jpg", help="测试图片本地路径（默认当前目录 cat.jpg）")
    parser.add_argument("--video", default="demo.mp4", help="测试视频本地路径（默认当前目录 demo.mp4）")
    args = parser.parse_args()

    try:
        from openai import OpenAI, APIStatusError
    except ImportError:
        print("请先安装: pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    print(f"目标服务: {args.base_url}  模型: {args.model}\n")

    # ============ 1. 模型列表 ============
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        ok = record("1. GET /v1/models", args.model in ids, f"ids={ids}")
        if not ok:
            print("!! 模型名不匹配，后续测试将失败。"); sys.exit(1)
    except Exception as e:
        record("1. GET /v1/models", False, str(e)); sys.exit(1)

    # ============ 2. 非流式对话（思考模式，难题触发 reasoning） ============
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "一个 8x8 棋盘放 8 个车，要求任意两个车不能同行同列。"
                                           "一共有多少种放法？请仔细推导后给出最终数字。"},
            ],
            temperature=1.0,
            top_p=0.95,
            max_tokens=2048,
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        usage = resp.usage
        reasoning = getattr(msg, "reasoning_content", None) or ""
        ok = (
            resp.choices[0].finish_reason in ("stop", "length")
            and usage.prompt_tokens > 0
            and usage.completion_tokens > 0
            and bool((msg.content or "").strip())
        )
        record("2. chat 非流式（思考模式）", ok,
               f"{dt:.1f}s, prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
               f"reasoning_chars={len(reasoning)}, finish_reason={resp.choices[0].finish_reason}")
        if reasoning:
            print(f"--- reasoning_content 前 120 字（reasoning-parser 生效验证）---")
            print(reasoning[:120])
        print("--- 正文前 150 字 ---")
        print((msg.content or "")[:150])
    except Exception as e:
        record("2. chat 非流式（思考模式）", False, str(e))

    # ============ 3. 流式对话（SSE + usage） ============
    try:
        t0 = time.time()
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "一句话解释什么是 KV Cache。"}],
            max_tokens=512,
            stream=True,
            stream_options={"include_usage": True},
        )
        n_reason, n_content, stream_usage, first_content = 0, 0, None, None
        content_all = ""
        for chunk in stream:
            if chunk.usage:
                stream_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                n_reason += 1
            if delta.content:
                if first_content is None:
                    first_content = time.time() - t0
                n_content += 1
                content_all += delta.content
        ok = stream_usage is not None and n_content > 0
        record("3. chat 流式（SSE + usage）", ok,
               f"first_content={round(first_content, 2) if first_content else None}s, "
               f"reasoning_chunks={n_reason}, content_chunks={n_content}, "
               f"usage.completion={stream_usage.completion_tokens if stream_usage else None}")
    except Exception as e:
        record("3. chat 流式（SSE + usage）", False, str(e))

    # ============ 4. 非思考模式（chat_template_kwargs 覆盖） ============
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "返回 1 到 5 的平方，用逗号分隔，不要解释。"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.7,
            top_p=0.80,
            max_tokens=64,
        )
        ans = (resp.choices[0].message.content or "").strip()
        ok = "1, 4, 9, 16, 25" in ans.replace("，", ",")
        record("4. 非思考模式（enable_thinking=false）", ok,
               f"{time.time() - t0:.1f}s -> {ans[:80]}")
    except Exception as e:
        record("4. 非思考模式（enable_thinking=false）", False, str(e))

    # ============ 5. 工具调用（OpenAI tools 协议，agent 依赖） ============
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，例如：北京"}},
                "required": ["city"],
            },
        },
    }]
    tool_ok, last_err = False, ""
    for attempt in range(2):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "请调用 get_weather 工具查询北京现在的天气，必须使用工具。"}],
                tools=tools,
                tool_choice="auto",
                temperature=1.0,
                top_p=0.95,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                fn = tcs[0].function
                try:
                    args_json = json.loads(fn.arguments)
                except Exception:
                    args_json = None
                ok = fn.name == "get_weather" and isinstance(args_json, dict) and "city" in args_json
                tool_ok = ok
                record("5. 工具调用（tools 协议）", ok,
                       f"{time.time() - t0:.1f}s, name={fn.name}, arguments={fn.arguments[:120]}")
                break
            last_err = f"attempt{attempt + 1}: 未返回 tool_calls（content={(msg.content or '')[:80]}）"
            print(f"      {last_err}，重试...")
        except Exception as e:
            last_err = str(e)
            print(f"      attempt{attempt + 1} 异常: {e}，重试...")
    if not tool_ok:
        record("5. 工具调用（tools 协议）", False, last_err)

    # ============ 6. 图像理解（cat.jpg，base64 data URL） ============
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片里有什么（一两句话）。"},
                    {"type": "image_url", "image_url": {"url": b64_data_url(args.image, "image/jpeg")}},
                ],
            }],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.7,
            top_p=0.80,
            max_tokens=512,
        )
        ans = (resp.choices[0].message.content or "")
        low = ans.lower()
        hit = any(k in low for k in ("cat", "猫"))
        record("6. 图像理解（cat.jpg）", hit, f"{time.time() - t0:.1f}s -> {ans.strip()[:100]}")
    except Exception as e:
        record("6. 图像理解（cat.jpg）", False, str(e))

    # ============ 7. 视频理解（demo.mp4，base64 data URL） ============
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这段视频的内容（一两句话）。"},
                    {"type": "video_url", "video_url": {"url": b64_data_url(args.video, "video/mp4")}},
                ],
            }],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.7,
            top_p=0.80,
            max_tokens=512,
        )
        ans = (resp.choices[0].message.content or "").strip()
        ok = len(ans) > 10
        record("7. 视频理解（demo.mp4）", ok, f"{time.time() - t0:.1f}s -> {ans[:100]}")
    except Exception as e:
        record("7. 视频理解（demo.mp4）", False, str(e))

    # ============ 8. 容器内文件路径模式（file:// URL 指向容器内文件，企业本地文件场景） ============
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "图片里是什么？（限 10 字内）"},
                    {"type": "image_url", "image_url": {"url": "file:///data/cat.jpg"}},
                ],
            }],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.7,
            top_p=0.80,
            max_tokens=128,
        )
        ans = (resp.choices[0].message.content or "").strip()
        hit = any(k in ans.lower() for k in ("cat", "猫"))
        record("8. 容器内文件路径输入（file:///data/cat.jpg）", hit, f"{time.time() - t0:.1f}s -> {ans[:60]}")
    except Exception as e:
        record("8. 容器内文件路径输入（file:///data/cat.jpg）", False, str(e))

    # ============ 9. 错误处理（错误模型名 → 标准 OpenAI 错误对象） ============
    try:
        try:
            client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "hi"}],
                                           max_tokens=8)
            record("9. 错误处理（404 错误格式）", False, "未抛出异常（不应到达）")
        except APIStatusError as e:
            body = e.response.json() if e.response is not None else {}
            err = body.get("error", {})
            ok = (e.status_code in (400, 404) and err.get("message") and err.get("type"))
            record("9. 错误处理（404 错误格式）", ok,
                   f"status={e.status_code}, error.type={err.get('type')}, error.code={err.get('code')}")
    except Exception as e:
        record("9. 错误处理（404 错误格式）", False, str(e))

    # ============ 汇总 ============
    print("\n" + "=" * 56)
    print(f"冒烟测试汇总：{PASS} 通过 / {FAIL} 失败")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
