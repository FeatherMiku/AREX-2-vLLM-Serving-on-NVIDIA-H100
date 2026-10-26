#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AREX-2 vLLM 服务单流速度基准（双卡 TP=2）

验收口径：
  1) 单流 decode 速度 = (completion_tokens - 1) / (末 token 时间 - 首 token 时间)
     —— 排除 prefill（TTFT）后的纯增量解码速度；非思考模式、多轮取中位数。
  2) 长生成单流速度：max_tokens=2048 的稳态 decode 速度（生产长回答场景）。
  3) 长上下文 TTFT：约 16K token 提示词的首 token 延迟（agent 长 system prompt 场景）。

用法：
    pip install openai
    python3 bench_speed.py --base-url http://127.0.0.1:8020/v1
    python3 bench_speed.py --base-url http://127.0.0.1:8020/v1 --runs 5 --target 60
"""
import argparse
import statistics
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("请先安装: pip install openai")
    sys.exit(1)


def stream_bench(client, model, messages, max_tokens, thinking=False, extra=None):
    """流式请求，返回 (ttft, decode_window, n_tokens, t_total)"""
    extra_body = {}
    if not thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        extra_body.update(extra)
    t0 = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.80,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body or None,
    )
    t_first, t_last, n_chunks, usage = None, None, 0, None
    for chunk in stream:
        now = time.time()
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if t_first is None:
                t_first = now
            t_last = now
            n_chunks += 1
    t_end = time.time()
    n_tokens = usage.completion_tokens if usage else n_chunks
    ttft = (t_first - t0) if t_first else float("nan")
    window = (t_last - t_first) if (t_first and t_last) else float("nan")
    return ttft, window, n_tokens, (t_end - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="arex-2")
    parser.add_argument("--target", type=float, default=60.0, help="单流 decode 速度验收目标（tokens/s）")
    parser.add_argument("--runs", type=int, default=5, help="单流速度重复轮数（取中位数）")
    parser.add_argument("--max-tokens", type=int, default=512, help="单流速度测试每轮输出上限")
    parser.add_argument("--skip-long", action="store_true", help="跳过长生成与长上下文测试")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900)

    print("[0/3] 预热（消除 CUDA graph / 编译冷启动影响）...")
    _ = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "你好"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_tokens=16,
    )
    print("  完成")

    prompt = ("请写一篇约 500 字的中文短文，主题：为什么 GPU 适合大模型推理。"
              "要求段落清晰、内容具体。")

    # 1) 单流 decode 速度
    print(f"\n[1/3] 单流 decode 速度（{args.runs} 轮 x {args.max_tokens} tokens，非思考模式）")
    speeds = []
    for i in range(args.runs):
        ttft, window, n, total = stream_bench(
            client, args.model,
            [{"role": "user", "content": prompt}], args.max_tokens)
        speed = (n - 1) / window if window and window > 0 else float("nan")
        speeds.append(speed)
        print(f"  run{i + 1}: completion={n} | TTFT={ttft:.2f}s | "
              f"decode_window={window:.2f}s | 总耗时={total:.2f}s | 速度={speed:.1f} tok/s")
    median = statistics.median(speeds)
    best = max(speeds)
    print(f"  ── 单流 decode 速度：中位数 {median:.1f} tok/s，最高 {best:.1f} tok/s"
          f"（判定目标 >= {args.target:.0f} tok/s）")

    # 2) 长生成单流速度（生产长回答场景）
    long_speed, long_ttft, long_n = float("nan"), float("nan"), 0
    if not args.skip_long:
        print(f"\n[2/3] 长生成单流速度（max_tokens=2048，非思考模式）")
        long_ttft, window, long_n, total = stream_bench(
            client, args.model,
            [{"role": "user", "content": "请写一篇约 1500 字的技术博客，主题：大模型推理服务化的工程实践，"
                                         "涵盖 KV Cache、连续批处理、张量并行与多模态预处理，结构完整。"}],
            2048)
        long_speed = (long_n - 1) / window if window and window > 0 else float("nan")
        print(f"  completion={long_n} | TTFT={long_ttft:.2f}s | 总耗时={total:.2f}s | 稳态速度={long_speed:.1f} tok/s")

        # 3) 长上下文 TTFT（约 16K token 提示词）
        print("\n[3/3] 长上下文 TTFT（约 12000 字中文提示词，非思考模式，max_tokens=32）")
        filler = ("这是一段用于测试长上下文首 token 延迟的填充文本。大模型推理服务在生产环境中需要处理"
                  "非常长的系统提示词与多轮对话历史，前缀缓存能够显著降低这类请求的首 token 延迟。")
        # 构造约 12000 字的中文提示词（中文约 0.7~1 token/字，≈ 10K~16K token）
        long_prompt = (filler * 250)[:12000] + "\n请只回答：收到。"
        ttft_l, window_l, n_l, total_l = stream_bench(
            client, args.model,
            [{"role": "user", "content": long_prompt}], 32)
        # 重复一次：命中前缀缓存后的 TTFT（agent 共享前缀场景的真实收益）
        ttft_l2, _, _, _ = stream_bench(
            client, args.model,
            [{"role": "user", "content": long_prompt}], 32)
        print(f"  提示词长度={len(long_prompt)} 字 | 首轮 TTFT={ttft_l:.2f}s | "
              f"第二轮（前缀缓存命中）TTFT={ttft_l2:.2f}s")

    else:
        print("\n[2/3][3/3] 已跳过（--skip-long）")

    print("\n" + "=" * 56)
    print(f"结果汇总：")
    print(f"  单流 decode 速度（{args.max_tokens} token 短输出）中位数: {median:.1f} tok/s")
    if not args.skip_long:
        print(f"  长生成（2048 token）稳态速度: {long_speed:.1f} tok/s")
    verdict = "PASS" if median >= args.target else "FAIL"
    print(f"判定: {verdict}（目标 >= {args.target:.0f} tok/s）")
    sys.exit(0 if median >= args.target else 1)


if __name__ == "__main__":
    main()
