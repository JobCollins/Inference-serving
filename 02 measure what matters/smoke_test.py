#!/usr/bin/env python3
"""
Day-1 baseline benchmark harness
================================
Measures TTFT (time-to-first-token), per-request latency, and aggregate
output throughput (tokens/sec) against ANY OpenAI-compatible endpoint:
vLLM, SGLang, Ollama (/v1), or llama.cpp's server.

Why it works everywhere: it only speaks the OpenAI chat-completions API,
so the exact same script benchmarks your local 4-bit model on the RTX 3060
and the FP16 baseline on Colab. The only variable you change is --model.

Setup
-----
    pip install openai

Start a server first, e.g. (local, WSL):
    vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
        --quantization awq_marlin --max-model-len 2048 \
        --gpu-memory-utilization 0.90 --dtype float16

Run
---
    python day1_baseline_benchmark.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-3B-Instruct-AWQ \
        --concurrency 1 4 8 \
        --max-tokens 128 \
        --requests 32 \
        --out results_awq_3060.json

Then re-run against the Colab FP16 endpoint with a different --out, and diff
the two JSON files for your before/after table.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip install openai")


# A fixed prompt set. Keep this identical across every run so comparisons are honest.
PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Summarise the causes of the 2008 financial crisis in five bullet points.",
    "Write a haiku about memory bandwidth.",
    "What is the difference between prefill and decode in LLM inference?",
    "Give three tips for reducing latency in a web API.",
    "Translate 'good morning, how are you?' into French, Swahili, and Japanese.",
    "Describe photosynthesis to a ten-year-old.",
    "List five common causes of GPU out-of-memory errors.",
    "What are the trade-offs between throughput and latency when batching requests?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain quantization in machine learning in two sentences.",
    "What is continuous batching and why does it help serving throughput?",
    "Give a short, balanced overview of nuclear energy.",
    "Draft a one-sentence commit message for fixing an off-by-one error.",
    "What does 'goodput' mean in the context of model serving?",
    "Name three metrics you would track for a production LLM endpoint.",
    "Explain the memory wall in modern GPUs.",
    "Write a polite email declining a meeting invitation.",
    "What is prefix caching and when does it save the most compute?",
    "Compare AWQ and GPTQ quantization in a few sentences.",
]


@dataclass
class Result:
    ttft: float          # seconds to first streamed token
    latency: float       # seconds, full request
    out_tokens: int      # completion tokens produced
    ok: bool = True


async def one_request(client, model, prompt, max_tokens) -> Result:
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out_tokens += 1  # rough live count; corrected by usage below if present
            if getattr(chunk, "usage", None):
                out_tokens = chunk.usage.completion_tokens
    except Exception as e:
        print(f"  request failed: {e}")
        latency = time.perf_counter() - t0
        return Result(ttft=latency, latency=latency, out_tokens=0, ok=False)

    latency = time.perf_counter() - t0
    if ttft is None:
        ttft = latency
    return Result(ttft=ttft, latency=latency, out_tokens=out_tokens)


async def run_level(client, model, concurrency, max_tokens, n_requests) -> dict:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []

    async def worker(p):
        async with sem:
            results.append(await one_request(client, model, p, max_tokens))

    wall0 = time.perf_counter()
    await asyncio.gather(*(worker(p) for p in prompts))
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r.ok]
    if not ok:
        return {"concurrency": concurrency, "requests": n_requests, "error": "all requests failed"}

    ttfts = sorted(r.ttft for r in ok)
    total_out = sum(r.out_tokens for r in ok)
    p95_idx = max(0, int(0.95 * len(ttfts)) - 1)

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "ok": len(ok),
        "wall_s": round(wall, 3),
        "throughput_tok_s": round(total_out / wall, 1) if wall else 0.0,
        "median_ttft_s": round(statistics.median(ttfts), 3),
        "p95_ttft_s": round(ttfts[p95_idx], 3),
        "median_latency_s": round(statistics.median(r.latency for r in ok), 3),
        "total_out_tokens": total_out,
    }


async def main():
    ap = argparse.ArgumentParser(description="Baseline benchmark for OpenAI-compatible LLM endpoints.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True, help="Served model id, e.g. Qwen/Qwen2.5-3B-Instruct-AWQ")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests", type=int, default=32, help="Requests per concurrency level")
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Model: {args.model}")
    print(f"Endpoint: {args.base_url}\n")
    print(f"{'conc':>4} {'tok/s':>8} {'med_ttft':>9} {'p95_ttft':>9} {'med_lat':>8} {'ok':>4}")
    print("-" * 48)

    rows = []
    for c in sorted(set(args.concurrency)):
        row = await run_level(client, args.model, c, args.max_tokens, args.requests)
        rows.append(row)
        if "error" in row:
            print(f"{c:>4}  {row['error']}")
        else:
            print(f"{row['concurrency']:>4} {row['throughput_tok_s']:>8} "
                  f"{row['median_ttft_s']:>9} {row['p95_ttft_s']:>9} "
                  f"{row['median_latency_s']:>8} {row['ok']:>4}")

    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests,
        "runs": rows,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
