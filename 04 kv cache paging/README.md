# Day 4 — KV cache paging

Why you can't just batch forever: after weights (~2.2 GB in 4-bit), the **KV cache**
is the second biggest consumer of VRAM. Weights are a one-time cost; the KV cache
is a running cost that scales with context length × concurrency — the capacity
ceiling behind Day 3's throughput curve.

## Size it from the config (`kv_math.py`)

Per-token KV (Qwen2.5-3B, from `config.json`):

```
2 × num_hidden_layers × kv_heads × head_dim × bytes
  = 2 × 36 × 2 × 128 × 2
  = 36,864 bytes ≈ 36 KB/token
```

`head_dim = hidden_size / num_attention_heads = 2048 / 16 = 128`. Bytes come from
bf16 (`torch_dtype`) → 2 bytes.

**GQA:** 16 query heads share only **2** key/value heads (`num_key_value_heads`),
so the cache stores 2, not 16 — **8× smaller** than full MHA would be.

At `max-model-len 2048`:

| | |
|---|---|
| KV per sequence | ~36 KB × 2048 ≈ **74 MB/seq** |
| Free VRAM after weights | ~2.5 GB |
| Sequences that fit | 2.5 GB / 74 MB ≈ **~33** (order-of-30 upper bound) |

That capacity limit is why Day-3 batching gains shrank as concurrency climbed —
not bandwidth, but room to hold KV caches.

```bash
python kv_math.py
```

## PagedAttention

Old serving reserved one big KV slot per request sized for the longest possible
answer; short outputs left most of that slot empty (60–80% waste).

**PagedAttention** cuts the cache into small fixed-size blocks and hands them out
on demand. Waste drops under ~4%. Because those blocks are shareable, requests
with the same opening text can point at the same blocks — that is what makes
**prefix caching** possible.

## Prefix caching (`prefix_test.py`)

Ten requests share one long system prompt (~1k tokens) and ask short questions.

| | Total time |
|---|---|
| Prefix caching ON | **1.33 s** |
| Prefix caching OFF | **3.23 s** |
| Saved | **~59%** |

Favorable case (long shared prefix, short questions) — not a universal number.
Needs the AWQ server running (vLLM enables prefix caching by default on this path).

```bash
# server up with the usual serve flags
python prefix_test.py
```

## Lever: `--max-model-len`

Halving context (2048 → 1024) halves KV per sequence (~38 MB) → roughly
**~66 sequences** in the same ~2.5 GB. Trade-off: each request holds half the
prompt+output tokens. Peak `nvidia-smi` VRAM looks similar either way because
`--gpu-memory-utilization 0.80` reserves the pool up front — the lever changes
how many sequences that pool is carved into, not the pool size.
