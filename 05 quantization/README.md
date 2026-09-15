# Day 5 — Quantization (4-bit AWQ)

We've been serving 4-bit all along; this day measures what that buys and what it
means. Decode is memory-bound (Day 1), so cutting bytes read per token lifts the
roofline.

## Measured on the RTX 3060 (`quant_report.py`)

| Metric | Value |
|---|---|
| Decode speed | **114.5 tok/s** |
| vs Day-1 4-bit ceiling (~177) | **65%** |
| `nvidia-smi` VRAM in use | **5945 MiB** |

That VRAM figure is weights (~**2.2 GB**) plus the KV-cache pool reserved by
`--gpu-memory-utilization 0.80` — not the model alone. The ~2.2 GB weights are
what leave KV headroom on a 6 GB card.

```bash
# server already up
python quant_report.py
```

## What "4-bit weight-only, group size 128" means

Weights normally live in high precision (FP16 = 16 bits per weight). **4-bit
weight-only** stores each weight in 4 bits — roughly a quarter the size — by
rounding. To avoid one coarse ruler for the whole tensor, quantization works in
small **groups** (commonly 128 weights). Each group gets its own scale so
important structure survives.

Decode already reads the full weight set per token. At 4 bits you move about
**one-third the bytes** per token versus FP16, which is why the memory-wall
ceiling rises.

## GPTQ vs AWQ

| | Approach |
|---|---|
| **GPTQ** | Quantize layer by layer; after rounding, nudge remaining weights to absorb error |
| **AWQ** | Use activation magnitudes to find important weights, scale them up before quantizing so they stay near-exact |

AWQ suits this laptop path: strong accuracy retention for transformers at 4-bit,
and `awq_marlin` is what vLLM served successfully on the 3060.

## Accuracy cost is small but not uniform

Aggregate scores usually hold. Specific, rarer behaviours can still shift — how
the model refuses unsafe requests, edge cases, fairness across groups. That
unevenness is the hook for later eval days: predict refusal consistency / group
fairness may move even when aggregate accuracy barely does.

---

## Day 6 — FP16 vs 4-bit before/after

Same model family, different precision and hardware: FP16 on a Colab **T4**,
4-bit AWQ on the laptop **3060**. Cross-GPU throughput is **illustrative**, not
a controlled like-for-like benchmark — the weight-size comparison is the clean
before/after.

### FP16 baseline (`colab_fp16_benchmark.py`)

Colab notebook twin: `colab_fp16_benchmark.ipynb`. Results:
`results_fp16_t4.json`.

| Config | Weights (approx.) | Batched tok/s | Notes |
|---|---|---|---|
| FP16 (T4) | **~5.8 GB** | **479.7** | full-precision serve |
| AWQ 4-bit (3060) | **~2.2 GB** | 116.2 @ c=1 / 1968 @ peak (Day 3) | laptop path |

Weight footprint shrinks **~2.6×** (5.8 → 2.2 GB). That is what makes the 3B
model fit a 6 GB card with KV headroom.

```bash
# print the before/after table (needs Day-3 sweep JSON next door)
python comparison_table.py
```

Do not read the tok/s columns as “4-bit is slower/faster than FP16” — different
GPUs, different batching setups. Use them as order-of-magnitude context; use the
GB column for the quantization claim.
