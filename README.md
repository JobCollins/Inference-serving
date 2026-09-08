# Working Setup — vLLM + Qwen2.5-3B-AWQ on an RTX 3060 Laptop (WSL2)

The exact path that got me serving, in the order that worked, including the two
errors I hit and the fixes that cleared them. Every value here is what actually
worked on my machine — adjust only if yours differs.

## My machine

- Lenovo Legion 5i, RTX 3060 **Laptop** GPU — **6 GB** VRAM (not the 12 GB desktop card)
- Windows host, NVIDIA driver 581.29, CUDA 13.0 exposed to WSL
- WSL2 running Ubuntu; project lives in the WSL filesystem at `~/Inference-serving`
- Model: `Qwen/Qwen2.5-3B-Instruct-AWQ` (4-bit), served with vLLM

### Repo layout

| Folder | Focus |
|---|---|
| `01 memory wall/` | Roofline ceiling + quick decode smoke test |
| `02 measure what matters/` | TTFT / latency / throughput baseline harness + Day-2 results |
| `03 continuous batching/` | Concurrency sweep, knee table/plot + Day-3 results |

---

## Step 0 — Put the laptop in Performance mode *first*

This mattered more than expected. On battery / Balanced mode the GPU was capped
at **50 W**, which throttles memory bandwidth and drags down tokens/sec. Fixing
it raised the cap to **126 W** (the card's real ceiling).

1. Plug in the original charger. (Lenovo blocks Performance mode on battery or an under-rated adapter.)
2. Press **Fn + Q** until the indicator shows Performance — or set it in Lenovo Vantage / Legion Toolkit.
3. Verify the cap rose. In WSL:
   ```bash
   nvidia-smi
   ```
   Look at `Pwr:Usage/Cap` — it should read `… / 126W`, not `… / 50W`.

Keep it in Performance mode only while benchmarking; Balanced for everyday editing.

---

## Step 1 — Confirm WSL2 sees the GPU

GPU passthrough must already work *before* installing anything. Inside Ubuntu:

```bash
nvidia-smi
```

This must list the RTX 3060. If it doesn't, fix the **Windows** NVIDIA driver —
never install a GPU driver inside WSL (it breaks the passthrough stub).

---

## Step 2 — Python environment + vLLM

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv ~/.venv/vllm
source ~/.venv/vllm/bin/activate
pip install --upgrade pip
pip install vllm          # ships with a matched CUDA/PyTorch build — don't hand-pin torch
pip install openai        # client used by the smoke test / benchmark
```

---

## Step 3 — The one-time fix that made vLLM start (UVA error)

Out of the box, the first serve failed with:

```
RuntimeError: UVA is not available
```

Cause: vLLM detects WSL and disables pinned memory by default, but a newer code
path needs it. Force it back on — and make it permanent so I never hit this again:

```bash
echo 'export VLLM_WSL2_ENABLE_PIN_MEMORY=1' >> ~/.bashrc
source ~/.bashrc
```

(As a bonus, this also removes vLLM's "pin_memory disabled, may be slow" penalty.)

---

## Step 4 — Serve the model (the settings that worked)

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1        # already in ~/.bashrc; harmless to repeat
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
  --quantization awq_marlin \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --dtype float16
```

Why `0.80` and not `0.90`: at `0.90` it failed with
`Free memory on device cuda:0 (5.0/6.0 GiB) … is less than desired (5.4 GiB)`
because the Windows desktop already holds ~1 GB of VRAM. `0.80` asks for ~4.8 GiB,
which fits, and still leaves plenty of KV-cache room after the ~2.2 GB of AWQ weights.

First run downloads the model (~2 GB), so give it a minute. Wait until you see:

```
INFO  Application startup complete.
INFO  Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal running — the server lives in the foreground here.

---

## Step 5 — Confirm it's actually up

In a **second** terminal:

```bash
curl http://localhost:8000/v1/models      # should list the model
```

If you get `curl: (7) Failed to connect to port 8000`, the server isn't ready yet
(still loading) or it crashed — check the serve terminal. Quick check for a
listener:

```bash
ss -ltnp | grep 8000                      # a line here means it's up
```

---

## Step 6 — Smoke test: confirm generation + measure decode speed

From the repo root, run the script in `01 memory wall/smoke_test.py`:

```bash
source ~/.venv/vllm/bin/activate
python "01 memory wall/smoke_test.py"
# -> 121 tokens in 1.05s -> 115.6 tok/s (measured 4-bit decode)
```

115.6 tok/s ≈ 65% of the ~177 tok/s roofline for this card — healthy for a single
unbatched stream. If it comes out far lower (~20%), suspect throttling: recheck
Step 0 (Performance mode, power cap under load with `watch -n 1 nvidia-smi`).

---

## Day 2 — Baseline (measure what matters)

Recorded against the same serve config as above (4-bit AWQ, RTX 3060 Laptop,
Performance mode). Harness: `02 measure what matters/day1_baseline_benchmark.py`.
Raw numbers: `results_day2_baseline.json`, VRAM samples: `vram_day2.log`.

| Metric | Value |
|---|---|
| Median TTFT | 0.019 s (19 ms) |
| p95 TTFT | 0.033 s (33 ms) |
| Median end-to-end latency | 1.015 s |
| Throughput @ concurrency 1 | 118.8 tok/s |
| Peak VRAM | 5961 MiB (~5.8 GB) |

Config: `Qwen2.5-3B-Instruct-AWQ` (4-bit), RTX 3060 Laptop 6 GB, WSL2, vLLM,
Performance mode.

I report the p95 alongside the median because a production SLA is written on
the tail (p95 or p99) — the worst cases real users hit — and not the mean,
which a few outliers can quietly distort. The median gives the typical case
for serving; the p95 guards the promise.

At 118.8 tok/s the single-stream decode throughput is **67% of the Day-1
~177 tok/s roofline**, which is within the healthy range for a single
unbatched stream.

Peak VRAM sits at 5961 of 6144 MiB not because the model is large (the 4-bit
weights are only ~2.2 GB) but because `--gpu-memory-utilization 0.80` tells
vLLM to pre-reserve ~80% of the card for the KV cache at startup — so this
figure is the ceiling I set, not the model's live footprint.

Re-run the concurrency-1 baseline (with the server already up):

```bash
source ~/.venv/vllm/bin/activate
python "02 measure what matters/day1_baseline_benchmark.py" \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-3B-Instruct-AWQ \
  --concurrency 1 \
  --max-tokens 128 \
  --requests 20 \
  --out "02 measure what matters/results_day2_baseline.json"
```

---

## Gotchas I actually hit (and the fix)

| Symptom | Fix |
|---|---|
| GPU capped at 50 W, low tok/s | Plug in + Performance mode (Fn+Q) → cap rises to 126 W |
| `Free memory … less than desired GPU memory utilization` | Lower `--gpu-memory-utilization` to `0.80` (or `0.75`); close browser windows |
| `RuntimeError: UVA is not available` | `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` (put it in `~/.bashrc`) |
| `curl: (7) Failed to connect to port 8000` | Server not up yet — wait for the `Uvicorn running` line; verify with `ss -ltnp \| grep 8000` |
| Code from a Word doc throws a syntax error | Smart quotes — retype `"` as straight quotes, or paste into a plain-text/code editor |
| Model loads slowly / stalls | Keep the project in the WSL filesystem (`~/…`), not `/mnt/c/…` (3–5× slower) |

---

## Next-time quick start

Once the environment exists, getting running again is just:

```bash
source ~/.venv/vllm/bin/activate
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
  --quantization awq_marlin --max-model-len 2048 \
  --gpu-memory-utilization 0.80 --dtype float16
# (VLLM_WSL2_ENABLE_PIN_MEMORY is already exported via ~/.bashrc)
```

Then `curl http://localhost:8000/v1/models` to confirm, and you're serving.
Remember to flip to Performance mode before any benchmarking.
