import json
fp16 = json.load(open("results_fp16_t4.json"))
sweep = json.load(open("../03 continuous batching/results_day3_sweep.json"))
runs = sweep["runs"]
c1   = next(r for r in runs if r["concurrency"] == 1)
knee = max(runs, key=lambda r: r.get("throughput_tok_s", 0))

print(f"{'config':16s}{'VRAM':>10s}{'tok/s c=1':>11s}{'tok/s knee':>12s}")
print(f"{'FP16 (T4)':16s}{fp16['peak_vram_gb']:>8}GB{'—':>11s}{fp16['throughput_tok_s']:>12}")
print(f"{'AWQ 4-bit (3060)':16s}{'~2.2GB':>10s}"
      f"{c1['throughput_tok_s']:>11}{knee['throughput_tok_s']:>12}")