import json, sys

data = json.load(open(sys.argv[1]))
print(f"{'conc':>5} {'tok/s':>8} {'gain':>7} {'med_lat_s':>10} {'p95_ttft_s':>11}")
prev = None
rows = []

for r in data["runs"]:
    if "throughput_tok_s" not in r: continue
    tp = r["throughput_tok_s"]
    gain = f"+{(tp/prev-1)*100:.0f}%" if prev else "-"
    print(f"{r['concurrency']:>5} {tp:>8} {gain:>7} {r['median_latency_s']:>10} {r['p95_ttft_s']:>11}")
    prev = tp
    rows.append(r)


def plot_sweep(rows, out="sweep.png", knee=8):
    """Throughput vs median latency across concurrency, dual y-axis."""
    import matplotlib
    matplotlib.use("Agg")                      # headless (WSL) — write a file, don't try to display
    import matplotlib.pyplot as plt

    conc = [r["concurrency"] for r in rows]
    tput = [r["throughput_tok_s"] for r in rows]
    lat  = [r["median_latency_s"] for r in rows]

    TP_C, LAT_C = "#0E8F6E", "#B5541E"         # green = throughput, amber = latency
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # left axis — throughput
    ax1.plot(conc, tput, "-o", color=TP_C, linewidth=2, label="throughput")
    ax1.set_xlabel("concurrency (requests in flight)")
    ax1.set_ylabel("throughput (tok/s)", color=TP_C)
    ax1.tick_params(axis="y", labelcolor=TP_C)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(conc); ax1.set_xticklabels(conc)   # show real concurrency values, not 2^n
    ax1.set_ylim(0, max(tput) * 1.12)
    for x, y in zip(conc, tput):                       # label each throughput point
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=TP_C)

    # right axis — median latency
    ax2 = ax1.twinx()
    ax2.plot(conc, lat, "--s", color=LAT_C, linewidth=2, label="median latency")
    ax2.set_ylabel("median end-to-end latency (s)", color=LAT_C)
    ax2.tick_params(axis="y", labelcolor=LAT_C)
    ax2.set_ylim(0, max(lat) * 1.25)

    # mark the interactive knee
    if knee in conc:
        ax1.axvline(knee, color="#6B7680", linestyle=":", linewidth=1)
        ax1.annotate("interactive knee",
                     (knee, max(tput) * 1.02), ha="center", fontsize=8, color="#6B7680")

    # one combined legend
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=9, frameon=False)

    plt.title("Continuous batching: throughput rises, latency follows", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


plot_sweep(rows)