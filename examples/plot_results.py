#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def col_idx(header, prefix):
    return [i for i, name in enumerate(header) if name.startswith(prefix)]


def parse_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def plot_trace(trace_path, out_path):
    header, rows = read_csv(trace_path)
    cols = {name: idx for idx, name in enumerate(header)}

    t = np.array([int(r[cols["t"]]) for r in rows])
    seg_idx = np.array([int(r[cols["seg_idx"]]) for r in rows])
    segments = [r[cols["segment"]] for r in rows]

    bayes_ema = np.array([parse_float(r[cols["bayes_ema"]]) for r in rows])
    switch_ema = np.array([parse_float(r[cols["switch_ema"]]) for r in rows])
    fading_ema = np.array([parse_float(r[cols["fading_ema"]]) for r in rows])
    mdl_ema = np.array([parse_float(r[cols["mdl_ema"]]) for r in rows])

    bayes_neff = np.array([parse_float(r[cols["bayes_neff"]]) for r in rows])
    switch_neff = np.array([parse_float(r[cols["switch_neff"]]) for r in rows])
    fading_neff = np.array([parse_float(r[cols["fading_neff"]]) for r in rows])

    bayes_entropy = np.array(
        [parse_float(r[cols["bayes_entropy_bits"]]) for r in rows]
    )
    switch_entropy = np.array(
        [parse_float(r[cols["switch_entropy_bits"]]) for r in rows]
    )
    fading_entropy = np.array(
        [parse_float(r[cols["fading_entropy_bits"]]) for r in rows]
    )

    bayes_w_cols = col_idx(header, "bayes_w_")
    switch_w_cols = col_idx(header, "switch_w_")
    fading_w_cols = col_idx(header, "fading_w_")
    exp_ema_cols = col_idx(header, "exp_ema_")

    expert_names = [header[i].replace("bayes_w_", "") for i in bayes_w_cols]

    bayes_w = np.array([[parse_float(r[i]) for i in bayes_w_cols] for r in rows])
    switch_w = np.array([[parse_float(r[i]) for i in switch_w_cols] for r in rows])
    fading_w = np.array([[parse_float(r[i]) for i in fading_w_cols] for r in rows])
    exp_ema = np.array([[parse_float(r[i]) for i in exp_ema_cols] for r in rows])

    best_exp_ema = np.nanmin(exp_ema, axis=1)

    boundaries = []
    labels = []
    last = seg_idx[0]
    start_t = t[0]
    for i in range(1, len(seg_idx)):
        if seg_idx[i] != last:
            boundaries.append(t[i])
            labels.append(segments[i])
            last = seg_idx[i]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, bayes_ema, label="Bayes EMA", lw=1.6)
    ax.plot(t, switch_ema, label="Switch EMA", lw=1.6)
    ax.plot(t, fading_ema, label="Fading EMA", lw=1.2)
    ax.plot(t, mdl_ema, label="MDL EMA", lw=1.2)
    ax.plot(t, best_exp_ema, label="Best expert EMA", lw=1.2, ls="--")
    ax.set_ylabel("bits/symbol (EMA)")
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1]
    for i, name in enumerate(expert_names):
        ax.plot(t, bayes_w[:, i], lw=0.9, label=name)
    ax.set_ylabel("Bayes posterior")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[2]
    for i, name in enumerate(expert_names):
        ax.plot(t, switch_w[:, i], lw=0.9, label=name)
    ax.set_ylabel("Switch posterior")

    ax = axes[3]
    ax.plot(t, bayes_neff, label="Neff Bayes", lw=1.2)
    ax.plot(t, switch_neff, label="Neff Switch", lw=1.2)
    ax.plot(t, fading_neff, label="Neff Fading", lw=1.2)
    ax.plot(t, bayes_entropy, label="H Bayes", lw=1.0, ls="--")
    ax.plot(t, switch_entropy, label="H Switch", lw=1.0, ls="--")
    ax.plot(t, fading_entropy, label="H Fading", lw=1.0, ls="--")
    ax.set_ylabel("Neff / entropy")
    ax.set_xlabel("time")
    ax.legend(ncol=3, fontsize=8)

    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="k", lw=0.6, alpha=0.3)

    if boundaries:
        for b, label in zip(boundaries, labels):
            axes[0].text(b, axes[0].get_ylim()[1], label, rotation=90, va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_sweep(sweep_path, out_path):
    header, rows = read_csv(sweep_path)
    cols = {name: idx for idx, name in enumerate(header)}

    data = []
    for r in rows:
        seg_len = int(r[cols["seg_len"]])
        alpha = float(r[cols["alpha"]])
        gap = r[cols["gap"]]
        bayes_collapse = parse_float(r[cols["bayes_collapse"]])
        switch_excess_v = parse_float(r[cols["switch_excess_viterbi_bits"]])
        data.append((seg_len, alpha, gap, bayes_collapse, switch_excess_v))

    seg_lens = sorted({d[0] for d in data})
    alphas = sorted({d[1] for d in data})
    gaps = sorted({d[2] for d in data})

    seg_index = {v: i for i, v in enumerate(seg_lens)}
    alpha_index = {v: i for i, v in enumerate(alphas)}

    fig, axes = plt.subplots(len(gaps), 2, figsize=(12, 4 * len(gaps)))
    if len(gaps) == 1:
        axes = np.array([axes])

    for g_i, gap in enumerate(gaps):
        collapse_grid = np.full((len(seg_lens), len(alphas)), np.nan)
        viterbi_grid = np.full((len(seg_lens), len(alphas)), np.nan)
        for seg_len, alpha, g, collapse, viterbi_bits in data:
            if g != gap:
                continue
            i = seg_index[seg_len]
            j = alpha_index[alpha]
            collapse_grid[i, j] = collapse / (3.0 * seg_len) if not math.isnan(collapse) else np.nan
            viterbi_grid[i, j] = viterbi_bits / (3.0 * seg_len)

        ax = axes[g_i, 0]
        im = ax.imshow(
            collapse_grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{gap}: Bayes collapse (fraction of total)")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.1e}" for a in alphas], rotation=45)
        ax.set_yticks(range(len(seg_lens)))
        ax.set_yticklabels(seg_lens)
        ax.set_xlabel("alpha")
        ax.set_ylabel("segment length")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[g_i, 1]
        im = ax.imshow(
            viterbi_grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{gap}: Switch excess vs Viterbi (bpb)")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.1e}" for a in alphas], rotation=45)
        ax.set_yticks(range(len(seg_lens)))
        ax.set_yticklabels(seg_lens)
        ax.set_xlabel("alpha")
        ax.set_ylabel("segment length")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_cost_pareto(pareto_path, out_path):
    header, rows = read_csv(pareto_path)
    cols = {name: idx for idx, name in enumerate(header)}

    lambdas = np.array([parse_float(r[cols["lambda"]]) for r in rows])
    total_bits = np.array([parse_float(r[cols["total_bits"]]) for r in rows])
    bps = np.array([parse_float(r[cols["bps"]]) for r in rows])
    expected_time_ns = np.array(
        [parse_float(r[cols["total_expected_time_ns"]]) for r in rows]
    )
    expected_time_per_sym = np.array(
        [parse_float(r[cols["expected_time_per_symbol_ns"]]) for r in rows]
    )

    expert_bits_cols = [i for i, name in enumerate(header) if name.startswith("bits_")]
    expert_time_cols = [i for i, name in enumerate(header) if name.startswith("time_ns_")]
    expert_names = [header[i].replace("bits_", "") for i in expert_bits_cols]

    expert_bits = np.array(
        [[parse_float(r[i]) for i in expert_bits_cols] for r in rows]
    )
    expert_times = np.array(
        [[parse_float(r[i]) for i in expert_time_cols] for r in rows]
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(
        expected_time_ns / 1e6,
        total_bits,
        c=np.log10(lambdas + 1e-15),
        cmap="viridis",
        s=60,
    )
    ax.plot(expected_time_ns / 1e6, total_bits, "k-", alpha=0.3, lw=1)
    ax.set_xlabel("Total Expected Time (ms)")
    ax.set_ylabel("Total Bits")
    ax.set_title("Pareto Frontier: Bits vs Expected Time")

    for i in range(0, len(rows), max(1, len(rows) // 5)):
        ax.annotate(
            f"λ={lambdas[i]:.1e}",
            (expected_time_ns[i] / 1e6, total_bits[i]),
            fontsize=7,
            alpha=0.7,
        )

    ax = axes[0, 1]
    ax.scatter(
        expected_time_per_sym / 1e3,
        bps,
        c=np.log10(lambdas + 1e-15),
        cmap="viridis",
        s=60,
    )
    ax.plot(expected_time_per_sym / 1e3, bps, "k-", alpha=0.3, lw=1)
    ax.set_xlabel("Expected Time per Symbol (μs)")
    ax.set_ylabel("Bits per Symbol")
    ax.set_title("Rate-Distortion: BPS vs Time/Symbol")

    ax = axes[1, 0]
    for i, name in enumerate(expert_names):
        ax.plot(lambdas, expert_bits[:, i], label=name, lw=1.2)
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_xlabel("λ (nats/ns)")
    ax.set_ylabel("Expert Bits")
    ax.set_title("Expert Cumulative Bits vs λ")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    for i, name in enumerate(expert_names):
        ax.plot(lambdas, expert_times[:, i] / 1e6, label=name, lw=1.2)
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_xlabel("λ (nats/ns)")
    ax.set_ylabel("Expert Time (ms)")
    ax.set_title("Expert Cumulative Time vs λ")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_cost_trace(trace_path, out_path):
    header, rows = read_csv(trace_path)
    cols = {name: idx for idx, name in enumerate(header)}

    t = np.array([int(r[cols["t"]]) for r in rows])
    bits_plain = np.array([parse_float(r[cols["bits_plain"]]) for r in rows])
    bits_aug = np.array([parse_float(r[cols["bits_aug"]]) for r in rows])
    expected_time = np.array([parse_float(r[cols["expected_time_ns"]]) for r in rows])
    cum_bits = np.array([parse_float(r[cols["cum_bits"]]) for r in rows])
    cum_expected_time = np.array(
        [parse_float(r[cols["cum_expected_time_ns"]]) for r in rows]
    )

    w_cols = [i for i, name in enumerate(header) if name.startswith("w_")]
    expert_names = [header[i].replace("w_", "") for i in w_cols]
    weights = np.array([[parse_float(r[i]) for i in w_cols] for r in rows])

    ema_alpha = 0.02
    ema_bits = np.zeros_like(bits_plain)
    ema_bits[0] = bits_plain[0]
    for i in range(1, len(bits_plain)):
        ema_bits[i] = (1 - ema_alpha) * ema_bits[i - 1] + ema_alpha * bits_plain[i]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, ema_bits, lw=1.2, label="Bits/symbol (EMA)")
    ax.set_ylabel("Bits/symbol (EMA)")
    ax.set_title("Cost-Aware Mixture Trace")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(t, expected_time / 1e3, lw=1.0, color="orange")
    ax.set_ylabel("Expected Time (μs)")

    ax = axes[2]
    for i, name in enumerate(expert_names):
        ax.plot(t, weights[:, i], label=name, lw=0.8)
    ax.set_ylabel("Expert Weights")
    ax.legend(ncol=3, fontsize=7, loc="upper right")

    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(t, cum_bits, "b-", lw=1.2, label="Cum. Bits")
    ax2.plot(t, cum_expected_time / 1e6, "r-", lw=1.2, label="Cum. Time (ms)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative Bits", color="b")
    ax2.set_ylabel("Cumulative Expected Time (ms)", color="r")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_adoption(sweep_path, out_path):
    header, rows = read_csv(sweep_path)
    cols = {name: idx for idx, name in enumerate(header)}

    def collect_pairs(prefix, seg):
        obs_col = f"{prefix}_oracle_adopt_seg{seg}"
        pred_col = f"{prefix}_predicted_time_seg{seg}"
        gap_col = "gap"
        if obs_col not in cols or pred_col not in cols:
            return []
        pairs = []
        for r in rows:
            gap = r[cols[gap_col]]
            obs = parse_float(r[cols[obs_col]])
            pred = parse_float(r[cols[pred_col]])
            if not math.isfinite(obs) or not math.isfinite(pred):
                continue
            if obs <= 0 or pred <= 0:
                continue
            pairs.append((obs, pred, gap, seg))
        return pairs

    mixtures = [
        ("bayes", "Bayes"),
        ("switch", "Switch"),
        ("fading", "Fading"),
    ]
    gaps = sorted({r[cols["gap"]] for r in rows})
    gap_colors = {g: c for g, c in zip(gaps, ["#1f77b4", "#ff7f0e", "#2ca02c"])}

    fig, axes = plt.subplots(1, len(mixtures), figsize=(15, 4), sharex=True, sharey=True)
    if len(mixtures) == 1:
        axes = [axes]

    for ax, (prefix, title) in zip(axes, mixtures):
        pairs = collect_pairs(prefix, 2) + collect_pairs(prefix, 3)
        if not pairs:
            ax.set_title(f"{title}: no data")
            continue
        for obs, pred, gap, seg in pairs:
            marker = "o" if seg == 2 else "^"
            ax.scatter(
                pred,
                obs,
                s=18,
                alpha=0.7,
                color=gap_colors.get(gap, "#333333"),
                marker=marker,
                label=f"{gap} seg{seg}",
            )
        min_v = min(p[0] for p in pairs + [(1.0, 1.0, "", 0)])
        max_v = max(p[1] for p in pairs + [(1.0, 1.0, "", 0)])
        lo = min_v * 0.8
        hi = max_v * 1.25
        ax.plot([lo, hi], [lo, hi], color="#444444", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("Predicted adopt time (steps)")
        ax.grid(True, which="both", ls=":", lw=0.5)

    axes[0].set_ylabel("Observed adopt time (steps)")

    handles = []
    labels = []
    for gap in gaps:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=gap_colors.get(gap, "#333333"),
                label=gap,
                markersize=6,
            )
        )
        labels.append(gap)
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            label="seg2",
            markersize=6,
            linestyle="None",
        )
    )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="k",
            label="seg3",
            markersize=6,
            linestyle="None",
        )
    )
    fig.legend(handles=handles, labels=[*labels, "seg2", "seg3"], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", help="Universal mixture trace CSV")
    parser.add_argument("--sweep", help="Universal mixture sweep CSV")
    parser.add_argument("--trace-out", help="Universal trace plot output")
    parser.add_argument("--sweep-out", help="Universal sweep plot output")
    parser.add_argument("--adopt-out", help="Universal adoption plot output")
    parser.add_argument("--cost-pareto", help="Cost-aware Pareto CSV")
    parser.add_argument("--cost-trace", help="Cost-aware trace CSV")
    parser.add_argument(
        "--cost-out",
        default="examples/outputs/cost_aware",
        help="Base output path for cost-aware plots",
    )
    args = parser.parse_args()

    ran_any = False

    if args.cost_pareto or args.cost_trace:
        base = Path(args.cost_out)
        stem = base.stem
        suffix = base.suffix or ".png"
        parent = base.parent if base.parent.as_posix() != "." else Path(".")
        if args.cost_pareto:
            out = parent / f"{stem}_pareto{suffix}"
            plot_cost_pareto(args.cost_pareto, str(out))
            print(f"Wrote {out}")
            ran_any = True
        if args.cost_trace:
            out = parent / f"{stem}_trace{suffix}"
            plot_cost_trace(args.cost_trace, str(out))
            print(f"Wrote {out}")
            ran_any = True

    if args.trace or args.sweep or args.adopt_out:
        trace = args.trace or "examples/outputs/universal_mixture_trace.csv"
        sweep = args.sweep or "examples/outputs/phase_sweep.csv"
        trace_out = args.trace_out or "examples/outputs/universal_mixture_trace.png"
        sweep_out = args.sweep_out or "examples/outputs/phase_sweep.png"
        adopt_out = args.adopt_out or "examples/outputs/phase_adoption.png"
        plot_trace(trace, trace_out)
        plot_sweep(sweep, sweep_out)
        plot_adoption(sweep, adopt_out)
        print(f"Wrote {trace_out}")
        print(f"Wrote {sweep_out}")
        print(f"Wrote {adopt_out}")
        ran_any = True

    if not ran_any:
        print("No plots requested. Use --trace/--sweep or --cost-pareto/--cost-trace.")


if __name__ == "__main__":
    main()
