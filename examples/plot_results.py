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
    parser.add_argument(
        "--trace",
        default="examples/outputs/universal_mixture_trace.csv",
        help="Path to trace CSV",
    )
    parser.add_argument(
        "--sweep",
        default="examples/outputs/phase_sweep.csv",
        help="Path to sweep CSV",
    )
    parser.add_argument(
        "--trace-out",
        default="examples/outputs/universal_mixture_trace.png",
        help="Output path for trace plot",
    )
    parser.add_argument(
        "--sweep-out",
        default="examples/outputs/phase_sweep.png",
        help="Output path for sweep plot",
    )
    parser.add_argument(
        "--adopt-out",
        default="examples/outputs/phase_adoption.png",
        help="Output path for adoption scatter plot",
    )
    args = parser.parse_args()

    plot_trace(args.trace, args.trace_out)
    plot_sweep(args.sweep, args.sweep_out)
    plot_adoption(args.sweep, args.adopt_out)

    print(f"Wrote {args.trace_out}")
    print(f"Wrote {args.sweep_out}")
    print(f"Wrote {args.adopt_out}")


if __name__ == "__main__":
    main()
