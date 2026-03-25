#!/usr/bin/env python3
"""Plot MC-AIXI competitor benchmark results.

This script reads benchmark outputs produced by
scripts/bench_aixi_competitors_runner.py and creates:

1) trial_metrics.png
   - RSS across implementations and trials
   - Speed (cycles/s) across implementations and trials
   - Reward across implementations and trials

2) scaling_comparison.png
   - RSS scaling vs workload scale
   - Speed scaling vs workload scale

Workload scale is defined as:
    horizon * num_simulations * eval_cycles
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RAW_INT_COLS = {
    "trial",
    "ct_depth",
    "horizon",
    "num_simulations",
    "eval_cycles",
    "seed",
    "rayon_threads",
    "steps",
}

RAW_FLOAT_COLS = {
    "wall_s",
    "rss_kb",
    "reward_avg",
    "reward_total",
    "cycles_per_sec",
}

SUMMARY_INT_COLS = {
    "horizon",
    "num_simulations",
    "eval_cycles",
    "n",
}

SUMMARY_FLOAT_COLS = {
    "wall_mean_s",
    "wall_std_s",
    "rss_mean_kb",
    "rss_std_kb",
    "reward_mean",
    "reward_std",
    "cycles_per_sec_mean",
    "cycles_per_sec_std",
}

DEFAULT_IMPL_ORDER = [
    "infotheory-rust",
    "infotheory-python",
    "mcaixi-cpp",
    "pyaixi",
    "infotheory-rust-fac-ctw",
    "infotheory-python-fac-ctw",
]

COLOR_MAP = {
    "infotheory-rust": "#1f78b4",
    "infotheory-python": "#33a02c",
    "mcaixi-cpp": "#ff7f00",
    "pyaixi": "#e31a1c",
    "infotheory-rust-fac-ctw": "#6a3d9a",
    "infotheory-python-fac-ctw": "#b15928",
}


def parse_number(value: str, *, as_int: bool) -> float | int:
    text = (value or "").strip()
    if as_int:
        return int(text)
    return float(text)


def read_tsv(path: Path, int_cols: set[str], float_cols: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                if key in int_cols:
                    row[key] = parse_number(value or "0", as_int=True)
                elif key in float_cols:
                    row[key] = parse_number(value or "nan", as_int=False)
                else:
                    row[key] = (value or "").strip()
            rows.append(row)
    return rows


def find_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = [
        p
        for p in base_dir.iterdir()
        if p.is_dir() and (p / "raw.tsv").is_file() and (p / "summary.tsv").is_file()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark runs found under: {base_dir}")
    return sorted(run_dirs, key=lambda p: p.name)[-1]


def filter_by_algorithm(
    raw_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    algorithm: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if algorithm == "all":
        return raw_rows, summary_rows
    raw_filtered = [r for r in raw_rows if r.get("algorithm") == algorithm]
    summary_filtered = [r for r in summary_rows if r.get("algorithm") == algorithm]
    return raw_filtered, summary_filtered


def implementation_order(rows: list[dict[str, object]]) -> list[str]:
    impls = sorted({str(r["impl"]) for r in rows})
    order_map = {name: idx for idx, name in enumerate(DEFAULT_IMPL_ORDER)}
    return sorted(impls, key=lambda name: (order_map.get(name, 999), name))


def implementation_color(impl: str, color_index: int) -> str:
    if impl in COLOR_MAP:
        return COLOR_MAP[impl]
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return palette[color_index % len(palette)]


def metric_label(metric_key: str) -> str:
    labels = {
        "rss_kb": "RSS (KB)",
        "cycles_per_sec": "Speed (cycles/s)",
        "reward_avg": "Average Reward",
    }
    return labels.get(metric_key, metric_key)


def build_trial_stats(
    rows: list[dict[str, object]],
    metric_key: str,
) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        impl = str(row["impl"])
        trial = int(row["trial"])
        value = float(row[metric_key])
        if math.isfinite(value):
            grouped[(impl, trial)].append(value)
    return grouped


def plot_trial_metrics(
    raw_rows: list[dict[str, object]],
    out_path: Path,
    algorithm_filter: str,
    run_name: str,
    dpi: int,
) -> None:
    if not raw_rows:
        raise ValueError("No raw rows available for trial plot")

    trials = sorted({int(r["trial"]) for r in raw_rows})
    impls = implementation_order(raw_rows)

    metrics = ["rss_kb", "cycles_per_sec", "reward_avg"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis_idx, metric in enumerate(metrics):
        ax = axes[axis_idx]
        grouped = build_trial_stats(raw_rows, metric)

        for impl_idx, impl in enumerate(impls):
            color = implementation_color(impl, impl_idx)
            x_vals: list[float] = []
            means: list[float] = []
            stds: list[float] = []

            for trial in trials:
                vals = grouped.get((impl, trial), [])
                if not vals:
                    continue
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                x_vals.append(float(trial))
                means.append(mean)
                stds.append(std)

                offset = (impl_idx - (len(impls) - 1) / 2.0) * 0.04
                x_jitter = [trial + offset] * len(vals)
                ax.scatter(
                    x_jitter,
                    vals,
                    color=color,
                    s=18,
                    alpha=0.25,
                    linewidths=0,
                )

            if not x_vals:
                continue

            ax.plot(
                x_vals,
                means,
                marker="o",
                markersize=5,
                linewidth=2.2,
                color=color,
                label=impl,
                zorder=3,
            )
            if any(s > 0 for s in stds):
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                ax.fill_between(x_vals, lower, upper, color=color, alpha=0.12)

        ax.set_ylabel(metric_label(metric))
        ax.set_title(metric_label(metric), loc="left", fontsize=11, fontweight="semibold")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Trial")
    axes[-1].set_xticks(trials)

    title_algo = "all algorithms" if algorithm_filter == "all" else algorithm_filter
    fig.suptitle(
        f"MC-AIXI Competitor Metrics by Trial ({title_algo})\n{run_name}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, max(1, len(labels))),
            frameon=True,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.965),
        )

    fig.text(
        0.5,
        0.01,
        "Solid lines: per-trial mean across scenarios. Faint points: individual scenario runs.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def workload_scale(row: dict[str, object]) -> float:
    return (
        float(row["horizon"])
        * float(row["num_simulations"])
        * float(row["eval_cycles"])
    )


def plot_scaling_comparison(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    algorithm_filter: str,
    run_name: str,
    dpi: int,
) -> None:
    if not summary_rows:
        raise ValueError("No summary rows available for scaling plot")

    impls = implementation_order(summary_rows)
    workloads = sorted({str(r["workload"]) for r in summary_rows})
    workload_markers = {
        workload: marker
        for workload, marker in zip(workloads, ["o", "s", "^", "D", "P", "X"])
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    panels = [
        (axes[0], "rss_mean_kb", "RSS Scaling (KB)"),
        (axes[1], "cycles_per_sec_mean", "Speed Scaling (cycles/s)"),
    ]

    for ax, metric_key, title in panels:
        for impl_idx, impl in enumerate(impls):
            color = implementation_color(impl, impl_idx)
            impl_rows = [r for r in summary_rows if str(r["impl"]) == impl]
            for workload in workloads:
                subset = [
                    r for r in impl_rows if str(r["workload"]) == workload
                ]
                if not subset:
                    continue
                subset = sorted(subset, key=workload_scale)
                x_vals = [workload_scale(r) for r in subset]
                y_vals = [float(r[metric_key]) for r in subset]
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    marker=workload_markers[workload],
                    linewidth=1.8,
                    markersize=5,
                    alpha=0.9,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Workload Scale (horizon * simulations * eval_cycles)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.grid(True, which="both", alpha=0.25)

    impl_handles = [
        Line2D([0], [0], color=implementation_color(impl, i), lw=2.2, label=impl)
        for i, impl in enumerate(impls)
    ]
    workload_handles = [
        Line2D([0], [0], color="#555555", marker=workload_markers[w], lw=0, ms=7, label=w)
        for w in workloads
    ]

    axes[0].legend(handles=impl_handles, title="Implementation", fontsize=8, title_fontsize=9)
    axes[1].legend(handles=workload_handles, title="Workload", fontsize=8, title_fontsize=9)

    title_algo = "all algorithms" if algorithm_filter == "all" else algorithm_filter
    fig.suptitle(
        f"MC-AIXI Scaling Comparison ({title_algo})\n{run_name}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MC-AIXI competitor benchmark outputs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Benchmark run directory containing raw.tsv and summary.tsv. Defaults to latest under target/aixi-competitors.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["all", "ac-ctw", "fac-ctw"],
        default="all",
        help="Filter rows by algorithm before plotting.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plot images. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_run_dir(Path("target") / "aixi-competitors")
    else:
        run_dir = args.run_dir

    raw_tsv = run_dir / "raw.tsv"
    summary_tsv = run_dir / "summary.tsv"

    if not raw_tsv.is_file() or not summary_tsv.is_file():
        raise FileNotFoundError(
            f"Expected raw.tsv and summary.tsv in run directory: {run_dir}"
        )

    out_dir = args.out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_tsv(raw_tsv, RAW_INT_COLS, RAW_FLOAT_COLS)
    summary_rows = read_tsv(summary_tsv, SUMMARY_INT_COLS, SUMMARY_FLOAT_COLS)

    raw_rows, summary_rows = filter_by_algorithm(raw_rows, summary_rows, args.algorithm)
    if not raw_rows:
        raise ValueError(f"No raw rows found after algorithm filter: {args.algorithm}")
    if not summary_rows:
        raise ValueError(f"No summary rows found after algorithm filter: {args.algorithm}")

    trial_plot = out_dir / "trial_metrics.png"
    scaling_plot = out_dir / "scaling_comparison.png"

    plot_trial_metrics(
        raw_rows=raw_rows,
        out_path=trial_plot,
        algorithm_filter=args.algorithm,
        run_name=run_dir.name,
        dpi=args.dpi,
    )
    plot_scaling_comparison(
        summary_rows=summary_rows,
        out_path=scaling_plot,
        algorithm_filter=args.algorithm,
        run_name=run_dir.name,
        dpi=args.dpi,
    )

    print(f"Run directory: {run_dir}")
    print(f"Wrote: {trial_plot}")
    print(f"Wrote: {scaling_plot}")


if __name__ == "__main__":
    main()
