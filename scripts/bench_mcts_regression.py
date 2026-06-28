#!/usr/bin/env python3
"""Baseline-vs-current regression gate for Tranche 3.5 Part 1 MCTS benches."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


EXPECTED_BENCHMARKS = (
    "mcts_planner_throughput/rho_uct/256",
    "mcts_planner_throughput/parallel_uct_wu_workers1/256",
    "mcts_planner_throughput/parallel_uct_wu_workers4/256",
    "mcts_planner_throughput/parallel_uct_bu_core_workers4/256",
    "mcts_tuner_shaped/rho_uct_actions64_h4/192",
    "mcts_tuner_shaped/parallel_uct_wu_actions64_h4/192",
    "mcts_tuner_shaped/parallel_uct_bu_core_actions64_h4/192",
)

TIME_LINE_RE = re.compile(
    r"time:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)\s*\]"
)
UNIT_TO_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}


@dataclass(frozen=True)
class BenchResult:
    name: str
    baseline_s: float
    current_s: float

    @property
    def delta_pct(self) -> float:
        return ((self.current_s / self.baseline_s) - 1.0) * 100.0

    @property
    def threshold_pct(self) -> float:
        if "/rho_uct" in self.name:
            return 5.0
        if "/parallel_uct_" in self.name:
            return 10.0
        raise ValueError(f"unknown benchmark family: {self.name}")

    @property
    def is_regression(self) -> bool:
        return self.delta_pct >= self.threshold_pct


def parse_bench_times(output: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    active_name: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip().replace("µs", "us")
        if not line:
            continue
        if line.startswith("mcts_"):
            active_name = line
            continue
        if active_name is None:
            continue
        match = TIME_LINE_RE.search(line)
        if match is None:
            continue
        median_value = float(match.group(3))
        median_unit = match.group(4)
        parsed[active_name] = median_value * UNIT_TO_SECONDS[median_unit]
        active_name = None

    return parsed


def run_command(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n\n{proc.stdout}"
        )
    return proc.stdout


def ensure_benchmark_target_exists(repo_root: pathlib.Path) -> None:
    bench_file = repo_root / "crates" / "infotheory" / "benches" / "mcts_planners.rs"
    if not bench_file.is_file():
        raise RuntimeError(
            "required benchmark target is unavailable in this checkout: "
            f"missing {bench_file}"
        )


def ensure_baseline_ref_exists(repo_root: pathlib.Path, baseline_ref: str) -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{baseline_ref}^{{commit}}"],
        cwd=str(repo_root),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"unknown baseline ref: {baseline_ref}")


def baseline_contains_mcts_harness(repo_root: pathlib.Path, baseline_ref: str) -> bool:
    proc = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{baseline_ref}:crates/infotheory/benches/mcts_planners.rs",
        ],
        cwd=str(repo_root),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def run_benchmark(repo_root: pathlib.Path, target_dir: pathlib.Path, tmp_dir: pathlib.Path) -> str:
    ensure_benchmark_target_exists(repo_root)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["CARGO_TERM_COLOR"] = "never"
    cmd = [
        "cargo",
        "bench",
        "-p",
        "infotheory",
        "--bench",
        "mcts_planners",
        "--",
        "--noplot",
    ]
    return run_command(cmd, cwd=repo_root, env=env)


def git_worktree_add(repo_root: pathlib.Path, path: pathlib.Path, commit: str) -> None:
    cmd = ["git", "worktree", "add", "--detach", str(path), commit]
    run_command(cmd, cwd=repo_root, env=os.environ.copy())


def git_worktree_remove(repo_root: pathlib.Path, path: pathlib.Path) -> None:
    cmd = ["git", "worktree", "remove", "--force", str(path)]
    run_command(cmd, cwd=repo_root, env=os.environ.copy())


def sync_submodules(repo_root: pathlib.Path) -> None:
    env = os.environ.copy()
    run_command(["git", "submodule", "sync", "--recursive"], cwd=repo_root, env=env)
    run_command(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=repo_root,
        env=env,
    )


def prepare_baseline_source(
    repo_root: pathlib.Path, baseline_dir: pathlib.Path, baseline_ref: str
) -> str:
    try:
        git_worktree_add(repo_root, baseline_dir, baseline_ref)
        sync_submodules(baseline_dir)
        return "worktree"
    except RuntimeError as exc:
        print(f"[mcts-bench] worktree setup failed, falling back to copied checkout: {exc}")

    ignore = shutil.ignore_patterns(
        ".bench-runs",
        ".codex-test-target",
        ".codex-tmp",
        "target",
        "__pycache__",
    )
    shutil.copytree(repo_root, baseline_dir, ignore=ignore)
    run_command(
        ["git", "checkout", "--detach", baseline_ref],
        cwd=baseline_dir,
        env=os.environ.copy(),
    )
    sync_submodules(baseline_dir)
    return "copy"


def format_seconds(value: float) -> str:
    if value < 1e-6:
        return f"{value * 1e9:.2f} ns"
    if value < 1e-3:
        return f"{value * 1e6:.2f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.4f} s"


def write_summary(run_dir: pathlib.Path, results: list[BenchResult]) -> pathlib.Path:
    out = run_dir / "summary.tsv"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("benchmark\tbaseline_s\tcurrent_s\tdelta_pct\tthreshold_pct\tstatus\n")
        for row in results:
            status = "FAIL" if row.is_regression else "OK"
            fh.write(
                f"{row.name}\t{row.baseline_s:.12f}\t{row.current_s:.12f}\t"
                f"{row.delta_pct:.6f}\t{row.threshold_pct:.1f}\t{status}\n"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-vs-current MCTS planner benchmarks and enforce "
            "Tranche 3.5 Part 1 regression gates."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline git commit/branch/tag")
    parser.add_argument(
        "--root",
        default="",
        help="Directory for benchmark artifacts (default: .bench-runs/mcts-regression)",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    root = (
        pathlib.Path(args.root).resolve()
        if args.root
        else (repo_root / ".bench-runs" / "mcts-regression")
    )
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    baseline_worktree = run_dir / "baseline-src"
    baseline_target = run_dir / "baseline-target"
    current_target = run_dir / "current-target"
    tmp_dir = run_dir / "tmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mcts-bench] repo: {repo_root}")
    print(f"[mcts-bench] baseline: {args.baseline}")
    print(f"[mcts-bench] artifacts: {run_dir}")

    ensure_baseline_ref_exists(repo_root, args.baseline)
    if not baseline_contains_mcts_harness(repo_root, args.baseline):
        raise RuntimeError(
            "baseline predates the MCTS benchmark harness; use the first "
            "harness-bearing anchor commit or a later baseline"
        )

    baseline_output = ""
    current_output = ""
    baseline_mode = ""
    try:
        baseline_mode = prepare_baseline_source(repo_root, baseline_worktree, args.baseline)
        print(f"[mcts-bench] baseline source mode: {baseline_mode}")
        print("[mcts-bench] running baseline benchmark...")
        baseline_output = run_benchmark(baseline_worktree, baseline_target, tmp_dir)
        (run_dir / "baseline.log").write_text(baseline_output, encoding="utf-8")

        print("[mcts-bench] running current benchmark...")
        current_output = run_benchmark(repo_root, current_target, tmp_dir)
        (run_dir / "current.log").write_text(current_output, encoding="utf-8")
    finally:
        if baseline_mode == "worktree":
            try:
                git_worktree_remove(repo_root, baseline_worktree)
            except Exception as exc:  # pragma: no cover
                print(f"[mcts-bench] warning: failed to remove worktree: {exc}", file=sys.stderr)

    baseline_times = parse_bench_times(baseline_output)
    current_times = parse_bench_times(current_output)

    missing = [
        name
        for name in EXPECTED_BENCHMARKS
        if name not in baseline_times or name not in current_times
    ]
    if missing:
        print("[mcts-bench] ERROR: missing benchmark rows:")
        for name in missing:
            print(f"  - {name}")
        print(f"[mcts-bench] see logs in: {run_dir}")
        return 2

    results: list[BenchResult] = [
        BenchResult(
            name=name,
            baseline_s=baseline_times[name],
            current_s=current_times[name],
        )
        for name in EXPECTED_BENCHMARKS
    ]
    summary_path = write_summary(run_dir, results)

    print("")
    print(
        f"{'benchmark':72} {'baseline':>12} {'current':>12} "
        f"{'change':>10} {'gate':>7} {'status':>6}"
    )
    print("-" * 126)
    failures = []
    for row in results:
        status = "FAIL" if row.is_regression else "OK"
        gate = f"{row.threshold_pct:.1f}%"
        change = f"{row.delta_pct:+.2f}%"
        print(
            f"{row.name:72} {format_seconds(row.baseline_s):>12} {format_seconds(row.current_s):>12} "
            f"{change:>10} {gate:>7} {status:>6}"
        )
        if row.is_regression:
            failures.append(row)

    print("")
    print(f"[mcts-bench] summary: {summary_path}")
    if failures:
        print("[mcts-bench] RESULT: FAIL (regression gate exceeded)")
        return 1
    print("[mcts-bench] RESULT: PASS (all regression gates satisfied)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
