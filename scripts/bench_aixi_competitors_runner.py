#!/usr/bin/env python3
"""Run deterministic MC-AIXI competitor benchmarks and emit plot-ready TSV outputs.

This runner executes four implementations under equivalent parameter settings:
- infotheory-rust (CLI)
- infotheory-python (infotheory_rs bindings)
- pyaixi (Python reference)
- mcaixi-cpp (C++ reference)

It writes:
- raw.tsv: one row per run
- summary.tsv: grouped means/stddev by scenario and implementation
- report.txt: human-readable overview
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SUMMARY_RE = re.compile(
    r"SUMMARY:?\s*\n\s*agent age:\s*([0-9]+)\s*\n\s*average reward:\s*([-+0-9.eE]+)",
    re.IGNORECASE | re.MULTILINE,
)
INF_EVAL_AVG_RE = re.compile(r"Eval Average Reward per Cycle:\s*([-+0-9.eE]+)")
INF_EVAL_TOTAL_RE = re.compile(r"Eval Total Reward:\s*([-+0-9.eE]+)")
TIME_RE = re.compile(r"wall=([-+0-9.eE]+)\s+rss_kb=([0-9]+)")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    workload: str
    ct_depth: int
    horizon: int
    num_simulations: int
    eval_cycles: int
    exploration_exploitation_ratio: float


@dataclass(frozen=True)
class InfotheoryVariant:
    algorithm: str
    comparable_to_references: bool


def infotheory_variants() -> List[InfotheoryVariant]:
    return [
        InfotheoryVariant(algorithm="ac-ctw", comparable_to_references=True),
        InfotheoryVariant(algorithm="fac-ctw", comparable_to_references=False),
    ]


def infotheory_impl_name(base_name: str, *, algorithm: str) -> str:
    if algorithm == "ac-ctw":
        return base_name
    return f"{base_name}-{algorithm}"


def derive_trial_seed(base_seed: int, trial: int) -> int:
    if trial < 1:
        raise ValueError(f"trial must be >= 1, got {trial}")
    # Keep seeds deterministic and distinct across trials while preserving base seed semantics.
    return (base_seed + (trial - 1)) & 0xFFFFFFFF


def default_scenarios() -> List[Scenario]:
    # Matrix gives scaling along both eval_cycles and horizon.
    return [
        Scenario("coinflip_h4_c250_s200", "coinflip", 4, 4, 200, 250, 2.0),
        Scenario("coinflip_h4_c500_s200", "coinflip", 4, 4, 200, 500, 2.0),
        Scenario("coinflip_h6_c500_s250", "coinflip", 4, 6, 250, 500, 2.0),
        Scenario("kuhn_h2_c120_s200", "kuhn", 42, 2, 200, 120, 2.0),
        Scenario("kuhn_h2_c240_s200", "kuhn", 42, 2, 200, 240, 2.0),
        Scenario("kuhn_h3_c240_s250", "kuhn", 42, 3, 250, 240, 2.0),
    ]


def quick_scenarios() -> List[Scenario]:
    return [
        Scenario("coinflip_h4_c80_s120", "coinflip", 4, 4, 120, 80, 2.0),
        Scenario("kuhn_h2_c60_s120", "kuhn", 42, 2, 120, 60, 2.0),
    ]


def parse_time_file(path: Path) -> Tuple[float, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = TIME_RE.search(text)
    if not m:
        raise RuntimeError(f"failed to parse timing data from {path}")
    return float(m.group(1)), int(m.group(2))


def run_timed(
    *,
    time_bin: str,
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    time_path: Path,
    commands_log: Path,
) -> Tuple[float, int]:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    with commands_log.open("a", encoding="utf-8") as f:
        f.write(f"[{cwd}] {cmd_str}\n")

    wrapped = [time_bin, "-f", "wall=%e rss_kb=%M", "-o", str(time_path)] + cmd
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(wrapped, cwd=str(cwd), env=env, stdout=out, stderr=err)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {cmd_str}\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )
    return parse_time_file(time_path)


def parse_summary_avg_and_age(text: str) -> Tuple[float, int]:
    m = SUMMARY_RE.search(text)
    if not m:
        raise RuntimeError("could not parse SUMMARY block (agent age / average reward)")
    age = int(m.group(1))
    avg = float(m.group(2))
    return avg, age


def parse_infotheory_cli_metrics(text: str, fallback_steps: int) -> Tuple[float, float, int]:
    avg_m = INF_EVAL_AVG_RE.search(text)
    if not avg_m:
        raise RuntimeError("could not parse infotheory CLI eval average reward")
    avg = float(avg_m.group(1))

    total_m = INF_EVAL_TOTAL_RE.search(text)
    if total_m:
        total = float(total_m.group(1))
    else:
        total = avg * float(fallback_steps)

    return avg, total, fallback_steps


def decode_reported_reward(*, workload: str, impl: str, reward_avg: float) -> float:
    """Normalize implementation-specific printed rewards to the native domain scale.

    C++ MC-AIXI and PyAIXI print encoded unsigned reward symbols for Kuhn Poker.
    Their values are offset by +2 relative to the native reward scale.
    """
    if workload == "kuhn" and impl in {"pyaixi", "mcaixi-cpp"}:
        return reward_avg - 2.0
    return reward_avg


def write_infotheory_cli_config(
    *,
    path: Path,
    scenario: Scenario,
    seed: int,
    algorithm: str,
) -> None:
    environment = "coin-flip" if scenario.workload == "coinflip" else "kuhn-poker"
    cfg = {
        "environment": environment,
        "planner": "mc-aixi",
        "algorithm": algorithm,
        "ct_depth": scenario.ct_depth,
        "agent_horizon": scenario.horizon,
        "num_simulations": scenario.num_simulations,
        "exploration_exploitation_ratio": scenario.exploration_exploitation_ratio,
        "discount_gamma": 1.0,
        "learn_cycles": 0,
        "eval_cycles": scenario.eval_cycles,
        "terminate-lifetime": scenario.eval_cycles,
        "explore_epsilon": 0.0,
        "explore_gamma": 1.0,
        "random_seed": seed,
        "log_every": 0,
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def write_cpp_config(
    *,
    path: Path,
    scenario: Scenario,
    seed: int,
    coin_flip_p: float,
) -> None:
    lines = [
        f"environment = {'coin-flip' if scenario.workload == 'coinflip' else 'kuhn-poker'}",
        f"ct-depth = {scenario.ct_depth}",
        f"agent-horizon = {scenario.horizon}",
        f"mc-simulations = {scenario.num_simulations}",
        "exploration = 0.0",
        "explore-decay = 1.0",
        "learning-period = 0",
        f"terminate-age = {max(0, scenario.eval_cycles - 1)}",
        f"random-seed = {seed}",
        "verbose = false",
    ]
    if scenario.workload == "coinflip":
        lines.insert(1, f"coin-flip-p = {coin_flip_p}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_python_runner(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
import argparse
import json

try:
    import infotheory_rs as ait
except ModuleNotFoundError:
    # In minimal maturin-only installs, the extension is available as `_core`.
    import _core as ait


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["coinflip", "kuhn"], required=True)
    p.add_argument("--ct-depth", type=int, required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--num-simulations", type=int, required=True)
    p.add_argument("--eval-cycles", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--algorithm", choices=["ac-ctw", "fac-ctw"], required=True)
    p.add_argument("--coin-flip-p", type=float, required=True)
    p.add_argument("--exploration-exploitation-ratio", type=float, required=True)
    args = p.parse_args()

    if args.workload == "coinflip":
        env = ait.CoinFlipEnv(args.coin_flip_p, args.seed)
        cfg = ait.AgentConfig(
            algorithm=args.algorithm,
            ct_depth=args.ct_depth,
            agent_horizon=args.horizon,
            observation_bits=1,
            observation_stream_len=1,
            reward_bits=1,
            agent_actions=2,
            num_simulations=args.num_simulations,
            exploration_exploitation_ratio=args.exploration_exploitation_ratio,
            discount_gamma=1.0,
            min_reward=0,
            max_reward=1,
            reward_offset=0,
            random_seed=args.seed,
        )
    else:
        env = ait.KuhnPokerEnv(args.seed)
        cfg = ait.AgentConfig(
            algorithm=args.algorithm,
            ct_depth=args.ct_depth,
            agent_horizon=args.horizon,
            observation_bits=3,
            observation_stream_len=1,
            reward_bits=3,
            agent_actions=2,
            num_simulations=args.num_simulations,
            exploration_exploitation_ratio=args.exploration_exploitation_ratio,
            discount_gamma=1.0,
            min_reward=-2,
            max_reward=2,
            reward_offset=2,
            random_seed=args.seed,
        )

    summary = ait.run_agent_with_environment(
        env,
        cfg,
        learn_cycles=0,
        eval_cycles=args.eval_cycles,
        terminate_lifetime=args.eval_cycles,
        explore_epsilon=0.0,
        explore_gamma=1.0,
        check_finished=False,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def make_common_env(base_env: Dict[str, str], rayon_threads: int) -> Dict[str, str]:
    env = dict(base_env)
    env["PYTHONHASHSEED"] = "0"
    env["RAYON_NUM_THREADS"] = str(rayon_threads)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    return env


def prepare_pyaixi_run_root(*, pyaixi_root: Path, runtime_dir: Path) -> Path:
    """Create an isolated PyAIXI run tree without root-level module shadowing.

    The upstream repository includes a top-level `six.py` that shadows the pip
    package `six` under Python 3. Copying only `aixi.py` plus the `pyaixi/`
    package reproduces the expected runtime layout used by prior harnesses.
    """
    src_aixi = pyaixi_root / "aixi.py"
    src_pkg = pyaixi_root / "pyaixi"
    if not src_aixi.is_file() or not src_pkg.is_dir():
        raise RuntimeError(
            "invalid --pyaixi-root: expected aixi.py and pyaixi/ package"
        )

    run_root = runtime_dir / "pyaixi_run"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_aixi, run_root / "aixi.py")
    shutil.copytree(src_pkg, run_root / "pyaixi")
    return run_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bench-python", required=True)
    p.add_argument("--infotheory-bin", required=True)
    p.add_argument("--pyaixi-root", required=True)
    p.add_argument("--cpp-root", required=True)
    p.add_argument("--time-bin", required=True)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rayon-threads", type=int, default=1)
    p.add_argument("--coin-flip-p", type=float, default=0.9)
    p.add_argument("--profile", choices=["default", "quick"], default="default")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    pyaixi_root = Path(args.pyaixi_root).resolve()
    cpp_root = Path(args.cpp_root).resolve()
    infotheory_bin = Path(args.infotheory_bin).resolve()
    # Preserve the passed interpreter path (do not dereference venv symlinks).
    bench_python = Path(args.bench_python).absolute()

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    commands_log = out_dir / "commands.log"
    raw_tsv = out_dir / "raw.tsv"
    summary_tsv = out_dir / "summary.tsv"
    report_txt = out_dir / "report.txt"

    py_runner = runtime_dir / "run_infotheory_python_case.py"
    ensure_python_runner(py_runner)
    pyaixi_run_root = prepare_pyaixi_run_root(
        pyaixi_root=pyaixi_root,
        runtime_dir=runtime_dir,
    )

    scenarios = default_scenarios() if args.profile == "default" else quick_scenarios()
    infotheory_models = infotheory_variants()
    base_env = make_common_env(os.environ, args.rayon_threads)

    rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        for trial in range(1, args.trials + 1):
            trial_seed = derive_trial_seed(args.seed, trial)
            tag = f"{scenario.scenario_id}_trial{trial}"

            for variant in infotheory_models:
                model_name = variant.algorithm

                # infotheory-rust
                rust_impl = infotheory_impl_name("infotheory-rust", algorithm=model_name)
                rust_cfg = runtime_dir / f"rust_cfg_{rust_impl}_{tag}.json"
                write_infotheory_cli_config(
                    path=rust_cfg,
                    scenario=scenario,
                    seed=trial_seed,
                    algorithm=model_name,
                )
                rust_out = logs_dir / f"{rust_impl}_{tag}.out"
                rust_err = logs_dir / f"{rust_impl}_{tag}.err"
                rust_time = logs_dir / f"{rust_impl}_{tag}.time"
                wall_s, rss_kb = run_timed(
                    time_bin=args.time_bin,
                    cmd=[str(infotheory_bin), "aixi", str(rust_cfg)],
                    cwd=repo_root,
                    env=base_env,
                    stdout_path=rust_out,
                    stderr_path=rust_err,
                    time_path=rust_time,
                    commands_log=commands_log,
                )
                rust_text = rust_out.read_text(encoding="utf-8", errors="replace")
                reward_avg, reward_total, steps = parse_infotheory_cli_metrics(
                    rust_text, scenario.eval_cycles
                )
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "workload": scenario.workload,
                        "impl": rust_impl,
                        "algorithm": model_name,
                        "trial": trial,
                        "ct_depth": scenario.ct_depth,
                        "horizon": scenario.horizon,
                        "num_simulations": scenario.num_simulations,
                        "eval_cycles": scenario.eval_cycles,
                        "seed": trial_seed,
                        "rayon_threads": args.rayon_threads,
                        "wall_s": wall_s,
                        "rss_kb": rss_kb,
                        "steps": steps,
                        "reward_avg": reward_avg,
                        "reward_total": reward_total,
                        "cycles_per_sec": (steps / wall_s) if wall_s > 0 else float("nan"),
                    }
                )

                # infotheory-python
                py_impl = infotheory_impl_name("infotheory-python", algorithm=model_name)
                py_out = logs_dir / f"{py_impl}_{tag}.out"
                py_err = logs_dir / f"{py_impl}_{tag}.err"
                py_time = logs_dir / f"{py_impl}_{tag}.time"
                wall_s, rss_kb = run_timed(
                    time_bin=args.time_bin,
                    cmd=[
                        str(bench_python),
                        str(py_runner),
                        "--workload",
                        scenario.workload,
                        "--ct-depth",
                        str(scenario.ct_depth),
                        "--horizon",
                        str(scenario.horizon),
                        "--num-simulations",
                        str(scenario.num_simulations),
                        "--eval-cycles",
                        str(scenario.eval_cycles),
                        "--seed",
                        str(trial_seed),
                        "--algorithm",
                        model_name,
                        "--coin-flip-p",
                        str(args.coin_flip_p),
                        "--exploration-exploitation-ratio",
                        str(scenario.exploration_exploitation_ratio),
                    ],
                    cwd=repo_root,
                    env=base_env,
                    stdout_path=py_out,
                    stderr_path=py_err,
                    time_path=py_time,
                    commands_log=commands_log,
                )
                summary_line = (
                    py_out.read_text(encoding="utf-8", errors="replace")
                    .strip()
                    .splitlines()
                )
                if not summary_line:
                    raise RuntimeError(f"missing output for infotheory-python run: {py_out}")
                py_summary = json.loads(summary_line[-1])
                steps = int(scenario.eval_cycles)
                reward_avg = float(py_summary["eval_average_reward"])
                reward_total = float(py_summary["eval_total_reward"])
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "workload": scenario.workload,
                        "impl": py_impl,
                        "algorithm": model_name,
                        "trial": trial,
                        "ct_depth": scenario.ct_depth,
                        "horizon": scenario.horizon,
                        "num_simulations": scenario.num_simulations,
                        "eval_cycles": scenario.eval_cycles,
                        "seed": trial_seed,
                        "rayon_threads": args.rayon_threads,
                        "wall_s": wall_s,
                        "rss_kb": rss_kb,
                        "steps": steps,
                        "reward_avg": reward_avg,
                        "reward_total": reward_total,
                        "cycles_per_sec": (steps / wall_s) if wall_s > 0 else float("nan"),
                    }
                )

            # pyaixi
            pyax_out = logs_dir / f"pyaixi_{tag}.out"
            pyax_err = logs_dir / f"pyaixi_{tag}.err"
            pyax_time = logs_dir / f"pyaixi_{tag}.time"
            terminate_age = max(0, scenario.eval_cycles - 1)
            pyax_cmd = [
                str(bench_python),
                "aixi.py",
                "-e",
                "coin_flip" if scenario.workload == "coinflip" else "kuhn_poker",
                "-t",
                str(scenario.ct_depth),
                "-h",
                str(scenario.horizon),
                "-m",
                str(scenario.num_simulations),
                "-l",
                "0",
                "-r",
                str(terminate_age),
                "-x",
                "0.0",
                "-d",
                "1.0",
                "-o",
                f"random-seed={trial_seed}",
            ]
            if scenario.workload == "coinflip":
                pyax_cmd += ["-o", f"coin-flip-p={args.coin_flip_p}"]
            wall_s, rss_kb = run_timed(
                time_bin=args.time_bin,
                cmd=pyax_cmd,
                cwd=pyaixi_run_root,
                env=base_env,
                stdout_path=pyax_out,
                stderr_path=pyax_err,
                time_path=pyax_time,
                commands_log=commands_log,
            )
            pyax_text = pyax_out.read_text(encoding="utf-8", errors="replace")
            reward_avg, steps = parse_summary_avg_and_age(pyax_text)
            reward_avg = decode_reported_reward(
                workload=scenario.workload,
                impl="pyaixi",
                reward_avg=reward_avg,
            )
            reward_total = reward_avg * float(steps)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "workload": scenario.workload,
                    "impl": "pyaixi",
                    "algorithm": "ac-ctw",
                    "trial": trial,
                    "ct_depth": scenario.ct_depth,
                    "horizon": scenario.horizon,
                    "num_simulations": scenario.num_simulations,
                    "eval_cycles": scenario.eval_cycles,
                    "seed": trial_seed,
                    "rayon_threads": args.rayon_threads,
                    "wall_s": wall_s,
                    "rss_kb": rss_kb,
                    "steps": steps,
                    "reward_avg": reward_avg,
                    "reward_total": reward_total,
                    "cycles_per_sec": (steps / wall_s) if wall_s > 0 else float("nan"),
                }
            )

            # mcaixi-cpp
            cpp_cfg = runtime_dir / f"cpp_cfg_{tag}.conf"
            cpp_log = runtime_dir / f"cpp_log_{tag}.csv"
            write_cpp_config(
                path=cpp_cfg,
                scenario=scenario,
                seed=trial_seed,
                coin_flip_p=args.coin_flip_p,
            )
            cpp_out = logs_dir / f"mcaixi-cpp_{tag}.out"
            cpp_err = logs_dir / f"mcaixi-cpp_{tag}.err"
            cpp_time = logs_dir / f"mcaixi-cpp_{tag}.time"
            wall_s, rss_kb = run_timed(
                time_bin=args.time_bin,
                cmd=["./aixi", str(cpp_cfg), str(cpp_log)],
                cwd=cpp_root,
                env=base_env,
                stdout_path=cpp_out,
                stderr_path=cpp_err,
                time_path=cpp_time,
                commands_log=commands_log,
            )
            cpp_text = cpp_out.read_text(encoding="utf-8", errors="replace")
            reward_avg, steps = parse_summary_avg_and_age(cpp_text)
            reward_avg = decode_reported_reward(
                workload=scenario.workload,
                impl="mcaixi-cpp",
                reward_avg=reward_avg,
            )
            reward_total = reward_avg * float(steps)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "workload": scenario.workload,
                    "impl": "mcaixi-cpp",
                    "algorithm": "ac-ctw",
                    "trial": trial,
                    "ct_depth": scenario.ct_depth,
                    "horizon": scenario.horizon,
                    "num_simulations": scenario.num_simulations,
                    "eval_cycles": scenario.eval_cycles,
                    "seed": trial_seed,
                    "rayon_threads": args.rayon_threads,
                    "wall_s": wall_s,
                    "rss_kb": rss_kb,
                    "steps": steps,
                    "reward_avg": reward_avg,
                    "reward_total": reward_total,
                    "cycles_per_sec": (steps / wall_s) if wall_s > 0 else float("nan"),
                }
            )

    # Write raw.tsv
    raw_fields = [
        "scenario_id",
        "workload",
        "impl",
        "algorithm",
        "trial",
        "ct_depth",
        "horizon",
        "num_simulations",
        "eval_cycles",
        "seed",
        "rayon_threads",
        "wall_s",
        "rss_kb",
        "steps",
        "reward_avg",
        "reward_total",
        "cycles_per_sec",
    ]
    with raw_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate summary
    grouped: Dict[Tuple[str, str, str, str, int, int, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scenario_id"]),
            str(row["workload"]),
            str(row["impl"]),
            str(row["algorithm"]),
            int(row["horizon"]),
            int(row["num_simulations"]),
            int(row["eval_cycles"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, vals in sorted(grouped.items()):
        scenario_id, workload, impl, algorithm, horizon, num_simulations, eval_cycles = key

        def mean_std(name: str) -> Tuple[float, float]:
            xs = [float(v[name]) for v in vals]
            mean = statistics.mean(xs)
            std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
            return mean, std

        wall_mean, wall_std = mean_std("wall_s")
        rss_mean, rss_std = mean_std("rss_kb")
        reward_mean, reward_std = mean_std("reward_avg")
        cps_mean, cps_std = mean_std("cycles_per_sec")

        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "workload": workload,
                "impl": impl,
                "algorithm": algorithm,
                "horizon": horizon,
                "num_simulations": num_simulations,
                "eval_cycles": eval_cycles,
                "n": len(vals),
                "wall_mean_s": wall_mean,
                "wall_std_s": wall_std,
                "rss_mean_kb": rss_mean,
                "rss_std_kb": rss_std,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "cycles_per_sec_mean": cps_mean,
                "cycles_per_sec_std": cps_std,
            }
        )

    summary_fields = [
        "scenario_id",
        "workload",
        "impl",
        "algorithm",
        "horizon",
        "num_simulations",
        "eval_cycles",
        "n",
        "wall_mean_s",
        "wall_std_s",
        "rss_mean_kb",
        "rss_std_kb",
        "reward_mean",
        "reward_std",
        "cycles_per_sec_mean",
        "cycles_per_sec_std",
    ]
    with summary_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    # Human-readable report
    lines: List[str] = []
    lines.append("[Benchmark Summary]")
    lines.append(f"scenarios={len(scenarios)} trials={args.trials} rows={len(rows)}")
    lines.append(
        f"trial seeds use deterministic schedule: seed_t=(base_seed + trial - 1) mod 2^32; base_seed={args.seed}"
    )
    lines.append("reward metric uses native domain scale (Kuhn offsets removed for C++/PyAIXI)")
    lines.append("comparable model class: AC-CTW (infotheory-rust / infotheory-python, pyaixi, mcaixi-cpp)")
    lines.append("standalone infotheory datapoint: FAC-CTW (infotheory-rust-fac-ctw / infotheory-python-fac-ctw)")
    lines.append("")

    by_scenario: Dict[str, List[Dict[str, object]]] = {}
    for r in summary_rows:
        by_scenario.setdefault(str(r["scenario_id"]), []).append(r)

    for sid in sorted(by_scenario):
        rows_for_sid = sorted(by_scenario[sid], key=lambda r: str(r["impl"]))
        first = rows_for_sid[0]
        lines.append(
            f"[{sid}] workload={first['workload']} horizon={first['horizon']} "
            f"num_simulations={first['num_simulations']} eval_cycles={first['eval_cycles']}"
        )
        for r in rows_for_sid:
            lines.append(
                "  "
                f"{r['impl']} [{r['algorithm']}]: wall={float(r['wall_mean_s']):.4f}s±{float(r['wall_std_s']):.4f} "
                f"rss={float(r['rss_mean_kb']):.1f}KB±{float(r['rss_std_kb']):.1f} "
                f"reward={float(r['reward_mean']):.6f}±{float(r['reward_std']):.6f} "
                f"cycles/s={float(r['cycles_per_sec_mean']):.2f}±{float(r['cycles_per_sec_std']):.2f}"
            )

        lookup = {str(r["impl"]): r for r in rows_for_sid}
        if "infotheory-rust" in lookup and "pyaixi" in lookup:
            speedup = float(lookup["pyaixi"]["wall_mean_s"]) / float(
                lookup["infotheory-rust"]["wall_mean_s"]
            )
            lines.append(f"  speedup infotheory-rust vs pyaixi: {speedup:.2f}x")
        if "infotheory-rust" in lookup and "mcaixi-cpp" in lookup:
            speedup = float(lookup["mcaixi-cpp"]["wall_mean_s"]) / float(
                lookup["infotheory-rust"]["wall_mean_s"]
            )
            lines.append(f"  speedup infotheory-rust vs mcaixi-cpp: {speedup:.2f}x")
        lines.append("")

    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Benchmark complete.")
    print(f"Raw TSV: {raw_tsv}")
    print(f"Summary TSV: {summary_tsv}")
    print(f"Report: {report_txt}")


if __name__ == "__main__":
    main()
