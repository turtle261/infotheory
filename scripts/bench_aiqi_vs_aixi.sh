#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out_dir=""
python_bin=""
uv_bin="uv"
rayon_threads="1"
coinflip_trials="2"
coinflip_cycles="500"
kuhn_trials="2"
kuhn_cycles="200"
num_simulations="200"
aiqi_return_bins="32"
aiqi_baseline_exploration="0.01"

usage() {
  cat <<'EOF'
Usage: scripts/bench_aiqi_vs_aixi.sh [options]

This is a quick in-repo benchmark harness for relative comparisons.
It does NOT replicate the evaluation setup from "Universal AI with Q-Induction",
and it does not compare against external pyaixi runs.

Options:
  --out-dir <path>                    Output directory (default: target/aiqi-vs-aixi/<timestamp>)
  --python-bin <path>                 Base Python used by uv venv creation (default: .venv/bin/python if present, else python3)
  --uv-bin <path>                     uv executable (default: uv)
  --rayon-threads <n>                 RAYON_NUM_THREADS for both runs (default: 1)
  --coinflip-trials <n>               Coin-flip trial count (default: 2)
  --coinflip-cycles <n>               Coin-flip eval cycles (default: 500)
  --kuhn-trials <n>                   Kuhn Poker trial count (default: 2)
  --kuhn-cycles <n>                   Kuhn Poker eval cycles (default: 200)
  --num-simulations <n>               MCTS simulations for MC-AIXI (default: 200)
  --aiqi-return-bins <n>              Return discretization bins M for AIQI (default: 32)
  --aiqi-baseline-exploration <p>     Baseline exploration tau for AIQI (default: 0.01)
  --help                              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --uv-bin)
      uv_bin="$2"
      shift 2
      ;;
    --rayon-threads)
      rayon_threads="$2"
      shift 2
      ;;
    --coinflip-trials)
      coinflip_trials="$2"
      shift 2
      ;;
    --coinflip-cycles)
      coinflip_cycles="$2"
      shift 2
      ;;
    --kuhn-trials)
      kuhn_trials="$2"
      shift 2
      ;;
    --kuhn-cycles)
      kuhn_cycles="$2"
      shift 2
      ;;
    --num-simulations)
      num_simulations="$2"
      shift 2
      ;;
    --aiqi-return-bins)
      aiqi_return_bins="$2"
      shift 2
      ;;
    --aiqi-baseline-exploration)
      aiqi_baseline_exploration="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$out_dir" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  out_dir="$repo_root/target/aiqi-vs-aixi/$stamp"
fi

mkdir -p "$out_dir"
logs_dir="$out_dir/logs"
mkdir -p "$logs_dir"

commands_log="$out_dir/commands.log"
: > "$commands_log"

record_cmd() {
  printf '%s\n' "$*" >> "$commands_log"
  eval "$@"
}

if [[ -z "$python_bin" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
  else
    python_bin="python3"
  fi
fi

if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "uv executable not found: $uv_bin" >&2
  exit 1
fi

bench_venv="$out_dir/.uv-bench-venv"
bench_python="$bench_venv/bin/python"

echo "[env] $uv_bin venv --python $python_bin $bench_venv"
record_cmd "'$uv_bin' venv --python '$python_bin' '$bench_venv'"

echo "[env] $uv_bin pip install --python $bench_python maturin"
record_cmd "'$uv_bin' pip install --python '$bench_python' 'maturin>=1.8'"

echo "[build] source $bench_venv/bin/activate && python -m maturin develop --release"
record_cmd "bash -lc 'source \"$bench_venv/bin/activate\" && python -m maturin develop --release'"

runner_py="$out_dir/run_infotheory_aiqi_vs_aixi.py"
cat > "$runner_py" <<'PY'
#!/usr/bin/env python
import argparse

import infotheory_rs as ait


def make_environment(name: str):
  if name == "coinflip":
    return ait.CoinFlipEnv(0.9)
  if name == "kuhn":
    return ait.KuhnPokerEnv()
  raise ValueError(f"unknown environment: {name}")


def run_mcaixi(args):
  env = make_environment(args.environment)
  if args.environment == "coinflip":
    observation_bits = 1
    reward_bits = 1
    agent_actions = 2
    min_reward = 0
    max_reward = 1
    cfg = ait.AgentConfig(
      algorithm="ac-ctw",
      ct_depth=4,
      agent_horizon=6,
      observation_bits=observation_bits,
      observation_stream_len=1,
      reward_bits=reward_bits,
      agent_actions=agent_actions,
      num_simulations=args.num_simulations,
      exploration_exploitation_ratio=12.0,
      discount_gamma=0.99,
      min_reward=min_reward,
      max_reward=max_reward,
      reward_offset=max(0, -min_reward),
    )
  else:
    observation_bits = 4
    reward_bits = 3
    agent_actions = 2
    min_reward = -2
    max_reward = 4
    cfg = ait.AgentConfig(
      algorithm="ac-ctw",
      ct_depth=42,
      agent_horizon=2,
      observation_bits=observation_bits,
      observation_stream_len=1,
      reward_bits=reward_bits,
      agent_actions=agent_actions,
      num_simulations=args.num_simulations,
      exploration_exploitation_ratio=2.0,
      discount_gamma=0.99,
      min_reward=min_reward,
      max_reward=max_reward,
      reward_offset=max(0, -min_reward),
    )
  return ait.run_agent_with_environment(
    env,
    cfg,
    learn_cycles=0,
    eval_cycles=args.eval_cycles,
    terminate_lifetime=args.eval_cycles,
    explore_epsilon=0.0,
    explore_gamma=1.0,
    check_finished=False,
  )


def run_aiqi(args):
  env = make_environment(args.environment)
  if args.environment == "coinflip":
    observation_bits = 1
    reward_bits = 1
    agent_actions = 2
    min_reward = 0
    max_reward = 1
    cfg = ait.AiqiConfig(
      algorithm="ac-ctw",
      ct_depth=4,
      observation_bits=observation_bits,
      observation_stream_len=1,
      reward_bits=reward_bits,
      agent_actions=agent_actions,
      min_reward=min_reward,
      max_reward=max_reward,
      reward_offset=max(0, -min_reward),
      discount_gamma=0.99,
      return_horizon=6,
      return_bins=args.aiqi_return_bins,
      augmentation_period=6,
      baseline_exploration=args.aiqi_baseline_exploration,
    )
  else:
    observation_bits = 4
    reward_bits = 3
    agent_actions = 2
    min_reward = -2
    max_reward = 4
    cfg = ait.AiqiConfig(
      algorithm="ac-ctw",
      ct_depth=42,
      observation_bits=observation_bits,
      observation_stream_len=1,
      reward_bits=reward_bits,
      agent_actions=agent_actions,
      min_reward=min_reward,
      max_reward=max_reward,
      reward_offset=max(0, -min_reward),
      discount_gamma=0.99,
      return_horizon=2,
      return_bins=args.aiqi_return_bins,
      augmentation_period=2,
      baseline_exploration=args.aiqi_baseline_exploration,
    )
  return ait.run_aiqi_with_environment(
    env,
    cfg,
    learn_cycles=0,
    eval_cycles=args.eval_cycles,
    terminate_lifetime=args.eval_cycles,
    explore_epsilon=0.0,
    explore_gamma=1.0,
    check_finished=False,
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--solver", choices=["mcaixi", "aiqi"], required=True)
  parser.add_argument("--environment", choices=["coinflip", "kuhn"], required=True)
  parser.add_argument("--eval-cycles", type=int, required=True)
  parser.add_argument("--num-simulations", type=int, default=200)
  parser.add_argument("--aiqi-return-bins", type=int, default=32)
  parser.add_argument("--aiqi-baseline-exploration", type=float, default=0.01)
  args = parser.parse_args()

  if args.solver == "mcaixi":
    summary = run_mcaixi(args)
  else:
    summary = run_aiqi(args)

  print(f"Eval Total Reward: {summary['eval_total_reward']}")
  print(f"Eval Average Reward per Cycle: {summary['eval_average_reward']:.6f}")
  print(f"Learn Total Reward: {summary['learn_total_reward']}")
  print(f"Learn cycles/s: {summary['learn_cycles_per_second']:.2f}")
  print(f"Eval cycles/s: {summary['eval_cycles_per_second']:.2f}")


if __name__ == "__main__":
  main()
PY

run_pair() {
  local workload="$1"
  local trial="$2"
  local mcaixi_cmd="$3"
  local aiqi_cmd="$4"

  local mcaixi_out="$logs_dir/mcaixi_${workload}_out_${trial}.txt"
  local mcaixi_time="$logs_dir/mcaixi_${workload}_time_${trial}.txt"
  local aiqi_out="$logs_dir/aiqi_${workload}_out_${trial}.txt"
  local aiqi_time="$logs_dir/aiqi_${workload}_time_${trial}.txt"

  if (( trial % 2 == 1 )); then
    echo "[run] mcaixi $workload trial $trial"
    record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$mcaixi_time' env RAYON_NUM_THREADS=$rayon_threads $mcaixi_cmd > '$mcaixi_out' 2>&1"

    echo "[run] aiqi $workload trial $trial"
    record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$aiqi_time' env RAYON_NUM_THREADS=$rayon_threads $aiqi_cmd > '$aiqi_out' 2>&1"
  else
    echo "[run] aiqi $workload trial $trial"
    record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$aiqi_time' env RAYON_NUM_THREADS=$rayon_threads $aiqi_cmd > '$aiqi_out' 2>&1"

    echo "[run] mcaixi $workload trial $trial"
    record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$mcaixi_time' env RAYON_NUM_THREADS=$rayon_threads $mcaixi_cmd > '$mcaixi_out' 2>&1"
  fi
}

for ((i=1; i<=coinflip_trials; i++)); do
  run_pair \
    "coinflip" \
    "$i" \
    "'$bench_python' '$runner_py' --solver mcaixi --environment coinflip --eval-cycles $coinflip_cycles --num-simulations $num_simulations --aiqi-return-bins $aiqi_return_bins --aiqi-baseline-exploration $aiqi_baseline_exploration" \
    "'$bench_python' '$runner_py' --solver aiqi --environment coinflip --eval-cycles $coinflip_cycles --num-simulations $num_simulations --aiqi-return-bins $aiqi_return_bins --aiqi-baseline-exploration $aiqi_baseline_exploration"
done

for ((i=1; i<=kuhn_trials; i++)); do
  run_pair \
    "kuhn" \
    "$i" \
    "'$bench_python' '$runner_py' --solver mcaixi --environment kuhn --eval-cycles $kuhn_cycles --num-simulations $num_simulations --aiqi-return-bins $aiqi_return_bins --aiqi-baseline-exploration $aiqi_baseline_exploration" \
    "'$bench_python' '$runner_py' --solver aiqi --environment kuhn --eval-cycles $kuhn_cycles --num-simulations $num_simulations --aiqi-return-bins $aiqi_return_bins --aiqi-baseline-exploration $aiqi_baseline_exploration"
done

summary_tsv="$out_dir/summary.tsv"
cat > "$summary_tsv" <<'TSV'
workload	impl	trial	wall_s	rss_kb	reward_metric
TSV

"$bench_python" - "$logs_dir" "$summary_tsv" <<'PY'
import glob
import pathlib
import re
import sys

logs = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])

time_re = re.compile(r"wall=([0-9.]+)\s+rss_kb=(\d+)")
rew_re = re.compile(r"Eval Average Reward per Cycle:\s*([0-9.+-]+)")

def parse_time(path):
    m = time_re.search(path.read_text())
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))

rows = []
for time_path in sorted(glob.glob(str(logs / "*_time_*.txt"))):
    tp = pathlib.Path(time_path)
    m = re.match(r"(mcaixi|aiqi)_(coinflip|kuhn)_time_(\d+)\.txt", tp.name)
    if not m:
        continue
    impl, workload, trial = m.group(1), m.group(2), int(m.group(3))
    wall_s, rss_kb = parse_time(tp)
    out_path = logs / f"{impl}_{workload}_out_{trial}.txt"
    reward_metric = None
    if out_path.exists():
        mm = rew_re.search(out_path.read_text())
        if mm:
            reward_metric = float(mm.group(1))
    rows.append((workload, impl, trial, wall_s, rss_kb, reward_metric))

with summary_tsv.open("a") as f:
    for r in rows:
        f.write(
            f"{r[0]}\t{r[1]}\t{r[2]}\t"
            f"{'' if r[3] is None else r[3]:}\t"
            f"{'' if r[4] is None else r[4]:}\t"
            f"{'' if r[5] is None else r[5]:}\n"
        )
PY

report_txt="$out_dir/report.txt"
"$bench_python" - "$summary_tsv" "$report_txt" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

summary_tsv = sys.argv[1]
report_txt = sys.argv[2]

data = defaultdict(lambda: defaultdict(list))

with open(summary_tsv, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        workload = row["workload"]
        impl = row["impl"]
        wall = float(row["wall_s"]) if row["wall_s"] else None
        rss = float(row["rss_kb"]) if row["rss_kb"] else None
        rew = float(row["reward_metric"]) if row["reward_metric"] else None
        data[workload][impl].append((wall, rss, rew))

lines = []
for workload in sorted(data.keys()):
    lines.append(f"[{workload}]")
    for impl in ("mcaixi", "aiqi"):
        vals = data[workload].get(impl, [])
        if not vals:
            lines.append(f"  {impl}: no data")
            continue
        walls = [v[0] for v in vals if v[0] is not None]
        rss = [v[1] for v in vals if v[1] is not None]
        rew = [v[2] for v in vals if v[2] is not None]
        lines.append(
            f"  {impl}: n={len(vals)} "
            f"wall_mean={statistics.mean(walls):.4f}s "
            f"rss_mean={statistics.mean(rss):.1f}KB "
            + (f"reward_mean={statistics.mean(rew):.6f}" if rew else "reward_mean=n/a")
        )
    if data[workload].get("mcaixi") and data[workload].get("aiqi"):
        mw = [v[0] for v in data[workload]["mcaixi"] if v[0] is not None]
        aw = [v[0] for v in data[workload]["aiqi"] if v[0] is not None]
        mr = [v[1] for v in data[workload]["mcaixi"] if v[1] is not None]
        ar = [v[1] for v in data[workload]["aiqi"] if v[1] is not None]
        if mw and aw:
            lines.append(f"  speedup aiqi_vs_mcaixi: {statistics.mean(mw)/statistics.mean(aw):.2f}x")
        if mr and ar:
            lines.append(f"  rss_ratio aiqi_vs_mcaixi: {statistics.mean(ar)/statistics.mean(mr):.2f}x")
    lines.append("")

with open(report_txt, "w") as f:
    f.write("\n".join(lines))
PY

echo "Benchmark complete."
echo "Output directory: $out_dir"
echo "Summary TSV: $summary_tsv"
echo "Report: $report_txt"
