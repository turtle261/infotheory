#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pyaixi_root="/var/tmp/pyaixi"
out_dir=""
python_bin=""
uv_bin="uv"
rayon_threads="1"
coinflip_trials="2"
coinflip_cycles="500"
kuhn_trials="1"
kuhn_cycles="200"
num_simulations="200"

usage() {
  cat <<'EOF'
Usage: scripts/bench_aixi_vs_pyaixi.sh [options]

Options:
  --pyaixi-root <path>       PyAIXI clone path (default: /var/tmp/pyaixi)
  --out-dir <path>           Output directory (default: target/aixi-vs-pyaixi/<timestamp>)
  --python-bin <path>        Base Python used by uv venv creation (default: .venv/bin/python if present, else python3)
  --uv-bin <path>            uv executable (default: uv)
  --rayon-threads <n>        RAYON_NUM_THREADS for infotheory runs (default: 1)
  --coinflip-trials <n>      Coin-flip trial count (default: 2)
  --coinflip-cycles <n>      Coin-flip eval cycles (default: 500)
  --kuhn-trials <n>          Kuhn Poker trial count (default: 1)
  --kuhn-cycles <n>          Kuhn Poker eval cycles (default: 200)
  --num-simulations <n>      MCTS simulations per step for both (default: 200)
  --help                     Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyaixi-root)
      pyaixi_root="$2"
      shift 2
      ;;
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

if [[ ! -d "$pyaixi_root" ]]; then
  echo "PyAIXI root does not exist: $pyaixi_root" >&2
  exit 1
fi

if [[ -z "$out_dir" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  out_dir="$repo_root/target/aixi-vs-pyaixi/$stamp"
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

echo "[env] $uv_bin pip install --python $bench_python maturin six"
record_cmd "'$uv_bin' pip install --python '$bench_python' 'maturin>=1.8' 'six>=1.16,<2'"

echo "[build] source $bench_venv/bin/activate && python -m maturin develop --release"
record_cmd "bash -lc 'source \"$bench_venv/bin/activate\" && python -m maturin develop --release'"

run_root="$out_dir/pyaixi_run"
rm -rf "$run_root"
mkdir -p "$run_root"
cp "$pyaixi_root/aixi.py" "$run_root/"
cp -R "$pyaixi_root/pyaixi" "$run_root/"

runner_py="$out_dir/run_infotheory_python_env.py"
cat > "$runner_py" <<'PY'
#!/usr/bin/env python
import argparse
import importlib
import inspect
import random
import sys
from pathlib import Path

import infotheory_rs as ait


def parse_env_options(items):
  options = {}
  for item in items:
    if "=" not in item:
      raise ValueError(f"invalid --env-option '{item}' (expected key=value)")
    key, value = item.split("=", 1)
    options[key] = value
  return options


def load_pyaixi_environment_class(module_name):
  module = importlib.import_module(f"pyaixi.environments.{module_name}")
  from pyaixi.environment import Environment as PyaixiEnvironmentBase

  for _, obj in inspect.getmembers(module, inspect.isclass):
    if obj.__module__ != module.__name__:
      continue
    if obj is PyaixiEnvironmentBase:
      continue
    if issubclass(obj, PyaixiEnvironmentBase):
      return obj

  raise RuntimeError(f"No pyaixi Environment subclass found in module '{module.__name__}'")


class PyaixiAdapter(ait.EnvironmentABC):
  def __init__(self, env):
    self._env = env
    raw_actions = [int(a) for a in getattr(env, "valid_actions", [])]
    if not raw_actions:
      raise ValueError("pyaixi environment exposed no valid_actions")
    self._action_from_index = sorted(raw_actions)

  def perform_action(self, action: int):
    raw_action = self._action_from_index[int(action)]
    return self._env.perform_action(raw_action)

  def get_observation(self) -> int:
    return int(self._env.observation)

  def get_reward(self) -> int:
    return int(self._env.reward)

  def is_finished(self) -> bool:
    return bool(self._env.is_finished)

  def get_observation_bits(self) -> int:
    return int(self._env.observation_bits())

  def get_reward_bits(self) -> int:
    return int(self._env.reward_bits())

  def get_action_bits(self) -> int:
    n = max(1, len(self._action_from_index))
    bits = 0
    while (1 << bits) < n:
      bits += 1
    return max(1, bits)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pyaixi-root", required=True)
  parser.add_argument("--environment", required=True)
  parser.add_argument("--algorithm", default="ac-ctw")
  parser.add_argument("--ct-depth", type=int, required=True)
  parser.add_argument("--horizon", type=int, required=True)
  parser.add_argument("--num-simulations", type=int, required=True)
  parser.add_argument("--learn-cycles", type=int, default=0)
  parser.add_argument("--eval-cycles", type=int, default=0)
  parser.add_argument("--terminate-lifetime", type=int, default=20)
  parser.add_argument("--explore-epsilon", type=float, default=0.0)
  parser.add_argument("--explore-gamma", type=float, default=1.0)
  parser.add_argument("--exploration-exploitation-ratio", type=float, default=1.41)
  parser.add_argument("--discount-gamma", type=float, default=1.0)
  parser.add_argument("--env-option", action="append", default=[])
  args = parser.parse_args()

  pyaixi_root = Path(args.pyaixi_root).resolve()
  sys.path.insert(0, str(pyaixi_root))

  options = parse_env_options(args.env_option)
  random.seed(int(options.get("random-seed", 0)))

  env_class = load_pyaixi_environment_class(args.environment)
  pyaixi_env = env_class(options=options)
  adapter = PyaixiAdapter(pyaixi_env)

  raw_rewards = [int(r) for r in getattr(pyaixi_env, "valid_rewards", [])]
  if not raw_rewards:
    raise ValueError("pyaixi environment exposed no valid_rewards")

  min_reward = min(raw_rewards)
  max_reward = max(raw_rewards)
  reward_offset = max(0, -min_reward)

  cfg = ait.AgentConfig(
    algorithm=args.algorithm,
    ct_depth=args.ct_depth,
    agent_horizon=args.horizon,
    observation_bits=adapter.get_observation_bits(),
    observation_stream_len=1,
    reward_bits=adapter.get_reward_bits(),
    agent_actions=len(adapter._action_from_index),
    num_simulations=args.num_simulations,
    exploration_exploitation_ratio=args.exploration_exploitation_ratio,
    discount_gamma=args.discount_gamma,
    min_reward=min_reward,
    max_reward=max_reward,
    reward_offset=reward_offset,
  )

  summary = ait.run_agent_with_environment(
    adapter,
    cfg,
    learn_cycles=args.learn_cycles,
    eval_cycles=args.eval_cycles,
    terminate_lifetime=args.terminate_lifetime,
    explore_epsilon=args.explore_epsilon,
    explore_gamma=args.explore_gamma,
    check_finished=False,
  )

  print(f"Eval Total Reward: {summary['eval_total_reward']}")
  print(f"Eval Average Reward per Cycle: {summary['eval_average_reward']:.6f}")
  print(f"Learn Total Reward: {summary['learn_total_reward']}")
  print(f"Learn cycles/s: {summary['learn_cycles_per_second']:.2f}")
  print(f"Eval cycles/s: {summary['eval_cycles_per_second']:.2f}")


if __name__ == "__main__":
  main()
PY

coinflip_terminate_age=$(( coinflip_cycles > 0 ? coinflip_cycles - 1 : 0 ))
kuhn_terminate_age=$(( kuhn_cycles > 0 ? kuhn_cycles - 1 : 0 ))

run_pair() {
  local workload="$1"
  local trial="$2"
  local inf_cmd="$3"
  local py_cmd="$4"

  local inf_out="$logs_dir/infotheory_${workload}_out_${trial}.txt"
  local inf_time="$logs_dir/infotheory_${workload}_time_${trial}.txt"
  local py_out="$logs_dir/pyaixi_${workload}_out_${trial}.txt"
  local py_time="$logs_dir/pyaixi_${workload}_time_${trial}.txt"

  echo "[run] infotheory $workload trial $trial"
  record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$inf_time' env RAYON_NUM_THREADS=$rayon_threads $inf_cmd > '$inf_out' 2>&1"

  echo "[run] pyaixi $workload trial $trial"
  record_cmd "/usr/bin/time -f 'wall=%e rss_kb=%M' -o '$py_time' bash -lc 'cd \"$run_root\" && env PYTHONHASHSEED=0 $py_cmd' > '$py_out' 2>&1"
}

for ((i=1; i<=coinflip_trials; i++)); do
  run_pair \
    "coinflip" \
    "$i" \
    "'$bench_python' '$runner_py' --pyaixi-root '$run_root' --environment coin_flip --algorithm ac-ctw --ct-depth 4 --horizon 6 --num-simulations $num_simulations --learn-cycles 0 --eval-cycles $coinflip_cycles --explore-epsilon 0.0 --explore-gamma 1.0 --exploration-exploitation-ratio 12.0 --discount-gamma 1.0 --env-option random-seed=0 --env-option coin-flip-p=0.9" \
    "'$bench_python' aixi.py -e coin_flip -t 4 -h 6 -m $num_simulations -l 0 -r $coinflip_terminate_age -x 0.0 -d 1.0 -o random-seed=0 -o coin-flip-p=0.9"
done

for ((i=1; i<=kuhn_trials; i++)); do
  run_pair \
    "kuhn" \
    "$i" \
    "'$bench_python' '$runner_py' --pyaixi-root '$run_root' --environment kuhn_poker --algorithm ac-ctw --ct-depth 42 --horizon 2 --num-simulations $num_simulations --learn-cycles 0 --eval-cycles $kuhn_cycles --explore-epsilon 0.0 --explore-gamma 1.0 --exploration-exploitation-ratio 2.0 --discount-gamma 1.0 --env-option random-seed=0" \
    "'$bench_python' aixi.py -e kuhn_poker -t 42 -h 2 -m $num_simulations -l 0 -r $kuhn_terminate_age -x 0.0 -d 1.0 -o random-seed=0"
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
inf_rew_re = re.compile(r"Eval Average Reward per Cycle:\s*([0-9.+-]+)")
py_rew_re = re.compile(r"SUMMARY:\s*\nagent age:\s*(\d+)\s*\naverage reward:\s*([0-9.+-]+)")

def parse_time(path):
    m = time_re.search(path.read_text())
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))

rows = []
for time_path in sorted(glob.glob(str(logs / "*_time_*.txt"))):
    tp = pathlib.Path(time_path)
    m = re.match(r"(infotheory|pyaixi)_(coinflip|kuhn)_time_(\d+)\.txt", tp.name)
    if not m:
        continue
    impl, workload, trial = m.group(1), m.group(2), int(m.group(3))
    wall_s, rss_kb = parse_time(tp)
    out_path = logs / f"{impl}_{workload}_out_{trial}.txt"
    reward_metric = None
    if out_path.exists():
        text = out_path.read_text()
        if impl == "infotheory":
            mm = inf_rew_re.search(text)
            if mm:
                reward_metric = float(mm.group(1))
        else:
            mm = py_rew_re.search(text)
            if mm:
                reward_metric = float(mm.group(2))
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
    for impl in ("infotheory", "pyaixi"):
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
    if data[workload].get("infotheory") and data[workload].get("pyaixi"):
        iw = [v[0] for v in data[workload]["infotheory"] if v[0] is not None]
        pw = [v[0] for v in data[workload]["pyaixi"] if v[0] is not None]
        ir = [v[1] for v in data[workload]["infotheory"] if v[1] is not None]
        pr = [v[1] for v in data[workload]["pyaixi"] if v[1] is not None]
        if iw and pw:
            lines.append(f"  speedup infotheory_vs_pyaixi: {statistics.mean(pw)/statistics.mean(iw):.2f}x")
        if ir and pr:
            lines.append(f"  rss_ratio infotheory_vs_pyaixi: {statistics.mean(ir)/statistics.mean(pr):.2f}x")
    lines.append("")

with open(report_txt, "w") as f:
    f.write("\n".join(lines))
PY

echo "Benchmark complete."
echo "Output directory: $out_dir"
echo "Summary TSV: $summary_tsv"
echo "Report: $report_txt"
