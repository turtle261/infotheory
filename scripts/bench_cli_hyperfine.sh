#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/bench_cli_hyperfine.sh <baseline-commit> [preset]

Builds a detached baseline tree and a snapshot of the current dirty working tree,
then compares curated CLI workloads with hyperfine.

Environment:
  INFOTHEORY_CLI_BENCH_ROOT=/var/tmp/infotheory_bench
  INFOTHEORY_CLI_BENCH_PROFILE=release
  INFOTHEORY_CLI_BENCH_RUNS=3
  INFOTHEORY_CLI_BENCH_WARMUPS=1
  INFOTHEORY_CLI_BENCH_BYTES=8192
  INFOTHEORY_CLI_BENCH_BUILD_MODE=native|portable
  INFOTHEORY_CLI_BENCH_CARGO_FEATURES=cli
  INFOTHEORY_CLI_BENCH_BASELINE_FEATURES=...
  INFOTHEORY_CLI_BENCH_CURRENT_FEATURES=...
  INFOTHEORY_CLI_BENCH_NO_DEFAULT_FEATURES=0|1

Presets:
  default  Broad sweep across core rate/compression backends (default)
  quick    Smaller smoke sweep
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

build_rustflags() {
  local cpu_flag
  case "${INFOTHEORY_CLI_BENCH_BUILD_MODE:-native}" in
    native) cpu_flag="-C target-cpu=native" ;;
    portable) cpu_flag="-C target-cpu=generic" ;;
    *)
      echo "invalid INFOTHEORY_CLI_BENCH_BUILD_MODE: ${INFOTHEORY_CLI_BENCH_BUILD_MODE}" >&2
      exit 1
      ;;
  esac

  case "$(uname -s)" in
    Linux|FreeBSD|OpenBSD) printf '%s\n' "${cpu_flag} -C link-arg=-fuse-ld=lld" ;;
    *) printf '%s\n' "${cpu_flag}" ;;
  esac
}

prepare_input() {
  local repo_root="$1"
  local bytes="$2"
  local out_dir="$3"
  local out_path="${out_dir}/portable_${bytes}.bin"
  local bench_two
  bench_two="$(bench_config_path "${repo_root}" "two.json")"
  mkdir -p "${out_dir}"
  : > "${out_path}"
  while [[ "$(wc -c < "${out_path}")" -lt "${bytes}" ]]; do
    cat \
      "${repo_root}/README.md" \
      "${repo_root}/Cargo.toml" \
      "${bench_two}" >> "${out_path}"
  done
  truncate -s "${bytes}" "${out_path}"
  printf '%s\n' "${out_path}"
}

bench_config_path() {
  local repo_root="$1"
  local name="$2"
  if [[ -f "${repo_root}/configs/bench/${name}" ]]; then
    printf '%s\n' "${repo_root}/configs/bench/${name}"
    return 0
  fi
  if [[ -f "${repo_root}/examples/${name}" ]]; then
    printf '%s\n' "${repo_root}/examples/${name}"
    return 0
  fi
  printf '%s\n' "${SCRIPT_REPO_ROOT}/configs/bench/${name}"
}

copy_dirty_tree() {
  local src_root="$1"
  local dst_root="$2"
  local bench_root="$3"
  mkdir -p "${dst_root}"
  local exclude_args=(
    --exclude=.git
    --exclude=target
    --exclude=.cargo-target-*
    --exclude=.tmp
    --exclude=.bench-runs
  )
  if [[ "${bench_root}" == "${src_root}"/* ]]; then
    exclude_args+=(--exclude="${bench_root#${src_root}/}")
  fi
  tar -C "${src_root}" -cf - "${exclude_args[@]}" . | tar -C "${dst_root}" -xf -
}

build_infotheory_bin() {
  local src_root="$1"
  local build_root="$2"
  local rustflags="$3"
  local features="$4"
  local profile="${INFOTHEORY_CLI_BENCH_PROFILE:-release}"
  local no_default=()
  if [[ "${INFOTHEORY_CLI_BENCH_NO_DEFAULT_FEATURES:-0}" == "1" ]]; then
    no_default+=(--no-default-features)
  fi

  mkdir -p "${build_root}"
  (
    cd "${src_root}"
    export TMPDIR="${build_root}/tmp"
    mkdir -p "${TMPDIR}"
    export CARGO_TARGET_DIR="${build_root}/target"
    export RUSTFLAGS="${rustflags}"
    cargo build --locked -p infotheory --bin infotheory --profile "${profile}" "${no_default[@]}" --features "${features}"
  )
  printf '%s\n' "${build_root}/target/${profile}/infotheory"
}

run_hyperfine_case() {
  local out_dir="$1"
  local label="$2"
  local baseline_cmd="$3"
  local current_cmd="$4"
  local json_path="${out_dir}/${label}.json"

  hyperfine \
    --warmup "${INFOTHEORY_CLI_BENCH_WARMUPS:-1}" \
    --runs "${INFOTHEORY_CLI_BENCH_RUNS:-3}" \
    --export-json "${json_path}" \
    --command-name baseline \
    --command-name current \
    "${baseline_cmd}" \
    "${current_cmd}" >/dev/null
}

append_summary_rows() {
  local cases_dir="$1"
  local summary_tsv="$2"
  python3 - "$cases_dir" "$summary_tsv" <<'PY'
import json
import math
import pathlib
import sys

cases_dir = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])

with summary_tsv.open("w", encoding="utf-8") as f:
    f.write("label\tbaseline_mean_s\tbaseline_stddev_s\tcurrent_mean_s\tcurrent_stddev_s\tratio_current_over_baseline\n")
    for path in sorted(cases_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        if len(results) != 2:
            continue
        base, current = results
        base_mean = float(base["mean"])
        current_mean = float(current["mean"])
        base_std = base.get("stddev")
        current_std = current.get("stddev")
        base_std = "" if base_std is None else f"{float(base_std):.9f}"
        current_std = "" if current_std is None else f"{float(current_std):.9f}"
        ratio = math.inf if base_mean == 0 else current_mean / base_mean
        f.write(
            f"{path.stem}\t{base_mean:.9f}\t{base_std}\t{current_mean:.9f}\t{current_std}\t{ratio:.6f}\n"
        )
PY
}

build_commands() {
  local bin_path="$1"
  local repo_root="$2"
  local input_path="$3"
  local out_dir="$4"
  local preset="$5"

  local rwkv_cfg="cfg:hidden=64,layers=1,intermediate=64,decay_rank=16,a_rank=16,v_rank=16,g_rank=16,seed=22,train=adam,lr=0.0009,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=1.0,momentum=0.9)"
  local mamba_cfg="cfg:hidden=64,layers=1,intermediate=128,state=16,conv=4,dt_rank=16,seed=26,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)"
  local two_json
  local particle_fast_json
  two_json="$(bench_config_path "${repo_root}" "two.json")"
  particle_fast_json="$(bench_config_path "${repo_root}" "particle_fast.json")"
  mkdir -p "${out_dir}"

  CASE_LABELS=()
  CASE_COMMANDS=()

  add_case() {
    CASE_LABELS+=("$1")
    CASE_COMMANDS+=("$2")
  }

  add_case "h_rosaplus" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend rosaplus"
  add_case "h_ctw" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend ctw --method 32"
  add_case "h_fac_ctw" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend fac-ctw --method 16"
  add_case "h_match" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend match"
  add_case "h_sparse_match" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend sparse-match"
  add_case "h_ppmd" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend ppmd"
  add_case "h_sequitur" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend sequitur"
  add_case "h_mixture" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend mixture --method \"${two_json}\""
  add_case "h_particle" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend particle --method \"${particle_fast_json}\""
  add_case "h_rwkv7_cfg" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend rwkv7 --method \"${rwkv_cfg}\""
  add_case "h_mamba_cfg" "\"${bin_path}\" h \"${input_path}\" --compression-backend rate-ac --rate-backend mamba --method \"${mamba_cfg}\""
  add_case "compress_rate_ac_ctw" "\"${bin_path}\" compress \"${input_path}\" \"${out_dir}/ctw_ac.itc\" --compression-backend rate-ac --rate-backend ctw --method 32"
  add_case "compress_rate_rans_ppmd" "\"${bin_path}\" compress \"${input_path}\" \"${out_dir}/ppmd_rans.itc\" --compression-backend rate-rans --rate-backend ppmd"

  if [[ "${preset}" == "quick" ]]; then
    CASE_LABELS=("${CASE_LABELS[@]:0:6}")
    CASE_COMMANDS=("${CASE_COMMANDS[@]:0:6}")
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -ge 1 ]] || {
  usage >&2
  exit 1
}

need_cmd bash
need_cmd cargo
need_cmd git
need_cmd tar
need_cmd python3

if ! command -v hyperfine >/dev/null 2>&1; then
  echo "hyperfine is required for CLI benchmarking." >&2
  echo "Install it with: cargo install --locked hyperfine" >&2
  exit 1
fi

baseline_commit="$1"
preset="${2:-default}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_REPO_ROOT="${repo_root}"
bench_root="${INFOTHEORY_CLI_BENCH_ROOT:-/var/tmp/infotheory_bench}"
stamp="$(date +%Y%m%d-%H%M%S)"
run_root="${bench_root}/${stamp}"
baseline_root="${run_root}/baseline-src"
current_root="${run_root}/current-src"
baseline_build="${run_root}/baseline-build"
current_build="${run_root}/current-build"
cases_dir="${run_root}/cases"
mkdir -p "${cases_dir}"

rustflags="$(build_rustflags)"
shared_features="${INFOTHEORY_CLI_BENCH_CARGO_FEATURES:-cli}"
baseline_features="${INFOTHEORY_CLI_BENCH_BASELINE_FEATURES:-${shared_features}}"
current_features="${INFOTHEORY_CLI_BENCH_CURRENT_FEATURES:-${shared_features}}"

git worktree add --detach "${baseline_root}" "${baseline_commit}" >/dev/null
trap 'git worktree remove --force "${baseline_root}" >/dev/null 2>&1 || true' EXIT
git -C "${baseline_root}" submodule update --init --recursive >/dev/null
copy_dirty_tree "${repo_root}" "${current_root}" "${bench_root}"

baseline_bin="$(build_infotheory_bin "${baseline_root}" "${baseline_build}" "${rustflags}" "${baseline_features}")"
current_bin="$(build_infotheory_bin "${current_root}" "${current_build}" "${rustflags}" "${current_features}")"

baseline_input="$(prepare_input "${baseline_root}" "${INFOTHEORY_CLI_BENCH_BYTES:-8192}" "${run_root}/inputs-baseline")"
current_input="$(prepare_input "${current_root}" "${INFOTHEORY_CLI_BENCH_BYTES:-8192}" "${run_root}/inputs-current")"

build_commands "${baseline_bin}" "${baseline_root}" "${baseline_input}" "${run_root}/baseline-outputs" "${preset}"
baseline_labels=("${CASE_LABELS[@]}")
baseline_cmds=("${CASE_COMMANDS[@]}")
build_commands "${current_bin}" "${current_root}" "${current_input}" "${run_root}/current-outputs" "${preset}"
current_labels=("${CASE_LABELS[@]}")
current_cmds=("${CASE_COMMANDS[@]}")

if [[ "${#baseline_labels[@]}" -ne "${#current_labels[@]}" ]]; then
  echo "benchmark command generation mismatch between baseline and current trees" >&2
  exit 1
fi

for ((i = 0; i < ${#baseline_labels[@]}; i++)); do
  if [[ "${baseline_labels[$i]}" != "${current_labels[$i]}" ]]; then
    echo "benchmark labels diverged at index ${i}" >&2
    exit 1
  fi
  run_hyperfine_case "${cases_dir}" "${baseline_labels[$i]}" "${baseline_cmds[$i]}" "${current_cmds[$i]}"
done

summary_tsv="${run_root}/summary.tsv"
append_summary_rows "${cases_dir}" "${summary_tsv}"

echo "CLI benchmark comparison complete."
echo "Run root: ${run_root}"
echo "Summary TSV: ${summary_tsv}"
