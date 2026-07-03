#!/usr/bin/env bash

# Shared workload configuration for perf and PGO scripts.
#
# Inputs:
# - WORKLOAD_PRESET: checked-in preset name (default: two-json)
# - INFOTHEORY_BENCH_INPUT: optional external corpus path
# - INFOTHEORY_BENCH_BYTES: optional byte limit for the portable default corpus
# - WORKLOAD_COMMANDS_FILE: optional shell file sourced after preset selection;
#   it may override WORKLOAD_LABELS / WORKLOAD_COMMANDS.
#
# Outputs:
# - WORKLOAD_PRESET_NAME
# - WORKLOAD_INPUT
# - WORKLOAD_BYTES
# - WORKLOAD_LABELS (bash array)
# - WORKLOAD_COMMANDS (bash array)

workload_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

workload_portable_input() {
  local repo_root="$1"
  local bytes="$2"
  local out_dir="${repo_root}/target/workloads"
  local out_path="${out_dir}/portable_${bytes}.bin"
  mkdir -p "${out_dir}"
  if [[ -f "${out_path}" ]] && [[ "$(wc -c < "${out_path}")" -eq "${bytes}" ]]; then
    printf '%s\n' "${out_path}"
    return
  fi
  : > "${out_path}"
  while [[ "$(wc -c < "${out_path}")" -lt "${bytes}" ]]; do
    cat \
      "${repo_root}/README.md" \
      "${repo_root}/LICENSE-APACHE" \
      "${repo_root}/Cargo.toml" \
      "${repo_root}/configs/bench/two.json" >> "${out_path}"
  done
  truncate -s "${bytes}" "${out_path}"
  printf '%s\n' "${out_path}"
}

configure_workload_preset() {
  local repo_root="$1"
  local bin_path="$2"
  local out_dir="$3"
  local preset="${WORKLOAD_PRESET:-two-json}"
  local bytes="${INFOTHEORY_BENCH_BYTES:-100000}"

  if [[ -n "${INFOTHEORY_BENCH_INPUT:-}" ]]; then
    WORKLOAD_INPUT="${INFOTHEORY_BENCH_INPUT}"
  else
    WORKLOAD_INPUT="$(workload_portable_input "${repo_root}" "${bytes}")"
  fi

  WORKLOAD_PRESET_NAME="${preset}"
  WORKLOAD_BYTES="${bytes}"
  WORKLOAD_LABELS=()
  WORKLOAD_COMMANDS=()

  case "${preset}" in
    two-json)
      WORKLOAD_LABELS=("two_json_rate_ac")
      WORKLOAD_COMMANDS=(
        "\"${bin_path}\" compress \"${WORKLOAD_INPUT}\" \"${out_dir}/two_json_rate_ac.itc\" --compression-backend rate-ac --rate-backend mixture --method \"${repo_root}/configs/bench/two.json\""
      )
      ;;
    one-sse)
      WORKLOAD_LABELS=("one_sse_rate_ac")
      WORKLOAD_COMMANDS=(
        "\"${bin_path}\" compress \"${WORKLOAD_INPUT}\" \"${out_dir}/one_sse_rate_ac.itc\" --compression-backend rate-ac --rate-backend calibrated --method \"${repo_root}/configs/bench/one_sse.json\""
      )
      ;;
    rwkv-all)
      WORKLOAD_LABELS=("rwkv_scope_all")
      WORKLOAD_COMMANDS=(
        "\"${bin_path}\" h \"${WORKLOAD_INPUT}\" --rate-backend rwkv7 --method \"cfg:hidden=64,layers=1,intermediate=64,decay_rank=16,a_rank=16,v_rank=16,g_rank=16,seed=22,train=adam,lr=0.0009,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=1.0,momentum=0.9)\""
      )
      ;;
    mamba-all)
      WORKLOAD_LABELS=("mamba_scope_all")
      WORKLOAD_COMMANDS=(
        "\"${bin_path}\" h \"${WORKLOAD_INPUT}\" --rate-backend mamba --method \"cfg:hidden=64,layers=1,intermediate=128,state=16,conv=4,dt_rank=16,seed=26,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)\""
      )
      ;;
    ctw-rate-ac)
      WORKLOAD_LABELS=("ctw_rate_ac")
      WORKLOAD_COMMANDS=(
        "\"${bin_path}\" compress \"${WORKLOAD_INPUT}\" \"${out_dir}/ctw_rate_ac.itc\" --compression-backend rate-ac --rate-backend ctw --method 32"
      )
      ;;
    *)
      echo "unknown workload preset: ${preset}" >&2
      return 1
      ;;
  esac

  if [[ -n "${WORKLOAD_COMMANDS_FILE:-}" ]]; then
    # The override file may mutate WORKLOAD_PRESET_NAME / WORKLOAD_INPUT /
    # WORKLOAD_LABELS / WORKLOAD_COMMANDS after inspecting the preset defaults.
    # shellcheck disable=SC1090
    source "${WORKLOAD_COMMANDS_FILE}"
  fi

  if [[ "${#WORKLOAD_COMMANDS[@]}" -eq 0 ]]; then
    echo "workload preset produced no commands" >&2
    return 1
  fi
}
