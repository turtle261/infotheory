#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/scripts/workload_presets.sh"

find_llvm_profdata() {
  if command -v llvm-profdata >/dev/null 2>&1; then
    command -v llvm-profdata
    return
  fi
  if command -v rustup >/dev/null 2>&1; then
    if rustup which llvm-profdata >/dev/null 2>&1; then
      rustup which llvm-profdata
      return
    fi
  fi
  local host
  host="$(rustc -vV | sed -n 's/^host: //p')"
  local bundled
  bundled="$(rustc --print sysroot)/lib/rustlib/${host}/bin/llvm-profdata"
  if [[ -x "${bundled}" ]]; then
    printf '%s\n' "${bundled}"
    return
  fi
  echo "llvm-profdata not found" >&2
  return 1
}

mode="${1:-all}"
features="${INFOTHEORY_FEATURES:-cli}"
preset="${WORKLOAD_PRESET:-two-json}"
profile_root="${PGO_PROFILE_DIR:-${repo_root}/target/pgo/${preset}}"
raw_dir="${profile_root}/raw"
merged_profdata="${profile_root}/merged.profdata"
record_target_dir="${profile_root}/target-instr"
use_target_dir="${profile_root}/target-use"
record_bin="${record_target_dir}/release/infotheory"

record_profiles() {
  mkdir -p "${raw_dir}"
  echo "Building instrumented binary"
  RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-generate=${raw_dir}" \
    CARGO_TARGET_DIR="${record_target_dir}" \
    cargo build --release --features "${features}" --quiet

  local run_dir="${profile_root}/runs"
  mkdir -p "${run_dir}"
  configure_workload_preset "${repo_root}" "${record_bin}" "${run_dir}"
  echo "Recording PGO with preset ${WORKLOAD_PRESET_NAME}"
  for idx in "${!WORKLOAD_COMMANDS[@]}"; do
    local label="cmd_$(printf '%02d' "$((idx + 1))")"
    if [[ "${idx}" -lt "${#WORKLOAD_LABELS[@]}" ]] && [[ -n "${WORKLOAD_LABELS[$idx]}" ]]; then
      label="${WORKLOAD_LABELS[$idx]}"
    fi
    bash -lc "${WORKLOAD_COMMANDS[$idx]}" > "${run_dir}/${label}.stdout" 2> "${run_dir}/${label}.stderr"
  done
}

merge_profiles() {
  mkdir -p "${profile_root}"
  local llvm_profdata
  llvm_profdata="$(find_llvm_profdata)"
  mapfile -t profraws < <(find "${raw_dir}" -type f -name '*.profraw' | sort)
  if [[ "${#profraws[@]}" -eq 0 ]]; then
    echo "no .profraw files found under ${raw_dir}" >&2
    return 1
  fi
  "${llvm_profdata}" merge -output="${merged_profdata}" "${profraws[@]}"
}

build_with_profiles() {
  if [[ ! -f "${merged_profdata}" ]]; then
    echo "missing merged profdata: ${merged_profdata}" >&2
    return 1
  fi
  echo "Building profile-use binary"
  RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-use=${merged_profdata} -Cllvm-args=-pgo-warn-missing-function" \
    CARGO_TARGET_DIR="${use_target_dir}" \
    cargo build --release --features "${features}" --quiet
  echo "PGO binary: ${use_target_dir}/release/infotheory"
}

clean_profiles() {
  rm -rf "${profile_root}"
}

case "${mode}" in
  record)
    record_profiles
    ;;
  merge)
    merge_profiles
    ;;
  build)
    build_with_profiles
    ;;
  clean)
    clean_profiles
    ;;
  all)
    clean_profiles
    record_profiles
    merge_profiles
    build_with_profiles
    ;;
  *)
    echo "unknown pgo mode: ${mode}" >&2
    exit 1
    ;;
esac
