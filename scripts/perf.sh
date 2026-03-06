#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/scripts/workload_presets.sh"

mode="${1:-run}"
build_mode="${BUILD_MODE:-release}"
features="${INFOTHEORY_FEATURES:-cli}"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/target}"
bin_path="${target_dir}/${build_mode}/infotheory"
out_dir="${repo_root}/target/perf/${WORKLOAD_PRESET:-two-json}"

mkdir -p "${out_dir}"

run_cmd() {
  local label="$1"
  local command_text="$2"
  local base="${out_dir}/${label}"
  case "${mode}" in
    run)
      /usr/bin/time -f 'real %e user %U sys %S peak_rss_kb %M' \
        -o "${base}.time" \
        bash -lc "${command_text}" > "${base}.stdout" 2> "${base}.stderr"
      ;;
    stat)
      perf stat -d -o "${base}.perfstat" -- bash -lc "${command_text}" \
        > "${base}.stdout" 2> "${base}.stderr"
      ;;
    record)
      perf record -o "${base}.data" --call-graph dwarf -- bash -lc "${command_text}" \
        > "${base}.stdout" 2> "${base}.stderr"
      perf report --stdio -i "${base}.data" > "${base}.report"
      ;;
    all)
      /usr/bin/time -f 'real %e user %U sys %S peak_rss_kb %M' \
        -o "${base}.time" \
        bash -lc "${command_text}" > "${base}.stdout" 2> "${base}.stderr"
      perf stat -d -o "${base}.perfstat" -- bash -lc "${command_text}" \
        > "${base}.perf.stdout" 2> "${base}.perf.stderr"
      perf record -o "${base}.data" --call-graph dwarf -- bash -lc "${command_text}" \
        > "${base}.record.stdout" 2> "${base}.record.stderr"
      perf report --stdio -i "${base}.data" > "${base}.report"
      ;;
    *)
      echo "unknown perf mode: ${mode}" >&2
      return 1
      ;;
  esac
}

echo "Building infotheory (${build_mode}, features=${features})"
cargo build --"${build_mode}" --features "${features}" --quiet

configure_workload_preset "${repo_root}" "${bin_path}" "${out_dir}"
echo "Preset: ${WORKLOAD_PRESET_NAME}"
echo "Input: ${WORKLOAD_INPUT}"
echo "Bytes: ${WORKLOAD_BYTES}"

for idx in "${!WORKLOAD_COMMANDS[@]}"; do
  label="cmd_$(printf '%02d' "$((idx + 1))")"
  if [[ "${idx}" -lt "${#WORKLOAD_LABELS[@]}" ]] && [[ -n "${WORKLOAD_LABELS[$idx]}" ]]; then
    label="${WORKLOAD_LABELS[$idx]}"
  fi
  echo "Running ${label}"
  run_cmd "${label}" "${WORKLOAD_COMMANDS[$idx]}"
done
