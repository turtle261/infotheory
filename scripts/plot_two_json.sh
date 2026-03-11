#!/bin/sh
set -eu

export LC_ALL=C

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PLOT_DIR=${INFOTHEORY_PLOT_OUTPUT_DIR:-/tmp/plotimgs}
SUMMARY_TSV=${INFOTHEORY_PLOT_SUMMARY_TSV:-}
WORK_DIR=
PLOT_WIDTH=${INFOTHEORY_PLOT_WIDTH:-2400}
PLOT_HEIGHT=${INFOTHEORY_PLOT_HEIGHT:-1400}
PLOT_THEME=${INFOTHEORY_PLOT_THEME:-light}
PLOT_PALETTE=${INFOTHEORY_PLOT_PALETTE:-okabe-ito}

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"; }

cleanup() {
  if [ -n "${WORK_DIR}" ] && [ -d "${WORK_DIR}" ]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT HUP INT TERM

usage() {
  cat <<EOF
Usage: sh ./scripts/plot_two_json.sh

Creates SVG plots for the most recent completed examples/two.json benchmark
summary in /tmp, or for INFOTHEORY_PLOT_SUMMARY_TSV if provided.

Outputs:
  ${PLOT_DIR}/*.svg

Environment:
  INFOTHEORY_PLOT_SUMMARY_TSV=/tmp/infotheory-two-json-summary-<stamp>.tsv
  INFOTHEORY_PLOT_OUTPUT_DIR=/tmp/plotimgs
  INFOTHEORY_PLOT_WIDTH=2400
  INFOTHEORY_PLOT_HEIGHT=1400
  INFOTHEORY_PLOT_THEME=light
  INFOTHEORY_PLOT_PALETTE=okabe-ito
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

need_cmd kuva
need_cmd awk
need_cmd mkdir
need_cmd ls
need_cmd head

latest_summary_tsv() {
  ls -1t /tmp/infotheory-two-json-summary-*.tsv 2>/dev/null | head -n 1 || true
}

resolve_summary_tsv() {
  if [ -n "${SUMMARY_TSV}" ]; then
    printf '%s\n' "${SUMMARY_TSV}"
    return 0
  fi
  latest=$(latest_summary_tsv)
  [ -n "${latest}" ] || fail "No completed benchmark summary TSV found in /tmp"
  printf '%s\n' "${latest}"
}

validate_summary_tsv() {
  header=$(head -n 1 "${SUMMARY_TSV}")
  case "${header}" in
    operation$(printf '\t')subject$(printf '\t')subject_kind$(printf '\t')expert_kind$(printf '\t')series$(printf '\t')size_bytes*)
      ;;
    *)
      fail "Unexpected summary TSV header in ${SUMMARY_TSV}"
      ;;
  esac
}

subset_tsv() {
  operation=$1
  out_path=$2
  awk -F '\t' -v operation="${operation}" 'NR == 1 || $1 == operation' "${SUMMARY_TSV}" > "${out_path}"
}

plot_line() {
  input_tsv=$1
  output_svg=$2
  x_col=$3
  y_col=$4
  color_col=$5
  title=$6
  x_label=$7
  y_label=$8
  shift 8

  kuva line "${input_tsv}" \
    --x "${x_col}" \
    --y "${y_col}" \
    --color-by "${color_col}" \
    --legend \
    --log-x \
    --stroke-width 3 \
    --width "${PLOT_WIDTH}" \
    --height "${PLOT_HEIGHT}" \
    --theme "${PLOT_THEME}" \
    --palette "${PLOT_PALETTE}" \
    --title "${title}" \
    --x-label "${x_label}" \
    --y-label "${y_label}" \
    -o "${output_svg}" \
    "$@"
}

SUMMARY_TSV=$(resolve_summary_tsv)
[ -f "${SUMMARY_TSV}" ] || fail "Summary TSV not found: ${SUMMARY_TSV}"
validate_summary_tsv

WORK_DIR=$(mktemp -d /tmp/infotheory-two-json-plot-work.XXXXXX)
mkdir -p "${PLOT_DIR}"

run_id=$(basename "${SUMMARY_TSV}")
run_id=${run_id#infotheory-two-json-summary-}
run_id=${run_id%.tsv}

h_tsv="${WORK_DIR}/h.tsv"
compress_tsv="${WORK_DIR}/compress.tsv"
decompress_tsv="${WORK_DIR}/decompress.tsv"

subset_tsv "h" "${h_tsv}"
subset_tsv "compress" "${compress_tsv}"
subset_tsv "decompress" "${decompress_tsv}"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-h-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "examples/two.json h RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${compress_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-compress-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "examples/two.json compress RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${decompress_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-decompress-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "examples/two.json decompress RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-h-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "examples/two.json h wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${compress_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-compress-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "examples/two.json compress wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${decompress_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-decompress-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "examples/two.json decompress wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/infotheory-two-json-h-entropy-${run_id}.svg" \
  "size_bytes" "entropy_bpb_median" "subject" \
  "examples/two.json h bits per byte vs size" \
  "size (bytes)" "bits per byte"

plot_line "${SUMMARY_TSV}" \
  "${PLOT_DIR}/infotheory-two-json-all-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "series" \
  "examples/two.json all operations wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${SUMMARY_TSV}" \
  "${PLOT_DIR}/infotheory-two-json-all-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "series" \
  "examples/two.json all operations RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

say "[plot] Summary TSV: ${SUMMARY_TSV}"
say "[plot] Output directory: ${PLOT_DIR}"
say "[plot] SVG files:"
ls -1 "${PLOT_DIR}"/infotheory-two-json-*-"${run_id}".svg
