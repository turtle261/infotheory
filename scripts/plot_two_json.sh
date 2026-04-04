#!/bin/sh
set -eu

export LC_ALL=C

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PLOT_SUITE=${INFOTHEORY_PLOT_SUITE:-${INFOTHEORY_BENCH_SUITE:-two-json}}
PLOT_DIR=${INFOTHEORY_PLOT_OUTPUT_DIR:-/tmp/plotimgs}
SUMMARY_TSV=${INFOTHEORY_PLOT_SUMMARY_TSV:-}
BASELINE_SUMMARY_TSV=${INFOTHEORY_BASELINE_SUMMARY_TSV:-}
SUBJECT_FILTER=${INFOTHEORY_PLOT_SUBJECTS:-}
WORK_DIR=
PLOT_WIDTH=${INFOTHEORY_PLOT_WIDTH:-2400}
PLOT_HEIGHT=${INFOTHEORY_PLOT_HEIGHT:-1400}
PLOT_THEME=${INFOTHEORY_PLOT_THEME:-light}
PLOT_PALETTE=${INFOTHEORY_PLOT_PALETTE:-okabe-ito}

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"; }

case "${PLOT_SUITE}" in
  two-json|two_json|two|core|full)
    PLOT_SUITE=two-json
    SUITE_DISPLAY="configs/bench/two.json"
    SUITE_PATH_PREFIX="infotheory-two-json"
    SUITE_FOCUS_SUBJECTS="neural_mixture rwkv"
    ;;
  extra)
    PLOT_SUITE=extra
    SUITE_DISPLAY="configs/bench/extra.json"
    SUITE_PATH_PREFIX="infotheory-extra"
    SUITE_FOCUS_SUBJECTS="neural_mixture mamba"
    ;;
  *)
    fail "INFOTHEORY_PLOT_SUITE must be 'two-json' or 'extra' (found '${PLOT_SUITE}')"
    ;;
esac

cleanup() {
  if [ -n "${WORK_DIR}" ] && [ -d "${WORK_DIR}" ]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT HUP INT TERM

usage() {
  cat <<EOF
Usage: sh ./scripts/plot_two_json.sh

Creates SVG plots for the most recent completed ${SUITE_DISPLAY} benchmark
summary in /tmp, or for INFOTHEORY_PLOT_SUMMARY_TSV if provided.

Outputs:
  ${PLOT_DIR}/*.svg

Environment:
  INFOTHEORY_PLOT_SUITE=two-json|extra
  INFOTHEORY_PLOT_SUMMARY_TSV=/tmp/${SUITE_PATH_PREFIX}-summary-<stamp>.tsv
  INFOTHEORY_BASELINE_SUMMARY_TSV=benchmarks/baselines/${SUITE_PATH_PREFIX}-summary-<stamp>.tsv
  INFOTHEORY_PLOT_SUBJECTS=rwkv
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
  ls -1t "/tmp/${SUITE_PATH_PREFIX}-summary-"*.tsv 2>/dev/null | head -n 1 || true
}

resolve_summary_tsv() {
  explicit_path=$1
  if [ -n "${explicit_path}" ]; then
    printf '%s\n' "${explicit_path}"
    return 0
  fi
  latest=$(latest_summary_tsv)
  [ -n "${latest}" ] || fail "No completed benchmark summary TSV found in /tmp"
  printf '%s\n' "${latest}"
}

validate_summary_tsv() {
  summary_path=$1
  header=$(head -n 1 "${summary_path}")
  case "${header}" in
    operation$(printf '\t')subject$(printf '\t')subject_kind$(printf '\t')expert_kind$(printf '\t')series$(printf '\t')size_bytes*)
      ;;
    *)
      fail "Unexpected summary TSV header in ${summary_path}"
      ;;
  esac
}

filter_summary_tsv() {
  input_path=$1
  output_path=$2
  python3 - "${input_path}" "${SUBJECT_FILTER}" "${output_path}" <<'PY'
import csv
import re
import sys

input_path = sys.argv[1]
raw_filter = sys.argv[2]
output_path = sys.argv[3]
selected = {token for token in re.split(r"[\s,]+", raw_filter.strip()) if token}
if not selected:
    raise SystemExit("INFOTHEORY_PLOT_SUBJECTS must contain at least one subject")

with open(input_path, newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    rows = list(reader)

known = {row["subject"] for row in rows}
unknown = sorted(selected - known)
if unknown:
    raise SystemExit(
        "unknown INFOTHEORY_PLOT_SUBJECTS entries: "
        + ", ".join(unknown)
        + " (known: "
        + ", ".join(sorted(known))
        + ")"
    )

with open(output_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if row["subject"] in selected:
            writer.writerow(row)
PY
}

filter_summary_tsv_exact_value() {
  input_path=$1
  field_name=$2
  field_value=$3
  output_path=$4
  python3 - "${input_path}" "${field_name}" "${field_value}" "${output_path}" <<'PY'
import csv
import sys

input_path, field_name, field_value, output_path = sys.argv[1:5]

with open(input_path, newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    rows = list(reader)

if not rows:
    raise SystemExit(f"{input_path}: no rows to filter")

if field_name not in rows[0]:
    raise SystemExit(f"{input_path}: missing field {field_name!r}")

matched = [row for row in rows if row[field_name] == field_value]
if not matched:
    raise SystemExit(
        f"{input_path}: no rows matched {field_name}={field_value!r}"
    )

with open(output_path, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=reader.fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(matched)
PY
}

subset_tsv() {
  input_path=$1
  operation=$2
  out_path=$3
  awk -F '\t' -v operation="${operation}" 'NR == 1 || $1 == operation' "${input_path}" > "${out_path}"
}

combine_summary_tsvs() {
  current_path=$1
  baseline_path=$2
  output_path=$3
  python3 - "${current_path}" "${baseline_path}" "${output_path}" <<'PY'
import csv
import sys

current_path, baseline_path, output_path = sys.argv[1:4]

def load_rows(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames, rows

current_fields, current_rows = load_rows(current_path)
baseline_fields, baseline_rows = load_rows(baseline_path)
if current_fields != baseline_fields:
    raise SystemExit(
        "summary TSV columns do not match between current and baseline inputs"
    )

key_fields = ("operation", "subject", "size_bytes", "compression_backend")
current_keys = {
    tuple(row[field] for field in key_fields)
    for row in current_rows
}
baseline_keys = {
    tuple(row[field] for field in key_fields)
    for row in baseline_rows
}
current_only = len(current_keys - baseline_keys)
baseline_only = len(baseline_keys - current_keys)
if current_only or baseline_only:
    print(
        f"[plot] Baseline overlay row-key mismatch: current_only={current_only}, baseline_only={baseline_only}",
        file=sys.stderr,
    )

fieldnames = current_fields + [
    "summary_source",
    "subject_overlay",
    "series_overlay",
]
with open(output_path, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for source, rows in (("baseline", baseline_rows), ("current", current_rows)):
        for row in rows:
            row = dict(row)
            row["summary_source"] = source
            row["subject_overlay"] = f"{row['subject']} ({source})"
            row["series_overlay"] = f"{row['series']} ({source})"
            writer.writerow(row)
PY
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

slugify() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-'
}

SUMMARY_TSV=$(resolve_summary_tsv "${SUMMARY_TSV}")
[ -f "${SUMMARY_TSV}" ] || fail "Summary TSV not found: ${SUMMARY_TSV}"
validate_summary_tsv "${SUMMARY_TSV}"

if [ -n "${BASELINE_SUMMARY_TSV}" ]; then
  [ -f "${BASELINE_SUMMARY_TSV}" ] || fail "Baseline summary TSV not found: ${BASELINE_SUMMARY_TSV}"
  validate_summary_tsv "${BASELINE_SUMMARY_TSV}"
fi

WORK_DIR=$(mktemp -d "/tmp/${SUITE_PATH_PREFIX}-plot-work.XXXXXX")
mkdir -p "${PLOT_DIR}"

run_id=$(basename "${SUMMARY_TSV}")
run_id=${run_id#${SUITE_PATH_PREFIX}-summary-}
run_id=${run_id%.tsv}

if [ -n "${SUBJECT_FILTER}" ]; then
  filtered_summary_tsv="${WORK_DIR}/summary-filtered.tsv"
  filter_summary_tsv "${SUMMARY_TSV}" "${filtered_summary_tsv}"
  SUMMARY_TSV="${filtered_summary_tsv}"
  if [ -n "${BASELINE_SUMMARY_TSV}" ]; then
    filtered_baseline_summary_tsv="${WORK_DIR}/baseline-summary-filtered.tsv"
    filter_summary_tsv "${BASELINE_SUMMARY_TSV}" "${filtered_baseline_summary_tsv}"
    BASELINE_SUMMARY_TSV="${filtered_baseline_summary_tsv}"
  fi
  run_id="${run_id}-$(printf '%s' "${SUBJECT_FILTER}" | tr ' ,' '--')"
fi

h_tsv="${WORK_DIR}/h.tsv"
compress_tsv="${WORK_DIR}/compress.tsv"
decompress_tsv="${WORK_DIR}/decompress.tsv"

subset_tsv "${SUMMARY_TSV}" "h" "${h_tsv}"
subset_tsv "${SUMMARY_TSV}" "compress" "${compress_tsv}"
subset_tsv "${SUMMARY_TSV}" "decompress" "${decompress_tsv}"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "${SUITE_DISPLAY} h RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${compress_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-compress-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "${SUITE_DISPLAY} compress RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${decompress_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-decompress-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "subject" \
  "${SUITE_DISPLAY} decompress RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "${SUITE_DISPLAY} h wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${compress_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-compress-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "${SUITE_DISPLAY} compress wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${decompress_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-decompress-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "subject" \
  "${SUITE_DISPLAY} decompress wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${h_tsv}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-entropy-${run_id}.svg" \
  "size_bytes" "entropy_bpb_median" "subject" \
  "${SUITE_DISPLAY} h bits per byte vs size" \
  "size (bytes)" "bits per byte"

plot_line "${SUMMARY_TSV}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-all-time-${run_id}.svg" \
  "size_bytes" "real_seconds_median" "series" \
  "${SUITE_DISPLAY} all operations wall time vs size" \
  "size (bytes)" "seconds"

plot_line "${SUMMARY_TSV}" \
  "${PLOT_DIR}/${SUITE_PATH_PREFIX}-all-rss-${run_id}.svg" \
  "size_bytes" "rss_kib_median" "series" \
  "${SUITE_DISPLAY} all operations RSS vs size" \
  "size (bytes)" "peak RSS (KiB)"

if [ -n "${BASELINE_SUMMARY_TSV}" ]; then
  combined_summary_tsv="${WORK_DIR}/summary-with-baseline.tsv"
  combined_h_tsv="${WORK_DIR}/h-with-baseline.tsv"
  combined_compress_tsv="${WORK_DIR}/compress-with-baseline.tsv"
  combined_decompress_tsv="${WORK_DIR}/decompress-with-baseline.tsv"

  combine_summary_tsvs "${SUMMARY_TSV}" "${BASELINE_SUMMARY_TSV}" "${combined_summary_tsv}"
  subset_tsv "${combined_summary_tsv}" "h" "${combined_h_tsv}"
  subset_tsv "${combined_summary_tsv}" "compress" "${combined_compress_tsv}"
  subset_tsv "${combined_summary_tsv}" "decompress" "${combined_decompress_tsv}"

  plot_line "${combined_h_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-rss-baseline-${run_id}.svg" \
    "size_bytes" "rss_kib_median" "subject_overlay" \
    "${SUITE_DISPLAY} h RSS vs size (current vs baseline)" \
    "size (bytes)" "peak RSS (KiB)"

  plot_line "${combined_compress_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-compress-rss-baseline-${run_id}.svg" \
    "size_bytes" "rss_kib_median" "subject_overlay" \
    "${SUITE_DISPLAY} compress RSS vs size (current vs baseline)" \
    "size (bytes)" "peak RSS (KiB)"

  plot_line "${combined_decompress_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-decompress-rss-baseline-${run_id}.svg" \
    "size_bytes" "rss_kib_median" "subject_overlay" \
    "${SUITE_DISPLAY} decompress RSS vs size (current vs baseline)" \
    "size (bytes)" "peak RSS (KiB)"

  plot_line "${combined_h_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-time-baseline-${run_id}.svg" \
    "size_bytes" "real_seconds_median" "subject_overlay" \
    "${SUITE_DISPLAY} h wall time vs size (current vs baseline)" \
    "size (bytes)" "seconds"

  plot_line "${combined_compress_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-compress-time-baseline-${run_id}.svg" \
    "size_bytes" "real_seconds_median" "subject_overlay" \
    "${SUITE_DISPLAY} compress wall time vs size (current vs baseline)" \
    "size (bytes)" "seconds"

  plot_line "${combined_decompress_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-decompress-time-baseline-${run_id}.svg" \
    "size_bytes" "real_seconds_median" "subject_overlay" \
    "${SUITE_DISPLAY} decompress wall time vs size (current vs baseline)" \
    "size (bytes)" "seconds"

  plot_line "${combined_h_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-h-entropy-baseline-${run_id}.svg" \
    "size_bytes" "entropy_bpb_median" "subject_overlay" \
    "${SUITE_DISPLAY} h bits per byte vs size (current vs baseline)" \
    "size (bytes)" "bits per byte"

  plot_line "${combined_summary_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-all-time-baseline-${run_id}.svg" \
    "size_bytes" "real_seconds_median" "series_overlay" \
    "${SUITE_DISPLAY} all operations wall time vs size (current vs baseline)" \
    "size (bytes)" "seconds"

  plot_line "${combined_summary_tsv}" \
    "${PLOT_DIR}/${SUITE_PATH_PREFIX}-all-rss-baseline-${run_id}.svg" \
    "size_bytes" "rss_kib_median" "series_overlay" \
    "${SUITE_DISPLAY} all operations RSS vs size (current vs baseline)" \
    "size (bytes)" "peak RSS (KiB)"

  for subject in ${SUITE_FOCUS_SUBJECTS}; do
    subject_slug=$(slugify "${subject}")
    subject_tsv="${WORK_DIR}/${subject_slug}-with-baseline.tsv"
    subject_h_tsv="${WORK_DIR}/${subject_slug}-h-with-baseline.tsv"
    subject_compress_tsv="${WORK_DIR}/${subject_slug}-compress-with-baseline.tsv"
    subject_decompress_tsv="${WORK_DIR}/${subject_slug}-decompress-with-baseline.tsv"

    filter_summary_tsv_exact_value "${combined_summary_tsv}" "subject" "${subject}" "${subject_tsv}"
    subset_tsv "${subject_tsv}" "h" "${subject_h_tsv}"
    subset_tsv "${subject_tsv}" "compress" "${subject_compress_tsv}"
    subset_tsv "${subject_tsv}" "decompress" "${subject_decompress_tsv}"

    plot_line "${subject_h_tsv}" \
      "${PLOT_DIR}/${SUITE_PATH_PREFIX}-${subject_slug}-h-time-baseline-${run_id}.svg" \
      "size_bytes" "real_seconds_median" "summary_source" \
      "${SUITE_DISPLAY} ${subject} h wall time vs size (current vs baseline)" \
      "size (bytes)" "seconds"

    plot_line "${subject_h_tsv}" \
      "${PLOT_DIR}/${SUITE_PATH_PREFIX}-${subject_slug}-h-entropy-baseline-${run_id}.svg" \
      "size_bytes" "entropy_bpb_median" "summary_source" \
      "${SUITE_DISPLAY} ${subject} h bits per byte vs size (current vs baseline)" \
      "size (bytes)" "bits per byte"

    plot_line "${subject_compress_tsv}" \
      "${PLOT_DIR}/${SUITE_PATH_PREFIX}-${subject_slug}-compress-time-baseline-${run_id}.svg" \
      "size_bytes" "real_seconds_median" "summary_source" \
      "${SUITE_DISPLAY} ${subject} compress wall time vs size (current vs baseline)" \
      "size (bytes)" "seconds"

    plot_line "${subject_compress_tsv}" \
      "${PLOT_DIR}/${SUITE_PATH_PREFIX}-${subject_slug}-compress-archive-ratio-baseline-${run_id}.svg" \
      "size_bytes" "archive_ratio_median" "summary_source" \
      "${SUITE_DISPLAY} ${subject} compress archive ratio vs size (current vs baseline)" \
      "size (bytes)" "archive/input ratio"

    plot_line "${subject_decompress_tsv}" \
      "${PLOT_DIR}/${SUITE_PATH_PREFIX}-${subject_slug}-decompress-time-baseline-${run_id}.svg" \
      "size_bytes" "real_seconds_median" "summary_source" \
      "${SUITE_DISPLAY} ${subject} decompress wall time vs size (current vs baseline)" \
      "size (bytes)" "seconds"
  done
fi

say "[plot] Summary TSV: ${SUMMARY_TSV}"
if [ -n "${BASELINE_SUMMARY_TSV}" ]; then
  say "[plot] Baseline summary TSV: ${BASELINE_SUMMARY_TSV}"
fi
if [ -n "${SUBJECT_FILTER}" ]; then
  say "[plot] Subjects: ${SUBJECT_FILTER}"
fi
say "[plot] Output directory: ${PLOT_DIR}"
say "[plot] SVG files:"
ls -1 "${PLOT_DIR}/${SUITE_PATH_PREFIX}"-*-"${run_id}".svg
