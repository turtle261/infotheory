#!/bin/sh
set -eu

export LC_ALL=C

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SOURCE_FILE=${INFOTHEORY_BENCH_SOURCE:-/tmp/enwik7}
TIME_CMD=/usr/bin/time
BENCH_SUITE=${INFOTHEORY_BENCH_SUITE:-two-json}
COMP_BACKEND=${INFOTHEORY_BENCH_COMPRESSION_BACKEND:-rate-ac}
BENCH_FEATURES=${INFOTHEORY_BENCH_FEATURES:-cli}
BENCH_BUILD_MODE=${INFOTHEORY_BENCH_BUILD_MODE:-${INFOTHEORY_BUILD_MODE:-native}}
REPEATS=${INFOTHEORY_BENCH_REPEATS:-3}
WARMUPS=${INFOTHEORY_BENCH_WARMUPS:-1}
SIZES=${INFOTHEORY_BENCH_SIZES:-"4096 16384 65536 262144 1048576 2097152 4194304 10000000"}
SUBJECT_FILTER=${INFOTHEORY_BENCH_SUBJECTS:-}
WORKDIR_ROOT=${INFOTHEORY_BENCH_WORKDIR_ROOT:-${TMPDIR:-/tmp}}
STAMP=$(date +%Y%m%d-%H%M%S)
RAW_TSV=
SUMMARY_TSV=
WORK_DIR=
OUTPUT_MODE=
RAW_HEADER="operation	subject	subject_kind	expert_kind	series	size_bytes	repetition	cpu	compression_backend	input_sha256	suite_spec_path	suite_spec_sha256	build_mode	build_features	archive_bytes	entropy_bpb	real_seconds	user_seconds	sys_seconds	rss_kib	verified"

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"; }

case "${BENCH_SUITE}" in
  two-json|two_json|two|core|full)
    BENCH_SUITE=two-json
    SUITE_SPEC_PATH="${ROOT_DIR}/configs/bench/two.json"
    SUITE_DISPLAY="configs/bench/two.json"
    SUITE_PATH_PREFIX="infotheory-two-json"
    ;;
  one-sse|one_sse|one)
    BENCH_SUITE=one-sse
    SUITE_SPEC_PATH="${ROOT_DIR}/configs/bench/one_sse.json"
    SUITE_DISPLAY="configs/bench/one_sse.json"
    SUITE_PATH_PREFIX="infotheory-one-sse"
    ;;
  extra)
    BENCH_SUITE=extra
    SUITE_SPEC_PATH="${ROOT_DIR}/configs/bench/extra.json"
    SUITE_DISPLAY="configs/bench/extra.json"
    SUITE_PATH_PREFIX="infotheory-extra"
    ;;
  *)
    fail "INFOTHEORY_BENCH_SUITE must be 'two-json', 'one-sse', or 'extra' (found '${BENCH_SUITE}')"
    ;;
esac

[ -f "${SUITE_SPEC_PATH}" ] || fail "Benchmark spec not found: ${SUITE_SPEC_PATH}"
SUITE_SPEC_RESOLVED=$(python3 - "${SUITE_SPEC_PATH}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
SUITE_SPEC_SHA256=$(sha256sum "${SUITE_SPEC_PATH}" | awk 'NR==1 { print $1 }')

cleanup() {
  if [ "${INFOTHEORY_BENCH_KEEP_WORKDIR:-0}" = "1" ]; then
    if [ -n "${WORK_DIR}" ] && [ -d "${WORK_DIR}" ]; then
      say "[bench] Keeping work directory: ${WORK_DIR}"
    fi
    return 0
  fi
  if [ -n "${WORK_DIR}" ] && [ -d "${WORK_DIR}" ]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT HUP INT TERM

usage() {
  cat <<EOF
Usage: sh ./scripts/bench_two_json.sh

Runs a sequential benchmark suite for subjects derived from ${SUITE_DISPLAY},
using only /tmp/enwik7 as the source corpus.

Resume behavior:
  By default, resumes the newest /tmp/${SUITE_PATH_PREFIX}-raw-*.tsv if one exists.
  Set INFOTHEORY_BENCH_FRESH=1 to force a new timestamped run.
  Set INFOTHEORY_BENCH_RAW_TSV=/tmp/${SUITE_PATH_PREFIX}-raw-<stamp>.tsv to resume
  or append to a specific run file.

Environment:
  INFOTHEORY_BENCH_SUITE=two-json|one-sse|extra
  INFOTHEORY_BENCH_REPEATS=3
  INFOTHEORY_BENCH_WARMUPS=1
  INFOTHEORY_BENCH_SIZES="4096 16384 ... 10000000"
  INFOTHEORY_BENCH_SUBJECTS=rwkv7
  INFOTHEORY_BENCH_FEATURES="cli backend-rwkv"
  INFOTHEORY_BENCH_BUILD_MODE=native|portable
  INFOTHEORY_BENCH_CPU=11
  INFOTHEORY_BENCH_COMPRESSION_BACKEND=rate-ac
  INFOTHEORY_BENCH_WORKDIR_ROOT=/var/tmp
  INFOTHEORY_BENCH_FRESH=1
  INFOTHEORY_BENCH_RAW_TSV=/tmp/custom-raw.tsv
  INFOTHEORY_BENCH_SUMMARY_TSV=/tmp/custom-summary.tsv
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

need_cmd cargo
need_cmd python3
need_cmd head
need_cmd taskset
need_cmd sha256sum
need_cmd cmp
need_cmd wc
need_cmd du
[ -x "${TIME_CMD}" ] || fail "Missing required command: ${TIME_CMD}"

[ -f "${SOURCE_FILE}" ] || fail "Expected source file at ${SOURCE_FILE}"
source_wc=$(wc -c < "${SOURCE_FILE}" | tr -d ' ')
[ "${source_wc}" = "10000000" ] || fail "${SOURCE_FILE} must be exactly 10000000 bytes by wc -c (found ${source_wc})"
source_du=$(du -b "${SOURCE_FILE}" | awk 'NR==1 { print $1 }')
[ "${source_du}" = "10000000" ] || fail "${SOURCE_FILE} must be exactly 10000000 bytes by du -b (found ${source_du})"

case "${COMP_BACKEND}" in
  rate-ac|rate-rans) ;;
  *)
    fail "INFOTHEORY_BENCH_COMPRESSION_BACKEND must be 'rate-ac' or 'rate-rans'"
    ;;
esac

BENCH_FEATURES=$(printf '%s' "${BENCH_FEATURES}" | tr ',' ' ' | xargs)
[ -n "${BENCH_FEATURES}" ] || BENCH_FEATURES="cli"
case " ${BENCH_FEATURES} " in
  *" cli "*) ;;
  *) BENCH_FEATURES="cli ${BENCH_FEATURES}" ;;
esac

case "${BENCH_BUILD_MODE}" in
  native|portable) ;;
  *)
    fail "INFOTHEORY_BENCH_BUILD_MODE (or INFOTHEORY_BUILD_MODE fallback) must be 'native' or 'portable'"
    ;;
esac

portable_rustflags() {
  case "$(uname -s)" in
    Linux|FreeBSD|OpenBSD) printf '%s' "-C target-cpu=generic -C link-arg=-fuse-ld=lld" ;;
    *) printf '%s' "-C target-cpu=generic" ;;
  esac
}

build_infotheory_cli() {
  if [ "${BENCH_BUILD_MODE}" = "portable" ]; then
    bench_rustflags=$(portable_rustflags)
    (
      cd "${ROOT_DIR}" && \
      CARGO_INCREMENTAL=0 \
      CARGO_BUILD_RUSTFLAGS="${bench_rustflags}" \
      RUSTDOCFLAGS="${RUSTDOCFLAGS:-${bench_rustflags}}" \
      cargo build --release --features "${BENCH_FEATURES}" --bin infotheory --locked
    )
    return 0
  fi

  (cd "${ROOT_DIR}" && CARGO_INCREMENTAL=0 cargo build --release --features "${BENCH_FEATURES}" --bin infotheory --locked)
}

case "${REPEATS}" in
  ''|*[!0-9]*)
    fail "INFOTHEORY_BENCH_REPEATS must be a non-negative integer"
    ;;
esac
case "${WARMUPS}" in
  ''|*[!0-9]*)
    fail "INFOTHEORY_BENCH_WARMUPS must be a non-negative integer"
    ;;
esac
[ "${REPEATS}" -gt 0 ] || fail "INFOTHEORY_BENCH_REPEATS must be > 0"

derive_summary_path() {
  raw_path=$1
  case "${raw_path}" in
    *-raw-*.tsv)
      printf '%s\n' "$(printf '%s' "${raw_path}" | sed 's/-raw-/-summary-/')"
      ;;
    *.tsv)
      printf '%s\n' "${raw_path%.tsv}.summary.tsv"
      ;;
    *)
      printf '%s.summary.tsv\n' "${raw_path}"
      ;;
  esac
}

latest_existing_raw_tsv() {
  ls -1t "/tmp/${SUITE_PATH_PREFIX}-raw-"*.tsv 2>/dev/null | head -n 1 || true
}

current_two_json_baseline_tsv() {
  ls -1t "${ROOT_DIR}/benchmarks/current/infotheory-two-json-summary"*.tsv 2>/dev/null | head -n 1 || true
}

resolve_output_paths() {
  latest_raw=
  if [ -n "${INFOTHEORY_BENCH_RAW_TSV:-}" ]; then
    RAW_TSV=${INFOTHEORY_BENCH_RAW_TSV}
    OUTPUT_MODE=explicit
  elif [ "${INFOTHEORY_BENCH_FRESH:-0}" = "1" ]; then
    RAW_TSV=/tmp/${SUITE_PATH_PREFIX}-raw-${STAMP}.tsv
    OUTPUT_MODE=fresh
  else
    latest_raw=$(latest_existing_raw_tsv)
    if [ -n "${latest_raw}" ]; then
      RAW_TSV=${latest_raw}
      OUTPUT_MODE=resume-latest
    else
      RAW_TSV=/tmp/${SUITE_PATH_PREFIX}-raw-${STAMP}.tsv
      OUTPUT_MODE=fresh
    fi
  fi

  if [ -n "${INFOTHEORY_BENCH_SUMMARY_TSV:-}" ]; then
    SUMMARY_TSV=${INFOTHEORY_BENCH_SUMMARY_TSV}
  else
    SUMMARY_TSV=$(derive_summary_path "${RAW_TSV}")
  fi
}

validate_existing_raw_tsv() {
  python3 - "${RAW_TSV}" "${RAW_HEADER}" <<'PY'
import csv
import sys

path = sys.argv[1]
expected = sys.argv[2].split("\t")
seen = set()

with open(path, newline="") as fh:
    reader = csv.reader(fh, delimiter="\t")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SystemExit(f"{path}: empty file") from exc
    if header != expected:
        raise SystemExit(f"{path}: unexpected header")
    index = {name: idx for idx, name in enumerate(header)}
    for lineno, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != len(expected):
            raise SystemExit(
                f"{path}: line {lineno}: expected {len(expected)} fields, found {len(row)}"
            )
        if row[index["verified"]] != "1":
            continue
        key = (
            row[index["operation"]],
            row[index["subject"]],
            row[index["size_bytes"]],
            row[index["repetition"]],
            row[index["cpu"]],
            row[index["compression_backend"]],
            row[index["input_sha256"]],
            row[index["suite_spec_sha256"]],
            row[index["build_mode"]],
            row[index["build_features"]],
        )
        if key in seen:
            raise SystemExit(f"{path}: duplicate verified row at line {lineno}: {key!r}")
        seen.add(key)
PY
}

initialize_raw_tsv() {
  if [ -f "${RAW_TSV}" ]; then
    validate_existing_raw_tsv
    return 0
  fi
  printf '%s\n' "${RAW_HEADER}" > "${RAW_TSV}"
}

mkdir -p "${WORKDIR_ROOT}"
WORK_DIR=$(mktemp -d "${WORKDIR_ROOT%/}/${SUITE_PATH_PREFIX}-work.XXXXXX")
BIN_PATH="${ROOT_DIR}/target/release/infotheory"
SUBJECTS_TSV="${WORK_DIR}/subjects.tsv"

resolve_output_paths
initialize_raw_tsv

choose_cpu() {
  if [ -n "${INFOTHEORY_BENCH_CPU:-}" ]; then
    printf '%s\n' "${INFOTHEORY_BENCH_CPU}"
    return 0
  fi

  allowed=$(taskset -pc $$ 2>/dev/null | awk -F': ' 'NR==1 { print $2 }')
  [ -n "${allowed}" ] || fail "Failed to determine allowed CPU affinity list"
  python3 - "${allowed}" <<'PY'
import sys

allowed = sys.argv[1].strip()
cpus = []
for part in allowed.split(','):
    part = part.strip()
    if not part:
        continue
    if '-' in part:
        lo, hi = part.split('-', 1)
        cpus.extend(range(int(lo), int(hi) + 1))
    else:
        cpus.append(int(part))
if not cpus:
    raise SystemExit("no CPUs available from taskset")
for preferred in (11, 0):
    if preferred in cpus:
        print(preferred)
        raise SystemExit(0)
print(min(cpus))
PY
}

CPU=$(choose_cpu)

say "[bench] Building release CLI binary with features: ${BENCH_FEATURES}"
build_infotheory_cli
[ -x "${BIN_PATH}" ] || fail "Expected built binary at ${BIN_PATH}"

python3 - "${SUITE_SPEC_PATH}" "${ROOT_DIR}" "${WORK_DIR}" "${SUITE_DISPLAY}" > "${SUBJECTS_TSV}" <<'PY'
import json
import re
import sys
from pathlib import Path

spec_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
work_dir = Path(sys.argv[3]).resolve()
suite_label = sys.argv[4]
subject_dir = work_dir / "subjects"
subject_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(spec_path.read_text())
spec_dir = spec_path.parent

PATH_KEYS = {
    "spec_path",
    "base_path",
    "path",
    "model_path",
    "rwkv_model_path",
    "mamba_model_path",
}


def looks_like_uri(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value))


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return text.strip("-").lower() or "expert"

def canonical_subject_name(expert):
    kind = str(expert.get("kind") or "")
    name = str(expert.get("name") or kind or "expert")
    # The canonical two-json CTW slot is now the factorized byte/MSB model.
    # Preserve old suite files that still named this subject "ctw" while keeping
    # deliberately custom names untouched.
    if kind == "fac-ctw" and slug(name) == "ctw":
        return "fac-ctw"
    return name

def resolve_path_like(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if looks_like_uri(value):
        raise SystemExit(f"{suite_label}: URI paths are not supported in benchmark suite files")
    return (base_dir / candidate).resolve()

def load_neural_mixture(path: Path, label: str):
    value = json.loads(path.read_text())
    if value.get("kind") != "neural":
        raise SystemExit(f"expected {label} kind=neural, found {value.get('kind')!r}")
    experts_value = value.get("experts")
    if not isinstance(experts_value, list) or not experts_value:
        raise SystemExit(f"{label} must contain a non-empty experts array")
    return value, experts_value, path.parent

def emit_experts(experts_value, expert_base_dir):
    for expert in experts_value:
        expert_resolved = canonicalize_relative_paths_with_base(expert, expert_base_dir)
        name = canonical_subject_name(expert_resolved)
        subject = slug(name)
        out_path = subject_dir / f"{subject}.json"
        out_path.write_text(json.dumps(expert_resolved, indent=2, sort_keys=True) + "\n")
        h_order = expert_resolved.get("max_order", "")
        print(
            "\t".join(
                [
                    subject,
                    "expert",
                    str(expert_resolved.get("kind", "")),
                    str(out_path),
                    "" if h_order == "" else str(h_order),
                ]
            )
        )

def canonicalize_relative_paths_with_base(node, base_dir):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if (
                isinstance(value, str)
                and value
                and (key in PATH_KEYS or key.endswith("_path"))
                and not Path(value).is_absolute()
                and not looks_like_uri(value)
            ):
                out[key] = str((base_dir / value).resolve())
            else:
                out[key] = canonicalize_relative_paths_with_base(value, base_dir)
        return out
    if isinstance(node, list):
        return [canonicalize_relative_paths_with_base(item, base_dir) for item in node]
    return node

print("subject\tsubject_kind\texpert_kind\tspec_path\th_order")

if data.get("kind") == "neural":
    _, experts, expert_dir = load_neural_mixture(spec_path, suite_label)
    print(
        "\t".join(
            [
                "neural_mixture",
                "mixture",
                "neural-mixture",
                str(spec_path),
                "",
            ]
        )
    )
    emit_experts(experts, expert_dir)
else:
    base = data.get("base")
    if not isinstance(base, dict) or base.get("kind") != "mixture":
        raise SystemExit(
            f"expected {suite_label} to be kind=neural or a calibrated spec with base.kind=mixture"
        )
    spec_ref = base.get("spec_path")
    if not isinstance(spec_ref, str) or not spec_ref:
        raise SystemExit(f"{suite_label} calibrated base must use a non-empty spec_path")
    base_path = resolve_path_like(spec_ref, spec_dir)
    _, experts, expert_dir = load_neural_mixture(base_path, f"{suite_label} base {spec_ref}")
    print(
        "\t".join(
            [
                "calibrated_mixture",
                "calibrated",
                "calibrated-mixture",
                str(spec_path),
                "",
            ]
        )
    )
    emit_experts(experts, expert_dir)
PY

if [ -n "${SUBJECT_FILTER}" ]; then
  filtered_subjects_tsv="${WORK_DIR}/subjects-filtered.tsv"
  python3 - "${SUBJECTS_TSV}" "${SUBJECT_FILTER}" > "${filtered_subjects_tsv}" <<'PY'
import csv
import re
import sys

subjects_path = sys.argv[1]
raw_filter = sys.argv[2]
selected = {token for token in re.split(r"[\s,]+", raw_filter.strip()) if token}
if not selected:
    raise SystemExit("INFOTHEORY_BENCH_SUBJECTS must contain at least one subject")

with open(subjects_path, newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    rows = list(reader)

known = {row["subject"] for row in rows}
unknown = sorted(selected - known)
if unknown:
    raise SystemExit(
        "unknown INFOTHEORY_BENCH_SUBJECTS entries: "
        + ", ".join(unknown)
        + " (known: "
        + ", ".join(sorted(known))
        + ")"
    )

writer = csv.DictWriter(
    sys.stdout,
    fieldnames=["subject", "subject_kind", "expert_kind", "spec_path", "h_order"],
    delimiter="\t",
    lineterminator="\n",
)
writer.writeheader()
for row in rows:
    if row["subject"] in selected:
        writer.writerow(row)
PY
  SUBJECTS_TSV="${filtered_subjects_tsv}"
fi

run_plain() {
  cmd_op=$1
  cmd_input_path=$2
  cmd_output_path=$3
  cmd_subject_kind=$4
  cmd_spec_path=$5
  cmd_h_order=$6

  set -- "${BIN_PATH}"
  case "${cmd_op}" in
    h)
      set -- "$@" h "${cmd_input_path}"
      if [ -n "${cmd_h_order}" ]; then
        set -- "$@" "${cmd_h_order}"
      fi
      ;;
    compress)
      set -- "$@" compress "${cmd_input_path}" "${cmd_output_path}" --compression-backend "${COMP_BACKEND}"
      ;;
    decompress)
      set -- "$@" decompress "${cmd_input_path}" "${cmd_output_path}" --compression-backend "${COMP_BACKEND}"
      ;;
    *)
      fail "Unsupported operation for run_plain: ${cmd_op}"
      ;;
  esac

  case "${cmd_subject_kind}" in
    mixture)
      set -- "$@" --rate-backend mixture --method "${cmd_spec_path}"
      ;;
    calibrated)
      set -- "$@" --rate-backend calibrated --method "${cmd_spec_path}"
      ;;
    expert)
      set -- "$@" --expert-spec "${cmd_spec_path}"
      ;;
    *)
      fail "Unsupported subject_kind: ${cmd_subject_kind}"
      ;;
  esac

  cmd_log_path=$(mktemp "${WORK_DIR}/plain-${cmd_op}.XXXXXX.log")
  if env RAYON_NUM_THREADS=1 taskset -c "${CPU}" "${@}" >"${cmd_log_path}" 2>&1; then
    rm -f "${cmd_log_path}"
    return 0
  fi

  cat "${cmd_log_path}" >&2 || true
  fail "Command failed: ${cmd_op} ${cmd_input_path}"
}

run_timed() {
  cmd_op=$1
  cmd_input_path=$2
  cmd_output_path=$3
  cmd_subject_kind=$4
  cmd_spec_path=$5
  cmd_h_order=$6
  cmd_stdout_path=$7
  cmd_stderr_path=$8
  cmd_time_path=$9

  set -- "${BIN_PATH}"
  case "${cmd_op}" in
    h)
      set -- "$@" h "${cmd_input_path}"
      if [ -n "${cmd_h_order}" ]; then
        set -- "$@" "${cmd_h_order}"
      fi
      ;;
    compress)
      set -- "$@" compress "${cmd_input_path}" "${cmd_output_path}" --compression-backend "${COMP_BACKEND}"
      ;;
    decompress)
      set -- "$@" decompress "${cmd_input_path}" "${cmd_output_path}" --compression-backend "${COMP_BACKEND}"
      ;;
    *)
      fail "Unsupported operation for run_timed: ${cmd_op}"
      ;;
  esac

  case "${cmd_subject_kind}" in
    mixture)
      set -- "$@" --rate-backend mixture --method "${cmd_spec_path}"
      ;;
    calibrated)
      set -- "$@" --rate-backend calibrated --method "${cmd_spec_path}"
      ;;
    expert)
      set -- "$@" --expert-spec "${cmd_spec_path}"
      ;;
    *)
      fail "Unsupported subject_kind: ${cmd_subject_kind}"
      ;;
  esac

  if env RAYON_NUM_THREADS=1 "${TIME_CMD}" -f '%e\t%U\t%S\t%M' -o "${cmd_time_path}" \
    taskset -c "${CPU}" "${@}" > "${cmd_stdout_path}" 2> "${cmd_stderr_path}"; then
    return 0
  fi

  cat "${cmd_stderr_path}" >&2 || true
  fail "Timed command failed: ${cmd_op} ${cmd_input_path}"
}

parse_time_file() {
  IFS="$(printf '\t')" read -r real_seconds user_seconds sys_seconds rss_kib < "$1"
  printf '%s\t%s\t%s\t%s\n' "${real_seconds}" "${user_seconds}" "${sys_seconds}" "${rss_kib}"
}

validate_float() {
  value=$1
  python3 - "${value}" <<'PY'
import math
import sys

value = sys.argv[1].strip()
try:
    parsed = float(value)
except ValueError as exc:
    raise SystemExit(f"not a float: {value!r}") from exc
if not math.isfinite(parsed):
    raise SystemExit(f"not finite: {value!r}")
PY
}

append_row() {
  sep=
  for field in "$@"; do
    printf '%s%s' "${sep}" "${field}" >> "${RAW_TSV}"
    sep="$(printf '\t')"
  done
  printf '\n' >> "${RAW_TSV}"
}

row_exists() {
  row_operation=$1
  row_subject=$2
  row_size_bytes=$3
  row_rep=$4
  row_cpu=$5
  row_compression_backend=$6
  row_input_sha256=$7
  row_suite_spec_sha256=$8
  row_build_mode=$9
  row_build_features=${10}

  awk -F '\t' \
    -v row_operation="${row_operation}" \
    -v row_subject="${row_subject}" \
    -v row_size_bytes="${row_size_bytes}" \
    -v row_rep="${row_rep}" \
    -v row_cpu="${row_cpu}" \
    -v row_compression_backend="${row_compression_backend}" \
    -v row_input_sha256="${row_input_sha256}" \
    -v row_suite_spec_sha256="${row_suite_spec_sha256}" \
    -v row_build_mode="${row_build_mode}" \
    -v row_build_features="${row_build_features}" \
    '
      NR > 1 &&
      $1 == row_operation &&
      $2 == row_subject &&
      $6 == row_size_bytes &&
      $7 == row_rep &&
      $8 == row_cpu &&
      $9 == row_compression_backend &&
      $10 == row_input_sha256 &&
      $12 == row_suite_spec_sha256 &&
      $13 == row_build_mode &&
      $14 == row_build_features &&
      $21 == "1" {
        found = 1
      }
      END {
        exit found ? 0 : 1
      }
    ' "${RAW_TSV}"
}

say "[bench] Source: ${SOURCE_FILE}"
say "[bench] Suite: ${BENCH_SUITE} (${SUITE_DISPLAY})"
say "[bench] Suite spec: ${SUITE_SPEC_RESOLVED}"
say "[bench] Suite spec digest: ${SUITE_SPEC_SHA256}"
say "[bench] CPU affinity: ${CPU}"
say "[bench] Compression backend: ${COMP_BACKEND}"
say "[bench] Build mode: ${BENCH_BUILD_MODE}"
say "[bench] Build features: ${BENCH_FEATURES}"
say "[bench] Repeats: ${REPEATS}"
say "[bench] Warmups: ${WARMUPS}"
say "[bench] Sizes: ${SIZES}"
if [ -n "${SUBJECT_FILTER}" ]; then
  say "[bench] Subjects: ${SUBJECT_FILTER}"
fi
say "[bench] Raw TSV: ${RAW_TSV}"
say "[bench] Summary TSV: ${SUMMARY_TSV}"
case "${OUTPUT_MODE}" in
  explicit)
    say "[bench] Output mode: explicit raw TSV"
    ;;
  resume-latest)
    say "[bench] Output mode: resume latest raw TSV"
    ;;
  fresh)
    say "[bench] Output mode: fresh run"
    ;;
esac

for size_bytes in ${SIZES}; do
  case "${size_bytes}" in
    ''|*[!0-9]*)
      fail "Every INFOTHEORY_BENCH_SIZES entry must be a positive integer"
      ;;
  esac
  [ "${size_bytes}" -gt 0 ] || fail "Benchmark size must be > 0"
  [ "${size_bytes}" -le 10000000 ] || fail "Benchmark size ${size_bytes} exceeds 10000000 bytes"

  input_path="${WORK_DIR}/input-${size_bytes}.bin"
  head -c "${size_bytes}" "${SOURCE_FILE}" > "${input_path}"
  actual_size=$(wc -c < "${input_path}" | tr -d ' ')
  [ "${actual_size}" = "${size_bytes}" ] || fail "Failed to create ${size_bytes}-byte slice from ${SOURCE_FILE}"
  input_sha256=$(sha256sum "${input_path}" | awk 'NR==1 { print $1 }')

  while IFS="$(printf '\t')" read -r subject subject_kind expert_kind spec_path h_order; do
    [ "${subject}" = "subject" ] && continue

    need_subject_work=0
    rep=1
    while [ "${rep}" -le "${REPEATS}" ]; do
      if ! row_exists "h" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "-" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_subject_work=1
        break
      fi
      if ! row_exists "compress" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_subject_work=1
        break
      fi
      if ! row_exists "decompress" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_subject_work=1
        break
      fi
      rep=$((rep + 1))
    done

    if [ "${need_subject_work}" -eq 0 ]; then
      say "[bench] size=${size_bytes} subject=${subject} (already complete)"
      continue
    fi

    say "[bench] size=${size_bytes} subject=${subject}"

    warmup_idx=1
    while [ "${warmup_idx}" -le "${WARMUPS}" ]; do
      archive_path="${WORK_DIR}/warmup-${subject}-${size_bytes}.itc"
      restored_path="${WORK_DIR}/warmup-${subject}-${size_bytes}.out"
      run_plain h "${input_path}" "${WORK_DIR}/unused" "${subject_kind}" "${spec_path}" "${h_order}"
      run_plain compress "${input_path}" "${archive_path}" "${subject_kind}" "${spec_path}" "${h_order}"
      run_plain decompress "${archive_path}" "${restored_path}" "${subject_kind}" "${spec_path}" "${h_order}"
      cmp -s "${input_path}" "${restored_path}" || fail "Warmup round-trip failed for ${subject} at ${size_bytes} bytes"
      rm -f "${archive_path}" "${restored_path}"
      warmup_idx=$((warmup_idx + 1))
    done

    rep=1
    while [ "${rep}" -le "${REPEATS}" ]; do
      stdout_path="${WORK_DIR}/stdout-${subject}-${size_bytes}-${rep}.txt"
      stderr_path="${WORK_DIR}/stderr-${subject}-${size_bytes}-${rep}.txt"
      time_path="${WORK_DIR}/time-${subject}-${size_bytes}-${rep}.txt"
      archive_path="${WORK_DIR}/archive-${subject}-${size_bytes}-${rep}.itc"
      restored_path="${WORK_DIR}/restored-${subject}-${size_bytes}-${rep}.bin"

      need_h=0
      need_compress=0
      need_decompress=0
      archive_ready=0

      if ! row_exists "h" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "-" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_h=1
      fi
      if ! row_exists "compress" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_compress=1
      fi
      if ! row_exists "decompress" "${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" "${input_sha256}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}"; then
        need_decompress=1
      fi

      if [ "${need_h}" -eq 1 ]; then
        run_timed h "${input_path}" "${WORK_DIR}/unused" "${subject_kind}" "${spec_path}" "${h_order}" \
          "${stdout_path}" "${stderr_path}" "${time_path}"
        entropy_bpb=$(tr -d '\r\n' < "${stdout_path}")
        validate_float "${entropy_bpb}"
        IFS="$(printf '\t')" read -r real_seconds user_seconds sys_seconds rss_kib <<EOF
$(parse_time_file "${time_path}")
EOF
        append_row \
          "h" "${subject}" "${subject_kind}" "${expert_kind}" "h:${subject}" "${size_bytes}" "${rep}" "${CPU}" "-" \
          "${input_sha256}" "${SUITE_SPEC_RESOLVED}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}" \
          "" "${entropy_bpb}" "${real_seconds}" "${user_seconds}" "${sys_seconds}" "${rss_kib}" "1"
      fi

      if [ "${need_compress}" -eq 1 ]; then
        run_timed compress "${input_path}" "${archive_path}" "${subject_kind}" "${spec_path}" "${h_order}" \
          "${stdout_path}" "${stderr_path}" "${time_path}"
        archive_ready=1
        archive_bytes=$(wc -c < "${archive_path}" | tr -d ' ')
        IFS="$(printf '\t')" read -r compress_real_seconds compress_user_seconds compress_sys_seconds compress_rss_kib <<EOF
$(parse_time_file "${time_path}")
EOF
        if [ "${need_decompress}" -eq 1 ]; then
          run_timed decompress "${archive_path}" "${restored_path}" "${subject_kind}" "${spec_path}" "${h_order}" \
            "${stdout_path}" "${stderr_path}" "${time_path}"
          cmp -s "${input_path}" "${restored_path}" || fail "Compression round-trip failed for ${subject} at ${size_bytes} bytes (rep ${rep})"
          IFS="$(printf '\t')" read -r decompress_real_seconds decompress_user_seconds decompress_sys_seconds decompress_rss_kib <<EOF
$(parse_time_file "${time_path}")
EOF
        else
          run_plain decompress "${archive_path}" "${restored_path}" "${subject_kind}" "${spec_path}" "${h_order}"
          cmp -s "${input_path}" "${restored_path}" || fail "Compression round-trip failed for ${subject} at ${size_bytes} bytes (rep ${rep})"
        fi
        rm -f "${restored_path}"
        append_row \
          "compress" "${subject}" "${subject_kind}" "${expert_kind}" "compress:${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" \
          "${input_sha256}" "${SUITE_SPEC_RESOLVED}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}" \
          "${archive_bytes}" "" "${compress_real_seconds}" "${compress_user_seconds}" "${compress_sys_seconds}" "${compress_rss_kib}" "1"
        if [ "${need_decompress}" -eq 1 ]; then
          append_row \
            "decompress" "${subject}" "${subject_kind}" "${expert_kind}" "decompress:${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" \
            "${input_sha256}" "${SUITE_SPEC_RESOLVED}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}" \
            "${archive_bytes}" "" "${decompress_real_seconds}" "${decompress_user_seconds}" "${decompress_sys_seconds}" "${decompress_rss_kib}" "1"
          need_decompress=0
        fi
      fi

      if [ "${need_decompress}" -eq 1 ]; then
        if [ "${archive_ready}" -eq 0 ]; then
          run_plain compress "${input_path}" "${archive_path}" "${subject_kind}" "${spec_path}" "${h_order}"
          archive_ready=1
        fi
        run_timed decompress "${archive_path}" "${restored_path}" "${subject_kind}" "${spec_path}" "${h_order}" \
          "${stdout_path}" "${stderr_path}" "${time_path}"
        cmp -s "${input_path}" "${restored_path}" || fail "Decompression validation failed for ${subject} at ${size_bytes} bytes (rep ${rep})"
        archive_bytes=$(wc -c < "${archive_path}" | tr -d ' ')
        IFS="$(printf '\t')" read -r real_seconds user_seconds sys_seconds rss_kib <<EOF
$(parse_time_file "${time_path}")
EOF
        append_row \
          "decompress" "${subject}" "${subject_kind}" "${expert_kind}" "decompress:${subject}" "${size_bytes}" "${rep}" "${CPU}" "${COMP_BACKEND}" \
          "${input_sha256}" "${SUITE_SPEC_RESOLVED}" "${SUITE_SPEC_SHA256}" "${BENCH_BUILD_MODE}" "${BENCH_FEATURES}" \
          "${archive_bytes}" "" "${real_seconds}" "${user_seconds}" "${sys_seconds}" "${rss_kib}" "1"
      fi
      rm -f "${archive_path}" "${restored_path}"

      rep=$((rep + 1))
    done
  done < "${SUBJECTS_TSV}"
done

python3 - "${RAW_TSV}" "${SUMMARY_TSV}" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

raw_path = sys.argv[1]
summary_path = sys.argv[2]

def parse_float(value):
    if value == "":
        return None
    return float(value)

def parse_int(value):
    if value == "":
        return None
    return int(value)

def mean(values):
    return statistics.fmean(values) if values else None

def stdev(values):
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)

def median(values):
    if not values:
        return None
    return statistics.median(values)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.12g}"
        return ""
    return str(value)

groups = defaultdict(list)
with open(raw_path, newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        row["size_bytes"] = int(row["size_bytes"])
        row["repetition"] = int(row["repetition"])
        row["real_seconds"] = float(row["real_seconds"])
        row["user_seconds"] = float(row["user_seconds"])
        row["sys_seconds"] = float(row["sys_seconds"])
        row["rss_kib"] = int(row["rss_kib"])
        row["archive_bytes"] = parse_int(row["archive_bytes"])
        row["entropy_bpb"] = parse_float(row["entropy_bpb"])
        row["verified"] = int(row["verified"])
        groups[
            (
                row["operation"],
                row["subject"],
                row["subject_kind"],
                row["expert_kind"],
                row["series"],
                row["size_bytes"],
                row["cpu"],
                row["compression_backend"],
                row["input_sha256"],
                row["suite_spec_path"],
                row["suite_spec_sha256"],
                row["build_mode"],
                row["build_features"],
            )
        ].append(row)

fieldnames = [
    "operation",
    "subject",
    "subject_kind",
    "expert_kind",
    "series",
    "size_bytes",
    "repeats",
    "cpu",
    "compression_backend",
    "input_sha256",
    "suite_spec_path",
    "suite_spec_sha256",
    "build_mode",
    "build_features",
    "real_seconds_mean",
    "real_seconds_stdev",
    "real_seconds_median",
    "real_seconds_min",
    "real_seconds_max",
    "user_seconds_mean",
    "sys_seconds_mean",
    "throughput_mib_s_mean",
    "throughput_mib_s_median",
    "rss_kib_mean",
    "rss_kib_stdev",
    "rss_kib_median",
    "rss_kib_min",
    "rss_kib_max",
    "archive_bytes_mean",
    "archive_bytes_median",
    "archive_ratio_mean",
    "archive_ratio_median",
    "entropy_bpb_mean",
    "entropy_bpb_median",
    "verified_all",
]

op_order = {"h": 0, "compress": 1, "decompress": 2}
sorted_keys = sorted(
    groups,
    key=lambda key: (
        op_order.get(key[0], 99),
        key[1],
        key[5],
    ),
)

with open(summary_path, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for key in sorted_keys:
        rows = groups[key]
        operation, subject, subject_kind, expert_kind, series, size_bytes, cpu, compression_backend, input_sha256, suite_spec_path, suite_spec_sha256, build_mode, build_features = key
        real = [row["real_seconds"] for row in rows]
        user = [row["user_seconds"] for row in rows]
        sysc = [row["sys_seconds"] for row in rows]
        rss = [float(row["rss_kib"]) for row in rows]
        archive = [float(row["archive_bytes"]) for row in rows if row["archive_bytes"] is not None]
        entropy = [row["entropy_bpb"] for row in rows if row["entropy_bpb"] is not None]
        mib = size_bytes / (1024.0 * 1024.0)
        throughput = [mib / value if value > 0.0 else float("inf") for value in real]
        archive_ratio = [value / size_bytes for value in archive]

        writer.writerow(
            {
                "operation": operation,
                "subject": subject,
                "subject_kind": subject_kind,
                "expert_kind": expert_kind,
                "series": series,
                "size_bytes": size_bytes,
                "repeats": len(rows),
                "cpu": cpu,
                "compression_backend": compression_backend,
                "input_sha256": input_sha256,
                "suite_spec_path": suite_spec_path,
                "suite_spec_sha256": suite_spec_sha256,
                "build_mode": build_mode,
                "build_features": build_features,
                "real_seconds_mean": fmt(mean(real)),
                "real_seconds_stdev": fmt(stdev(real)),
                "real_seconds_median": fmt(median(real)),
                "real_seconds_min": fmt(min(real)),
                "real_seconds_max": fmt(max(real)),
                "user_seconds_mean": fmt(mean(user)),
                "sys_seconds_mean": fmt(mean(sysc)),
                "throughput_mib_s_mean": fmt(mean(throughput)),
                "throughput_mib_s_median": fmt(median(throughput)),
                "rss_kib_mean": fmt(mean(rss)),
                "rss_kib_stdev": fmt(stdev(rss)),
                "rss_kib_median": fmt(median(rss)),
                "rss_kib_min": fmt(min(rss)),
                "rss_kib_max": fmt(max(rss)),
                "archive_bytes_mean": fmt(mean(archive)),
                "archive_bytes_median": fmt(median(archive)),
                "archive_ratio_mean": fmt(mean(archive_ratio)),
                "archive_ratio_median": fmt(median(archive_ratio)),
                "entropy_bpb_mean": fmt(mean(entropy)),
                "entropy_bpb_median": fmt(median(entropy)),
                "verified_all": "1" if all(row["verified"] == 1 for row in rows) else "0",
            }
        )
PY

say "[bench] Raw TSV: ${RAW_TSV}"
say "[bench] Summary TSV: ${SUMMARY_TSV}"
if [ "${BENCH_SUITE}" = "two-json" ]; then
  CURRENT_BASELINE_TSV=$(current_two_json_baseline_tsv)
  if [ -n "${CURRENT_BASELINE_TSV}" ]; then
    say "[bench] Compare against the checked-in baseline:"
    say "  '${ROOT_DIR}/scripts/compare_bench_two_json.lua' --baseline '${CURRENT_BASELINE_TSV}' '${SUMMARY_TSV}'"
  else
    say "[bench] No checked-in two-json baseline summary found under benchmarks/current."
  fi
else
  say "[bench] No checked-in baseline comparator is configured for suite '${BENCH_SUITE}'."
fi
say "[bench] Plot and inspect the results via benchman TUI instead of legacy Kuva commands:"
say "  '${ROOT_DIR}/projman.sh' tui '${BENCH_SUITE}' --summary-tsv '${SUMMARY_TSV}'"
