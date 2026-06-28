#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/bench_cli_hyperfine.sh <baseline-commit> [preset]
       ./scripts/bench_cli_hyperfine.sh --plan [preset]

Builds a detached baseline tree and a snapshot of the current dirty working tree,
then compares CLI workloads with hyperfine.

The benchmark runs two full passes to reduce order effects:
  1) baseline then current
  2) current then baseline

Environment:
  INFOTHEORY_CLI_BENCH_ROOT=/var/tmp/infotheory_bench
  INFOTHEORY_CLI_BENCH_PROFILE=release
  INFOTHEORY_CLI_BENCH_RUNS=<preset default>
  INFOTHEORY_CLI_BENCH_WARMUPS=<preset default>
  INFOTHEORY_CLI_BENCH_BYTES=<preset default>
  INFOTHEORY_CLI_BENCH_BUILD_MODE=native|portable
  INFOTHEORY_CLI_BENCH_CARGO_FEATURES=cli
  INFOTHEORY_CLI_BENCH_BASELINE_FEATURES=...
  INFOTHEORY_CLI_BENCH_CURRENT_FEATURES=...
  INFOTHEORY_CLI_BENCH_NO_DEFAULT_FEATURES=0|1

Presets:
  default  Full matrix, tuned for signal quality (runs=10, warmups=3, bytes=32768)
  quick    Full matrix with faster defaults (runs=5, warmups=1, bytes=16384)

Modes:
  --plan   Print the expanded workload plan and effective tuning without running builds
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

fail() {
  echo "error: $*" >&2
  exit 1
}

sanitize_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

join_by() {
  local sep="$1"
  shift || true
  local out=""
  local item
  for item in "$@"; do
    if [[ -n "${out}" ]]; then
      out+="${sep}"
    fi
    out+="${item}"
  done
  printf '%s' "${out}"
}

require_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    fail "${name} must be a non-negative integer (got '${value}')"
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  require_non_negative_int "${name}" "${value}"
  if [[ "${value}" -le 0 ]]; then
    fail "${name} must be greater than zero (got '${value}')"
  fi
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

resolve_bench_tuning() {
  local preset="$1"
  local preset_runs
  local preset_warmups
  local preset_bytes

  case "${preset}" in
    default)
      preset_runs=10
      preset_warmups=3
      preset_bytes=32768
      ;;
    quick)
      preset_runs=5
      preset_warmups=1
      preset_bytes=16384
      ;;
    *)
      fail "unknown preset '${preset}' (expected default or quick)"
      ;;
  esac

  BENCH_RUNS="${INFOTHEORY_CLI_BENCH_RUNS:-${preset_runs}}"
  BENCH_WARMUPS="${INFOTHEORY_CLI_BENCH_WARMUPS:-${preset_warmups}}"
  BENCH_BYTES="${INFOTHEORY_CLI_BENCH_BYTES:-${preset_bytes}}"

  require_positive_int "INFOTHEORY_CLI_BENCH_RUNS" "${BENCH_RUNS}"
  require_non_negative_int "INFOTHEORY_CLI_BENCH_WARMUPS" "${BENCH_WARMUPS}"
  require_positive_int "INFOTHEORY_CLI_BENCH_BYTES" "${BENCH_BYTES}"
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
      "${repo_root}/LICENSE-APACHE" \
      "${repo_root}/benchmarks/baseline/infotheory-two-json-summary-20260310-212017.tsv" \
      "${bench_two}" >> "${out_path}"
  done
  truncate -s "${bytes}" "${out_path}"
  printf '%s\n' "${out_path}"
}

write_calibrated_spec() {
  local out_path="$1"
  mkdir -p "$(dirname "${out_path}")"
  cat >"${out_path}" <<'EOF'
{
  "kind": "calibrated",
  "context": "text",
  "bins": 33,
  "learning_rate": 0.02,
  "bias_clip": 4.0,
  "base": {
    "kind": "match"
  }
}
EOF
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

format_command() {
  local quoted=()
  local token
  for token in "$@"; do
    printf -v token '%q' "${token}"
    quoted+=("${token}")
  done
  (IFS=' '; printf '%s' "${quoted[*]}")
}

run_command_checked() {
  local label="$1"
  local stage="$2"
  local command_line="$3"
  local log_dir="$4"
  local log_path

  log_path="${log_dir}/$(sanitize_name "${label}_${stage}").log"
  if ! bash -c "${command_line}" >"${log_path}" 2>&1; then
    echo "error: ${stage} failed for case '${label}'" >&2
    echo "command: ${command_line}" >&2
    if [[ -s "${log_path}" ]]; then
      sed -n '1,120p' "${log_path}" >&2
    fi
    exit 1
  fi
}

run_hyperfine_case() {
  local out_dir="$1"
  local label="$2"
  local order="$3"
  local first_name="$4"
  local second_name="$5"
  local first_cmd="$6"
  local second_cmd="$7"
  local runs="$8"
  local warmups="$9"
  local log_dir="${10}"
  local json_path="${out_dir}/${label}__${order}.json"
  local log_path

  log_path="${log_dir}/$(sanitize_name "hyperfine_${label}_${order}").log"
  if ! hyperfine \
    --shell=none \
    --style none \
    --warmup "${warmups}" \
    --runs "${runs}" \
    --export-json "${json_path}" \
    --command-name "${first_name}" \
    --command-name "${second_name}" \
    "${first_cmd}" \
    "${second_cmd}" >"${log_path}" 2>&1; then
    echo "error: hyperfine failed for case '${label}' (order: ${order})" >&2
    echo "first (${first_name}): ${first_cmd}" >&2
    echo "second (${second_name}): ${second_cmd}" >&2
    if [[ -s "${log_path}" ]]; then
      sed -n '1,120p' "${log_path}" >&2
    fi
    exit 1
  fi

  [[ -s "${json_path}" ]] || fail "missing hyperfine JSON output for case '${label}' (${order})"
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


def stats(values: list[float]) -> tuple[int, float, float, float, float]:
    n = len(values)
    if n == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    mean = sum(values) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
    else:
        var = 0.0
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return n, mean, std, sem, var


def fmt(value: float, digits: int = 9) -> str:
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    if value > 0:
        return "inf"
    if value < 0:
        return "-inf"
    return "nan"


samples: dict[str, dict[str, list[float]]] = {}

for path in sorted(cases_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    label = path.stem.split("__", 1)[0]
    bucket = samples.setdefault(label, {"baseline": [], "current": []})
    for result in data.get("results", []):
        command_name = str(result.get("command", "")).strip().lower()
        if command_name not in bucket:
            continue
        times = result.get("times")
        if isinstance(times, list) and times:
            for value in times:
                try:
                    bucket[command_name].append(float(value))
                except (TypeError, ValueError):
                    continue
            continue
        try:
            bucket[command_name].append(float(result["mean"]))
        except (KeyError, TypeError, ValueError):
            continue

if not samples:
    raise SystemExit("no hyperfine case JSON files found")

with summary_tsv.open("w", encoding="utf-8") as f:
    f.write(
        "label\tbaseline_mean_s\tbaseline_stddev_s\tcurrent_mean_s\tcurrent_stddev_s"
        "\tratio_current_over_baseline\tbaseline_n\tcurrent_n\tbaseline_sem_s"
        "\tcurrent_sem_s\tdelta_s\tse_delta_s\tt_like\tci95_ratio_low"
        "\tci95_ratio_high\tpooled_residual_var_s2\tresidual_bits_gaussian\n"
    )

    for label in sorted(samples):
        base_vals = samples[label]["baseline"]
        current_vals = samples[label]["current"]
        if not base_vals or not current_vals:
            raise SystemExit(f"missing baseline/current timing samples for case '{label}'")

        b_n, b_mean, b_std, b_sem, _ = stats(base_vals)
        c_n, c_mean, c_std, c_sem, _ = stats(current_vals)

        ratio = math.inf if b_mean == 0 else c_mean / b_mean
        delta = c_mean - b_mean
        se_delta = math.sqrt((b_sem * b_sem) + (c_sem * c_sem))
        t_like = math.inf if se_delta == 0 else abs(delta) / se_delta

        if (
            b_mean > 0
            and c_mean > 0
            and math.isfinite(b_sem)
            and math.isfinite(c_sem)
        ):
            log_ratio = math.log(c_mean / b_mean)
            se_log_ratio = math.sqrt((b_sem / b_mean) ** 2 + (c_sem / c_mean) ** 2)
            ci95_low = math.exp(log_ratio - 1.96 * se_log_ratio)
            ci95_high = math.exp(log_ratio + 1.96 * se_log_ratio)
        else:
            ci95_low = math.nan
            ci95_high = math.nan

        pooled_denom = (b_n - 1) + (c_n - 1)
        pooled_var = (
            (sum((x - b_mean) ** 2 for x in base_vals) + sum((x - c_mean) ** 2 for x in current_vals))
            / pooled_denom
            if pooled_denom > 0
            else 0.0
        )
        if pooled_var > 0:
            residual_bits = 0.5 * math.log2(2.0 * math.pi * math.e * pooled_var)
        else:
            residual_bits = float("-inf")

        f.write(
            "\t".join(
                [
                    label,
                    fmt(b_mean),
                    fmt(b_std),
                    fmt(c_mean),
                    fmt(c_std),
                    fmt(ratio, 6),
                    str(b_n),
                    str(c_n),
                    fmt(b_sem),
                    fmt(c_sem),
                    fmt(delta),
                    fmt(se_delta),
                    fmt(t_like, 6),
                    fmt(ci95_low, 6),
                    fmt(ci95_high, 6),
                    fmt(pooled_var),
                    fmt(residual_bits, 6),
                ]
            )
            + "\n"
        )
PY
}

add_case() {
  CASE_LABELS+=("$1")
  CASE_BASELINE_COMMANDS+=("$2")
  CASE_CURRENT_COMMANDS+=("$3")
}

add_roundtrip_case() {
  ROUNDTRIP_LABELS+=("$1")
  ROUNDTRIP_INPUT_PATHS+=("$2")
  ROUNDTRIP_BASELINE_COMPRESS_COMMANDS+=("$3")
  ROUNDTRIP_CURRENT_COMPRESS_COMMANDS+=("$4")
  ROUNDTRIP_BASELINE_DECOMPRESS_COMMANDS+=("$5")
  ROUNDTRIP_CURRENT_DECOMPRESS_COMMANDS+=("$6")
  ROUNDTRIP_BASELINE_VERIFY_OUTPUTS+=("$7")
  ROUNDTRIP_CURRENT_VERIFY_OUTPUTS+=("$8")
}

add_rate_spec() {
  RATE_SPEC_LABELS+=("$1")
  RATE_SPEC_BACKENDS+=("$2")
  RATE_SPEC_METHODS+=("$3")
}

build_rate_backend_specs() {
  local run_root="$1"
  local rwkv_cfg="cfg:hidden=64,layers=1,intermediate=64,decay_rank=16,a_rank=16,v_rank=16,g_rank=16,seed=22,train=adam,lr=0.0009,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=1.0,momentum=0.9)"
  local mamba_cfg="cfg:hidden=64,layers=1,intermediate=128,state=16,conv=4,dt_rank=16,seed=26,train=adam,lr=0.001,stride=1;policy:schedule=0..100:train(scope=all,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)"
  local two_json
  local particle_fast_json
  local calibrated_json

  two_json="$(bench_config_path "${SCRIPT_REPO_ROOT}" "two.json")"
  particle_fast_json="$(bench_config_path "${SCRIPT_REPO_ROOT}" "particle_fast.json")"
  calibrated_json="${run_root}/specs/calibrated_match.json"
  write_calibrated_spec "${calibrated_json}"

  RATE_SPEC_LABELS=()
  RATE_SPEC_BACKENDS=()
  RATE_SPEC_METHODS=()

  add_rate_spec "rosaplus" "rosaplus" ""
  add_rate_spec "ctw" "ctw" "32"
  add_rate_spec "fac_ctw" "fac-ctw" "16"
  add_rate_spec "match" "match" ""
  add_rate_spec "sparse_match" "sparse-match" ""
  add_rate_spec "ppmd" "ppmd" "10"
  add_rate_spec "sequitur" "sequitur" "64"
  add_rate_spec "calibrated" "calibrated" "${calibrated_json}"
  add_rate_spec "mixture" "mixture" "${two_json}"
  add_rate_spec "particle" "particle" "${particle_fast_json}"
  add_rate_spec "mamba_cfg" "mamba" "${mamba_cfg}"
  add_rate_spec "rwkv7_cfg" "rwkv7" "${rwkv_cfg}"
}

build_rate_backend_flags() {
  local idx="$1"
  RATE_BACKEND_FLAGS=(--rate-backend "${RATE_SPEC_BACKENDS[$idx]}")
  if [[ -n "${RATE_SPEC_METHODS[$idx]}" ]]; then
    RATE_BACKEND_FLAGS+=(--method "${RATE_SPEC_METHODS[$idx]}")
  fi
}

find_rate_spec_index() {
  local label="$1"
  local i
  for ((i = 0; i < ${#RATE_SPEC_LABELS[@]}; i++)); do
    if [[ "${RATE_SPEC_LABELS[$i]}" == "${label}" ]]; then
      printf '%s\n' "${i}"
      return 0
    fi
  done
  return 1
}

build_cases() {
  local baseline_bin="$1"
  local current_bin="$2"
  local input_path="$3"
  local run_root="$4"
  local i

  mkdir -p \
    "${run_root}/prepared/baseline" \
    "${run_root}/prepared/current" \
    "${run_root}/bench-outputs/baseline" \
    "${run_root}/bench-outputs/current"

  build_rate_backend_specs "${run_root}"

  CASE_LABELS=()
  CASE_BASELINE_COMMANDS=()
  CASE_CURRENT_COMMANDS=()
  ROUNDTRIP_LABELS=()
  ROUNDTRIP_INPUT_PATHS=()
  ROUNDTRIP_BASELINE_COMPRESS_COMMANDS=()
  ROUNDTRIP_CURRENT_COMPRESS_COMMANDS=()
  ROUNDTRIP_BASELINE_DECOMPRESS_COMMANDS=()
  ROUNDTRIP_CURRENT_DECOMPRESS_COMMANDS=()
  ROUNDTRIP_BASELINE_VERIFY_OUTPUTS=()
  ROUNDTRIP_CURRENT_VERIFY_OUTPUTS=()

  for ((i = 0; i < ${#RATE_SPEC_LABELS[@]}; i++)); do
    local rate_label="${RATE_SPEC_LABELS[$i]}"
    build_rate_backend_flags "${i}"

    local baseline_h_cmd
    local current_h_cmd
    baseline_h_cmd="$(format_command "${baseline_bin}" h "${input_path}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
    current_h_cmd="$(format_command "${current_bin}" h "${input_path}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
    add_case "h_${rate_label}" "${baseline_h_cmd}" "${current_h_cmd}"
  done

  for ((i = 0; i < ${#RATE_SPEC_LABELS[@]}; i++)); do
    local rate_label="${RATE_SPEC_LABELS[$i]}"
    build_rate_backend_flags "${i}"

    local prepared_baseline_comp="${run_root}/prepared/baseline/${rate_label}_ac.itc"
    local prepared_current_comp="${run_root}/prepared/current/${rate_label}_ac.itc"
    local verify_baseline_out="${run_root}/prepared/baseline/${rate_label}_ac.roundtrip.bin"
    local verify_current_out="${run_root}/prepared/current/${rate_label}_ac.roundtrip.bin"

    local prep_baseline_compress
    local prep_current_compress
    local prep_baseline_decompress
    local prep_current_decompress

    prep_baseline_compress="$(format_command "${baseline_bin}" compress "${input_path}" "${prepared_baseline_comp}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
    prep_current_compress="$(format_command "${current_bin}" compress "${input_path}" "${prepared_current_comp}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
    prep_baseline_decompress="$(format_command "${baseline_bin}" decompress "${prepared_baseline_comp}" "${verify_baseline_out}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
    prep_current_decompress="$(format_command "${current_bin}" decompress "${prepared_current_comp}" "${verify_current_out}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"

    add_roundtrip_case \
      "rate_ac_${rate_label}" \
      "${input_path}" \
      "${prep_baseline_compress}" \
      "${prep_current_compress}" \
      "${prep_baseline_decompress}" \
      "${prep_current_decompress}" \
      "${verify_baseline_out}" \
      "${verify_current_out}"

    local bench_baseline_comp="${run_root}/bench-outputs/baseline/${rate_label}_ac.itc"
    local bench_current_comp="${run_root}/bench-outputs/current/${rate_label}_ac.itc"
    local bench_baseline_dec="${run_root}/bench-outputs/baseline/${rate_label}_ac.decompressed.bin"
    local bench_current_dec="${run_root}/bench-outputs/current/${rate_label}_ac.decompressed.bin"

    add_case \
      "compress_rate_ac_${rate_label}" \
      "$(format_command "${baseline_bin}" compress "${input_path}" "${bench_baseline_comp}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")" \
      "$(format_command "${current_bin}" compress "${input_path}" "${bench_current_comp}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"

    add_case \
      "decompress_rate_ac_${rate_label}" \
      "$(format_command "${baseline_bin}" decompress "${prepared_baseline_comp}" "${bench_baseline_dec}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")" \
      "$(format_command "${current_bin}" decompress "${prepared_current_comp}" "${bench_current_dec}" --compression-backend rate-ac "${RATE_BACKEND_FLAGS[@]}")"
  done

  local rans_label
  for rans_label in ctw ppmd; do
    local idx
    idx="$(find_rate_spec_index "${rans_label}")" || fail "missing rate spec label '${rans_label}'"
    build_rate_backend_flags "${idx}"

    local prepared_baseline_comp="${run_root}/prepared/baseline/${rans_label}_rans.itc"
    local prepared_current_comp="${run_root}/prepared/current/${rans_label}_rans.itc"
    local verify_baseline_out="${run_root}/prepared/baseline/${rans_label}_rans.roundtrip.bin"
    local verify_current_out="${run_root}/prepared/current/${rans_label}_rans.roundtrip.bin"

    local prep_baseline_compress
    local prep_current_compress
    local prep_baseline_decompress
    local prep_current_decompress

    prep_baseline_compress="$(format_command "${baseline_bin}" compress "${input_path}" "${prepared_baseline_comp}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"
    prep_current_compress="$(format_command "${current_bin}" compress "${input_path}" "${prepared_current_comp}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"
    prep_baseline_decompress="$(format_command "${baseline_bin}" decompress "${prepared_baseline_comp}" "${verify_baseline_out}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"
    prep_current_decompress="$(format_command "${current_bin}" decompress "${prepared_current_comp}" "${verify_current_out}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"

    add_roundtrip_case \
      "rate_rans_${rans_label}" \
      "${input_path}" \
      "${prep_baseline_compress}" \
      "${prep_current_compress}" \
      "${prep_baseline_decompress}" \
      "${prep_current_decompress}" \
      "${verify_baseline_out}" \
      "${verify_current_out}"

    local bench_baseline_comp="${run_root}/bench-outputs/baseline/${rans_label}_rans.itc"
    local bench_current_comp="${run_root}/bench-outputs/current/${rans_label}_rans.itc"
    local bench_baseline_dec="${run_root}/bench-outputs/baseline/${rans_label}_rans.decompressed.bin"
    local bench_current_dec="${run_root}/bench-outputs/current/${rans_label}_rans.decompressed.bin"

    add_case \
      "compress_rate_rans_${rans_label}" \
      "$(format_command "${baseline_bin}" compress "${input_path}" "${bench_baseline_comp}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")" \
      "$(format_command "${current_bin}" compress "${input_path}" "${bench_current_comp}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"

    add_case \
      "decompress_rate_rans_${rans_label}" \
      "$(format_command "${baseline_bin}" decompress "${prepared_baseline_comp}" "${bench_baseline_dec}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")" \
      "$(format_command "${current_bin}" decompress "${prepared_current_comp}" "${bench_current_dec}" --compression-backend rate-rans "${RATE_BACKEND_FLAGS[@]}")"
  done
}

prepare_roundtrip_artifacts() {
  local roundtrip_tsv="$1"
  local log_dir="$2"
  local i

  printf 'label\tsubject\tstatus\n' >"${roundtrip_tsv}"

  for ((i = 0; i < ${#ROUNDTRIP_LABELS[@]}; i++)); do
    local label="${ROUNDTRIP_LABELS[$i]}"
    local input_path="${ROUNDTRIP_INPUT_PATHS[$i]}"

    run_command_checked "${label}" "baseline_compress_preflight" "${ROUNDTRIP_BASELINE_COMPRESS_COMMANDS[$i]}" "${log_dir}"
    run_command_checked "${label}" "baseline_decompress_preflight" "${ROUNDTRIP_BASELINE_DECOMPRESS_COMMANDS[$i]}" "${log_dir}"
    if ! cmp -s "${input_path}" "${ROUNDTRIP_BASELINE_VERIFY_OUTPUTS[$i]}"; then
      fail "roundtrip verification failed for baseline case '${label}'"
    fi
    printf '%s\tbaseline\tpass\n' "${label}" >>"${roundtrip_tsv}"

    run_command_checked "${label}" "current_compress_preflight" "${ROUNDTRIP_CURRENT_COMPRESS_COMMANDS[$i]}" "${log_dir}"
    run_command_checked "${label}" "current_decompress_preflight" "${ROUNDTRIP_CURRENT_DECOMPRESS_COMMANDS[$i]}" "${log_dir}"
    if ! cmp -s "${input_path}" "${ROUNDTRIP_CURRENT_VERIFY_OUTPUTS[$i]}"; then
      fail "roundtrip verification failed for current case '${label}'"
    fi
    printf '%s\tcurrent\tpass\n' "${label}" >>"${roundtrip_tsv}"
  done
}

print_plan() {
  local preset="$1"
  local plan_root
  plan_root="$(mktemp -d)"
  trap 'rm -rf "${plan_root}"' RETURN

  build_cases "/tmp/infotheory-bench-baseline" "/tmp/infotheory-bench-current" "/tmp/infotheory-bench-input.bin" "${plan_root}"

  printf 'PLAN\tpreset\t%s\n' "${preset}"
  printf 'PLAN\truns\t%s\n' "${BENCH_RUNS}"
  printf 'PLAN\twarmups\t%s\n' "${BENCH_WARMUPS}"
  printf 'PLAN\tbytes\t%s\n' "${BENCH_BYTES}"
  printf 'PLAN\trate_backends\t%s\n' "$(join_by , "${RATE_SPEC_BACKENDS[@]}")"
  printf 'PLAN\tcases\t%s\n' "${#CASE_LABELS[@]}"
  printf 'PLAN\troundtrip_cases\t%s\n' "${#ROUNDTRIP_LABELS[@]}"

  local label
  for label in "${CASE_LABELS[@]}"; do
    printf 'CASE\t%s\n' "${label}"
  done
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local mode="run"
  local baseline_commit=""
  local preset="default"

  if [[ "${1:-}" == "--plan" ]]; then
    mode="plan"
    preset="${2:-default}"
  else
    [[ $# -ge 1 ]] || {
      usage >&2
      exit 1
    }
    baseline_commit="$1"
    preset="${2:-default}"
  fi

  repo_root="$(cd "$(dirname "$0")/.." && pwd)"
  SCRIPT_REPO_ROOT="${repo_root}"
  resolve_bench_tuning "${preset}"

  if [[ "${mode}" == "plan" ]]; then
    print_plan "${preset}"
    exit 0
  fi

  need_cmd bash
  need_cmd cargo
  need_cmd git
  need_cmd tar
  need_cmd python3
  need_cmd cmp

  if ! command -v hyperfine >/dev/null 2>&1; then
    echo "hyperfine is required for CLI benchmarking." >&2
    echo "Install it with: cargo install --locked hyperfine" >&2
    exit 1
  fi

  local bench_root="${INFOTHEORY_CLI_BENCH_ROOT:-/var/tmp/infotheory_bench}"
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local run_root="${bench_root}/${stamp}"
  local baseline_root="${run_root}/baseline-src"
  local current_root="${run_root}/current-src"
  local baseline_build="${run_root}/baseline-build"
  local current_build="${run_root}/current-build"
  local cases_dir="${run_root}/cases"
  local log_dir="${run_root}/logs"
  mkdir -p "${cases_dir}" "${log_dir}"

  local rustflags
  rustflags="$(build_rustflags)"
  local shared_features="${INFOTHEORY_CLI_BENCH_CARGO_FEATURES:-cli}"
  local baseline_features="${INFOTHEORY_CLI_BENCH_BASELINE_FEATURES:-${shared_features}}"
  local current_features="${INFOTHEORY_CLI_BENCH_CURRENT_FEATURES:-${shared_features}}"

  git worktree add --detach "${baseline_root}" "${baseline_commit}" >/dev/null
  trap 'git worktree remove --force "${baseline_root}" >/dev/null 2>&1 || true' EXIT
  git -C "${baseline_root}" submodule update --init --recursive >/dev/null
  copy_dirty_tree "${repo_root}" "${current_root}" "${bench_root}"

  local baseline_bin
  local current_bin
  baseline_bin="$(build_infotheory_bin "${baseline_root}" "${baseline_build}" "${rustflags}" "${baseline_features}")"
  current_bin="$(build_infotheory_bin "${current_root}" "${current_build}" "${rustflags}" "${current_features}")"

  # Allow clocks and thermal state to settle immediately after compilation.
  sleep 5

  local input_path
  input_path="$(prepare_input "${SCRIPT_REPO_ROOT}" "${BENCH_BYTES}" "${run_root}/inputs")"

  build_cases "${baseline_bin}" "${current_bin}" "${input_path}" "${run_root}"

  local roundtrip_tsv="${run_root}/roundtrip.tsv"
  prepare_roundtrip_artifacts "${roundtrip_tsv}" "${log_dir}"

  echo "Benchmark preset: ${preset}"
  echo "Benchmark tuning: runs=${BENCH_RUNS}, warmups=${BENCH_WARMUPS}, bytes=${BENCH_BYTES}"
  echo "Cases: ${#CASE_LABELS[@]}"
  echo "Roundtrip preflight cases: ${#ROUNDTRIP_LABELS[@]}"

  local order
  local i
  for order in baseline-current current-baseline; do
    echo "Running order: ${order}"
    for ((i = 0; i < ${#CASE_LABELS[@]}; i++)); do
      local first_name
      local second_name
      local first_cmd
      local second_cmd

      if [[ "${order}" == "baseline-current" ]]; then
        first_name="baseline"
        second_name="current"
        first_cmd="${CASE_BASELINE_COMMANDS[$i]}"
        second_cmd="${CASE_CURRENT_COMMANDS[$i]}"
      else
        first_name="current"
        second_name="baseline"
        first_cmd="${CASE_CURRENT_COMMANDS[$i]}"
        second_cmd="${CASE_BASELINE_COMMANDS[$i]}"
      fi

      run_hyperfine_case \
        "${cases_dir}" \
        "${CASE_LABELS[$i]}" \
        "${order}" \
        "${first_name}" \
        "${second_name}" \
        "${first_cmd}" \
        "${second_cmd}" \
        "${BENCH_RUNS}" \
        "${BENCH_WARMUPS}" \
        "${log_dir}"
    done

    if [[ "${order}" == "baseline-current" ]]; then
      # Reduce drift from immediate back-to-back pass ordering.
      sleep 5
    fi
  done

  local summary_tsv="${run_root}/summary.tsv"
  append_summary_rows "${cases_dir}" "${summary_tsv}"

  echo "CLI benchmark comparison complete."
  echo "Run root: ${run_root}"
  echo "Summary TSV: ${summary_tsv}"
  echo "Roundtrip TSV: ${roundtrip_tsv}"
}

main "$@"
