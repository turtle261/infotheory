#!/usr/bin/env bash
set -euo pipefail

# Reproducible three-mode tuner comparison anchored to examples/two.json.
# Modes: annealed_hill_climbing, mc_aixi_fac_ctw, aiqi_warmstart_exact_jh.
#
# Positional arguments (all optional):
#   1) input file path
#   2) per-evaluation time limit (seconds, > 0)
#   3) memory cap (GB, >= 1)
#
# Defaults:
#   - input: `git show HEAD:README.md` written to /tmp run directory
#   - eval_time_limit_seconds: 2
#   - max_memory_gb: 1
#   - strict RSS/accounting mode: hybrid_strict_max (Linux + delegated cgroup-v2 required)
#
# Environment overrides:
#   TUNER_MAX_EVALUATIONS      (default: 120)
#   TUNER_TIME_BUDGET_SECONDS  (default: 600)
#   TUNER_OUTPUT_ROOT          (default: /tmp/infotheory-tuner-two-json)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_FILE="${1:-}"
EVAL_TIME_LIMIT_SECONDS="${2:-2}"
MAX_MEMORY_GB="${3:-1}"
MAX_EVALUATIONS="${TUNER_MAX_EVALUATIONS:-120}"
TIME_BUDGET_SECONDS="${TUNER_TIME_BUDGET_SECONDS:-600}"
OUTPUT_ROOT="${TUNER_OUTPUT_ROOT:-/tmp/infotheory-tuner-two-json}"

# Keep this list explicit so runs are auditable and reproducible.
FEATURES="tuner cli backend-ctw backend-mixture backend-ppmd backend-rosa backend-match backend-rwkv"
SCALAR_REP="finite-ieee754-f64-nonfinite-forbidden-v1"
OBS_ADAPTER_REF="single-channel-conditional-byte-adapter-v1"
TWO_JSON_PATH="$REPO_ROOT/examples/two.json"
STRICT_RSS_MODE="hybrid_strict_max"
EVAL_CGROUP_PARENT="${INFOTHEORY_TUNER_EVAL_CGROUP_PARENT:-}"

die() {
    echo "Error: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command '$1' is not available"
}

need_cmd cargo
need_cmd git
need_cmd python3

if [ "$(uname -s)" != "Linux" ]; then
    die "strict theorem-facing benchmark requires Linux (requested rss mode: $STRICT_RSS_MODE)"
fi
if [ -z "$EVAL_CGROUP_PARENT" ]; then
    die "INFOTHEORY_TUNER_EVAL_CGROUP_PARENT is required for strict theorem-facing runs"
fi
if [ ! -d "$EVAL_CGROUP_PARENT" ]; then
    die "delegated cgroup parent does not exist: $EVAL_CGROUP_PARENT"
fi
if [ -f "$EVAL_CGROUP_PARENT/cgroup.subtree_control" ] && ! grep -Eq '(^|[[:space:]])memory($|[[:space:]])' "$EVAL_CGROUP_PARENT/cgroup.subtree_control"; then
    die "delegated cgroup parent must have memory enabled in cgroup.subtree_control: $EVAL_CGROUP_PARENT"
fi

python3 - "$EVAL_TIME_LIMIT_SECONDS" "$MAX_MEMORY_GB" "$MAX_EVALUATIONS" "$TIME_BUDGET_SECONDS" <<'PY'
import sys

eval_limit = float(sys.argv[1])
mem_gb = float(sys.argv[2])
max_evals = int(sys.argv[3])
time_budget = float(sys.argv[4])
if not (eval_limit > 0.0):
    raise SystemExit("per-evaluation time limit must be > 0")
if not (mem_gb >= 1.0):
    raise SystemExit("memory cap (GB) must be >= 1")
if max_evals <= 0:
    raise SystemExit("TUNER_MAX_EVALUATIONS must be > 0")
if not (time_budget > 0.0):
    raise SystemExit("TUNER_TIME_BUDGET_SECONDS must be > 0")
PY

RUN_DIR="$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

SUBJECT_PATH="$RUN_DIR/subject.bin"
if [ -n "$INPUT_FILE" ]; then
    [ -f "$INPUT_FILE" ] || die "input file '$INPUT_FILE' does not exist"
    cp "$INPUT_FILE" "$SUBJECT_PATH"
    SUBJECT_SOURCE="$INPUT_FILE"
else
    git -C "$REPO_ROOT" show HEAD:README.md > "$SUBJECT_PATH"
    SUBJECT_SOURCE="git show HEAD:README.md"
fi

SUBJECT_BYTES="$(wc -c < "$SUBJECT_PATH" | tr -d '[:space:]')"
[ "$SUBJECT_BYTES" -gt 0 ] || die "subject file is empty: $SUBJECT_PATH"

MIN_THROUGHPUT="$(python3 - "$SUBJECT_BYTES" "$EVAL_TIME_LIMIT_SECONDS" <<'PY'
import sys
b = int(sys.argv[1])
t = float(sys.argv[2])
print(f"{b / t:.6f}")
PY
)"

MAX_MEMORY_BYTES="$(python3 - "$MAX_MEMORY_GB" <<'PY'
import sys
gb = float(sys.argv[1])
print(int(gb * (1024 ** 3)))
PY
)"

CANONICAL_BASELINE_RATE="$RUN_DIR/two-json-rate-backend-canonical.json"
ANNEALED_SPEC="$RUN_DIR/two-json-annealed-spec.json"
MCAIXI_SPEC="$RUN_DIR/two-json-mcaixi-spec.json"
WARMSTART_SPEC="$RUN_DIR/two-json-warmstart-spec.json"
WARMSTART_TEACHER="$RUN_DIR/warmstart-teacher.json"
MCAIXI_CERT="$RUN_DIR/mcaixi-exact-reward-cert.json"
WARMSTART_CERT="$RUN_DIR/warmstart-exact-reward-cert.json"

ANNEALED_OUTPUT="$RUN_DIR/annealed-output.json"
ANNEALED_REPORT="$RUN_DIR/annealed-report.json"
MCAIXI_OUTPUT="$RUN_DIR/mcaixi-output.json"
MCAIXI_REPORT="$RUN_DIR/mcaixi-report.json"
WARMSTART_OUTPUT="$RUN_DIR/warmstart-output.json"
WARMSTART_REPORT="$RUN_DIR/warmstart-report.json"
SUMMARY_JSON="$RUN_DIR/comparison-summary.json"
SUMMARY_TSV="$RUN_DIR/comparison-summary.tsv"
RUN_LOG="$RUN_DIR/benchmark.log"

canonicalize_two_json_rate_backend() {
    local input_path="$1"
    local output_path="$2"
    python3 - "$input_path" "$output_path" <<'PY'
import json
import pathlib
import struct
import sys

input_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
doc = json.loads(input_path.read_text())

if not isinstance(doc, dict):
    raise SystemExit("examples/two.json must be a JSON object")
if doc.get("kind") != "neural":
    raise SystemExit("examples/two.json kind must be 'neural'")
experts = doc.get("experts")
if not isinstance(experts, list) or len(experts) == 0:
    raise SystemExit("examples/two.json experts must be a non-empty array")


def parse_rwkv_cfg_string(raw: str):
    if not raw.startswith("cfg:"):
        raise SystemExit("rwkv7 method must start with 'cfg:' in examples/two.json")
    policy = None
    cfg_part = raw
    if ";policy:" in raw:
        cfg_part, policy = raw.split(";policy:", 1)
    cfg_fields = cfg_part[len("cfg:"):].split(",")
    parsed = {}
    for field in cfg_fields:
        key, sep, value = field.partition("=")
        if sep != "=":
            raise SystemExit(f"invalid rwkv7 cfg field '{field}'")
        parsed[key.strip()] = value.strip()

    def parse_int(key: str) -> int:
        if key not in parsed:
            raise SystemExit(f"rwkv7 cfg missing '{key}'")
        return int(parsed[key], 10)

    def parse_float(key: str) -> float:
        if key not in parsed:
            raise SystemExit(f"rwkv7 cfg missing '{key}'")
        value64 = float(parsed[key])
        packed = struct.pack("!f", value64)
        return struct.unpack("!f", packed)[0]

    train_raw = parsed.get("train", "none")
    train_mode = {
        "none": "none",
        "sgd": "sgd",
        "adam": "adam",
    }.get(train_raw)
    if train_mode is None:
        raise SystemExit(f"unsupported rwkv7 cfg train mode '{train_raw}'")

    cfg = {
        "hidden": parse_int("hidden"),
        "layers": parse_int("layers"),
        "intermediate": parse_int("intermediate"),
        "decay_rank": parse_int("decay_rank"),
        "a_rank": parse_int("a_rank"),
        "v_rank": parse_int("v_rank"),
        "g_rank": parse_int("g_rank"),
        "seed": parse_int("seed"),
        "train_mode": train_mode,
        "lr": parse_float("lr"),
        "stride": parse_int("stride"),
    }

    method = {
        "kind": "online",
        "cfg": cfg,
        "policy": policy if policy is not None and len(policy) > 0 else None,
    }
    return method

mixture_spec = {
    "kind": "neural",
    "schedule": "default",
    "alpha": float(doc.get("alpha", 0.01)),
    "decay": None,
    "experts": [],
}

for expert in experts:
    if not isinstance(expert, dict):
        raise SystemExit("each expert in examples/two.json must be an object")
    kind = expert.get("kind")
    if not isinstance(kind, str):
        raise SystemExit("each expert in examples/two.json must include string kind")

    out = {
        "kind": kind,
        "log_prior": float(expert.get("log_prior", expert.get("prior", 0.0))),
    }
    if isinstance(expert.get("name"), str):
        out["name"] = expert["name"]

    if kind == "ctw":
        out["depth"] = int(expert.get("depth", 16))
    elif kind == "ppmd":
        out["order"] = int(expert.get("order", 10))
        out["memory_mb"] = int(expert.get("memory_mb", 64))
    elif kind == "rosaplus":
        out["max_order"] = int(expert.get("max_order", -1))
    elif kind == "match":
        out["hash_bits"] = int(expert.get("hash_bits", 20))
        out["min_len"] = int(expert.get("min_len", 4))
        out["max_len"] = int(expert.get("max_len", 255))
        out["base_mix"] = float(expert.get("base_mix", 0.02))
        out["confidence_scale"] = float(expert.get("confidence_scale", 1.0))
    elif kind == "rwkv7":
        method = expert.get("method")
        if isinstance(method, str):
            out["method"] = parse_rwkv_cfg_string(method)
        elif isinstance(method, dict):
            out["method"] = method
        else:
            raise SystemExit("rwkv7 expert in examples/two.json must include method")
    else:
        raise SystemExit(f"unsupported expert kind '{kind}' in examples/two.json")

    mixture_spec["experts"].append(out)

canonical_rate_backend = {
    "kind": "mixture",
    "spec": mixture_spec,
}
output_path.write_text(json.dumps(canonical_rate_backend, indent=2) + "\n")
PY
}

build_specs() {
    canonicalize_two_json_rate_backend "$TWO_JSON_PATH" "$CANONICAL_BASELINE_RATE"
    python3 - "$CANONICAL_BASELINE_RATE" "$SUBJECT_PATH" "$EVAL_TIME_LIMIT_SECONDS" "$TIME_BUDGET_SECONDS" "$MIN_THROUGHPUT" "$MAX_MEMORY_BYTES" "$ANNEALED_OUTPUT" "$ANNEALED_REPORT" "$MCAIXI_OUTPUT" "$MCAIXI_REPORT" "$WARMSTART_OUTPUT" "$WARMSTART_REPORT" "$WARMSTART_TEACHER" "$ANNEALED_SPEC" "$MCAIXI_SPEC" "$WARMSTART_SPEC" <<'PY'
import copy
import json
import pathlib
import sys

(
    baseline_rate_path,
    subject_path,
    eval_time_limit_seconds,
    time_budget_seconds,
    min_throughput,
    max_memory_bytes,
    annealed_output,
    annealed_report,
    mcaixi_output,
    mcaixi_report,
    warmstart_output,
    warmstart_report,
    warmstart_teacher_path,
    annealed_spec_path,
    mcaixi_spec_path,
    warmstart_spec_path,
) = sys.argv[1:]

baseline_rate = json.loads(pathlib.Path(baseline_rate_path).read_text())
baseline_candidate = {
    "kind": "rate-ac",
    "rate_backend": baseline_rate,
    "framing": "framed",
}

bounds = {
    "allowed_backends": ["ctw", "ppmd", "rosaplus", "match", "rwkv7", "mixture"],
    "forbidden_backends": [],
    "parameter_ranges": [
        {"parameter": "rate_backend.spec.alpha", "min": 0.005, "max": 0.20},
        {"parameter": "rate_backend.spec.experts[0].depth", "min": 8.0, "max": 96.0},
        {"parameter": "rate_backend.spec.experts[1].order", "min": 4.0, "max": 16.0},
        {"parameter": "rate_backend.spec.experts[1].memory_mb", "min": 64.0, "max": 768.0},
        {"parameter": "rate_backend.spec.experts[2].max_order", "min": -1.0, "max": 128.0},
        {"parameter": "rate_backend.spec.experts[3].hash_bits", "min": 16.0, "max": 22.0},
        {"parameter": "rate_backend.spec.experts[3].min_len", "min": 2.0, "max": 16.0},
        {"parameter": "rate_backend.spec.experts[3].max_len", "min": 32.0, "max": 255.0},
        {"parameter": "rate_backend.spec.experts[3].base_mix", "min": 0.005, "max": 0.10},
        {"parameter": "rate_backend.spec.experts[3].confidence_scale", "min": 0.5, "max": 2.0},
    ],
    "max_experts": 8,
    "max_mixture_nesting_depth": 3,
    "min_experts": 3,
    "allow_duplicate_experts": False,
    "required_experts": ["ctw", "ppmd"],
    "forbidden_expert_pairs": [],
}

common = {
    "schema_version": 1,
    "kind": "tune",
    "assets": [{"id": "dataset", "path": subject_path}],
    "input_asset": "dataset",
    "baseline_candidate": baseline_candidate,
    "bounds": bounds,
    "eval_time_limit_seconds": float(eval_time_limit_seconds),
    "time_budget_seconds": float(time_budget_seconds),
    "min_throughput_bytes_per_second": float(min_throughput),
    "max_memory_bytes": int(max_memory_bytes),
    "seed": 1337,
}

annealed = copy.deepcopy(common)
annealed["controller"] = {
    "kind": "annealed_hill_climbing",
    "max_mutation_radius": 4,
}
annealed["output_config_path"] = annealed_output
annealed["report_path"] = annealed_report

planner_interface = {
    "observation_bits": 8,
    "observation_stream_len": 1,
    "observation_key_mode": "full_stream",
    "reward_bits": 32,
    "agent_actions": 20,
}

mcaixi = copy.deepcopy(common)
mcaixi["controller"] = {
    "kind": "mc_aixi_fac_ctw",
    "interface": planner_interface,
    "planner_simulations_per_step": 24,
}
mcaixi["output_config_path"] = mcaixi_output
mcaixi["report_path"] = mcaixi_report

warmstart = copy.deepcopy(common)
warmstart["assets"] = [
    {"id": "dataset", "path": subject_path},
    {"id": "teacher", "path": warmstart_teacher_path},
]
warmstart["controller"] = {
    "kind": "aiqi_warmstart_exact_jh",
    "interface": planner_interface,
    "planner_simulations_per_step": 24,
    "return_horizon": 4,
    "label_phase_period": 8,
    "warmstart_teacher_dataset_asset": "teacher",
}
warmstart["output_config_path"] = warmstart_output
warmstart["report_path"] = warmstart_report

pathlib.Path(annealed_spec_path).write_text(json.dumps(annealed, indent=2) + "\n")
pathlib.Path(mcaixi_spec_path).write_text(json.dumps(mcaixi, indent=2) + "\n")
pathlib.Path(warmstart_spec_path).write_text(json.dumps(warmstart, indent=2) + "\n")
PY
}

run_tune() {
    local mode="$1"
    shift
    echo "[$(date +%H:%M:%S)] running $mode" | tee -a "$RUN_LOG"
    (
        cd "$REPO_ROOT"
        cargo run --release -p infotheory --no-default-features --features "$FEATURES" -- "$@"
    ) 2>&1 | tee -a "$RUN_LOG"
}

emit_exact_cert() {
    local spec_path="$1"
    local cert_path="$2"
    run_tune "emit-cert:$cert_path" \
        tune "$spec_path" \
        --rss-mode "$STRICT_RSS_MODE" \
        --evaluator-cgroup-parent "$EVAL_CGROUP_PARENT" \
        --emit-exact-reward-encoding-certificate "$cert_path"
}

compute_crc32_hex_of_file_bytes() {
    local path="$1"
    python3 - "$path" <<'PY'
import pathlib
import sys
import zlib
p = pathlib.Path(sys.argv[1])
raw = p.read_bytes()
print(f"{zlib.crc32(raw) & 0xffffffff:08x}")
PY
}

extract_observation_adapter_crc() {
    local report_path="$1"
    python3 - "$report_path" <<'PY'
import json
import pathlib
import sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
value = report["provenance"]["observation_adapter_content_crc32"]
print(value)
PY
}

write_provisional_warmstart_teacher() {
    local teacher_path="$1"
    local adapter_crc="$2"
    local reward_cert_crc="$3"
    cat > "$teacher_path" <<JSON
{
  "schema_version": 1,
  "contract": {
    "task_fingerprint": "pending",
    "action_alphabet_size": 20,
    "observation_bits": 8,
    "observation_stream_len": 1,
    "observation_key_mode": "full_stream",
    "observation_adapter_spec_ref": "$OBS_ADAPTER_REF",
    "observation_adapter_content_crc32": "$adapter_crc",
    "reward_bits": 32,
    "return_horizon": 4,
    "label_phase_period": 8,
    "scalar_representation": "$SCALAR_REP",
    "exact_reward_encoding_certificate": "$reward_cert_crc"
  },
  "traces": [
    {
      "transitions": [
        { "action": 0, "observations": [12], "reward": 1 },
        { "action": 1, "observations": [16], "reward": 0 },
        { "action": 2, "observations": [21], "reward": 1 },
        { "action": 3, "observations": [25], "reward": 0 },
        { "action": 4, "observations": [29], "reward": 1 },
        { "action": 5, "observations": [33], "reward": 0 },
        { "action": 6, "observations": [37], "reward": 1 },
        { "action": 7, "observations": [40], "reward": 0 },
        { "action": 0, "observations": [44], "reward": 1 },
        { "action": 1, "observations": [48], "reward": 0 },
        { "action": 2, "observations": [52], "reward": 1 },
        { "action": 3, "observations": [57], "reward": 0 },
        { "action": 4, "observations": [61], "reward": 1 },
        { "action": 5, "observations": [66], "reward": 0 },
        { "action": 6, "observations": [70], "reward": 1 },
        { "action": 7, "observations": [74], "reward": 0 }
      ]
    }
  ]
}
JSON
}

patch_warmstart_teacher_task_fingerprint() {
    local teacher_path="$1"
    local fingerprint="$2"
    python3 - "$teacher_path" "$fingerprint" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
fingerprint = sys.argv[2]
doc = json.loads(path.read_text())
doc["contract"]["task_fingerprint"] = fingerprint
path.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

extract_expected_fingerprint_from_error_log() {
    local error_log="$1"
    python3 - "$error_log" <<'PY'
import pathlib
import re
import sys
text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(
    r"teacher task_fingerprint '[^']*' does not match current planner_run '([0-9a-f]{8})'",
    text,
)
if match is None:
    print("")
else:
    print(match.group(1))
PY
}

write_summary() {
    python3 - "$ANNEALED_REPORT" "$MCAIXI_REPORT" "$WARMSTART_REPORT" "$SUMMARY_JSON" "$SUMMARY_TSV" <<'PY'
import json
import pathlib
import sys

annealed_report = pathlib.Path(sys.argv[1])
mcaixi_report = pathlib.Path(sys.argv[2])
warmstart_report = pathlib.Path(sys.argv[3])
summary_json = pathlib.Path(sys.argv[4])
summary_tsv = pathlib.Path(sys.argv[5])

def load(path):
    return json.loads(path.read_text())

reports = {
    "annealed": load(annealed_report),
    "mcaixi": load(mcaixi_report),
    "warmstart_exact_jh": load(warmstart_report),
}

def row(mode, report):
    best = report["best"]
    cache = report["cache"]
    return {
        "mode": mode,
        "status": report.get("status"),
        "objective_bits": best.get("objective_bits"),
        "deployable": best.get("deployable"),
        "throughput_bytes_per_second": best.get("throughput_bytes_per_second"),
        "peak_memory_bytes": best.get("peak_memory_bytes"),
        "target_loss_bits": best.get("target_loss_bits"),
        "candidate_evaluations_executed": cache.get("candidate_evaluations_executed"),
        "cache_hits": cache.get("cache_hits"),
        "cache_misses": cache.get("cache_misses"),
    }

rows = [row(name, report) for name, report in reports.items()]
summary = {
    "modes": rows,
    "winner_by_objective_bits": min(
        rows,
        key=lambda x: float("inf") if x["objective_bits"] is None else x["objective_bits"],
    )["mode"],
}

summary_json.write_text(json.dumps(summary, indent=2) + "\n")
with summary_tsv.open("w", encoding="utf-8") as fh:
    fh.write(
        "mode\tstatus\tobjective_bits\tdeployable\tthroughput_bytes_per_second\tpeak_memory_bytes\ttarget_loss_bits\tcandidate_evaluations_executed\tcache_hits\tcache_misses\n"
    )
    for item in rows:
        fh.write(
            f"{item['mode']}\t{item['status']}\t{item['objective_bits']}\t{item['deployable']}\t"
            f"{item['throughput_bytes_per_second']}\t{item['peak_memory_bytes']}\t{item['target_loss_bits']}\t"
            f"{item['candidate_evaluations_executed']}\t{item['cache_hits']}\t{item['cache_misses']}\n"
        )
PY
}

main() {
    : > "$RUN_LOG"
    {
        echo "run_dir=$RUN_DIR"
        echo "subject_source=$SUBJECT_SOURCE"
        echo "subject_path=$SUBJECT_PATH"
        echo "subject_bytes=$SUBJECT_BYTES"
        echo "eval_time_limit_seconds=$EVAL_TIME_LIMIT_SECONDS"
        echo "max_memory_gb=$MAX_MEMORY_GB"
        echo "max_memory_bytes=$MAX_MEMORY_BYTES"
        echo "min_throughput_bytes_per_second=$MIN_THROUGHPUT"
        echo "max_evaluations=$MAX_EVALUATIONS"
        echo "time_budget_seconds=$TIME_BUDGET_SECONDS"
        echo "features=$FEATURES"
        echo "two_json_source=$TWO_JSON_PATH"
        echo "rss_mode=$STRICT_RSS_MODE"
        echo "evaluator_cgroup_parent=$EVAL_CGROUP_PARENT"
    } | tee -a "$RUN_LOG"

    [ -f "$TWO_JSON_PATH" ] || die "missing baseline source: $TWO_JSON_PATH"

    build_specs

    run_tune "annealed" \
        tune "$ANNEALED_SPEC" \
        --rss-mode "$STRICT_RSS_MODE" \
        --evaluator-cgroup-parent "$EVAL_CGROUP_PARENT" \
        --max-evaluations "$MAX_EVALUATIONS"

    emit_exact_cert "$MCAIXI_SPEC" "$MCAIXI_CERT"
    run_tune "mcaixi" \
        tune "$MCAIXI_SPEC" \
        --rss-mode "$STRICT_RSS_MODE" \
        --evaluator-cgroup-parent "$EVAL_CGROUP_PARENT" \
        --exact-reward-encoding-certificate "$MCAIXI_CERT" \
        --max-evaluations "$MAX_EVALUATIONS"

    emit_exact_cert "$WARMSTART_SPEC" "$WARMSTART_CERT"
    warm_reward_crc="$(compute_crc32_hex_of_file_bytes "$WARMSTART_CERT")"
    adapter_crc="$(extract_observation_adapter_crc "$ANNEALED_REPORT")"
    write_provisional_warmstart_teacher "$WARMSTART_TEACHER" "$adapter_crc" "$warm_reward_crc"

    warm_probe_err="$RUN_DIR/warmstart-probe-error.log"
    set +e
    (
        cd "$REPO_ROOT"
        cargo run --release -p infotheory --no-default-features --features "$FEATURES" -- \
            tune "$WARMSTART_SPEC" \
            --rss-mode "$STRICT_RSS_MODE" \
            --evaluator-cgroup-parent "$EVAL_CGROUP_PARENT" \
            --exact-reward-encoding-certificate "$WARMSTART_CERT" \
            --max-evaluations 1
    ) >"$RUN_DIR/warmstart-probe-stdout.log" 2>"$warm_probe_err"
    probe_status=$?
    set -e
    if [ "$probe_status" -ne 0 ]; then
        expected_fp="$(extract_expected_fingerprint_from_error_log "$warm_probe_err")"
        if [ -z "$expected_fp" ]; then
            cat "$warm_probe_err" >&2
            die "warmstart probe failed, and expected task_fingerprint could not be extracted"
        fi
        patch_warmstart_teacher_task_fingerprint "$WARMSTART_TEACHER" "$expected_fp"
    fi

    run_tune "warmstart" \
        tune "$WARMSTART_SPEC" \
        --rss-mode "$STRICT_RSS_MODE" \
        --evaluator-cgroup-parent "$EVAL_CGROUP_PARENT" \
        --exact-reward-encoding-certificate "$WARMSTART_CERT" \
        --max-evaluations "$MAX_EVALUATIONS"

    write_summary

    {
        echo
        echo "Completed three-mode comparison anchored to examples/two.json."
        echo "Run directory: $RUN_DIR"
        echo "Summary JSON: $SUMMARY_JSON"
        echo "Summary TSV:  $SUMMARY_TSV"
        echo "Reports:"
        echo "  annealed:  $ANNEALED_REPORT"
        echo "  mcaixi:    $MCAIXI_REPORT"
        echo "  warmstart: $WARMSTART_REPORT"
    } | tee -a "$RUN_LOG"
}

main
