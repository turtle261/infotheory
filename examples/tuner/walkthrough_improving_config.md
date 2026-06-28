# Walkthrough: Improving `examples/two.json` with Theorem-Facing Tuner Modes

This walkthrough is a reproducible, strict (theorem-facing) comparison across:

1. `annealed_hill_climbing`
2. `mc_aixi_fac_ctw`
3. `aiqi_warmstart_exact_jh`

All three runs use one shared baseline family anchored to `examples/two.json`, one shared bounds space, one shared dataset, one shared strict evaluator profile, and one shared deployability envelope.

Before copy-pasting commands below, set the repository root once:

```bash
export INFOTHEORY_REPO=/path/to/infotheory
```

## What this compares fairly

- Same baseline candidate family for all three modes: `rate-ac` + neural mixture derived from `examples/two.json`.
- Same search space (`bounds`) for all three modes.
- Same per-eval deadline and memory envelope.
- Same strict memory mode for all three modes:
  - `--rss-mode hybrid_strict_max`
  - delegated cgroup-v2 parent via `--evaluator-cgroup-parent`
- Same evaluation cap and time budget.


## Included example specs

These three committed specs demonstrate the shared design:

- [two-json-annealed-spec.json](two-json-annealed-spec.json)
- [two-json-mcaixi-spec.json](two-json-mcaixi-spec.json)
- [two-json-warmstart-spec.json](two-json-warmstart-spec.json)

They use this in-tree dataset by default:

- `$INFOTHEORY_REPO/benchmarks/6f464811/infotheory-two-json-summary-full.tsv`
- Reason: it is in-tree, stable for audit, and size-appropriate for tuner comparisons.

## Reproducible strict benchmark script

Use:

- [benchmark_tuner_two_json_modes.sh](../../scripts/benchmark_tuner_two_json_modes.sh)

Behavior:

- Anchors from `examples/two.json` and canonicalizes it into tune-acceptable baseline JSON.
- Runs annealed, MC-AIXI, then warmstart sequentially.
- Emits exact reward certificates under the strict evaluator profile.
- Bootstraps warmstart teacher fingerprint via explicit probe/patch flow.
- Writes machine-readable comparison artifacts.

Reversibility note (important for audit):

- This benchmark *does* include `rate_backend.spec.experts[2].max_order` in the search space.
- A prior reversibility failure on this dimension was traced to mutation-kind instability at the
  `-1 <-> 0` boundary (JSON numeric typing flipped from signed to unsigned by value).
- The kernel now preserves signed integer mutation semantics whenever the configured range has a
  negative lower bound, so `max_order` tuning remains enabled without violating reversible-kernel checks.

Defaults:

- Subject data: `git show HEAD:README.md` written into run directory.
- Per-eval limit: `2` seconds.
- Memory cap: `1` GiB.
- Max evaluations per mode: `120`.
- Time budget per mode: `600` seconds.

## Strict setup and run

One-time delegated cgroup setup:

```bash
cd "$INFOTHEORY_REPO"
sudo ./scripts/delegate_tuner_cgroup_v2.sh setup theo infotheory-tuner
```

Run full strict comparison (default subject = `README.md` at HEAD):

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  ./scripts/benchmark_tuner_two_json_modes.sh
```

Run with explicit subject file + limits:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  ./scripts/benchmark_tuner_two_json_modes.sh /path/to/input.bin 2 1
```

Optional 30-minute envelope (example):

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  TUNER_MAX_EVALUATIONS=180 TUNER_TIME_BUDGET_SECONDS=1800 \
  ./scripts/benchmark_tuner_two_json_modes.sh /path/to/input.bin 2 1
```

## Output artifacts

Per run, script prints `run_dir` under:

- `/tmp/infotheory-tuner-two-json/run-YYYYMMDD-HHMMSS`

Key outputs:

- `comparison-summary.json`
- `comparison-summary.tsv`
- `benchmark.log`
- `annealed-report.json`
- `mcaixi-report.json`
- `warmstart-report.json`
- strict exact reward cert files for MC-AIXI/warmstart

## What to return for verification

After running, return:

1. `run_dir`
2. `comparison-summary.tsv`
3. `comparison-summary.json`
4. last ~120 lines of `benchmark.log`
5. confirmation of theorem-facing strict markers in reports:
   - `provenance.executor_controls.rss_mode.strict_theorem_memory_certified: true`
   - strict memory provenance (`cgroup_v2_peak` / strict hybrid path)

Copy-paste checks:

```bash
RUN_DIR="/tmp/infotheory-tuner-two-json/run-YYYYMMDD-HHMMSS"
ls -l "$RUN_DIR"/{comparison-summary.tsv,comparison-summary.json,annealed-report.json,mcaixi-report.json,warmstart-report.json}
cat "$RUN_DIR/comparison-summary.tsv"
python3 - <<'PY' "$RUN_DIR"
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
for name in ["annealed-report.json", "mcaixi-report.json", "warmstart-report.json"]:
    doc = json.loads((run / name).read_text())
    rss = (
        doc.get("provenance", {})
        .get("executor_controls", {})
        .get("rss_mode", {})
    )
    print(name)
    print("  status:", doc.get("status"))
    print("  objective_bits:", doc.get("best", {}).get("objective_bits"))
    print("  strict_theorem_memory_certified:", rss.get("strict_theorem_memory_certified"))
PY
tail -n 120 "$RUN_DIR/benchmark.log"
```
