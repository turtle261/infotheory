# Tuner Strict Validation Examples

These examples are intended for local strict-mode validation runs that are
substantially stronger than the tiny smoke spec.

Dataset choice:
- `/home/theo/dev/infotheory/benchmarks/6f464811/infotheory-two-json-summary-full.tsv`
- Reason: it is in-tree and won't be altered, and is an appropriate size for
  repeatable local validation.

Matrix included here (non-AIQI theorem-facing controller modes):
- `annealed_hill_climbing`: simple + advanced
- `mc_aixi_fac_ctw`: simple + advanced

Advanced variants use a **neural-kind mixture** (`kind: "neural"`) with
multiple experts so tuning operates over mixture settings in a realistic setup.

## Quick Validation Commands

Simple annealed:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
  tune examples/tuner/strict-annealed-simple-tsv-spec.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1
```

Advanced annealed neural-mixture:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
  tune examples/tuner/strict-annealed-advanced-neural-mixture-tsv-spec.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1
```

Simple MC-AIXI:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
  tune examples/tuner/strict-mcaixi-simple-tsv-spec.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1 \
  --exact-reward-encoding-certificate strict-mcaixi-simple-tsv-exact-reward-cert-process-rss.json
```

Advanced MC-AIXI neural-mixture:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
  tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1 \
  --exact-reward-encoding-certificate strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-process-rss.json
```

Note:
- The two MC-AIXI example certificates are intentionally scoped to the
  demonstrated `process_rss_peak` evaluator profile (they bind dataset, bounds,
  evaluator profile, and controller kind).
- For strict Linux cgroup-v2 (`hybrid_strict_max`) runs, regenerate certificates
  for the strict evaluator profile before expecting theorem-facing certificate
  acceptance.
