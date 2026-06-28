# Developer Testing and Coverage

This project now tracks Rust and Python binding quality with explicit parity and
coverage workflows.

## Local Quick Start

```bash
# Rust tests (default features)
cargo test -p infotheory --locked

# Rust CLI + broad backend parity pass
cargo test -p infotheory --no-default-features --features "cli all-backends" --locked

# VM-focused Rust tests
cargo test -p infotheory --no-default-features --features "vm backend-ctw" --locked
```

```bash
# Build Python extension in editable mode using the repo's pyproject/maturin config
uv run maturin develop

# Python tests
uv run pytest -q python/tests

# Python coverage (wrapper module)
uv run pytest \
  --cov=infotheory_rs \
  --cov-report=term-missing \
  --cov-report=xml:target/python-coverage.xml \
  --cov-fail-under=100 \
  python/tests
```

## Rust Coverage

`cargo-llvm-cov` is used for line coverage over the first-party `infotheory`
crate (library + CLI tests).

```bash
cargo llvm-cov -p infotheory --tests --features "cli all-backends" --locked --summary-only
```

CI enforces a minimum line coverage threshold for this command.

## Rustdoc Coverage

Rustdoc item coverage is measured with nightly rustdoc:

```bash
cargo +nightly rustdoc -p infotheory --all-features -- \
  -Z unstable-options --show-coverage --output-format json \
  > /tmp/rustdoc_cov.json
```

CI currently enforces full documented-item coverage (100%) using this report.

## Golden and Parity Tests

The suite includes:

- Rust API ↔ CLI batch parity (`tests/cli_api_parity.rs`)
- Python bindings ↔ CLI parity (`python/tests/test_cli_parity_expanded.py`)
- Python backend parity for `match`, `sparse-match`, `ppmd`, `mixture`, `particle`,
  `calibrated`, `mamba`, and `rwkv7` string parsing (`python/tests/test_api_surface.py`)
- Compression/decompression roundtrip checks in Rust and Python
- VM stats-backend parsing and predictor-backed trace-model coverage for the new
  backends (`crates/infotheory/src/main.rs`, `crates/infotheory/src/aixi/vm_nyx.rs`)
- Deterministic fixture hash checks (`crates/infotheory/tests/roundtrip_hashes.rs`, `python/tests/test_golden_hashes.py`)
- RWKV method parsing/canonicalization tests (`crates/infotheory/tests/rwkv_method_canonicalization.rs`)

These tests are designed to catch semantic drift and output regressions across
interfaces.

## Local CI Preflight

For a local CI-like pass, prefer the project wrapper:

```bash
./projman.sh test_ci
```

Useful controls:

- `INFOTHEORY_BUILD_MODE=native|portable`
- `INFOTHEORY_CI_INCLUDE_VM=1`
- `INFOTHEORY_CI_SKIP_RUST_LINE_COVERAGE=1`
- `INFOTHEORY_CI_SKIP_RUSTDOC_COVERAGE=1`
- `INFOTHEORY_CI_SKIP_FEATURE_GATES=1`
- `INFOTHEORY_CI_SKIP_PYTHON=1`

Avoid indiscriminate workspace all-features sweeps; they pull in heavyweight
optional surfaces that are intentionally tested through curated CI slices.

## Benchmark Provenance Checks

The `two-json` benchmark suite is pinned to the historical canonical
`configs/bench/two.json` / `examples/two.json` spec with `alpha = 0.03`.

The benchmark harness and comparator now enforce provenance:

- `scripts/bench_two_json.sh` records the resolved suite-spec path, suite-spec
  SHA-256 digest, build mode, and build features in raw and summary TSVs.
- `scripts/compare_bench_two_json.lua` rejects baseline/current comparisons when
  the suite-spec digests differ.
- Rust and Python tests assert that the checked-in `two.json` benchmark specs
  stay byte-identical and preserve the historical `alpha = 0.03` setting.

This is the guardrail against benchmark-subject drift being mistaken for a code
regression.

## MC-AIXI Competitor Benchmark Validation

Use the reproducible benchmark harness to validate cross-implementation parity
for MC-AIXI behavior and reporting:

```bash
./projman.sh bench_aixi_competitors --profile default --trials 1
```

Parity/correctness expectations for this benchmark:

- Rust environments used in the run are reference-aligned with C++/PyAIXI for
  Kuhn Poker and Biased Rock-Paper-Scissors dynamics.
- `random_seed`/`rng_seed` deterministically seeds both agent and environment
  stochasticity.
- Reported rewards are on a common native domain scale (Kuhn offset removed for
  C++/PyAIXI outputs).
- MC-AIXI uses reference-style UCB scaling while retaining reward-sensitive
  chance-node tree reuse to avoid percept collisions in generic environments.
