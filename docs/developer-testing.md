# Developer Testing and Coverage

This project now tracks Rust and Python binding quality with explicit parity and
coverage workflows.

## Local Quick Start

```bash
# Rust tests (default features)
cargo test --locked

# Rust tests with CLI enabled (includes CLI/API parity + search tests)
cargo test --features cli --locked

# VM-focused Rust tests
cargo test --features vm --test nyx_vm_tests --locked
```

```bash
# Build Python extension in editable mode
uv run maturin develop --release

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
cargo llvm-cov -p infotheory --tests --features cli --locked --summary-only
```

CI enforces a minimum line coverage threshold for this command.

## Rustdoc Coverage

Rustdoc item coverage is measured with nightly rustdoc:

```bash
cargo +nightly rustdoc -p infotheory --all-features -- \
  -Z unstable-options --show-coverage --output-format json \
  > /tmp/rustdoc_cov.json
```

CI enforces a minimum documented-item percentage using this report.

## Golden and Parity Tests

The suite includes:

- Rust API ↔ CLI batch parity (`tests/cli_api_parity.rs`)
- Python bindings ↔ CLI parity (`python/tests/test_cli_parity_expanded.py`)
- Python backend parity for `match`, `sparse-match`, `ppmd`, `mixture`, `particle`,
  `calibrated`, `mamba`, and `rwkv7` string parsing (`python/tests/test_api_surface.py`)
- Compression/decompression roundtrip checks in Rust and Python
- VM stats-backend parsing and predictor-backed trace-model coverage for the new
  backends (`src/main.rs`, `src/aixi/vm_nyx.rs`)
- Deterministic fixture hash checks (`tests/roundtrip_hashes.rs`, `python/tests/test_golden_hashes.py`)
- RWKV method parsing/canonicalization tests (`tests/rwkv_method_canonicalization.rs`)

These tests are designed to catch semantic drift and output regressions across
interfaces.

## MC-AIXI Competitor Benchmark Validation

Use the reproducible benchmark harness to validate cross-implementation parity
for MC-AIXI behavior and reporting:

```bash
./projman.sh bench__aixi_competitors --profile default --trials 1
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
