# InfoTheory

InfoTheory is a Rust library, CLI, and Python extension for algorithmic
information theory and related information-theoretic functions. It's focused on
predictive modelling, compression-derived complexity estimates, AIXI approximate (and other universal-ish) agents, and tooling.

## What Is Included

- Information metrics: NCD, Normalized Entropy Distance, Normalized Transform Effort, Mutual Information, entropy/cross-entropy/conditional
  entropy, intrinsic dependence, resistance to transformation, KL, JS, TVD, and
  Hellinger distance.
- Predictive backends: ROSA+, CTW/FAC-CTW, PPMD, Sequitur, match/sparse-match,
  ZPAQ-as-rate, Mamba, RWKV7, mixtures, particle filters, and calibrated
  wrappers. Availability is feature-gated.
- Compression surfaces: native ZPAQ, generic AC/rANS rate-coded compression over any rate backend.
- Agents: canonical `planner_run` execution for MC-AIXI, AIQI, and VM-backed
  environments through Nyx-Lite.
- Tuner: canonical `tune` specs for bounded task-specific backend search. 
- Python bindings: `infotheory-rs` on PyPI, imported as `infotheory_rs`.

## Repository Layout

- `crates/infotheory`: primary Rust library and optional CLI binary.
- `crates/infotheory_py`: PyO3/maturin Python extension.
- `crates/benchman`: TUI benchmark-summary inspector.
- `configs`: canonical checked-in planner and benchmark configs.
- `examples`: runnable examples, including tuner validation specs.
- `docs`: developer, performance, tuner, warm-start, and canonical-pipeline docs.
- `scripts` and `projman.sh`: local workflow, benchmark, CI-preflight, and
  conversion tooling.
- `vendor/zpaq_rs`, `vendor/nyx-lite`, `vendor/gameengine`: optional/path
  dependencies used by selected features.

## Build

Rust stable is required. Some workflows also need `uv`, `maturin`, Lean, KVM, or
platform tools; see [docs/developer-testing.md](docs/developer-testing.md).

Build the CLI:

```bash
cargo build -p infotheory --release --features cli --bin infotheory --locked
```

Run the default Rust test slice:

```bash
cargo test -p infotheory --locked
```

Run the CLI plus broad backend parity slice:

```bash
cargo test -p infotheory --no-default-features --features "cli all-backends" --locked
```

Local workflows respect `INFOTHEORY_BUILD_MODE`:

- `native`: local development/benchmark default, using repository-default native CPU
  tuning.
- `portable`: generic CPU flags to match portable CI/release behavior.

## CLI Quick Start

Build the binary first, or use `cargo run -p infotheory --features cli -- ...`.

```bash
infotheory h README.md
infotheory h_rate README.md --rate-backend ctw --method 32
infotheory ncd a.bin b.bin --compression-backend zpaq --method 5
infotheory ncd a.bin b.bin --compression-backend rate-ac --rate-backend ctw --method 16
infotheory compress in.bin out.itc --compression-backend rate-rans --rate-backend fac-ctw --method 32
infotheory decompress out.itc restored.bin --compression-backend rate-rans --rate-backend fac-ctw --method 32
cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
infotheory aixi configs/aixi/paper_kuhn_poker.json
```

The CLI has topic help:

```bash
infotheory --help
infotheory help backends
infotheory help tune
infotheory ncd --help
infotheory warmstart --help
```

Important CLI surfaces:

- `batch`: line-oriented JSON request/response mode.
- `aixi`: executes canonical `planner_run` documents.
- `warmstart`: exports, converts, and merges teacher datasets.
- `tune`: executes canonical `tune` documents.
- `ac-log-loss`: emits exact mixture AC/log-loss diagnostics as TSV.

## Rust API Quick Start

```rust
use infotheory::api::{CompressionBackend, InfotheoryCtx, RateBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = InfotheoryCtx::from_specs(
        RateBackend::Ctw { depth: 16 },
        CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 16 },
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        },
    )?;

    let bits = ctx.try_entropy_rate_bytes(b"abracadabra")?;
    assert!(bits.is_finite());
    Ok(())
}
```

`InfotheoryCtx::from_specs()` validates, canonicalizes, and compiles wrapper specs
before execution. Advanced callers can explicitly use `validate()` and
`compile()` on `RateBackend` and `CompressionBackend`, then construct
`InfotheoryCtx::new(compiled_rate, compiled_compression)`.

## Python Quick Start

Local editable build:

```bash
uv run maturin develop
uv run pytest -q python/tests
```

Example:

```python
import infotheory_rs as ait

backend = ait.RateBackend.ctw(16)
assert ait.entropy_rate_backend(b"abracadabra", backend=backend) >= 0.0

framed = ait.CompressionBackend.rate_ac(backend, "framed")
blob = ait.compress_bytes_backend(b"payload", compression_backend=framed)
assert ait.decompress_bytes_backend(blob, compression_backend=framed) == b"payload"
```

The Python package is built as `abi3-py310`. Published wheels are portable and exclude VM support by default; Linux source builds can opt into VM
bindings with the Rust `vm` feature, through git. The published versions on crates.io have a shimmed(useless) VM feature.

## Documentation Map

- [Canonical pipeline invariants](docs/developer-canonical-pipeline-invariants.md)
  defines the normative parse/validate/compile/runtime boundary.
- [Developer testing](docs/developer-testing.md) records local test, coverage,
  rustdoc, Python, and CI-preflight commands.
- [CLI reference](docs/cli.md) summarizes commands, topic help, and backend
  selection.
- [Performance and benchmarking](docs/perf.md) covers build modes and benchmark
  workflow details.
  [examples/tuner](examples/tuner/README.md) cover tuner specs and manual
  validation commands.

## Platform Notes

CI covers Linux GNU/musl, macOS, Windows, *BSD, AArch64 variants, and
WASM slices where applicable. ZPAQ is not supported on WASM. VM support is
Linux/KVM-specific and depends on `vendor/nyx-lite`.

For local VM setup:

```bash
./projman.sh init-vm
cargo build -p infotheory --release --features vm --locked
```

## License

InfoTheory is available under either the Apache-2.0 License or the ISC License,
at your choice. Contributing to this repository means you agree to submit
contributions under that dual-license arrangement.
