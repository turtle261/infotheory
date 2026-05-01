# InfoTheory

### 1. Unified Information Estimation
Estimate core measures using both **Empirical** (aka Marginal/IID) and **Rate** (predictive-based) approaches:
- **NCD (Normalized Compression Distance)**: Approximates information distance using compression.
- **MI (Mutual Information)**: Quantifies shared information between sequences.
- **NED (Normalized Entropy Distance)**: A metric distance based on mutual information.
- **NTE (Normalized Transform Effort)**: Variation of Information (VI).
- **Intrinsic Dependence**: Redundancy Ratio.
- **Resistance**: Information preservation under noise/transform.

### 2. Multi-Backend Predictive Engine
The library exposes two layers for predictive backends:

- `RateBackend` / `CompressionBackend`: wrapper ASTs and compatibility specs used by CLI, JSON, and Python-facing surfaces.
- `CompiledRateBackend` / `CompiledCompressionBackend`: canonicalized, validated, immutable execution plans used by the generic Rust API.

In other words, wrapper specs are the configuration layer, and compiled plans are the execution layer. This keeps parsing, canonicalization, feature validation, and method-string resolution out of hot runtime paths.

Switch between different `RateBackend` families seamlessly:
- **ROSA+ (Rapid Online Suffix Automaton + Witten Bell)**: A fast statistical LM. Default backend. 
- **CTW (Context Tree Weighting)**: Historically standard for AIXI. Accurate bit-level Bayesian model (KT-estimator).
- **Sequitur**: Exact online grammar induction with Sequitur normalization plus predictive suffix-context readout.
- **Mamba (Neural Network)**: Deterministic CPU-first Mamba-1 backend with online mode + export.
- **RWKV (Neural Network)**: Portable SIMD RWKV7 CPU inference backend (`wide`-based).

The same `RateBackend` wrapper family also supports ensemble world models. `RateBackend::Mixture` combines `RateBackend` experts into a single predictive model: `Bayes`, `Switching`, and `Convex` follow *On Ensemble Techniques for AIXI Approximation*, while `FadingBayes`, `Mdl`, and `Neural` are extensions implemented in this repository.

### 3. Integrated MC-AIXI Agent
Includes a full implementation of the **Monte Carlo AIXI (MC-AIXI)** agent described by Hutter et al. It approximates incomputable AIXI with Monte-Carlo Tree Search and can use the library's `RateBackend` model class, including mixture-based ensemble world models, as its world model.

You can use a trained neural model (Mamba-1 or RWKV7) as a rate backend ("world model") for MC-AIXI.

- `planner: "mc-aixi"` selects the classic MCTS-based MC-AIXI planner.
- MC-AIXI can also take a full `rate_backend` object instead of relying only on `algorithm`, including nested mixture backends built from other `RateBackend` experts.
- **Mixture families from *On Ensemble Techniques for AIXI Approximation***: `Bayes` and `Convex` are exposed directly, and `Switching` follows the fixed-share update from *On Ensemble Techniques for AIXI Approximation* with a constant switch-rate `alpha`.
- **Extensions**: `FadingBayes`, `Mdl`, and `Neural` remain available.
- **Why recursive `zpaq` is rejected in generic MC-AIXI configs**: `zpaq` cannot roll predictor state backward after hypothetical actions, so it does not satisfy the reversible action-conditioning requirement used by *A Monte-Carlo AIXI Approximation*. The older standalone `algorithm: "zpaq"` mode still exists, but it does not provide that exact rollback behavior.
- **UCB tie-breaking from *A Monte-Carlo AIXI Approximation***: MC-AIXI chooses uniformly at random among unvisited actions and among exactly tied maximal UCB actions.

### 4. Integrated AIQI Agent
The repository also includes **AIQI**, the model-free return-prediction agent introduced in *A Model-Free Universal AI* by Yegon Kim and Juho Lee, with periodic augmentation (`N >= H`) and discretized H-step return targets.

- `planner: "aiqi"` enables AIQI in `infotheory aixi <config.json>`.
- `planner: "mc-aixi"` (default) keeps MC-AIXI as the default planner.
- **Direct AIQI-CTW configuration from *A Model-Free Universal AI***: `algorithm: "ac-ctw"` (or `"ctw"`) selects the AIQI-CTW setup described in *A Model-Free Universal AI*.
- **Extensions**: AIQI also supports `fac-ctw`, `rosa`, `rwkv`, and generic `rate_backend` predictors, including the same mixture JSON format used elsewhere in the repo.
- **Why `zpaq` is excluded from AIQI**: AIQI needs exact frozen predictor states while it scores hypothetical actions and return bins, and `zpaq` does not provide that interface.
- **Validation from *A Model-Free Universal AI***: AIQI enforces `discount_gamma in (0,1)` and `baseline_exploration (tau) in (0,1]`.
- **Tie-breaking from *A Model-Free Universal AI***: greedy action selection uses a fixed tie-break rule (first maximizing action) to match the fixed tie-breaking assumption in *A Model-Free Universal AI*.
- **Optional bounded memory**: set `history_prune_keep_steps` (or `aiqi_history_prune_keep_steps`) to retain only recent history while still keeping the steps needed for exact H-step return construction.
- **Reproducibility**: set `random_seed` in config (or planner-specific `aiqi_random_seed` / `mcaixi_random_seed`) to make agent-side randomness deterministic across runs.
- AIQI uses the same environment interfaces as MC-AIXI, including VM environments.

---

## Compilation & Installation
### Platform Support (tested)
`infotheory` is currently tested on:
- **Linux (GNU libc)** (`x86_64-unknown-linux-gnu`)
- **Linux (musl)** (`x86_64-unknown-linux-musl`)
- **macOS (Intel)** (`x86_64-apple-darwin`)
- **macOS (Apple Silicon)** (`aarch64-apple-darwin`)
- **Windows** (`x86_64-pc-windows-msvc`)
- **FreeBSD** (`x86_64-unknown-freebsd`)
- **OpenBSD** (`x86_64-unknown-openbsd`)
- **NetBSD** (`x86_64-unknown-netbsd`)
- **AArch64 Linux (GNU/musl)** (`aarch64-unknown-linux-gnu`, `aarch64-unknown-linux-musl`)
- **AArch64 Windows** (`aarch64-pc-windows-msvc`)
- **WASM** (`wasm32-unknown-unknown`)

<small>ZPAQ feature is not supported on WASM targets</small>

### Build Prerequisites
- Rust toolchain (stable): `rustup` recommended.
- C/C++ toolchain: `clang` + `lld` recommended on Unix-like systems.
- For local repository builds with VM support available: clone recursively (`--recurse-submodules`) so `vendor/nyx-lite` is present.

### Build Configuration
- By default, .cargo/config.toml is set to use march=native as the target-cpu, which will allow LLVM to make full use of your specific CPU. This can improve performance by roughly 2x for the RWKV Model. This may affect binary compatibility depending on your usecase.
- Local `projman.sh` workflows now make this explicit:
  - `INFOTHEORY_BUILD_MODE=native` keeps the repository default `target-cpu=native` behavior for local development and benchmarking.
  - `INFOTHEORY_BUILD_MODE=portable` overrides local cargo invocations to `target-cpu=generic`, matching the portable CI/release intent.
  - `INFOTHEORY_CLI_BENCH_BUILD_MODE=native|portable` does the same specifically for `./projman.sh bench cli ...`.

### Build the CLI
Enable the `cli` feature (the binary is feature-gated):

```bash
cargo build --release --features cli --bin infotheory
```

Output binary:
- `./target/release/infotheory` (host target)
- `./target/<target-triple>/release/infotheory` (cross target)

### Build as a library
Add the dependency in your `Cargo.toml`:

```toml
[dependencies]
# From git (current development):
# infotheory = { git = "https://github.com/turtle261/infotheory", branch = "main" }

# From crates.io (when released):
# infotheory = "1.2"

# With feature selection:
# infotheory = { version = "1.2", features = ["cli", "aixi"] } # Infotheory is highly configurable at the feature-level, you can compile in what you need.
```

### Library API layering

The generic Rust API is compiled-plan-first. Internally, specs are validated and canonicalized once, then compiled to immutable execution plans. Typical library usage is:

```rust
use infotheory::api::{CompressionBackend, InfotheoryCtx, RateBackend};

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
```

`InfotheoryCtx::from_specs()` handles validation and compilation internally, ensuring a single consistent compiled plan across both rate and compression backends. The compiled backends are also accessible via `ctx.rate_backend` and `ctx.compression_backend` if you need explicit pre-compiled access for repeated operations.

For advanced use cases, explicit validation and compilation helpers are available:

- `RateBackend::validate()` / `CompressionBackend::validate()` — produce `ValidatedRateBackend` / `ValidatedCompressionBackend`
- `RateBackend::compile()` / `CompressionBackend::compile()` — produce `CompiledRateBackend` / `CompiledCompressionBackend`
- `InfotheoryCtx::new(compiled_rate, compiled_compression)` — construct context from pre-compiled backends

All validated and compiled types are re-exported from `infotheory::api`.

### Building Nyx-Lite
The VM backend is optional (`--features vm`) and depends on `vendor/nyx-lite` (and its vendored submodule code). Build it with:
```bash
cargo build --release --features vm
```
Notes:
- VM is Linux/KVM-oriented (`/dev/kvm` required).
- Some `vendor/nyx-lite` tests also require VM image artifacts under `vendor/nyx-lite/vm_image`.

### Additional notes
Platform caveats:
- **OpenBSD/NetBSD**: kernel W^X policies can break ZPAQ JIT at runtime. Set `ZPAQ_NOJIT=1`.
- **NetBSD**: release LTO is problematic in common toolchains; disable release LTO if needed (see `.cargo/config.toml` comments).
- **MacOS**: Supported on both Intel and Apple Silicon natively.

Optional tooling used by some tests/workflows:
- docker (for tests, or if you want to use it for rootfs generation)
- cpio
- wget (for tests, or to use the provided kernel. you can also use curl instead manually on the download_kernel.sh file )
- cmake (for VM feature, firecracker needs it)
- Lean4 (Toolchain Version 4.14.0)
---

## CLI Usage

The `infotheory` binary provides a powerful interface for file analysis.

### Primitives
```bash
# Calculate Mutual Information (Empirical IID Shannon plugin)
./infotheory mi file1.txt file2.txt

# Use CTW backend for NTE (Normalized Transform Effort)
./infotheory nte file1.txt file2.txt --rate-backend ctw

# Calculate NCD with custom ZPAQ method
./infotheory ncd file1.txt file2.txt 5
```

### Compression Backends

`CompressionBackend` is the canonical compression enum in the library.

CLI:

```bash
# ZPAQ standalone (as before)
./infotheory ncd a.bin b.bin --compression-backend zpaq --method 5

# Turn any rate backend into a compressor via AC/rANS
./infotheory ncd a.bin b.bin --compression-backend rate-ac --rate-backend ctw --method 16
./infotheory ncd a.bin b.bin --compression-backend rate-rans --rate-backend fac-ctw --method 16
```

For rate-coded metrics, raw framing is used by default to avoid framing overhead.
Explicit `compress_bytes_backend` / `decompress_bytes_backend` APIs support framed payloads for roundtrip verification.

For direct Rust usage, those compression APIs take `CompiledCompressionBackend`.
Compile once up front and reuse the compiled plan when you care about minimizing repeated setup overhead.

### AC Log-Loss Diagnostics

`ac-log-loss` runs the exact arithmetic-coding predictor path for a top-level mixture spec and streams per-position diagnostics to TSV without keeping the full trace in memory.

```bash
RAYON_NUM_THREADS=4 ./infotheory ac-log-loss corpus.bin \
  --mixture configs/bench/mixture.json \
  --out-prefix /tmp/mixture-diagnostic
```

It writes:

- `/tmp/mixture-diagnostic.trace.tsv`: per-position mixture probability/bits, oracle fields, root weight statistics, and per-node `prob` / `bits` / `local_weight` / `effective_weight`
- `/tmp/mixture-diagnostic.nodes.tsv`: flattened mixture-tree metadata with stable node ids
- `/tmp/mixture-diagnostic.summary.tsv`: total bits, oracle regret, switch counts, AC payload bits, and per-node aggregates

### Neural Method Strings

Mamba and RWKV can be configured with either a model file or compact method string:

- `file:/abs/or/relative/model.safetensors`
- `file:/abs/or/relative/model.safetensors;policy:...`
- `cfg:key=value,...[;policy:...]`

`file:` method paths may contain reserved delimiters like `;` when they are percent-escaped in canonical form; raw `file:...;policy:...` strings are accepted and normalized for you.

Supported `cfg:` keys:
- RWKV7: `hidden,layers,intermediate,decay_rank,a_rank,v_rank,g_rank,seed,train,lr,stride`
- Mamba-1: `hidden,layers,intermediate,state,conv,dt_rank,seed,train,lr,stride`

`train` supports: `none`, `sgd`, `adam`.
`policy` supports `schedule=...` rules (for example `0..100:infer` or `0..100:train(scope=head+bias,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)`).
For RWKV full-parameter training scopes (`scope` touching non-head parameters), `bptt<=1` resolves to the fast default window `8`; specify a larger explicit `bptt` to override it.

Example:

```bash
./infotheory h file.txt \
  --rate-backend rwkv7 \
  --method "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=7,train=sgd,lr=0.01,stride=1;policy:schedule=0..100:train(scope=head+bias,opt=sgd,lr=0.01,stride=1,bptt=1,clip=0,momentum=0.9)"
```

For `configs/bench/two.json` benchmark plotting, `scripts/plot_two_json.sh` also accepts `INFOTHEORY_BASELINE_SUMMARY_TSV=/path/to/baseline-summary.tsv` to emit additional baseline-overlay SVGs.

`scripts/bench_two_json.sh` now records benchmark provenance in both raw and
summary TSVs:

- resolved suite spec path
- suite spec SHA-256 digest
- build mode
- build features

`scripts/compare_bench_two_json.lua` rejects comparisons when the suite spec
digest differs between baseline and candidate summaries. That prevents
benchmark-spec drift from being misreported as a performance regression.

The benchmark tooling also supports an `extra` suite for additional rate backends
not in `configs/bench/two.json` (currently `mamba`, `particle` via
`configs/bench/particle_fast.json`, and `sparse-match`):

```bash
./projman.sh bench extra
./projman.sh plot extra
./projman.sh tui extra
```

For interactive benchmark analysis (all `plot_two_json.sh` graph families, subject focus, exact point inspection, overlap-aware readouts), use:

```bash
./projman.sh tui --summary-tsv /tmp/infotheory-two-json-summary-<stamp>.tsv
```

Manual:

```bash
./projman.sh tui man
```

Optional online export after processing input:

```bash
./infotheory h file.txt --rate-backend mamba --method "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4;policy:schedule=0..100:infer" --model-export ./mamba_online.safetensors
```

This writes:
- `mamba_online.safetensors`
- `mamba_online.json` (sidecar with resolved config + metadata)

### AIXI Agent Mode
```bash
# Run the AIXI agent using config-specified backend
./infotheory aixi conf/kuhn_poker.json
```

Planner switch in config:

```json
{
  "planner": "aiqi",
  "rate_backend": {
    "kind": "ctw",
    "depth": 8
  },
  "random_seed": 12345,
  "discount_gamma": 0.99,
  "return_horizon": 6,
  "return_bins": 32,
  "augmentation_period": 6,
  "history_prune_keep_steps": 2048,
  "baseline_exploration": 0.01
}
```

Both planners also accept a `rate_backend` object using the same `RateBackend` schema and mixture language as the rest of the library. This is how the library's model class becomes the planner world model. Recursive `zpaq` is rejected here because these planner integrations need exact reversible or frozen conditioning during planning:

```json
{
  "planner": "aiqi",
  "rate_backend": {
    "kind": "ppmd",
    "order": 10,
    "memory_mb": 64
  }
}
```

Canonical planner configs should use `rate_backend` directly. Configs made for Infotheory versions prior to 1.2.0 should be ran through the conversion tool:  `./projman.sh legacy_aixi_convert <input_file>`

Example MC-AIXI convex mixture override:

```json
{
  "planner": "mc-aixi",
  "rate_backend": {
    "kind": "mixture",
    "spec": {
      "kind": "convex",
      "alpha": 1.25,
      "experts": [
        {"name": "ctw", "kind": "ctw", "depth": 8},
        {"name": "ppmd", "kind": "ppmd", "order": 8, "memory_mb": 16}
      ]
    }
  }
}
```

### AIXI Agent Mode (VM via Nyx-Lite)
```bash
# VM-backed environment using high-performance Firecracker (Nyx-Lite)
./infotheory aixi configs/aixi/vm_example.json
```

Quick benchmark (AIQI vs MC-AIXI):

```bash
./scripts/bench_aiqi_vs_aixi.sh
```

Reproducible competitor benchmark (Infotheory Rust/Python vs PyAIXI + C++ MC-AIXI):

```bash
./projman.sh bench_aixi_competitors --profile default --trials 1
```

Benchmark correctness notes:
- Stochastic environments are seeded from `random_seed` (or `rng_seed`) in CLI and Python run loops for reproducible trajectories.
- Reward reporting is normalized to native domain scale in competitor reports (for example Kuhn offset removal for C++/PyAIXI), so cross-implementation reward means are apples-to-apples.
- MC-AIXI tree search uses the same UCB scaling convention as common MC-AIXI reference implementations, the uniform-max tie-breaking rule from *A Monte-Carlo AIXI Approximation*, and chance-node cache keys that include reward as well as observation so environments with repeated observations but different rewards are handled correctly.

VM config highlights:
- **Environment**: Use `"environment": "vm"` (requires `vm` feature).
- **Core Config**:
  - `vm_config.kernel_image_path`: Path to `vmlinux` kernel.
  - `vm_config.rootfs_image_path`: Path to `rootfs.ext4`.
  - `vm_config.instance_id`: Unique ID for the VM instance.
- **Performance**:
  - `vm_config.shared_memory_policy`: Use `"snapshot"` for fast resets (fork-server style).
  - `vm_config.observation_policy`: `"shared_memory"` for zero-copy observations.
- **Rewards & Observations**:
  - `vm_reward.mode`: `"guest"` (guest writes to specific address), `"pattern"`, or `"trace-entropy"`.
  - `vm_observation.mode`: `"raw"` (bytes) or hash-based.
  - `observation_stream_len`: **Critical** for planning consistency. Must match guest output.

**Prerequisites**:
- Linux with KVM enabled (`/dev/kvm` accessible).
- `vmlinux` kernel and `rootfs.ext4` image valid for Firecracker.
- `vendor/nyx-lite` submodule (path dependency, kept outside the main workspace members).

**Setup**:
1. Ensure you have the `vmlinux-6.1.58` kernel in the project root (or update config).
2. Ensure `vendor/nyx-lite/vm_image/dockerimage/rootfs.ext4` exists or provide your own.
3. Enable the feature: `cargo build --release --features vm`.

---

## Library Usage

```rust
use infotheory::*;

// Entropy rate of a sequence (uses ROSA by default)
let h = try_entropy_rate_bytes(data).unwrap();

// Switch the entire thread to use CTW for all subsequent calls
set_default_ctx(
    InfotheoryCtx::from_specs(
        RateBackend::Ctw { depth: 32 },
        CompressionBackend::try_default().expect("default compression backend"),
    )
    .expect("ctw context"),
);
```

---

## Supported Primitives

| Command | Description | Domain |
| :--- | :--- | :--- |
| `ncd` | Normalized Compression Distance | Compression |
| `ned` | Normalized Entropy Distance | Shannon |
| `nte` | Variation of Information | Shannon |
| `mi`  | Mutual Information | Shannon |
| `id`  | Internal Redundancy | Algorithmic |
| `rt`  | Resistance to Transform | Algorithmic |
and more!
---

## Python Bindings (`infotheory-rs`)

This repository now includes PyO3/maturin bindings with package name:
- PyPI distribution: `infotheory-rs`
- Python import: `infotheory_rs`

Quickstart (local, via `uv`):

```bash
uv run maturin develop --release
uv run python -c "import infotheory_rs as ait; print(ait.ncd_paths('README.md','README.md', backend='zpaq', method='5', variant='vitanyi'))"
```

Python exposes both string-based backend parsing and direct backend objects. The
Python API includes `RateBackend.match(...)`, `RateBackend.sparse_match(...)`,
`RateBackend.ppmd(...)`, `RateBackend.mixture(...)`, `RateBackend.particle(...)`,
and `RateBackend.calibrated(...)`, plus `CalibrationContextKind` for calibrated
backends.

Example:

```python
import infotheory_rs as ait

match_backend = ait.RateBackend.match()
particle_backend = ait.RateBackend.particle(
    ait.ParticleSpec(num_particles=4, num_cells=4, cell_dim=8)
)
cal_backend = ait.RateBackend.calibrated(
    ait.RateBackend.ctw(8),
    ait.CalibrationContextKind.Text,
)

assert ait.entropy_rate_backend(b"abracadabra", backend=match_backend) >= 0.0
framed = ait.CompressionBackend.rate_rans(particle_backend, "framed")
blob = ait.compress_bytes_backend(b"payload", compression_backend=framed)
assert ait.decompress_bytes_backend(blob, compression_backend=framed) == b"payload"
assert ait.compress_size_backend(
    b"payload",
    compression_backend="rwkv7",
    method="cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer",
) > 0
```

Run Python tests:

```bash
uv run pytest -q python/tests
```

Run Python wrapper coverage (enforced in CI):

```bash
uv run pytest \
  --cov=infotheory_rs \
  --cov-report=term-missing \
  --cov-report=xml:target/python-coverage.xml \
  --cov-fail-under=100 \
  python/tests
```

For full developer test and coverage workflows (Rust + Python + VM), see:
`docs/developer-testing.md`.

Notes:
- Built as `abi3-py310` (compatible with Python 3.10+).
- Published wheels are intended to be portable and exclude `vm` support by default.
- Linux source builds can opt into VM bindings by enabling the Rust `vm` feature when building the extension.
  Example: `uv run maturin develop --release --features vm`
- Python trait-callback adapters (`PredictorABC`, `EnvironmentABC`, `AgentSimulatorABC`) are fail-fast:
  unhandled callback exceptions terminate the process after printing traceback context. This prevents
  silently continuing planning/search with invalid fallback values.

## License
- This is free software, which you may use under either the Apache-2.0 License, or the ISC License, at your choice. Those are available at LICENSE-APACHE and LICENSE respectively.
- Contributing to this repository means you agree to submit all contributions under the above Licensing arrangement. In other words, such that it is available to others under either license(ISC and Apache-2.0), at the others choice. 
- Don't forget to add your Copyright notice to the LICENSE file.
