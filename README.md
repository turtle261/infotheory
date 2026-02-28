# InfoTheory

### 1. Unified Information Estimation
Estimate core measures using both **Marginal** (distribution-based) and **Rate** (predictive-based) approaches:
- **NCD (Normalized Compression Distance)**: Approximates information distance using compression.
- **MI (Mutual Information)**: Quantifies shared information between sequences.
- **NED (Normalized Entropy Distance)**: A metric distance based on mutual information.
- **NTE (Normalized Transform Effort)**: Variation of Information (VI).
- **Intrinsic Dependence**: Redundancy Ratio.
- **Resistance**: Information preservation under noise/transform.

### 2. Multi-Backend Predictive Engine
Switch between different modeling paradigms seamlessly:
- **ROSA+ (Rapid Online Suffix Automaton + Witten Bell)**: A fast statistical LM. Default backend. 
- **CTW (Context Tree Weighting)**: Historically standard for AIXI. Accurate bit-level Bayesian model (KT-estimator).
- **RWKV (Neural Network)**: Portable SIMD RWKV7 CPU inference backend (`wide`-based).

### 3. Integrated MC-AIXI Agent
Includes a full implementation of the **Monte Carlo AIXI (MC-AIXI)** agent described by Hutter et al. This approximates the incomputable AIXI Agent using Monte-Carlo Tree Search, and is **backend-agnostic** and can utilize any of the available predictive backends (ROSA, CTW, or RWKV) for universal reinforcement learning.

You can use a trained RWKV7 model as a rate backend ("world model") for MC-AIXI.

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
- Cross-target compile validation for RWKV portability:
  - **AArch64 Linux (GNU/musl)** (`aarch64-unknown-linux-gnu`, `aarch64-unknown-linux-musl`)
  - **AArch64 Windows** (`aarch64-pc-windows-msvc`)
  - **WASM** (`wasm32-unknown-unknown`, RWKV-only/no-zpaq profile)

<small>WASM support is compile-target validation for the RWKV path (no zpaq/VM feature path).</small>

### Build Prerequisites
- Rust toolchain (stable): `rustup` recommended.
- C/C++ toolchain: `clang` + `lld` recommended on Unix-like systems.
- For local repository builds with VM support available: clone recursively (`--recurse-submodules`) so `nyx-lite` is present.

### Build Configuration
- By default, .cargo/config.toml is set to use march=native as the target-cpu, which will allow LLVM to make full use of your specific CPU. This can improve performance by roughly 2x for the RWKV Model. This may affect binary compatibility depending on your usecase.

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
infotheory = { path = "." } # Or git or whatever, you know rust.
```

### Building nyx-lite
The VM backend is optional (`--features vm`) and depends on `nyx-lite` (and its vendored submodule code). Build it with:
```bash
cargo build --release --features vm
```
Notes:
- VM is Linux/KVM-oriented (`/dev/kvm` required).
- Some `nyx-lite` tests also require VM image artifacts under `nyx-lite/vm_image`.

### Additional notes
Platform caveats:
- **OpenBSD/NetBSD**: kernel W^X policies can break ZPAQ JIT at runtime. Set `CARGO_FEATURE_NOJIT=true`.
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
# Calculate Mutual Information (ROSA backend, order 8)
./infotheory mi file1.txt file2.txt 8

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

### RWKV Method Strings

RWKV can be configured with either a model file or compact method string:

- `file:/abs/or/relative/model.safetensors`
- `cfg:key=value,...`

Supported `cfg:` keys:
`hidden,layers,intermediate,decay_rank,a_rank,v_rank,g_rank,seed,train,lr,stride`

`train` supports: `none`, `sgd`, `adam`.

Example:

```bash
./infotheory h file.txt \
  --rate-backend rwkv7 \
  --method "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=7,train=sgd,lr=0.01,stride=1"
```

Optional online export after processing input:

```bash
./infotheory h file.txt --rate-backend rwkv7 --method "cfg:hidden=64,layers=1,intermediate=64" --rwkv-export ./rwkv_online.safetensors
```

This writes:
- `rwkv_online.safetensors`
- `rwkv_online.json` (sidecar with resolved config + metadata)

### AIXI Agent Mode
```bash
# Run the AIXI agent using config-specified backend
./infotheory aixi conf/kuhn_poker.json
```

### AIXI Agent Mode (VM via Nyx-Lite)
```bash
# VM-backed environment using high-performance Firecracker (Nyx-Lite)
./infotheory aixi aixi_confs/vm_example.json
```

VM config highlights:
- **Environment**: Use `"environment": "nyx-vm"` or `"vm"` (requires `vm` feature).
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
- `nyx-lite` crate (included in workspace).

**Setup**:
1. Ensure you have the `vmlinux-6.1.58` kernel in the project root (or update config).
2. Ensure `nyx-lite/vm_image/dockerimage/rootfs.ext4` exists or provide your own.
3. Enable the feature: `cargo build --release --features vm`.

---

## Library Usage

```rust
use infotheory::*;

// Entropy rate of a sequence (uses ROSA by default)
let h = entropy_rate_bytes(data, 8);

// Switch the entire thread to use CTW for all subsequent calls
set_default_ctx(InfotheoryCtx::new(
    RateBackend::Ctw { depth: 32 },
    CompressionBackend::default()
));
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
