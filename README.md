# InfoTheory

### 1. Unified Information Estimation
Estimate core measures using both **Marginal** (distribution-based) and **Rate** (predictive-based) approaches:
- **NCD (Normalized Compression Distance)**: Approximates information distance using real-world compressors (ZPAQ).
- **MI (Mutual Information)**: Quantifies shared information between sequences.
- **NED (Normalized Entropy Distance)**: A metric distance based on mutual information.
- **NTE (Normalized Transform Effort)**: Variation of Information (VI).
- **Intrinsic Dependence**: Redundancy Ratio.
- **Resistance**: Information preservation under noise/transform.

### 2. Multi-Backend Predictive Engine
Switch between different modeling paradigms seamlessly:
- **ROSA+ (Rapid Online Suffix Automaton + Witten Bell)**: A statistical LM. Default backend. Extremely fast online learning. Highly optimized for x86_64, memory tuned, parallelized, and with disk-caching.
- **CTW (Context Tree Weighting)**: Historically standard for AIXI. Accurate bit-level Bayesian model (KT-estimator).
- **RWKV (Neural Network)**: Highly optimized x86_64 RWKV7 LLM CPU inference kernel, and training (requires CUDA only for training).

### 3. Integrated MC-AIXI Agent
Includes a full implementation of the **Monte Carlo AIXI (MC-AIXI)** agent described by Hutter et al. This approximates the incomputable AIXI Agent using Monte-Carlo Tree Search, and is **backend-agnostic** and can utilize any of the available predictive backends (ROSA, CTW, or RWKV) for universal reinforcement learning.
As of my knowledge, this is the *first* real AIXI approximation that can be used for universal reinforcement learning, due to the libraries design of allowing any rate backend to be used rather than merely CTW. I am not aware of any other implementations of AIXI that are not merely CTW (Which performs poorly on non-Markovian tasks, and is not universal).

Provided, our library full includes native RWKV7 Model Training (Hybrid CPU/GPU) -- and a native optimized CPU inference Kernel(which will be faster than GPU for all but huge models). Training REQUIRES CUDA, but you can bring your own model instead. CPU Inference is explicitly SIMD optimized, for x86_64 -- so: non x86_64 architectures will be slower or perhaps not work at all for RWKV -- same goes for really old x86_64 without FMA/AVX2.
Therefore, you can use a trained RWKV7 model as a rate backend/"World Model" for MC-AIXI. Meaning, you can get information inside the Agent's mind before it ever makes a decision or plan. You can train the model on agent output, etc. 

---

## 🛠 Compilation & Installation
### Compiling Infotheory
X86_64 Linux TLDR: Install Rust, Clang, and do `cargo build --release`. That's all.
Infotheory is tested on x86_64 architecture only. It should work on other architectures, but I have not tested it yet.
It is known to work with the Following OS's:
- **Linux**: Install Rust via Rustup, and install clang++ and lld from your distribution's package manager.
- **FreeBSD**: `pkg install rust`
- **OpenBSD**: `pkg_add rust`
- **NetBSD**\*: `pkg_add rust clang lld` 

* NetBSD will need manual configuration to get this compiling, but is tested to work. Read the comments in the netbsd section of.cargo/.config.toml in this repository. TLDR: LTO breaks it on NetBSD, so disable it.

NOTE for NetBSD, OpenBSD, non-x86_64, and potentially other systems:
If your Kernel enforces W^X protection (as NetBSD and OpenBSD do), you will need to set the environment variable `CARGO_FEATURE_NOJIT` equal to something, such as "true". This is very important, as ZPAQ will fail at **runtime** otherwise.
If you are not using x86_64, ZPAQ JIT will also not work, and should be disabled.
You will get innacurate NCD results otherwise. JIT should work fine on Linux(x86_64!), and you should not set the env variable there--enjoy the better performance.


if using as a CLI:
0. Install dependencies as noted above.
1. Use git to clone the repository (recursively) -- configure as needed for your platform (x86_64 Linux, FreeBSD will work by default)
2. Run `cargo build --release` and the infotheory CLI will be present at `./target/release/infotheory`.

if using as a library:
Add the following to your `Cargo.toml`:

```toml
[dependencies]
infotheory = { path = "." } # Or git or whatever, you know rust.
```

### Building nyx-lite
`nyx-lite` is included as a workspace member. Build it with:
```bash
cargo build -p nyx-lite
```
Note: some nyx-lite tests require `/dev/kvm` and VM image artifacts under `nyx-lite/vm_image`.

### Additional notes
Some tests/dependencies which may be optional in some cases but not all:
- docker (for tests, or if you want to use it for rootfs generation)
- cpio
- wget (for tests, or to use the provided kernel. you can also use curl instead manually on the download_kernel.sh file )
- cmake (for VM feature, firecracker needs it)
- Lean4 (Toolchain Version 4.14.0)
---

## 💻 CLI Usage

The `infotheory` binary provides a powerful interface for file analysis.

### Information Theoretic Primitives
```bash
# Calculate Mutual Information (ROSA backend, order 8)
./infotheory mi file1.txt file2.txt 8

# Use CTW backend for NTE (Normalized Transform Effort)
./infotheory nte file1.txt file2.txt --rate-backend ctw

# Calculate NCD with custom ZPAQ method
./infotheory ncd file1.txt file2.txt 5
```

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

## 🦀 Library Usage

```rust
use infotheory::*;

// Entropy rate of a sequence (uses ROSA by default)
let h = entropy_rate_bytes(data, 8);

// Switch the entire thread to use CTW for all subsequent calls
set_default_ctx(InfotheoryCtx::new(
    RateBackend::Ctw { depth: 32 },
    NcdBackend::default()
));
```

---

## 📊 Supported Primitives

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

## 📄 License

Apache License, Version 2.0.
