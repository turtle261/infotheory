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

### AIXI Agent Mode (VM via libvirt + SSH)
```bash
# VM-backed environment using libvirt + SSH
./infotheory aixi aixi_confs/vm_example.json

# VM-backed environment with trace-entropy reward
./infotheory aixi aixi_confs/vm_trace_example.json
```

VM config highlights (see `aixi_confs/vm_example.json` for a full reference):
- `vm_config.domain`, `vm_config.snapshot`, `vm_config.transport` control libvirt lifecycle; use `transport: "ssh"` for SSH-only operation.
- `vm_config.ssh` configures SSH (host, port, user, password/key) and the per-action command to run in the guest.
- `vm_config.auto_snapshot` enables automatic snapshot creation when missing, after provisioning steps run.
- `vm_config.ssh.provision_steps` is a dockerfile-like list of `run`, `upload`, or `script` steps executed via SSH before snapshot creation.
- `vm_config.stats_backend` selects the rate backend used for info-theoretic scoring (entropy, novelty). If omitted, it defaults to the agent algorithm settings.
- `vm_protocol` optionally defines a line-based wire protocol (`OBS`, `REW`, `DATA`) if your SSH command emits structured output.
- `vm_actions` can be literal payloads or `fuzz` mutators with seeds and dictionaries; for SSH, payloads are sent to stdin when `vm_config.ssh.action_command.stdin_payload=true`.
- `vm_reward` supports guest-provided rewards, pattern matches, entropy-reduction, or trace-entropy signals.
- `vm_trace` can pull trace bytes via `ssh` command output when `vm_reward.mode = "trace-entropy"` (see `aixi_confs/vm_trace_example.json`).
  - Example: `vm_trace.mode = "ssh"` with `command = { "cmd": "cat", "args": ["/tmp/trace.bin"] }`.
- `vm_observation` controls observation streams:
  - `mode: "raw"` streams raw output bytes as observation symbols (no hashing).
  - `stream_len` + `stream_mode` define fixed-length normalization for planning consistency.
- `observation_stream_len` and `observation_key_mode` select how observation streams map to search-tree keys (`first`, `last`, `stream-hash`).
- `discount_gamma` (optional) enables discounted UCT for long-horizon approximations.

Tip: For raw text output, set `vm_observation.mode = "raw"` and `observation_bits = 8`, and choose a fixed `stream_len` to keep planning consistent.
- `vm_filter` enables optional info-theoretic pruning gates (entropy / intrinsic dependence / novelty).

Prerequisites:
- libvirt daemon accessible via the configured `libvirt_uri` (uses the `virt` crate bindings).
- Guest OS with SSH enabled (root login allowed for isolated VMs), and an IP/port reachable from the host.
- libssh2 available for the `ssh2` crate (package name is usually `libssh2`).

SSH setup quick notes:
- Enable root SSH login (isolated VM), or use a dedicated user and set `vm_config.ssh.user`.
- Set `vm_config.ssh.action_command` to the command you want each action to run; when `stdin_payload=true`, the fuzzed bytes are piped to stdin.
- Use `vm_config.ssh.provision_steps` to install packages, add users, and copy scripts before auto-snapshot creation.

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

---

## 📄 License

Apache License, Version 2.0.
