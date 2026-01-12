# InfoTheory

A high-performance, versatile Rust crate for **Information Theoretic Primitives**, **Sequential Metrics**, and **Autonomous Agents**.

`infotheory` provides a unified framework for quantifying complexity, dependence, and similarity between data sequences. It bridges classical Shannon entropy with algorithmic information theory (Kolmogorov Complexity), supported by multiple predictive backends.

## 🚀 Key Features

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
- **ROSA (Suffix Automaton)**: Default backend. Extremely fast online learning with Witten-Bell smoothing.
- **CTW (Context Tree Weighting)**: Historically standard for AIXI. Accurate bit-level Bayesian model (KT-estimator).
- **RWKV (Neural Network)**: Modern neural sequence prediction (requires CUDA).

### 3. Integrated MC-AIXI Agent
Includes a full implementation of the **Monte Carlo AIXI (MC-AIXI)** agent. Unlike traditional implementations restricted to CTW, this agent is **backend-agnostic** and can utilize any of the available predictive backends (ROSA, CTW, or RWKV) for universal reinforcement learning.

---

## 🛠 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
infotheory = { path = "." }
```

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
