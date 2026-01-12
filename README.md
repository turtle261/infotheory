# InfoTheory

A high-performance Rust crate for Information Theoretic Estimators and Metrics.

This library provides a comprehensive suite of tools for quantifying complexity, dependence, and similarity between data sequences using both **Compression-based** (Kolmogorov Complexity) and **Entropy-based** (Shannon Information) approaches.

## Features

*   **NCD (Normalized Compression Distance)**: Uses ZPAQ compression to estimate information distance.
*   **Entropy & Mutual Information**:
    *   **Marginal**: Exact calculation for i.i.d. data (histograms).
    *   **Rate**: Predictive entropy rate estimation using ROSA (Suffix Automaton + Witten-Bell smoothing) for sequential data.
*   **Advanced Metrics**:
    *   **NED**: Normalized Entropy Distance.
    *   **NTE**: Normalized Transform Effort (Variation of Information).
    *   **TVD**: Total Variation Distance.
    *   **NHD**: Normalized Hellinger Distance.
    *   **KL / JS Divergence**: Kullback-Leibler and Jensen-Shannon divergences.
*   **Structural Primitives**:
    *   **Intrinsic Dependence**: Measures internal redundancy/predictability.
    *   **Resistance**: Measures information preservation under transformation.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
infotheory = { path = "." } # Or git repository
```

## CLI Usage

The crate includes a binary for command-line usage.

```bash
# Build the binary
cargo build --release

# Run NCD between two files
./target/release/infotheory ncd file1.bin file2.bin

# Calculate Mutual Information (Marginal / i.i.d.)
./target/release/infotheory mi file1.bin file2.bin 0

# Calculate Mutual Information (Entropy Rate, max_order=8)
./target/release/infotheory mi file1.bin file2.bin 8

# Calculate Intrinsic Dependence
./target/release/infotheory id file1.bin
```

### Supported Primitives

| Command | Description |
|---------|-------------|
| `ncd` | Normalized Compression Distance (Vitanyi) |
| `ncd_sym` | Symmetric NCD |
| `ned` | Normalized Entropy Distance |
| `nte` | Normalized Transform Effort |
| `mi` | Mutual Information |
| `entropy` | Shannon Entropy (Marginal or Rate) |
| `kl` | KL Divergence |
| `js` | JS Divergence |
| `id` | Intrinsic Dependence |
| `rt` | Resistance to Transformation |

## Library Usage

```rust
use infotheory::{ncd_vitanyi, mutual_information_bytes};

fn main() {
    let x = b"hello world hello world";
    let y = b"hello world hello earth";

    // Compression-based distance
    // Note: NCD functions typically take file paths, but byte-based variants exist.
    // let d = ncd_vitanyi("path/to/x", "path/to/y", "5");

    // Mutual Information (Marginal)
    let mi = mutual_information_bytes(x, y, 0);
    println!("MI (Marginal): {}", mi);

    // Mutual Information (Rate, context order 4)
    let mi_rate = mutual_information_bytes(x, y, 4);
    println!("MI (Rate): {}", mi_rate);
}
```

## Mathematical Details

### Compression-Based (NCD)
Approximates Kolmogorov complexity `K(x)` using compressed size `C(x)`.

```
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
```

### Entropy-Based (ROSA)
For sequential data, we estimate the entropy rate `Ĥ(X)` using a predictive model (ROSA) that builds a Suffix Automaton and applies Witten-Bell smoothing.

```
Ĥ(X) = -1/N * Σ log P(x_t | x_{<t})
```

This allows accurate estimation of Mutual Information and other metrics even for non-i.i.d. sources (e.g., text, code, DNA).


## TODO
 Levin Search
