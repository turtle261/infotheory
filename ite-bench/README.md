# ITE-Bench: Information Theoretic Estimator Benchmark

A self-contained validation framework for information-theoretic estimators.

## Overview

This project validates the `infotheory` Rust crate against mathematically-defined oracle truths (for synthetic data) and fundamental information-theoretic identities/inequalities. It uses **Lean 4** as a test runner and orchestrator to ensure rigorous verification.

## Directory Structure

*   `infotheory/`: The Rust crate containing the estimator implementations (ZPAQ, ROSA, CTW, RWKV backends, etc.).
*   `src/`: Lean 4 source code for the benchmark runner and data generation oracles.
*   `lakefile.lean`: Build configuration for the Lean project.

## Running the Benchmark

To run the full validation suite:

```bash
lake build
lake exe runner
```

This will:
1.  Compile the Rust `infotheory` binary.
2.  Generate synthetic data (Uniform, Independent, etc.) using Lean oracles.
3.  Run the Rust estimator on this data.
4.  Verify results against Oracle Truth (theoretical value) and structural identities.

## Validation Results

The benchmark currently tests:
1.  **Shannon Entropy $H(X)$**
2.  **Mutual Information $I(X;Y)$**
3.  **KL Divergence $D_{KL}(P||Q)$**
4.  **Jensen-Shannon Divergence $D_{JS}(P||Q)$**
5.  **Conditional Entropy $H(X|Y)$**
6.  **Joint Entropy $H(X,Y)$**
7.  **Cross-Entropy $H(P,Q)$** 

### Findings

*   **Rust Estimator (`infotheory`)**:
    *   Correctly handles discrete data (integers, bytes).
    *   Matches Oracle truths with high accuracy for uniform distributions.
    *   Passes all non-negativity and bound checks.
    *   Supports both i.i.d. (Marginal) and sequential (Rate) estimation.

This benchmark intentionally does not depend on any external estimator codebase. All validation is performed via oracle truths and mathematically-required properties.

## Methodology

The benchmark generates "bundles" of data with known properties (e.g., two independent uniform random variables). It then asks each estimator to compute metrics on these bundles.

*   **Accuracy**: $|Estimate - Truth| \le Tolerance$
*   **Internal consistency**: checks that quantities satisfy required relationships and bounds.
*   **Properties**: Checks mathematical bounds (e.g., $I(X;Y) \ge 0$).

## Mathematical Rigor

The `infotheory` crate implements estimators based on:
*   **ZPAQ**: A high-ratio compressor for approximating Kolmogorov Complexity.
*   **ROSA**: A Suffix Automaton with Witten-Bell smoothing for estimating Entropy Rate $\hat{H}(X)$.

These allow for "universal" distance metrics (NCD, NED) that work on arbitrary byte sequences without assuming a specific parametric distribution.
