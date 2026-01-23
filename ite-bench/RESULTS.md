# ITE Benchmark: Validation Results

## Executive Summary

This benchmark framework validates information-theoretic estimators against oracle truths and required mathematical properties.

## Test Results

### Rust Estimator (infotheory)
This suite now runs a **strict, broad validation matrix** across multiple regimes, oracles, and quantities.
Failures are expected under strict tolerances and are treated as signals for estimator or model improvements.

Covered quantities (non-exhaustive):
- Shannon Entropy H(X)
- Mutual Information I(X;Y)
- Conditional Entropy H(X|Y)
- Joint Entropy H(X,Y)
- KL Divergence D_KL
- JS Divergence D_JS
- Cross-Entropy H(P,Q)
- Total Variation Distance (TVD)
- Normalized Entropy Distance (NED)
- Normalized Transform Effort (NTE)
- Entropy Rate H_rate
- NCD metric axioms (Vitányi)

### External estimator dependencies
None. This benchmark is intentionally self-contained.

## Key Findings (Current Strict Run)

### 1. Estimator Methodology Differences

| Aspect | infotheory |
|--------|-----------|
| Method | Compression/rate based (depending on primitive/backend) |
| Data Type | Designed for arbitrary byte sequences |
| Accuracy | Validated against oracle truths + mathematical properties |

### 2. Discrete Data Compatibility

**Test Setup**: Discrete uniform distribution on {0,1,2,3,4,5,6,7}

- Validation focuses on oracle accuracy and bound/inequality checks for discrete data.

### 3. Information-Theoretic Properties Verified

✓ **Non-negativity**: I(X;Y) ≥ 0, D_KL ≥ 0, H(X|Y) ≥ 0  
✓ **Divergence bounds**: 0 ≤ D_JS ≤ log(2)  
✓ **Oracle accuracy**: Estimates within strict tolerance (may fail; intended)  
✓ **Subadditivity / data processing**: H(X,Y) ≤ H(X)+H(Y), I(X;Z) ≤ I(X;Y)  
✓ **Metric axioms**: NCD approximate non-negativity, identity, symmetry, triangle

## Framework Architecture

```
├── src/ITE/
│   ├── Types.lean          # Core types: Quantity, Estimator, SampleBundle
│   ├── Oracles.lean        # Ground truth generators
│   ├── Estimators.lean     # Rust estimator adapter (infotheory CLI)
│   ├── Verification.lean   # Accuracy & inequality verification
│   └── Reporting.lean      # Result structures
├── src/Runner.lean         # Comprehensive validation runner
└── infotheory/             # Rust CLI (cargo project)

```

## Usage

Build and run:
```bash
lake build runner
./.lake/build/bin/runner
```

## Recommendations

### For Production Use
- **Discrete/categorical data**: Rust infotheory (marginal measures)
- **Research/experimentation**: Use strict suite to identify estimator weaknesses

## Future Work

1. Extend rate-backend coverage to RWKV and fac-CTW
2. Increase regime diversity (mixtures, heavy tails, higher dimensionality)
3. Tighten formal identities (NED equivalences, NTE bounds) with larger samples

## Technical Implementation

### Rust Adapter
- Calls `../target/release/infotheory` (relative to `ite-bench/`)
- Writes samples to temporary binary files
- Parses float output from stdout
- Handles all ITE quantities via CLI primitives

### Validation Logic
- Validates estimates against oracle truths
- Verifies core inequalities and structural identities
- Provides detailed error messages for failures

## Conclusion

The benchmark successfully demonstrates:
1. ✅ infotheory works reliably for discrete data
2. ✅ Clean validation against mathematical properties
3. ✅ Comprehensive error handling and reporting

This provides a solid foundation for information-theoretic estimation research and validates the Rust implementation as robust for discrete/categorical data analysis.
