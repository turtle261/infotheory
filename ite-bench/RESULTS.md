# ITE Benchmark: Validation Results

## Executive Summary

This benchmark framework validates information-theoretic estimators against oracle truths and required mathematical properties.

## Test Results

### Rust Estimator (infotheory)
✅ **7/7 tests passed**

All quantities correctly estimated with compression-based methods:
- Shannon Entropy H(X)
- Mutual Information I(X;Y)
- KL Divergence D_KL
- JS Divergence D_JS
- Conditional Entropy H(X|Y)
- Joint Entropy H(X,Y)
- Cross-Entropy (Rust-only)

### External estimator dependencies
None. This benchmark is intentionally self-contained.

## Key Findings

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
✓ **Oracle accuracy**: Estimates within sample tolerance
✓ **Subadditivity**: H(X,Y) ≤ H(X) + H(Y)

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
- **Discrete/categorical data**: Rust infotheory
- **Research/experimentation**: Rust (more robust)

## Future Work

1. Increase regime coverage (sample size, alphabet size, stationarity)
2. Add deeper end-to-end tests for rate backends (ROSA/CTW/RWKV) and entropy-rate primitives
3. Extend formal verification layer to cover more identities/inequalities and edge-cases

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
