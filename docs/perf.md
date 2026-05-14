# Performance Tuning

## AC fast-bitwise compile-time configuration

In the AC compression/decompression paths, there exist some compile-time configurations that can affect runtime performance varyingly depending on the usecase.

These are controlled crate-locally through `crates/infotheory/build.rs` with environment variables, so global `RUSTFLAGS` edits are not required.

```bash
# default:
#   encode: inline
#   decode: deinline
cargo build -p infotheory --release --features cli --bin infotheory --locked

# both de-inlined:
INFOTHEORY_AC_ENCODE_DEINLINE=1 \
cargo build -p infotheory --release --features cli --bin infotheory --locked

# both inlined:
INFOTHEORY_AC_DECODE_INLINE=1 \
cargo build -p infotheory --release --features cli --bin infotheory --locked

# custom combination (example for benchmarking):
INFOTHEORY_AC_ENCODE_DEINLINE=1 \
INFOTHEORY_AC_DECODE_INLINE=1 \
CARGO_INCREMENTAL=0 \
cargo build -p infotheory --release --features cli --bin infotheory --locked
```
## AC fast-bitwise helper inlining matrix

The AC fast-bitwise encode/decode helper boundary is intentionally compile-time configurable. A local matrix benchmark was run on the `two-json` suite to compare the four layout variants against the default monolithic baseline.

Benchmark conditions:

- Suite: `two-json`
- Backend: `rate-ac`
- Sizes: `4096 16384 65536 262144 1048576 2097152 4194304`
- Repeats: 3
- Warmups: 0
- Metric: mean of summed raw `real_seconds` per repeat
- Output stability: all variants preserved archive bytes, entropy values, and verification status.

Variant meanings:

| Variant | Encode helper | Decode helper |
|---|---|---|
| `baseline` | original inline body | original inline body |
| `both-deinlined` | `inline(never)` | `inline(never)` |
| `comp-inlined` | `inline(always)` | `inline(never)` |
| `dec-inlined` | `inline(never)` | `inline(always)` |
| `both-inlined` | `inline(always)` | `inline(always)` |

Whole-matrix result:

| Variant | Mean total time | Stdev | Delta vs baseline |
|---|---:|---:|---:|
| `baseline` | `2136.093s` | `1.459s` | — |
| `both-deinlined` | `2139.473s` | `1.223s` | `+0.158%` |
| `comp-inlined` | `2135.147s` | `0.257s` | `-0.044%` |
| `dec-inlined` | `2135.733s` | `2.072s` | `-0.017%` |
| `both-inlined` | `2133.747s` | `1.427s` | `-0.110%` |

Targeted effects:

| Metric | Baseline | `both-deinlined` | `comp-inlined` | `dec-inlined` | `both-inlined` |
|---|---:|---:|---:|---:|---:|
| `match` decompress | `4.347s` | `4.027s` (`-7.36%`) | `4.073s` (`-6.29%`) | `4.023s` (`-7.44%`) | `4.393s` (`+1.07%`) |
| `rwkv7` total | `328.420s` | `330.987s` (`+0.78%`) | `329.157s` (`+0.22%`) | `330.503s` (`+0.63%`) | `327.790s` (`-0.19%`) |
| `rwkv7` compress+decompress | `223.267s` | `224.713s` (`+0.65%`) | `223.503s` (`+0.11%`) | `224.173s` (`+0.41%`) | `222.637s` (`-0.28%`) |
| `neural_mixture` total | `1103.503s` | `1104.830s` (`+0.12%`) | `1104.813s` (`+0.12%`) | `1103.713s` (`+0.02%`) | `1102.653s` (`-0.08%`) |
| `ctw` total | `349.577s` | `349.140s` (`-0.12%`) | `348.550s` (`-0.29%`) | `348.843s` (`-0.21%`) | `348.923s` (`-0.19%`) |
| non-`match` total | `2126.863s` | `2130.570s` (`+0.17%`) | `2126.177s` (`-0.03%`) | `2126.827s` (`-0.00%`) | `2124.470s` (`-0.11%`) |

Chosen default:

| Helper | Default |
|---|---|
| encode fast-bitwise AC helper | `inline(always)` |
| decode fast-bitwise AC helper | `inline(never)` |

This corresponds to the `comp-inlined` variant.
