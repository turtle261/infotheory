# Infotheory Performance Optimization Roadmap - 2026-05

This is a local, disposable working document for choosing the next performance and memory-efficiency experiments in Infotheory. It is deliberately not a permanent design spec. Its purpose is to keep optimization work scientifically grounded, Hutter-relevant, and resistant to seductive but low-yield micro-tuning.

The current branch state materially changes the prioritization:

1. **CTW's shared log cache asymptotic memory problem is addressed for this branch scope**. The bounded exact log-cache work landed, the hot-prefix sweep was completed, and depth `12` is the current chosen default. CTW should now be treated as done for this tranche unless later validation uncovers a new issue.
2. **PPMD's rolling suffix key and exact-entry update work landed, while sparse/exact query routing was investigated and rejected under the current representation**. Further PPMD work is now a later representation question, not the active branch target.
3. **RWKV7's next memory/time work should target TBPTT replay workspace churn before exotic kernels**. Fixed-shape GEMV specialization is plausible, but the current online TBPTT path still allocates replay-local vectors, per-step PDF vectors, traces, checkpoints, full gradient state, and recurrent gradient state per segment flush. Reusing this workspace and freeing it after the final full-train window is the active `A(n)/Q(n)/S(n)` target for this branch.
4. **Expert marginal utility still matters, but it is not the immediate branch task**. Ablation remains important for later resource-MDL decision-making, yet it is intentionally out of scope until the current optimization tranche is satisfactorily implemented.
5. **AC inlining and binary decode are now guardrail knobs, not the main research frontier**. They should remain in the benchmark matrix, but further AC work is lower priority unless a new profile contradicts the current evidence.

## Hutter-aware objective

For exact implementation optimizations, the predictive distribution and entropy coder semantics should be preserved. For model-changing experiments, the relevant objective is a resource-bounded MDL score, not raw archive size alone.

Useful decomposition:

```text
W(n): semantic and implementation arithmetic per byte
Q(n): copied bytes / memory traffic per byte
A(n): allocation events / allocator pressure per byte
S(n): live state size / peak RSS / cache footprint
B:    release binary/program/model size after stripping/UPX when relevant
L(n): compressed codelength/archive bytes for the target corpus
```

For a Hutter-like candidate, think in terms of:

```text
J(candidate) =
    archive_bytes
  + program_and_model_description_bytes
  + hard penalties for time, RAM, disk, and decompression failures
  + local soft penalties used only for search/ranking
```

The implementation tracks in this document should reduce one or more of `W/Q/A/S/B` while preserving `L(n)` exactly or nearly exactly. The model-science tracks may intentionally change `L(n)`, but then they must improve the measured resource frontier rather than merely be “interesting.”

## Hutter feasibility gates

The Hutter Prize resource envelope changes which optimizations are rational. For enwik9-scale work, the hard constraints imply:

```text
input bytes:        1_000_000_000
time limit:         50 hours
minimum throughput: ~5_556 bytes/s, before decompression/verification overhead
RAM limit:          10 GB
disk limit:         100 GB
target archive:     < 110 MB to beat the current record class
```

The current 10 MB neural-mixture timing is not obviously time-infeasible under linear extrapolation, but its memory curve is the dangerous term. A 2.6 GiB RSS at 10 MB is already too close to the RAM limit if any major component grows linearly or superlinearly. Therefore the immediate Hutter-relevant question is:

```text
Can the best compression-effective mixture be made sublinear/bounded enough in memory
while preserving most of its codelength advantage?
```

Every proposed optimization should be classified into one of four buckets:

1. **Feasibility fix**: changes an impossible asymptotic or large constant that blocks enwik9-scale execution.
2. **Frontier improvement**: improves archive/time/RSS Pareto position on verified prefixes.
3. **Search enabler**: makes future model/tuner exploration cheaper without degrading compression.
4. **Cosmetic/local micro-win**: does not change feasibility or the Pareto frontier; deprioritize.

The first two buckets are the only default priorities for Hutter progress. Search enablers are valuable when they unlock many evaluations. Cosmetic/local wins should be kept only if they are simple, exact, and non-invasive.

## Evidence baseline

The current `configs/bench/two.json` neural mixture is:

```text
ctw(depth=32)
ppmd(order=12, memory_mb=256)
rosaplus(max_order=-1)
match
rwkv7(hidden=64,layers=1,intermediate=64,ranks=16,early full training, long inference tail)
```

Representative measured/profile facts:

- **Neural mixture compression is strong but slow**: for 262144 bytes, neural mixture archive size is around `67454` bytes, versus CTW around `80004` and RWKV around `137010`.
- **Neural mixture runtime is expert-bound**: the neural combiner itself is around 1% self-time in the cited profile; CTW, RWKV, PPMD, and ROSA dominate.
- **Large-prefix memory is expert-state dominated**: 10 MB neural mixture RSS in the current checked-in data is around 2.6 GiB.
- **RWKV standalone RSS is low for the tiny config**, but its arithmetic and training replay are significant.
- **CTW standalone memory and time grow materially**, and its hidden shared-log-cache term is asymptotically dangerous at enwik9 scale.

Representative 262144-byte neural-mixture compression self-time:

```text
CtEngine::predict_one                              ~16.05%
rwkv7::Model::forward_with_sink                   ~15.74%
rwkv7::Model::accumulate_token_step_gradients     ~10.29%
exp                                                ~9.61%
CtEngine::update_prepared                          ~6.35%
PpmdModel::ensure_pdf_inner                        ~5.22%
RosaPlus::fill_probs_for_last_bytes                ~4.10%
PpmdModel::update                                  ~3.17%
RosaPlus::train_byte                               ~2.80%
rwkvzip::Compressor::logits_to_pdf                ~1.96%
RatePdfPredictor::prepare_cached_cdf_fast_bitwise  ~1.77%
NeuralMixCore::compute_context_mixtures            ~1.03%
MatchModel::ensure_pdf_inner                       ~1.02%
```

Important current-code facts verified in the second pass:

- `PpmdModel::context_key(ord)` hashes the suffix from scratch, and `PpmdModel::ensure_pdf_inner` rebuilds dense 256-entry distributions order by order.
- `MixturePredictor::ac_step_bitwise` already uses CTW's bitwise prepared path, and for several non-CTW experts calls `prepare_cached_cdf_fast_bitwise`, which still materializes full CDFs.
- `CtwPredictor::bit_prob_one_msb` already calls `FacContextTree::predict_one`, and `CtwPredictor::update_bit_msb` already calls `update_predicted`. A naive “reuse prediction path for update” idea is redundant.
- `CtEngine::with_logs` previously used a thread-local `SharedLogCache` with two `Vec<f64>` caches growing to `root_visits + 1`; this branch now bounds that exact cache path.
- The CTW hot-prefix sweep for this branch was completed and selected `12` as the current default.
- `Model::online_train_segment_tbptt` allocates replay-local gradient/checkpoint/trace/PDF structures inside each segment training call.
- `OnlineRuntime` keeps `full_tbptt` and `full_adam` state after they become unnecessary unless explicit lifecycle logic is added.

## Acceptance rules

### Exact implementation optimizations

Use these for changes that claim not to change the model:

- **Archive bytes**: exact archive parity is preferred; any byte difference must be treated as a floating-order/model-change event until proven harmless.
- **Roundtrip**: decompressed bytes must equal original bytes.
- **Entropy/loss**: `h` output should be bit-identical or within an explicitly justified floating tolerance only if the operation order changed.
- **Speed**: require a stable whole-subject win, not only a microbenchmark win. For hot expert changes, target at least 1-2% whole-subject improvement or a clear allocation/RSS win.
- **Memory**: for Hutter-scale asymptotic fixes, an optional mode can be accepted even with speed cost if it changes the feasibility class.
- **Binary size**: any specialization that increases text size must report stripped and optionally UPX-compressed binary deltas.

### Model-changing or compute-allocation experiments

Use these for expert pruning, approximate scheduling, changed floating semantics, or altered probability models:

- **Primary**: archive bytes and decompression correctness.
- **Secondary**: time, RSS, temp disk, binary/model bytes.
- **Decision metric**: accept only if the Pareto frontier improves. A faster but materially worse compressor is useful only as an explicit speed-tier configuration, not as the default Hutter candidate.

## Measurement protocol

### Inner loop

For local exact changes:

```bash
cargo build -p infotheory --release --features cli --bin infotheory --locked

INFOTHEORY_BENCH_WARMUPS=0 \
INFOTHEORY_BENCH_REPEATS=2 \
INFOTHEORY_BENCH_SUITE=two-json \
INFOTHEORY_BENCH_CPU=11 \
INFOTHEORY_BENCH_FRESH=1 \
INFOTHEORY_BENCH_SUBJECTS='ctw rwkv7 neural_mixture ppmd rosa match' \
INFOTHEORY_BENCH_SIZES=262144 \
./projman.sh bench
```

For allocation/memory hypotheses, do not trust wall time alone. Capture at least:

```text
real_seconds_mean
throughput_mib_s_mean
rss_kib_mean
archive_bytes_mean
verified_all
```

If available, supplement with `perf stat -d --repeat 3` and an allocation profiler. Do not make those tools mandatory for correctness.

### Scale loop

For candidates that survive the inner loop:

```bash
INFOTHEORY_BENCH_WARMUPS=0 \
INFOTHEORY_BENCH_REPEATS=2 \
INFOTHEORY_BENCH_SUITE=two-json \
INFOTHEORY_BENCH_CPU=11 \
INFOTHEORY_BENCH_FRESH=1 \
INFOTHEORY_BENCH_SUBJECTS='ctw rwkv7 neural_mixture ppmd rosa' \
INFOTHEORY_BENCH_SIZES='262144 1048576 4194304 10000000' \
./projman.sh bench
```

For Hutter-relevance, eventually test prefixes from:

```text
~/dev/rmix/enwik9.zst
```

Use controlled decompressed prefixes rather than repeatedly manipulating the full compressed source.

Do not run full coverage or repeated long `llvm-cov` in the performance inner loop. Use targeted tests and `git diff --check`; reserve full coverage for finalized changes.

## Universal kill criteria

Apply these before spending multiple implementation days on any path:

- **No verified bottleneck**: kill or defer if the target function/state is not visible in profile/RSS/allocation data for the relevant subject.
- **No parity story**: kill exact-optimization framing if the change cannot specify how archive parity or dense-reference equivalence will be tested.
- **No scaling story**: kill Hutter-priority status if the improvement is only a small constant on 262144-byte data and does not plausibly change 10 MB/enwik9 time or memory.
- **No marginal codelength value**: kill deep optimization of an expert if ablation shows it does not materially improve compressed size in the mixture.
- **Binary-size negative trade**: kill fixed-specialization work if speed gain is smaller than noise or does not justify added program bytes.
- **Complexity pollution**: kill patches that make core abstractions less truthful or harder to audit unless the measured frontier gain is substantial.

Use “kill” literally: the best outcome of many experiments is an early, well-evidenced decision not to continue.

## Current tranche status and active next target

For this branch, the tranche ordering above is no longer future work; most of it is already resolved:

1. **CTW feasibility work is complete for this branch scope.**
   - Bounded exact log lookup landed.
   - The hot-prefix sweep was completed.
   - Depth `12` is the chosen branch default.
   - Treat further CTW work as closed unless later testing reveals a new issue.
2. **PPMD exact optimization work is complete for this branch scope.**
   - Rolling suffix keys landed.
   - Exact-entry update improvements landed.
   - Sparse/exact query routing was investigated and rejected under the current representation.
3. **RWKV7 replay-workspace reuse is the active implementation target.**
   - Remove per-flush allocation churn in `Model::online_train_segment_tbptt`.
   - Reuse checkpoints, step states, traces, PDF storage, full gradient state, recurrent gradient state, and bias-gradient scratch.
   - Preserve training semantics exactly.
   - Prefer runtime-local caching rather than widening public API surface.
4. **RWKV7 post-training lifecycle cleanup remains a later follow-up.**
   - Releasing no-longer-needed full-training state is still useful, but it is secondary to removing replay allocation churn.

This tranche is deliberately centered on finishing the current exact optimization path, not on switching to ablation or another CTW round.

## Post-feasibility resource-MDL search

After the P0 feasibility fixes, stop hand-choosing a single mixture by intuition. Use a finite, auditable search over configurations under the resource-bounded objective `J`.

Candidate dimensions:

```text
expert subset:       CTW, PPMD, ROSA, Match, RWKV7
CTW:                 depth, hot-prefix depth, log-cache threshold/mode
PPMD:                order, memory cap, query mode, compact-count mode
ROSA:                max_order and bounded-state options if implemented
RWKV7:               hidden, layers, ranks, train policy, optimizer, BPTT window
mixture:             Bayes/Fading/Neural, learning rates, context features
coder:               AC/rANS only if archive/time tradeoff is relevant
```

Do not make this an unbounded heuristic sweep. Use a finite grammar and report:

```text
candidate description bytes
archive bytes
time
RSS
binary/model bytes
verification result
objective J
```

The point is MDL discipline: a larger or more complex model must pay for itself in codelength under the actual resource envelope. This search should happen only after CTW/PPMD memory feasibility is no longer confounding the results.

## Priority table

| Priority | Track | Main term | Why it is high-signal |
|---|---|---:|---|
| P0 | RWKV7 TBPTT replay workspace reuse | `A(n), Q(n), S(n)` | Current TBPTT replay still allocates large fixed-shape scratch structures per flush; reuse is the clearest remaining exact optimization target on this branch. |
| P0 | PPMD probability-query interface | `W(n), Q(n)` | Avoids dense 256-way interpolation/CDF construction when only `P(symbol)` or binary interval masses are needed. |
| P1 | PPMD rolling suffix keys + update lookup tightening | `W(n)` | Exact `Theta(k^2) -> Theta(k)` key maintenance at order `k`, localized and low semantic risk. |
| P1 | RWKV7 TBPTT replay workspace reuse | `A(n), Q(n), S(n)` | Current full-training replay allocates per segment; profile shows training gradients are expensive. |
| P1 | Expert marginal-utility audit | `L/W/S` | Prevents optimizing costly experts that do not pay their codelength/resource rent. |
| P1 | CTW representation phase diagram | `W(n), S(n)` | Hot-prefix depth and segment representation are a core speed/memory frontier. |
| P2 | PPMD compact continuation storage | `S(n), A(n)` | Likely strong after query/key fixes, but more invasive than rolling keys. |
| P2 | RWKV7 fixed-shape GEMV specialization | `W(n), B` | Plausible for `64/16/256` shapes, but binary-size and floating-order risks must be measured. |
| P2 | ROSA+ probability/memory audit | `W(n), S(n)` | Relevant but more invasive; should follow direct measurement of ROSA's marginal utility. |
| P3 | AC inlining/helper tuning | `W(n)` | Already improved; remaining deltas are likely backend-specific knobs. |

## P0 track: CTW bounded exact log lookup

### Files

```text
crates/infotheory/src/backends/ctw.rs
```

### Current implementation

`SharedLogCache` stores:

```rust
log_int: Vec<f64>
log_half: Vec<f64>
```

and `with_shared_log_cache(upto, ...)` ensures both arrays contain every entry through `upto`. CTW update/revert paths call this with visit counts derived from `root_visits`.

At `n = 1_000_000_000`, two `f64` arrays are about:

```text
2 * 8 * n = 16 GB
```

That is already beyond the Hutter RAM limit before nodes, segments, experts, neural state, entropy coder state, and OS overhead. Therefore this is not a micro-optimization. It is a feasibility issue.

### Hypothesis

Replace unbounded direct slices with a bounded exact lookup abstraction:

```text
LogLookup {
    cached_int: &[f64],
    cached_half: &[f64],
    cache_limit: usize
}

log_int(n):
    if n < cached_int.len(): cached_int[n]
    else if n == 0: -inf
    else: ln(n as f64)

log_half(n):
    if n < cached_half.len(): cached_half[n]
    else: ln(n as f64 + 0.5)
```

Then cap cache growth at a configurable threshold, such as:

```text
2^16, 2^20, 2^22, unlimited
```

The first implementation should be private and compile-time or test-configurable. Do not expose a public API until the speed/memory phase diagram is known.

Thresholds should be chosen by memory budget, not aesthetics:

```text
cache_bytes ≈ 2 * threshold * sizeof(f64)
2^20 entries ≈ 16 MiB
2^22 entries ≈ 64 MiB
2^24 entries ≈ 256 MiB
```

For the default library mode, prefer the smallest threshold that preserves speed on current benchmarks. For a Hutter-specific memory-first mode, a lower threshold is acceptable if total RSS stays inside the 10 GB envelope and the time penalty remains under the 50-hour bound.

### Why this is mathematically valid

The cached value is exactly the same expression currently inserted into the cache:

```text
(n as f64).ln()
(n as f64 + 0.5).ln()
```

Therefore, above-threshold direct computation should be semantically identical modulo libm determinism of the same expression. Archive parity is plausible and must be tested.

### Implementation caution

Do not insert an unpredictable branch at every hot arithmetic site if avoidable. Prefer passing a small `LogLookup` object with `#[inline(always)]` accessors, and keep the cached fast path branch simple. Audit all direct `log_int[...]` and `log_half[...]` uses.

### Expected impact

- **Reduces**: `S(n)` from unbounded cache memory to `O(threshold)`.
- **May increase**: `W(n)` through direct `ln` calls for high counts.
- **Hutter relevance**: mandatory if CTW/FAC-CTW is used on enwik9-scale streams.

### Acceptance

- Exact archive parity on small and medium CTW/neural-mixture runs if possible.
- RSS decreases for large prefixes when threshold is below observed root visits.
- Runtime penalty quantified. A memory-first mode can be accepted even if slower; a default change requires a tolerable speed/RSS tradeoff.
- CTW memory reports should separate arena memory from shared log-cache memory, otherwise the improvement can be hidden by other state.

### Implementation status: bounded exact lookup accepted

Implemented in `crates/infotheory/src/backends/ctw.rs`.

design:
  cached-prefix and bounded-fallback KT log access are now split again
  `CachedLogs` keeps the under-threshold hot path on direct unchecked slice reads
  `BoundedLogs` is used only once visits exceed the cap
  shared CTW cache growth is capped by a private compile-time limit
  accepted default cap is `2^24` entries, approximately 256 MiB for the two f64 caches
  above-cap values compute the exact same expressions directly:
    `ln(n as f64)`
    `ln(n as f64 + 0.5)`
  the first implementation remains private/test-configurable, as intended

coverage:
  all CTW/FAC-CTW update, prepared-update, revert, and FAC byte fast paths route
  through the shared bounded cache machinery
  the under-threshold fast path is still monomorphized on direct cached slices
  test-only scoped limits force fallback at tiny thresholds
  guardrails assert bounded logical cache length and separated log-cache memory reporting
  `FacContextTree::memory_usage_breakdown()` now separates:
    tree bytes
    shared log-cache bytes
    shared history bytes

validation:
  `cargo fmt --all -- --check`
  `cargo test -p infotheory --no-default-features --features backend-ctw backends::ctw::tests:: --locked`
  `cargo test -p infotheory --features all-backends compression::tests::roundtrip_rate_ac_ctw --locked`
  `cargo test -p infotheory --features 'cli all-backends' --test cli_commands cli_compression_backend_json_roundtrips_rate_ac_ctw --locked`
  `cargo test -p infotheory --no-default-features --features backend-ctw --test oracle_tests ctw_matches_theoretical_markov_entropy --locked`
  `RUSTFLAGS='-D warnings' cargo check -p infotheory --no-default-features --features backend-ctw --locked`
  `RUSTFLAGS='-D warnings' cargo check -p infotheory --no-default-features --features 'cli backend-ctw backend-mixture' --locked`
  `RUSTFLAGS='-D warnings' cargo check -p infotheory --no-default-features --features 'tuner backend-ctw' --locked`
  `git diff --check`

current caveat:
  lower defaults such as `2^20`, `2^21`, and `2^23` were measured and rejected for
  the default mode because they introduced about 4-6% CTW `h` regressions on the
  current 10 MB benchmark corpus when the accessor abstraction stayed on the hot path
  or the threshold was crossed too early
  with the accepted `2^24` cap plus the cached/bounded split:
    10 MB CTW `h` baseline/current on CPU 11 was about `161.9s -> 163.8s` (`+1.14%`)
    10 MB CTW `compress` was about `181.91s -> 182.41s` (`+0.28%`)
    10 MB CTW `decompress` was about `182.28s -> 182.83s` (`+0.30%`)
    20 MB CTW `h` was about `350.26s -> 356.20s` (`+1.70%`) while RSS fell
    from about `1,027,124 KiB` to `976,484 KiB` (`-4.93%`)
  this is the accepted default-mode compromise for now: bounded Hutter-feasible
  log-cache memory without the earlier large current-suite slowdown

## P0 track: PPMD probability-query interface

### Files

```text
crates/infotheory/src/backends/ppmd.rs
crates/infotheory/src/compression/mod.rs
```

### Current implementation

`PpmdModel::ensure_pdf_inner` computes a full byte distribution:

```text
lower = uniform[256]
for ord in 0..=max_order:
    key = context_key(ord)
    if context exists:
        lower = interpolate_context(ctx, lower)
normalize lower
```

`interpolate_context` writes a fresh `[f64; 256]`-sized array for each active context. In the neural-mixture fast-bitwise path, `RatePdfPredictor::prepare_cached_cdf_fast_bitwise` calls `PpmdModel::cdf()`, so PPMD materializes a full PDF/CDF even though the binary coder asks only eight conditional interval questions.

For neural-mixture likelihood updates, many call sites need `P(symbol)` rather than all 256 probabilities.

### Stronger hypothesis than suffix hashing

Add exact query methods that evaluate only the requested quantity:

```text
PpmdModel::symbol_prob(symbol) -> f64
PpmdModel::interval_mass(lo, hi) -> f64
```

For a single symbol:

```text
p = 1/256
for ord in active orders:
    if ctx exists:
        denom = total + distinct + 1
        escape = (distinct + 1) / denom
        p = escape * p + count(symbol) / denom
return normalized/floored equivalent
```

For an interval:

```text
mass = (hi - lo) / 256
for ord in active orders:
    if ctx exists:
        denom = total + distinct + 1
        escape = (distinct + 1) / denom
        mass = escape * mass + sum_counts(symbol in [lo, hi)) / denom
return normalized/floored equivalent
```

The exact difficulty is the current final `PDF_MIN` flooring and normalization. A query path must reproduce:

```text
normalize_pdf_and_maybe_cdf
```

or it is a model/coder change. There are two possible designs:

1. **Exact dense-equivalent query**: account for flooring/renormalization exactly. This may require knowing how many symbols fall below `PDF_MIN`, which may force more work.
2. **Unfloored internal query mode**: explicitly change semantics and evaluate archive effects. This is model-changing and should not be called an implementation optimization.

Before designing the final query path, measure how often flooring is actually active:

```text
min_unfloored_probability
count(pdf[i] < PDF_MIN)
mass_added_by_flooring
normalization_factor_after_flooring
```

If flooring is effectively never active for the benchmark/corpus region, the exact query path is simple and high-confidence. If flooring is common, dense-equivalent interval queries may need a compact prefix-summary of floored symbols rather than a scalar recurrence; otherwise the optimization may be illusory.

Start with exact dense-equivalence tests before measuring performance.

### Better binary-query shape

For the AC bitwise path, do not call `interval_mass` eight times with repeated order walks if avoidable. Compute the binary prefix tree masses in one pass per symbol:

```text
for each active context:
    accumulate sparse counts into the 8 prefix intervals
    update all interval masses by escape interpolation
```

This is `O(order * distinct_context_counts + 8 * order)` rather than `O(order * 256)` dense work.

### Expected impact

- **Reduces**: `W(n)` and `Q(n)` in PPMD inside neural mixture.
- **May reduce**: pressure from `prepare_cached_cdf_fast_bitwise`.
- **Neutral**: `S(n)` unless combined with compact counts.

### Acceptance

- For every test stream, `symbol_prob(symbol)` equals `pdf()[symbol]` within exact intended tolerance.
- For every binary prefix interval, `interval_mass(lo, hi)` equals `cdf[hi] - cdf[lo]`.
- Neural-mixture archive parity for AC fast-bitwise.
- `PpmdModel::ensure_pdf_inner` self-time decreases in the relevant path, or is bypassed in profiles.

## P1 track: PPMD rolling suffix keys and update lookup tightening

### Files

```text
crates/infotheory/src/backends/ppmd.rs
```

### Current implementation

`context_key(ord)` hashes `history[history.len() - ord..]` from scratch. Both `update` and `ensure_pdf_inner` walk active orders. At order `k`, key construction is `Theta(k^2)` byte mixing per symbol in the worst case.

`update` also does:

```text
contains_key
insert if absent
get_mut
```

which is avoidable double lookup for common existing contexts.

### Hypothesis

Maintain cached suffix hashes:

```text
suffix_hash[0] = 0 for order zero key
suffix_hash[1] = hash(last byte)
suffix_hash[2] = hash(last two bytes)
...
suffix_hash[k] = hash(last k bytes)
```

On byte append:

```text
new_suffix[0] = 0
new_suffix[1] = extend(FNV_OFFSET, byte)
new_suffix[j] = extend(old_suffix[j - 1], byte)
```

Then `context_key(ord)` becomes `O(1)` for active orders. Use the `entry` API in `update` to eliminate redundant map lookups.

### Correctness invariant

For every stream prefix and order:

```text
cached_suffix_key(ord) == hash_bytes(&history[history.len() - ord..])
```

Reset and history-only update paths must update or clear the cache consistently.

### Expected impact

- **Reduces**: `W(n)` in PPMD `update` and `ensure_pdf_inner`.
- **Slightly increases**: `S(n)` by `O(order)` words.
- **Archive**: exact parity required.

### Acceptance

This is localized and should be accepted only with parity and non-negative runtime. It is weaker than the query-interface track but low risk enough to do early.

## P1 track: RWKV7 TBPTT replay workspace reuse

### Files

```text
crates/infotheory/src/backends/rwkvzip/mod.rs
crates/infotheory/src/backends/rwkvzip/rwkv7/model.rs
```

### Current implementation

`Model::online_train_segment_tbptt` allocates inside each segment flush:

```text
FullGradState
RecurrentGradState
Vec<State> checkpoints
Vec<State> step_states
Vec<TokenTrainTrace> step_traces
Vec<Vec<f64>> step_pdfs
optional Vec<f32> bias_grad
```

The accepted RWKV patch removed several clones around policy windows and segment extraction, but this replay-local allocation/copy structure remains.

### Hypothesis

Introduce a reusable private replay workspace, probably owned by `OnlineRuntime` or `ScratchBuffers`:

```text
TbpttReplayWorkspace {
    grads: FullGradState
    recurrent: RecurrentGradState
    checkpoints: Vec<State>
    step_states: Vec<State>
    step_traces: Vec<TokenTrainTrace>
    step_pdfs: Vec<f64>  // flat chunk_len * vocab
    bias_grad: Vec<f32>
}
```

Use `clear`/`zero`/`clone_from` rather than dropping and reallocating every flush. Replace `Vec<Vec<f64>>` with a flat contiguous buffer.

### Correctness invariant

TBPTT replay semantics require the same forward states, traces, PDFs, and reverse-time gradient accumulation. Reusing storage must not reuse stale values. Every gradient/recurrent buffer needs explicit zeroing at the same semantic boundaries as today.

### Expected impact

- **Reduces**: `A(n)` and `Q(n)` during full-training windows.
- **May reduce**: wall time in `accumulate_token_step_gradients` regions by improving locality.
- **May reduce**: peak transient memory during training.

### Acceptance

- Archive parity for current RWKV and neural-mixture policies.
- Allocation count decreases in a profiling run or can be inferred from code plus stable timing.
- No long-inference slowdown.

## P1 track: RWKV7 post-training lifecycle release

### Files

```text
crates/infotheory/src/backends/rwkvzip/mod.rs
crates/infotheory/src/backends/llm_policy.rs
```

### Current implementation

`OnlineRuntime` can allocate:

```text
full_tbptt
full_adam
lm_head_adam_m/v
out_bias adam_m/v
training trace buffers in ScratchBuffers
```

The current code uses `should_capture_full_trace_for_next_step` to avoid unnecessary per-step state snapshots during inference windows, but it does not drop all full-training-only structures after the final full-parameter training interval.

### Hypothesis

Extend compiled policy/runtime introspection to answer:

```text
will_any_future_action_train_non_head_params(position, total_len) -> bool
will_any_future_action_use_adam_for_scope(scope, position, total_len) -> bool
```

After flushing pending TBPTT state and proving no future action needs full traces or full Adam:

```text
online.full_tbptt = None
online.full_adam = None
scratch.set_capture_train_trace(false)
drop or shrink replay workspace
```

### Correctness invariant

This is valid only when policy compilation makes the future finite and knowable for the stream. For repeat schedules or unknown total length, fail closed and keep state. Never drop optimizer state for a parameter family that may train again.

### Expected impact

- **Reduces**: `S(n)` in long inference tails and larger RWKV configs.
- **May reduce**: cache footprint.
- **Current tiny RWKV**: likely modest, but important for larger Hutter candidates.

## P1 track: expert marginal-utility audit

### Files

```text
configs/bench/two.json
crates/infotheory/src/mixture.rs
crates/infotheory/src/neural_mix.rs
crates/infotheory/src/compression/mod.rs
```

### Motivation

The Hutter Prize is not won by making every component faster in isolation. It is won by spending limited computation and memory on predictors whose codelength reduction exceeds their resource and description costs.

### Hypothesis

Add or script an ablation matrix:

```text
full neural mixture
drop CTW
drop PPMD
drop ROSA
drop Match
drop RWKV
pairs/triples of high-cost experts
```

Record:

```text
archive bytes
entropy bpb
time
RSS
expert cumulative log loss
neural expert weights over time
```

Then compute marginal value:

```text
Δcompressed_bits / Δseconds
Δcompressed_bits / ΔRSS_bytes
Δcompressed_bits / Δprogram_or_model_bytes
```

Use incremental, not standalone, value. An expert with excellent standalone compression can still be redundant inside the mixture if another expert already captures the same regularities. The useful quantity is:

```text
L(mixture_without_expert) - L(full_mixture)
```

under the same coder, same training policy, same prefix, and same verification conditions.

### Actionable outcomes

- If an expert is high-cost and low-marginal, prune or schedule it sparsely before optimizing internals.
- If an expert is high-cost and high-marginal, it becomes a priority for exact optimization.
- If an expert is useful only on certain corpus regions, investigate dynamic scheduling/gating as a model-changing frontier.
- If an expert improves compression only after a long warmup, measure whether the warmup cost is amortized on enwik9 rather than on short prefixes.
- If an expert is useful but memory-superlinear, treat bounded-state redesign as mandatory before further speed work.

### Acceptance

This is an analysis track. It should produce a table that determines which implementation tracks are worth pursuing for Hutter-scale work. The table should explicitly mark each expert as:

```text
keep/default
keep/optional speed-tier
bounded-state redesign required
drop or defer
needs longer-prefix evidence
```

## P1 track: CTW representation phase diagram

### Files

```text
crates/infotheory/src/backends/ctw.rs
```

### Current implementation

CTW stores explicit nodes through `HOT_PREFIX_DEPTH` and compressed unary/path segments deeper in the tree. This is the central speed/memory representation tradeoff.

### Hypothesis

Sweep `HOT_PREFIX_DEPTH` locally:

```text
6, 8, 10, 12, 14
```

and measure:

```text
CTW standalone h/compress/decompress
neural mixture h/compress/decompress
RSS
archive bytes
prepared path case distribution
```

Also instrument:

```text
PreparedEnd frequencies
prepared_steps.len()
node versus segment steps
has_sibling frequency
segment span distribution
```

### Expected impact

- **Trades**: `W(n)` against `S(n)`.
- **May reveal**: whether current `HOT_PREFIX_DEPTH = 10` is on the Pareto frontier.
- **May guide**: targeted CTW branch specializations.

### Acceptance

Do not change the default from one benchmark point. Require a phase diagram across at least 262144, 1 MiB, 4 MiB, and 10 MB. If archive bytes differ due floating-order effects, treat as a model/numerical change and evaluate resource frontier rather than claiming exact parity.

## P2 track: PPMD compact continuation storage

### Files

```text
crates/infotheory/src/backends/ppmd.rs
```

### Current implementation

Each context stores:

```rust
counts: Vec<(u8, u16)>
total: u32
```

For many contexts with few continuations, a separate `Vec` allocation per context is likely expensive in both allocator pressure and memory overhead.

### Hypothesis

Use an inline-small representation:

```text
TinyCounts {
    len: u8
    inline: [(u8, u16); N]
    spill: Option<Vec<(u8, u16)>>
}
```

Test `N = 2, 4, 6`. Do this after probability queries and suffix keys, because those may change the profile enough to alter the best storage design.

### Correctness invariant

The count multiset, `total`, rescale behavior, and interpolation distribution must match the current representation exactly.

### Expected impact

- **Reduces**: `S(n)` and `A(n)`.
- **May reduce**: `W(n)` through locality.
- **Risk**: branchier code and larger context structs if `N` is wrong.

## P2 track: exact interval/probability query API for other experts

### Files

```text
crates/infotheory/src/compression/mod.rs
crates/infotheory/src/backends/match_model.rs
crates/infotheory/src/backends/rosaplus.rs
crates/infotheory/src/backends/rwkvzip/mod.rs
```

### Current implementation

`MixturePredictor::ac_step_bitwise` has three modes:

```text
0: CTW bitwise prepared prediction/update
2: expert cached CDF branch probability
1: fallback local PDF-derived CDF row
```

For non-CTW byte experts, the model state is not updated until the full byte is known. Therefore exact byte-distribution interval queries are valid if they reproduce the same distribution.

### Match

Match has the simplest exact interval mass. Its distribution is uniform or a single predicted-byte spike over a uniform background:

```text
mass([lo, hi)) = rest * len + spike_mass_if_predicted_inside
```

This can avoid `pdf[256]` and `cdf[257]` construction in bitwise mode. The whole-suite gain may be small because Match is not a major self-time contributor, so implement only if the abstraction is needed for PPMD/ROSA anyway.

### ROSA+

ROSA interval mass may be possible from LM alphabet/count structures, but it is more invasive. Do not start here. First measure ROSA marginal utility and identify whether `fill_probs_for_last_bytes` is a dominant cost on compression/decompression, not only on `h`.

### RWKV7

RWKV needs logits over the full vocabulary for the softmax denominator. Interval queries cannot avoid the LM-head logits. A direct logits-to-CDF path may save a `f64` PDF pass, but `rwkvzip::Compressor::logits_to_pdf` is only about 2% self-time in the cited profile, so this is not a P0 track.

### Acceptance

For each expert:

```text
all binary prefix masses == dense CDF reference
archive parity
decompression roundtrip
whole neural-mixture signal non-negative
```

## P2 track: RWKV7 fixed-shape GEMV specialization

### Files

```text
crates/infotheory/src/backends/rwkvzip/rwkv7/kernel.rs
crates/infotheory/src/backends/rwkvzip/rwkv7/model.rs
```

### Current implementation

`gemv_avx` is generic. The current benchmark RWKV config has stable shapes:

```text
hidden = 64
vocab = 256
low ranks = 16
layers = 1
```

Hot shapes include:

```text
256 x 64  LM head
64 x 64   dense projections
16 x 64   low-rank down projections
64 x 16   low-rank up projections
```

### Hypothesis

Prototype private fixed-shape kernels:

```text
gemv_256x64
gemv_64x64
gemv_16x64
gemv_64x16
```

Dispatch only on exact dimensions. Preserve accumulation order as much as possible. Do not use FMA or reordered reductions unless archive parity still holds.

### Why this is P2, not P0

It may improve `W(n)`, but it increases code size and may perturb floating behavior. It is also less asymptotically important than CTW log-cache memory and PPMD dense-query elimination.

### Acceptance

- Specialized kernel outputs match generic kernel on randomized tests.
- RWKV and neural-mixture archive bytes unchanged, or the change is explicitly treated as numerical/model-changing.
- Binary size delta is reported.
- Whole RWKV speed improves enough to justify added code.

## P2 track: ROSA+ memory and probability audit

### Files

```text
crates/infotheory/src/backends/rosaplus.rs
```

### Hypotheses

1. **Probability query**: avoid dense byte PDF materialization if interval or symbol probabilities can be computed exactly from LM state.
2. **State compaction**: inspect SAM/LM node fields and index widths for safe narrowing or packing.
3. **Build/update scheduling**: verify LM build/finalization is not happening more often than required in compression/decompression.

### Caution

ROSA code is more complex than PPMD/Match, and the `h` path can be much slower than compression/decompression. Prioritize only after measuring marginal archive contribution and actual compression/decompression self-time.

## P3 track: AC and coder glue

### Files

```text
crates/infotheory/src/coders/ac.rs
crates/infotheory/src/compression/mod.rs
docs/perf.md
```

### Current state

Recent work already:

- split fast/nonfast AC paths once per stream;
- reused CDF scratch buffers;
- added binary AC decode;
- added compile-time inlining/deinlining knobs.

### Recommendation

Keep AC in the benchmark guardrails, but do not spend the next major optimization pass here unless a new profile shows coder glue back above the expert costs.

## Rejected or downgraded ideas

Do not retry these without new evidence:

- **NeuralMixCore same-context no-op**: measured regression.
- **Scratch-backed predictive weights in generic mixture path**: measured slower.
- **Removing binary split clamp**: measured regression.
- **Generic CTW predict-update fusion**: already present through `predict_one` and `update_predicted`.
- **RWKV fixed kernels as the first next step**: plausible, but lower priority than TBPTT workspace and CTW/PPMD asymptotic work.
- **Optimizing an expert solely because standalone speed is poor**: optimize only if marginal mixture/Hutter value justifies it.

## Recommended execution order

### 1. CTW bounded log-cache prototype

This is the highest Hutter-feasibility item. Implement a private thresholded exact lookup and sweep thresholds.

Required outputs:

```text
threshold
archive bytes
CTW/neural-mixture speed
RSS
shared log cache capacity/memory
```

### 2. PPMD exact query tests and symbol-probability path

First write tests proving query equivalence to dense PDF/CDF. Then route the neural-mixture likelihood update and/or bitwise path through query methods where possible.

Required outputs:

```text
symbol_prob == pdf[symbol]
interval_mass == cdf[hi] - cdf[lo]
archive parity
PpmdModel::ensure_pdf_inner profile reduction
```

### 3. PPMD rolling suffix keys and `entry` update

Do this as a low-risk exact improvement after or alongside query tests.

Progress:

```text
2026-05-13:
  implemented private rolling suffix-key cache in PpmdModel;
  replaced contains_key + insert + get_mut with entry-based update;
  converted dense PPMD context interpolation to in-place update, removing
  the per-active-context 256-entry temporary/copy while preserving bit identity;
  added invariant test comparing cached keys with recomputed suffix hashes
  across update, update_history_only, reset_history, and Clone;
  added bit-identity interpolation regression against the old out-of-place form;
  added PPMD checkpoint regression for mixed learned/frozen updates.
```

Validation:

```text
cargo fmt --all -- --check
TMPDIR=/home/theo/dev/infotheory/target/tmp \
  cargo test -p infotheory --no-default-features --features backend-ppmd \
  backends::ppmd::tests::rolling_suffix_keys_match_recomputed_suffix_hashes --locked
TMPDIR=/home/theo/dev/infotheory/target/tmp \
  cargo test -p infotheory --no-default-features --features backend-ppmd \
  backends::ppmd::tests::in_place_interpolation_matches_out_of_place_reference --locked
TMPDIR=/home/theo/dev/infotheory/target/tmp \
  cargo test -p infotheory --no-default-features --features backend-ppmd \
  mixture::tests::predictor_fill_matches_symbol_queries_for_ppmd_backend --locked
TMPDIR=/home/theo/dev/infotheory/target/tmp \
  cargo test -p infotheory --no-default-features --features backend-ppmd \
  mixture::tests::ppmd_checkpoint_restores_mixed_learned_and_frozen_updates --locked
TMPDIR=/home/theo/dev/infotheory/target/tmp RUSTFLAGS='-D warnings' \
  cargo check -p infotheory --no-default-features --features backend-ppmd --locked
```

Short A/B evidence, noisy workstation, native release, baseline `HEAD` versus dirty current, `cli backend-ppmd`, PPMD order 12:

```text
input: deterministic 1 MiB local prefix
parity:
  h output identical: 0.08754313906153081
  rate-ac archive byte-identical
  roundtrip verified for baseline and current

hyperfine, 10 runs, h:
  baseline mean 2.524110 s, stddev 0.030596
  current  mean 1.732328 s, stddev 0.041615
  rough speedup: 1.46x

hyperfine, 8 runs, rate-ac compress:
  baseline mean 2.932226 s, stddev 0.054539
  current  mean 2.151286 s, stddev 0.053199
  rough speedup: 1.36x

perf stat, 3 runs, h:
  baseline cycles      10,177,137,209
  current  cycles       6,948,914,653
  baseline instructions 13,571,757,869
  current  instructions  7,337,571,191

cachegrind, 32 KiB h:
  baseline I refs 425,764,652
  current  I refs 253,391,317

memory checks:
  theoretical persistent model-state delta:
    one Vec header plus (order + 1) u64 suffix keys per PpmdModel
    order 12 payload ~= 104 bytes, plus allocator/header effects
  /usr/bin/time -v, 1 MiB h:
    PPMD-alone max RSS stayed within noise: about 13.2--13.5 MiB
    ctw+ppmd mixture max RSS stayed within noise: about 40.1--40.3 MiB
  massif, 32 KiB h:
    PPMD-alone peak heap 8,203,332 B -> 8,203,439 B
    ctw+ppmd peak heap 10,904,815 B -> 10,904,970 B
  binary size, native release:
    cli backend-ppmd: 1,426,280 B -> 1,427,008 B
    cli backend-mixture/backend-ppmd/backend-ctw: 1,767,280 B -> 1,768,216 B

ctw+ppmd mixture rough performance, 1 MiB h:
  parity: h identical, archive byte-identical, roundtrip verified
  /usr/bin/time -v user time:
    baseline about 10.33--10.46 s
    current  about  9.34-- 9.53 s
  perf stat, 2 runs:
    cycles       42,318,389,366 -> 38,710,946,165
    instructions 58,971,036,540 -> 52,728,971,888
```

Interpretation: this is not a final benchmark claim because the workstation was not isolated, but the agreement between parity, hyperfine, perf counters, and cachegrind instruction counts is strong evidence that the PPMD implementation work is a real strict improvement.

Exact query follow-up:

```text
implemented validation:
  exact symbol query equals dense pdf()[symbol]
  exact interval query equals dense cdf[hi] - cdf[lo]
  flooring diagnostics cover min unfloored probability, floored count,
    mass added by flooring, and post-flooring normalization factor

production routing result:
  always-sparse exact query prototype was rejected
  conservative sparse/fallback prototype was also rejected
  final production PPMD path remains the dense cached normalization path
  final native release binary sizes match the committed baseline:
    cli backend-ppmd: 1,427,008 B -> 1,427,008 B
    cli backend-mixture/backend-ppmd/backend-ctw: 1,768,216 B -> 1,768,216 B

isolated A/B against committed PPMD suffix-key baseline, noisy workstation:
  parity:
    PPMD-alone h identical, ctw+ppmd h identical
    archives byte-identical, roundtrip verified
  always-sparse prototype, 1 MiB h:
    PPMD-alone baseline 1.723205 s, prototype 1.909546 s
    ctw+ppmd   baseline 9.506916 s, prototype 9.871999 s
  conservative sparse/fallback prototype, 1 MiB h:
    PPMD-alone baseline 1.739279 s, prototype 1.919832 s
    ctw+ppmd   baseline 9.447059 s, prototype 9.612215 s
```

Conclusion: exact dense-equivalence tests and flooring diagnostics are now in place, but the production sparse-query hypothesis is a measured non-improvement under the current PPMD representation. Do not route hot callers through a sparse PPMD query path unless the representation itself changes enough to make the query state cheaper than dense normalization. The PPMD work accepted for production in this tranche is therefore the rolling suffix-key, map-entry tightening, and in-place interpolation work; the next high-confidence implementation item should move to CTW bounded/exact log lookup.

### 4. RWKV TBPTT replay workspace reuse

Reduce allocation churn in full training windows. Use flat PDF buffers and reusable vectors.

### 5. Expert marginal-utility ablation matrix

Before deeper ROSA/RWKV/Match work, measure which experts pay rent in the mixture. Use the result to choose the next deep implementation target.

### 6. Resource-MDL configuration search

After feasibility fixes and expert-rent measurements, run a finite configuration search over expert subsets and high-impact hyperparameters. Do not tune a memory-infeasible configuration.

### 7. CTW hot-prefix/segment phase diagram

Run after bounded log-cache work so the memory picture is not confounded by the unbounded cache.

### 8. PPMD compact continuation storage

Proceed only if PPMD remains memory-significant after query/key improvements.

### 9. RWKV fixed-shape GEMV

Prototype only after the above higher-confidence work, and require binary-size reporting.

## Guardrail tests to add before invasive patches

### CTW

- bounded log lookup equals cached lookup for a broad range of counts;
- archive parity for CTW standalone and neural mixture;
- `predict_one` + `update_predicted` remains equivalent to the existing fast bitwise path;
- hot-prefix variants preserve roundtrip and report archive differences explicitly.

### PPMD

- suffix cache equals `hash_bytes` for randomized streams and all active orders;
- `symbol_prob` equals dense `pdf()[symbol]`;
- all binary prefix interval masses equal dense CDF masses;
- compact counts preserve observe/rescale/prune semantics.

### RWKV7

- TBPTT workspace reuse preserves archive parity;
- no stale gradient state after workspace reuse;
- post-training release only triggers when compiled policy future is finite and safe;
- fixed-shape kernels equal generic kernels and preserve archive bytes.

### Mixture

- expert ablations produce verified archives;
- direct probability/interval query paths match dense references;
- fast-bitwise compress/decompress roundtrip remains exact.

## Final principle

The strongest next work is not “more micro-optimization.” It is removing asymptotic infeasibilities and dense computations that are not demanded by the coding decision:

```text
do not store O(n) logs if O(1) memory exact lookup is acceptable;
do not build 256 probabilities if the coder needs one symbol or one interval;
do not retain training state after the policy can no longer train;
do not optimize a costly expert until its marginal codelength value is known.
```

That is the path most aligned with MDL, algorithmic information theory, and a realistic Hutter Prize resource envelope.
