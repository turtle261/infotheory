# CLI Reference

The `infotheory` binary is available only when the `cli` feature is enabled:

```bash
cargo build -p infotheory --release --features cli --bin infotheory --locked
```

For local development, the same commands can be run through Cargo:

```bash
cargo run -p infotheory --features cli -- <command> [args...] [options]
```

## Help

The top-level help is intentionally short:

```bash
infotheory --help
```

Use topic help for details:

```bash
infotheory help backends
infotheory help compression
infotheory help generation
infotheory help batch
infotheory help aixi
infotheory help warmstart
infotheory help tune
infotheory help diagnostics
infotheory help sequitur
infotheory help search
```

Representative command help also routes to the matching topic:

```bash
infotheory ncd --help
infotheory generate --help
infotheory warmstart --help
infotheory tune --help
```

## Metrics

Single-file metrics:

```bash
infotheory h README.md
infotheory h_rate README.md --rate-backend ctw --method 32
infotheory id README.md --rate-backend fac-ctw --method 32 --msb-first
```

Two-file metrics:

```bash
infotheory mi a.bin b.bin
infotheory mi a.bin b.bin --rate-backend ctw --method 16
infotheory ned a.bin b.bin
infotheory nte a.bin b.bin
infotheory kl a.bin b.bin
infotheory js a.bin b.bin
```

`h` and the empirical two-file metrics use order-0 byte statistics unless a rate
backend is explicitly selected. `h_rate` always uses the active rate backend.

## Backend Selection

Common backend options:

```bash
--rate-backend <name>
--compression-backend <name>
--method <value>
--rate-backend-json <path>
--compression-backend-json <path>
--expert-spec <path>
--model-export <path>
```

Use `infotheory help backends` for the feature-dependent backend list reported
by the binary you built.

Canonical JSON inputs are preferred when a backend is produced by another
InfoTheory tool, especially tuner output. Relative asset paths inside canonical
backend JSON resolve against the JSON file's directory.

## Compression

```bash
infotheory ncd a.bin b.bin --compression-backend zpaq --method 5
infotheory ncd a.bin b.bin --compression-backend rate-ac --rate-backend ctw --method 16
infotheory compress in.bin out.itc --compression-backend rate-rans --rate-backend fac-ctw --method 32
infotheory decompress out.itc restored.bin --compression-backend rate-rans --rate-backend fac-ctw --method 32
```

Use `rate-ac` or `rate-rans` when the compressor should be driven by a
predictive `RateBackend`.

## Generation

```bash
cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
infotheory generate prompt.txt --rate-backend match --bytes 16 --sample --seed 7
```

Generation supports greedy or sampled continuation, top-k/top-p filtering, and
adaptive continuation. See `infotheory help generation`.

## Spec-Driven Commands

`aixi` executes canonical `planner_run` documents:

```bash
infotheory aixi configs/aixi/paper_kuhn_poker.json
```

Legacy pre-1.2 AIXI JSON configs are rejected. Convert them with:

```bash
./projman.sh legacy_aixi_convert <input_file>
```

`warmstart` tools export, convert, and merge exact-J_H teacher datasets:

```bash
infotheory warmstart teacher planner-run --target target.json --teacher teacher.json --out teacher-dataset.json
infotheory warmstart teacher from-jsonl --target target.json --jsonl run.jsonl --out teacher-dataset.json
infotheory warmstart teacher merge --target target.json --out merged.json --teacher a.json --teacher b.json
```

`tune` executes canonical `tune` JSON or binary spec documents:

```bash
infotheory tune examples/tuner/strict-smoke-spec.json --max-evaluations 1
```

The tuner has many executor and certificate controls; keep those in topic help
and tuner docs rather than top-level help.

## Diagnostics

```bash
RAYON_NUM_THREADS=4 infotheory ac-log-loss corpus.bin \
  --mixture configs/bench/mixture.json \
  --out-prefix /tmp/mixture-diagnostic

infotheory sequitur-debug --hex 616263616263 --alphabet-prefix 8
```

`ac-log-loss` writes `<prefix>.trace.tsv`, `<prefix>.nodes.tsv`, and
`<prefix>.summary.tsv`. `ctw-profile` is available only when the binary is built
with `backend-ctw research-tooling`.
