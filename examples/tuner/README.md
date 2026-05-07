# Tuner Validation Command Suite

This README is a depot for manual validation, as much of it requires root and is not automated.

Before copy-pasting the commands below, set the repository root once:

```bash
export INFOTHEORY_REPO=/path/to/infotheory
```

The tuner, at least for passive compression, outputs a raw CompressionBackend JSON Object.  By the library, you can use parse_compression_backend_json directly on this output.

You can compress with the CompressionBackend object like this:
```bash
infotheory compress /input.bin /output.bin --compression-backend-json /path/to/compression_backend.json
```

## 1) One-time delegated cgroup-v2 setup

```bash
cd "$INFOTHEORY_REPO"
sudo ./scripts/delegate_tuner_cgroup_v2.sh setup theo infotheory-tuner
```

## 2) Strict smoke checks

Strict non-MC-AIXI smoke:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-spec.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --max-evaluations 1
```

Strict MC-AIXI smoke:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-mc-aixi-spec.json \
    --exact-reward-encoding-certificate examples/tuner/strict-smoke-mc-aixi-reward-cert.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --max-evaluations 1
```

## 3) Strict TSV examples

Annealed simple TSV:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-annealed-simple-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --max-evaluations 1
```

Annealed advanced neural-mixture TSV:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-annealed-advanced-neural-mixture-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --max-evaluations 1
```

MC-AIXI advanced neural-mixture TSV:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
    --exact-reward-encoding-certificate examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-hybrid-strict-max.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --max-evaluations 1
```

## 4) Strict three-mode `two.json` comparison artifact

Walkthrough:

- [walkthrough_improving_config.md](walkthrough_improving_config.md)

Script:

- [benchmark_tuner_two_json_modes.sh](../../scripts/benchmark_tuner_two_json_modes.sh)

Run default strict comparison (subject = `git show HEAD:README.md`):

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  ./scripts/benchmark_tuner_two_json_modes.sh
```

Run strict comparison on explicit subject file:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  ./scripts/benchmark_tuner_two_json_modes.sh /path/to/input.bin 2 1
```

## 5) Emit strict exact reward certificates only

Strict MC-AIXI smoke cert:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-mc-aixi-spec.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --emit-exact-reward-encoding-certificate examples/tuner/strict-smoke-mc-aixi-reward-cert.json
```

Strict MC-AIXI advanced TSV cert:

```bash
cd "$INFOTHEORY_REPO"
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --evaluator-cgroup-parent /sys/fs/cgroup/infotheory-tuner/evals \
    --emit-exact-reward-encoding-certificate examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-hybrid-strict-max.json
```
