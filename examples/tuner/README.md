# Tuner Validation Command Suite

This folder contains copy-paste runnable validation examples.

Dataset choice for TSV examples:
- `/home/theo/dev/infotheory/benchmarks/6f464811/infotheory-two-json-summary-full.tsv`
- Reason: it is in-tree and won't be altered and is an appropriate size.

## 1) One-time strict cgroup setup

```bash
cd /home/theo/dev/infotheory
sudo ./scripts/delegate_tuner_cgroup_v2.sh setup theo infotheory-tuner
```

## 2) Strict Linux (`hybrid_strict_max`) suite

Strict smoke (non-MC-AIXI):

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-spec.json \
    --rss-mode hybrid_strict_max \
    --max-evaluations 1
```

Strict smoke MC-AIXI (fixed certificate):

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-mc-aixi-spec.json \
    --exact-reward-encoding-certificate strict-smoke-mc-aixi-reward-cert.json \
    --rss-mode hybrid_strict_max \
    --max-evaluations 1
```

Strict annealed simple TSV:

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-annealed-simple-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --max-evaluations 1
```

Strict annealed advanced neural-mixture TSV:

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-annealed-advanced-neural-mixture-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --max-evaluations 1
```

Strict MC-AIXI advanced neural-mixture TSV:

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
    --exact-reward-encoding-certificate strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-hybrid-strict-max.json \
    --rss-mode hybrid_strict_max \
    --max-evaluations 1
```

## 3) MC-AIXI TSV examples (provided certificates are process-RSS profile)

Simple MC-AIXI TSV:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
  tune examples/tuner/strict-mcaixi-simple-tsv-spec.json \
  --exact-reward-encoding-certificate strict-mcaixi-simple-tsv-exact-reward-cert-process-rss.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1
```

Advanced MC-AIXI neural-mixture TSV:

```bash
cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
  tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
  --exact-reward-encoding-certificate strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-process-rss.json \
  --rss-mode process_rss_peak \
  --max-evaluations 1
```

Notes:
- The two TSV MC-AIXI certificate files bind dataset + bounds + evaluator profile + controller kind.
- They are intentionally scoped to `process_rss_peak`.
- `strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-hybrid-strict-max.json` is scoped to strict Linux `hybrid_strict_max`.
- If you want strict `hybrid_strict_max` for those two TSV MC-AIXI examples, regenerate exact reward certificates under the strict evaluator profile.

## 4) Emit strict exact-reward certificates from resolved profile (no trial run)

Strict MC-AIXI smoke certificate (emit and exit):

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw' -- \
    tune examples/tuner/strict-smoke-mc-aixi-spec.json \
    --rss-mode hybrid_strict_max \
    --emit-exact-reward-encoding-certificate examples/tuner/strict-smoke-mc-aixi-reward-cert.json
```

Strict MC-AIXI advanced neural-mixture TSV certificate (emit and exit):

```bash
sudo --preserve-env=PATH ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
  env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
  cargo run -p infotheory --no-default-features --features 'tuner cli backend-ctw backend-mixture' -- \
    tune examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-spec.json \
    --rss-mode hybrid_strict_max \
    --emit-exact-reward-encoding-certificate examples/tuner/strict-mcaixi-advanced-neural-mixture-tsv-exact-reward-cert-hybrid-strict-max.json
```
