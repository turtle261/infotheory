fn quoted_backend_list(names: &[&str]) -> String {
    names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if idx == 0 {
                format!("'{name}' (default)")
            } else {
                format!("'{name}'")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn rate_backends() -> String {
    quoted_backend_list(&infotheory::backends::available_rate_backends())
}

fn compression_backends() -> String {
    quoted_backend_list(&infotheory::backends::available_compression_backends())
}

fn ctw_profile_line() -> &'static str {
    if cfg!(all(feature = "backend-ctw", feature = "research-tooling")) {
        "  ctw-profile    FAC-CTW arena telemetry JSONL\n"
    } else {
        ""
    }
}

/// Print the top-level CLI help.
pub(crate) fn print_global_help() {
    eprintln!(
        r#"InfoTheory CLI
Usage:
  infotheory <command> [args...] [options]
  infotheory help [topic]

Core commands:
  h, entropy     Empirical byte entropy; uses rate backend when explicitly selected
  h_rate         Algorithmic entropy rate via the active rate backend
  mi, xe, ce     Mutual information, cross entropy, conditional entropy
  ncd            Normalized compression distance
  ned, nte       Normalized entropy distance and transform effort
  kl, js, tvd    Empirical byte-distribution divergences/distances
  compress       Compress a file with the selected compression backend
  decompress     Decompress a framed file
  generate       Continue file or piped bytes with the active rate backend
  search         Rank code/text matches with information-theoretic scoring
  batch          JSONL batch API
  aixi           Run a canonical planner_run document
  warmstart      Export, convert, or merge warm-start teacher datasets
  tune           Run a canonical tune document
  ac-log-loss    Exact AC/log-loss diagnostics for mixture specs
  sequitur-debug Sequitur grammar and bounded predictive trace debugging
{ctw_profile_line}
Common backend options:
  --rate-backend <name>          Rate backend: {rate_backends}
  --compression-backend <name>   Compression backend: {compression_backends}
  --method <value>               Backend method/config, model method, or spec path
  --rate-backend-json <path>     Canonical RateBackend JSON
  --compression-backend-json <path>
                                  Canonical CompressionBackend JSON
  --expert-spec <path>           Standalone expert JSON
  --model-export <path>          Export updated online model and sidecar

Topics:
  metrics, backends, compression, generation, batch, aixi, warmstart, tune,
  diagnostics, sequitur, search

Examples:
  infotheory h README.md
  infotheory h_rate README.md --rate-backend ctw --method 32
  infotheory ncd a.bin b.bin --compression-backend zpaq --method 5
  infotheory compress in.bin out.itc --compression-backend rate-ac --rate-backend ctw
  cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
  infotheory aixi configs/aixi/paper_kuhn_poker.json

Use `infotheory help <topic>` or `infotheory <command> --help` for details.
"#,
        ctw_profile_line = ctw_profile_line(),
        rate_backends = rate_backends(),
        compression_backends = compression_backends()
    );
}

/// Print a topic-specific help page. Unknown topics fall back to global help.
pub(crate) fn print_topic_help(topic: &str) {
    match normalize_topic(topic).as_str() {
        "metrics"
        | "h"
        | "entropy"
        | "h_rate"
        | "entropy_rate"
        | "mi"
        | "mutual_info"
        | "xe"
        | "cross_entropy"
        | "ce"
        | "conditional_entropy"
        | "joint_entropy"
        | "h_xy"
        | "id"
        | "ned"
        | "nte"
        | "rt"
        | "resistance"
        | "kl"
        | "kl_divergence"
        | "js"
        | "js_divergence"
        | "tvd"
        | "nhd" => print_metrics_help(),
        "backends" | "backend" | "rate-backend" | "compression-backend" => print_backends_help(),
        "compression" | "compress" | "decompress" | "ncd" | "ncd_sym" | "ncd_cons"
        | "ncd_sym_cons" => print_compression_help(),
        "generation" | "generate" => print_generation_help(),
        "batch" => print_batch_help(),
        "aixi" | "planner" | "planner_run" | "planner-run" => print_aixi_help(),
        "warmstart" => print_warmstart_help(),
        "tune" | "tuner" => print_tune_help(),
        "diagnostics" | "diagnostic" | "ac-log-loss" | "ac_log_loss" | "ctw-profile"
        | "ctw_profile" => print_diagnostics_help(),
        "sequitur" | "sequitur-debug" | "sequitur_debug" => print_sequitur_help(),
        "search" => print_search_help(),
        _ => print_global_help(),
    }
}

fn normalize_topic(topic: &str) -> String {
    topic.trim().to_ascii_lowercase()
}

fn print_metrics_help() {
    eprintln!(
        r#"InfoTheory metrics
Usage:
  infotheory h <file> [backend options]
  infotheory h_rate <file> [backend options]
  infotheory <metric> <file1> <file2> [backend options]

Single-file metrics:
  h, entropy             Empirical order-0 byte entropy unless a rate backend is selected
  h_rate, entropy_rate   Algorithmic entropy rate via active RateBackend
  id                     Intrinsic dependence from empirical entropy and entropy rate

Two-file metrics:
  mi, mutual_info        Mutual information
  xe, cross_entropy      Cross entropy
  ce, conditional_entropy
  joint_entropy, h_xy
  ned, ned_cons          Normalized entropy distance variants
  nte                    Normalized transform effort
  rt, resistance         Resistance to transformation
  kl, js, tvd, nhd       Empirical byte-distribution divergences/distances

Backend selection:
  Add --rate-backend, --rate-backend-json, or --expert-spec to use the
  algorithmic/rate-backed path where the metric supports it.

Examples:
  infotheory h README.md
  infotheory h_rate README.md --rate-backend fac-ctw --method 32 --msb-first
  infotheory mi a.bin b.bin --rate-backend ctw --method 16
"#
    );
}

fn print_backends_help() {
    eprintln!(
        r#"InfoTheory backend selection
Usage:
  infotheory <command> ... [backend options]

Rate backends:
  {rate_backends}

Compression backends:
  {compression_backends}

Options:
  --rate-backend <name>          Shorthand rate backend name
  --compression-backend <name>   Shorthand compression backend name
  --method <value>               Backend method/config or spec path
  --rate-backend-json <path>     Canonical RateBackend JSON; relative asset paths resolve
                                  against the JSON file directory
  --compression-backend-json <path>
                                  Canonical CompressionBackend JSON, including tuner output
  --expert-spec <path>           One standalone mixture expert JSON
  --msb-first | --lsb-first      FAC-CTW bit order; requires --rate-backend fac-ctw
  --model-export <path>          Export updated online neural model plus JSON sidecar

Method examples:
  --method 5
  --method 32
  --method mixture.json
  --method "file:/path/model.safetensors;policy:..."
  --method "cfg:hidden=64,layers=1,intermediate=64,...;policy:..."
"#,
        rate_backends = rate_backends(),
        compression_backends = compression_backends()
    );
}

fn print_compression_help() {
    eprintln!(
        r#"InfoTheory compression and NCD
Usage:
  infotheory ncd <file1> <file2> [method] [backend options]
  infotheory ncd_sym <file1> <file2> [backend options]
  infotheory ncd_cons <file1> <file2> [backend options]
  infotheory compress <input> <output> [backend options]
  infotheory decompress <input> <output> [backend options]

Compression backend options:
  --compression-backend zpaq|rate-ac|rate-rans|rwkv7
  --compression-backend-json <path>
  --rate-backend <name> --method <value>     For rate-ac/rate-rans wrappers

Examples:
  infotheory ncd a.bin b.bin --compression-backend zpaq --method 5
  infotheory ncd a.bin b.bin --compression-backend rate-ac --rate-backend ctw --method 16
  infotheory compress in.bin out.itc --compression-backend rate-rans --rate-backend fac-ctw --method 32
  infotheory decompress out.itc restored.bin --compression-backend rate-rans --rate-backend fac-ctw --method 32
"#
    );
}

fn print_generation_help() {
    eprintln!(
        r#"InfoTheory generation
Usage:
  infotheory generate [file] [backend options] [generation options]
  cat prompt.txt | infotheory generate [backend options] [generation options]

Options:
  --bytes <n>          Bytes to generate (default: 8)
  --sample             Use seeded sampling
  --greedy             Force deterministic greedy generation
  --adaptive           Fit on generated bytes instead of frozen continuation
  --seed <u64>         RNG seed; implies sampling
  --temperature <x>    Sampling temperature (default: 1.0)
  --top-k <n>          Sample from top-k bytes; 0 disables
  --top-p <p>          Nucleus threshold in (0, 1]

Examples:
  cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
  infotheory generate prompt.txt --rate-backend match --bytes 16 --sample --seed 7
"#
    );
}

fn print_batch_help() {
    eprintln!(
        r#"InfoTheory JSONL batch API
Usage:
  infotheory batch < input.jsonl > output.jsonl
  echo '{{"op":"help"}}' | infotheory batch

Batch operations:
  help, metrics, metrics_file, ncd, ncd_files, rosa_dist, cross_entropy,
  batch_metrics, ncd_matrix, rosa_matrix, spam_check

The batch API is line-oriented: each input line is one JSON request and each
output line is one JSON response.
"#
    );
}

fn print_aixi_help() {
    eprintln!(
        r#"InfoTheory AIXI/planner-run mode
Usage:
  infotheory aixi <planner-run.json>

The AIXI CLI executes canonical spec documents:
  {{"schema_version": 1, "kind": "planner_run", ...}}

Checked-in examples:
  infotheory aixi configs/aixi/paper_kuhn_poker.json
  infotheory aixi configs/aixi/builtin_tictactoe.json

Legacy pre-1.2 AIXI JSON configs are intentionally rejected. From the Infotheory repository, convert them with:
  ./projman.sh legacy_aixi_convert <input_file>

For VM-backed environments, build with the vm feature and provide valid Nyx-Lite
VM assets referenced by the planner_run document.
"#
    );
}

fn print_warmstart_help() {
    eprintln!(
        r#"InfoTheory warm-start teacher tools
Usage:
  infotheory warmstart teacher planner-run --target <warmstart-planner-run> --teacher <teacher-planner-run> --out <teacher.json>
  infotheory warmstart teacher from-jsonl --target <warmstart-planner-run> --jsonl <run.jsonl> --out <teacher.json>
  infotheory warmstart teacher merge --target <warmstart-planner-run> --out <teacher.json> --teacher <teacher-a.json> [...]

Subcommands:
  planner-run   Execute a compatible teacher planner_run and export a same-task
                warm-start teacher dataset.
  from-jsonl    Convert normalized planner JSONL telemetry to a teacher dataset.
  merge         Deterministically merge same-task teacher datasets.

The target must be an aiqi_warmstart_exact_jh planner_run. Teacher datasets are
validated against the compiled target contract before writing.
"#
    );
}

fn print_tune_help() {
    eprintln!(
        r#"InfoTheory tuner
Usage:
  infotheory tune <spec.json|spec.itsd> [options]

Core options:
  --exec-config <path>                  Executor profile JSON
  --max-evaluations <n>                 Optional evaluation cap
  --annealer-kernel-profile <name>      reversible_elementary_metropolis or
                                        compiled_uniform_metropolis_hastings
  --cpu-affinity <csv>                  Comma-separated core ids
  --threads <n>                         Executor thread hint
  --evaluator-worker-executable <path>  Explicit evaluator worker executable
  --evaluator-cgroup-parent <path>      Delegated cgroup-v2 eval parent
  --warmup-baseline-runs <n>
  --self-improvement-rounds <n>
  --stagnation-reset-evals <n>
  --log-path <path>                     JSONL executor event log
  --diagnostic-chunk-bytes <n>
  --rss-mode <mode>                     process_rss_peak, backend_reported,
                                        or hybrid_strict_max
  --planner-deployable-model
  --warmstart-trace-refresh

Certificate/theorem-facing options:
  --timing-tier <tier>
  --determinism-deadline-certificate <ref>
  --deterministic-evaluator-table <ref>
  --finite-planner-state-certificate <ref>
  --no-hidden-state-certificate <ref>
  --exact-reward-encoding-certificate <ref>
  --emit-exact-reward-encoding-certificate <path>
  --exact-state-observation-certificate <ref>
  --observation-adapter-spec-ref <ref>
  --exact-state-encoder-spec-ref <ref>
  --scalar-representation-ref <ref>
  --claim-exact-finite-mdp
  --claim-exact-observed-markov
  --claim-planner-convergence

Examples live under examples/tuner/.
"#
    );
}

fn print_diagnostics_help() {
    eprintln!(
        r#"InfoTheory diagnostics
Usage:
  infotheory ac-log-loss <input> --mixture <spec.json> --out-prefix <prefix>
  infotheory ctw-profile <input|-> [--depth N]

ac-log-loss writes:
  <prefix>.trace.tsv
  <prefix>.nodes.tsv
  <prefix>.summary.tsv

ctw-profile requires features backend-ctw and research-tooling. It emits FAC-CTW
arena telemetry as JSONL.
"#
    );
}

fn print_sequitur_help() {
    eprintln!(
        r#"InfoTheory Sequitur debug
Usage:
  infotheory sequitur-debug <input> [options]
  infotheory sequitur-debug --hex <hex> [--hex <hex> ...] [options]

Options:
  --hex <hex>             Hex-encoded byte string; repeatable
  --context-bytes <n>     Sequitur context width (default: 64)
  --alphabet-prefix <n>   Prefix of predictive PDF to emit

Example:
  infotheory sequitur-debug --hex 616263616263 --alphabet-prefix 8
"#
    );
}

fn print_search_help() {
    eprintln!(
        r#"InfoTheory search
Usage:
  infotheory search <query> <target> [options]

Common options:
  --prior <text>          Extra codebase/domain context
  --level snippet|file    Search granularity
  --top-k <n>             Maximum results

Example:
  infotheory search "encryption" ./crates/infotheory/src --prior "codebase context"
"#
    );
}
