#![cfg(feature = "cli")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

fn temp_path(name: &str, ext: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("infotheory_ac_log_loss_{name}_{ts}.{ext}"))
}

fn temp_prefix(name: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("infotheory_ac_log_loss_{name}_{ts}"))
}

fn write_temp_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).expect("write temp file");
}

fn run_ac_log_loss(input: &Path, mixture: &Path, prefix: &Path, threads: &str) {
    let bin = env!("CARGO_BIN_EXE_infotheory");
    let output = Command::new(bin)
        .arg("ac-log-loss")
        .arg(input)
        .arg("--mixture")
        .arg(mixture)
        .arg("--out-prefix")
        .arg(prefix)
        .env("RAYON_NUM_THREADS", threads)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn ac-log-loss");

    assert!(
        output.status.success(),
        "ac-log-loss failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn nested_mixture_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "kind": "switching",
        "alpha": 0.15,
        "experts": [
            {
                "name": "ctw",
                "kind": "ctw",
                "depth": 6
            },
            {
                "name": "nested",
                "kind": "mixture",
                "spec": {
                    "kind": "bayes",
                    "experts": [
                        {
                            "name": "fac",
                            "kind": "fac-ctw",
                            "base_depth": 5,
                            "encoding_bits": 8,
                            "num_percept_bits": 8
                        },
                        {
                            "name": "ppmd",
                            "kind": "ppmd",
                            "order": 4,
                            "memory_mb": 8
                        }
                    ]
                }
            }
        ]
    }))
    .expect("mixture json")
}

fn parallel_mixture_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "kind": "switching",
        "alpha": 0.12,
        "experts": [
            {
                "name": "ctw",
                "kind": "ctw",
                "depth": 6
            },
            {
                "name": "fac-root",
                "kind": "fac-ctw",
                "base_depth": 5,
                "encoding_bits": 8,
                "num_percept_bits": 8
            },
            {
                "name": "ppmd-root",
                "kind": "ppmd",
                "order": 4,
                "memory_mb": 8
            },
            {
                "name": "nested",
                "kind": "mixture",
                "spec": {
                    "kind": "bayes",
                    "experts": [
                        {
                            "name": "fac",
                            "kind": "fac-ctw",
                            "base_depth": 5,
                            "encoding_bits": 8,
                            "num_percept_bits": 8
                        },
                        {
                            "name": "ppmd",
                            "kind": "ppmd",
                            "order": 4,
                            "memory_mb": 8
                        }
                    ]
                }
            }
        ]
    }))
    .expect("parallel mixture json")
}

#[test]
fn ac_log_loss_cli_writes_expected_tsvs() {
    let input_path = temp_path("input", "bin");
    let mixture_path = temp_path("mixture", "json");
    let prefix = temp_prefix("out");
    write_temp_file(&input_path, b"cli diagnostic payload");
    write_temp_file(&mixture_path, &nested_mixture_json());

    run_ac_log_loss(&input_path, &mixture_path, &prefix, "1");

    let trace_path = prefix.with_extension("trace.tsv");
    let nodes_path = prefix.with_extension("nodes.tsv");
    let summary_path = prefix.with_extension("summary.tsv");
    let trace = fs::read_to_string(&trace_path).expect("trace exists");
    let nodes = fs::read_to_string(&nodes_path).expect("nodes exists");
    let summary = fs::read_to_string(&summary_path).expect("summary exists");

    let trace_lines: Vec<&str> = trace.lines().collect();
    assert_eq!(trace_lines.len(), b"cli diagnostic payload".len() + 1);
    assert!(trace_lines[0].contains("mix_prob"));
    assert!(trace_lines[0].contains("root_weight_entropy_bits"));
    assert!(trace_lines[0].contains("oracle_best_id"));
    assert!(trace_lines[0].contains("n1__prob"));
    assert!(trace_lines[0].contains("n4__effective_weight"));

    let node_lines: Vec<&str> = nodes.lines().collect();
    assert_eq!(node_lines.len(), 6);
    assert!(node_lines[0].contains("node_id\tparent_id\tdepth"));
    assert!(nodes.contains("0:root"));
    assert!(nodes.contains("nested"));

    let summary_lines: Vec<&str> = summary.lines().collect();
    assert_eq!(summary_lines.len(), 2);
    assert!(summary_lines[0].contains("ac_payload_bits_raw"));
    assert!(summary_lines[0].contains("n1__total_bits"));

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(mixture_path);
    let _ = fs::remove_file(trace_path);
    let _ = fs::remove_file(nodes_path);
    let _ = fs::remove_file(summary_path);
}

#[test]
fn ac_log_loss_cli_is_deterministic_across_thread_counts() {
    let input_path = temp_path("det_input", "bin");
    let mixture_path = temp_path("det_mixture", "json");
    let prefix_one = temp_prefix("det_one");
    let prefix_two = temp_prefix("det_two");
    write_temp_file(
        &input_path,
        b"deterministic diagnostic payload with nested mixture experts",
    );
    write_temp_file(&mixture_path, &parallel_mixture_json());

    run_ac_log_loss(&input_path, &mixture_path, &prefix_one, "1");
    run_ac_log_loss(&input_path, &mixture_path, &prefix_two, "2");

    for ext in ["trace.tsv", "nodes.tsv", "summary.tsv"] {
        let lhs = fs::read_to_string(prefix_one.with_extension(ext)).expect("lhs file");
        let rhs = fs::read_to_string(prefix_two.with_extension(ext)).expect("rhs file");
        assert_eq!(lhs, rhs, "mismatch for {ext}");
        let _ = fs::remove_file(prefix_one.with_extension(ext));
        let _ = fs::remove_file(prefix_two.with_extension(ext));
    }

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(mixture_path);
}
