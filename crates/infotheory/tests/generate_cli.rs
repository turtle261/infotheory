#![cfg(all(feature = "cli", feature = "all-backends"))]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

const PROMPT: &[u8] = b"If a frog is green, dogs are red.\nIf a toad is green, cats are red.\nIf a dog is green, frogs are red.\nIf a cat is green, toads are red.\nIf a frog is red, dogs are green.\nIf a toad is red, cats are green.\nIf a dog is red, frogs are green.\nIf a cat is red, toads are \n";

fn temp_path(name: &str, ext: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("infotheory_generate_{name}_{ts}.{ext}"))
}

fn write_temp_file(name: &str, ext: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_path(name, ext);
    fs::write(&path, bytes).expect("write temp file");
    path
}

fn run_generate(args: &[&str], stdin_bytes: Option<&[u8]>) -> Vec<u8> {
    let bin = env!("CARGO_BIN_EXE_infotheory");
    let mut child = Command::new(bin)
        .args(args)
        .stdin(if stdin_bytes.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn generate");

    if let Some(bytes) = stdin_bytes {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(bytes).expect("write stdin");
    }

    let output = child.wait_with_output().expect("wait generate");
    assert!(
        output.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn generate_cli_rosaplus_predicts_green_from_file_and_stdin() {
    let prompt_path = write_temp_file("prompt", "txt", PROMPT);
    let path_str = prompt_path.to_string_lossy().to_string();

    let from_file = run_generate(
        &[
            "generate",
            &path_str,
            "--rate-backend",
            "rosaplus",
            "--bytes",
            "8",
            "--sample",
            "--seed",
            "42",
        ],
        None,
    );
    assert_eq!(from_file, b" green.\n");

    let from_stdin = run_generate(
        &[
            "generate",
            "--rate-backend",
            "rosaplus",
            "--bytes",
            "8",
            "--sample",
            "--seed",
            "42",
        ],
        Some(PROMPT),
    );
    assert_eq!(from_stdin, b" green.\n");

    let _ = fs::remove_file(prompt_path);
}

/// When stdin is piped and the sole positional parses as an integer,
/// the CLI should interpret it as `max_order` (not try to open it as a file).
#[test]
fn generate_cli_stdin_with_max_order_positional() {
    let from_stdin_with_order = run_generate(
        &[
            "generate",
            "8",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--bytes",
            "4",
            "--greedy",
        ],
        Some(PROMPT),
    );
    assert_eq!(
        from_stdin_with_order.len(),
        4,
        "should interpret '8' as max_order and read prompt from stdin"
    );
}

#[test]
fn generate_cli_backend_matrix_emits_requested_bytes() {
    let prompt_path = write_temp_file("matrix_prompt", "txt", PROMPT);
    let path_str = prompt_path.to_string_lossy().to_string();

    let cases = [
        vec![
            "generate",
            path_str.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "32",
            "--bytes",
            "8",
            "--greedy",
        ],
        vec![
            "generate",
            path_str.as_str(),
            "--rate-backend",
            "rosaplus",
            "--bytes",
            "8",
            "--sample",
            "--seed",
            "42",
        ],
        vec![
            "generate",
            path_str.as_str(),
            "--rate-backend",
            "match",
            "--bytes",
            "8",
            "--greedy",
        ],
        vec![
            "generate",
            path_str.as_str(),
            "--rate-backend",
            "ppmd",
            "--method",
            "12",
            "--bytes",
            "8",
            "--greedy",
        ],
        vec![
            "generate",
            path_str.as_str(),
            "--rate-backend",
            "sequitur",
            "--method",
            "64",
            "--bytes",
            "8",
            "--greedy",
        ],
    ];

    for args in cases {
        let out = run_generate(&args, None);
        assert_eq!(out.len(), 8, "args={args:?}");
    }

    let _ = fs::remove_file(prompt_path);
}

#[cfg(feature = "backend-rwkv")]
#[test]
fn generate_cli_rwkv_emits_requested_bytes() {
    let prompt_path = write_temp_file("rwkv_prompt", "txt", PROMPT);
    let path_str = prompt_path.to_string_lossy().to_string();
    let out = run_generate(
        &[
            "generate",
            &path_str,
            "--rate-backend",
            "rwkv",
            "--method",
            "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer",
            "--bytes",
            "8",
            "--sample",
            "--seed",
            "42",
        ],
        None,
    );
    assert_eq!(out.len(), 8);
    let _ = fs::remove_file(prompt_path);
}

#[test]
fn generate_cli_supports_expert_spec_and_mixture_spec() {
    let prompt_path = write_temp_file("spec_prompt", "txt", PROMPT);
    let prompt_str = prompt_path.to_string_lossy().to_string();

    let expert_path = write_temp_file(
        "expert",
        "json",
        serde_json::to_vec(&json!({
            "name": "ppmd",
            "kind": "ppmd",
            "order": 12,
            "memory_mb": 8
        }))
        .expect("expert json")
        .as_slice(),
    );
    let expert_str = expert_path.to_string_lossy().to_string();
    let expert_out = run_generate(
        &[
            "generate",
            &prompt_str,
            "--expert-spec",
            &expert_str,
            "--bytes",
            "8",
            "--greedy",
        ],
        None,
    );
    assert_eq!(expert_out.len(), 8);

    let experts = vec![
        json!({"name": "ctw", "kind": "ctw", "depth": 32, "log_prior": 0.0}),
        json!({"name": "ppmd", "kind": "ppmd", "order": 12, "memory_mb": 8, "log_prior": 0.0}),
        json!({"name": "rosa", "kind": "rosaplus", "max_order": -1, "log_prior": 0.0}),
        json!({"name": "match", "kind": "match", "log_prior": 0.0}),
    ];
    #[cfg(feature = "backend-rwkv")]
    let experts = {
        let mut experts = experts;
        experts.push(json!({
            "name": "rwkv",
            "kind": "rwkv",
            "method": "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,v_rank=8,g_rank=8,seed=31,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer",
            "log_prior": 0.0
        }));
        experts
    };

    let mixture_path = write_temp_file(
        "mixture",
        "json",
        serde_json::to_vec(&json!({
            "kind": "neural",
            "alpha": 0.03,
            "experts": experts
        }))
        .expect("mixture json")
        .as_slice(),
    );
    let mixture_str = mixture_path.to_string_lossy().to_string();
    let mix_out = run_generate(
        &[
            "generate",
            &prompt_str,
            "--rate-backend",
            "mixture",
            "--method",
            &mixture_str,
            "--bytes",
            "8",
            "--sample",
            "--seed",
            "42",
        ],
        None,
    );
    assert_eq!(mix_out.len(), 8);

    let _ = fs::remove_file(prompt_path);
    let _ = fs::remove_file(expert_path);
    let _ = fs::remove_file(mixture_path);
}
