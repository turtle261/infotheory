#![cfg(all(feature = "cli", feature = "all-backends"))]

use infotheory::api::{empirical_entropy_bytes, try_entropy_rate_bytes};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str, ext: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("infotheory_cli_{name}_{ts}.{ext}"))
}

fn write_temp_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).expect("write temp file");
}

fn run_cli(args: &[&str], stdin_bytes: Option<&[u8]>) -> Output {
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
        .expect("spawn cli");

    if let Some(bytes) = stdin_bytes {
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(bytes)
            .expect("write stdin");
    }

    child.wait_with_output().expect("wait cli")
}

fn stdout_string(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout utf8")
}

fn stderr_string(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr utf8")
}

fn parse_stdout_f64(output: &Output) -> f64 {
    stdout_string(output)
        .trim()
        .parse::<f64>()
        .expect("stdout should be numeric")
}

fn parse_stdout_json_lines(output: &Output) -> Vec<Value> {
    stdout_string(output)
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stdout line should be JSON"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tol,
        "{label} mismatch: actual={actual}, expected={expected}, delta={delta}, tol={tol}"
    );
}

#[test]
fn cli_usage_and_unknown_primitive_paths_are_stable() {
    let no_args = run_cli(&[], None);
    assert!(no_args.status.success());
    assert!(stderr_string(&no_args).contains("InfoTheory CLI"));

    let help = run_cli(&["--help"], None);
    assert!(help.status.success());
    assert!(stderr_string(&help).contains("Usage: infotheory"));

    let a_path = temp_path("unknown_a", "txt");
    let b_path = temp_path("unknown_b", "txt");
    write_temp_file(&a_path, b"a");
    write_temp_file(&b_path, b"b");
    let unknown = run_cli(
        &[
            "definitely-unknown-primitive",
            a_path.to_string_lossy().as_ref(),
            b_path.to_string_lossy().as_ref(),
        ],
        None,
    );
    assert!(unknown.status.success());
    let stderr = stderr_string(&unknown);
    assert!(stderr.contains("Unknown primitive"));
    assert!(stderr.contains("InfoTheory CLI"));

    let _ = fs::remove_file(a_path);
    let _ = fs::remove_file(b_path);
}

#[test]
fn direct_cli_primitives_cover_empirical_and_backend_paths() {
    let a_path = temp_path("a", "txt");
    let b_path = temp_path("b", "txt");
    write_temp_file(&a_path, b"abracadabra abracadabra abracadabra");
    write_temp_file(&b_path, b"alakazam alakazam alakazam");
    let a = a_path.to_string_lossy().to_string();
    let b = b_path.to_string_lossy().to_string();

    let single_file_cases = [
        vec!["h", a.as_str()],
        vec![
            "h_rate",
            a.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["id", a.as_str(), "--rate-backend", "ctw", "--method", "8"],
    ];
    for args in single_file_cases {
        let output = run_cli(&args, None);
        assert!(
            output.status.success(),
            "single-file cli failed: args={args:?}, stderr={}",
            stderr_string(&output)
        );
        let value = parse_stdout_f64(&output);
        assert!(value.is_finite(), "args={args:?}");
    }

    let pair_cases = [
        vec!["mi", a.as_str(), b.as_str()],
        vec![
            "mi",
            a.as_str(),
            b.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["xe", a.as_str(), b.as_str()],
        vec![
            "xe",
            a.as_str(),
            b.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["ce", a.as_str(), b.as_str()],
        vec!["joint_entropy", a.as_str(), b.as_str()],
        vec![
            "joint_entropy",
            a.as_str(),
            b.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["ned", a.as_str(), b.as_str()],
        vec![
            "ned",
            a.as_str(),
            b.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["ned_cons", a.as_str(), b.as_str()],
        vec!["nte", a.as_str(), b.as_str()],
        vec![
            "rt",
            a.as_str(),
            b.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["tvd", a.as_str(), b.as_str()],
        vec!["nhd", a.as_str(), b.as_str()],
        vec!["kl", a.as_str(), b.as_str()],
        vec!["js", a.as_str(), b.as_str()],
        vec![
            "ncd",
            a.as_str(),
            b.as_str(),
            "--ncd-backend",
            "rate-ac",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        vec!["ncd_sym_cons", a.as_str(), b.as_str(), "5"],
    ];
    for args in pair_cases {
        let output = run_cli(&args, None);
        assert!(
            output.status.success(),
            "pair cli failed: args={args:?}, stderr={}",
            stderr_string(&output)
        );
        let value = parse_stdout_f64(&output);
        assert!(value.is_finite(), "args={args:?}");
    }

    let _ = fs::remove_file(a_path);
    let _ = fs::remove_file(b_path);
}

#[test]
fn search_cli_supports_prior_modes_and_granularity_flags() {
    let target_root = temp_path("search_target", "dir");
    let prior_root = temp_path("search_prior", "dir");
    fs::create_dir_all(&target_root).expect("create target dir");
    fs::create_dir_all(&prior_root).expect("create prior dir");

    write_temp_file(
        &target_root.join("match.txt"),
        b"needle exact search phrase\nneedle exact search phrase\n",
    );
    write_temp_file(
        &target_root.join("other.txt"),
        b"unrelated background text without the exact phrase\n",
    );
    write_temp_file(
        &prior_root.join("prior.txt"),
        b"needle prior corpus bytes\nexact search phrase prior bytes\n",
    );

    let target = target_root.to_string_lossy().to_string();
    let prior = prior_root.to_string_lossy().to_string();
    for mode in ["use", "summarize", "none"] {
        let output = run_cli(
            &[
                "search",
                "needle exact search phrase",
                target.as_str(),
                "--level",
                "file",
                "--prior",
                prior.as_str(),
                "--stage2-prior-mode",
                mode,
                "--top-k",
                "2",
                "--rate-backend",
                "ctw",
                "--method",
                "8",
            ],
            None,
        );
        assert!(
            output.status.success(),
            "search failed for mode {mode}: {}",
            stderr_string(&output)
        );
        let stdout = stdout_string(&output);
        assert!(stdout.contains("sed -n"));
        assert!(stdout.contains("match.txt"));
    }

    let _ = fs::remove_dir_all(target_root);
    let _ = fs::remove_dir_all(prior_root);
}

#[test]
fn compress_and_decompress_roundtrip_for_zpaq_and_rate_backends() {
    let input_path = temp_path("compress_input", "bin");
    let zpaq_out = temp_path("compress_zpaq", "itc");
    let zpaq_roundtrip = temp_path("compress_zpaq_roundtrip", "bin");
    let rate_out = temp_path("compress_rate", "itc");
    let rate_roundtrip = temp_path("compress_rate_roundtrip", "bin");
    let payload = b"roundtrip payload for compress/decompress command coverage";
    write_temp_file(&input_path, payload);
    let input = input_path.to_string_lossy().to_string();

    let zpaq_compress = run_cli(
        &[
            "compress",
            input.as_str(),
            zpaq_out.to_string_lossy().as_ref(),
            "--compression-backend",
            "zpaq",
            "--method",
            "5",
        ],
        None,
    );
    assert!(
        zpaq_compress.status.success(),
        "{}",
        stderr_string(&zpaq_compress)
    );
    assert!(stdout_string(&zpaq_compress).contains("compressed"));

    let zpaq_decompress = run_cli(
        &[
            "decompress",
            zpaq_out.to_string_lossy().as_ref(),
            zpaq_roundtrip.to_string_lossy().as_ref(),
            "--compression-backend",
            "zpaq",
            "--method",
            "5",
        ],
        None,
    );
    assert!(
        zpaq_decompress.status.success(),
        "{}",
        stderr_string(&zpaq_decompress)
    );
    assert_eq!(fs::read(&zpaq_roundtrip).expect("zpaq roundtrip"), payload);

    let rate_compress = run_cli(
        &[
            "compress",
            input.as_str(),
            rate_out.to_string_lossy().as_ref(),
            "--compression-backend",
            "rate-ac",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        None,
    );
    assert!(
        rate_compress.status.success(),
        "{}",
        stderr_string(&rate_compress)
    );

    let rate_decompress = run_cli(
        &[
            "decompress",
            rate_out.to_string_lossy().as_ref(),
            rate_roundtrip.to_string_lossy().as_ref(),
            "--compression-backend",
            "rate-ac",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        None,
    );
    assert!(
        rate_decompress.status.success(),
        "{}",
        stderr_string(&rate_decompress)
    );
    assert_eq!(fs::read(&rate_roundtrip).expect("rate roundtrip"), payload);

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(zpaq_out);
    let _ = fs::remove_file(zpaq_roundtrip);
    let _ = fs::remove_file(rate_out);
    let _ = fs::remove_file(rate_roundtrip);
}

#[test]
fn sequitur_debug_accepts_hex_and_file_input() {
    let output = run_cli(
        &[
            "sequitur-debug",
            "--hex",
            "616263616263",
            "--context-bytes",
            "32",
            "--alphabet-prefix",
            "8",
        ],
        None,
    );
    assert!(output.status.success(), "{}", stderr_string(&output));
    let parsed: Value = serde_json::from_slice(&output.stdout).expect("sequitur json");
    assert_eq!(
        parsed.get("context_bytes").and_then(Value::as_u64),
        Some(32)
    );
    assert_eq!(
        parsed.get("alphabet_prefix").and_then(Value::as_u64),
        Some(8)
    );
    assert_eq!(
        parsed
            .get("cases")
            .and_then(Value::as_array)
            .expect("cases")
            .len(),
        1
    );

    let input_path = temp_path("sequitur_input", "bin");
    write_temp_file(&input_path, b"abcabc");
    let file_output = run_cli(
        &[
            "sequitur-debug",
            input_path.to_string_lossy().as_ref(),
            "--context-bytes",
            "16",
        ],
        None,
    );
    assert!(
        file_output.status.success(),
        "{}",
        stderr_string(&file_output)
    );
    let parsed: Value = serde_json::from_slice(&file_output.stdout).expect("sequitur file json");
    assert_eq!(
        parsed.get("context_bytes").and_then(Value::as_u64),
        Some(16)
    );

    let _ = fs::remove_file(input_path);
}

#[test]
fn batch_cli_stream_handles_mixed_lines_with_error_isolation() {
    let fixture_path = temp_path("batch_fixture", "txt");
    let fixture_bytes = b"batch fixture entropy text";
    write_temp_file(&fixture_path, fixture_bytes);

    let missing_path = temp_path("batch_missing", "txt");
    let fixture_str = fixture_path.to_string_lossy().to_string();
    let missing_str = missing_path.to_string_lossy().to_string();

    let line1 = serde_json::json!({
        "op": "metrics",
        "text": "batch fixture entropy text"
    })
    .to_string();
    let line3 = serde_json::json!({
        "op": "metrics_file",
        "path": fixture_str
    })
    .to_string();
    let line4 = serde_json::json!({
        "op": "metrics_file",
        "path": missing_str
    })
    .to_string();
    let line6 = serde_json::json!({
        "op": "batch_metrics",
        "texts": ["abcabc", ""]
    })
    .to_string();

    let input = [
        line1,
        "{ invalid".to_string(),
        line3,
        line4,
        serde_json::json!({ "op": "nope" }).to_string(),
        line6,
        "".to_string(),
    ]
    .join("\n")
        + "\n";

    let output = run_cli(&["batch"], Some(input.as_bytes()));
    assert!(output.status.success(), "{}", stderr_string(&output));

    let lines = parse_stdout_json_lines(&output);
    assert_eq!(
        lines.len(),
        7,
        "batch should emit one JSON row per input line"
    );

    let h0 = empirical_entropy_bytes(fixture_bytes);
    let h_rate = try_entropy_rate_bytes(fixture_bytes).expect("h_rate");
    let id = ((h0 - h_rate) / h0).clamp(0.0, 1.0);

    assert_close(
        lines[0]["h0"].as_f64().expect("line 1 h0"),
        h0,
        1e-5,
        "line 1 h0",
    );
    assert_close(
        lines[0]["h_rate"].as_f64().expect("line 1 h_rate"),
        h_rate,
        1e-5,
        "line 1 h_rate",
    );
    assert_close(
        lines[0]["id"].as_f64().expect("line 1 id"),
        id,
        1e-5,
        "line 1 id",
    );
    assert_eq!(lines[0]["len"].as_u64(), Some(fixture_bytes.len() as u64));

    assert!(
        lines[1]["error"]
            .as_str()
            .expect("line 2 error")
            .contains("invalid json"),
        "line 2 should report malformed JSON"
    );

    assert_close(
        lines[2]["h0"].as_f64().expect("line 3 h0"),
        h0,
        1e-5,
        "line 3 h0",
    );
    assert_close(
        lines[2]["h_rate"].as_f64().expect("line 3 h_rate"),
        h_rate,
        1e-5,
        "line 3 h_rate",
    );
    assert_close(
        lines[2]["id"].as_f64().expect("line 3 id"),
        id,
        1e-5,
        "line 3 id",
    );
    assert_eq!(lines[2]["len"].as_u64(), Some(fixture_bytes.len() as u64));

    assert!(
        lines[3]["error"]
            .as_str()
            .expect("line 4 error")
            .contains("failed to read file"),
        "line 4 should report read failure for missing path"
    );

    assert_eq!(lines[4]["error"], "unknown op: nope");

    let batch_results = lines[5]["results"]
        .as_array()
        .expect("line 6 results array");
    assert_eq!(batch_results.len(), 2);
    assert_eq!(batch_results[1]["len"], 0);
    assert_eq!(batch_results[1]["h_rate"], 0);

    assert_eq!(lines[6]["error"], "empty input");

    let _ = fs::remove_file(fixture_path);
}

#[test]
fn search_cli_default_and_topk_paths_return_relevant_hit() {
    let target_root = temp_path("search_default_target", "dir");
    fs::create_dir_all(&target_root).expect("create target dir");

    write_temp_file(
        &target_root.join("match.txt"),
        b"needle exact CLI search phrase\nneedle exact CLI search phrase\n",
    );
    write_temp_file(
        &target_root.join("other.txt"),
        b"background text without the exact phrase\n",
    );

    let target = target_root.to_string_lossy().to_string();

    let default_output = run_cli(
        &["search", "needle exact CLI search phrase", target.as_str()],
        None,
    );
    assert!(
        default_output.status.success(),
        "{}",
        stderr_string(&default_output)
    );
    let default_stdout = stdout_string(&default_output);
    assert!(default_stdout.contains("sed -n"));
    assert!(default_stdout.contains("match.txt"));

    let topk_output = run_cli(
        &[
            "search",
            "needle exact CLI search phrase",
            target.as_str(),
            "--level",
            "file",
            "--top-k",
            "1",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        None,
    );
    assert!(
        topk_output.status.success(),
        "{}",
        stderr_string(&topk_output)
    );
    let topk_stdout = stdout_string(&topk_output);
    let topk_lines: Vec<&str> = topk_stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(
        topk_lines.len(),
        1,
        "--top-k 1 should emit one search command"
    );
    assert!(topk_lines[0].contains("match.txt"));

    let _ = fs::remove_dir_all(target_root);
}

#[test]
fn search_cli_reports_argument_and_target_failures() {
    let missing_target = run_cli(&["search", "needle only"], None);
    assert!(!missing_target.status.success());
    assert!(
        stderr_string(&missing_target).contains("Error: 'search' requires query and target path")
    );

    let nonexistent = temp_path("missing_search_target", "dir");
    let nonexistent_str = nonexistent.to_string_lossy().to_string();
    let missing_target_path = run_cli(
        &[
            "search",
            "needle exact phrase",
            nonexistent_str.as_str(),
            "--level",
            "file",
            "--rate-backend",
            "ctw",
            "--method",
            "8",
        ],
        None,
    );
    assert!(!missing_target_path.status.success());
    assert!(stderr_string(&missing_target_path).contains("Error: search failed"));
}

#[test]
fn search_cli_handles_expert_spec_unknown_flags_and_invalid_stage2_mode() {
    let target_root = temp_path("search_expert_target", "dir");
    fs::create_dir_all(&target_root).expect("create target dir");
    write_temp_file(
        &target_root.join("match.txt"),
        b"needle exact expert-spec query phrase\n",
    );

    let expert_spec_path = temp_path("search_expert_spec", "json");
    write_temp_file(
        &expert_spec_path,
        br#"{"name":"leaf","kind":"ctw","depth":8,"log_prior":0.0}"#,
    );

    let target = target_root.to_string_lossy().to_string();
    let expert = expert_spec_path.to_string_lossy().to_string();
    let output = run_cli(
        &[
            "search",
            "needle exact expert-spec query phrase",
            target.as_str(),
            "--stage2-prior-mode",
            "unknown-mode",
            "--bogus-flag",
            "ignored",
            "--expert-spec",
            expert.as_str(),
            "--level",
            "file",
            "--top-k",
            "1",
        ],
        None,
    );
    assert!(output.status.success());
    let stdout = stdout_string(&output);
    assert!(stdout.contains("sed -n"));
    assert!(stdout.contains("match.txt"));

    let _ = fs::remove_file(expert_spec_path);
    let _ = fs::remove_dir_all(target_root);
}

#[test]
fn ac_log_loss_reports_mixture_load_failures() {
    let input_path = temp_path("ac_log_loss_input", "txt");
    write_temp_file(&input_path, b"ac log loss fixture bytes");
    let input = input_path.to_string_lossy().to_string();
    let missing_spec = temp_path("ac_log_loss_missing", "json");
    let missing = missing_spec.to_string_lossy().to_string();
    let out_prefix = temp_path("ac_log_loss_out", "prefix");
    let out = out_prefix.to_string_lossy().to_string();

    let output = run_cli(
        &[
            "ac-log-loss",
            input.as_str(),
            "--mixture",
            missing.as_str(),
            "--out-prefix",
            out.as_str(),
        ],
        None,
    );
    assert!(!output.status.success());
    assert!(
        stderr_string(&output).contains("Error: failed to load mixture spec"),
        "stderr={}",
        stderr_string(&output)
    );

    let _ = fs::remove_file(input_path);
}

#[test]
fn generate_cli_rejects_invalid_numeric_sampling_flags() {
    let prompt_path = temp_path("generate_invalid_flags", "txt");
    write_temp_file(&prompt_path, b"prompt bytes for invalid flag validation");
    let prompt = prompt_path.to_string_lossy().to_string();

    let invalid_temperature = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--sample",
            "--temperature",
            "-1",
        ],
        None,
    );
    assert!(!invalid_temperature.status.success());
    assert!(
        stderr_string(&invalid_temperature)
            .contains("Error: --temperature must be finite and non-negative")
    );

    let invalid_top_p = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--sample",
            "--top-p",
            "1.5",
        ],
        None,
    );
    assert!(!invalid_top_p.status.success());
    assert!(stderr_string(&invalid_top_p).contains("Error: --top-p must be in (0, 1]"));

    let invalid_temperature_parse = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--sample",
            "--temperature",
            "nan-not-a-number",
        ],
        None,
    );
    assert!(!invalid_temperature_parse.status.success());
    assert!(
        stderr_string(&invalid_temperature_parse)
            .contains("Error: --temperature must be a finite number")
    );

    let invalid_seed = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--seed",
            "not-a-u64",
        ],
        None,
    );
    assert!(!invalid_seed.status.success());
    assert!(stderr_string(&invalid_seed).contains("Error: --seed must be an unsigned integer"));

    let invalid_top_k = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--sample",
            "--top-k",
            "not-a-usize",
        ],
        None,
    );
    assert!(!invalid_top_k.status.success());
    assert!(
        stderr_string(&invalid_top_k).contains("Error: --top-k must be a non-negative integer")
    );

    let invalid_top_p_parse = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--sample",
            "--top-p",
            "not-a-float",
        ],
        None,
    );
    assert!(!invalid_top_p_parse.status.success());
    assert!(
        stderr_string(&invalid_top_p_parse).contains("Error: --top-p must be a number in (0, 1]")
    );

    let adaptive = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--bytes",
            "4",
            "--adaptive",
        ],
        None,
    );
    assert!(adaptive.status.success());
    assert_eq!(adaptive.stdout.len(), 4);

    let invalid_bytes = run_cli(
        &[
            "generate",
            prompt.as_str(),
            "--rate-backend",
            "ctw",
            "--method",
            "8",
            "--bytes",
            "-1",
        ],
        None,
    );
    assert!(!invalid_bytes.status.success());
    assert!(
        stderr_string(&invalid_bytes).contains("Error: --bytes must be a non-negative integer")
    );

    let _ = fs::remove_file(prompt_path);
}

#[test]
fn sequitur_debug_rejects_invalid_numeric_flags_and_hex_payloads() {
    let invalid_context = run_cli(
        &[
            "sequitur-debug",
            "--hex",
            "616263",
            "--context-bytes",
            "abc",
        ],
        None,
    );
    assert!(!invalid_context.status.success());
    assert!(
        stderr_string(&invalid_context)
            .contains("Error: --context-bytes must be a positive integer")
    );

    let invalid_prefix = run_cli(
        &[
            "sequitur-debug",
            "--hex",
            "616263",
            "--alphabet-prefix",
            "abc",
        ],
        None,
    );
    assert!(!invalid_prefix.status.success());
    assert!(
        stderr_string(&invalid_prefix)
            .contains("Error: --alphabet-prefix must be a positive integer")
    );

    let invalid_hex = run_cli(&["sequitur-debug", "--hex", "xyz"], None);
    assert!(!invalid_hex.status.success());
    assert!(stderr_string(&invalid_hex).contains("invalid --hex input"));
}

#[test]
fn aixi_cli_requires_config_argument() {
    let output = run_cli(&["aixi"], None);
    assert!(!output.status.success());
    assert!(stderr_string(&output).contains("Error: 'aixi' requires config.json"));
}

#[test]
fn aixi_cli_reports_missing_gameengine_feature_for_builtin_environments() {
    let config_path = temp_path("aixi_builtin_config", "json");
    write_temp_file(
        &config_path,
        br#"{
  "schema_version": 1,
  "kind": "planner_run",
  "assets": [],
  "environment": { "kind": "builtin", "name": "coin_flip" },
  "interface": {
    "observation_bits": 1,
    "observation_stream_len": 1,
    "observation_key_mode": "full_stream",
    "reward_bits": 1,
    "agent_actions": 2,
    "min_reward": 0,
    "max_reward": 1,
    "reward_offset": 0
  },
  "controller": {
    "kind": "aiqi_discounted",
    "predictor": { "kind": "ctw", "depth": 8 },
    "discount_gamma": 0.99,
    "return_horizon": 2,
    "return_bins": 8,
    "augmentation_period": 2,
    "baseline_exploration": 0.01
  },
  "runtime": {
    "random_seed": 7,
    "learn_cycles": 1,
    "eval_cycles": 1,
    "terminate_lifetime": 2,
    "log_every": 1,
    "perf": false,
    "vm_perf_only": false,
    "explore_epsilon": 0.0,
    "explore_gamma": 1.0
  }
}"#,
    );
    let config = config_path.to_string_lossy().to_string();

    let output = run_cli(&["aixi", config.as_str()], None);
    assert!(!output.status.success());
    assert!(
        stderr_string(&output).contains("requires feature 'aixi-gameengine'"),
        "stderr={}",
        stderr_string(&output)
    );

    let _ = fs::remove_file(config_path);
}

#[test]
fn decompress_cli_reports_corruption_instead_of_succeeding() {
    let input_path = temp_path("corrupt_input", "bin");
    let compressed_path = temp_path("corrupt_compressed", "itc");
    let corrupt_path = temp_path("corrupt_truncated", "itc");
    let output_path = temp_path("corrupt_output", "bin");

    let payload = b"payload for corruption failure contract";
    write_temp_file(&input_path, payload);

    let compress = run_cli(
        &[
            "compress",
            input_path.to_string_lossy().as_ref(),
            compressed_path.to_string_lossy().as_ref(),
            "--compression-backend",
            "zpaq",
            "--method",
            "5",
        ],
        None,
    );
    assert!(compress.status.success(), "{}", stderr_string(&compress));

    let mut compressed = fs::read(&compressed_path).expect("read compressed artifact");
    assert!(
        compressed.len() > 8,
        "compressed payload too small for truncation test"
    );
    compressed.truncate(compressed.len() / 2);
    write_temp_file(&corrupt_path, &compressed);

    let decompress = run_cli(
        &[
            "decompress",
            corrupt_path.to_string_lossy().as_ref(),
            output_path.to_string_lossy().as_ref(),
            "--compression-backend",
            "zpaq",
            "--method",
            "5",
        ],
        None,
    );
    assert!(
        !decompress.status.success(),
        "corrupted stream should fail decompression"
    );
    assert!(
        stderr_string(&decompress).contains("decompression failed"),
        "stderr should expose decompression failure"
    );
    assert!(
        !output_path.exists(),
        "failed decompression must not materialize output file"
    );

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(compressed_path);
    let _ = fs::remove_file(corrupt_path);
}
