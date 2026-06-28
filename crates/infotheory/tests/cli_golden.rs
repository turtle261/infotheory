#![cfg(all(
    feature = "cli",
    feature = "backend-rosa",
    feature = "backend-rwkv",
    feature = "backend-zpaq"
))]

use std::io::Write;
use std::process::{Command, Stdio};

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    input: Value,
    expected: Value,
    float_tolerance: Option<f64>,
}

fn assert_json_close(expected: &Value, actual: &Value, tolerance: f64, path: &str) {
    match (expected, actual) {
        (Value::Null, Value::Null) => {}
        (Value::Bool(a), Value::Bool(b)) => assert_eq!(a, b, "{path}: bool mismatch"),
        (Value::String(a), Value::String(b)) => assert_eq!(a, b, "{path}: string mismatch"),
        (Value::Number(a), Value::Number(b)) => {
            let a_f = a
                .as_f64()
                .unwrap_or_else(|| panic!("{path}: expected number not representable as f64"));
            let b_f = b
                .as_f64()
                .unwrap_or_else(|| panic!("{path}: actual number not representable as f64"));
            assert!(
                (a_f - b_f).abs() <= tolerance,
                "{path}: number mismatch, expected {a_f}, got {b_f}, tolerance {tolerance}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: array length mismatch");
            for (index, (ea, eb)) in a.iter().zip(b.iter()).enumerate() {
                assert_json_close(ea, eb, tolerance, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: object size mismatch");
            for (key, expected_value) in a {
                let next_path = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}.{key}")
                };
                let actual_value = b
                    .get(key)
                    .unwrap_or_else(|| panic!("{next_path}: missing key in output"));
                assert_json_close(expected_value, actual_value, tolerance, &next_path);
            }
        }
        _ => panic!(
            "{path}: type mismatch, expected {expected:?}, got {actual:?}",
            expected = expected,
            actual = actual
        ),
    }
}

fn run_batch_case(input: &Value) -> Value {
    let bin = env!("CARGO_BIN_EXE_infotheory");
    let mut child = Command::new(bin)
        .arg("batch")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn infotheory batch");

    {
        let stdin = child.stdin.as_mut().expect("failed to open stdin");
        let line = serde_json::to_string(input).expect("failed to serialize input JSON");
        writeln!(stdin, "{line}").expect("failed to write batch input");
    }

    let output = child
        .wait_with_output()
        .expect("failed to wait for batch process");
    assert!(
        output.status.success(),
        "infotheory batch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout is not valid UTF-8");
    let line = stdout.lines().next().unwrap_or("").trim();
    assert!(!line.is_empty(), "infotheory batch returned empty output");
    serde_json::from_str(line).expect("batch output is not valid JSON")
}

#[test]
fn cli_batch_golden_outputs_match() {
    let cases: Vec<GoldenCase> =
        serde_json::from_str(include_str!("cli_golden_cases.json")).expect("invalid golden JSON");

    for case in &cases {
        let actual = run_batch_case(&case.input);
        let tolerance = case.float_tolerance.unwrap_or(0.0);
        assert_json_close(&case.expected, &actual, tolerance, &case.name);
    }
}

#[test]
fn cli_warmstart_usage_error_is_stable() {
    let bin = env!("CARGO_BIN_EXE_infotheory");
    let output = Command::new(bin)
        .args(["warmstart", "teacher", "merge"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn warmstart merge without options");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    assert!(
        stderr.contains("Error: warmstart failed: missing required --teacher"),
        "stderr={stderr}"
    );
}
