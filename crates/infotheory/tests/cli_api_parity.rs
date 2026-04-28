#![cfg(all(feature = "cli", feature = "backend-rosa", feature = "backend-zpaq"))]

use std::io::Write;
use std::process::{Command, Stdio};

use infotheory::api::{
    NcdVariant, empirical_entropy_bytes, try_biased_entropy_rate_bytes,
    try_cross_entropy_rate_bytes, try_entropy_rate_bytes, try_ncd_matrix_bytes, try_ncd_paths,
};
use serde_json::Value;

fn run_batch(input: &Value) -> Value {
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
        let line = serde_json::to_string(input).expect("failed to encode json input");
        writeln!(stdin, "{line}").expect("failed to write json input");
    }

    let output = child
        .wait_with_output()
        .expect("failed to read batch output");
    assert!(
        output.status.success(),
        "batch exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout is not utf-8");
    serde_json::from_str(stdout.lines().next().unwrap_or("")).expect("invalid batch json")
}

fn as_f64(obj: &Value, key: &str) -> f64 {
    obj.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing/invalid key '{key}' in {obj}"))
}

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tol,
        "{label} mismatch: actual={actual}, expected={expected}, delta={delta}, tol={tol}"
    );
}

fn rosa_distance_like_cli(x: &[u8], y: &[u8]) -> f64 {
    let h_x_x = try_biased_entropy_rate_bytes(x).expect("biased entropy x");
    let h_y_y = try_biased_entropy_rate_bytes(y).expect("biased entropy y");
    let h_y_x = try_cross_entropy_rate_bytes(x, y).expect("cross entropy y|x");
    let h_x_y = try_cross_entropy_rate_bytes(y, x).expect("cross entropy x|y");
    if h_x_x < 1e-9 || h_y_y < 1e-9 {
        return 1.0;
    }
    (0.5 * (h_y_x / h_x_x + h_x_y / h_y_y) - 1.0).clamp(0.0, 1.0)
}

#[test]
fn metrics_text_parity_with_library() {
    let text = "entropy parity text";
    let out = run_batch(&serde_json::json!({
        "op": "metrics",
        "text": text,
    }));

    let data = text.as_bytes();
    let h0 = empirical_entropy_bytes(data);
    let h_rate = try_entropy_rate_bytes(data).expect("h_rate");
    let id = ((h0 - h_rate) / h0).clamp(0.0, 1.0);

    assert_close(as_f64(&out, "h0"), h0, 1e-6, "h0");
    assert_close(as_f64(&out, "h_rate"), h_rate, 1e-6, "h_rate");
    assert_close(as_f64(&out, "id"), id, 1e-6, "id");
    assert_eq!(
        out.get("len").and_then(Value::as_u64),
        Some(data.len() as u64),
        "length mismatch"
    );
}

#[test]
fn metrics_file_parity_with_library() {
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fixture_a.txt");
    let bytes = std::fs::read(fixture).expect("failed to read fixture");
    let out = run_batch(&serde_json::json!({
        "op": "metrics_file",
        "path": fixture,
    }));

    let h0 = empirical_entropy_bytes(&bytes);
    let h_rate = try_entropy_rate_bytes(&bytes).expect("h_rate");
    let id = if h0 < 1e-9 {
        0.0
    } else {
        ((h0 - h_rate) / h0).clamp(0.0, 1.0)
    };

    assert_close(as_f64(&out, "h0"), h0, 1e-6, "h0");
    assert_close(as_f64(&out, "h_rate"), h_rate, 1e-6, "h_rate");
    assert_close(as_f64(&out, "id"), id, 1e-6, "id");
    assert_eq!(
        out.get("len").and_then(Value::as_u64),
        Some(bytes.len() as u64),
        "length mismatch"
    );
}

#[test]
fn ncd_file_parity_with_library() {
    let a = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fixture_a.txt");
    let b = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fixture_b.txt");
    let out = run_batch(&serde_json::json!({
        "op": "ncd_files",
        "path1": a,
        "path2": b,
        "method": "5",
        "variant": "vitanyi",
    }));
    let rust_val = try_ncd_paths(a, b, "5", NcdVariant::Vitanyi).expect("ncd");
    assert_close(as_f64(&out, "ncd"), rust_val, 1e-6, "ncd");
}

#[test]
fn cross_entropy_parity_with_library() {
    let x = "abracadabra";
    let y = "alakazam";
    let out = run_batch(&serde_json::json!({
        "op": "cross_entropy",
        "text_x": x,
        "text_y": y,
    }));
    let rust_val = try_cross_entropy_rate_bytes(x.as_bytes(), y.as_bytes()).expect("cross entropy");
    assert_close(
        as_f64(&out, "cross_entropy"),
        rust_val,
        1e-6,
        "cross_entropy",
    );
}

#[test]
fn batch_metrics_parity_with_library() {
    let texts = vec!["abracadabra", "alakazam", "xyzxyz"];
    let out = run_batch(&serde_json::json!({
        "op": "batch_metrics",
        "texts": texts,
    }));
    let rows = out
        .get("results")
        .and_then(Value::as_array)
        .expect("missing results array");
    assert_eq!(rows.len(), texts.len());

    for (idx, text) in texts.iter().enumerate() {
        let data = text.as_bytes();
        let h0 = empirical_entropy_bytes(data);
        let h_rate = try_entropy_rate_bytes(data).expect("h_rate");
        let id = if h0 < 1e-9 {
            0.0
        } else {
            ((h0 - h_rate) / h0).clamp(0.0, 1.0)
        };
        assert_close(as_f64(&rows[idx], "h0"), h0, 1e-6, "h0");
        assert_close(as_f64(&rows[idx], "h_rate"), h_rate, 1e-6, "h_rate");
        assert_close(as_f64(&rows[idx], "id"), id, 1e-6, "id");
        assert_eq!(
            rows[idx].get("len").and_then(Value::as_u64),
            Some(data.len() as u64)
        );
    }
}

#[test]
fn ncd_matrix_parity_with_library() {
    let texts = vec!["abracadabra", "alakazam", "xyzxyz"];
    let datas: Vec<Vec<u8>> = texts.iter().map(|s| s.as_bytes().to_vec()).collect();
    let out = run_batch(&serde_json::json!({
        "op": "ncd_matrix",
        "texts": texts,
        "method": "5",
        "variant": "sym",
    }));
    let n = out.get("n").and_then(Value::as_u64).expect("missing n") as usize;
    let matrix = out
        .get("matrix")
        .and_then(Value::as_array)
        .expect("missing matrix");
    assert_eq!(matrix.len(), n);
    let rust_flat =
        try_ncd_matrix_bytes(&datas, "5", NcdVariant::SymVitanyi).expect("ncd matrix bytes");
    for i in 0..n {
        let row = matrix[i].as_array().expect("row must be array");
        for j in 0..n {
            let val = row[j].as_f64().expect("entry must be number");
            assert_close(val, rust_flat[i * n + j], 1e-6, "ncd_matrix");
        }
    }
}

#[test]
fn rosa_matrix_parity_with_library_formula() {
    let texts = vec!["abracadabra", "alakazam", "xyzxyz"];
    let out = run_batch(&serde_json::json!({
        "op": "rosa_matrix",
        "texts": texts,
    }));
    let n = out.get("n").and_then(Value::as_u64).expect("missing n") as usize;
    let matrix = out
        .get("matrix")
        .and_then(Value::as_array)
        .expect("missing matrix");
    assert_eq!(matrix.len(), n);

    for i in 0..n {
        let row = matrix[i].as_array().expect("row must be array");
        for j in 0..n {
            let val = row[j].as_f64().expect("entry must be number");
            let expected = if i == j {
                0.0
            } else {
                rosa_distance_like_cli(texts[i].as_bytes(), texts[j].as_bytes())
            };
            assert_close(val, expected, 1e-6, "rosa_matrix");
        }
    }
}
