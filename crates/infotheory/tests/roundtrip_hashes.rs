#![cfg(feature = "backend-zpaq")]

use infotheory::api::{
    CompressionBackend, try_compress_bytes_backend, try_decompress_bytes_backend,
};
use sha2::{Digest, Sha256};
#[cfg(feature = "cli")]
use std::io::Write;
#[cfg(feature = "cli")]
use std::path::Path;
#[cfg(feature = "cli")]
use std::process::{Command, Stdio};

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn zpaq_roundtrip_fixture_a_and_hash_stability() {
    let input = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/fixture_a.txt"
    ))
    .expect("failed to read fixture_a");
    let backend = CompressionBackend::Zpaq {
        method: "5".to_string(),
    };
    let compressed = try_compress_bytes_backend(&input, &backend).expect("compress failed");
    let restored = try_decompress_bytes_backend(&compressed, &backend).expect("decompress failed");
    assert_eq!(restored, input, "zpaq roundtrip mismatch");
    assert_eq!(
        sha256_hex(&compressed),
        "26ad22d35f5f014d7b99a403af46a0c2b172986352ffee21a03d1f7a39d67498",
        "compressed bytes hash changed unexpectedly"
    );
}

#[test]
fn zpaq_roundtrip_fixture_b_and_hash_stability() {
    let input = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/fixture_b.txt"
    ))
    .expect("failed to read fixture_b");
    let backend = CompressionBackend::Zpaq {
        method: "5".to_string(),
    };
    let compressed = try_compress_bytes_backend(&input, &backend).expect("compress failed");
    let restored = try_decompress_bytes_backend(&compressed, &backend).expect("decompress failed");
    assert_eq!(restored, input, "zpaq roundtrip mismatch");
    assert_eq!(
        sha256_hex(&compressed),
        "df691b88c9c1a9791b57f3e7d70fc05c6bb7a324f71e9b45900696472befb837",
        "compressed bytes hash changed unexpectedly"
    );
}

#[cfg(feature = "cli")]
#[test]
fn batch_metrics_output_hash_stability() {
    let Some(bin) = option_env!("CARGO_BIN_EXE_infotheory") else {
        // Binary is only built in cli-enabled test runs.
        return;
    };
    if !Path::new(bin).exists() {
        // Some test invocations expose CARGO_BIN_EXE_* without building
        // the required-features binary; skip in those configurations.
        return;
    }
    let mut child = Command::new(bin)
        .arg("batch")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn infotheory batch");
    {
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        writeln!(
            stdin,
            "{}",
            serde_json::json!({
                "op": "metrics",
                "text": "abracadabra",
                "max_order": 3,
            })
        )
        .expect("failed to write payload");
    }

    let output = child.wait_with_output().expect("failed to wait on batch");
    assert!(
        output.status.success(),
        "batch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8(output.stdout)
        .expect("stdout must be utf8")
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    assert_eq!(
        line, r#"{"h0":2.040373,"h_rate":1.763318,"id":0.135787,"len":11}"#,
        "batch output changed unexpectedly"
    );
    assert_eq!(
        sha256_hex(line.as_bytes()),
        "11c696ec8f63b2b95df1ed16d6e3360fb3788788a4bde80c2510d45a75cd8f88",
        "batch metrics line hash changed unexpectedly"
    );
}
