#![cfg(all(feature = "cli", feature = "backend-ctw"))]

use std::process::{Command, Stdio};

#[test]
fn warmstart_dispatch_does_not_require_default_backends() {
    let output = Command::new(env!("CARGO_BIN_EXE_infotheory"))
        .args(["warmstart", "teacher", "merge"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn warmstart command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    assert!(
        stderr.contains("Error: warmstart failed: missing required --teacher"),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("requires infotheory feature 'backend-rosa'"),
        "warmstart dispatch must not build the default rate backend before parsing: {stderr}"
    );
    assert!(
        !stderr.contains("requires infotheory feature 'backend-zpaq'"),
        "warmstart dispatch must not build the default compression backend before parsing: {stderr}"
    );
}
