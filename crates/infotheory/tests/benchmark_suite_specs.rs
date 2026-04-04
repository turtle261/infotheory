use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn load_example(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join("bench")
        .join(name);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse {} as JSON: {e}", path.display()))
}

#[test]
fn extra_suite_includes_expected_uncovered_backends() {
    let v = load_example("extra.json");
    assert_eq!(v["kind"], "neural");
    let experts = v["experts"]
        .as_array()
        .expect("extra.json must contain experts array");
    assert!(
        !experts.is_empty(),
        "extra.json experts array must be non-empty"
    );

    let mut saw_mamba = false;
    let mut saw_particle_fast = false;
    let mut saw_sparse_match = false;

    for expert in experts {
        let kind = expert["kind"].as_str().unwrap_or_default();
        match kind {
            "mamba" => {
                saw_mamba = true;
                let method = expert["method"]
                    .as_str()
                    .expect("mamba expert must define method");
                assert!(
                    method.contains("policy:schedule="),
                    "mamba method should include an explicit schedule policy"
                );
            }
            "particle" => {
                let spec_path = expert["spec_path"]
                    .as_str()
                    .expect("particle expert must use spec_path");
                if spec_path == "particle_fast.json" {
                    saw_particle_fast = true;
                }
            }
            "sparse-match" => {
                saw_sparse_match = true;
            }
            _ => {}
        }
    }

    assert!(saw_mamba, "extra.json must include a mamba expert");
    assert!(
        saw_particle_fast,
        "extra.json must include particle_fast.json-backed particle expert"
    );
    assert!(
        saw_sparse_match,
        "extra.json must include a sparse-match expert"
    );
}
