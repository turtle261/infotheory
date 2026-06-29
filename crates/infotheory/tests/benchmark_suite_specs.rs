use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn read_json(path: &PathBuf) -> Value {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse {} as JSON: {e}", path.display()))
}

fn load_example(name: &str) -> Value {
    let path = repo_root().join("configs").join("bench").join(name);
    read_json(&path)
}

#[test]
fn two_json_benchmark_specs_are_pinned_and_canonical() {
    let root = repo_root();
    let config_path = root.join("configs").join("bench").join("two.json");
    let example_path = root.join("examples").join("two.json");
    let config_raw = fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", config_path.display()));
    let example_raw = fs::read_to_string(&example_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", example_path.display()));

    assert_eq!(
        config_raw, example_raw,
        "configs/bench/two.json and examples/two.json must stay byte-identical"
    );

    let v: Value = serde_json::from_str(&config_raw)
        .unwrap_or_else(|e| panic!("failed to parse {} as JSON: {e}", config_path.display()));
    assert_eq!(v["kind"], "neural");
    assert_eq!(
        v["alpha"].as_f64(),
        Some(0.03),
        "two.json preserves the historical alpha used by benchmark baselines"
    );

    let experts = v["experts"]
        .as_array()
        .expect("two.json must contain experts array");
    assert!(
        experts
            .iter()
            .any(|expert| expert["kind"] == "rwkv7" && expert["name"] == "rwkv7"),
        "two.json must use canonical rwkv7 kind/name spelling"
    );
    assert!(
        experts.iter().all(|expert| expert["name"] != "rwkv"),
        "two.json must not retain stale rwkv expert labels"
    );

    let fac_ctw = experts
        .iter()
        .find(|expert| expert["kind"] == "fac-ctw")
        .expect("two.json must include the canonical factorized CTW subject");
    assert_eq!(
        fac_ctw["name"], "fac-ctw",
        "the canonical CTW benchmark slot must be named fac-ctw, not stale ctw"
    );
    assert_eq!(
        fac_ctw["encoding_bits"].as_u64(),
        Some(8),
        "canonical fac-ctw benchmark subject must be byte-width"
    );
    assert_eq!(
        fac_ctw["num_percept_bits"].as_u64(),
        Some(8),
        "canonical fac-ctw benchmark subject must expose the 8-bit percept width"
    );
    assert_eq!(
        fac_ctw["msb_first"].as_bool(),
        Some(true),
        "canonical fac-ctw benchmark subject must explicitly select MSB-first byte order"
    );
}

#[test]
fn two_sse_profile_wraps_canonical_two_json() {
    let v = load_example("two_sse.json");
    assert_eq!(v["base"]["kind"], "mixture");
    assert_eq!(v["base"]["spec_path"], "two.json");
    assert_eq!(v["context"], "textrepeat");
    assert_eq!(v["bins"].as_u64(), Some(32));
    assert_eq!(v["learning_rate"].as_f64(), Some(0.03125));
    assert_eq!(v["bias_clip"].as_f64(), Some(16.0));

    #[cfg(all(
        feature = "backend-calibrated",
        feature = "backend-mixture",
        feature = "backend-ctw",
        feature = "backend-ppmd",
        feature = "backend-rosa",
        feature = "backend-match",
        feature = "backend-rwkv"
    ))]
    {
        let root = repo_root();
        let path = root.join("configs").join("bench").join("two_sse.json");
        let spec = infotheory::spec::load_calibrated_spec(
            path.to_str().expect("two_sse path should be UTF-8"),
        )
        .expect("two_sse.json should load as a calibrated spec");
        let backend = infotheory::api::RateBackend::Calibrated {
            spec: std::sync::Arc::new(spec),
        };
        backend
            .compile()
            .expect("two_sse calibrated backend should compile");
    }
}

#[test]
fn two_all_sse_profile_calibrates_root_and_each_expert() {
    let v = load_example("two_all_sse.json");
    assert_eq!(v["base"]["kind"], "mixture");
    assert_eq!(v["base"]["spec"]["kind"], "neural");
    assert_eq!(v["context"], "textrepeat");

    let experts = v["base"]["spec"]["experts"]
        .as_array()
        .expect("two_all_sse.json must contain mixture experts");
    assert_eq!(experts.len(), 5);
    assert!(
        experts.iter().all(|expert| expert["kind"] == "calibrated"),
        "every two_all_sse expert should be individually calibrated"
    );
    assert!(
        experts
            .iter()
            .all(|expert| expert["spec"]["context"] == "textrepeat")
    );

    #[cfg(all(
        feature = "backend-calibrated",
        feature = "backend-mixture",
        feature = "backend-ctw",
        feature = "backend-ppmd",
        feature = "backend-rosa",
        feature = "backend-match",
        feature = "backend-rwkv"
    ))]
    {
        let root = repo_root();
        let path = root.join("configs").join("bench").join("two_all_sse.json");
        let spec = infotheory::spec::load_calibrated_spec(
            path.to_str().expect("two_all_sse path should be UTF-8"),
        )
        .expect("two_all_sse.json should load as a calibrated spec");
        let backend = infotheory::api::RateBackend::Calibrated {
            spec: std::sync::Arc::new(spec),
        };
        backend
            .compile()
            .expect("two_all_sse calibrated backend should compile");
    }
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
