use infotheory::{entropy_rate_backend, RateBackend};

#[test]
fn zpaq_rate_backend_compresses_copy_data() {
    let mut data = Vec::new();
    let pattern = b"copy-like-pattern-";
    for _ in 0..512 {
        data.extend_from_slice(pattern);
    }
    let backend = RateBackend::Zpaq {
        method: "2".to_string(),
    };
    let rate = entropy_rate_backend(&data, -1, &backend);
    assert!(
        rate < 0.5,
        "expected low entropy rate for copy-like data, got {rate:.4}"
    );
}
