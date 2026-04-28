#[cfg(feature = "backend-zpaq")]
use infotheory::api::{RateBackend, try_entropy_rate_backend};

#[test]
#[cfg(feature = "backend-zpaq")]
fn zpaq_rate_backend_compresses_copy_data() {
    let mut data = Vec::new();
    let pattern = b"copy-like-pattern-";
    for _ in 0..512 {
        data.extend_from_slice(pattern);
    }
    let backend = RateBackend::Zpaq {
        method: infotheory::api::ZpaqMethodSpec::literal("2"),
    };
    let backend = backend.compile().expect("compile zpaq rate backend");
    let rate = try_entropy_rate_backend(&data, &backend).expect("entropy rate");
    assert!(
        rate < 0.5,
        "expected low entropy rate for copy-like data, got {rate:.4}"
    );
}
