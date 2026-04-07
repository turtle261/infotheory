#![cfg(feature = "backend-rosa")]

use infotheory::api::{CompressionBackend, RateBackend, validate_compression_backend};
use infotheory::coders::CoderType;
use infotheory::compression::FramingMode;

#[test]
fn validate_compression_backend_accepts_rate_ac_and_rate_rans_without_rwkv_feature() {
    let ac = CompressionBackend::Rate {
        rate_backend: RateBackend::RosaPlus,
        coder: CoderType::AC,
        framing: FramingMode::Framed,
    };
    validate_compression_backend(&ac).expect("rate-ac validation should not panic or fail");

    let rans = CompressionBackend::Rate {
        rate_backend: RateBackend::RosaPlus,
        coder: CoderType::RANS,
        framing: FramingMode::Framed,
    };
    validate_compression_backend(&rans).expect("rate-rans validation should not panic or fail");
}
