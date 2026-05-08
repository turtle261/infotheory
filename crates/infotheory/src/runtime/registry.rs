//! Backend registry descriptor lookups and feature-gate error helpers.

use super::{
    BackendDescriptor, COMPRESSION_BACKEND_REGISTRY, CompressionBackendDescriptor,
    CompressionBackendKind, RATE_BACKEND_REGISTRY, RateBackendDescriptor, RateBackendKind,
};

pub(crate) fn find_backend_descriptor_in_registry<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    input: &str,
) -> Option<&'static BackendDescriptor<K>> {
    let key = input.trim().to_ascii_lowercase();
    registry
        .iter()
        .find(|descriptor| descriptor.aliases.iter().any(|alias| *alias == key))
}

fn backend_descriptor_by_kind<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    kind: K,
) -> Option<&'static BackendDescriptor<K>> {
    registry.iter().find(|descriptor| descriptor.kind == kind)
}

pub(crate) fn backend_descriptor_by_kind_checked<K: Copy + Eq>(
    registry: &'static [BackendDescriptor<K>],
    kind: K,
    registry_name: &'static str,
) -> Result<&'static BackendDescriptor<K>, String>
where
    K: std::fmt::Debug,
{
    backend_descriptor_by_kind(registry, kind).ok_or_else(|| {
        format!(
            "internal backend registry mismatch: backend '{kind:?}' is missing from {registry_name}"
        )
    })
}

pub(crate) fn describe_rate_backend_kind(
    kind: RateBackendKind,
) -> Result<&'static RateBackendDescriptor, String> {
    backend_descriptor_by_kind_checked(RATE_BACKEND_REGISTRY, kind, "RATE_BACKEND_REGISTRY")
}

pub(crate) fn describe_compression_backend_kind(
    kind: CompressionBackendKind,
) -> Result<&'static CompressionBackendDescriptor, String> {
    backend_descriptor_by_kind_checked(
        COMPRESSION_BACKEND_REGISTRY,
        kind,
        "COMPRESSION_BACKEND_REGISTRY",
    )
}

#[cfg(any(
    test,
    not(feature = "backend-rosa"),
    not(feature = "backend-ctw"),
    not(feature = "backend-match"),
    not(feature = "backend-ppmd"),
    not(feature = "backend-sequitur"),
    not(feature = "backend-zpaq"),
    not(feature = "backend-mixture"),
    not(feature = "backend-particle"),
    not(feature = "backend-calibrated"),
    not(feature = "backend-mamba"),
    not(feature = "backend-rwkv")
))]
pub(super) fn rate_backend_feature_error(kind: RateBackendKind) -> String {
    describe_rate_backend_kind(kind)
        .map(|descriptor| match descriptor.feature {
            Some(feature) => format!(
                "backend '{}' requires infotheory feature '{}'",
                descriptor.canonical, feature
            ),
            None => format!("backend '{}' is unavailable", descriptor.canonical),
        })
        .unwrap_or_else(|err| err)
}

#[cfg(any(test, not(feature = "backend-rwkv")))]
pub(super) fn compression_backend_feature_error(kind: CompressionBackendKind) -> String {
    describe_compression_backend_kind(kind)
        .map(|descriptor| match descriptor.feature {
            Some(feature) => format!(
                "compression backend '{}' requires infotheory feature '{}'",
                descriptor.canonical, feature
            ),
            None => format!(
                "compression backend '{}' is unavailable",
                descriptor.canonical
            ),
        })
        .unwrap_or_else(|err| err)
}
