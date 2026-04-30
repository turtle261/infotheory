//! Disk/document I/O policy for top-level spec documents.

use super::{DOCUMENT_MAGIC, SpecDocument, SpecError, SpecResult};
use std::path::Path;

/// Load a spec document from disk.
///
/// Parsing is format-directed:
/// - `.json` files are parsed as JSON documents only.
/// - `.itsd` files (or payloads with the `itsd` magic envelope) are decoded as
///   binary documents.
/// - Otherwise, JSON parsing is attempted and JSON errors are reported
///   directly without a binary fallthrough.
pub fn load_spec_document(path: &str) -> SpecResult<SpecDocument> {
    let full = Path::new(path);
    let base_dir = full.parent().unwrap_or_else(|| Path::new("."));
    let raw = std::fs::read(full)?;

    let extension = full.extension().and_then(|value| value.to_str());
    if extension.is_some_and(|value| value.eq_ignore_ascii_case("json")) {
        let value = serde_json::from_slice::<serde_json::Value>(&raw).map_err(|err| {
            SpecError::new(format!(
                "invalid JSON spec document '{}': {err}",
                full.display()
            ))
        })?;
        return SpecDocument::parse_json_value(&value, base_dir);
    }

    if extension.is_some_and(|value| value.eq_ignore_ascii_case("itsd"))
        || raw.starts_with(DOCUMENT_MAGIC)
    {
        return SpecDocument::from_binary(&raw, base_dir);
    }

    let value = serde_json::from_slice::<serde_json::Value>(&raw).map_err(|err| {
        SpecError::new(format!(
            "failed to parse spec document '{}': expected JSON or binary '{}' envelope; JSON parse error: {err}",
            full.display(),
            String::from_utf8_lossy(DOCUMENT_MAGIC)
        ))
    })?;
    SpecDocument::parse_json_value(&value, base_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "backend-ctw")]
    use crate::api::RateBackend;
    #[cfg(feature = "backend-ctw")]
    use crate::spec::CanonicalJson;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(prefix: &str, ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("infotheory-spec-io-{prefix}-{nanos}.{ext}"))
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn load_spec_document_detects_json_extension() {
        let path = temp_path("json", "json");
        let expected = SpecDocument::RateBackend(RateBackend::Ctw { depth: 8 });
        std::fs::write(&path, expected.to_canonical_json().expect("json"))
            .expect("write json spec");

        let parsed = load_spec_document(path.to_string_lossy().as_ref()).expect("load json spec");
        assert_eq!(
            parsed.to_canonical_json().expect("parsed json"),
            expected.to_canonical_json().expect("expected json")
        );

        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn load_spec_document_detects_binary_extension_and_magic() {
        let path = temp_path("binary", "itsd");
        let expected = SpecDocument::RateBackend(RateBackend::Ctw { depth: 12 });
        std::fs::write(&path, expected.to_binary()).expect("write binary spec");

        let parsed =
            load_spec_document(path.to_string_lossy().as_ref()).expect("load binary extension");
        assert_eq!(
            parsed.to_canonical_json().expect("parsed json"),
            expected.to_canonical_json().expect("expected json")
        );

        let magic_path = temp_path("magic", "bin");
        std::fs::write(&magic_path, expected.to_binary()).expect("write magic-detected binary");
        let magic_parsed =
            load_spec_document(magic_path.to_string_lossy().as_ref()).expect("load binary magic");
        assert_eq!(
            magic_parsed.to_canonical_json().expect("parsed json"),
            expected.to_canonical_json().expect("expected json")
        );

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(magic_path);
    }

    #[test]
    fn load_spec_document_reports_non_json_non_binary_payloads_directly() {
        let path = temp_path("invalid", "txt");
        std::fs::write(&path, b"not json and not binary").expect("write invalid payload");

        let err = match load_spec_document(path.to_string_lossy().as_ref()) {
            Ok(_) => panic!("invalid payload must fail"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(msg.contains("expected JSON or binary 'itsd' envelope"));
        assert!(msg.contains(path.to_string_lossy().as_ref()));

        let _ = std::fs::remove_file(path);
    }
}
