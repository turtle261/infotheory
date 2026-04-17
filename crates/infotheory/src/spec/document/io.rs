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
