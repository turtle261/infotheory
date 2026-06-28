//! Shared public error types for fallible infotheory APIs.

use std::error::Error;
use std::fmt;

/// Result type used by fallible infotheory APIs.
pub type InfotheoryResult<T> = Result<T, InfotheoryError>;

/// Public error type for spec validation, backend construction, and runtime failures.
#[derive(Debug)]
pub enum InfotheoryError {
    /// Invalid or unsupported backend/spec configuration supplied by the caller.
    InvalidBackendConfig(String),
    /// Runtime execution failure while scoring, generating, or compressing.
    Runtime(String),
    /// Requested operation is not supported for the chosen backend.
    Unsupported(String),
    /// I/O failure surfaced through infotheory APIs.
    Io(std::io::Error),
    /// Shared spec/config parsing error.
    Spec(crate::spec::SpecError),
}

impl InfotheoryError {
    /// Build an invalid-backend/spec configuration error.
    pub fn invalid_backend_config(message: impl Into<String>) -> Self {
        Self::InvalidBackendConfig(message.into())
    }

    /// Build a runtime execution error.
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    /// Build an unsupported-operation error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }
}

impl fmt::Display for InfotheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBackendConfig(message) => {
                write!(f, "invalid backend configuration: {message}")
            }
            Self::Runtime(message) => write!(f, "runtime failure: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported operation: {message}"),
            Self::Io(err) => write!(f, "i/o failure: {err}"),
            Self::Spec(err) => write!(f, "spec error: {err}"),
        }
    }
}

impl Error for InfotheoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Spec(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InfotheoryError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<crate::spec::SpecError> for InfotheoryError {
    fn from(value: crate::spec::SpecError) -> Self {
        Self::Spec(value)
    }
}

impl From<String> for InfotheoryError {
    fn from(value: String) -> Self {
        Self::Runtime(value)
    }
}

impl From<&str> for InfotheoryError {
    fn from(value: &str) -> Self {
        Self::Runtime(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_and_display_messages_are_stable() {
        assert_eq!(
            InfotheoryError::invalid_backend_config("bad depth").to_string(),
            "invalid backend configuration: bad depth"
        );
        assert_eq!(
            InfotheoryError::runtime("decoder stalled").to_string(),
            "runtime failure: decoder stalled"
        );
        assert_eq!(
            InfotheoryError::unsupported("requires vm").to_string(),
            "unsupported operation: requires vm"
        );
    }

    #[test]
    fn source_and_from_conversions_preserve_error_context() {
        let io = std::io::Error::other("disk broke");
        let io_error = InfotheoryError::from(io);
        assert!(io_error.to_string().contains("i/o failure: disk broke"));
        assert!(io_error.source().is_some());

        let spec = crate::spec::SpecError::new("bad spec");
        let spec_error = InfotheoryError::from(spec.clone());
        assert_eq!(spec_error.to_string(), "spec error: bad spec");
        assert_eq!(
            spec_error
                .source()
                .expect("spec error should retain source")
                .to_string(),
            spec.to_string()
        );

        let runtime_from_string = InfotheoryError::from("runtime text");
        assert_eq!(
            runtime_from_string.to_string(),
            "runtime failure: runtime text"
        );

        let runtime_from_owned = InfotheoryError::from(String::from("owned runtime"));
        assert_eq!(
            runtime_from_owned.to_string(),
            "runtime failure: owned runtime"
        );
    }
}
