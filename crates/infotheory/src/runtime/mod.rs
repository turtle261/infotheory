//! Internal runtime builders and backend registry metadata.
//!
//! This module is the spec -> runtime boundary for predictor and compression
//! execution paths.

use crate::api::{CompressionBackend, RateBackend, validate_compression_backend};
use crate::error::{InfotheoryError, InfotheoryResult};

/// Canonical backend metadata entry shared by parsers and runtime builders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendDescriptor {
    /// Canonical backend name.
    pub canonical: &'static str,
    /// Accepted aliases for the backend, including the canonical spelling.
    pub aliases: &'static [&'static str],
    /// Required Cargo feature for availability, when the backend is feature-gated.
    pub feature: Option<&'static str>,
    /// Whether the backend is enabled in the current build.
    pub enabled: bool,
}

/// Registry of rate-backend metadata.
pub const RATE_BACKEND_REGISTRY: &[BackendDescriptor] = &[
    BackendDescriptor {
        canonical: "rosaplus",
        aliases: &["rosaplus", "rosa"],
        feature: Some("backend-rosa"),
        enabled: cfg!(feature = "backend-rosa"),
    },
    BackendDescriptor {
        canonical: "ctw",
        aliases: &["ctw"],
        feature: Some("backend-ctw"),
        enabled: cfg!(feature = "backend-ctw"),
    },
    BackendDescriptor {
        canonical: "fac-ctw",
        aliases: &["fac-ctw", "facctw"],
        feature: Some("backend-ctw"),
        enabled: cfg!(feature = "backend-ctw"),
    },
    BackendDescriptor {
        canonical: "match",
        aliases: &["match"],
        feature: Some("backend-match"),
        enabled: cfg!(feature = "backend-match"),
    },
    BackendDescriptor {
        canonical: "sparse-match",
        aliases: &["sparse-match", "sparse_match", "sparsematch"],
        feature: Some("backend-match"),
        enabled: cfg!(feature = "backend-match"),
    },
    BackendDescriptor {
        canonical: "ppmd",
        aliases: &["ppmd", "ppm"],
        feature: Some("backend-ppmd"),
        enabled: cfg!(feature = "backend-ppmd"),
    },
    BackendDescriptor {
        canonical: "sequitur",
        aliases: &["sequitur"],
        feature: Some("backend-sequitur"),
        enabled: cfg!(feature = "backend-sequitur"),
    },
    BackendDescriptor {
        canonical: "calibrated",
        aliases: &["calibrated", "cal"],
        feature: Some("backend-calibrated"),
        enabled: cfg!(feature = "backend-calibrated"),
    },
    BackendDescriptor {
        canonical: "zpaq",
        aliases: &["zpaq"],
        feature: Some("backend-zpaq"),
        enabled: cfg!(feature = "backend-zpaq"),
    },
    BackendDescriptor {
        canonical: "mixture",
        aliases: &["mixture", "mix"],
        feature: Some("backend-mixture"),
        enabled: cfg!(feature = "backend-mixture"),
    },
    BackendDescriptor {
        canonical: "particle",
        aliases: &["particle", "particles"],
        feature: Some("backend-particle"),
        enabled: cfg!(feature = "backend-particle"),
    },
    BackendDescriptor {
        canonical: "mamba",
        aliases: &["mamba", "mamba1"],
        feature: Some("backend-mamba"),
        enabled: cfg!(feature = "backend-mamba"),
    },
    BackendDescriptor {
        canonical: "rwkv7",
        aliases: &["rwkv7", "rwkv"],
        feature: Some("backend-rwkv"),
        enabled: cfg!(feature = "backend-rwkv"),
    },
];

/// Registry of compression-backend metadata.
pub const COMPRESSION_BACKEND_REGISTRY: &[BackendDescriptor] = &[
    BackendDescriptor {
        canonical: "zpaq",
        aliases: &["zpaq"],
        feature: Some("backend-zpaq"),
        enabled: cfg!(feature = "backend-zpaq"),
    },
    BackendDescriptor {
        canonical: "rwkv7",
        aliases: &["rwkv7", "rwkv"],
        feature: Some("backend-rwkv"),
        enabled: cfg!(feature = "backend-rwkv"),
    },
    BackendDescriptor {
        canonical: "rate-ac",
        aliases: &["rate-ac", "rate_ac", "rateac"],
        feature: None,
        enabled: true,
    },
    BackendDescriptor {
        canonical: "rate-rans",
        aliases: &["rate-rans", "rate_rans", "raterans"],
        feature: None,
        enabled: true,
    },
];

/// Shared byte-level runtime predictor trait.
pub trait BytePredictor: crate::mixture::OnlineBytePredictor {}

impl<T> BytePredictor for T where T: crate::mixture::OnlineBytePredictor + ?Sized {}

/// Runtime predictors that support checkpoint/rollback.
#[allow(dead_code)]
pub trait CheckpointablePredictor {
    /// Concrete checkpoint type.
    type Checkpoint: Clone;

    /// Snapshot the current runtime state.
    fn checkpoint(&mut self) -> Self::Checkpoint;

    /// Restore a previous snapshot.
    fn restore_checkpoint(&mut self, checkpoint: &Self::Checkpoint);
}

/// Shared runtime factory trait for byte-level predictors.
pub trait PredictorFactory {
    /// Predictor type produced by this factory.
    type Predictor: BytePredictor + CheckpointablePredictor;

    /// Build a predictor runtime from a spec object.
    fn build_predictor(&self, max_order: i64, min_prob: f64) -> Result<Self::Predictor, String>;
}

/// Shared runtime trait for compression-capable backends.
pub trait CompressionRuntime {
    /// Compressed size of a single byte slice.
    fn compress_size(&mut self, data: &[u8]) -> InfotheoryResult<u64>;

    /// Compressed size of chained slices encoded as one stream.
    fn compress_size_chain(&mut self, parts: &[&[u8]]) -> InfotheoryResult<u64>;

    /// Encode raw bytes.
    fn compress_bytes(&mut self, data: &[u8]) -> InfotheoryResult<Vec<u8>>;

    /// Decode previously encoded bytes.
    fn decompress_bytes(&mut self, input: &[u8]) -> InfotheoryResult<Vec<u8>>;
}

/// Shared runtime factory trait for compression backends.
pub trait CompressionFactory {
    /// Runtime type produced by this factory.
    type Runtime: CompressionRuntime;

    /// Build a compression runtime from a spec object.
    fn build_compression_runtime(&self) -> Result<Self::Runtime, String>;
}

impl CheckpointablePredictor for crate::mixture::RateBackendPredictor {
    type Checkpoint = crate::mixture::RateBackendPredictorCheckpoint;

    fn checkpoint(&mut self) -> Self::Checkpoint {
        crate::mixture::RateBackendPredictor::checkpoint(self)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self::Checkpoint) {
        crate::mixture::RateBackendPredictor::restore_checkpoint(self, checkpoint);
    }
}

impl PredictorFactory for RateBackend {
    type Predictor = crate::mixture::RateBackendPredictor;

    fn build_predictor(&self, max_order: i64, min_prob: f64) -> Result<Self::Predictor, String> {
        crate::mixture::RateBackendPredictor::try_from_backend(self.clone(), max_order, min_prob)
    }
}

struct SliceChainReader<'a> {
    parts: &'a [&'a [u8]],
    i: usize,
    off: usize,
}

impl<'a> SliceChainReader<'a> {
    fn new(parts: &'a [&'a [u8]]) -> Self {
        Self {
            parts,
            i: 0,
            off: 0,
        }
    }
}

impl<'a> std::io::Read for SliceChainReader<'a> {
    fn read(&mut self, mut buf: &mut [u8]) -> std::io::Result<usize> {
        let mut total = 0;
        if buf.is_empty() {
            return Ok(0);
        }
        while self.i < self.parts.len() {
            let p = self.parts[self.i];
            if self.off >= p.len() {
                self.i += 1;
                self.off = 0;
                continue;
            }
            let n = (p.len() - self.off).min(buf.len());
            buf[..n].copy_from_slice(&p[self.off..self.off + n]);
            self.off += n;
            total += n;
            let tmp = buf;
            buf = &mut tmp[n..];
            if buf.is_empty() {
                break;
            }
        }
        Ok(total)
    }
}

/// Concrete compression runtime handle built from a [`CompressionBackend`] spec.
pub enum CompressionRuntimeHandle {
    Zpaq {
        method: String,
    },
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        method: String,
        coder: crate::coders::CoderType,
    },
    Rate {
        rate_backend: RateBackend,
        coder: crate::coders::CoderType,
        framing: crate::compression::FramingMode,
    },
}

impl CompressionRuntime for CompressionRuntimeHandle {
    fn compress_size(&mut self, data: &[u8]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => {
                crate::try_zpaq_compress_size_bytes(data, method.as_str())
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 { method, coder } => {
                crate::with_rwkv_method_tls(method, |c| {
                    c.compress_size(data, *coder).map_err(|err| {
                        InfotheoryError::runtime(format!("rwkv7 compression failed: {err:#}"))
                    })
                })
            }
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_size(data, rate_backend, -1, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded compression failed: {err:#}"))
                }),
        }
    }

    fn compress_size_chain(&mut self, parts: &[&[u8]]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => {
                let reader = SliceChainReader::new(parts);
                crate::try_zpaq_compress_size_stream(reader, method.as_str())
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 { method, coder } => {
                crate::with_rwkv_method_tls(method, |c| {
                    c.compress_size_chain(parts, *coder).map_err(|err| {
                        InfotheoryError::runtime(format!("rwkv7 chain compression failed: {err:#}"))
                    })
                })
            }
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_size_chain(
                parts,
                rate_backend,
                -1,
                *coder,
                *framing,
            )
            .map_err(|err| {
                InfotheoryError::runtime(format!("rate-coded chain compression failed: {err:#}"))
            }),
        }
    }

    fn compress_bytes(&mut self, data: &[u8]) -> InfotheoryResult<Vec<u8>> {
        match self {
            CompressionRuntimeHandle::Zpaq { method } => crate::zpaq_compress_to_vec(data, method)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("zpaq byte compression failed: {err:#}"))
                }),
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 { method, coder } => {
                crate::with_rwkv_method_tls(method, |c| c.compress(data, *coder)).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 byte compression failed: {err:#}"))
                })
            }
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_bytes(data, rate_backend, -1, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded byte compression failed: {err:#}"))
                }),
        }
    }

    fn decompress_bytes(&mut self, input: &[u8]) -> InfotheoryResult<Vec<u8>> {
        match self {
            CompressionRuntimeHandle::Zpaq { .. } => {
                crate::zpaq_decompress_to_vec(input).map_err(|err| {
                    InfotheoryError::runtime(format!("zpaq decompression failed: {err:#}"))
                })
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 { method, .. } => {
                crate::with_rwkv_method_tls(method, |c| c.decompress(input)).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 decompression failed: {err:#}"))
                })
            }
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                crate::compression::decompress_rate_bytes(input, rate_backend, -1, *coder, *framing)
                    .map_err(|err| {
                        InfotheoryError::runtime(format!(
                            "rate-coded decompression failed: {err:#}"
                        ))
                    })
            }
        }
    }
}

impl CompressionFactory for CompressionBackend {
    type Runtime = CompressionRuntimeHandle;

    fn build_compression_runtime(&self) -> Result<Self::Runtime, String> {
        validate_compression_backend(self).map_err(|err| err.to_string())?;
        Ok(match self {
            CompressionBackend::Zpaq { method } => CompressionRuntimeHandle::Zpaq {
                method: method.clone(),
            },
            #[cfg(feature = "backend-rwkv")]
            CompressionBackend::Rwkv7 { method, coder } => CompressionRuntimeHandle::Rwkv7 {
                method: method.clone(),
                coder: *coder,
            },
            CompressionBackend::Rate {
                rate_backend,
                coder,
                framing,
            } => CompressionRuntimeHandle::Rate {
                rate_backend: rate_backend.clone(),
                coder: *coder,
                framing: *framing,
            },
        })
    }
}

/// Shared alias lookup for rate backends.
pub fn find_rate_backend_descriptor(input: &str) -> Option<&'static BackendDescriptor> {
    let key = input.trim().to_ascii_lowercase();
    RATE_BACKEND_REGISTRY
        .iter()
        .find(|descriptor| descriptor.aliases.iter().any(|alias| *alias == key))
}

/// Shared alias lookup for compression backends.
pub fn find_compression_backend_descriptor(input: &str) -> Option<&'static BackendDescriptor> {
    let key = input.trim().to_ascii_lowercase();
    COMPRESSION_BACKEND_REGISTRY
        .iter()
        .find(|descriptor| descriptor.aliases.iter().any(|alias| *alias == key))
}

/// Return registry metadata for a concrete rate-backend spec.
pub fn describe_rate_backend(backend: &RateBackend) -> &'static BackendDescriptor {
    match backend {
        RateBackend::RosaPlus => &RATE_BACKEND_REGISTRY[0],
        RateBackend::Ctw { .. } => &RATE_BACKEND_REGISTRY[1],
        RateBackend::FacCtw { .. } => &RATE_BACKEND_REGISTRY[2],
        RateBackend::Match { .. } => &RATE_BACKEND_REGISTRY[3],
        RateBackend::SparseMatch { .. } => &RATE_BACKEND_REGISTRY[4],
        RateBackend::Ppmd { .. } => &RATE_BACKEND_REGISTRY[5],
        RateBackend::Sequitur { .. } => &RATE_BACKEND_REGISTRY[6],
        RateBackend::Calibrated { .. } => &RATE_BACKEND_REGISTRY[7],
        RateBackend::Zpaq { .. } => &RATE_BACKEND_REGISTRY[8],
        RateBackend::Mixture { .. } => &RATE_BACKEND_REGISTRY[9],
        RateBackend::Particle { .. } => &RATE_BACKEND_REGISTRY[10],
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { .. } => &RATE_BACKEND_REGISTRY[11],
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { .. } => &RATE_BACKEND_REGISTRY[12],
    }
}

/// Return registry metadata for a concrete compression-backend spec.
pub fn describe_compression_backend(backend: &CompressionBackend) -> &'static BackendDescriptor {
    match backend {
        CompressionBackend::Zpaq { .. } => &COMPRESSION_BACKEND_REGISTRY[0],
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { .. } => &COMPRESSION_BACKEND_REGISTRY[1],
        CompressionBackend::Rate { coder, .. } => match coder {
            crate::coders::CoderType::AC => &COMPRESSION_BACKEND_REGISTRY[2],
            crate::coders::CoderType::RANS => &COMPRESSION_BACKEND_REGISTRY[3],
        },
    }
}

/// Shared spec -> predictor runtime builder using the default probability floor.
pub(crate) fn build_rate_backend_predictor(
    backend: &RateBackend,
    max_order: i64,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    backend.build_predictor(max_order, min_prob)
}

/// Shared spec -> predictor runtime builder using the library's default probability floor.
pub(crate) fn build_rate_backend_predictor_default(
    backend: &RateBackend,
    max_order: i64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    build_rate_backend_predictor(backend, max_order, crate::mixture::DEFAULT_MIN_PROB)
}

/// Shared spec -> compression predictor runtime builder.
pub(crate) fn build_rate_pdf_predictor(
    backend: &RateBackend,
    max_order: i64,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    crate::compression::RatePdfPredictor::from_rate_backend(backend.clone(), max_order)
}

/// Shared spec -> compression runtime builder.
pub(crate) fn build_compression_runtime(
    backend: &CompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    backend.build_compression_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_backend_registry_resolves_aliases() {
        let rosa = find_rate_backend_descriptor("rosa").expect("rosa descriptor");
        assert_eq!(rosa.canonical, "rosaplus");

        let mix = find_rate_backend_descriptor("mix").expect("mixture descriptor");
        assert_eq!(mix.canonical, "mixture");
    }

    #[test]
    fn compression_registry_resolves_aliases() {
        let ac = find_compression_backend_descriptor("rate_ac").expect("rate-ac descriptor");
        assert_eq!(ac.canonical, "rate-ac");
    }
}
