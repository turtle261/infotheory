//! Internal runtime builders and backend registry metadata.
//!
//! This module is the spec -> runtime boundary for predictor and compression
//! execution paths.

use crate::api::{
    CompressionBackend, RateBackend, validate_compression_backend, validate_rate_backend,
};
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree};
#[cfg(feature = "backend-particle")]
use crate::backends::particle::ParticleRuntime;
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
use crate::error::{InfotheoryError, InfotheoryResult};

/// Stable internal identity for each rate-backend family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RateBackendKind {
    RosaPlus,
    Ctw,
    FacCtw,
    Match,
    SparseMatch,
    Ppmd,
    Sequitur,
    Calibrated,
    Zpaq,
    Mixture,
    Particle,
    Mamba,
    Rwkv7,
}

/// Stable internal identity for each compression-backend family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CompressionBackendKind {
    Zpaq,
    Rwkv7,
    RateAc,
    RateRans,
}

/// Stable identity for method-backed neural families shared by VM glue.
#[cfg(feature = "vm")]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MethodBackendKind {
    Mamba,
    Rwkv7,
}

/// Shared trace-model execution strategy used by VM glue.
#[cfg(feature = "vm")]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TraceModelStrategy {
    Rosa,
    Ctw,
    FacCtw,
    PredictorBacked,
    Zpaq,
    Mamba,
    Rwkv7,
}

/// Canonical backend metadata entry shared by parsers and runtime builders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendDescriptor<K> {
    /// Stable internal backend identity.
    pub kind: K,
    /// Canonical backend name.
    pub canonical: &'static str,
    /// Accepted aliases for the backend, including the canonical spelling.
    pub aliases: &'static [&'static str],
    /// Required Cargo feature for availability, when the backend is feature-gated.
    pub feature: Option<&'static str>,
    /// Whether the backend is enabled in the current build.
    pub enabled: bool,
}

/// Canonical metadata for rate backends.
pub type RateBackendDescriptor = BackendDescriptor<RateBackendKind>;
/// Canonical metadata for compression backends.
pub type CompressionBackendDescriptor = BackendDescriptor<CompressionBackendKind>;

/// Registry of rate-backend metadata.
pub const RATE_BACKEND_REGISTRY: &[RateBackendDescriptor] = &[
    BackendDescriptor {
        kind: RateBackendKind::RosaPlus,
        canonical: "rosaplus",
        aliases: &["rosaplus", "rosa"],
        feature: Some("backend-rosa"),
        enabled: cfg!(feature = "backend-rosa"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Ctw,
        canonical: "ctw",
        aliases: &["ctw"],
        feature: Some("backend-ctw"),
        enabled: cfg!(feature = "backend-ctw"),
    },
    BackendDescriptor {
        kind: RateBackendKind::FacCtw,
        canonical: "fac-ctw",
        aliases: &["fac-ctw", "facctw"],
        feature: Some("backend-ctw"),
        enabled: cfg!(feature = "backend-ctw"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Match,
        canonical: "match",
        aliases: &["match"],
        feature: Some("backend-match"),
        enabled: cfg!(feature = "backend-match"),
    },
    BackendDescriptor {
        kind: RateBackendKind::SparseMatch,
        canonical: "sparse-match",
        aliases: &["sparse-match", "sparse_match", "sparsematch"],
        feature: Some("backend-match"),
        enabled: cfg!(feature = "backend-match"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Ppmd,
        canonical: "ppmd",
        aliases: &["ppmd", "ppm"],
        feature: Some("backend-ppmd"),
        enabled: cfg!(feature = "backend-ppmd"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Sequitur,
        canonical: "sequitur",
        aliases: &["sequitur"],
        feature: Some("backend-sequitur"),
        enabled: cfg!(feature = "backend-sequitur"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Calibrated,
        canonical: "calibrated",
        aliases: &["calibrated", "cal"],
        feature: Some("backend-calibrated"),
        enabled: cfg!(feature = "backend-calibrated"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Zpaq,
        canonical: "zpaq",
        aliases: &["zpaq"],
        feature: Some("backend-zpaq"),
        enabled: cfg!(feature = "backend-zpaq"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Mixture,
        canonical: "mixture",
        aliases: &["mixture", "mix"],
        feature: Some("backend-mixture"),
        enabled: cfg!(feature = "backend-mixture"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Particle,
        canonical: "particle",
        aliases: &["particle", "particles"],
        feature: Some("backend-particle"),
        enabled: cfg!(feature = "backend-particle"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Mamba,
        canonical: "mamba",
        aliases: &["mamba", "mamba1"],
        feature: Some("backend-mamba"),
        enabled: cfg!(feature = "backend-mamba"),
    },
    BackendDescriptor {
        kind: RateBackendKind::Rwkv7,
        canonical: "rwkv7",
        aliases: &["rwkv7", "rwkv"],
        feature: Some("backend-rwkv"),
        enabled: cfg!(feature = "backend-rwkv"),
    },
];

/// Registry of compression-backend metadata.
pub const COMPRESSION_BACKEND_REGISTRY: &[CompressionBackendDescriptor] = &[
    BackendDescriptor {
        kind: CompressionBackendKind::Zpaq,
        canonical: "zpaq",
        aliases: &["zpaq"],
        feature: Some("backend-zpaq"),
        enabled: cfg!(feature = "backend-zpaq"),
    },
    BackendDescriptor {
        kind: CompressionBackendKind::Rwkv7,
        canonical: "rwkv7",
        aliases: &["rwkv7", "rwkv"],
        feature: Some("backend-rwkv"),
        enabled: cfg!(feature = "backend-rwkv"),
    },
    BackendDescriptor {
        kind: CompressionBackendKind::RateAc,
        canonical: "rate-ac",
        aliases: &["rate-ac", "rate_ac", "rateac"],
        feature: None,
        enabled: true,
    },
    BackendDescriptor {
        kind: CompressionBackendKind::RateRans,
        canonical: "rate-rans",
        aliases: &["rate-rans", "rate_rans", "raterans"],
        feature: None,
        enabled: true,
    },
];

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

fn backend_descriptor_by_kind_checked<K: Copy + Eq>(
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

#[cfg(feature = "vm")]
#[allow(dead_code)]
pub(crate) fn rate_backend_method(
    backend: &RateBackend,
    family: MethodBackendKind,
) -> Option<&str> {
    match (family, backend) {
        #[cfg(feature = "backend-rwkv")]
        (MethodBackendKind::Rwkv7, RateBackend::Rwkv7Method { method }) => Some(method.as_str()),
        #[cfg(feature = "backend-mamba")]
        (MethodBackendKind::Mamba, RateBackend::MambaMethod { method }) => Some(method.as_str()),
        _ => None,
    }
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
pub(crate) fn rate_backend_trace_model_strategy(backend: &RateBackend) -> TraceModelStrategy {
    match backend.kind() {
        RateBackendKind::RosaPlus => TraceModelStrategy::Rosa,
        RateBackendKind::Ctw => TraceModelStrategy::Ctw,
        RateBackendKind::FacCtw => TraceModelStrategy::FacCtw,
        RateBackendKind::Match
        | RateBackendKind::SparseMatch
        | RateBackendKind::Ppmd
        | RateBackendKind::Sequitur
        | RateBackendKind::Calibrated
        | RateBackendKind::Mixture
        | RateBackendKind::Particle => TraceModelStrategy::PredictorBacked,
        RateBackendKind::Zpaq => TraceModelStrategy::Zpaq,
        RateBackendKind::Mamba => TraceModelStrategy::Mamba,
        RateBackendKind::Rwkv7 => TraceModelStrategy::Rwkv7,
    }
}

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

pub(crate) fn try_describe_rate_backend(
    backend: &RateBackend,
) -> Result<&'static RateBackendDescriptor, String> {
    describe_rate_backend_kind(backend.kind())
}

pub(crate) fn try_describe_compression_backend(
    backend: &CompressionBackend,
) -> Result<&'static CompressionBackendDescriptor, String> {
    describe_compression_backend_kind(backend.kind())
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

fn interleave_aligned_bytes(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut joint = Vec::with_capacity(x.len() * 2);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        joint.push(xb);
        joint.push(yb);
    }
    joint
}

fn execute_entropy_rate_backend(
    data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    Ok(match backend {
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus => {
            let mut model = RosaPlus::new(max_order, false, 0, 42);
            model.predictive_entropy_rate(data)
        }
        #[cfg(not(feature = "backend-rosa"))]
        RateBackend::RosaPlus => unreachable!("validation should reject disabled rosaplus"),
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Sequitur { .. }
        | RateBackend::Calibrated { .. } => {
            crate::try_prequential_rate_backend(data, &[], max_order, backend)?
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => crate::with_rwkv_method_tls(method, |c| {
            c.cross_entropy(data).map_err(|err| {
                InfotheoryError::runtime(format!("rwkv method entropy scoring failed: {err:#}"))
            })
        })?,
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => crate::with_mamba_method_tls(method, |c| {
            c.cross_entropy(data).map_err(|err| {
                InfotheoryError::runtime(format!("mamba method entropy scoring failed: {err:#}"))
            })
        })?,
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(data);
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => unreachable!("validation should reject disabled zpaq"),
        #[cfg(feature = "backend-mixture")]
        RateBackend::Mixture { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let experts = spec.build_experts();
            let mut mix =
                crate::mixture::build_mixture_runtime(spec.as_ref(), &experts).map_err(|err| {
                    InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {err}"))
                })?;
            mix.begin_stream(Some(data.len() as u64)).map_err(|err| {
                InfotheoryError::runtime(format!("Mixture stream init failed: {err}"))
            })?;
            let mut bits = 0.0;
            for &byte in data {
                bits -= mix.step(byte) / std::f64::consts::LN_2;
            }
            mix.finish_stream().map_err(|err| {
                InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
            })?;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-mixture"))]
        RateBackend::Mixture { .. } => unreachable!("validation should reject disabled mixture"),
        #[cfg(feature = "backend-particle")]
        RateBackend::Particle { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut runtime = ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &byte in data {
                bits -= runtime.step(byte) / std::f64::consts::LN_2;
            }
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-particle"))]
        RateBackend::Particle { .. } => unreachable!("validation should reject disabled particle"),
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut fac = FacContextTree::new(*depth, 8);
            fac.reserve_for_symbols(data.len());
            for &byte in data {
                fac.update_byte_msb(byte);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::Ctw { .. } => unreachable!("validation should reject disabled ctw"),
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = FacContextTree::new(*base_depth, bits_per_byte);
            fac.reserve_for_symbols(data.len());
            for &byte in data {
                fac.update_byte_lsb(byte);
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (data.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::FacCtw { .. } => unreachable!("validation should reject disabled fac-ctw"),
    })
}

pub(crate) fn try_entropy_rate_backend_direct(
    data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    execute_entropy_rate_backend(data, max_order, backend)
}

pub(crate) fn try_cross_entropy_rate_backend_direct(
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    #[cfg(feature = "backend-zpaq")]
    if let RateBackend::Zpaq { method } = backend {
        if test_data.is_empty() {
            return Ok(0.0);
        }
        let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
        model.update_and_score(train_data);
        let bits = model.update_and_score(test_data);
        return Ok(bits / (test_data.len() as f64));
    }
    crate::try_frozen_plugin_rate_backend(test_data, &[train_data], max_order, backend)
}

pub(crate) fn try_joint_entropy_rate_backend_direct(
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let n = x.len().min(y.len());
    let x = &x[..n];
    let y = &y[..n];

    Ok(match backend {
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus => {
            let joint_symbols: Vec<u32> = (0..x.len())
                .map(|idx| (x[idx] as u32) * 256 + (y[idx] as u32))
                .collect();
            let mut model = RosaPlus::new(max_order, false, 0, 42);
            model.entropy_rate_cps(&joint_symbols)
        }
        #[cfg(not(feature = "backend-rosa"))]
        RateBackend::RosaPlus => unreachable!("validation should reject disabled rosaplus"),
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Sequitur { .. }
        | RateBackend::Calibrated { .. } => {
            let joint = interleave_aligned_bytes(x, y);
            execute_entropy_rate_backend(&joint, max_order, backend)? * 2.0
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => crate::with_rwkv_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
                InfotheoryError::runtime(format!(
                    "rwkv method joint-entropy scoring failed: {err:#}"
                ))
            })
        })?,
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => crate::with_mamba_method_tls(method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
                InfotheoryError::runtime(format!(
                    "mamba method joint-entropy scoring failed: {err:#}"
                ))
            })
        })?,
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            let joint = interleave_aligned_bytes(x, y);
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            let bits = model.update_and_score(&joint);
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => unreachable!("validation should reject disabled zpaq"),
        #[cfg(feature = "backend-mixture")]
        RateBackend::Mixture { spec } => {
            let joint = interleave_aligned_bytes(x, y);
            let experts = spec.build_experts();
            let mut mix =
                crate::mixture::build_mixture_runtime(spec.as_ref(), &experts).map_err(|err| {
                    InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {err}"))
                })?;
            mix.begin_stream(Some(joint.len() as u64)).map_err(|err| {
                InfotheoryError::runtime(format!("Mixture stream init failed: {err}"))
            })?;
            let mut bits = 0.0;
            for &byte in &joint {
                bits -= mix.step(byte) / std::f64::consts::LN_2;
            }
            mix.finish_stream().map_err(|err| {
                InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
            })?;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-mixture"))]
        RateBackend::Mixture { .. } => unreachable!("validation should reject disabled mixture"),
        #[cfg(feature = "backend-particle")]
        RateBackend::Particle { spec } => {
            let joint = interleave_aligned_bytes(x, y);
            let mut runtime = ParticleRuntime::new(spec.as_ref());
            let mut bits = 0.0;
            for &byte in &joint {
                bits -= runtime.step(byte) / std::f64::consts::LN_2;
            }
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-particle"))]
        RateBackend::Particle { .. } => unreachable!("validation should reject disabled particle"),
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => {
            let mut fac = FacContextTree::new(*depth, 16);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                for bit_idx in 0..8 {
                    fac.update(((xb >> (7 - bit_idx)) & 1) == 1, bit_idx);
                    fac.update(((yb >> (7 - bit_idx)) & 1) == 1, bit_idx + 8);
                }
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::Ctw { .. } => unreachable!("validation should reject disabled ctw"),
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = FacContextTree::new(*base_depth, bits_per_byte * 2);
            for (&xb, &yb) in x.iter().zip(y.iter()) {
                for idx in 0..bits_per_byte {
                    let bit_idx_x = idx * 2;
                    let bit_idx_y = bit_idx_x + 1;
                    fac.update(((xb >> idx) & 1) == 1, bit_idx_x);
                    fac.update(((yb >> idx) & 1) == 1, bit_idx_y);
                }
            }
            let ln_p = fac.get_log_block_probability();
            let bits = -ln_p / std::f64::consts::LN_2;
            bits / (x.len() as f64)
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::FacCtw { .. } => unreachable!("validation should reject disabled fac-ctw"),
    })
}

pub(crate) fn try_cross_entropy_conditional_chain_backend(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &RateBackend,
) -> InfotheoryResult<f64> {
    validate_rate_backend(backend)?;
    match backend {
        #[cfg(feature = "backend-rosa")]
        RateBackend::RosaPlus => {
            crate::try_frozen_plugin_rate_backend(data, prefix_parts, -1, &RateBackend::RosaPlus)
        }
        #[cfg(not(feature = "backend-rosa"))]
        RateBackend::RosaPlus => unreachable!("validation should reject disabled rosaplus"),
        RateBackend::Match { .. }
        | RateBackend::SparseMatch { .. }
        | RateBackend::Ppmd { .. }
        | RateBackend::Sequitur { .. }
        | RateBackend::Calibrated { .. } => {
            crate::try_prequential_rate_backend(data, prefix_parts, -1, backend)
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => crate::with_rwkv_method_tls(method, |c| {
            c.cross_entropy_conditional_chain(prefix_parts, data)
                .map_err(|err| {
                    InfotheoryError::runtime(format!(
                        "rwkv method conditional-chain scoring failed: {err:#}"
                    ))
                })
        }),
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => crate::with_mamba_method_tls(method, |c| {
            c.cross_entropy_conditional_chain(prefix_parts, data)
                .map_err(|err| {
                    InfotheoryError::runtime(format!(
                        "mamba method conditional-chain scoring failed: {err:#}"
                    ))
                })
        }),
        #[cfg(feature = "backend-ctw")]
        RateBackend::Ctw { depth } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut tree = ContextTree::new(*depth);
            for &part in prefix_parts {
                for &byte in part {
                    for idx in (0..8).rev() {
                        tree.update(((byte >> idx) & 1) == 1);
                    }
                }
            }
            let log_p_prefix = tree.get_log_block_probability();
            for &byte in data {
                for idx in (0..8).rev() {
                    tree.update(((byte >> idx) & 1) == 1);
                }
            }
            let log_p_joint = tree.get_log_block_probability();
            let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
            Ok(bits / (data.len() as f64))
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::Ctw { .. } => unreachable!("validation should reject disabled ctw"),
        #[cfg(feature = "backend-zpaq")]
        RateBackend::Zpaq { method } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut model = ZpaqRateModel::new(method.clone(), 2f64.powi(-24));
            for &part in prefix_parts {
                model.update_and_score(part);
            }
            let bits = model.update_and_score(data);
            Ok(bits / (data.len() as f64))
        }
        #[cfg(not(feature = "backend-zpaq"))]
        RateBackend::Zpaq { .. } => unreachable!("validation should reject disabled zpaq"),
        #[cfg(feature = "backend-mixture")]
        RateBackend::Mixture { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let experts = spec.build_experts();
            let mut mix =
                crate::mixture::build_mixture_runtime(spec.as_ref(), &experts).map_err(|err| {
                    InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {err}"))
                })?;
            let total = prefix_parts
                .iter()
                .map(|part| part.len() as u64)
                .sum::<u64>()
                .saturating_add(data.len() as u64);
            mix.begin_stream(Some(total)).map_err(|err| {
                InfotheoryError::runtime(format!("Mixture stream init failed: {err}"))
            })?;
            for &part in prefix_parts {
                for &byte in part {
                    mix.step(byte);
                }
            }
            let mut bits = 0.0;
            for &byte in data {
                bits -= mix.step(byte) / std::f64::consts::LN_2;
            }
            Ok(bits / (data.len() as f64))
        }
        #[cfg(not(feature = "backend-mixture"))]
        RateBackend::Mixture { .. } => unreachable!("validation should reject disabled mixture"),
        #[cfg(feature = "backend-particle")]
        RateBackend::Particle { spec } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let mut runtime = ParticleRuntime::new(spec.as_ref());
            for &part in prefix_parts {
                for &byte in part {
                    runtime.step(byte);
                }
            }
            let mut bits = 0.0;
            for &byte in data {
                bits -= runtime.step(byte) / std::f64::consts::LN_2;
            }
            Ok(bits / (data.len() as f64))
        }
        #[cfg(not(feature = "backend-particle"))]
        RateBackend::Particle { .. } => unreachable!("validation should reject disabled particle"),
        #[cfg(feature = "backend-ctw")]
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
        } => {
            if data.is_empty() {
                return Ok(0.0);
            }
            let bits_per_byte = (*encoding_bits).clamp(1, 8);
            let mut fac = FacContextTree::new(*base_depth, bits_per_byte);
            for &part in prefix_parts {
                for &byte in part {
                    for idx in 0..bits_per_byte {
                        fac.update(((byte >> idx) & 1) == 1, idx);
                    }
                }
            }
            let log_p_prefix = fac.get_log_block_probability();
            for &byte in data {
                for idx in 0..bits_per_byte {
                    fac.update(((byte >> idx) & 1) == 1, idx);
                }
            }
            let log_p_joint = fac.get_log_block_probability();
            let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
            Ok(bits / (data.len() as f64))
        }
        #[cfg(not(feature = "backend-ctw"))]
        RateBackend::FacCtw { .. } => unreachable!("validation should reject disabled fac-ctw"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn assert_registry_is_injective<K>(registry: &[BackendDescriptor<K>], label: &str)
    where
        K: Copy + Eq + std::fmt::Debug + std::hash::Hash,
    {
        let mut kinds = HashSet::new();
        let mut aliases = HashSet::new();

        for descriptor in registry {
            assert!(
                kinds.insert(descriptor.kind),
                "{label} duplicates backend kind {:?}",
                descriptor.kind
            );
            assert!(
                descriptor.aliases.contains(&descriptor.canonical),
                "{label} descriptor '{}' must list its canonical alias",
                descriptor.canonical
            );
            for alias in descriptor.aliases {
                assert!(
                    aliases.insert(*alias),
                    "{label} alias '{alias}' is assigned to multiple backends"
                );
            }
        }
    }

    #[test]
    fn rate_backend_registry_resolves_aliases() {
        let rosa = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "rosa")
            .expect("rosa descriptor");
        assert_eq!(rosa.canonical, "rosaplus");

        let mix = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "mix")
            .expect("mixture descriptor");
        assert_eq!(mix.canonical, "mixture");
    }

    #[test]
    fn compression_registry_resolves_aliases() {
        let ac = find_backend_descriptor_in_registry(COMPRESSION_BACKEND_REGISTRY, "rate_ac")
            .expect("rate-ac descriptor");
        assert_eq!(ac.canonical, "rate-ac");
    }

    #[test]
    fn rate_backend_registry_has_unique_kinds_and_aliases() {
        assert_registry_is_injective(RATE_BACKEND_REGISTRY, "RATE_BACKEND_REGISTRY");
    }

    #[test]
    fn compression_backend_registry_has_unique_kinds_and_aliases() {
        assert_registry_is_injective(COMPRESSION_BACKEND_REGISTRY, "COMPRESSION_BACKEND_REGISTRY");
    }

    #[test]
    fn describe_compression_backend_uses_canonical_lookup_not_positional_indices() {
        let ac = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus,
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-ac");
        assert_eq!(ac.canonical, "rate-ac");

        let rans = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus,
            coder: crate::coders::CoderType::RANS,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-rans");
        assert_eq!(rans.canonical, "rate-rans");
    }

    #[test]
    fn missing_descriptor_reports_registry_mismatch_error() {
        let err = backend_descriptor_by_kind_checked(
            &[],
            RateBackendKind::RosaPlus,
            "RATE_BACKEND_REGISTRY",
        )
        .expect_err("missing descriptor should return an error");
        assert!(err.contains("internal backend registry mismatch"));
        assert!(err.contains("RosaPlus"));
    }
}
