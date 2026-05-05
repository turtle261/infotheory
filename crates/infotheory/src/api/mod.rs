//! Public, spec-first infotheory API surface.

pub(crate) mod compression;
pub(crate) mod context;
pub(crate) mod generation;
pub(crate) mod metrics;
pub(crate) mod paths;
pub(crate) mod types;

pub use self::context::{InfotheoryCtx, RateBackendSession, get_default_ctx, set_default_ctx};
pub use self::types::{
    CalibratedSpec, CalibrationContextKind, CompressionBackend, GenerationConfig,
    GenerationStrategy, GenerationUpdateMode, MAX_MIXTURE_NESTING, MixtureExpertSpec, MixtureKind,
    MixtureScheduleMode, MixtureSpec, ParticleSpec, RateBackend, ZpaqMethodSpec,
    parse_mixture_kind_name, parse_mixture_schedule_name, validate_compression_backend,
    validate_rate_backend,
};
pub use crate::spec::{
    AssetRef, CanonicalBytes, CanonicalJson, CompiledCompressionBackend, CompiledRateBackend,
    CompressionBackendCapabilities, MethodBackendFamily, RateBackendCapabilities,
    RateBackendTraceStrategy, SpecEnvironment, ValidatedCompressionBackend, ValidatedRateBackend,
};

pub use self::compression::{
    NcdVariant, try_compress_bytes_backend, try_compress_size_backend,
    try_compress_size_chain_backend, try_decompress_bytes_backend, try_ncd_bytes,
    try_ncd_bytes_backend, try_ncd_bytes_default, try_ncd_matrix_bytes,
    try_ncd_matrix_bytes_backend, try_ncd_matrix_bytes_default,
};
pub use self::generation::{
    try_generate_bytes, try_generate_bytes_conditional_chain,
    try_generate_bytes_conditional_chain_with_config, try_generate_bytes_with_config,
};
pub use self::metrics::{
    d_kl_bytes, empirical_cross_entropy_bytes, empirical_entropy_bytes,
    empirical_joint_entropy_bytes, empirical_mutual_information_bytes, empirical_ned_bytes,
    empirical_ned_cons_bytes, empirical_nte_bytes, empirical_resistance_to_transformation_bytes,
    js_div_bytes, nhd_bytes, try_biased_entropy_rate_backend, try_biased_entropy_rate_bytes,
    try_conditional_entropy_bytes, try_conditional_entropy_rate_bytes, try_cross_entropy_bytes,
    try_cross_entropy_rate_backend, try_cross_entropy_rate_bytes, try_entropy_rate_backend,
    try_entropy_rate_bytes, try_intrinsic_dependence_bytes, try_joint_entropy_rate_backend,
    try_joint_entropy_rate_bytes, try_mutual_information_bytes,
    try_mutual_information_rate_backend, try_mutual_information_rate_bytes, try_ned_bytes,
    try_ned_cons_bytes, try_ned_cons_rate_bytes, try_ned_rate_backend, try_ned_rate_bytes,
    try_nte_bytes, try_nte_rate_backend, try_nte_rate_bytes,
    try_resistance_to_transformation_bytes, tvd_bytes,
};
pub use self::paths::{
    try_conditional_entropy_paths, try_cross_entropy_paths, try_get_bytes_from_paths,
    try_get_compressed_size, try_get_compressed_size_parallel, try_get_compressed_sizes_from_paths,
    try_get_parallel_compressed_sizes_from_parallel_paths,
    try_get_parallel_compressed_sizes_from_sequential_paths,
    try_get_sequential_compressed_sizes_from_parallel_paths,
    try_get_sequential_compressed_sizes_from_sequential_paths, try_js_divergence_paths,
    try_kl_divergence_paths, try_mutual_information_paths, try_ncd_matrix_paths,
    try_ncd_matrix_paths_backend, try_ncd_paths, try_ncd_paths_backend,
    try_ncd_paths_compiled_backend, try_ned_paths, try_nhd_paths, try_nte_paths, try_tvd_paths,
};
