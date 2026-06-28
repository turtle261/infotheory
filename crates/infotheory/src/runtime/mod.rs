//! Internal runtime builders and backend registry metadata.
//!
//! This module is the spec -> runtime boundary for predictor and compression
//! execution paths.

use self::plan_macros::expect_plan_ref;
use crate::api::{CompressionBackend, RateBackend};
#[cfg(feature = "backend-calibrated")]
use crate::backends::calibration::CalibratorCore;
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree, ctw_symbol_bit_msb};
#[cfg(feature = "backend-match")]
use crate::backends::match_model::MatchModel;
#[cfg(feature = "backend-particle")]
use crate::backends::particle::ParticleRuntime;
#[cfg(feature = "backend-ppmd")]
use crate::backends::ppmd::PpmdModel;
#[cfg(feature = "backend-rosa")]
use crate::backends::rosaplus::RosaPlus;
#[cfg(feature = "backend-sequitur")]
use crate::backends::sequitur::SequiturModel;
#[cfg(feature = "backend-match")]
use crate::backends::sparse_match::SparseMatchModel;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
use crate::error::{InfotheoryError, InfotheoryResult};
#[cfg(feature = "backend-mamba")]
use crate::mambazip;
#[cfg(feature = "backend-rwkv")]
use crate::rwkvzip;
use crate::spec::core::{
    CompressionBackendCapabilities, CompressionBackendPlan, MethodBackendFamily,
    RateBackendCapabilities, RateBackendPlan, RateBackendTraceStrategy as PublicTraceStrategy,
    SpecEnvironment,
};
use crate::spec::{CompiledCompressionBackend, CompiledRateBackend, SpecResult};
use std::sync::Arc;

mod pdf_predictor_builders;
mod plan_macros;
mod predictor_builders;
mod registry;

pub(crate) use registry::{
    describe_compression_backend_kind, describe_rate_backend_kind,
    find_backend_descriptor_in_registry,
};

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

/// Shared trace-model execution strategy used by VM glue.
#[cfg(feature = "vm")]
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

macro_rules! backend_feature_option {
    (none) => {
        None
    };
    ($feature:literal) => {
        Some($feature)
    };
}

macro_rules! backend_feature_enabled {
    (none) => {
        true
    };
    ($feature:literal) => {
        cfg!(feature = $feature)
    };
}

// Feature-gated kernel pointers deliberately distinguish "anchor" from "drop".
//
// The runtime catalog contains one row per known backend even when a backend's
// implementation feature is disabled, so disabled builds still report stable
// names, aliases, canonical errors, and capability metadata without retaining
// heavyweight model code. Use `drop` for kernels whose real implementation may
// be compiled out completely; the catalog stores only the fallback pointer in
// disabled builds. Use `anchor` only for helpers that must remain available and
// type-checked when the backend feature is off, such as canonical spec helpers
// and cheap capability predicates. An anchored function must therefore be
// feature-independent and must not pull backend runtime state into disabled
// binaries.
macro_rules! feature_gated_kernel_ptr_anchor {
    ($feature:literal, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        {
            #[cfg(feature = $feature)]
            { $func as $ptr_type }
            #[cfg(not(feature = $feature))]
            {
                #[allow(dead_code)]
                fn reachability_anchor() {
                    let _ = $func as $ptr_type;
                }
                fn fallback $args $(-> $ret)? $body
                fallback as $ptr_type
            }
        }
    };
    (none, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        $func as $ptr_type
    };
}

macro_rules! feature_gated_kernel_ptr_drop {
    ($feature:literal, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        {
            #[cfg(feature = $feature)]
            { $func as $ptr_type }
            #[cfg(not(feature = $feature))]
            {
                fn fallback $args $(-> $ret)? $body
                fallback as $ptr_type
            }
        }
    };
    (none, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        $func as $ptr_type
    };
}

macro_rules! feature_gated_kernel_ptr_mode {
    (anchor, $feature:tt, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        feature_gated_kernel_ptr_anchor!($feature, $ptr_type, $func, fn fallback $args $(-> $ret)? $body)
    };
    (drop, $feature:tt, $ptr_type:ident, $func:path, fn fallback $args:tt $(-> $ret:ty)? $body:block) => {
        feature_gated_kernel_ptr_drop!($feature, $ptr_type, $func, fn fallback $args $(-> $ret)? $body)
    };
}

/// Shared declarative catalog for rate-backend metadata and runtime wiring.
///
/// Adding a new rate backend should require:
/// 1. the backend-specific implementation code,
/// 2. one entry in this catalog.
///
/// The catalog emits both the user-facing registry metadata and the runtime
/// kernel dispatch table so the family list cannot silently drift.
macro_rules! define_rate_backend_catalog {
    ($(
        backend {
            kind: $kind:ident,
            canonical: $canonical:literal,
            aliases: [$($alias:literal),* $(,)?],
            feature: $feature:tt,
            spec_helper_refs: $spec_helper_refs:ident,
            metric_helper_refs: $metric_helper_refs:ident,
            compile_plan: $compile_plan:path,
            to_wrapper: $to_wrapper:path,
            encode_payload: $encode_payload:path,
            display_label: $display_label:path,
            default_name: $default_name:path,
            trace_strategy: $trace_strategy:expr,
            supports_biased_entropy: $supports_biased_entropy:expr,
            supports_frozen_conditioning: $supports_frozen_conditioning:expr,
            supports_rate_coded_compression: $supports_rate_coded_compression:expr,
            supports_native_bit_prediction: $supports_native_bit_prediction:path,
            supports_byte_prefix_mass: $supports_byte_prefix_mass:path,
            supports_efficient_byte_packed_bit_sessions: $supports_efficient_byte_packed_bit_sessions:path,
            supports_reversible_bit_updates: $supports_reversible_bit_updates:path,
            method_family: $method_family:expr,
            contains_zpaq: $contains_zpaq:path,
            build_predictor: $build_predictor:path,
            build_binary_token_predictor: $build_binary_token_predictor:path,
            build_pdf_predictor: $build_pdf_predictor:path,
            entropy_rate: $entropy_rate:path,
            joint_entropy_rate: $joint_entropy_rate:path,
            conditional_chain_rate: $conditional_chain_rate:path,
        }
    ),* $(,)?) => {
        /// Registry of rate-backend metadata.
        pub const RATE_BACKEND_REGISTRY: &[RateBackendDescriptor] = &[
            $(
                BackendDescriptor {
                    kind: RateBackendKind::$kind,
                    canonical: $canonical,
                    aliases: &[$($alias),*],
                    feature: backend_feature_option!($feature),
                    enabled: backend_feature_enabled!($feature),
                },
            )*
        ];

        pub(crate) const RATE_BACKEND_KERNELS: &[RateBackendKernel] = &[
            $(
                RateBackendKernel {
                    kind: RateBackendKind::$kind,
                    compile_plan: feature_gated_kernel_ptr_mode!(
                        $spec_helper_refs,
                        $feature, RatePlanCompiler, $compile_plan,
                        fn fallback(_b: &RateBackend, _e: &SpecEnvironment, _d: usize) -> SpecResult<RateBackendPlan> {
                            unreachable!("kernel should never compile without its feature")
                        }
                    ),
                    to_wrapper: feature_gated_kernel_ptr_mode!(
                        $spec_helper_refs,
                        $feature, RateWrapperBuilder, $to_wrapper,
                        fn fallback(_p: &RateBackendPlan) -> RateBackend {
                            unreachable!("kernel should never emit a wrapper without its feature")
                        }
                    ),
                    encode_payload: feature_gated_kernel_ptr_mode!(
                        $spec_helper_refs,
                        $feature, RatePayloadEncoder, $encode_payload,
                        fn fallback(_p: &RateBackendPlan, _o: &mut Vec<u8>) {
                            unreachable!("kernel should never encode payload without its feature")
                        }
                    ),
                    display_label: feature_gated_kernel_ptr_mode!(
                        $spec_helper_refs,
                        $feature, RateDisplayLabelFn, $display_label,
                        fn fallback(_p: &RateBackendPlan) -> String {
                            unreachable!("kernel should never format a label without its feature")
                        }
                    ),
                    default_name: feature_gated_kernel_ptr_mode!(
                        $spec_helper_refs,
                        $feature, RateDefaultNameFn, $default_name,
                        fn fallback(_p: &RateBackendPlan) -> String {
                            unreachable!("kernel should never format a default name without its feature")
                        }
                    ),
                    trace_strategy: $trace_strategy,
                    supports_biased_entropy: $supports_biased_entropy,
                    supports_frozen_conditioning: $supports_frozen_conditioning,
                    supports_rate_coded_compression: $supports_rate_coded_compression,
                    supports_native_bit_prediction: feature_gated_kernel_ptr_anchor!(
                        $feature, RateCapabilityFn, $supports_native_bit_prediction,
                        fn fallback(_p: &RateBackendPlan) -> bool {
                            false
                        }
                    ),
                    supports_byte_prefix_mass: feature_gated_kernel_ptr_anchor!(
                        $feature, RateCapabilityFn, $supports_byte_prefix_mass,
                        fn fallback(_p: &RateBackendPlan) -> bool {
                            false
                        }
                    ),
                    supports_efficient_byte_packed_bit_sessions: feature_gated_kernel_ptr_anchor!(
                        $feature, RateCapabilityFn, $supports_efficient_byte_packed_bit_sessions,
                        fn fallback(_p: &RateBackendPlan) -> bool {
                            false
                        }
                    ),
                    supports_reversible_bit_updates: feature_gated_kernel_ptr_anchor!(
                        $feature, RateCapabilityFn, $supports_reversible_bit_updates,
                        fn fallback(_p: &RateBackendPlan) -> bool {
                            false
                        }
                    ),
                    method_family: $method_family,
                    contains_zpaq: $contains_zpaq,
                    build_predictor: feature_gated_kernel_ptr_drop!(
                        $feature, RateBackendPredictorBuilder, $build_predictor,
                        fn fallback(_b: &CompiledRateBackend, _p: f64) -> Result<crate::mixture::RateBackendPredictor, String> {
                            Err(registry::rate_backend_feature_error(RateBackendKind::$kind))
                        }
                    ),
                    build_binary_token_predictor: feature_gated_kernel_ptr_drop!(
                        $feature, RateBackendBinaryTokenPredictorBuilder, $build_binary_token_predictor,
                        fn fallback(_b: &CompiledRateBackend, _p: f64) -> Result<crate::mixture::RateBackendPredictor, String> {
                            Err(registry::rate_backend_feature_error(RateBackendKind::$kind))
                        }
                    ),
                    build_pdf_predictor: feature_gated_kernel_ptr_drop!(
                        $feature, RatePdfPredictorBuilder, $build_pdf_predictor,
                        fn fallback(_b: &CompiledRateBackend) -> anyhow::Result<crate::compression::RatePdfPredictor> {
                            Err(anyhow::anyhow!(registry::rate_backend_feature_error(RateBackendKind::$kind)))
                        }
                    ),
                    entropy_rate: feature_gated_kernel_ptr_mode!(
                        $metric_helper_refs,
                        $feature, RateEntropyFn, $entropy_rate,
                        fn fallback(_d: &[u8], _b: &CompiledRateBackend) -> InfotheoryResult<f64> {
                            Err(InfotheoryError::unsupported(registry::rate_backend_feature_error(RateBackendKind::$kind)))
                        }
                    ),
                    joint_entropy_rate: feature_gated_kernel_ptr_mode!(
                        $metric_helper_refs,
                        $feature, RateJointEntropyFn, $joint_entropy_rate,
                        fn fallback(_x: &[u8], _y: &[u8], _b: &CompiledRateBackend) -> InfotheoryResult<f64> {
                            Err(InfotheoryError::unsupported(registry::rate_backend_feature_error(RateBackendKind::$kind)))
                        }
                    ),
                    conditional_chain_rate: feature_gated_kernel_ptr_mode!(
                        $metric_helper_refs,
                        $feature, RateConditionalChainFn, $conditional_chain_rate,
                        fn fallback(_p: &[&[u8]], _d: &[u8], _b: &CompiledRateBackend) -> InfotheoryResult<f64> {
                            Err(InfotheoryError::unsupported(registry::rate_backend_feature_error(RateBackendKind::$kind)))
                        }
                    ),
                },
            )*
        ];
    };
}

/// Shared declarative catalog for compression-backend metadata and runtime wiring.
macro_rules! define_compression_backend_catalog {
    ($(
        backend {
            kind: $kind:ident,
            canonical: $canonical:literal,
            aliases: [$($alias:literal),* $(,)?],
            feature: $feature:tt,
            helper_refs: $helper_refs:ident,
            compile_plan: $compile_plan:path,
            to_wrapper: $to_wrapper:path,
            encode_payload: $encode_payload:path,
            display_label: $display_label:path,
            uses_rate_backend: $uses_rate_backend:expr,
            supports_decompression: $supports_decompression:expr,
            build_runtime: $build_runtime:path,
        }
    ),* $(,)?) => {
        /// Registry of compression-backend metadata.
        pub const COMPRESSION_BACKEND_REGISTRY: &[CompressionBackendDescriptor] = &[
            $(
                BackendDescriptor {
                    kind: CompressionBackendKind::$kind,
                    canonical: $canonical,
                    aliases: &[$($alias),*],
                    feature: backend_feature_option!($feature),
                    enabled: backend_feature_enabled!($feature),
                },
            )*
        ];

        pub(crate) const COMPRESSION_BACKEND_KERNELS: &[CompressionBackendKernel] = &[
            $(
                CompressionBackendKernel {
                    kind: CompressionBackendKind::$kind,
                    compile_plan: feature_gated_kernel_ptr_mode!(
                        $helper_refs,
                        $feature, CompressionPlanCompiler, $compile_plan,
                        fn fallback(_b: &CompressionBackend, _e: &SpecEnvironment) -> SpecResult<CompressionBackendPlan> {
                            unreachable!("compression kernel should never compile without its feature")
                        }
                    ),
                    to_wrapper: feature_gated_kernel_ptr_mode!(
                        $helper_refs,
                        $feature, CompressionWrapperBuilder, $to_wrapper,
                        fn fallback(_p: &CompressionBackendPlan) -> CompressionBackend {
                            unreachable!("compression kernel should never emit a wrapper without its feature")
                        }
                    ),
                    encode_payload: feature_gated_kernel_ptr_mode!(
                        $helper_refs,
                        $feature, CompressionPayloadEncoder, $encode_payload,
                        fn fallback(_p: &CompressionBackendPlan, _o: &mut Vec<u8>) {
                            unreachable!("compression kernel should never encode payload without its feature")
                        }
                    ),
                    display_label: feature_gated_kernel_ptr_mode!(
                        $helper_refs,
                        $feature, CompressionDisplayLabelFn, $display_label,
                        fn fallback(_p: &CompressionBackendPlan) -> String {
                            unreachable!("compression kernel should never format a label without its feature")
                        }
                    ),
                    uses_rate_backend: $uses_rate_backend,
                    supports_decompression: $supports_decompression,
                    build_runtime: feature_gated_kernel_ptr_mode!(
                        $helper_refs,
                        $feature, CompressionRuntimeBuilder, $build_runtime,
                        fn fallback(_b: &CompiledCompressionBackend) -> Result<CompressionRuntimeHandle, String> {
                            Err(registry::compression_backend_feature_error(CompressionBackendKind::$kind))
                        }
                    ),
                },
            )*
        ];
    };
}

type RateBackendPredictorBuilder =
    fn(&CompiledRateBackend, f64) -> Result<crate::mixture::RateBackendPredictor, String>;
type RateBackendBinaryTokenPredictorBuilder =
    fn(&CompiledRateBackend, f64) -> Result<crate::mixture::RateBackendPredictor, String>;
type RatePdfPredictorBuilder =
    fn(&CompiledRateBackend) -> anyhow::Result<crate::compression::RatePdfPredictor>;
type RateEntropyFn = fn(&[u8], &CompiledRateBackend) -> InfotheoryResult<f64>;
type RateJointEntropyFn = fn(&[u8], &[u8], &CompiledRateBackend) -> InfotheoryResult<f64>;
type RateConditionalChainFn = fn(&[&[u8]], &[u8], &CompiledRateBackend) -> InfotheoryResult<f64>;
type RatePlanCompiler = fn(&RateBackend, &SpecEnvironment, usize) -> SpecResult<RateBackendPlan>;
type RateWrapperBuilder = fn(&RateBackendPlan) -> RateBackend;
type RatePayloadEncoder = fn(&RateBackendPlan, &mut Vec<u8>);
type RateDisplayLabelFn = fn(&RateBackendPlan) -> String;
type RateDefaultNameFn = fn(&RateBackendPlan) -> String;
type RateCapabilityFn = fn(&RateBackendPlan) -> bool;
type RateContainsZpaqFn = fn(&RateBackendPlan) -> bool;
type CompressionRuntimeBuilder =
    fn(&CompiledCompressionBackend) -> Result<CompressionRuntimeHandle, String>;
type CompressionPlanCompiler =
    fn(&CompressionBackend, &SpecEnvironment) -> SpecResult<CompressionBackendPlan>;
type CompressionWrapperBuilder = fn(&CompressionBackendPlan) -> CompressionBackend;
type CompressionPayloadEncoder = fn(&CompressionBackendPlan, &mut Vec<u8>);
type CompressionDisplayLabelFn = fn(&CompressionBackendPlan) -> String;

#[derive(Clone, Copy)]
pub(crate) struct RateBackendKernel {
    pub kind: RateBackendKind,
    pub compile_plan: RatePlanCompiler,
    pub to_wrapper: RateWrapperBuilder,
    pub encode_payload: RatePayloadEncoder,
    pub display_label: RateDisplayLabelFn,
    pub default_name: RateDefaultNameFn,
    pub trace_strategy: PublicTraceStrategy,
    pub supports_biased_entropy: bool,
    pub supports_frozen_conditioning: bool,
    pub supports_rate_coded_compression: bool,
    pub supports_native_bit_prediction: RateCapabilityFn,
    pub supports_byte_prefix_mass: RateCapabilityFn,
    pub supports_efficient_byte_packed_bit_sessions: RateCapabilityFn,
    pub supports_reversible_bit_updates: RateCapabilityFn,
    pub method_family: Option<MethodBackendFamily>,
    pub contains_zpaq: RateContainsZpaqFn,
    pub build_predictor: RateBackendPredictorBuilder,
    pub build_binary_token_predictor: RateBackendBinaryTokenPredictorBuilder,
    pub build_pdf_predictor: RatePdfPredictorBuilder,
    pub entropy_rate: RateEntropyFn,
    pub joint_entropy_rate: RateJointEntropyFn,
    pub conditional_chain_rate: RateConditionalChainFn,
}

#[derive(Clone, Copy)]
pub(crate) struct CompressionBackendKernel {
    pub kind: CompressionBackendKind,
    pub compile_plan: CompressionPlanCompiler,
    pub to_wrapper: CompressionWrapperBuilder,
    pub encode_payload: CompressionPayloadEncoder,
    pub display_label: CompressionDisplayLabelFn,
    pub uses_rate_backend: bool,
    pub supports_decompression: bool,
    pub build_runtime: CompressionRuntimeBuilder,
}

define_rate_backend_catalog! {
    backend {
        kind: RosaPlus,
        canonical: "rosaplus",
        aliases: ["rosaplus", "rosa"],
        feature: "backend-rosa",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_rosa,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_rosa,
        encode_payload: crate::spec::core::encode_rate_payload_rosa,
        display_label: crate::spec::core::rate_plan_display_label_rosa,
        default_name: crate::spec::core::rate_plan_default_name_rosa,
        trace_strategy: PublicTraceStrategy::Rosa,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_rosa,
        build_binary_token_predictor: predictor_builders::build_predictor_rosa,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_rosa,
        entropy_rate: entropy_rosa,
        joint_entropy_rate: joint_entropy_rosa,
        conditional_chain_rate: conditional_chain_rosa,
    },
    backend {
        kind: Match,
        canonical: "match",
        aliases: ["match"],
        feature: "backend-match",
        spec_helper_refs: anchor,
        metric_helper_refs: anchor,
        compile_plan: crate::spec::core::compile_rate_plan_match,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_match,
        encode_payload: crate::spec::core::encode_rate_payload_match,
        display_label: crate::spec::core::rate_plan_display_label_match,
        default_name: crate::spec::core::rate_plan_default_name_match,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_match,
        build_binary_token_predictor: predictor_builders::build_predictor_match,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_match,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: SparseMatch,
        canonical: "sparse-match",
        aliases: ["sparse-match"],
        feature: "backend-match",
        spec_helper_refs: anchor,
        metric_helper_refs: anchor,
        compile_plan: crate::spec::core::compile_rate_plan_sparse_match,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_sparse_match,
        encode_payload: crate::spec::core::encode_rate_payload_sparse_match,
        display_label: crate::spec::core::rate_plan_display_label_sparse_match,
        default_name: crate::spec::core::rate_plan_default_name_sparse_match,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_sparse_match,
        build_binary_token_predictor: predictor_builders::build_predictor_sparse_match,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_sparse_match,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Ppmd,
        canonical: "ppmd",
        aliases: ["ppmd"],
        feature: "backend-ppmd",
        spec_helper_refs: anchor,
        metric_helper_refs: anchor,
        compile_plan: crate::spec::core::compile_rate_plan_ppmd,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_ppmd,
        encode_payload: crate::spec::core::encode_rate_payload_ppmd,
        display_label: crate::spec::core::rate_plan_display_label_ppmd,
        default_name: crate::spec::core::rate_plan_default_name_ppmd,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_ppmd,
        build_binary_token_predictor: predictor_builders::build_predictor_ppmd,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_ppmd,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Sequitur,
        canonical: "sequitur",
        aliases: ["sequitur"],
        feature: "backend-sequitur",
        spec_helper_refs: anchor,
        metric_helper_refs: anchor,
        compile_plan: crate::spec::core::compile_rate_plan_sequitur,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_sequitur,
        encode_payload: crate::spec::core::encode_rate_payload_sequitur,
        display_label: crate::spec::core::rate_plan_display_label_sequitur,
        default_name: crate::spec::core::rate_plan_default_name_sequitur,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_sequitur,
        build_binary_token_predictor: predictor_builders::build_predictor_sequitur,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_sequitur,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Ctw,
        canonical: "ctw",
        aliases: ["ctw", "ac-ctw"],
        feature: "backend-ctw",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_ctw,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_ctw,
        encode_payload: crate::spec::core::encode_rate_payload_ctw,
        display_label: crate::spec::core::rate_plan_display_label_ctw,
        default_name: crate::spec::core::rate_plan_default_name_ctw,
        trace_strategy: PublicTraceStrategy::Ctw,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_true,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_ctw,
        build_binary_token_predictor: predictor_builders::build_predictor_binary_tokens_ctw,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_ctw,
        entropy_rate: entropy_ctw,
        joint_entropy_rate: joint_entropy_ctw,
        conditional_chain_rate: conditional_chain_ctw,
    },
    backend {
        kind: FacCtw,
        canonical: "fac-ctw",
        aliases: ["fac-ctw"],
        feature: "backend-ctw",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_fac_ctw,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_fac_ctw,
        encode_payload: crate::spec::core::encode_rate_payload_fac_ctw,
        display_label: crate::spec::core::rate_plan_display_label_fac_ctw,
        default_name: crate::spec::core::rate_plan_default_name_fac_ctw,
        trace_strategy: PublicTraceStrategy::FacCtw,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_true,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_true,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_fac_ctw,
        build_binary_token_predictor: predictor_builders::build_predictor_binary_tokens_fac_ctw,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_fac_ctw,
        entropy_rate: entropy_fac_ctw,
        joint_entropy_rate: joint_entropy_fac_ctw,
        conditional_chain_rate: conditional_chain_fac_ctw,
    },
    backend {
        kind: Zpaq,
        canonical: "zpaq",
        aliases: ["zpaq"],
        feature: "backend-zpaq",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_zpaq,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_zpaq,
        encode_payload: crate::spec::core::encode_rate_payload_zpaq,
        display_label: crate::spec::core::rate_plan_display_label_zpaq,
        default_name: crate::spec::core::rate_plan_default_name_zpaq,
        trace_strategy: PublicTraceStrategy::Zpaq,
        supports_biased_entropy: false,
        supports_frozen_conditioning: false,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_false,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_true,
        build_predictor: predictor_builders::build_predictor_zpaq,
        build_binary_token_predictor: predictor_builders::build_predictor_zpaq,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_zpaq,
        entropy_rate: entropy_zpaq,
        joint_entropy_rate: joint_entropy_zpaq,
        conditional_chain_rate: conditional_chain_zpaq,
    },
    backend {
        kind: Mixture,
        canonical: "mixture",
        aliases: ["mixture"],
        feature: "backend-mixture",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_mixture,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_mixture,
        encode_payload: crate::spec::core::encode_rate_payload_mixture,
        display_label: crate::spec::core::rate_plan_display_label_mixture,
        default_name: crate::spec::core::rate_plan_default_name_mixture,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::mixture_supports_native_bit_prediction,
        supports_byte_prefix_mass: crate::runtime::mixture_supports_byte_prefix_mass,
        supports_efficient_byte_packed_bit_sessions:
            crate::runtime::mixture_supports_efficient_byte_packed_bit_sessions,
        supports_reversible_bit_updates: crate::runtime::mixture_supports_reversible_bit_updates,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_mixture,
        build_predictor: predictor_builders::build_predictor_mixture,
        build_binary_token_predictor: predictor_builders::build_predictor_binary_tokens_mixture,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_mixture,
        entropy_rate: entropy_mixture,
        joint_entropy_rate: joint_entropy_mixture,
        conditional_chain_rate: conditional_chain_mixture,
    },
    backend {
        kind: Particle,
        canonical: "particle",
        aliases: ["particle"],
        feature: "backend-particle",
        spec_helper_refs: anchor,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_particle,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_particle,
        encode_payload: crate::spec::core::encode_rate_payload_particle,
        display_label: crate::spec::core::rate_plan_display_label_particle,
        default_name: crate::spec::core::rate_plan_default_name_particle,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_particle,
        build_binary_token_predictor: predictor_builders::build_predictor_particle,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_particle,
        entropy_rate: entropy_particle,
        joint_entropy_rate: joint_entropy_particle,
        conditional_chain_rate: conditional_chain_particle,
    },
    backend {
        kind: Calibrated,
        canonical: "calibrated",
        aliases: ["calibrated"],
        feature: "backend-calibrated",
        spec_helper_refs: anchor,
        metric_helper_refs: anchor,
        compile_plan: crate::spec::core::compile_rate_plan_calibrated,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_calibrated,
        encode_payload: crate::spec::core::encode_rate_payload_calibrated,
        display_label: crate::spec::core::rate_plan_display_label_calibrated,
        default_name: crate::spec::core::rate_plan_default_name_calibrated,
        trace_strategy: PublicTraceStrategy::PredictorBacked,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::calibrated_supports_native_bit_prediction,
        supports_byte_prefix_mass: crate::runtime::calibrated_supports_byte_prefix_mass,
        supports_efficient_byte_packed_bit_sessions:
            crate::runtime::calibrated_supports_efficient_byte_packed_bit_sessions,
        supports_reversible_bit_updates: crate::runtime::calibrated_supports_reversible_bit_updates,
        method_family: None,
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_calibrated,
        build_predictor: predictor_builders::build_predictor_calibrated,
        build_binary_token_predictor: predictor_builders::build_predictor_binary_tokens_calibrated,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_calibrated,
        entropy_rate: entropy_prequential,
        joint_entropy_rate: joint_entropy_prequential,
        conditional_chain_rate: conditional_chain_prequential,
    },
    backend {
        kind: Mamba,
        canonical: "mamba",
        aliases: ["mamba"],
        feature: "backend-mamba",
        spec_helper_refs: drop,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_mamba,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_mamba,
        encode_payload: crate::spec::core::encode_rate_payload_mamba,
        display_label: crate::spec::core::rate_plan_display_label_mamba,
        default_name: crate::spec::core::rate_plan_default_name_mamba,
        trace_strategy: PublicTraceStrategy::Mamba,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: Some(MethodBackendFamily::Mamba),
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_mamba,
        build_binary_token_predictor: predictor_builders::build_predictor_mamba,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_mamba,
        entropy_rate: entropy_mamba,
        joint_entropy_rate: joint_entropy_mamba,
        conditional_chain_rate: conditional_chain_mamba,
    },
    backend {
        kind: Rwkv7,
        canonical: "rwkv7",
        aliases: ["rwkv7"],
        feature: "backend-rwkv",
        spec_helper_refs: drop,
        metric_helper_refs: drop,
        compile_plan: crate::spec::core::compile_rate_plan_rwkv7,
        to_wrapper: crate::spec::core::rate_plan_to_wrapper_rwkv7,
        encode_payload: crate::spec::core::encode_rate_payload_rwkv7,
        display_label: crate::spec::core::rate_plan_display_label_rwkv7,
        default_name: crate::spec::core::rate_plan_default_name_rwkv7,
        trace_strategy: PublicTraceStrategy::Rwkv7,
        supports_biased_entropy: true,
        supports_frozen_conditioning: true,
        supports_rate_coded_compression: true,
        supports_native_bit_prediction: crate::runtime::capability_always_false,
        supports_byte_prefix_mass: crate::runtime::capability_always_true,
        supports_efficient_byte_packed_bit_sessions: crate::runtime::capability_always_true,
        supports_reversible_bit_updates: crate::runtime::capability_always_false,
        method_family: Some(MethodBackendFamily::Rwkv7),
        contains_zpaq: crate::spec::core::rate_plan_contains_zpaq_false,
        build_predictor: predictor_builders::build_predictor_rwkv,
        build_binary_token_predictor: predictor_builders::build_predictor_rwkv,
        build_pdf_predictor: pdf_predictor_builders::build_pdf_predictor_rwkv,
        entropy_rate: entropy_rwkv,
        joint_entropy_rate: joint_entropy_rwkv,
        conditional_chain_rate: conditional_chain_rwkv,
    },
}

define_compression_backend_catalog! {
    backend {
        kind: Zpaq,
        canonical: "zpaq",
        aliases: ["zpaq"],
        feature: "backend-zpaq",
        helper_refs: anchor,
        compile_plan: crate::spec::core::compile_compression_plan_zpaq,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_zpaq,
        encode_payload: crate::spec::core::encode_compression_payload_zpaq,
        display_label: crate::spec::core::compression_plan_display_label_zpaq,
        uses_rate_backend: false,
        supports_decompression: true,
        build_runtime: build_compression_runtime_zpaq,
    },
    backend {
        kind: Rwkv7,
        canonical: "rwkv7",
        aliases: ["rwkv7"],
        feature: "backend-rwkv",
        helper_refs: drop,
        compile_plan: crate::spec::core::compile_compression_plan_rwkv7,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rwkv7,
        encode_payload: crate::spec::core::encode_compression_payload_rwkv7,
        display_label: crate::spec::core::compression_plan_display_label_rwkv7,
        uses_rate_backend: false,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rwkv,
    },
    backend {
        kind: RateAc,
        canonical: "rate-ac",
        aliases: ["rate-ac"],
        feature: none,
        helper_refs: anchor,
        compile_plan: crate::spec::core::compile_compression_plan_rate,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rate,
        encode_payload: crate::spec::core::encode_compression_payload_rate,
        display_label: crate::spec::core::compression_plan_display_label_rate,
        uses_rate_backend: true,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rate,
    },
    backend {
        kind: RateRans,
        canonical: "rate-rans",
        aliases: ["rate-rans"],
        feature: none,
        helper_refs: anchor,
        compile_plan: crate::spec::core::compile_compression_plan_rate,
        to_wrapper: crate::spec::core::compression_plan_to_wrapper_rate,
        encode_payload: crate::spec::core::encode_compression_payload_rate,
        display_label: crate::spec::core::compression_plan_display_label_rate,
        uses_rate_backend: true,
        supports_decompression: true,
        build_runtime: build_compression_runtime_rate,
    },
}

pub(crate) fn default_rate_backend_spec(kind: RateBackendKind) -> Option<RateBackend> {
    crate::rate_defaults::runtime_default_rate_backend_spec(kind)
}

pub(crate) fn first_enabled_default_rate_backend_spec() -> Option<RateBackend> {
    RATE_BACKEND_REGISTRY
        .iter()
        .filter(|descriptor| descriptor.enabled)
        .find_map(|descriptor| default_rate_backend_spec(descriptor.kind))
}

pub(crate) fn rate_backend_kernel(kind: RateBackendKind) -> &'static RateBackendKernel {
    RATE_BACKEND_KERNELS
        .iter()
        .find(|kernel| kernel.kind == kind)
        .unwrap_or_else(|| panic!("missing RATE_BACKEND_KERNELS entry for {kind:?}"))
}

pub(crate) fn compression_backend_kernel(
    kind: CompressionBackendKind,
) -> &'static CompressionBackendKernel {
    COMPRESSION_BACKEND_KERNELS
        .iter()
        .find(|kernel| kernel.kind == kind)
        .unwrap_or_else(|| panic!("missing COMPRESSION_BACKEND_KERNELS entry for {kind:?}"))
}

pub(crate) fn rate_backend_canonical_name(kind: RateBackendKind) -> &'static str {
    describe_rate_backend_kind(kind)
        .unwrap_or_else(|err| panic!("{err}"))
        .canonical
}

pub(crate) fn compression_backend_canonical_name(kind: CompressionBackendKind) -> &'static str {
    describe_compression_backend_kind(kind)
        .unwrap_or_else(|err| panic!("{err}"))
        .canonical
}

pub(crate) fn compile_rate_backend_plan_via_kernel(
    kind: RateBackendKind,
    backend: &RateBackend,
    env: &SpecEnvironment,
    depth: usize,
) -> SpecResult<RateBackendPlan> {
    (rate_backend_kernel(kind).compile_plan)(backend, env, depth)
}

pub(crate) fn compile_compression_backend_plan_via_kernel(
    kind: CompressionBackendKind,
    backend: &CompressionBackend,
    env: &SpecEnvironment,
) -> SpecResult<CompressionBackendPlan> {
    (compression_backend_kernel(kind).compile_plan)(backend, env)
}

pub(crate) fn rate_backend_wrapper_via_kernel(plan: &RateBackendPlan) -> RateBackend {
    (rate_backend_kernel(plan.kind()).to_wrapper)(plan)
}

pub(crate) fn compression_backend_wrapper_via_kernel(
    plan: &CompressionBackendPlan,
) -> CompressionBackend {
    (compression_backend_kernel(plan.kind()).to_wrapper)(plan)
}

pub(crate) fn encode_rate_backend_payload_via_kernel(plan: &RateBackendPlan, out: &mut Vec<u8>) {
    (rate_backend_kernel(plan.kind()).encode_payload)(plan, out);
}

pub(crate) fn encode_compression_backend_payload_via_kernel(
    plan: &CompressionBackendPlan,
    out: &mut Vec<u8>,
) {
    (compression_backend_kernel(plan.kind()).encode_payload)(plan, out);
}

pub(crate) fn rate_backend_display_label_via_kernel(plan: &RateBackendPlan) -> String {
    (rate_backend_kernel(plan.kind()).display_label)(plan)
}

pub(crate) fn rate_backend_default_name_via_kernel(plan: &RateBackendPlan) -> String {
    (rate_backend_kernel(plan.kind()).default_name)(plan)
}

pub(crate) fn compression_backend_display_label_via_kernel(
    plan: &CompressionBackendPlan,
) -> String {
    (compression_backend_kernel(plan.kind()).display_label)(plan)
}

pub(crate) fn rate_backend_capabilities_via_kernel(
    plan: &RateBackendPlan,
) -> RateBackendCapabilities {
    let kernel = rate_backend_kernel(plan.kind());
    RateBackendCapabilities {
        canonical_name: rate_backend_canonical_name(plan.kind()),
        display_label: Arc::<str>::from((kernel.display_label)(plan)),
        trace_strategy: kernel.trace_strategy,
        supports_biased_entropy: kernel.supports_biased_entropy,
        supports_frozen_conditioning: kernel.supports_frozen_conditioning,
        supports_rate_coded_compression: kernel.supports_rate_coded_compression,
        supports_native_bit_prediction: (kernel.supports_native_bit_prediction)(plan),
        supports_byte_prefix_mass: (kernel.supports_byte_prefix_mass)(plan),
        supports_efficient_byte_packed_bit_sessions: (kernel
            .supports_efficient_byte_packed_bit_sessions)(
            plan
        ),
        supports_reversible_bit_updates: (kernel.supports_reversible_bit_updates)(plan),
        contains_zpaq: (kernel.contains_zpaq)(plan),
        method_family: kernel.method_family,
    }
}

pub(crate) fn compression_backend_capabilities_via_kernel(
    plan: &CompressionBackendPlan,
) -> CompressionBackendCapabilities {
    let kernel = compression_backend_kernel(plan.kind());
    CompressionBackendCapabilities {
        canonical_name: compression_backend_canonical_name(plan.kind()),
        display_label: Arc::<str>::from(compression_backend_display_label_via_kernel(plan)),
        uses_rate_backend: kernel.uses_rate_backend,
        supports_decompression: kernel.supports_decompression,
    }
}

#[cfg(feature = "vm")]
pub(crate) fn rate_backend_trace_model_strategy(
    backend: &CompiledRateBackend,
) -> TraceModelStrategy {
    match backend.capabilities().trace_strategy {
        PublicTraceStrategy::Rosa => TraceModelStrategy::Rosa,
        PublicTraceStrategy::Ctw => TraceModelStrategy::Ctw,
        PublicTraceStrategy::FacCtw => TraceModelStrategy::FacCtw,
        PublicTraceStrategy::PredictorBacked => TraceModelStrategy::PredictorBacked,
        PublicTraceStrategy::Zpaq => TraceModelStrategy::Zpaq,
        PublicTraceStrategy::Mamba => TraceModelStrategy::Mamba,
        PublicTraceStrategy::Rwkv7 => TraceModelStrategy::Rwkv7,
    }
}

#[cfg(feature = "backend-rwkv")]
/// Execute `f` with the RWKV method string + parsed spec from a compiled backend.
fn with_rwkv_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(&str, &crate::rwkvzip::MethodSpec) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Rwkv7 {
            method,
            parsed_method,
            ..
        },
        "rwkv kernel used with non-rwkv plan"
    );
    f(method, parsed_method)
}

#[cfg(feature = "backend-mamba")]
/// Execute `f` with the Mamba method string + parsed spec from a compiled backend.
fn with_mamba_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(&str, &crate::mambazip::MethodSpec) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Mamba {
            method,
            parsed_method,
            ..
        },
        "mamba kernel used with non-mamba plan"
    );
    f(method, parsed_method)
}

#[cfg(feature = "backend-zpaq")]
/// Execute `f` with the ZPAQ method string from a compiled backend.
fn with_zpaq_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(&str) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Zpaq { method },
        "zpaq kernel used with non-zpaq plan"
    );
    f(method)
}

#[cfg(feature = "backend-particle")]
/// Execute `f` with the particle spec from a compiled backend.
fn with_particle_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(&crate::api::ParticleSpec) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Particle { spec },
        "particle kernel used with non-particle plan"
    );
    f(spec)
}

#[cfg(feature = "backend-ctw")]
/// Execute `f` with CTW depth from a compiled backend.
fn with_ctw_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(usize) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::Ctw { depth },
        "ctw kernel used with non-ctw plan"
    );
    f(*depth)
}

#[cfg(feature = "backend-ctw")]
/// Execute `f` with FAC-CTW `(base_depth, encoding_bits)` from a compiled backend.
fn with_fac_ctw_backend_plan<T>(
    backend: &CompiledRateBackend,
    f: impl FnOnce(usize, usize, bool) -> InfotheoryResult<T>,
) -> InfotheoryResult<T> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::RateBackendPlan::FacCtw {
            base_depth,
            num_percept_bits: _,
            encoding_bits,
            msb_first,
        },
        "fac-ctw kernel used with non-fac-ctw plan"
    );
    f(*base_depth, *encoding_bits, *msb_first)
}

/// Execute `f` with the ZPAQ compression method from a compiled compression backend.
fn with_zpaq_compression_plan<T>(
    backend: &CompiledCompressionBackend,
    f: impl FnOnce(&str, usize) -> Result<T, String>,
) -> Result<T, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::CompressionBackendPlan::Zpaq { method, threads },
        "zpaq compression kernel used with non-zpaq plan"
    );
    f(method, *threads)
}

/// Execute `f` with `(rate_backend, coder, framing)` from a rate-coded compression backend.
fn with_rate_compression_plan<T>(
    backend: &CompiledCompressionBackend,
    f: impl FnOnce(
        &Arc<RateBackendPlan>,
        crate::coders::CoderType,
        crate::compression::FramingMode,
    ) -> Result<T, String>,
) -> Result<T, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::CompressionBackendPlan::Rate {
            rate_backend,
            coder,
            framing,
        },
        "rate compression kernel used with non-rate compression plan"
    );
    f(rate_backend, *coder, *framing)
}

#[cfg(feature = "backend-rwkv")]
/// Execute `f` with RWKV compression `(method, parsed_method, coder)` from a compiled backend.
fn with_rwkv_compression_plan<T>(
    backend: &CompiledCompressionBackend,
    f: impl FnOnce(&str, &crate::rwkvzip::MethodSpec, crate::coders::CoderType) -> Result<T, String>,
) -> Result<T, String> {
    expect_plan_ref!(
        backend.plan(),
        crate::spec::core::CompressionBackendPlan::Rwkv7 {
            method,
            parsed_method,
            coder,
            ..
        },
        "rwkv compression kernel used with non-rwkv compression plan"
    );
    f(method, parsed_method, *coder)
}

fn entropy_prequential(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    crate::try_prequential_rate_backend(data, &[], backend)
}

fn joint_entropy_prequential(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    entropy_prequential(&joint, backend).map(|bits| bits * 2.0)
}

fn conditional_chain_prequential(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::try_prequential_rate_backend(data, prefix_parts, backend)
}

#[cfg(feature = "backend-rosa")]
fn entropy_rosa(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let crate::spec::core::RateBackendPlan::RosaPlus { max_order } = backend.plan() else {
        unreachable!("rosa kernel used with non-rosa plan")
    };
    let mut model = RosaPlus::new(*max_order, false, 0, 42);
    Ok(model.predictive_entropy_rate(data))
}

#[cfg(feature = "backend-rosa")]
fn joint_entropy_rosa(x: &[u8], y: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let crate::spec::core::RateBackendPlan::RosaPlus { max_order } = backend.plan() else {
        unreachable!("rosa kernel used with non-rosa plan")
    };
    let joint_symbols: Vec<u32> = (0..x.len())
        .map(|idx| (x[idx] as u32) * 256 + (y[idx] as u32))
        .collect();
    let mut model = RosaPlus::new(*max_order, false, 0, 42);
    Ok(model.entropy_rate_cps(&joint_symbols))
}

#[cfg(feature = "backend-rosa")]
fn conditional_chain_rosa(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    crate::try_frozen_plugin_rate_backend(data, prefix_parts, backend)
}

#[cfg(feature = "backend-rwkv")]
fn entropy_rwkv(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_rwkv_backend_plan(backend, |method, parsed_method| {
        crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy(data).map_err(|err| {
                InfotheoryError::runtime(format!("rwkv method entropy scoring failed: {err:#}"))
            })
        })
    })
}

#[cfg(feature = "backend-rwkv")]
fn joint_entropy_rwkv(x: &[u8], y: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_rwkv_backend_plan(backend, |method, parsed_method| {
        crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
                InfotheoryError::runtime(format!(
                    "rwkv method joint-entropy scoring failed: {err:#}"
                ))
            })
        })
    })
}

#[cfg(feature = "backend-rwkv")]
fn conditional_chain_rwkv(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_rwkv_backend_plan(backend, |method, parsed_method| {
        crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy_conditional_chain(prefix_parts, data)
                .map_err(|err| {
                    InfotheoryError::runtime(format!(
                        "rwkv method conditional-chain scoring failed: {err:#}"
                    ))
                })
        })
    })
}

#[cfg(feature = "backend-mamba")]
fn entropy_mamba(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_mamba_backend_plan(backend, |method, parsed_method| {
        crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy(data).map_err(|err| {
                InfotheoryError::runtime(format!("mamba method entropy scoring failed: {err:#}"))
            })
        })
    })
}

#[cfg(feature = "backend-mamba")]
fn joint_entropy_mamba(x: &[u8], y: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_mamba_backend_plan(backend, |method, parsed_method| {
        crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
            c.joint_cross_entropy_aligned_min(x, y).map_err(|err| {
                InfotheoryError::runtime(format!(
                    "mamba method joint-entropy scoring failed: {err:#}"
                ))
            })
        })
    })
}

#[cfg(feature = "backend-mamba")]
fn conditional_chain_mamba(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_mamba_backend_plan(backend, |method, parsed_method| {
        crate::with_mamba_method_spec_tls(method, parsed_method, |c| {
            c.cross_entropy_conditional_chain(prefix_parts, data)
                .map_err(|err| {
                    InfotheoryError::runtime(format!(
                        "mamba method conditional-chain scoring failed: {err:#}"
                    ))
                })
        })
    })
}

#[cfg(feature = "backend-zpaq")]
fn entropy_zpaq(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_zpaq_backend_plan(backend, |method| {
        zpaq_conditional_chain_rate_bits(method, &[], data)
    })
}

#[cfg(feature = "backend-zpaq")]
fn joint_entropy_zpaq(x: &[u8], y: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_zpaq_backend_plan(backend, |method| zpaq_joint_entropy_rate_bits(method, x, y))
}

#[cfg(feature = "backend-zpaq")]
fn conditional_chain_zpaq(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_zpaq_backend_plan(backend, |method| {
        zpaq_conditional_chain_rate_bits(method, prefix_parts, data)
    })
}

#[cfg(feature = "backend-mixture")]
fn entropy_mixture(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    mixture_entropy_rate_bits(data, backend)
}

#[cfg(feature = "backend-mixture")]
fn joint_entropy_mixture(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    mixture_joint_entropy_rate_bits(x, y, backend)
}

#[cfg(feature = "backend-mixture")]
fn conditional_chain_mixture(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    mixture_conditional_chain_rate_bits(prefix_parts, data, backend)
}

#[cfg(feature = "backend-particle")]
fn entropy_particle(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_particle_backend_plan(backend, |spec| {
        particle_stream_entropy_rate_bits(data, spec)
    })
}

#[cfg(feature = "backend-particle")]
fn joint_entropy_particle(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_particle_backend_plan(backend, |spec| particle_joint_entropy_rate_bits(x, y, spec))
}

#[cfg(feature = "backend-particle")]
fn conditional_chain_particle(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_particle_backend_plan(backend, |spec| {
        particle_conditional_chain_rate_bits(prefix_parts, data, spec)
    })
}

#[cfg(feature = "backend-ctw")]
fn entropy_ctw(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_ctw_backend_plan(backend, |depth| ctw_entropy_rate_bits(depth, data))
}

#[cfg(feature = "backend-ctw")]
fn joint_entropy_ctw(x: &[u8], y: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_ctw_backend_plan(backend, |depth| ctw_joint_entropy_rate_bits(depth, x, y))
}

#[cfg(feature = "backend-ctw")]
fn conditional_chain_ctw(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_ctw_backend_plan(backend, |depth| {
        ctw_conditional_chain_rate_bits(depth, prefix_parts, data)
    })
}

#[cfg(feature = "backend-ctw")]
fn entropy_fac_ctw(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    with_fac_ctw_backend_plan(backend, |base_depth, encoding_bits, msb_first| {
        fac_ctw_entropy_rate_bits(base_depth, encoding_bits, msb_first, data)
    })
}

#[cfg(feature = "backend-ctw")]
fn joint_entropy_fac_ctw(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_fac_ctw_backend_plan(backend, |base_depth, encoding_bits, msb_first| {
        fac_ctw_joint_entropy_rate_bits(base_depth, encoding_bits, msb_first, x, y)
    })
}

#[cfg(feature = "backend-ctw")]
fn conditional_chain_fac_ctw(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    with_fac_ctw_backend_plan(backend, |base_depth, encoding_bits, msb_first| {
        fac_ctw_conditional_chain_rate_bits(
            base_depth,
            encoding_bits,
            msb_first,
            prefix_parts,
            data,
        )
    })
}

fn build_compression_runtime_zpaq(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    with_zpaq_compression_plan(backend, |method, threads| {
        Ok(CompressionRuntimeHandle::Zpaq {
            method: method.to_string(),
            threads,
        })
    })
}

#[cfg(feature = "backend-rwkv")]
fn build_compression_runtime_rwkv(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    with_rwkv_compression_plan(backend, |method, parsed_method, coder| {
        Ok(CompressionRuntimeHandle::Rwkv7 {
            method: method.to_string(),
            parsed_method: parsed_method.clone(),
            coder,
        })
    })
}

fn build_compression_runtime_rate(
    backend: &CompiledCompressionBackend,
) -> Result<CompressionRuntimeHandle, String> {
    with_rate_compression_plan(backend, |rate_backend, coder, framing| {
        Ok(CompressionRuntimeHandle::Rate {
            rate_backend: crate::spec::core::compiled_rate_backend_from_plan(rate_backend.clone())
                .map_err(|err| format!("failed to compile rate compression backend plan: {err}"))?,
            coder,
            framing,
        })
    })
}

fn build_rate_backend_predictor_via_kernel(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    (rate_backend_kernel(backend.plan().kind()).build_predictor)(backend, min_prob)
}

fn build_rate_backend_binary_token_predictor_via_kernel(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    (rate_backend_kernel(backend.plan().kind()).build_binary_token_predictor)(backend, min_prob)
}

fn build_rate_pdf_predictor_via_kernel(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    (rate_backend_kernel(backend.plan().kind()).build_pdf_predictor)(backend)
}

#[cfg(feature = "backend-calibrated")]
pub(super) fn compile_calibrated_base_backend(
    base: &Arc<RateBackendPlan>,
) -> Result<CompiledRateBackend, String> {
    crate::spec::core::compiled_rate_backend_from_plan(base.clone())
        .map_err(|err| format!("failed to compile calibrated base backend plan: {err}"))
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
        threads: usize,
    },
    #[cfg(feature = "backend-rwkv")]
    Rwkv7 {
        method: String,
        parsed_method: crate::rwkvzip::MethodSpec,
        coder: crate::coders::CoderType,
    },
    Rate {
        rate_backend: CompiledRateBackend,
        coder: crate::coders::CoderType,
        framing: crate::compression::FramingMode,
    },
}

impl CompressionRuntime for CompressionRuntimeHandle {
    fn compress_size(&mut self, data: &[u8]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method, threads } => {
                if *threads <= 1 {
                    crate::try_zpaq_compress_size_bytes(data, method.as_str())
                } else {
                    crate::try_zpaq_compress_size_parallel_bytes(data, method.as_str(), *threads)
                }
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress_size(data, *coder).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 compression failed: {err:#}"))
                })
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_size(data, rate_backend, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded compression failed: {err:#}"))
                }),
        }
    }

    fn compress_size_chain(&mut self, parts: &[&[u8]]) -> InfotheoryResult<u64> {
        match self {
            CompressionRuntimeHandle::Zpaq { method, threads } => {
                let reader = SliceChainReader::new(parts);
                if *threads <= 1 {
                    crate::try_zpaq_compress_size_stream(reader, method.as_str())
                } else {
                    crate::try_zpaq_compress_size_stream_parallel(reader, method.as_str(), *threads)
                }
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress_size_chain(parts, *coder).map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 chain compression failed: {err:#}"))
                })
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                crate::compression::compress_rate_size_chain(parts, rate_backend, *coder, *framing)
                    .map_err(|err| {
                        InfotheoryError::runtime(format!(
                            "rate-coded chain compression failed: {err:#}"
                        ))
                    })
            }
        }
    }

    fn compress_bytes(&mut self, data: &[u8]) -> InfotheoryResult<Vec<u8>> {
        match self {
            CompressionRuntimeHandle::Zpaq { method, .. } => {
                crate::zpaq_compress_to_vec(data, method).map_err(|err| {
                    InfotheoryError::runtime(format!("zpaq byte compression failed: {err:#}"))
                })
            }
            #[cfg(feature = "backend-rwkv")]
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                coder,
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| {
                c.compress(data, *coder)
            })
            .map_err(|err| {
                InfotheoryError::runtime(format!("rwkv7 byte compression failed: {err:#}"))
            }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::compress_rate_bytes(data, rate_backend, *coder, *framing)
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
            CompressionRuntimeHandle::Rwkv7 {
                method,
                parsed_method,
                ..
            } => crate::with_rwkv_method_spec_tls(method, parsed_method, |c| c.decompress(input))
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rwkv7 decompression failed: {err:#}"))
                }),
            CompressionRuntimeHandle::Rate {
                rate_backend,
                coder,
                framing,
            } => crate::compression::decompress_rate_bytes(input, rate_backend, *coder, *framing)
                .map_err(|err| {
                    InfotheoryError::runtime(format!("rate-coded decompression failed: {err:#}"))
                }),
        }
    }
}

impl CompressionFactory for CompiledCompressionBackend {
    type Runtime = CompressionRuntimeHandle;

    fn build_compression_runtime(&self) -> Result<Self::Runtime, String> {
        (compression_backend_kernel(self.plan().kind()).build_runtime)(self)
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
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    build_rate_backend_predictor_via_kernel(backend, min_prob)
}

/// Shared spec -> predictor runtime builder using the library's default probability floor.
pub(crate) fn build_rate_backend_predictor_default(
    backend: &CompiledRateBackend,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    build_rate_backend_predictor(backend, crate::mixture::DEFAULT_MIN_PROB)
}

/// Shared spec -> predictor runtime builder for binary-token applications.
///
/// Native bit backends expose their true bit-token predictors here. Byte-native
/// backends are adapted by observing the literal byte symbols `0` and `1`,
/// with downstream callers normalizing those two log-probabilities into a
/// binary prediction.
pub(crate) fn build_rate_backend_binary_token_predictor(
    backend: &CompiledRateBackend,
    min_prob: f64,
) -> Result<crate::mixture::RateBackendPredictor, String> {
    build_rate_backend_binary_token_predictor_via_kernel(backend, min_prob)
}

/// Shared spec -> compression predictor runtime builder.
pub(crate) fn build_rate_pdf_predictor(
    backend: &CompiledRateBackend,
) -> anyhow::Result<crate::compression::RatePdfPredictor> {
    build_rate_pdf_predictor_via_kernel(backend)
}

/// Shared spec -> compression runtime builder.
pub(crate) fn build_compression_runtime(
    backend: &CompiledCompressionBackend,
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

#[cfg(feature = "backend-zpaq")]
fn zpaq_conditional_chain_rate_bits(
    method: &str,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut model = ZpaqRateModel::new(method.to_owned(), 2f64.powi(-24));
    for &part in prefix_parts {
        model.update_and_score(part);
    }
    let bits = model.update_and_score(data);
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-zpaq")]
fn zpaq_joint_entropy_rate_bits(method: &str, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut model = ZpaqRateModel::new(method.to_owned(), 2f64.powi(-24));
    let bits = model.update_and_score(&joint);
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn build_compiled_mixture_runtime(
    backend: &CompiledRateBackend,
) -> Result<crate::mixture::MixtureRuntime, InfotheoryError> {
    let experts = crate::mixture::expert_configs_from_compiled_mixture(backend)?;
    crate::mixture::build_mixture_runtime_from_compiled(backend, &experts).map_err(|err| {
        InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {err}"))
    })
}

#[cfg(feature = "backend-mixture")]
fn mixture_entropy_rate_bits(data: &[u8], backend: &CompiledRateBackend) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut mix = build_compiled_mixture_runtime(backend)?;
    mix.begin_stream(Some(data.len() as u64))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
    let mut bits = 0.0;
    for &byte in data {
        bits -= mix.step(byte) / std::f64::consts::LN_2;
    }
    mix.finish_stream().map_err(|err| {
        InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
    })?;
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn mixture_joint_entropy_rate_bits(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut mix = build_compiled_mixture_runtime(backend)?;
    mix.begin_stream(Some(joint.len() as u64))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
    let mut bits = 0.0;
    for &byte in &joint {
        bits -= mix.step(byte) / std::f64::consts::LN_2;
    }
    mix.finish_stream().map_err(|err| {
        InfotheoryError::runtime(format!("Mixture stream finalize failed: {err}"))
    })?;
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-mixture")]
fn mixture_conditional_chain_rate_bits(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut mix = build_compiled_mixture_runtime(backend)?;
    let total = prefix_parts
        .iter()
        .map(|part| part.len() as u64)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    mix.begin_stream(Some(total))
        .map_err(|err| InfotheoryError::runtime(format!("Mixture stream init failed: {err}")))?;
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

#[cfg(feature = "backend-particle")]
fn particle_stream_entropy_rate_bits(
    data: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut runtime = ParticleRuntime::new(spec);
    let mut bits = 0.0;
    for &byte in data {
        bits -= runtime.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-particle")]
fn particle_conditional_chain_rate_bits(
    prefix_parts: &[&[u8]],
    data: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut runtime = ParticleRuntime::new(spec);
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

#[cfg(feature = "backend-particle")]
fn particle_joint_entropy_rate_bits(
    x: &[u8],
    y: &[u8],
    spec: &crate::api::ParticleSpec,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let joint = interleave_aligned_bytes(x, y);
    let mut runtime = ParticleRuntime::new(spec);
    let mut bits = 0.0;
    for &byte in &joint {
        bits -= runtime.step(byte) / std::f64::consts::LN_2;
    }
    Ok(bits / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
#[inline]
fn ctw_byte_bit_msb(byte: u8, bit_idx: usize) -> bool {
    ctw_symbol_bit_msb(byte, 8, bit_idx)
}

#[cfg(feature = "backend-ctw")]
#[inline]
fn ctw_update_byte_msb(tree: &mut ContextTree, byte: u8) {
    for bit_idx in 0..8usize {
        tree.update(ctw_byte_bit_msb(byte, bit_idx));
    }
}

#[cfg(feature = "backend-ctw")]
fn ctw_entropy_rate_bits(depth: usize, data: &[u8]) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut tree = ContextTree::new(depth);
    for &byte in data {
        ctw_update_byte_msb(&mut tree, byte);
    }
    let ln_p = tree.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_entropy_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    msb_first: bool,
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let bits_per_byte = encoding_bits;
    let mut fac = FacContextTree::new(base_depth, bits_per_byte);
    fac.reserve_for_symbols(data.len());
    for &byte in data {
        if msb_first {
            fac.update_byte_msb(byte);
        } else {
            fac.update_byte_lsb(byte);
        }
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn ctw_joint_entropy_rate_bits(depth: usize, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    let mut tree = ContextTree::new(depth);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        ctw_update_byte_msb(&mut tree, xb);
        ctw_update_byte_msb(&mut tree, yb);
    }
    let ln_p = tree.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_joint_entropy_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    msb_first: bool,
    x: &[u8],
    y: &[u8],
) -> InfotheoryResult<f64> {
    let bits_per_byte = encoding_bits;
    let mut fac = FacContextTree::new(base_depth, bits_per_byte * 2);
    for (&xb, &yb) in x.iter().zip(y.iter()) {
        for idx in 0..bits_per_byte {
            let bit_idx_x = idx * 2;
            let bit_idx_y = bit_idx_x + 1;
            let x_bit = if msb_first {
                ctw_symbol_bit_msb(xb, bits_per_byte, idx)
            } else {
                ((xb >> idx) & 1) == 1
            };
            let y_bit = if msb_first {
                ctw_symbol_bit_msb(yb, bits_per_byte, idx)
            } else {
                ((yb >> idx) & 1) == 1
            };
            fac.update(x_bit, bit_idx_x);
            fac.update(y_bit, bit_idx_y);
        }
    }
    let ln_p = fac.get_log_block_probability();
    Ok((-ln_p / std::f64::consts::LN_2) / (x.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn ctw_conditional_chain_rate_bits(
    depth: usize,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut tree = ContextTree::new(depth);
    for &part in prefix_parts {
        for &byte in part {
            ctw_update_byte_msb(&mut tree, byte);
        }
    }
    let log_p_prefix = tree.get_log_block_probability();
    for &byte in data {
        ctw_update_byte_msb(&mut tree, byte);
    }
    let log_p_joint = tree.get_log_block_probability();
    let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
    Ok(bits / (data.len() as f64))
}

#[cfg(feature = "backend-ctw")]
fn fac_ctw_conditional_chain_rate_bits(
    base_depth: usize,
    encoding_bits: usize,
    msb_first: bool,
    prefix_parts: &[&[u8]],
    data: &[u8],
) -> InfotheoryResult<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let bits_per_byte = encoding_bits;
    let mut fac = FacContextTree::new(base_depth, bits_per_byte);
    for &part in prefix_parts {
        for &byte in part {
            for idx in 0..bits_per_byte {
                let bit = if msb_first {
                    ctw_symbol_bit_msb(byte, bits_per_byte, idx)
                } else {
                    ((byte >> idx) & 1) == 1
                };
                fac.update(bit, idx);
            }
        }
    }
    let log_p_prefix = fac.get_log_block_probability();
    for &byte in data {
        for idx in 0..bits_per_byte {
            let bit = if msb_first {
                ctw_symbol_bit_msb(byte, bits_per_byte, idx)
            } else {
                ((byte >> idx) & 1) == 1
            };
            fac.update(bit, idx);
        }
    }
    let log_p_joint = fac.get_log_block_probability();
    let bits = -(log_p_joint - log_p_prefix) / std::f64::consts::LN_2;
    Ok(bits / (data.len() as f64))
}

fn execute_entropy_rate_backend(
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    (rate_backend_kernel(backend.plan().kind()).entropy_rate)(data, backend)
}

pub(crate) fn try_entropy_rate_backend_direct(
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    execute_entropy_rate_backend(data, backend)
}

pub(crate) fn try_cross_entropy_rate_backend_direct(
    test_data: &[u8],
    train_data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if backend.plan().kind() == RateBackendKind::Zpaq {
        return (rate_backend_kernel(backend.plan().kind()).conditional_chain_rate)(
            &[train_data],
            test_data,
            backend,
        );
    }
    crate::try_frozen_plugin_rate_backend(test_data, &[train_data], backend)
}

pub(crate) fn try_joint_entropy_rate_backend_direct(
    x: &[u8],
    y: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(0.0);
    }
    let n = x.len().min(y.len());
    let x = &x[..n];
    let y = &y[..n];

    (rate_backend_kernel(backend.plan().kind()).joint_entropy_rate)(x, y, backend)
}

pub(crate) fn try_cross_entropy_conditional_chain_backend(
    prefix_parts: &[&[u8]],
    data: &[u8],
    backend: &CompiledRateBackend,
) -> InfotheoryResult<f64> {
    (rate_backend_kernel(backend.plan().kind()).conditional_chain_rate)(prefix_parts, data, backend)
}

pub(crate) fn capability_always_true(_plan: &crate::spec::core::RateBackendPlan) -> bool {
    true
}

pub(crate) fn capability_always_false(_plan: &crate::spec::core::RateBackendPlan) -> bool {
    false
}

pub(crate) fn mixture_supports_native_bit_prediction(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!()
    };
    experts.iter().all(|e| {
        (crate::runtime::rate_backend_kernel(e.backend.kind()).supports_native_bit_prediction)(
            &e.backend,
        )
    })
}

pub(crate) fn mixture_supports_byte_prefix_mass(plan: &crate::spec::core::RateBackendPlan) -> bool {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!()
    };
    experts.iter().all(|e| {
        (crate::runtime::rate_backend_kernel(e.backend.kind()).supports_byte_prefix_mass)(
            &e.backend,
        )
    })
}

pub(crate) fn mixture_supports_efficient_byte_packed_bit_sessions(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!()
    };
    experts.iter().all(|e| {
        (crate::runtime::rate_backend_kernel(e.backend.kind())
            .supports_efficient_byte_packed_bit_sessions)(&e.backend)
    })
}

pub(crate) fn mixture_supports_reversible_bit_updates(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Mixture { experts, .. } = plan else {
        unreachable!()
    };
    experts.iter().all(|e| {
        (crate::runtime::rate_backend_kernel(e.backend.kind()).supports_reversible_bit_updates)(
            &e.backend,
        )
    })
}

pub(crate) fn calibrated_supports_native_bit_prediction(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!()
    };
    (crate::runtime::rate_backend_kernel(base.kind()).supports_native_bit_prediction)(base)
}

pub(crate) fn calibrated_supports_byte_prefix_mass(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!()
    };
    (crate::runtime::rate_backend_kernel(base.kind()).supports_byte_prefix_mass)(base)
}

pub(crate) fn calibrated_supports_efficient_byte_packed_bit_sessions(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!()
    };
    (crate::runtime::rate_backend_kernel(base.kind()).supports_efficient_byte_packed_bit_sessions)(
        base,
    )
}

pub(crate) fn calibrated_supports_reversible_bit_updates(
    plan: &crate::spec::core::RateBackendPlan,
) -> bool {
    let crate::spec::core::RateBackendPlan::Calibrated { base, .. } = plan else {
        unreachable!()
    };
    (crate::runtime::rate_backend_kernel(base.kind()).supports_reversible_bit_updates)(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::io::Read;

    #[cfg(feature = "backend-calibrated")]
    use crate::api::{CalibratedSpec, CalibrationContextKind};
    use crate::api::{CompressionBackend, RateBackend};
    #[cfg(feature = "backend-mixture")]
    use crate::api::{MixtureExpertSpec, MixtureKind, MixtureSpec};
    use crate::mixture::OnlineBytePredictor;
    #[cfg(any(feature = "backend-mixture", feature = "backend-calibrated"))]
    use std::sync::Arc;

    #[cfg(any(
        feature = "backend-ctw",
        feature = "backend-zpaq",
        feature = "backend-mixture",
        feature = "backend-rwkv",
        feature = "backend-mamba",
        feature = "all-backends"
    ))]
    fn compiled_rate_backend(backend: &RateBackend) -> CompiledRateBackend {
        backend.compile().expect("compiled rate backend")
    }

    #[cfg(feature = "backend-ctw")]
    fn compiled_compression_backend(backend: &CompressionBackend) -> CompiledCompressionBackend {
        backend.compile().expect("compiled compression backend")
    }

    fn sample_rate_backend_for_kind(kind: RateBackendKind) -> Option<RateBackend> {
        match kind {
            RateBackendKind::Mixture => {
                #[cfg(feature = "backend-mixture")]
                {
                    let base = first_enabled_default_rate_backend_spec()?;
                    Some(RateBackend::Mixture {
                        spec: Arc::new(MixtureSpec::new(
                            MixtureKind::Bayes,
                            vec![MixtureExpertSpec::new(base)],
                        )),
                    })
                }
                #[cfg(not(feature = "backend-mixture"))]
                {
                    None
                }
            }
            RateBackendKind::Calibrated => {
                #[cfg(feature = "backend-calibrated")]
                {
                    let base = first_enabled_default_rate_backend_spec()?;
                    Some(RateBackend::Calibrated {
                        spec: Arc::new(CalibratedSpec::new(base, CalibrationContextKind::Global)),
                    })
                }
                #[cfg(not(feature = "backend-calibrated"))]
                {
                    None
                }
            }
            _ => default_rate_backend_spec(kind),
        }
    }

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

        let mixture = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "mixture")
            .expect("mixture descriptor");
        assert_eq!(mixture.canonical, "mixture");

        let missing = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "mix");
        assert!(missing.is_none(), "legacy alias 'mix' must be rejected");
    }

    #[test]
    fn compression_registry_resolves_aliases() {
        let ac = find_backend_descriptor_in_registry(COMPRESSION_BACKEND_REGISTRY, "rate-ac")
            .expect("rate-ac descriptor");
        assert_eq!(ac.canonical, "rate-ac");

        let missing = find_backend_descriptor_in_registry(COMPRESSION_BACKEND_REGISTRY, "rate_ac");
        assert!(missing.is_none(), "legacy alias 'rate_ac' must be rejected");
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
    fn rate_backend_registry_kinds_have_runtime_kernels() {
        let kernel_kinds: HashSet<_> = RATE_BACKEND_KERNELS
            .iter()
            .map(|kernel| kernel.kind)
            .collect();
        for descriptor in RATE_BACKEND_REGISTRY {
            assert!(
                kernel_kinds.contains(&descriptor.kind),
                "missing runtime kernel for rate backend kind {:?}",
                descriptor.kind
            );
        }
    }

    #[test]
    fn enabled_rate_backend_catalog_entries_compile_and_build_consistently() {
        for descriptor in RATE_BACKEND_REGISTRY
            .iter()
            .filter(|descriptor| descriptor.enabled)
        {
            let backend = sample_rate_backend_for_kind(descriptor.kind)
                .unwrap_or_else(|| panic!("missing sample backend for {:?}", descriptor.kind));
            let compiled = backend.compile().unwrap_or_else(|err| {
                panic!(
                    "sample backend {:?} failed to compile: {err}",
                    descriptor.kind
                )
            });

            assert_eq!(compiled.canonical_name(), descriptor.canonical);
            assert_eq!(compiled.capabilities().canonical_name, descriptor.canonical);

            let mut predictor =
                build_rate_backend_predictor_default(&compiled).unwrap_or_else(|err| {
                    panic!("failed to build predictor for {:?}: {err}", descriptor.kind)
                });
            predictor.begin_stream(Some(1)).unwrap_or_else(|err| {
                panic!(
                    "failed to begin predictor stream for {:?}: {err}",
                    descriptor.kind
                )
            });
            predictor.finish_stream().unwrap_or_else(|err| {
                panic!(
                    "failed to finish predictor stream for {:?}: {err}",
                    descriptor.kind
                )
            });

            if compiled.supports_rate_coded_compression() {
                let _pdf = build_rate_pdf_predictor(&compiled).unwrap_or_else(|err| {
                    panic!(
                        "failed to build pdf predictor for {:?}: {err}",
                        descriptor.kind
                    )
                });
            }

            let mut bit_predictor = build_rate_backend_binary_token_predictor(
                &compiled,
                crate::mixture::DEFAULT_MIN_PROB,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "failed to build binary-token predictor for {:?}: {err}",
                    descriptor.kind
                )
            });
            let prediction = crate::prediction::binary_prediction_from_log_probs(
                bit_predictor.log_prob(0),
                bit_predictor.log_prob(1),
                crate::mixture::DEFAULT_MIN_PROB,
            );
            assert!(
                (prediction.p0 + prediction.p1 - 1.0).abs() < 1e-12,
                "binary-token predictor for {:?} must normalize to 1.0, got p0={} p1={}",
                descriptor.kind,
                prediction.p0,
                prediction.p1
            );
            bit_predictor.begin_stream(Some(1)).unwrap_or_else(|err| {
                panic!(
                    "failed to begin binary-token stream for {:?}: {err}",
                    descriptor.kind
                )
            });
            bit_predictor.finish_stream().unwrap_or_else(|err| {
                panic!(
                    "failed to finish binary-token stream for {:?}: {err}",
                    descriptor.kind
                )
            });
        }
    }

    #[test]
    fn compression_backend_registry_kinds_have_runtime_kernels() {
        let kernel_kinds: HashSet<_> = COMPRESSION_BACKEND_KERNELS
            .iter()
            .map(|kernel| kernel.kind)
            .collect();
        for descriptor in COMPRESSION_BACKEND_REGISTRY {
            assert!(
                kernel_kinds.contains(&descriptor.kind),
                "missing runtime kernel for compression backend kind {:?}",
                descriptor.kind
            );
        }
    }

    #[test]
    fn describe_compression_backend_uses_canonical_lookup_not_positional_indices() {
        let ac = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus { max_order: -1 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-ac");
        assert_eq!(ac.canonical, "rate-ac");

        let rans = try_describe_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::RosaPlus { max_order: -1 },
            coder: crate::coders::CoderType::RANS,
            framing: crate::compression::FramingMode::Framed,
        })
        .expect("descriptor for rate-rans");
        assert_eq!(rans.canonical, "rate-rans");
    }

    #[test]
    fn missing_descriptor_reports_registry_mismatch_error() {
        let err = registry::backend_descriptor_by_kind_checked(
            &[],
            RateBackendKind::RosaPlus,
            "RATE_BACKEND_REGISTRY",
        )
        .expect_err("missing descriptor should return an error");
        assert!(err.contains("internal backend registry mismatch"));
        assert!(err.contains("RosaPlus"));
    }

    #[test]
    fn registry_descriptors_round_trip_and_feature_messages_are_stable() {
        let rosa = find_backend_descriptor_in_registry(RATE_BACKEND_REGISTRY, "  ROSA  ")
            .expect("trimmed case-insensitive alias should resolve");
        assert_eq!(rosa.canonical, "rosaplus");

        for descriptor in RATE_BACKEND_REGISTRY {
            let described = describe_rate_backend_kind(descriptor.kind)
                .expect("rate descriptor kind lookup must succeed");
            assert_eq!(described.canonical, descriptor.canonical);
        }

        for descriptor in COMPRESSION_BACKEND_REGISTRY {
            let described = describe_compression_backend_kind(descriptor.kind)
                .expect("compression descriptor kind lookup must succeed");
            assert_eq!(described.canonical, descriptor.canonical);
        }

        let rate_with_feature = RATE_BACKEND_REGISTRY
            .iter()
            .find(|descriptor| descriptor.feature.is_some())
            .expect("at least one feature-gated rate backend");
        let rate_feature_message = registry::rate_backend_feature_error(rate_with_feature.kind);
        assert!(rate_feature_message.contains(rate_with_feature.canonical));
        assert!(rate_feature_message.contains("requires infotheory feature"));

        if let Some(rate_without_feature) = RATE_BACKEND_REGISTRY
            .iter()
            .find(|descriptor| descriptor.feature.is_none())
        {
            let rate_unavailable_message =
                registry::rate_backend_feature_error(rate_without_feature.kind);
            assert!(rate_unavailable_message.contains(rate_without_feature.canonical));
            assert!(rate_unavailable_message.contains("is unavailable"));
        }

        let compression_with_feature = COMPRESSION_BACKEND_REGISTRY
            .iter()
            .find(|descriptor| descriptor.feature.is_some())
            .expect("at least one feature-gated compression backend");
        let compression_feature_message =
            registry::compression_backend_feature_error(compression_with_feature.kind);
        assert!(compression_feature_message.contains(compression_with_feature.canonical));
        assert!(compression_feature_message.contains("requires infotheory feature"));

        let compression_without_feature = COMPRESSION_BACKEND_REGISTRY
            .iter()
            .find(|descriptor| descriptor.feature.is_none())
            .expect("at least one always-enabled compression backend");
        let compression_unavailable_message =
            registry::compression_backend_feature_error(compression_without_feature.kind);
        assert!(compression_unavailable_message.contains(compression_without_feature.canonical));
        assert!(compression_unavailable_message.contains("is unavailable"));
    }

    #[test]
    fn interleave_aligned_bytes_uses_shorter_input_and_preserves_pair_order() {
        let interleaved = interleave_aligned_bytes(&[1, 2, 3], &[9, 8]);
        assert_eq!(interleaved, vec![1, 9, 2, 8]);
    }

    #[test]
    fn slice_chain_reader_reads_across_empty_and_nonempty_parts() {
        let parts: [&[u8]; 4] = [b"ab", b"", b"c", b"def"];
        let mut reader = SliceChainReader::new(&parts);
        let mut out = [0u8; 6];

        let first = reader.read(&mut out[..3]).expect("first read");
        let second = reader.read(&mut out[3..]).expect("second read");
        let eof = reader.read(&mut out[0..1]).expect("eof read");

        assert_eq!(first, 3);
        assert_eq!(second, 3);
        assert_eq!(eof, 0);
        assert_eq!(&out, b"abcdef");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn rate_runtime_chain_matches_concatenated_stream_and_roundtrips() {
        let backend = compiled_compression_backend(&CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 5 },
            coder: crate::coders::CoderType::AC,
            framing: crate::compression::FramingMode::Framed,
        });
        let mut runtime = build_compression_runtime(&backend).expect("rate runtime");
        let parts: [&[u8]; 3] = [b"alpha", b"-", b"beta"];
        let joined = b"alpha-beta";

        let chain_size = runtime.compress_size_chain(&parts).expect("chain size");
        let joined_size = runtime.compress_size(joined).expect("joined size");
        assert_eq!(chain_size, joined_size);

        let encoded = runtime.compress_bytes(joined).expect("compress bytes");
        let decoded = runtime
            .decompress_bytes(&encoded)
            .expect("decompress bytes");
        assert_eq!(decoded, joined);
    }

    #[cfg(feature = "backend-zpaq")]
    #[test]
    fn zpaq_runtime_helpers_cover_empty_and_conditioned_paths() {
        let backend = compiled_rate_backend(&RateBackend::Zpaq {
            method: crate::api::ZpaqMethodSpec::literal("2"),
        });
        assert_eq!(
            zpaq_conditional_chain_rate_bits("2", &[b"prefix"], b"").expect("empty zpaq chain"),
            0.0
        );

        let entropy = entropy_zpaq(b"banana", &backend).expect("zpaq entropy");
        let joint = joint_entropy_zpaq(b"banana", b"bandit", &backend).expect("zpaq joint");
        let cond = conditional_chain_zpaq(&[b"ban"], b"ana", &backend).expect("zpaq conditional");
        assert!(entropy.is_finite() && entropy >= 0.0);
        assert!(joint.is_finite() && joint >= 0.0);
        assert!(cond.is_finite() && cond >= 0.0);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn ctw_family_runtime_entropy_helpers_are_finite() {
        let ctw = compiled_rate_backend(&RateBackend::Ctw { depth: 5 });
        let fac = compiled_rate_backend(&RateBackend::FacCtw {
            base_depth: 5,
            num_percept_bits: 8,
            encoding_bits: 8,
            msb_first: None,
        });

        let ctw_entropy = entropy_ctw(b"abracadabra", &ctw).expect("ctw entropy");
        let ctw_joint = joint_entropy_ctw(b"aaaa", b"bbbb", &ctw).expect("ctw joint");
        let ctw_cond = conditional_chain_ctw(&[b"abra"], b"cad", &ctw).expect("ctw conditional");
        assert!(ctw_entropy.is_finite() && ctw_entropy >= 0.0);
        assert!(ctw_joint.is_finite() && ctw_joint >= 0.0);
        assert!(ctw_cond.is_finite() && ctw_cond >= 0.0);

        let fac_entropy = entropy_fac_ctw(b"abracadabra", &fac).expect("fac entropy");
        let fac_joint = joint_entropy_fac_ctw(b"aaaa", b"bbbb", &fac).expect("fac joint");
        let fac_cond =
            conditional_chain_fac_ctw(&[b"abra"], b"cad", &fac).expect("fac conditional");
        assert!(fac_entropy.is_finite() && fac_entropy >= 0.0);
        assert!(fac_joint.is_finite() && fac_joint >= 0.0);
        assert!(fac_cond.is_finite() && fac_cond >= 0.0);
    }

    #[cfg(feature = "backend-mixture")]
    #[test]
    fn mixture_runtime_entropy_helpers_are_finite() {
        let mixture = RateBackend::Mixture {
            spec: Arc::new(MixtureSpec::new(
                MixtureKind::Bayes,
                vec![MixtureExpertSpec::new(RateBackend::Ctw { depth: 4 })],
            )),
        };
        let backend = compiled_rate_backend(&mixture);

        let entropy = entropy_mixture(b"mixture bytes", &backend).expect("mixture entropy");
        let joint = joint_entropy_mixture(b"abcd", b"wxyz", &backend).expect("mixture joint");
        let cond =
            conditional_chain_mixture(&[b"mix"], b"ture", &backend).expect("mixture conditional");
        assert!(entropy.is_finite() && entropy >= 0.0);
        assert!(joint.is_finite() && joint >= 0.0);
        assert!(cond.is_finite() && cond >= 0.0);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn rwkv_runtime_entropy_helpers_are_finite() {
        let backend = compiled_rate_backend(
            &default_rate_backend_spec(RateBackendKind::Rwkv7).expect("default rwkv spec"),
        );

        let entropy = entropy_rwkv(b"rwkv bytes", &backend).expect("rwkv entropy");
        let joint = joint_entropy_rwkv(b"abc", b"xyz", &backend).expect("rwkv joint");
        let cond = conditional_chain_rwkv(&[b"seed"], b"more", &backend).expect("rwkv conditional");
        assert!(entropy.is_finite() && entropy >= 0.0);
        assert!(joint.is_finite() && joint >= 0.0);
        assert!(cond.is_finite() && cond >= 0.0);
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn mamba_runtime_entropy_helpers_are_finite() {
        let backend = compiled_rate_backend(
            &default_rate_backend_spec(RateBackendKind::Mamba).expect("default mamba spec"),
        );

        let entropy = entropy_mamba(b"mamba bytes", &backend).expect("mamba entropy");
        let joint = joint_entropy_mamba(b"abc", b"xyz", &backend).expect("mamba joint");
        let cond =
            conditional_chain_mamba(&[b"seed"], b"more", &backend).expect("mamba conditional");
        assert!(entropy.is_finite() && entropy >= 0.0);
        assert!(joint.is_finite() && joint >= 0.0);
        assert!(cond.is_finite() && cond >= 0.0);
    }
}
