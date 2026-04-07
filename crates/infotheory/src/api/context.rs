//! Stateful context and session API surface.

use super::compression::{NcdVariant, try_ncd_bytes_backend};
use super::generation::{GenerationRng, pick_generated_byte, try_generate_rate_backend_chain};
use super::metrics::{
    byte_histogram, joint_marginal_entropy_bytes, marginal_entropy_bytes,
    mutual_information_marg_bytes, ned_marg_bytes, nte_marg_bytes, try_biased_entropy_rate_backend,
    try_cross_entropy_rate_backend, try_entropy_rate_backend, try_joint_entropy_rate_backend,
    try_mutual_information_rate_backend, try_ned_rate_backend, try_nte_rate_backend,
};
use super::types::{
    CompressionBackend, GenerationConfig, GenerationUpdateMode, RateBackend, validate_rate_backend,
};
use crate::aligned_prefix;
#[cfg(feature = "backend-ctw")]
use crate::backends::ctw::{ContextTree, FacContextTree};
#[cfg(feature = "backend-particle")]
use crate::backends::particle::ParticleRuntime;
#[cfg(feature = "backend-zpaq")]
use crate::backends::zpaq_rate::ZpaqRateModel;
use crate::error::{InfotheoryError, InfotheoryResult};
use crate::mixture::OnlineBytePredictor;
#[cfg(any(
    feature = "backend-match",
    feature = "backend-ppmd",
    feature = "backend-sequitur",
    feature = "backend-calibrated"
))]
use crate::try_prequential_rate_backend;
#[cfg(feature = "backend-mamba")]
use crate::with_mamba_method_tls;
#[cfg(feature = "backend-rwkv")]
use crate::with_rwkv_method_tls;

/// Returns the current default information theory context for this thread.
pub fn get_default_ctx() -> InfotheoryCtx {
    crate::get_default_ctx()
}

/// Sets the current default information theory context for this thread.
pub fn set_default_ctx(ctx: InfotheoryCtx) {
    crate::set_default_ctx(ctx);
}

/// Reusable execution context holding default rate and compression backends.
#[derive(Clone, Default)]
pub struct InfotheoryCtx {
    /// Default rate backend for entropy/rate metrics.
    pub rate_backend: RateBackend,
    /// Default compression backend for NCD/compression primitives.
    pub compression_backend: CompressionBackend,
}

/// Stateful rate-backend session for fitting, conditioning, and continuation.
pub struct RateBackendSession {
    predictor: crate::mixture::RateBackendPredictor,
}

impl RateBackendSession {
    /// Create a session from an explicit backend.
    pub fn from_backend(
        backend: RateBackend,
        max_order: i64,
        total_symbols: Option<u64>,
    ) -> InfotheoryResult<Self> {
        validate_rate_backend(&backend)?;
        let mut predictor =
            crate::runtime::build_rate_backend_predictor_default(&backend, max_order)
                .map_err(InfotheoryError::invalid_backend_config)?;
        predictor
            .begin_stream(total_symbols)
            .map_err(InfotheoryError::runtime)?;
        Ok(Self { predictor })
    }

    /// Observe bytes while adapting/fitting the model.
    pub fn observe(&mut self, data: &[u8]) {
        for &byte in data {
            self.predictor.update(byte);
        }
    }

    /// Advance conditioning state without changing fitted parameters/statistics.
    pub fn condition(&mut self, data: &[u8]) {
        for &byte in data {
            self.predictor.update_frozen(byte);
        }
    }

    /// Reset dynamic conditioning state while preserving fitted parameters/statistics.
    pub fn reset_frozen(&mut self, total_symbols: Option<u64>) -> InfotheoryResult<()> {
        self.predictor
            .reset_frozen(total_symbols)
            .map_err(InfotheoryError::runtime)
    }

    /// Fill the 256-way next-byte log-probabilities.
    pub fn fill_log_probs(&mut self, out: &mut [f64; 256]) {
        self.predictor.fill_log_probs(out);
    }

    /// Generate continuation bytes from the current state.
    pub fn generate_bytes(&mut self, bytes: usize, config: GenerationConfig) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(bytes);
        let mut logps = [0.0f64; 256];
        let mut rng = GenerationRng::new(config.seed);

        for _ in 0..bytes {
            match &mut self.predictor {
                #[cfg(feature = "backend-rosa")]
                crate::mixture::RateBackendPredictor::Rosa { .. } => {
                    for (sym, slot) in logps.iter_mut().enumerate() {
                        *slot = self.predictor.log_prob(sym as u8);
                    }
                }
                _ => self.predictor.fill_log_probs(&mut logps),
            }
            let byte = pick_generated_byte(&logps, config, &mut rng);
            match config.update_mode {
                GenerationUpdateMode::Adaptive => self.predictor.update(byte),
                GenerationUpdateMode::Frozen => self.predictor.update_frozen(byte),
            }
            out.push(byte);
        }

        out
    }

    /// Finalize the underlying stream if the backend needs it.
    pub fn finish(&mut self) -> InfotheoryResult<()> {
        self.predictor
            .finish_stream()
            .map_err(InfotheoryError::runtime)
    }
}

impl InfotheoryCtx {
    /// Create a context from explicit rate and compression backends.
    pub fn new(rate_backend: RateBackend, compression_backend: CompressionBackend) -> Self {
        Self {
            rate_backend,
            compression_backend,
        }
    }

    /// Create a context with ROSA+ rate backend and ZPAQ compression backend.
    pub fn with_zpaq(method: impl Into<String>) -> Self {
        Self {
            rate_backend: RateBackend::RosaPlus,
            compression_backend: CompressionBackend::Zpaq {
                method: method.into(),
            },
        }
    }

    /// Compressed length of one byte slice under this context's compressor.
    pub fn try_compress_size(&self, data: &[u8]) -> InfotheoryResult<u64> {
        crate::api::compression::try_compress_size_backend(data, &self.compression_backend)
    }

    /// Compressed length of chained slices under one stream.
    pub fn try_compress_size_chain(&self, parts: &[&[u8]]) -> InfotheoryResult<u64> {
        crate::api::compression::try_compress_size_chain_backend(parts, &self.compression_backend)
    }

    /// Create a stateful session for the active rate backend.
    pub fn rate_backend_session(
        &self,
        max_order: i64,
        total_symbols: Option<u64>,
    ) -> InfotheoryResult<RateBackendSession> {
        RateBackendSession::from_backend(self.rate_backend.clone(), max_order, total_symbols)
    }

    /// Fallible entropy-rate estimate for `data` under this context's rate backend.
    pub fn try_entropy_rate_bytes(&self, data: &[u8], max_order: i64) -> InfotheoryResult<f64> {
        try_entropy_rate_backend(data, max_order, &self.rate_backend)
    }

    /// Fallible biased entropy-rate estimate (plugin variant) for `data`.
    pub fn try_biased_entropy_rate_bytes(
        &self,
        data: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        try_biased_entropy_rate_backend(data, max_order, &self.rate_backend)
    }

    /// Fallible cross entropy of `test_data` under model trained on `train_data`.
    pub fn try_cross_entropy_rate_bytes(
        &self,
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        try_cross_entropy_rate_backend(test_data, train_data, max_order, &self.rate_backend)
    }

    /// Cross entropy with order-0 fast-path fallback when `max_order == 0`.
    pub fn try_cross_entropy_bytes(
        &self,
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        if max_order == 0 {
            if test_data.is_empty() {
                return Ok(0.0);
            }
            let p_x = byte_histogram(test_data);
            let p_y = byte_histogram(train_data);
            let mut h = 0.0f64;
            for i in 0..256 {
                if p_x[i] > 0.0 {
                    let q_y = p_y[i].max(1e-12);
                    h -= p_x[i] * q_y.log2();
                }
            }
            Ok(h)
        } else {
            self.try_cross_entropy_rate_bytes(test_data, train_data, max_order)
        }
    }

    /// Fallible joint entropy-rate estimate `H(X,Y)` under aligned-prefix semantics.
    pub fn try_joint_entropy_rate_bytes(
        &self,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return Ok(0.0);
        }
        try_joint_entropy_rate_backend(x, y, max_order, &self.rate_backend)
    }

    /// Fallible conditional entropy-rate estimate `H(X|Y)`.
    pub fn try_conditional_entropy_rate_bytes(
        &self,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return Ok(0.0);
        }
        let h_xy = self.try_joint_entropy_rate_bytes(x, y, max_order)?;
        let h_y = self.try_entropy_rate_bytes(y, max_order)?;
        Ok((h_xy - h_y).max(0.0))
    }

    /// Fallible `H(data | prefix_parts)` by conditioning the active rate backend
    /// on an explicit prefix chain.
    #[allow(unreachable_patterns)]
    #[cfg_attr(
        not(any(
            feature = "backend-rosa",
            feature = "backend-ctw",
            feature = "backend-match",
            feature = "backend-ppmd",
            feature = "backend-sequitur",
            feature = "backend-mixture",
            feature = "backend-particle",
            feature = "backend-calibrated",
            feature = "backend-zpaq",
            feature = "backend-rwkv",
            feature = "backend-mamba"
        )),
        allow(unused_variables)
    )]
    pub fn try_cross_entropy_conditional_chain(
        &self,
        prefix_parts: &[&[u8]],
        data: &[u8],
    ) -> InfotheoryResult<f64> {
        match &self.rate_backend {
            #[cfg(feature = "backend-rosa")]
            RateBackend::RosaPlus => crate::try_frozen_plugin_rate_backend(
                data,
                prefix_parts,
                -1,
                &RateBackend::RosaPlus,
            ),
            #[cfg(not(feature = "backend-rosa"))]
            RateBackend::RosaPlus => Err(InfotheoryError::invalid_backend_config(
                "backend 'rosaplus' requires infotheory feature 'backend-rosa'".to_string(),
            )),
            #[cfg(any(
                feature = "backend-match",
                feature = "backend-ppmd",
                feature = "backend-sequitur",
                feature = "backend-calibrated"
            ))]
            RateBackend::Match { .. }
            | RateBackend::SparseMatch { .. }
            | RateBackend::Ppmd { .. }
            | RateBackend::Sequitur { .. }
            | RateBackend::Calibrated { .. } => {
                try_prequential_rate_backend(data, prefix_parts, -1, &self.rate_backend)
            }
            #[cfg(not(feature = "backend-match"))]
            RateBackend::Match { .. } | RateBackend::SparseMatch { .. } => {
                Err(InfotheoryError::invalid_backend_config(
                    "backend 'match' requires infotheory feature 'backend-match'".to_string(),
                ))
            }
            #[cfg(not(feature = "backend-ppmd"))]
            RateBackend::Ppmd { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'ppmd' requires infotheory feature 'backend-ppmd'".to_string(),
            )),
            #[cfg(not(feature = "backend-sequitur"))]
            RateBackend::Sequitur { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'sequitur' requires infotheory feature 'backend-sequitur'".to_string(),
            )),
            #[cfg(not(feature = "backend-calibrated"))]
            RateBackend::Calibrated { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'calibrated' requires infotheory feature 'backend-calibrated'".to_string(),
            )),
            #[cfg(feature = "backend-rwkv")]
            RateBackend::Rwkv7Method { method } => with_rwkv_method_tls(method, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .map_err(|e| {
                        InfotheoryError::runtime(format!(
                            "rwkv method conditional-chain scoring failed: {e:#}"
                        ))
                    })
            }),
            #[cfg(feature = "backend-mamba")]
            RateBackend::MambaMethod { method } => with_mamba_method_tls(method, |c| {
                c.cross_entropy_conditional_chain(prefix_parts, data)
                    .map_err(|e| {
                        InfotheoryError::runtime(format!(
                            "mamba method conditional-chain scoring failed: {e:#}"
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
                    for &b in part {
                        for i in (0..8).rev() {
                            tree.update(((b >> i) & 1) == 1);
                        }
                    }
                }
                let log_p_prefix = tree.get_log_block_probability();
                for &b in data {
                    for i in (0..8).rev() {
                        tree.update(((b >> i) & 1) == 1);
                    }
                }
                let log_p_joint = tree.get_log_block_probability();
                let log_p_cond = log_p_joint - log_p_prefix;
                let bits = -log_p_cond / std::f64::consts::LN_2;
                Ok(bits / (data.len() as f64))
            }
            #[cfg(not(feature = "backend-ctw"))]
            RateBackend::Ctw { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'ctw' requires infotheory feature 'backend-ctw'".to_string(),
            )),
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
            RateBackend::Zpaq { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'zpaq' requires infotheory feature 'backend-zpaq'".to_string(),
            )),
            #[cfg(feature = "backend-mixture")]
            RateBackend::Mixture { spec } => {
                if data.is_empty() {
                    return Ok(0.0);
                }
                let experts = spec.build_experts();
                let mut mix = crate::mixture::build_mixture_runtime(spec.as_ref(), &experts)
                    .map_err(|e| {
                        InfotheoryError::invalid_backend_config(format!("MixtureSpec invalid: {e}"))
                    })?;
                let total = prefix_parts
                    .iter()
                    .map(|p| p.len() as u64)
                    .sum::<u64>()
                    .saturating_add(data.len() as u64);
                mix.begin_stream(Some(total)).map_err(|e| {
                    InfotheoryError::runtime(format!("Mixture stream init failed: {e}"))
                })?;
                for &part in prefix_parts {
                    for &b in part {
                        mix.step(b);
                    }
                }
                let mut bits = 0.0;
                for &b in data {
                    bits -= mix.step(b) / std::f64::consts::LN_2;
                }
                Ok(bits / (data.len() as f64))
            }
            #[cfg(not(feature = "backend-mixture"))]
            RateBackend::Mixture { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'mixture' requires infotheory feature 'backend-mixture'".to_string(),
            )),
            #[cfg(feature = "backend-particle")]
            RateBackend::Particle { spec } => {
                if data.is_empty() {
                    return Ok(0.0);
                }
                let mut runtime = ParticleRuntime::new(spec.as_ref());
                for &part in prefix_parts {
                    for &b in part {
                        runtime.step(b);
                    }
                }
                let mut bits = 0.0;
                for &b in data {
                    bits -= runtime.step(b) / std::f64::consts::LN_2;
                }
                Ok(bits / (data.len() as f64))
            }
            #[cfg(not(feature = "backend-particle"))]
            RateBackend::Particle { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'particle' requires infotheory feature 'backend-particle'".to_string(),
            )),
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
                    for &b in part {
                        for i in 0..bits_per_byte {
                            fac.update(((b >> i) & 1) == 1, i);
                        }
                    }
                }
                let log_p_prefix = fac.get_log_block_probability();
                for &b in data {
                    for i in 0..bits_per_byte {
                        fac.update(((b >> i) & 1) == 1, i);
                    }
                }
                let log_p_joint = fac.get_log_block_probability();
                let log_p_cond = log_p_joint - log_p_prefix;
                let bits = -log_p_cond / std::f64::consts::LN_2;
                Ok(bits / (data.len() as f64))
            }
            #[cfg(not(feature = "backend-ctw"))]
            RateBackend::FacCtw { .. } => Err(InfotheoryError::invalid_backend_config(
                "backend 'fac-ctw' requires infotheory feature 'backend-ctw'".to_string(),
            )),
        }
    }

    /// Generate a continuation from `prompt` with [`GenerationConfig::default()`].
    pub fn try_generate_bytes(
        &self,
        prompt: &[u8],
        bytes: usize,
        max_order: i64,
    ) -> InfotheoryResult<Vec<u8>> {
        self.try_generate_bytes_with_config(prompt, bytes, max_order, GenerationConfig::default())
    }

    /// Fallible continuation generation from `prompt` using an explicit config.
    pub fn try_generate_bytes_with_config(
        &self,
        prompt: &[u8],
        bytes: usize,
        max_order: i64,
        config: GenerationConfig,
    ) -> InfotheoryResult<Vec<u8>> {
        try_generate_rate_backend_chain(&[prompt], bytes, max_order, &self.rate_backend, config)
    }

    /// Generate a continuation after conditioning on an explicit chain of prefix parts.
    pub fn try_generate_bytes_conditional_chain(
        &self,
        prefix_parts: &[&[u8]],
        bytes: usize,
        max_order: i64,
    ) -> InfotheoryResult<Vec<u8>> {
        self.try_generate_bytes_conditional_chain_with_config(
            prefix_parts,
            bytes,
            max_order,
            GenerationConfig::default(),
        )
    }

    /// Fallible continuation generation after conditioning on an explicit chain of prefix parts.
    pub fn try_generate_bytes_conditional_chain_with_config(
        &self,
        prefix_parts: &[&[u8]],
        bytes: usize,
        max_order: i64,
        config: GenerationConfig,
    ) -> InfotheoryResult<Vec<u8>> {
        try_generate_rate_backend_chain(prefix_parts, bytes, max_order, &self.rate_backend, config)
    }

    /// NCD between byte slices using this context's compression backend.
    pub fn try_ncd_bytes(&self, x: &[u8], y: &[u8], variant: NcdVariant) -> InfotheoryResult<f64> {
        try_ncd_bytes_backend(x, y, &self.compression_backend, variant)
    }

    /// Rate-backend mutual information estimate.
    pub fn try_mutual_information_rate_bytes(
        &self,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        try_mutual_information_rate_backend(x, y, max_order, &self.rate_backend)
    }

    /// Mutual information with `max_order == 0` marginal fast-path.
    pub fn try_mutual_information_bytes(
        &self,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        if max_order == 0 {
            Ok(mutual_information_marg_bytes(x, y))
        } else {
            self.try_mutual_information_rate_bytes(x, y, max_order)
        }
    }

    /// Conditional entropy with aligned-prefix semantics.
    pub fn try_conditional_entropy_bytes(
        &self,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        if max_order == 0 {
            let h_xy = joint_marginal_entropy_bytes(x, y);
            let h_y = marginal_entropy_bytes(y);
            Ok((h_xy - h_y).max(0.0))
        } else {
            let h_xy = self.try_joint_entropy_rate_bytes(x, y, max_order)?;
            let h_y = self.try_entropy_rate_bytes(y, max_order)?;
            Ok((h_xy - h_y).max(0.0))
        }
    }

    /// Normalized entropy distance (NED) under this context.
    pub fn try_ned_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
        if max_order == 0 {
            Ok(ned_marg_bytes(x, y))
        } else {
            try_ned_rate_backend(x, y, max_order, &self.rate_backend)
        }
    }

    /// Conservative NED normalization variant.
    pub fn try_ned_cons_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        let (h_x, h_y, h_xy) = if max_order == 0 {
            (
                marginal_entropy_bytes(x),
                marginal_entropy_bytes(y),
                joint_marginal_entropy_bytes(x, y),
            )
        } else {
            (
                self.try_entropy_rate_bytes(x, max_order)?,
                self.try_entropy_rate_bytes(y, max_order)?,
                self.try_joint_entropy_rate_bytes(x, y, max_order)?,
            )
        };
        let min_h = h_x.min(h_y);
        if h_xy == 0.0 {
            Ok(0.0)
        } else {
            Ok(((h_xy - min_h) / h_xy).clamp(0.0, 1.0))
        }
    }

    /// Normalized transform effort (NTE) under this context.
    pub fn try_nte_bytes(&self, x: &[u8], y: &[u8], max_order: i64) -> InfotheoryResult<f64> {
        if max_order == 0 {
            Ok(nte_marg_bytes(x, y))
        } else {
            try_nte_rate_backend(x, y, max_order, &self.rate_backend)
        }
    }

    /// Intrinsic dependence score in `[0,1]`.
    pub fn try_intrinsic_dependence_bytes(
        &self,
        data: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        let h_marginal = marginal_entropy_bytes(data);
        if h_marginal < 1e-9 {
            return Ok(0.0);
        }
        let h_rate = self.try_entropy_rate_bytes(data, max_order)?;
        Ok(((h_marginal - h_rate) / h_marginal).clamp(0.0, 1.0))
    }

    /// Resistance-to-transformation ratio `I(X;T(X))/H(X)` in `[0,1]`.
    pub fn try_resistance_to_transformation_bytes(
        &self,
        x: &[u8],
        tx: &[u8],
        max_order: i64,
    ) -> InfotheoryResult<f64> {
        let (x, tx) = aligned_prefix(x, tx);
        let h_x = if max_order == 0 {
            marginal_entropy_bytes(x)
        } else {
            self.try_entropy_rate_bytes(x, max_order)?
        };
        if h_x < 1e-9 {
            return Ok(0.0);
        }
        let mi = self.try_mutual_information_bytes(x, tx, max_order)?;
        Ok((mi / h_x).clamp(0.0, 1.0))
    }
}
