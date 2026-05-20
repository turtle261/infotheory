//! Stateful context and session API surface.

use super::compression::{NcdVariant, try_ncd_bytes_backend};
use super::generation::{GenerationRng, pick_generated_byte, try_generate_rate_backend_chain};
use super::metrics::{
    empirical_entropy_bytes, try_biased_entropy_rate_backend, try_cross_entropy_rate_backend,
    try_entropy_rate_backend, try_joint_entropy_rate_backend, try_mutual_information_rate_backend,
    try_ned_rate_backend, try_nte_rate_backend,
};
use super::types::{CompressionBackend, GenerationConfig, GenerationUpdateMode, RateBackend};
use crate::aligned_prefix;
use crate::error::{InfotheoryError, InfotheoryResult};
use crate::mixture::OnlineBytePredictor;
use crate::prediction::{
    BinaryPrediction, BitOrder, BitStreamSemantics, BytePrefixMass,
    binary_prediction_from_log_probs,
};
use crate::spec::{CompiledCompressionBackend, CompiledRateBackend};

/// Returns the current default information theory context for this thread.
pub fn get_default_ctx() -> InfotheoryResult<InfotheoryCtx> {
    crate::get_default_ctx()
}

/// Sets the current default information theory context for this thread.
pub fn set_default_ctx(ctx: InfotheoryCtx) {
    crate::set_default_ctx(ctx);
}

/// Reusable execution context holding default rate and compression backends.
#[derive(Clone)]
pub struct InfotheoryCtx {
    /// Default rate backend for entropy/rate metrics.
    pub rate_backend: CompiledRateBackend,
    /// Default compression backend for NCD/compression primitives.
    pub compression_backend: CompiledCompressionBackend,
}

/// Stateful rate-backend session for fitting, conditioning, and continuation.
pub struct RateBackendSession {
    predictor: crate::mixture::RateBackendPredictor,
}

/// Stateful bit-level session over a rate backend.
///
/// Byte-packed sessions keep the underlying backend byte-native and expose it
/// through a lazy prefix-mass view. They therefore require whole-byte stream
/// boundaries: `total_bits`, when provided, must be a multiple of `8`, and
/// `finish` must not leave a dangling partial byte. Binary-token sessions model
/// each bit as the literal byte symbol `0` or `1`, so they support arbitrary
/// bit lengths directly.
pub struct RateBackendBitSession {
    predictor: crate::mixture::RateBackendPredictor,
    semantics: BitStreamSemantics,
    prefix: Option<BytePrefixMass>,
}

fn byte_packed_total_symbols(total_bits: Option<u64>) -> Result<Option<u64>, String> {
    let Some(total_bits) = total_bits else {
        return Ok(None);
    };
    if total_bits % 8 != 0 {
        return Err(format!(
            "byte-packed bit streams require a whole number of bytes; got {total_bits} bits. Use BitStreamSemantics::BinaryTokens for arbitrary-length bit streams"
        ));
    }
    Ok(Some(total_bits / 8))
}

fn total_symbols_for_bit_semantics(
    total_bits: Option<u64>,
    semantics: BitStreamSemantics,
) -> Result<Option<u64>, String> {
    match semantics {
        BitStreamSemantics::BytePacked { .. } => byte_packed_total_symbols(total_bits),
        BitStreamSemantics::BinaryTokens => Ok(total_bits),
    }
}

impl RateBackendBitSession {
    /// Create a bit session from an explicit compiled backend.
    ///
    /// For [`BitStreamSemantics::BytePacked`], `total_bits` must be `None` or a
    /// multiple of `8`.
    pub fn from_backend(
        backend: CompiledRateBackend,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
    ) -> InfotheoryResult<Self> {
        let backend = if matches!(semantics, BitStreamSemantics::BinaryTokens)
            && backend.supports_bit_token_adaptation()
        {
            backend
                .adapt_for_bit_tokens()
                .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?
        } else {
            backend
        };
        let total_symbols = total_symbols_for_bit_semantics(total_bits, semantics)
            .map_err(InfotheoryError::runtime)?;
        let mut predictor = crate::runtime::build_rate_backend_predictor_default(&backend)
            .map_err(InfotheoryError::invalid_backend_config)?;
        predictor
            .begin_stream(total_symbols)
            .map_err(InfotheoryError::runtime)?;
        Ok(Self {
            predictor,
            semantics,
            prefix: None,
        })
    }

    /// Create a bit session from a wrapper backend spec.
    ///
    /// For [`BitStreamSemantics::BytePacked`], `total_bits` must be `None` or a
    /// multiple of `8`.
    pub fn from_spec(
        backend: RateBackend,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
    ) -> InfotheoryResult<Self> {
        let compiled = backend
            .compile()
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        Self::from_backend(compiled, total_bits, semantics)
    }

    /// Predict the next bit without updating state.
    pub fn predict_bit(&mut self) -> BinaryPrediction {
        match self.semantics {
            BitStreamSemantics::BinaryTokens => binary_prediction_from_log_probs(
                self.predictor.log_prob(0),
                self.predictor.log_prob(1),
                crate::mixture::DEFAULT_MIN_PROB,
            ),
            BitStreamSemantics::BytePacked { order } => {
                self.ensure_prefix(order);
                self.prefix
                    .as_ref()
                    .expect("prefix initialized")
                    .prediction()
            }
        }
    }

    /// Convenience prediction for `P(bit = 1)`.
    pub fn predict_one(&mut self) -> f64 {
        self.predict_bit().p1
    }

    /// Predict and then observe one adaptive/fitting bit.
    pub fn step_bit(&mut self, bit: bool) -> BinaryPrediction {
        let prediction = self.predict_bit();
        self.observe_bit(bit);
        prediction
    }

    /// Observe one adaptive/fitting bit.
    pub fn observe_bit(&mut self, bit: bool) {
        match self.semantics {
            BitStreamSemantics::BinaryTokens => self.predictor.update(u8::from(bit)),
            BitStreamSemantics::BytePacked { order } => {
                self.ensure_prefix(order);
                let prefix = self.prefix.as_mut().expect("prefix initialized");
                prefix.observe(bit);
                if prefix.is_complete() {
                    let symbol = prefix.symbol();
                    self.predictor.update(symbol);
                    self.prefix = None;
                }
            }
        }
    }

    /// Advance conditioning state with one bit without fitting/adapting.
    pub fn condition_bit(&mut self, bit: bool) {
        match self.semantics {
            BitStreamSemantics::BinaryTokens => self.predictor.update_frozen(u8::from(bit)),
            BitStreamSemantics::BytePacked { order } => {
                self.ensure_prefix(order);
                let prefix = self.prefix.as_mut().expect("prefix initialized");
                prefix.observe(bit);
                if prefix.is_complete() {
                    let symbol = prefix.symbol();
                    self.predictor.update_frozen(symbol);
                    self.prefix = None;
                }
            }
        }
    }

    /// Reset dynamic conditioning state while preserving fitted parameters/statistics.
    pub fn reset_frozen(&mut self, total_bits: Option<u64>) -> InfotheoryResult<()> {
        self.prefix = None;
        let total_symbols = total_symbols_for_bit_semantics(total_bits, self.semantics)
            .map_err(InfotheoryError::runtime)?;
        self.predictor
            .reset_frozen(total_symbols)
            .map_err(InfotheoryError::runtime)
    }

    /// Finalize the underlying stream if the backend needs it.
    pub fn finish(&mut self) -> InfotheoryResult<()> {
        if matches!(self.semantics, BitStreamSemantics::BytePacked { .. })
            && self.prefix.is_some()
        {
            return Err(InfotheoryError::runtime(
                "byte-packed bit streams must finish on a whole-byte boundary; use BitStreamSemantics::BinaryTokens for arbitrary-length bit streams",
            ));
        }
        self.predictor
            .finish_stream()
            .map_err(InfotheoryError::runtime)
    }

    fn ensure_prefix(&mut self, order: BitOrder) {
        if self.prefix.is_some() {
            return;
        }
        let mut logps = [0.0f64; 256];
        self.predictor.fill_log_probs(&mut logps);
        let mut pdf = [0.0f64; 256];
        for (dst, &lp) in pdf.iter_mut().zip(logps.iter()) {
            *dst = if lp.is_finite() { lp.exp() } else { 0.0 };
        }
        self.prefix = Some(BytePrefixMass::from_pdf(&pdf, order));
    }
}

impl crate::prediction::OnlineBitPredictor for RateBackendBitSession {
    fn begin_bit_stream(
        &mut self,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
    ) -> Result<(), String> {
        if semantics != self.semantics {
            return Err(
                "bit stream semantics are fixed for a RateBackendBitSession; create a new session"
                    .to_string(),
            );
        }
        self.reset_frozen(total_bits).map_err(|err| err.to_string())
    }

    fn finish_bit_stream(&mut self) -> Result<(), String> {
        self.finish().map_err(|err| err.to_string())
    }

    fn bit_prediction(&mut self) -> BinaryPrediction {
        self.predict_bit()
    }

    fn update_bit(&mut self, bit: bool) {
        self.observe_bit(bit);
    }

    fn update_bit_frozen(&mut self, bit: bool) {
        self.condition_bit(bit);
    }
}

impl RateBackendSession {
    /// Create a session from an explicit backend.
    ///
    /// Algorithmic configuration (such as ROSA's `max_order`) lives inside the
    /// backend's variant; the session does not take it as an argument.
    pub fn from_backend(
        backend: CompiledRateBackend,
        total_symbols: Option<u64>,
    ) -> InfotheoryResult<Self> {
        let mut predictor = crate::runtime::build_rate_backend_predictor_default(&backend)
            .map_err(InfotheoryError::invalid_backend_config)?;
        predictor
            .begin_stream(total_symbols)
            .map_err(InfotheoryError::runtime)?;
        Ok(Self { predictor })
    }

    /// Create a session from a wrapper backend spec.
    pub fn from_spec(backend: RateBackend, total_symbols: Option<u64>) -> InfotheoryResult<Self> {
        let compiled = backend
            .compile()
            .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?;
        Self::from_backend(compiled, total_symbols)
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
    /// Create the current build's implicit default context.
    pub fn try_default() -> InfotheoryResult<Self> {
        Self::from_specs(
            RateBackend::try_default()?,
            CompressionBackend::try_default()?,
        )
    }

    /// Create a context from explicit rate and compression backends.
    pub fn new(
        rate_backend: CompiledRateBackend,
        compression_backend: CompiledCompressionBackend,
    ) -> Self {
        Self {
            rate_backend,
            compression_backend,
        }
    }

    /// Create a context from wrapper backend specs.
    pub fn from_specs(
        rate_backend: RateBackend,
        compression_backend: CompressionBackend,
    ) -> InfotheoryResult<Self> {
        Ok(Self {
            rate_backend: rate_backend
                .compile()
                .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?,
            compression_backend: compression_backend
                .compile()
                .map_err(|err| InfotheoryError::invalid_backend_config(err.to_string()))?,
        })
    }

    /// Create a context with adaptive ROSA+ rate backend and ZPAQ compression backend.
    pub fn try_with_zpaq(method: impl Into<crate::api::ZpaqMethodSpec>) -> InfotheoryResult<Self> {
        Self::from_specs(
            RateBackend::RosaPlus { max_order: -1 },
            CompressionBackend::zpaq(method),
        )
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
        total_symbols: Option<u64>,
    ) -> InfotheoryResult<RateBackendSession> {
        RateBackendSession::from_backend(self.rate_backend.clone(), total_symbols)
    }

    /// Create a stateful bit-level session for the active rate backend.
    pub fn rate_backend_bit_session(
        &self,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
    ) -> InfotheoryResult<RateBackendBitSession> {
        RateBackendBitSession::from_backend(self.rate_backend.clone(), total_bits, semantics)
    }

    /// Fallible entropy-rate estimate for `data` under this context's rate backend.
    pub fn try_entropy_rate_bytes(&self, data: &[u8]) -> InfotheoryResult<f64> {
        try_entropy_rate_backend(data, &self.rate_backend)
    }

    /// Fallible biased entropy-rate estimate (plugin variant) for `data`.
    pub fn try_biased_entropy_rate_bytes(&self, data: &[u8]) -> InfotheoryResult<f64> {
        try_biased_entropy_rate_backend(data, &self.rate_backend)
    }

    /// Fallible cross entropy of `test_data` under model trained on `train_data`.
    pub fn try_cross_entropy_rate_bytes(
        &self,
        test_data: &[u8],
        train_data: &[u8],
    ) -> InfotheoryResult<f64> {
        try_cross_entropy_rate_backend(test_data, train_data, &self.rate_backend)
    }

    /// Cross entropy under the active rate backend.
    pub fn try_cross_entropy_bytes(
        &self,
        test_data: &[u8],
        train_data: &[u8],
    ) -> InfotheoryResult<f64> {
        self.try_cross_entropy_rate_bytes(test_data, train_data)
    }

    /// Fallible joint entropy-rate estimate `H(X,Y)` under aligned-prefix semantics.
    pub fn try_joint_entropy_rate_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return Ok(0.0);
        }
        try_joint_entropy_rate_backend(x, y, &self.rate_backend)
    }

    /// Fallible conditional entropy-rate estimate `H(X|Y)`.
    pub fn try_conditional_entropy_rate_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        if x.is_empty() {
            return Ok(0.0);
        }
        let h_xy = self.try_joint_entropy_rate_bytes(x, y)?;
        let h_y = self.try_entropy_rate_bytes(y)?;
        Ok((h_xy - h_y).max(0.0))
    }

    /// Fallible `H(data | prefix_parts)` by conditioning the active rate backend
    /// on an explicit prefix chain.
    pub fn try_cross_entropy_conditional_chain(
        &self,
        prefix_parts: &[&[u8]],
        data: &[u8],
    ) -> InfotheoryResult<f64> {
        crate::runtime::try_cross_entropy_conditional_chain_backend(
            prefix_parts,
            data,
            &self.rate_backend,
        )
    }

    /// Generate a continuation from `prompt` with [`GenerationConfig::default()`].
    pub fn try_generate_bytes(&self, prompt: &[u8], bytes: usize) -> InfotheoryResult<Vec<u8>> {
        self.try_generate_bytes_with_config(prompt, bytes, GenerationConfig::default())
    }

    /// Fallible continuation generation from `prompt` using an explicit config.
    pub fn try_generate_bytes_with_config(
        &self,
        prompt: &[u8],
        bytes: usize,
        config: GenerationConfig,
    ) -> InfotheoryResult<Vec<u8>> {
        try_generate_rate_backend_chain(&[prompt], bytes, &self.rate_backend, config)
    }

    /// Generate a continuation after conditioning on an explicit chain of prefix parts.
    pub fn try_generate_bytes_conditional_chain(
        &self,
        prefix_parts: &[&[u8]],
        bytes: usize,
    ) -> InfotheoryResult<Vec<u8>> {
        self.try_generate_bytes_conditional_chain_with_config(
            prefix_parts,
            bytes,
            GenerationConfig::default(),
        )
    }

    /// Fallible continuation generation after conditioning on an explicit chain of prefix parts.
    pub fn try_generate_bytes_conditional_chain_with_config(
        &self,
        prefix_parts: &[&[u8]],
        bytes: usize,
        config: GenerationConfig,
    ) -> InfotheoryResult<Vec<u8>> {
        try_generate_rate_backend_chain(prefix_parts, bytes, &self.rate_backend, config)
    }

    /// NCD between byte slices using this context's compression backend.
    pub fn try_ncd_bytes(&self, x: &[u8], y: &[u8], variant: NcdVariant) -> InfotheoryResult<f64> {
        try_ncd_bytes_backend(x, y, &self.compression_backend, variant)
    }

    /// Rate-backend mutual information estimate.
    pub fn try_mutual_information_rate_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        try_mutual_information_rate_backend(x, y, &self.rate_backend)
    }

    /// Mutual information under the active rate backend.
    pub fn try_mutual_information_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        self.try_mutual_information_rate_bytes(x, y)
    }

    /// Conditional entropy `H(X|Y)` under the active rate backend.
    pub fn try_conditional_entropy_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        let h_xy = self.try_joint_entropy_rate_bytes(x, y)?;
        let h_y = self.try_entropy_rate_bytes(y)?;
        Ok((h_xy - h_y).max(0.0))
    }

    /// Normalized entropy distance (NED) under this context's rate backend.
    pub fn try_ned_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        try_ned_rate_backend(x, y, &self.rate_backend)
    }

    /// Conservative NED normalization variant under this context's rate backend.
    pub fn try_ned_cons_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        let (x, y) = aligned_prefix(x, y);
        let h_x = self.try_entropy_rate_bytes(x)?;
        let h_y = self.try_entropy_rate_bytes(y)?;
        let h_xy = self.try_joint_entropy_rate_bytes(x, y)?;
        let min_h = h_x.min(h_y);
        if h_xy == 0.0 {
            Ok(0.0)
        } else {
            Ok(((h_xy - min_h) / h_xy).clamp(0.0, 1.0))
        }
    }

    /// Normalized transform effort (NTE) under this context's rate backend.
    pub fn try_nte_bytes(&self, x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
        try_nte_rate_backend(x, y, &self.rate_backend)
    }

    /// Intrinsic dependence score in `[0,1]` driven by `(H₀(X) - Ĥ(X)) / H₀(X)`,
    /// where `H₀` is the order-0 / empirical entropy and `Ĥ` is the entropy rate
    /// produced by this context's rate backend.
    pub fn try_intrinsic_dependence_bytes(&self, data: &[u8]) -> InfotheoryResult<f64> {
        let h_empirical = empirical_entropy_bytes(data);
        if h_empirical < 1e-9 {
            return Ok(0.0);
        }
        let h_rate = self.try_entropy_rate_bytes(data)?;
        Ok(((h_empirical - h_rate) / h_empirical).clamp(0.0, 1.0))
    }

    /// Resistance-to-transformation ratio `I(X;T(X))/H(X)` in `[0,1]` under this context's rate backend.
    pub fn try_resistance_to_transformation_bytes(
        &self,
        x: &[u8],
        tx: &[u8],
    ) -> InfotheoryResult<f64> {
        let (x, tx) = aligned_prefix(x, tx);
        let h_x = self.try_entropy_rate_bytes(x)?;
        if h_x < 1e-9 {
            return Ok(0.0);
        }
        let mi = self.try_mutual_information_bytes(x, tx)?;
        Ok((mi / h_x).clamp(0.0, 1.0))
    }
}
