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
use crate::mixture::{OnlineBytePredictor, RateBackendPredictorCheckpoint};
use crate::prediction::{
    BinaryPrediction, BitOrder, BitStreamSemantics, BytePrefixMass,
    binary_prediction_from_log_probs,
};
use crate::spec::{CanonicalBytes, CompiledCompressionBackend, CompiledRateBackend};

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
/// `finish` must not leave a dangling partial byte. Within one buffered byte,
/// callers must also stay within either adaptive updates or conditioning-only
/// updates; switching modes mid-byte is rejected because the backend only
/// commits whole-byte symbols. Binary-token sessions model each bit either
/// through the backend's native binary-token application or, for byte-native
/// backends, by adapting the predictor to the literal byte symbols `0` and `1`
/// and renormalizing those two choices.
///
/// The complete checkpoint/restore contract is available in both Rust and
/// Python (`infotheory_rs.RateBackendBitSession` with
/// `RateBackendBitSessionCheckpoint`).
#[derive(Clone)]
pub struct RateBackendBitSession {
    backend_code: CanonicalBytes,
    predictor: crate::mixture::RateBackendPredictor,
    semantics: BitStreamSemantics,
    min_prob: f64,
    prefix: Option<BufferedBytePrefix>,
    discardable_scopes: usize,
}

/// Opaque checkpoint for restoring a [`RateBackendBitSession`].
///
/// Checkpoints capture the underlying rate predictor plus any in-flight
/// byte-prefix state, so they are valid even between byte-packed bits before a
/// full byte has been committed to the backend.
///
/// Snapshot-backed predictors restore their predictive state exactly. Compact
/// journaled predictors may replay reversible markers during restore, so their
/// floating-point probabilities are restored up to normal round-off while the
/// discrete model state and stream position are restored to the checkpoint.
///
/// (Python exposes this as `infotheory_rs.RateBackendBitSessionCheckpoint`.)
#[derive(Clone)]
pub struct RateBackendBitSessionCheckpoint {
    backend_code: CanonicalBytes,
    predictor: crate::mixture::RateBackendPredictorCheckpoint,
    semantics: BitStreamSemantics,
    prefix: Option<BufferedBytePrefix>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BufferedByteUpdateMode {
    Adaptive,
    Frozen,
}

impl BufferedByteUpdateMode {
    fn verb(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Frozen => "conditioning-only",
        }
    }
}

#[derive(Clone)]
struct BufferedBytePrefix {
    kind: BufferedBytePrefixKind,
    update_mode: Option<BufferedByteUpdateMode>,
}

#[derive(Clone)]
// `BytePrefixMass` is the resident prefix tree for the byte-packed path and is
// intentionally inline. Native-MSB start checkpoints are boxed below so the
// small native state does not inline a full predictor checkpoint.
#[allow(clippy::large_enum_variant)]
enum BufferedBytePrefixKind {
    Mass(BytePrefixMass),
    NativeMsb {
        // Boxed so partial-byte abort/replay stores a checkpoint out-of-line;
        // the checkpoint enum itself is pointer-sized (Full holds
        // Box<RateBackendPredictor>). Allocation is paid only when a native-MSB
        // prefix needs rollback across lifecycle reset or frozen byte replay.
        start_checkpoint: Option<Box<RateBackendPredictorCheckpoint>>,
        symbol: u8,
        bits: usize,
    },
}

impl BufferedBytePrefix {
    fn new_mass(mass: BytePrefixMass) -> Self {
        Self {
            kind: BufferedBytePrefixKind::Mass(mass),
            update_mode: None,
        }
    }

    fn new_native_msb(start_checkpoint: Option<RateBackendPredictorCheckpoint>) -> Self {
        Self {
            kind: BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: start_checkpoint.map(Box::new),
                symbol: 0,
                bits: 0,
            },
            update_mode: None,
        }
    }

    fn has_partial_bits(&self) -> bool {
        match &self.kind {
            BufferedBytePrefixKind::Mass(mass) => mass.has_partial_bits(),
            BufferedBytePrefixKind::NativeMsb { bits, .. } => *bits > 0,
        }
    }

    fn record_mode(&mut self, requested: BufferedByteUpdateMode) -> InfotheoryResult<()> {
        if let Some(active) = self.update_mode
            && self.has_partial_bits()
            && active != requested
        {
            return Err(InfotheoryError::runtime(format!(
                "byte-packed bit sessions cannot mix {} and {} updates within the same buffered byte; finish the byte with one mode, use `try_observe_bit`/`try_condition_bit` to handle this error explicitly, or switch to BitStreamSemantics::BinaryTokens for mid-byte mode changes",
                active.verb(),
                requested.verb(),
            )));
        }
        self.update_mode = Some(requested);
        Ok(())
    }
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
    fn observed_native_prefix_bit(symbol: u8, bit_idx: usize) -> bool {
        (symbol & (1u8 << (7 - bit_idx))) != 0
    }

    /// Restore to `start`, abort empty native MSB prefix, then unconditionally discard `start`.
    ///
    /// Discard only adjusts journal depth; predictor state is already at `start` after restore.
    fn restore_start_checkpoint_abort_and_discard(
        &mut self,
        start: RateBackendPredictorCheckpoint,
    ) -> Result<bool, String> {
        self.predictor.restore_checkpoint(&start);
        let abort_res = self.predictor.abort_empty_native_msb_byte_prefix();
        self.predictor.discard_checkpoint(start);
        abort_res
    }

    fn release_inflight_prefix_checkpoint_for_restore(&mut self) {
        let Some(prefix) = self.prefix.take() else {
            return;
        };
        if let BufferedBytePrefixKind::NativeMsb {
            start_checkpoint: Some(checkpoint),
            ..
        } = prefix.kind
        {
            self.predictor.discard_checkpoint(*checkpoint);
        }
    }

    fn discard_inflight_prefix_checkpoint(&mut self) {
        let Some(prefix) = self.prefix.take() else {
            return;
        };
        match prefix.kind {
            BufferedBytePrefixKind::Mass(_) => {}
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: Some(checkpoint),
                ..
            } => {
                self.restore_start_checkpoint_abort_and_discard(*checkpoint)
                    .expect("native MSB prefix start checkpoint must be empty");
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                bits,
                ..
            } => {
                if bits == 0 {
                    self.predictor
                        .abort_empty_native_msb_byte_prefix()
                        .expect("empty native MSB prefix abort must succeed");
                } else {
                    panic!(
                        "discarding an adaptive native MSB byte-prefix with observed bits; \
                         missing prefix start checkpoint"
                    );
                }
            }
        }
    }

    fn abandon_inflight_prefix_for_lifecycle_reset(&mut self) {
        let Some(prefix) = self.prefix.take() else {
            return;
        };
        match prefix.kind {
            BufferedBytePrefixKind::Mass(_) => {}
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: Some(checkpoint),
                ..
            } => {
                self.restore_start_checkpoint_abort_and_discard(*checkpoint)
                    .expect("native MSB prefix start checkpoint must be empty");
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                bits: 0,
                ..
            } => {
                self.predictor
                    .abort_empty_native_msb_byte_prefix()
                    .expect("empty native MSB prefix abort must succeed");
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                bits: _,
                ..
            } => {
                self.predictor
                    .abandon_incomplete_native_msb_byte_prefix_for_lifecycle()
                    .expect("native MSB prefix lifecycle abandonment must succeed");
            }
        }
    }

    /// Make clearing checkpoint journals preserve the byte-prefix invariant:
    /// `NativeMsb` session state exists only while the predictor has an active
    /// native MSB prefix. Partial native prefixes are downgraded to the generic
    /// mass prefix before clearing so their rollback checkpoint can be released.
    fn normalize_prefix_for_checkpoint_clear(&mut self) -> InfotheoryResult<bool> {
        let Some(prefix) = self.prefix.take() else {
            return Ok(true);
        };
        let update_mode = prefix.update_mode;
        match prefix.kind {
            BufferedBytePrefixKind::Mass(mass) => {
                self.prefix = Some(BufferedBytePrefix {
                    kind: BufferedBytePrefixKind::Mass(mass),
                    update_mode,
                });
                Ok(true)
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                bits: 0,
                ..
            } => {
                self.predictor
                    .abort_empty_native_msb_byte_prefix()
                    .map_err(InfotheoryError::runtime)?;
                Ok(true)
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: Some(checkpoint),
                symbol,
                bits,
            } => {
                let checkpoint = *checkpoint;
                self.restore_start_checkpoint_abort_and_discard(checkpoint)
                    .map_err(InfotheoryError::runtime)?;

                if bits == 0 {
                    return Ok(true);
                }

                let mut logps = [0.0f64; 256];
                match update_mode.unwrap_or(BufferedByteUpdateMode::Adaptive) {
                    BufferedByteUpdateMode::Adaptive => self.predictor.fill_log_probs(&mut logps),
                    BufferedByteUpdateMode::Frozen => {
                        self.predictor.fill_log_probs_frozen(&mut logps);
                    }
                }

                let mut mass = BytePrefixMass::from_log_probs(&logps, BitOrder::MsbFirst);
                for bit_idx in 0..bits {
                    mass.observe(Self::observed_native_prefix_bit(symbol, bit_idx));
                }
                self.prefix = Some(BufferedBytePrefix {
                    kind: BufferedBytePrefixKind::Mass(mass),
                    update_mode,
                });
                Ok(true)
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                symbol,
                bits,
            } => {
                self.prefix = Some(BufferedBytePrefix {
                    kind: BufferedBytePrefixKind::NativeMsb {
                        start_checkpoint: None,
                        symbol,
                        bits,
                    },
                    update_mode,
                });
                Ok(false)
            }
        }
    }

    /// Create a bit session from an explicit compiled backend.
    ///
    /// For [`BitStreamSemantics::BytePacked`], `total_bits` must be `None` or a
    /// multiple of `8`.
    pub fn from_backend(
        backend: CompiledRateBackend,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
    ) -> InfotheoryResult<Self> {
        Self::from_backend_with_min_prob(
            backend,
            total_bits,
            semantics,
            crate::mixture::DEFAULT_MIN_PROB,
        )
    }

    pub(crate) fn from_backend_with_min_prob(
        backend: CompiledRateBackend,
        total_bits: Option<u64>,
        semantics: BitStreamSemantics,
        min_prob: f64,
    ) -> InfotheoryResult<Self> {
        let total_symbols = total_symbols_for_bit_semantics(total_bits, semantics)
            .map_err(InfotheoryError::runtime)?;
        let mut predictor = match semantics {
            BitStreamSemantics::BinaryTokens => {
                crate::runtime::build_rate_backend_binary_token_predictor(&backend, min_prob)
            }
            BitStreamSemantics::BytePacked { .. } => {
                if !backend.supports_byte_prefix_mass() {
                    return Err(InfotheoryError::invalid_backend_config(format!(
                        "backend '{}' does not support BitStreamSemantics::BytePacked",
                        backend.canonical_name()
                    )));
                }
                if !backend.supports_efficient_byte_packed_bit_sessions() {
                    return Err(InfotheoryError::invalid_backend_config(format!(
                        "backend '{}' can expose byte probabilities but does not support efficient BitStreamSemantics::BytePacked sessions; use BitStreamSemantics::BinaryTokens or a backend with native or cached byte-prefix support",
                        backend.canonical_name()
                    )));
                }
                crate::runtime::build_rate_backend_predictor(&backend, min_prob)
            }
        }
        .map_err(InfotheoryError::invalid_backend_config)?;
        predictor
            .begin_stream(total_symbols)
            .map_err(InfotheoryError::runtime)?;
        let backend_code = backend.canonical_bytes().clone();
        Ok(Self {
            backend_code,
            predictor,
            semantics,
            min_prob,
            prefix: None,
            discardable_scopes: 0,
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
                self.min_prob,
            ),
            BitStreamSemantics::BytePacked { order } => {
                if self.prefix.is_none()
                    && order == BitOrder::MsbFirst
                    && self.predictor.has_native_msb_byte_prefix()
                {
                    self.try_begin_native_msb_prefix(None).unwrap_or_else(|err| {
                        panic!(
                            "RateBackendPredictor failed to begin native MSB byte-prefix: {err} \
                             (contract violation; BytePrefixMass fallback is not safe on Err)"
                        )
                    });
                }
                self.ensure_mass_prefix(order, BufferedByteUpdateMode::Adaptive);
                let native_bits = match &self.prefix.as_ref().expect("prefix initialized").kind {
                    BufferedBytePrefixKind::Mass(mass) => return mass.prediction(),
                    BufferedBytePrefixKind::NativeMsb { bits, .. } => *bits,
                };
                let p1 = self
                    .predictor
                    .native_msb_prefix_prob_one(native_bits)
                    .expect(
                        "native_msb_prefix_prob_one failed or returned error for active NativeMsb \
                         prefix (RateBackendBitSession invariant; predictors must uphold finite \
                         contract or report via Result consistently)",
                    );
                BinaryPrediction::from_prob_one(p1, self.min_prob)
            }
        }
    }

    /// Convenience prediction for `P(bit = 1)`.
    pub fn predict_one(&mut self) -> f64 {
        self.predict_bit().p1
    }

    /// Capture a reversible checkpoint for later restoration.
    ///
    /// Backends that store full snapshots restore bit-identical floating-point
    /// predictions. Backends that use compact reversible journals may differ by
    /// a few ULP after restore because floating-point accumulators are replayed
    /// instead of cloned byte-for-byte.
    pub fn checkpoint(&mut self) -> RateBackendBitSessionCheckpoint {
        debug_assert_eq!(
            self.discardable_scopes, 0,
            "RateBackendBitSession checkpoints are invalid inside discardable simulation scopes"
        );
        RateBackendBitSessionCheckpoint {
            backend_code: self.backend_code.clone(),
            predictor: self.predictor.checkpoint(),
            semantics: self.semantics,
            prefix: self.prefix.clone(),
        }
    }

    #[cfg(any(feature = "aixi", all(test, feature = "backend-ctw")))]
    pub(crate) fn begin_discardable_scope(&mut self) {
        self.discardable_scopes = self.discardable_scopes.saturating_add(1);
    }

    #[cfg(feature = "aixi")]
    pub(crate) fn clear_discardable_scopes(&mut self) {
        self.discardable_scopes = 0;
    }

    /// Restore the session to a previously captured checkpoint.
    ///
    /// Snapshot-backed predictors restore bit-identical predictions. Compact
    /// journaled predictors restore the same discrete predictor state and stream
    /// position, with predictions equal up to floating-point round-off.
    ///
    /// A checkpoint is tied to the backend and bit-stream semantics it was
    /// created from. Restoring a checkpoint into a different bit session is a
    /// programmer error and returns an explicit runtime error.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &RateBackendBitSessionCheckpoint,
    ) -> InfotheoryResult<()> {
        if self.backend_code != checkpoint.backend_code || self.semantics != checkpoint.semantics {
            return Err(InfotheoryError::runtime(
                "RateBackendBitSession checkpoint belongs to a different backend or bit semantics",
            ));
        }
        self.release_inflight_prefix_checkpoint_for_restore();
        self.predictor.restore_checkpoint(&checkpoint.predictor);
        let mut restored_prefix = checkpoint.prefix.clone();
        if let Some(BufferedBytePrefix {
            kind:
                BufferedBytePrefixKind::NativeMsb {
                    start_checkpoint: Some(start_checkpoint),
                    symbol,
                    bits,
                },
            ..
        }) = checkpoint.prefix.as_ref()
        {
            let restored_state = self.predictor.checkpoint();
            self.predictor.restore_checkpoint(start_checkpoint.as_ref());
            let abort_res = self.predictor.abort_empty_native_msb_byte_prefix();
            if let Err(err) = abort_res {
                self.predictor.restore_checkpoint(&restored_state);
                self.predictor.discard_checkpoint(restored_state);
                return Err(InfotheoryError::runtime(err));
            }
            let fresh_start = self.predictor.native_prefix_start_checkpoint();
            match self.predictor.begin_native_msb_byte_prefix() {
                Ok(true) => {}
                Ok(false) => {
                    self.predictor.discard_checkpoint(fresh_start);
                    self.predictor.restore_checkpoint(&restored_state);
                    self.predictor.discard_checkpoint(restored_state);
                    return Err(InfotheoryError::runtime(
                        "stored checkpoint requires native MSB-first byte-prefix support during restore",
                    ));
                }
                Err(err) => {
                    self.predictor.discard_checkpoint(fresh_start);
                    self.predictor.restore_checkpoint(&restored_state);
                    self.predictor.discard_checkpoint(restored_state);
                    return Err(InfotheoryError::runtime(err));
                }
            }
            for bit_idx in 0..*bits {
                let bit = Self::observed_native_prefix_bit(*symbol, bit_idx);
                if let Err(err) = self.predictor.observe_native_msb_prefix_bit(bit_idx, bit) {
                    self.predictor.restore_checkpoint(&restored_state);
                    self.predictor.discard_checkpoint(fresh_start);
                    self.predictor.discard_checkpoint(restored_state);
                    return Err(InfotheoryError::runtime(err));
                }
            }
            self.predictor.discard_checkpoint(restored_state);
            if let Some(prefix) = restored_prefix.as_mut()
                && let BufferedBytePrefixKind::NativeMsb {
                    start_checkpoint, ..
                } = &mut prefix.kind
            {
                *start_checkpoint = Some(Box::new(fresh_start));
            }
            self.prefix = restored_prefix;
            return Ok(());
        }
        self.prefix = restored_prefix;
        Ok(())
    }

    /// Clear compact checkpoint journals when no stored checkpoints remain.
    ///
    /// This is an optimization hint for backends with journaled checkpoints.
    /// Calling it while a checkpoint may still be restored violates the
    /// checkpoint contract. Byte-packed sessions preserve the invariant between
    /// buffered prefix state and predictor-native prefix state before clearing.
    pub fn clear_checkpoints_if_supported(&mut self) {
        match self.normalize_prefix_for_checkpoint_clear() {
            Ok(true) => self.predictor.clear_checkpoints_if_supported(),
            Ok(false) => {}
            Err(err) => {
                panic!("failed to normalize native byte-prefix before checkpoint clear: {err}")
            }
        }
    }

    /// Predict and then observe one adaptive/fitting bit.
    pub fn try_step_bit(&mut self, bit: bool) -> InfotheoryResult<BinaryPrediction> {
        let prediction = self.predict_bit();
        self.try_observe_bit(bit)?;
        Ok(prediction)
    }

    /// Predict and then observe one adaptive/fitting bit.
    pub fn step_bit(&mut self, bit: bool) -> BinaryPrediction {
        self.try_step_bit(bit)
            .unwrap_or_else(|err| panic!("step_bit rejected an invalid bit-session update: {err}"))
    }

    /// Observe one adaptive/fitting bit.
    pub fn try_observe_bit(&mut self, bit: bool) -> InfotheoryResult<()> {
        match self.semantics {
            BitStreamSemantics::BinaryTokens => {
                self.predictor.update(u8::from(bit));
                Ok(())
            }
            BitStreamSemantics::BytePacked { order } => {
                self.update_byte_packed_bit(bit, order, BufferedByteUpdateMode::Adaptive)
            }
        }
    }

    /// Observe one adaptive/fitting bit.
    pub fn observe_bit(&mut self, bit: bool) {
        self.try_observe_bit(bit).unwrap_or_else(|err| {
            panic!("observe_bit rejected an invalid bit-session update: {err}")
        });
    }

    /// Advance conditioning state with one bit without fitting/adapting.
    pub fn try_condition_bit(&mut self, bit: bool) -> InfotheoryResult<()> {
        match self.semantics {
            BitStreamSemantics::BinaryTokens => {
                self.predictor.update_frozen(u8::from(bit));
                Ok(())
            }
            BitStreamSemantics::BytePacked { order } => {
                self.update_byte_packed_bit(bit, order, BufferedByteUpdateMode::Frozen)
            }
        }
    }

    /// Advance conditioning state with one bit without fitting/adapting.
    pub fn condition_bit(&mut self, bit: bool) {
        self.try_condition_bit(bit).unwrap_or_else(|err| {
            panic!("condition_bit rejected an invalid bit-session update: {err}")
        });
    }

    /// Reset dynamic conditioning state while preserving fitted parameters/statistics.
    pub fn reset_frozen(&mut self, total_bits: Option<u64>) -> InfotheoryResult<()> {
        let total_symbols = total_symbols_for_bit_semantics(total_bits, self.semantics)
            .map_err(InfotheoryError::runtime)?;
        self.abandon_inflight_prefix_for_lifecycle_reset();
        self.predictor
            .reset_frozen(total_symbols)
            .map_err(InfotheoryError::runtime)
    }

    /// Finalize the underlying stream if the backend needs it.
    pub fn finish(&mut self) -> InfotheoryResult<()> {
        if matches!(self.semantics, BitStreamSemantics::BytePacked { .. })
            && self
                .prefix
                .as_ref()
                .is_some_and(BufferedBytePrefix::has_partial_bits)
        {
            return Err(InfotheoryError::runtime(
                "byte-packed bit streams must finish on a whole-byte boundary; use BitStreamSemantics::BinaryTokens for arbitrary-length bit streams",
            ));
        }
        self.discard_inflight_prefix_checkpoint();
        self.predictor
            .finish_stream()
            .map_err(InfotheoryError::runtime)
    }

    fn ensure_prefix_for_update(
        &mut self,
        order: BitOrder,
        update_mode: BufferedByteUpdateMode,
    ) -> InfotheoryResult<()> {
        if self.prefix.is_some() {
            return Ok(());
        }
        if order == BitOrder::MsbFirst && self.predictor.has_native_msb_byte_prefix() {
            let checkpoint = if update_mode == BufferedByteUpdateMode::Frozen {
                Some(self.predictor.native_prefix_start_checkpoint())
            } else {
                None
            };
            match self.try_begin_native_msb_prefix(checkpoint) {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(err) => return Err(InfotheoryError::runtime(err)),
            }
        }
        self.ensure_mass_prefix(order, update_mode);
        Ok(())
    }

    /// Try to enter native MSB byte-prefix mode for byte-packed sessions.
    ///
    /// - `Ok(true)`: native prefix active (`self.prefix` set)
    /// - `Ok(false)`: caller should use [`Self::ensure_mass_prefix`] (predictor unchanged)
    /// - `Err`: predictor state-machine failure; caller must not silently fall back
    fn try_begin_native_msb_prefix(
        &mut self,
        start_checkpoint: Option<RateBackendPredictorCheckpoint>,
    ) -> Result<bool, String> {
        match self.predictor.begin_native_msb_byte_prefix() {
            Ok(true) => {
                self.prefix = Some(BufferedBytePrefix::new_native_msb(start_checkpoint));
                Ok(true)
            }
            Ok(false) => {
                if let Some(checkpoint) = start_checkpoint {
                    self.predictor.discard_checkpoint(checkpoint);
                }
                Ok(false)
            }
            Err(err) => {
                if let Some(checkpoint) = start_checkpoint {
                    self.predictor.discard_checkpoint(checkpoint);
                }
                Err(err)
            }
        }
    }

    fn ensure_mass_prefix(&mut self, order: BitOrder, update_mode: BufferedByteUpdateMode) {
        if self.prefix.is_some() {
            return;
        }
        let mut logps = [0.0f64; 256];
        match update_mode {
            BufferedByteUpdateMode::Adaptive => self.predictor.fill_log_probs(&mut logps),
            BufferedByteUpdateMode::Frozen => self.predictor.fill_log_probs_frozen(&mut logps),
        }
        self.prefix = Some(BufferedBytePrefix::new_mass(
            BytePrefixMass::from_log_probs(&logps, order),
        ));
    }

    fn update_byte_packed_bit(
        &mut self,
        bit: bool,
        order: BitOrder,
        update_mode: BufferedByteUpdateMode,
    ) -> InfotheoryResult<()> {
        self.ensure_prefix_for_update(order, update_mode)?;
        let needs_prefix_checkpoint =
            update_mode != BufferedByteUpdateMode::Adaptive || self.discardable_scopes == 0;
        let prefix = self.prefix.as_mut().expect("prefix initialized");
        prefix.record_mode(update_mode)?;
        match &mut prefix.kind {
            BufferedBytePrefixKind::Mass(mass) => {
                mass.observe(bit);
                if mass.is_complete() {
                    let symbol = mass.symbol();
                    match update_mode {
                        BufferedByteUpdateMode::Adaptive => self.predictor.update(symbol),
                        BufferedByteUpdateMode::Frozen => self.predictor.update_frozen(symbol),
                    }
                    self.prefix = None;
                }
            }
            BufferedBytePrefixKind::NativeMsb {
                start_checkpoint,
                symbol,
                bits,
            } => {
                if start_checkpoint.is_none() && *bits == 0 && needs_prefix_checkpoint {
                    *start_checkpoint =
                        Some(Box::new(self.predictor.native_prefix_start_checkpoint()));
                }
                match update_mode {
                    BufferedByteUpdateMode::Adaptive => self
                        .predictor
                        .observe_native_msb_prefix_bit(*bits, bit)
                        .map_err(InfotheoryError::runtime)?,
                    BufferedByteUpdateMode::Frozen => self
                        .predictor
                        .condition_native_msb_prefix_bit_for_rollback(*bits, bit)
                        .map_err(InfotheoryError::runtime)?,
                }
                if bit {
                    *symbol |= 1u8 << (7 - *bits);
                }
                *bits += 1;
                if *bits == 8 {
                    let completed_symbol = *symbol;
                    if update_mode == BufferedByteUpdateMode::Frozen {
                        let checkpoint_box = start_checkpoint.take().ok_or_else(|| {
                            InfotheoryError::runtime(
                                "native byte-prefix frozen update is missing its start checkpoint",
                            )
                        })?;
                        let checkpoint = *checkpoint_box;
                        self.restore_start_checkpoint_abort_and_discard(checkpoint)
                            .map_err(InfotheoryError::runtime)?;
                        self.predictor.update_frozen(completed_symbol);
                    } else {
                        self.predictor
                            .finish_native_msb_byte_prefix(completed_symbol)
                            .map_err(InfotheoryError::runtime)?;
                        if let Some(checkpoint) = start_checkpoint.take() {
                            self.predictor.discard_checkpoint(*checkpoint);
                        }
                    }
                    self.prefix = None;
                }
            }
        }
        Ok(())
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
        let total_symbols = total_symbols_for_bit_semantics(total_bits, self.semantics)?;
        self.abandon_inflight_prefix_for_lifecycle_reset();
        self.predictor.begin_fresh_stream(total_symbols)
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

    /// Start a new stream while preserving each backend's semantic contract.
    ///
    /// Backends that support frozen-reset semantics will restart via
    /// `reset_frozen`. Backends that do not (for example ZPAQ) restart through
    /// ordinary stream lifecycle hooks instead.
    pub fn begin_stream(&mut self, total_symbols: Option<u64>) -> InfotheoryResult<()> {
        self.predictor
            .begin_fresh_stream(total_symbols)
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
                        *slot = match config.update_mode {
                            GenerationUpdateMode::Adaptive => self.predictor.log_prob(sym as u8),
                            GenerationUpdateMode::Frozen => {
                                self.predictor.log_prob_frozen(sym as u8)
                            }
                        };
                    }
                }
                _ => match config.update_mode {
                    GenerationUpdateMode::Adaptive => self.predictor.fill_log_probs(&mut logps),
                    GenerationUpdateMode::Frozen => {
                        self.predictor.fill_log_probs_frozen(&mut logps);
                    }
                },
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

#[cfg(test)]
mod tests {
    use super::*;
    // Re-exported for trait methods (begin_bit_stream etc.) exercised in
    // feature-specific tests. Some narrow backend slices do not use the trait
    // directly, but broader bit-session test slices do.
    #[allow(unused_imports)]
    use crate::prediction::OnlineBitPredictor;

    #[cfg(feature = "backend-ctw")]
    fn ctw_checkpoint_depth(session: &RateBackendBitSession) -> usize {
        match &session.predictor {
            crate::mixture::RateBackendPredictor::Ctw {
                checkpoint_depth, ..
            } => *checkpoint_depth,
            crate::mixture::RateBackendPredictor::FacCtw {
                checkpoint_depth, ..
            } => *checkpoint_depth,
            _ => panic!("expected ctw predictor"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn ctw_native_prefix_progress(session: &RateBackendBitSession) -> Option<usize> {
        match &session.predictor {
            crate::mixture::RateBackendPredictor::Ctw {
                native_prefix_progress,
                ..
            }
            | crate::mixture::RateBackendPredictor::FacCtw {
                native_prefix_progress,
                ..
            } => *native_prefix_progress,
            _ => panic!("expected ctw predictor"),
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn buffered_native_prefix_bits(session: &RateBackendBitSession) -> Option<usize> {
        match session.prefix.as_ref().map(|prefix| &prefix.kind) {
            Some(BufferedBytePrefixKind::NativeMsb { bits, .. }) => Some(*bits),
            _ => None,
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn buffered_native_prefix_has_start_checkpoint(
        session: &RateBackendBitSession,
    ) -> Option<bool> {
        match session.prefix.as_ref().map(|prefix| &prefix.kind) {
            Some(BufferedBytePrefixKind::NativeMsb {
                start_checkpoint, ..
            }) => Some(start_checkpoint.is_some()),
            _ => None,
        }
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
    fn buffered_native_prefix_has_calibrated_start_checkpoint(
        session: &RateBackendBitSession,
    ) -> Option<bool> {
        match session.prefix.as_ref().map(|prefix| &prefix.kind) {
            Some(BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: Some(checkpoint),
                ..
            }) => Some(matches!(
                checkpoint.as_ref(),
                crate::mixture::RateBackendPredictorCheckpoint::CalibratedNativePrefixStart { .. }
            )),
            Some(BufferedBytePrefixKind::NativeMsb {
                start_checkpoint: None,
                ..
            }) => Some(false),
            _ => None,
        }
    }

    #[cfg(feature = "backend-ctw")]
    fn new_ctw_byte_packed_session() -> RateBackendBitSession {
        RateBackendBitSession::from_spec(
            RateBackend::Ctw { depth: 4 },
            Some(8),
            BitStreamSemantics::BytePacked {
                order: BitOrder::MsbFirst,
            },
        )
        .expect("ctw byte-packed session")
    }

    #[cfg(feature = "backend-ctw")]
    fn new_fac_ctw_byte_packed_session() -> RateBackendBitSession {
        RateBackendBitSession::from_spec(
            RateBackend::FacCtw {
                base_depth: 4,
                num_percept_bits: 8,
                encoding_bits: 8,
                msb_first: None,
            },
            Some(8),
            BitStreamSemantics::BytePacked {
                order: BitOrder::MsbFirst,
            },
        )
        .expect("fac-ctw byte-packed session")
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
    fn new_calibrated_ctw_byte_packed_session() -> RateBackendBitSession {
        RateBackendBitSession::from_spec(
            RateBackend::Calibrated {
                spec: std::sync::Arc::new(crate::api::CalibratedSpec::new(
                    RateBackend::Ctw { depth: 4 },
                    crate::api::CalibrationContextKind::TextRepeat,
                )),
            },
            Some(8),
            BitStreamSemantics::BytePacked {
                order: BitOrder::MsbFirst,
            },
        )
        .expect("calibrated ctw byte-packed session")
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn frozen_native_byte_completion_releases_start_checkpoint() {
        for mut session in [
            new_ctw_byte_packed_session(),
            new_fac_ctw_byte_packed_session(),
        ] {
            for bit in [true, false, true, false, false, true, true, false] {
                session
                    .try_condition_bit(bit)
                    .expect("condition full frozen native byte");
            }
            assert!(session.prefix.is_none());
            assert_eq!(ctw_checkpoint_depth(&session), 0);
            assert_eq!(ctw_native_prefix_progress(&session), None);

            let pred = session.predict_bit();
            assert!(pred.p0.is_finite() && pred.p1.is_finite());
            assert_eq!(buffered_native_prefix_bits(&session), Some(0));
            assert_eq!(ctw_native_prefix_progress(&session), Some(0));
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn reset_frozen_discards_inflight_native_prefix_checkpoint() {
        let mut session = new_ctw_byte_packed_session();
        session.try_condition_bit(true).expect("conditioning bit");
        assert_eq!(ctw_checkpoint_depth(&session), 1);

        session.reset_frozen(Some(8)).expect("reset frozen");
        assert_eq!(ctw_checkpoint_depth(&session), 0);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn begin_bit_stream_discards_inflight_native_prefix_checkpoint() {
        let mut session = new_ctw_byte_packed_session();
        session.try_condition_bit(true).expect("conditioning bit");
        assert_eq!(ctw_checkpoint_depth(&session), 1);

        session
            .begin_bit_stream(
                Some(8),
                BitStreamSemantics::BytePacked {
                    order: BitOrder::MsbFirst,
                },
            )
            .expect("begin bit stream");
        assert_eq!(ctw_checkpoint_depth(&session), 0);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn clear_checkpoints_after_empty_native_restore_drops_prefix_cleanly() {
        for mut session in [
            new_ctw_byte_packed_session(),
            new_fac_ctw_byte_packed_session(),
        ] {
            let _ = session.predict_bit();
            let checkpoint = session.checkpoint();
            session
                .try_condition_bit(true)
                .expect("conditioning bit after checkpoint");

            session
                .restore_checkpoint(&checkpoint)
                .expect("restore empty native prefix checkpoint");
            assert_eq!(buffered_native_prefix_bits(&session), Some(0));
            assert_eq!(ctw_native_prefix_progress(&session), Some(0));

            session.clear_checkpoints_if_supported();
            assert!(session.prefix.is_none());
            assert_eq!(ctw_checkpoint_depth(&session), 0);
            assert_eq!(ctw_native_prefix_progress(&session), None);

            let pred = session.predict_bit();
            assert!(pred.p0.is_finite() && pred.p1.is_finite());
            assert_eq!(buffered_native_prefix_bits(&session), Some(0));
            assert_eq!(ctw_native_prefix_progress(&session), Some(0));
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn clear_checkpoints_converts_partial_native_prefix_to_mass() {
        for mut session in [
            new_ctw_byte_packed_session(),
            new_fac_ctw_byte_packed_session(),
        ] {
            session.try_condition_bit(true).expect("first prefix bit");
            session.try_condition_bit(false).expect("second prefix bit");
            assert_eq!(buffered_native_prefix_bits(&session), Some(2));
            assert_eq!(ctw_native_prefix_progress(&session), Some(2));

            let before_clear = session.predict_bit();
            session.clear_checkpoints_if_supported();

            assert!(matches!(
                session.prefix.as_ref().map(|prefix| &prefix.kind),
                Some(BufferedBytePrefixKind::Mass(_))
            ));
            assert_eq!(ctw_checkpoint_depth(&session), 0);
            assert_eq!(ctw_native_prefix_progress(&session), None);

            let after_clear = session.predict_bit();
            assert!(
                (before_clear.p1 - after_clear.p1).abs() <= 1e-12,
                "native-to-mass conversion changed prefix prediction: before={} after={}",
                before_clear.p1,
                after_clear.p1
            );

            for bit in [true, false, true, false, true, false] {
                session
                    .try_condition_bit(bit)
                    .expect("finish converted mass prefix");
            }
            session
                .finish()
                .expect("finish whole byte after conversion");
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn reset_frozen_rolls_back_adaptive_partial_native_prefix() {
        for mut session in [
            new_ctw_byte_packed_session(),
            new_fac_ctw_byte_packed_session(),
        ] {
            session.try_observe_bit(true).expect("adaptive prefix bit");
            assert_eq!(ctw_checkpoint_depth(&session), 1);
            assert_eq!(ctw_native_prefix_progress(&session), Some(1));

            session.reset_frozen(Some(8)).expect("reset frozen");

            assert_eq!(ctw_checkpoint_depth(&session), 0);
            assert_eq!(ctw_native_prefix_progress(&session), None);
            let mut fresh = match &session.predictor {
                crate::mixture::RateBackendPredictor::Ctw { .. } => new_ctw_byte_packed_session(),
                crate::mixture::RateBackendPredictor::FacCtw { .. } => {
                    new_fac_ctw_byte_packed_session()
                }
                _ => unreachable!("test only constructs CTW-family sessions"),
            };
            let reset_prediction = session.predict_bit();
            let fresh_prediction = fresh.predict_bit();
            assert!(
                (reset_prediction.p1 - fresh_prediction.p1).abs() <= 1e-12,
                "partial adaptive native prefix leaked into reset state: reset={} fresh={}",
                reset_prediction.p1,
                fresh_prediction.p1
            );
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn discardable_scope_avoids_adaptive_native_prefix_checkpoint() {
        for mut session in [
            new_ctw_byte_packed_session(),
            new_fac_ctw_byte_packed_session(),
        ] {
            session.begin_discardable_scope();
            let _ = session.predict_bit();
            session
                .try_observe_bit(true)
                .expect("discardable adaptive prefix bit");

            assert_eq!(buffered_native_prefix_bits(&session), Some(1));
            assert_eq!(
                buffered_native_prefix_has_start_checkpoint(&session),
                Some(false),
                "discardable adaptive prefixes must not retain restore-only checkpoints",
            );
            assert_eq!(ctw_checkpoint_depth(&session), 0);

            for bit in [false, true, false, true, false, true, false] {
                session
                    .try_observe_bit(bit)
                    .expect("finish discardable adaptive byte");
            }
            assert!(session.prefix.is_none());
            assert_eq!(ctw_checkpoint_depth(&session), 0);
            assert_eq!(ctw_native_prefix_progress(&session), None);
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn discardable_scope_keeps_frozen_native_prefix_checkpoint() {
        let mut session = new_ctw_byte_packed_session();
        session.begin_discardable_scope();
        session
            .try_condition_bit(true)
            .expect("discardable frozen prefix bit");

        assert_eq!(buffered_native_prefix_bits(&session), Some(1));
        assert_eq!(
            buffered_native_prefix_has_start_checkpoint(&session),
            Some(true),
            "frozen native prefixes still need a start checkpoint to avoid learning action bytes",
        );
        assert_eq!(ctw_checkpoint_depth(&session), 1);
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
    #[test]
    fn calibrated_native_prefix_supports_update_only_bits() {
        let bits = [true, false, true, false, false, true, true, false];

        let mut adaptive = new_calibrated_ctw_byte_packed_session();
        for bit in bits {
            adaptive
                .try_observe_bit(bit)
                .expect("adaptive calibrated update-only bit");
        }
        assert!(adaptive.prefix.is_none());
        adaptive.finish().expect("finish adaptive calibrated byte");

        let mut frozen = new_calibrated_ctw_byte_packed_session();
        for bit in bits {
            frozen
                .try_condition_bit(bit)
                .expect("frozen calibrated update-only bit");
        }
        assert!(frozen.prefix.is_none());
        frozen.finish().expect("finish frozen calibrated byte");
    }

    #[cfg(all(feature = "backend-calibrated", feature = "backend-ctw"))]
    #[test]
    fn calibrated_adaptive_prefix_uses_lightweight_start_checkpoint() {
        let mut session = new_calibrated_ctw_byte_packed_session();
        session
            .try_observe_bit(true)
            .expect("adaptive calibrated prefix bit");
        assert_eq!(buffered_native_prefix_bits(&session), Some(1));
        assert_eq!(
            buffered_native_prefix_has_calibrated_start_checkpoint(&session),
            Some(true),
            "calibrated adaptive prefixes must store only the wrapped predictor checkpoint"
        );

        session.reset_frozen(Some(8)).expect("reset frozen");
        let mut fresh = new_calibrated_ctw_byte_packed_session();
        let reset_prediction = session.predict_bit();
        let fresh_prediction = fresh.predict_bit();
        assert!(
            (reset_prediction.p1 - fresh_prediction.p1).abs() <= 1e-12,
            "abandoned calibrated prefix leaked into reset state: reset={} fresh={}",
            reset_prediction.p1,
            fresh_prediction.p1
        );
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn begin_bit_stream_rolls_back_adaptive_partial_native_prefix() {
        let mut session = new_fac_ctw_byte_packed_session();
        session.try_observe_bit(true).expect("adaptive prefix bit");
        assert_eq!(ctw_checkpoint_depth(&session), 1);
        assert_eq!(ctw_native_prefix_progress(&session), Some(1));

        session
            .begin_bit_stream(
                Some(8),
                BitStreamSemantics::BytePacked {
                    order: BitOrder::MsbFirst,
                },
            )
            .expect("begin bit stream");

        assert_eq!(ctw_checkpoint_depth(&session), 0);
        assert_eq!(ctw_native_prefix_progress(&session), None);
        session
            .try_observe_bit(false)
            .expect("native prefix re-entry after adaptive abort");
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn restore_checkpoint_discards_abandoned_native_prefix_checkpoint() {
        let mut session = new_ctw_byte_packed_session();
        let checkpoint = session.checkpoint();
        assert_eq!(ctw_checkpoint_depth(&session), 1);

        session.try_condition_bit(true).expect("conditioning bit");
        assert_eq!(ctw_checkpoint_depth(&session), 2);

        session
            .restore_checkpoint(&checkpoint)
            .expect("restore checkpoint");
        assert_eq!(ctw_checkpoint_depth(&session), 1);
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn restore_mid_prefix_checkpoint_rebuilds_frozen_native_rollback_point() {
        let mut session = new_ctw_byte_packed_session();
        session.try_condition_bit(true).expect("conditioning bit");
        let checkpoint = session.checkpoint();
        assert_eq!(ctw_checkpoint_depth(&session), 2);

        session
            .try_condition_bit(false)
            .expect("conditioning second bit");
        session
            .restore_checkpoint(&checkpoint)
            .expect("restore mid-prefix checkpoint");
        assert_eq!(ctw_checkpoint_depth(&session), 2);

        for bit in [false, true, false, true, false, true, false] {
            session
                .try_condition_bit(bit)
                .expect("finish restored byte");
        }
        assert_eq!(ctw_checkpoint_depth(&session), 1);
    }

    /// When static native MSB support is advertised without the empty-abort
    /// capability needed by mixture setup rollback, `predict_bit` must fall back
    /// to the BytePrefixMass path without debug-only false positives.
    #[cfg(feature = "backend-mixture")]
    #[test]
    fn predict_bit_mass_fallback_when_native_setup_is_not_abortable() {
        use crate::mixture::{
            BayesMixture, DEFAULT_MIN_PROB, ExpertConfig, MixtureRuntime, RateBackendPredictor,
        };

        #[derive(Clone)]
        struct MockNativeNoCheckpoint;

        impl OnlineBytePredictor for MockNativeNoCheckpoint {
            fn log_prob(&mut self, _symbol: u8) -> f64 {
                -(256.0f64).ln()
            }

            fn update(&mut self, _symbol: u8) {}

            fn has_native_msb_byte_prefix(&self) -> bool {
                true
            }
        }

        let configs = [ExpertConfig::uniform("native", || {
            Box::new(MockNativeNoCheckpoint) as Box<dyn OnlineBytePredictor>
        })];
        let runtime = MixtureRuntime::Bayes(BayesMixture::new(&configs));
        let mut predictor = RateBackendPredictor::Mixture { runtime };
        predictor.begin_stream(Some(1)).expect("begin_stream");

        let mut session = RateBackendBitSession {
            backend_code: CanonicalBytes::from(b"test-native-fallback".to_vec()),
            predictor,
            semantics: BitStreamSemantics::BytePacked {
                order: BitOrder::MsbFirst,
            },
            min_prob: DEFAULT_MIN_PROB,
            prefix: None,
            discardable_scopes: 0,
        };
        let pred = session.predict_bit();
        assert!(
            (pred.p0 + pred.p1 - 1.0).abs() < 1e-9,
            "mass fallback prediction must normalize: p0={} p1={}",
            pred.p0,
            pred.p1
        );
        assert!(pred.p1.is_finite());
        assert!(matches!(
            session.prefix.as_ref().map(|p| &p.kind),
            Some(BufferedBytePrefixKind::Mass(_))
        ));
    }

    /// Guards the byte-prefix buffering enum against accidentally embedding a
    /// full predictor checkpoint inline in the native-MSB variant.
    #[test]
    fn record_buffered_byte_prefix_kind_and_related_sizes() {
        let byte_prefix_mass = std::mem::size_of::<BytePrefixMass>();
        let buffered_kind = std::mem::size_of::<BufferedBytePrefixKind>();
        let rate_ckpt = std::mem::size_of::<RateBackendPredictorCheckpoint>();
        let opt_rate_ckpt = std::mem::size_of::<Option<RateBackendPredictorCheckpoint>>();
        eprintln!(
            "BufferedBytePrefixKind sizes: \
             BufferedBytePrefixKind={}B (Mass~{}B inline; NativeMsb Option<RateCheckpoint>~{}B; \
             RateBackendPredictorCheckpoint enum={}B). start_checkpoint is boxed; Full variant \
             is Box<RateBackendPredictor> so compact checkpoint variants stay small.",
            buffered_kind, byte_prefix_mass, opt_rate_ckpt, rate_ckpt
        );
        // Native-MSB frozen rewinds need a predictor checkpoint, but that
        // checkpoint may contain full model state. Keeping it boxed prevents the
        // cheaper byte-packed mass path from inheriting that storage cost.
        assert_eq!(
            buffered_kind, byte_prefix_mass,
            "post-box BufferedBytePrefixKind (NativeMsb) must not exceed Mass variant size ({} vs {}); \
             native-MSB rewind checkpoints must stay out-of-line",
            buffered_kind, byte_prefix_mass
        );
    }
}
