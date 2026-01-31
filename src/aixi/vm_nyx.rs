//! High-performance VM-backed AIXI environment using nyx-lite (Firecracker).
//!
//! This module provides a VM environment implementation built on top of nyx-lite,
//! enabling high-frequency snapshot-based resets for fast experimentation (hardware and
//! guest behavior dependent).
//!
//! ## Architecture
//!
//! The environment uses Firecracker's KVM-based microVM with nyx-lite's incremental
//! snapshot and reset capabilities. Communication with the guest occurs via:
//!
//! 1. **Shared Memory**: Zero-copy data transfer between host and guest
//! 2. **Hypercalls**: Control plane communication (snapshot, done, etc.)
//! 3. **Serial PTY**: Optional console I/O for simpler protocols
//!
//! ## Design Principles
//!
//! - **Universal**: Not biased towards any specific use case (fuzzing, etc.)
//! - **High Performance**: Leverages incremental snapshots and dirty page tracking
//! - **Configurable**: Pluggable reward policies, action sources, observation modes
//! - **Information-Theoretic**: Built-in support for entropy-based metrics

use crate::aixi::common::{Action, PerceptVal, RandomGenerator, Reward};
use crate::aixi::environment::Environment;
use crate::{
    RateBackend, cross_entropy_rate_backend, entropy_rate_backend, marginal_entropy_bytes,
};
use crate::zpaq_rate::ZpaqRateModel;
use rosaplus::RosaPlus;
use rwkvzip::Compressor;
use rwkvzip::coders::softmax_pdf_inplace;
use serde_json::Value;
use std::borrow::Cow;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Re-export nyx-lite types for external use
pub use nyx_lite::mem::SharedMemoryRegion;
pub use nyx_lite::snapshot::NyxSnapshot;
pub use nyx_lite::{ExitReason, NyxVM, SharedMemoryPolicy};

// ============================================================================
// Encoding Types
// ============================================================================

/// Payload encoding for wire protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadEncoding {
    Utf8,
    Hex,
}

impl PayloadEncoding {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "utf8" | "text" => Some(Self::Utf8),
            "hex" => Some(Self::Hex),
            _ => None,
        }
    }

    pub fn decode(self, s: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Utf8 => Ok(s.as_bytes().to_vec()),
            Self::Hex => hex_decode(s),
        }
    }

    pub fn encode(self, bytes: &[u8]) -> String {
        match self {
            Self::Utf8 => String::from_utf8_lossy(bytes).to_string(),
            Self::Hex => hex_encode(bytes),
        }
    }
}

fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut buf = 0u8;
    let mut high = true;
    for c in s.bytes() {
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(anyhow::anyhow!("invalid hex byte: {}", c as char)),
        };
        if high {
            buf = v << 4;
            high = false;
        } else {
            buf |= v;
            out.push(buf);
            high = true;
        }
    }
    if !high {
        return Err(anyhow::anyhow!("hex string has odd length"));
    }
    Ok(out)
}

fn resolve_relative_path(base: &Path, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        base.join(p).to_string_lossy().to_string()
    }
}

fn rewrite_firecracker_config_paths(config_path: &str, raw_json: &str) -> anyhow::Result<String> {
    let base_dir = Path::new(config_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut v: Value = serde_json::from_str(raw_json)?;

    if let Some(boot) = v.get_mut("boot-source") {
        if let Some(path_val) = boot.get_mut("kernel_image_path") {
            if let Some(path_str) = path_val.as_str() {
                let resolved = resolve_relative_path(base_dir, path_str);
                *path_val = Value::String(resolved);
            }
        }
        if let Some(path_val) = boot.get_mut("initrd_path") {
            if let Some(path_str) = path_val.as_str() {
                let resolved = resolve_relative_path(base_dir, path_str);
                *path_val = Value::String(resolved);
            }
        }
    }

    if let Some(drives) = v.get_mut("drives").and_then(|d| d.as_array_mut()) {
        for drive in drives {
            if let Some(path_val) = drive.get_mut("path_on_host") {
                if let Some(path_str) = path_val.as_str() {
                    let resolved = resolve_relative_path(base_dir, path_str);
                    *path_val = Value::String(resolved);
                }
            }
        }
    }

    Ok(serde_json::to_string(&v)?)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(hex_digit(b >> 4));
        s.push(hex_digit(b & 0x0F));
    }
    s
}

fn hex_digit(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        _ => (b'a' + (v - 10)) as char,
    }
}

// ============================================================================
// Guest Communication Protocol
// ============================================================================

/// Hypercall identifiers (must match guest implementation).
/// These are exported for use by custom guest programs.
#[allow(dead_code)]
pub const HYPERCALL_EXECDONE: u64 = 0x656e6f6463657865; // "execdone"
#[allow(dead_code)]
pub const HYPERCALL_SNAPSHOT: u64 = 0x746f687370616e73; // "snapshot"
#[allow(dead_code)]
pub const HYPERCALL_NYX_LITE: u64 = 0x6574696c2d78796e; // "nyx-lite"
#[allow(dead_code)]
pub const HYPERCALL_SHAREMEM: u64 = 0x6d656d6572616873; // "sharemem"
#[allow(dead_code)]
pub const HYPERCALL_DBGPRINT: u64 = 0x746e697270676264; // "dbgprint"

const SHARED_ACTION_LEN_OFFSET: u64 = 0;
const SHARED_RESP_LEN_OFFSET: u64 = 8;
const SHARED_PAYLOAD_OFFSET: u64 = 16;

/// Protocol configuration for structured communication.
#[derive(Clone, Debug)]
pub struct NyxProtocolConfig {
    /// Prefix for action messages.
    pub action_prefix: String,
    /// Suffix for action messages.
    pub action_suffix: String,
    /// Prefix for observation responses.
    pub obs_prefix: String,
    /// Prefix for reward responses.
    pub rew_prefix: String,
    /// Prefix for done indicator.
    pub done_prefix: String,
    /// Prefix for data payloads.
    pub data_prefix: String,
    /// Wire encoding for payloads.
    pub wire_encoding: PayloadEncoding,
}

impl Default for NyxProtocolConfig {
    fn default() -> Self {
        Self {
            action_prefix: "ACT ".to_string(),
            action_suffix: "\n".to_string(),
            obs_prefix: "OBS ".to_string(),
            rew_prefix: "REW ".to_string(),
            done_prefix: "DONE ".to_string(),
            data_prefix: "DATA ".to_string(),
            wire_encoding: PayloadEncoding::Hex,
        }
    }
}

// ============================================================================
// Action Configuration
// ============================================================================

/// A single action specification.
#[derive(Clone, Debug)]
pub struct NyxActionSpec {
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Raw payload bytes to send.
    pub payload: Vec<u8>,
}

/// Fuzzing mutator types.
#[derive(Clone, Debug)]
pub enum FuzzMutator {
    FlipBit,
    FlipByte,
    InsertByte,
    DeleteByte,
    SpliceSeed,
    ResetSeed,
    Havoc,
}

/// Fuzzing configuration for action generation.
#[derive(Clone, Debug)]
pub struct NyxFuzzConfig {
    pub seeds: Vec<Vec<u8>>,
    pub mutators: Vec<FuzzMutator>,
    pub min_len: usize,
    pub max_len: usize,
    pub dictionary: Vec<Vec<u8>>,
    pub rng_seed: u64,
}

/// Source of actions for the environment.
#[derive(Clone, Debug)]
pub enum NyxActionSource {
    /// Fixed set of action payloads.
    Literal(Vec<NyxActionSpec>),
    /// Mutation-based action generation.
    Fuzz(NyxFuzzConfig),
}

// ============================================================================
// Observation Configuration
// ============================================================================

/// How observations are derived from guest output.
#[derive(Clone, Copy, Debug)]
pub enum NyxObservationPolicy {
    /// Parse structured OBS/REW/DONE messages from guest.
    FromGuest,
    /// Hash raw output to derive observation.
    OutputHash,
    /// Use raw output bytes as observation stream.
    RawOutput,
    /// Use shared memory contents as observation.
    SharedMemory,
}

/// Stream normalization mode.
#[derive(Clone, Copy, Debug)]
pub enum NyxObservationStreamMode {
    /// Pad short streams, truncate long ones.
    PadTruncate,
    /// Only pad short streams.
    Pad,
    /// Only truncate long streams.
    Truncate,
}

// ============================================================================
// Reward Configuration
// ============================================================================

/// How rewards are computed.
#[derive(Clone)]
pub enum NyxRewardPolicy {
    /// Parse reward from guest response.
    FromGuest,
    /// Pattern matching on output.
    Pattern {
        pattern: String,
        base_reward: i64,
        bonus_reward: i64,
    },
    /// Custom reward function (callback-based).
    Custom(Arc<dyn Fn(&NyxStepResult) -> Reward + Send + Sync>),
}

/// Optional reward shaping (additive to base reward).
#[derive(Clone, Debug)]
pub enum NyxRewardShaping {
    /// Entropy reduction vs baseline.
    EntropyReduction {
        baseline_bytes: Vec<u8>,
        max_order: i64,
        scale: f64,
        crash_bonus: Option<i64>,
        timeout_bonus: Option<i64>,
    },
    /// Entropy of trace data (online learning).
    TraceEntropy {
        max_order: i64,
        scale: f64,
        normalize: bool,
    },
}

impl std::fmt::Debug for NyxRewardPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FromGuest => write!(f, "FromGuest"),
            Self::Pattern {
                pattern,
                base_reward,
                bonus_reward,
            } => f
                .debug_struct("Pattern")
                .field("pattern", pattern)
                .field("base_reward", base_reward)
                .field("bonus_reward", bonus_reward)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

// ============================================================================
// Action Filtering
// ============================================================================

/// Information-theoretic action filtering.
#[derive(Clone, Debug)]
pub struct NyxActionFilter {
    /// Minimum entropy threshold.
    pub min_entropy: Option<f64>,
    /// Maximum entropy threshold.
    pub max_entropy: Option<f64>,
    /// Minimum intrinsic dependence.
    pub min_intrinsic_dependence: Option<f64>,
    /// Minimum novelty (cross-entropy vs prior).
    pub min_novelty: Option<f64>,
    /// Prior corpus for novelty computation.
    pub novelty_prior: Option<Vec<u8>>,
    /// Max order for entropy estimation.
    pub max_order: i64,
    /// Reward to assign when action is rejected.
    pub reject_reward: Option<i64>,
}

// ============================================================================
// Trace Configuration
// ============================================================================

/// Configuration for trace collection and analysis.
#[derive(Clone, Debug)]
pub struct NyxTraceConfig {
    /// Shared memory region name for trace data.
    pub shared_region_name: Option<String>,
    /// Maximum bytes to collect per step.
    pub max_bytes: usize,
    /// Reset trace model on episode boundary.
    pub reset_on_episode: bool,
}

// ============================================================================
// Main Configuration
// ============================================================================

/// Complete configuration for the nyx-lite VM environment.
#[derive(Clone)]
pub struct NyxVmConfig {
    /// Path to Firecracker JSON config.
    pub firecracker_config: String,
    /// Instance ID for the VM.
    pub instance_id: String,

    // Shared memory configuration
    /// Name of the shared memory region for communication.
    pub shared_region_name: String,
    /// Size of the shared memory region.
    pub shared_region_size: usize,
    /// Shared memory policy (snapshot vs preserve).
    pub shared_memory_policy: SharedMemoryPolicy,

    // Timing configuration
    /// Timeout for each step.
    pub step_timeout: Duration,
    /// Timeout for initial boot.
    pub boot_timeout: Duration,

    // Episode configuration
    /// Number of steps per episode.
    pub episode_steps: usize,
    /// Cost subtracted from reward each step.
    pub step_cost: i64,

    // Observation configuration
    /// Observation derivation policy.
    pub observation_policy: NyxObservationPolicy,
    /// Bits per observation symbol.
    pub observation_bits: usize,
    /// Number of observation symbols per action.
    pub observation_stream_len: usize,
    /// Stream normalization mode.
    pub observation_stream_mode: NyxObservationStreamMode,
    /// Padding byte for short streams.
    pub observation_pad_byte: u8,

    // Reward configuration
    /// Bits for reward encoding.
    pub reward_bits: usize,
    /// Reward computation policy.
    pub reward_policy: NyxRewardPolicy,
    /// Optional reward shaping (additive; non-canonical).
    pub reward_shaping: Option<NyxRewardShaping>,

    // Action configuration
    /// Source of actions.
    pub action_source: NyxActionSource,
    /// Optional action filter.
    pub action_filter: Option<NyxActionFilter>,

    // Protocol configuration
    /// Wire protocol for structured communication.
    pub protocol: NyxProtocolConfig,

    // Statistics backend
    /// Backend for entropy estimation.
    pub stats_backend: RateBackend,

    // Trace configuration
    /// Optional trace collection.
    pub trace: Option<NyxTraceConfig>,

    // Debug mode
    pub debug_mode: bool,

    // Crash logging
    /// Path to log crashes/interesting behaviors (JSONL format).
    pub crash_log: Option<String>,

}

impl Default for NyxVmConfig {
    fn default() -> Self {
        Self {
            firecracker_config: String::new(),
            instance_id: "aixi-nyx".to_string(),
            shared_region_name: "shared".to_string(),
            shared_region_size: 4096,
            shared_memory_policy: SharedMemoryPolicy::Snapshot,
            step_timeout: Duration::from_millis(100),
            boot_timeout: Duration::from_secs(30),
            episode_steps: 100,
            step_cost: 0,
            observation_policy: NyxObservationPolicy::SharedMemory,
            observation_bits: 8,
            observation_stream_len: 64,
            observation_stream_mode: NyxObservationStreamMode::PadTruncate,
            observation_pad_byte: 0,
            reward_bits: 8,
            reward_policy: NyxRewardPolicy::FromGuest,
            reward_shaping: None,
            action_source: NyxActionSource::Literal(vec![]),
            action_filter: None,
            protocol: NyxProtocolConfig::default(),
            stats_backend: RateBackend::default(),
            trace: None,
            debug_mode: false,
            crash_log: None,
        }
    }
}

// ============================================================================
// Step Result
// ============================================================================

/// Result of a single environment step.
#[derive(Clone, Debug)]
pub struct NyxStepResult {
    /// Exit reason from the VM.
    pub exit_reason: NyxExitKind,
    /// Raw output data from guest.
    pub output: Vec<u8>,
    /// Parsed observation (if any).
    pub parsed_obs: Option<u64>,
    /// Parsed reward (if any).
    pub parsed_rew: Option<i64>,
    /// Done flag.
    pub done: bool,
    /// Trace data (if collected).
    pub trace_data: Vec<u8>,
    /// Shared memory contents snapshot.
    pub shared_memory: Vec<u8>,
}

/// Simplified exit reason categories.
#[derive(Clone, Debug)]
pub enum NyxExitKind {
    ExecDone(u64),
    Timeout,
    Shutdown,
    Hypercall {
        code: u64,
        arg1: u64,
        arg2: u64,
        arg3: u64,
        arg4: u64,
    },
    DebugPrint(String),
    Breakpoint,
    Other(String),
}

impl From<ExitReason> for NyxExitKind {
    fn from(reason: ExitReason) -> Self {
        match reason {
            ExitReason::ExecDone(code) => Self::ExecDone(code),
            ExitReason::Timeout => Self::Timeout,
            ExitReason::Shutdown => Self::Shutdown,
            ExitReason::Hypercall(r8, r9, r10, r11, r12) => Self::Hypercall {
                code: r8,
                arg1: r9,
                arg2: r10,
                arg3: r11,
                arg4: r12,
            },
            ExitReason::DebugPrint(s) => Self::DebugPrint(s),
            ExitReason::Breakpoint => Self::Breakpoint,
            ExitReason::RequestSnapshot => Self::Other("RequestSnapshot".to_string()),
            ExitReason::SharedMem(name, _, _) => Self::Other(format!("SharedMem({})", name)),
            ExitReason::SingleStep => Self::Other("SingleStep".to_string()),
            ExitReason::Interrupted => Self::Other("Interrupted".to_string()),
            ExitReason::HWBreakpoint(n) => Self::Other(format!("HWBreakpoint({})", n)),
            ExitReason::BadMemoryAccess(_) => Self::Other("BadMemoryAccess".to_string()),
        }
    }
}

// ============================================================================
// Trace Model
// ============================================================================

/// Predictive model for trace-based reward computation.
enum TraceModel {
    Rosa {
        model: RosaPlus,
        max_order: i64,
    },
    Ctw {
        tree: crate::ctw::ContextTree,
    },
    FacCtw {
        tree: crate::ctw::FacContextTree,
        bits_per_symbol: usize,
    },
    Rwkv7 {
        compressor: Compressor,
        primed: bool,
    },
    Zpaq {
        model: ZpaqRateModel,
    },
    Mixture {
        backend: RateBackend,
        model: crate::mixture::RateBackendPredictor,
    },
}

impl TraceModel {
    fn new(backend: &RateBackend, max_order: i64) -> Self {
        match backend {
            RateBackend::RosaPlus => {
                let mut model = RosaPlus::new(max_order, false, 0, 42);
                model.build_lm_full_bytes_no_finalize_endpos();
                TraceModel::Rosa { model, max_order }
            }
            RateBackend::Rwkv7 { model } => {
                let compressor = Compressor::new_from_model(model.clone());
                TraceModel::Rwkv7 {
                    compressor,
                    primed: false,
                }
            }
            RateBackend::Zpaq { method } => TraceModel::Zpaq {
                model: ZpaqRateModel::new(method.clone(), 2f64.powi(-24)),
            },
            RateBackend::Mixture { spec } => {
                let backend = RateBackend::Mixture { spec: spec.clone() };
                let model = crate::mixture::RateBackendPredictor::from_backend(
                    backend.clone(),
                    -1,
                    2f64.powi(-24),
                );
                TraceModel::Mixture { backend, model }
            }
            RateBackend::Ctw { depth } => TraceModel::Ctw {
                tree: crate::ctw::ContextTree::new(*depth),
            },
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                let bits_per_symbol = (*encoding_bits).min(8).max(1);
                TraceModel::FacCtw {
                    tree: crate::ctw::FacContextTree::new(*base_depth, bits_per_symbol),
                    bits_per_symbol,
                }
            }
        }
    }

    fn reset(&mut self) {
        match self {
            TraceModel::Rosa { model, max_order } => {
                let mut fresh = RosaPlus::new(*max_order, false, 0, 42);
                fresh.build_lm_full_bytes_no_finalize_endpos();
                *model = fresh;
            }
            TraceModel::Ctw { tree } => tree.clear(),
            TraceModel::FacCtw { tree, .. } => tree.clear(),
            TraceModel::Rwkv7 { compressor, primed } => {
                compressor.state.reset();
                *primed = false;
            }
            TraceModel::Zpaq { model } => {
                model.reset();
            }
            TraceModel::Mixture { backend, model } => {
                *model = crate::mixture::RateBackendPredictor::from_backend(
                    backend.clone(),
                    -1,
                    2f64.powi(-24),
                );
            }
        }
    }

    /// Update the model with new data and return the surprise (bits).
    fn update_and_score(&mut self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        match self {
            TraceModel::Rosa { model, .. } => {
                let mut bits = 0.0;
                let mut tx = model.begin_tx();
                for &b in data {
                    let p = model.prob_for_last(b as u32).max(1e-12);
                    bits -= p.log2();
                    model.train_sequence_tx(&mut tx, &[b]);
                }
                bits
            }
            TraceModel::Ctw { tree } => {
                let log_before = tree.get_log_block_probability();
                for &b in data {
                    for i in (0..8).rev() {
                        tree.update(((b >> i) & 1) == 1);
                    }
                }
                let log_after = tree.get_log_block_probability();
                let log_delta = log_after - log_before;
                -log_delta / std::f64::consts::LN_2
            }
            TraceModel::FacCtw {
                tree,
                bits_per_symbol,
            } => {
                let log_before = tree.get_log_block_probability();
                for &b in data {
                    for i in 0..*bits_per_symbol {
                        tree.update(((b >> i) & 1) == 1, i);
                    }
                }
                let log_after = tree.get_log_block_probability();
                let log_delta = log_after - log_before;
                -log_delta / std::f64::consts::LN_2
            }
            TraceModel::Rwkv7 { compressor, primed } => {
                if !*primed {
                    let vocab_size = compressor.vocab_size();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    softmax_pdf_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
                    *primed = true;
                }
                let mut bits = 0.0;
                let vocab_size = compressor.vocab_size();
                for &b in data {
                    let p = compressor.pdf_buffer[b as usize].max(1e-12);
                    bits -= p.log2();
                    let logits = compressor.model.forward(
                        &mut compressor.scratch,
                        b as u32,
                        &mut compressor.state,
                    );
                    softmax_pdf_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
                }
                bits
            }
            TraceModel::Zpaq { model } => model.update_and_score(data),
            TraceModel::Mixture { model, .. } => {
                let mut bits = 0.0;
                for &b in data {
                    let logp = model.log_prob(b);
                    bits -= logp / std::f64::consts::LN_2;
                    model.update(b);
                }
                bits
            }
        }
    }
}

// ============================================================================
// Fuzz State
// ============================================================================

struct FuzzState {
    current: Vec<u8>,
    rng: RandomGenerator,
}

// ============================================================================
// NyxVmEnvironment
// ============================================================================

/// High-performance VM environment using nyx-lite.
pub struct NyxVmEnvironment {
    /// Configuration.
    config: NyxVmConfig,
    /// The nyx-lite VM instance.
    vm: NyxVM,
    /// Base snapshot for episode resets.
    base_snapshot: Option<Arc<NyxSnapshot>>,
    /// Shared memory virtual address in guest.
    shared_vaddr: Option<u64>,
    /// CR3 used when shared memory was registered.
    shared_cr3: Option<u64>,
    /// Trace model for entropy-based rewards.
    trace_model: Option<TraceModel>,
    /// Baseline entropy for entropy reduction rewards.
    baseline_entropy: Option<f64>,
    /// Effective reward shaping policy (additive).
    reward_shaping: Option<NyxRewardShaping>,
    /// Fuzzing state.
    fuzz_state: Option<FuzzState>,

    // Current step state
    /// Current observation.
    obs: PerceptVal,
    /// Current reward.
    rew: Reward,
    /// Current observation stream.
    obs_stream: Vec<PerceptVal>,
    /// Step within current episode.
    step_in_episode: usize,
    /// Whether the environment needs reset.
    needs_reset: bool,
    /// Whether the VM has been initialized.
    initialized: bool,
}

impl NyxVmEnvironment {
    /// Creates a new NyxVmEnvironment with the given configuration.
    pub fn new(config: NyxVmConfig) -> anyhow::Result<Self> {
        // Validate configuration
        if config.firecracker_config.is_empty() {
            return Err(anyhow::anyhow!("firecracker_config path must be set"));
        }
        if config.episode_steps == 0 {
            return Err(anyhow::anyhow!("episode_steps must be > 0"));
        }
        if matches!(config.observation_policy, NyxObservationPolicy::RawOutput)
            && config.observation_stream_len == 0
        {
            return Err(anyhow::anyhow!(
                "observation_stream_len must be > 0 for RawOutput policy"
            ));
        }

        // Load Firecracker config and resolve relative paths
        let fc_config_raw = std::fs::read_to_string(&config.firecracker_config)
            .map_err(|e| anyhow::anyhow!("Failed to read firecracker config: {}", e))?;
        let fc_config =
            rewrite_firecracker_config_paths(&config.firecracker_config, &fc_config_raw)
                .map_err(|e| anyhow::anyhow!("Failed to parse firecracker config: {}", e))?;

        // Create the VM
        let vm = NyxVM::new(config.instance_id.clone(), &fc_config);

        // Initialize reward shaping
        let reward_shaping = config.reward_shaping.clone();

        if matches!(reward_shaping, Some(NyxRewardShaping::TraceEntropy { .. }))
            && config.trace.is_none()
        {
            return Err(anyhow::anyhow!(
                "vm_trace must be configured for vm_reward_shaping.mode=trace-entropy"
            ));
        }

        // Initialize trace model if needed
        let trace_model = match &reward_shaping {
            Some(NyxRewardShaping::TraceEntropy { max_order, .. }) => {
                Some(TraceModel::new(&config.stats_backend, *max_order))
            }
            _ => None,
        };

        // Compute baseline entropy if needed
        let baseline_entropy = match &reward_shaping {
            Some(NyxRewardShaping::EntropyReduction {
                baseline_bytes,
                max_order,
                ..
            }) => {
                let h = if *max_order == 0 {
                    marginal_entropy_bytes(baseline_bytes)
                } else {
                    entropy_rate_backend(baseline_bytes, *max_order, &config.stats_backend)
                };
                Some(h)
            }
            _ => None,
        };

        // Initialize fuzz state if needed
        let fuzz_state = match &config.action_source {
            NyxActionSource::Fuzz(fuzz) => {
                if fuzz.seeds.is_empty() {
                    return Err(anyhow::anyhow!("Fuzz mode requires at least one seed"));
                }
                if fuzz.mutators.is_empty() {
                    return Err(anyhow::anyhow!("Fuzz mode requires at least one mutator"));
                }
                let seed = fuzz.seeds[0].clone();
                Some(FuzzState {
                    current: seed,
                    rng: RandomGenerator::new().fork_with(fuzz.rng_seed),
                })
            }
            NyxActionSource::Literal(actions) => {
                if actions.is_empty() {
                    return Err(anyhow::anyhow!("Literal mode requires at least one action"));
                }
                None
            }
        };

        let mut env = Self {
            config,
            vm,
            base_snapshot: None,
            shared_vaddr: None,
            shared_cr3: None,
            trace_model,
            baseline_entropy,
            reward_shaping,
            fuzz_state,
            obs: 0,
            rew: 0,
            obs_stream: Vec::new(),
            step_in_episode: 0,
            needs_reset: true,
            initialized: false,
        };

        // Boot and initialize
        env.initialize()?;

        Ok(env)
    }

    /// Initializes the VM by booting to the snapshot point.
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.initialized {
            return Ok(());
        }

        if self.config.debug_mode {
            eprintln!("[NyxVm] Booting VM...");
        }

        // Run until we get the shared memory registration
        let start = Instant::now();
        loop {
            if start.elapsed() > self.config.boot_timeout {
                return Err(anyhow::anyhow!("Boot timeout waiting for shared memory"));
            }

            let exit = self.vm.run(Duration::from_secs(1));
            match exit {
                ExitReason::SharedMem(name, vaddr, size) => {
                    if self.config.debug_mode {
                        eprintln!(
                            "[NyxVm] Shared memory registered: {} @ {:#x} ({} bytes)",
                            name, vaddr, size
                        );
                    }
                    if name.trim_end_matches('\0') == self.config.shared_region_name {
                        self.shared_vaddr = Some(vaddr);
                        self.shared_cr3 = Some(self.vm.sregs().cr3);
                        // Register the shared region with the configured policy
                        let _ = self.vm.register_shared_region_current(
                            vaddr,
                            size,
                            self.config.shared_memory_policy,
                        );
                        break;
                    }
                }
                ExitReason::DebugPrint(msg) => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Guest: {}", msg);
                    }
                }
                ExitReason::Shutdown => {
                    return Err(anyhow::anyhow!("VM shut down during boot"));
                }
                _ => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Boot exit: {:?}", exit);
                    }
                    // Continue waiting
                }
            }
        }

        // Continue running until snapshot request
        loop {
            if start.elapsed() > self.config.boot_timeout {
                return Err(anyhow::anyhow!("Boot timeout waiting for snapshot request"));
            }

            let exit = self.vm.run(Duration::from_secs(1));
            match exit {
                ExitReason::RequestSnapshot => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Taking base snapshot...");
                    }
                    self.base_snapshot = Some(self.vm.take_base_snapshot());
                    break;
                }
                ExitReason::DebugPrint(msg) => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Guest: {}", msg);
                    }
                }
                ExitReason::Shutdown => {
                    return Err(anyhow::anyhow!("VM shut down before snapshot"));
                }
                _ => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Snapshot wait exit: {:?}", exit);
                    }
                    // Continue waiting
                }
            }
        }

        if self.config.debug_mode {
            eprintln!("[NyxVm] Initialization complete");
        }

        self.initialized = true;
        self.needs_reset = false;
        Ok(())
    }

    /// Resets to the base snapshot.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        let snapshot = self
            .base_snapshot
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No base snapshot available"))?
            .clone();

        self.vm.apply_snapshot(&snapshot);

        // Reset trace model if configured
        if let Some(trace_cfg) = &self.config.trace {
            if trace_cfg.reset_on_episode {
                if let Some(model) = &mut self.trace_model {
                    model.reset();
                }
            }
        }

        self.step_in_episode = 0;
        self.needs_reset = false;

        Ok(())
    }

    /// Writes action data to shared memory.
    fn write_action_to_shared_memory(&mut self, payload: &[u8]) -> anyhow::Result<()> {
        let vaddr = self
            .shared_vaddr
            .ok_or_else(|| anyhow::anyhow!("Shared memory not initialized"))?;
        let cr3 = self
            .shared_cr3
            .ok_or_else(|| anyhow::anyhow!("Shared memory CR3 not initialized"))?;
        let process = self.vm.process_memory(cr3);

        // Ensure guest has cleared the previous message length to avoid races.
        let wait_start = Instant::now();
        loop {
            let cur_len = process
                .read_u64(vaddr + SHARED_ACTION_LEN_OFFSET)
                .unwrap_or(0);
            if cur_len == 0 {
                break;
            }
            if wait_start.elapsed() > self.config.step_timeout {
                return Err(anyhow::anyhow!("shared buffer busy (len={cur_len})"));
            }
            std::thread::yield_now();
        }

        // Write length as first 8 bytes (u64 LE)
        let len = payload.len() as u64;
        process
            .write_u64(vaddr + SHARED_ACTION_LEN_OFFSET, len)
            .map_err(|e| anyhow::anyhow!("write len failed: {e}"))?;
        let _ = process.write_u64(vaddr + SHARED_RESP_LEN_OFFSET, 0);

        // Write payload starting at offset 8
        let max_len = self
            .config
            .shared_region_size
            .saturating_sub(SHARED_PAYLOAD_OFFSET as usize);
        let write_len = payload.len().min(max_len);
        if write_len > 0 {
            let _ = process
                .write_bytes(vaddr + SHARED_PAYLOAD_OFFSET, &payload[..write_len])
                .map_err(|e| anyhow::anyhow!("write payload failed: {e}"))?;
        }

        if self.config.debug_mode {
            let verify = process
                .read_u64(vaddr + SHARED_ACTION_LEN_OFFSET)
                .unwrap_or(0) as usize;
            eprintln!(
                "[NyxVm] Wrote action len={}, verified len={}",
                write_len, verify
            );
        }

        Ok(())
    }

    /// Reads response from shared memory.
    fn read_shared_memory(&self) -> Vec<u8> {
        let Some(vaddr) = self.shared_vaddr else {
            return Vec::new();
        };
        let Some(cr3) = self.shared_cr3 else {
            return Vec::new();
        };
        let process = self.vm.process_memory(cr3);

        // Read length from first 8 bytes
        let len = process
            .read_u64(vaddr + SHARED_RESP_LEN_OFFSET)
            .unwrap_or(0) as usize;
        let max_len = self
            .config
            .shared_region_size
            .saturating_sub(SHARED_PAYLOAD_OFFSET as usize);
        let read_len = len.min(max_len);

        if read_len == 0 {
            return Vec::new();
        }

        let mut buf = vec![0u8; read_len];
        let _ = process.read_bytes(vaddr + SHARED_PAYLOAD_OFFSET, &mut buf);
        buf
    }

    fn clear_shared_length(&self) {
        let (Some(vaddr), Some(cr3)) = (self.shared_vaddr, self.shared_cr3) else {
            return;
        };
        let process = self.vm.process_memory(cr3);
        let _ = process.write_u64(vaddr + SHARED_ACTION_LEN_OFFSET, 0);
        let _ = process.write_u64(vaddr + SHARED_RESP_LEN_OFFSET, 0);
    }

    /// Runs a single step, returning detailed results.
    pub fn run_step(&mut self, payload: &[u8]) -> anyhow::Result<NyxStepResult> {
        // Write action to shared memory
        self.write_action_to_shared_memory(payload)?;

        // Run the VM until we get a meaningful exit
        let start = Instant::now();
        let mut output = Vec::new();
        let mut trace_data = Vec::new();
        let mut parsed_obs = None;
        let mut parsed_rew = None;
        let mut done = false;
        let exit_kind;
        let collect_output =
            matches!(
                self.config.observation_policy,
                NyxObservationPolicy::OutputHash | NyxObservationPolicy::RawOutput
            ) || matches!(self.config.reward_policy, NyxRewardPolicy::Pattern { .. })
                || matches!(
                    self.reward_shaping,
                    Some(NyxRewardShaping::EntropyReduction { .. })
                );

        loop {
            let remaining = self
                .config
                .step_timeout
                .checked_sub(start.elapsed())
                .unwrap_or(Duration::ZERO);

            if remaining.is_zero() {
                exit_kind = NyxExitKind::Timeout;
                break;
            }

            let exit = self.vm.run(remaining);
            match exit {
                ExitReason::ExecDone(code) => {
                    exit_kind = NyxExitKind::ExecDone(code);
                    done = true;
                    break;
                }
                ExitReason::Timeout => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Step timeout");
                    }
                    exit_kind = NyxExitKind::Timeout;
                    break;
                }
                ExitReason::Shutdown => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] VM shutdown during step");
                    }
                    exit_kind = NyxExitKind::Shutdown;
                    done = true;
                    break;
                }
                ExitReason::DebugPrint(msg) => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Guest: {}", msg);
                    }
                    // Accumulate debug output
                    if collect_output {
                        output.extend_from_slice(msg.as_bytes());
                    }
                    // Continue running
                }
                ExitReason::Hypercall(r8, r9, r10, r11, r12) => {
                    exit_kind = NyxExitKind::Hypercall {
                        code: r8,
                        arg1: r9,
                        arg2: r10,
                        arg3: r11,
                        arg4: r12,
                    };
                    // Attempt to parse structured response
                    if let Some(obs) = Self::try_parse_u64(r9) {
                        parsed_obs = Some(obs);
                    }
                    if let Some(rew) = Self::try_parse_i64(r10) {
                        parsed_rew = Some(rew);
                    }
                    break;
                }
                ExitReason::Breakpoint => {
                    if self.config.debug_mode {
                        eprintln!("[NyxVm] Breakpoint exit during step");
                    }
                    exit_kind = NyxExitKind::Breakpoint;
                    break;
                }
                _ => {
                    // Continue for other exits
                }
            }
        }

        // Read shared memory contents (only if needed)
        let need_shared_memory = matches!(
            self.config.observation_policy,
            NyxObservationPolicy::SharedMemory
        ) || matches!(
            self.config.reward_policy,
            NyxRewardPolicy::Pattern { .. }
        ) || matches!(
            self.reward_shaping,
            Some(NyxRewardShaping::EntropyReduction { .. })
        ) || self.config.trace.is_some();
        let shared_memory = if need_shared_memory {
            self.read_shared_memory()
        } else {
            Vec::new()
        };

        // Clear shared length to avoid host/guest races on the next step.
        self.clear_shared_length();

        // Collect trace data if configured
        if let Some(trace_cfg) = &self.config.trace {
            if trace_cfg.shared_region_name.is_some() {
                // Read from trace shared memory region (implementation-specific)
                // For now, use main shared memory as fallback
                trace_data = shared_memory.clone();
                if trace_data.len() > trace_cfg.max_bytes {
                    trace_data.truncate(trace_cfg.max_bytes);
                }
            }
        }

        Ok(NyxStepResult {
            exit_reason: exit_kind,
            output,
            parsed_obs,
            parsed_rew,
            done,
            trace_data,
            shared_memory,
        })
    }

    fn try_parse_u64(val: u64) -> Option<u64> {
        // Hypercall args are already u64
        Some(val)
    }

    fn try_parse_i64(val: u64) -> Option<i64> {
        Some(val as i64)
    }

    /// Gets the action payload for the given action index.
    fn get_action_payload(&mut self, action: Action) -> anyhow::Result<Cow<'_, [u8]>> {
        match &self.config.action_source {
            NyxActionSource::Literal(actions) => {
                let idx = action as usize;
                if idx >= actions.len() {
                    return Err(anyhow::anyhow!("Action index out of range"));
                }
                Ok(Cow::Borrowed(actions[idx].payload.as_slice()))
            }
            NyxActionSource::Fuzz(fuzz) => {
                let state = self
                    .fuzz_state
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Fuzz state missing"))?;
                let idx = action as usize % fuzz.mutators.len();
                let mut input = state.current.clone();
                let mutator = &fuzz.mutators[idx];
                apply_mutator(mutator, &mut input, fuzz, &mut state.rng);
                if input.len() < fuzz.min_len {
                    input.resize(fuzz.min_len, 0);
                }
                if input.len() > fuzz.max_len {
                    input.truncate(fuzz.max_len);
                }
                state.current = input.clone();
                Ok(Cow::Owned(input))
            }
        }
    }

    /// Applies action filtering, returning reject reward if filtered.
    fn filter_action(&self, payload: &[u8]) -> Option<i64> {
        let filter = self.config.action_filter.as_ref()?;
        if payload.is_empty() {
            return filter.reject_reward;
        }

        let (entropy, intrinsic, novelty) = self.compute_filter_metrics(payload, filter);

        if let Some(min_entropy) = filter.min_entropy {
            if entropy < min_entropy {
                return filter.reject_reward;
            }
        }
        if let Some(max_entropy) = filter.max_entropy {
            if entropy > max_entropy {
                return filter.reject_reward;
            }
        }
        if let Some(min_intrinsic) = filter.min_intrinsic_dependence {
            if intrinsic < min_intrinsic {
                return filter.reject_reward;
            }
        }
        if let Some(min_novelty) = filter.min_novelty {
            if filter.novelty_prior.is_some() && novelty < min_novelty {
                return filter.reject_reward;
            }
        }
        None
    }

    fn wrap_action_payload(&self, payload: &[u8]) -> Vec<u8> {
        let p = &self.config.protocol;
        let mut wrapped = p.action_prefix.clone().into_bytes();
        wrapped.extend_from_slice(p.wire_encoding.encode(payload).as_bytes());
        wrapped.extend_from_slice(p.action_suffix.as_bytes());
        wrapped
    }

    fn compute_filter_metrics(&self, payload: &[u8], filter: &NyxActionFilter) -> (f64, f64, f64) {
        let h_marg = marginal_entropy_bytes(payload);
        let h_rate = if filter.max_order == 0 {
            h_marg
        } else {
            entropy_rate_backend(payload, filter.max_order, &self.config.stats_backend)
        };

        let intrinsic = if h_marg < 1e-9 {
            0.0
        } else {
            ((h_marg - h_rate) / h_marg).clamp(0.0, 1.0)
        };

        let novelty = if let Some(ref prior) = filter.novelty_prior {
            cross_entropy_rate_backend(payload, prior, filter.max_order, &self.config.stats_backend)
        } else {
            0.0
        };

        (h_rate, intrinsic, novelty)
    }

    /// Computes reward from step result.
    fn compute_reward(&mut self, result: &NyxStepResult) -> Reward {
        let base_reward = match &self.config.reward_policy {
            NyxRewardPolicy::FromGuest => result.parsed_rew.unwrap_or(0),
            NyxRewardPolicy::Pattern {
                pattern,
                base_reward,
                bonus_reward,
            } => {
                let text = String::from_utf8_lossy(&result.output);
                let shared_text = String::from_utf8_lossy(&result.shared_memory);
                if text.contains(pattern) || shared_text.contains(pattern) {
                    base_reward + bonus_reward
                } else {
                    *base_reward
                }
            }
            NyxRewardPolicy::Custom(f) => f(result),
        };

        let shaping_reward = if let Some(shaping) = self.reward_shaping.clone() {
            self.compute_reward_shaping(&shaping, result)
        } else {
            0
        };

        let mut reward = base_reward.saturating_add(shaping_reward);

        reward = reward.saturating_sub(self.config.step_cost);
        let min_reward = self.min_reward();
        let max_reward = self.max_reward();
        reward.clamp(min_reward, max_reward)
    }

    fn compute_reward_shaping(
        &mut self,
        shaping: &NyxRewardShaping,
        result: &NyxStepResult,
    ) -> Reward {
        match shaping {
            NyxRewardShaping::EntropyReduction {
                max_order,
                scale,
                crash_bonus,
                timeout_bonus,
                ..
            } => {
                let mut base_reward = {
                    let data = if result.shared_memory.is_empty() {
                        &result.output
                    } else {
                        &result.shared_memory
                    };
                    let h_obs = if *max_order == 0 {
                        marginal_entropy_bytes(data)
                    } else {
                        entropy_rate_backend(data, *max_order, &self.config.stats_backend)
                    };
                    let h_base = self.baseline_entropy.unwrap_or(0.0);
                    let er = (h_base - h_obs) * scale;
                    er.round() as i64
                };

                // Add bonuses for interesting behaviors (bugs/crashes)
                match &result.exit_reason {
                    NyxExitKind::Shutdown | NyxExitKind::Breakpoint => {
                        if let Some(bonus) = crash_bonus {
                            base_reward = base_reward.saturating_add(*bonus);
                        }
                    }
                    NyxExitKind::Timeout => {
                        if let Some(bonus) = timeout_bonus {
                            base_reward = base_reward.saturating_add(*bonus);
                        }
                    }
                    _ => {}
                }

                base_reward
            }
            NyxRewardShaping::TraceEntropy {
                scale, normalize, ..
            } => {
                let data = &result.trace_data;
                let bits = match self.trace_model.as_mut() {
                    Some(model) => model.update_and_score(data),
                    None => 0.0,
                };
                let bits = if *normalize && !data.is_empty() {
                    bits / data.len() as f64
                } else {
                    bits
                };
                (bits * scale).round() as i64
            }
        }
    }

    fn mask_observation(&self, value: u64) -> u64 {
        let bits = self.config.observation_bits;
        if bits >= 64 {
            value
        } else if bits == 0 {
            0
        } else {
            value & ((1u64 << bits) - 1)
        }
    }

    fn build_observation_stream(&self, result: &NyxStepResult) -> Vec<PerceptVal> {
        let mut observations = match self.config.observation_policy {
            NyxObservationPolicy::FromGuest => {
                if let Some(obs) = result.parsed_obs {
                    vec![self.mask_observation(obs)]
                } else {
                    vec![self.hash_observation(&result.shared_memory)]
                }
            }
            NyxObservationPolicy::OutputHash => {
                vec![self.hash_observation(&result.output)]
            }
            NyxObservationPolicy::RawOutput => {
                result.output.iter().map(|b| *b as PerceptVal).collect()
            }
            NyxObservationPolicy::SharedMemory => result
                .shared_memory
                .iter()
                .map(|b| *b as PerceptVal)
                .collect(),
        };

        if observations.is_empty() {
            observations.push(0);
        }

        self.normalize_observation_stream(&mut observations);
        observations
    }

    fn hash_observation(&self, data: &[u8]) -> PerceptVal {
        let h = robust_hash_bytes(data);
        self.mask_observation(h)
    }

    fn normalize_observation_stream(&self, observations: &mut Vec<PerceptVal>) {
        let mask = if self.config.observation_bits >= 64 {
            u64::MAX
        } else if self.config.observation_bits == 0 {
            0
        } else {
            (1u64 << self.config.observation_bits) - 1
        };

        for obs in observations.iter_mut() {
            *obs &= mask;
        }

        let target = self.config.observation_stream_len;
        if target == 0 {
            return;
        }

        if observations.len() > target {
            match self.config.observation_stream_mode {
                NyxObservationStreamMode::Truncate | NyxObservationStreamMode::PadTruncate => {
                    observations.truncate(target);
                }
                NyxObservationStreamMode::Pad => {}
            }
        } else if observations.len() < target {
            match self.config.observation_stream_mode {
                NyxObservationStreamMode::Pad | NyxObservationStreamMode::PadTruncate => {
                    let pad = self.config.observation_pad_byte as PerceptVal;
                    observations.resize(target, pad);
                }
                NyxObservationStreamMode::Truncate => {}
            }
        }
    }

    fn action_count(&self) -> usize {
        match &self.config.action_source {
            NyxActionSource::Literal(actions) => actions.len(),
            NyxActionSource::Fuzz(fuzz) => fuzz.mutators.len(),
        }
    }

    /// Direct access to the underlying NyxVM for advanced use cases.
    pub fn vm(&self) -> &NyxVM {
        &self.vm
    }

    /// Mutable access to the underlying NyxVM.
    pub fn vm_mut(&mut self) -> &mut NyxVM {
        &mut self.vm
    }

    /// Takes a new snapshot at the current state.
    pub fn take_snapshot(&mut self) -> Arc<NyxSnapshot> {
        self.vm.take_snapshot()
    }

    /// Applies a specific snapshot.
    pub fn apply_snapshot(&mut self, snapshot: &Arc<NyxSnapshot>) {
        self.vm.apply_snapshot(snapshot);
    }

    /// Resets trace model.
    pub fn reset_trace_model(&mut self) {
        if let Some(model) = &mut self.trace_model {
            model.reset();
        }
    }

    /// Logs crashes and interesting behaviors to file.
    fn log_crash(&self, action_payload: &[u8], result: &NyxStepResult, reward: i64) {
        let Some(log_path) = &self.config.crash_log else {
            return;
        };

        // Only log interesting exits
        let is_interesting = matches!(
            result.exit_reason,
            NyxExitKind::Shutdown | NyxExitKind::Breakpoint | NyxExitKind::Timeout
        );

        if !is_interesting {
            return;
        }

        let log_entry = serde_json::json!({
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "exit_reason": format!("{:?}", result.exit_reason),
            "action_payload": hex_encode(action_payload),
            "action_payload_str": String::from_utf8_lossy(action_payload),
            "output": String::from_utf8_lossy(&result.output),
            "shared_memory": hex_encode(&result.shared_memory),
            "reward": reward,
            "parsed_obs": result.parsed_obs,
            "parsed_rew": result.parsed_rew,
        });

        // Append to JSONL file
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
            if let Ok(json_str) = serde_json::to_string(&log_entry) {
                let _ = writeln!(file, "{}", json_str);
            }
        }
    }
}

// ============================================================================
// Environment Trait Implementation
// ============================================================================

impl Environment for NyxVmEnvironment {
    fn perform_action(&mut self, action: Action) {
        if self.needs_reset {
            if let Err(e) = self.reset() {
                if self.config.debug_mode {
                    eprintln!("[NyxVm] Reset failed: {}", e);
                }
            }
        }

        let payload = match self.get_action_payload(action) {
            Ok(payload) => payload.into_owned(),
            Err(e) => {
                if self.config.debug_mode {
                    eprintln!("[NyxVm] Invalid action: {}", e);
                }
                self.obs = 0;
                self.rew = self.min_reward();
                self.obs_stream.clear();
                self.obs_stream.push(0);
                self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
                if self.step_in_episode == 0 {
                    self.needs_reset = true;
                }
                return;
            }
        };

        // Check action filter
        if let Some(reject_reward) = self.filter_action(&payload) {
            self.obs = 0;
            self.rew = reject_reward.clamp(self.min_reward(), self.max_reward());
            self.obs_stream.clear();
            self.obs_stream.push(0);
            self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
            if self.step_in_episode == 0 {
                self.needs_reset = true;
            }
            return;
        }

        // Run the step
        let wrapped_payload = self.wrap_action_payload(&payload);
        let result = match self.run_step(&wrapped_payload) {
            Ok(result) => result,
            Err(e) => {
                if self.config.debug_mode {
                    eprintln!("[NyxVm] Step failed: {}", e);
                }
                self.obs = 0;
                self.rew = self.min_reward();
                self.obs_stream.clear();
                self.obs_stream.push(0);
                self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
                if self.step_in_episode == 0 {
                    self.needs_reset = true;
                }
                return;
            }
        };

        // Process results
        self.obs_stream = self.build_observation_stream(&result);
        self.obs = self.obs_stream.first().copied().unwrap_or(0);
        self.rew = self.compute_reward(&result);

        // Log crashes and interesting behaviors
        self.log_crash(&payload, &result, self.rew);

        if self.config.debug_mode {
            eprintln!(
                "[NyxVm] Action={} Obs={} Rew={} Done={:?} Exit={:?}",
                action, self.obs, self.rew, result.done, result.exit_reason
            );
        }

        self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
        if self.step_in_episode == 0 || result.done {
            self.needs_reset = true;
        }
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }

    fn drain_observations(&mut self) -> Vec<PerceptVal> {
        if self.obs_stream.is_empty() {
            vec![self.obs]
        } else {
            std::mem::take(&mut self.obs_stream)
        }
    }

    fn get_reward(&self) -> Reward {
        self.rew
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        self.config.observation_bits
    }

    fn get_reward_bits(&self) -> usize {
        self.config.reward_bits
    }

    fn get_action_bits(&self) -> usize {
        let n = self.action_count();
        if n <= 1 {
            return 1;
        }
        (n as f64).log2().ceil() as usize
    }

    fn get_num_actions(&self) -> usize {
        self.action_count()
    }

    fn max_reward(&self) -> Reward {
        let bits = self.config.reward_bits;
        if bits >= 64 {
            i64::MAX
        } else if bits == 0 {
            0
        } else {
            (1i64 << (bits - 1)) - 1
        }
    }

    fn min_reward(&self) -> Reward {
        let bits = self.config.reward_bits;
        if bits >= 64 {
            i64::MIN
        } else if bits == 0 {
            0
        } else {
            -(1i64 << (bits - 1))
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn robust_hash_bytes(data: &[u8]) -> u64 {
    let mut h = 0u64;
    for &b in data {
        h = h.rotate_left(7) ^ (b as u64);
    }
    h
}

fn apply_mutator(
    mutator: &FuzzMutator,
    input: &mut Vec<u8>,
    fuzz: &NyxFuzzConfig,
    rng: &mut RandomGenerator,
) {
    match mutator {
        FuzzMutator::FlipBit => {
            if input.is_empty() {
                input.push(0);
            }
            let idx = rng.gen_range(input.len());
            let bit = rng.gen_range(8);
            input[idx] ^= 1u8 << bit;
        }
        FuzzMutator::FlipByte => {
            if input.is_empty() {
                input.push(0);
            }
            let idx = rng.gen_range(input.len());
            input[idx] ^= rng.next_u64() as u8;
        }
        FuzzMutator::InsertByte => {
            let idx = if input.is_empty() {
                0
            } else {
                rng.gen_range(input.len() + 1)
            };
            let byte = if !fuzz.dictionary.is_empty() {
                let d = rng.gen_range(fuzz.dictionary.len());
                let entry = &fuzz.dictionary[d];
                if entry.is_empty() {
                    0
                } else {
                    entry[rng.gen_range(entry.len())]
                }
            } else {
                rng.next_u64() as u8
            };
            input.insert(idx, byte);
        }
        FuzzMutator::DeleteByte => {
            if input.len() > 1 {
                let idx = rng.gen_range(input.len());
                input.remove(idx);
            }
        }
        FuzzMutator::SpliceSeed => {
            if fuzz.seeds.is_empty() {
                return;
            }
            let seed = &fuzz.seeds[rng.gen_range(fuzz.seeds.len())];
            if input.is_empty() {
                input.extend_from_slice(seed);
            } else if !seed.is_empty() {
                let cut = rng.gen_range(input.len());
                let seed_cut = rng.gen_range(seed.len());
                let mut out = Vec::new();
                out.extend_from_slice(&input[..cut]);
                out.extend_from_slice(&seed[seed_cut..]);
                *input = out;
            }
        }
        FuzzMutator::ResetSeed => {
            if fuzz.seeds.is_empty() {
                return;
            }
            *input = fuzz.seeds[rng.gen_range(fuzz.seeds.len())].clone();
        }
        FuzzMutator::Havoc => {
            let flips = 1 + rng.gen_range(8);
            for _ in 0..flips {
                if input.is_empty() {
                    input.push(0);
                }
                let idx = rng.gen_range(input.len());
                input[idx] ^= rng.next_u64() as u8;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_encoding() {
        let data = b"hello";
        let encoded = hex_encode(data);
        assert_eq!(encoded, "68656c6c6f");
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_robust_hash() {
        let data1 = b"test data";
        let data2 = b"test data";
        let data3 = b"different";

        assert_eq!(robust_hash_bytes(data1), robust_hash_bytes(data2));
        assert_ne!(robust_hash_bytes(data1), robust_hash_bytes(data3));
    }

    #[test]
    fn test_payload_encoding() {
        let utf8 = PayloadEncoding::Utf8;
        let hex = PayloadEncoding::Hex;

        let data = b"test";
        assert_eq!(utf8.encode(data), "test");
        assert_eq!(hex.encode(data), "74657374");

        assert_eq!(utf8.decode("test").unwrap(), data);
        assert_eq!(hex.decode("74657374").unwrap(), data);
    }
}
