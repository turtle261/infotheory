//! # InfoTheory CLI
//!
//! Command-line interface for the `infotheory` library.
//! Provides access to compression-based (NCD) and entropy-based (Shannon, ROSA, CTW)
//! estimators for files, as well as AIXI agents.
//!
//! ## Usage
//!
//! ### Single-file mode:
//! ```bash
//! infotheory <primitive> <file1> <file2> [method/max_order]
//! ```
//!
//! ### Search mode:
//! ```bash
//! infotheory search <query> <target> [options]
//! ```
//!
//! ### AIXI Agent mode:
//! ```bash
//! infotheory aixi <config.json>
//! ```
//!
//! ### Batch JSON mode (for programmatic use):
//! ```bash
//! infotheory batch < input.json > output.json
//! echo '{"op":"metrics","text":"hello world"}' | infotheory batch
//! ```
//!
//! See `print_usage` for details on supported primitives.

use infotheory::aixi::agent::{Agent, AgentConfig};
use infotheory::aixi::common::{ObservationKeyMode, RandomGenerator};
use infotheory::aixi::environment::{
    BiasedRockPaperScissor, CoinFlip, CtwTest, Environment, ExtendedTiger, KuhnPoker, TicTacToe,
};
#[cfg(feature = "vm")]
use infotheory::aixi::vm_nyx::{
    FuzzMutator as NyxFuzzMutator, NyxActionFilter, NyxActionSource, NyxActionSpec, NyxFuzzConfig,
    NyxObservationPolicy, NyxObservationStreamMode, NyxProtocolConfig, NyxRewardPolicy,
    NyxRewardShaping, NyxTraceConfig, NyxVmConfig, NyxVmEnvironment,
    PayloadEncoding as NyxPayloadEncoding,
};
use infotheory::*;
#[cfg(feature = "vm")]
use nyx_lite::SharedMemoryPolicy;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, IsTerminal, Read, Write};
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "vm")]
use std::time::{Duration, Instant};

#[cfg(not(feature = "vm"))]
use std::time::Instant;

use infotheory::search;

struct AixiRunLogger {
    bits01: Option<BufWriter<File>>,
    jsonl: Option<BufWriter<File>>,
    flush_every: usize,
    step: usize,
}

impl AixiRunLogger {
    fn new(v: &serde_json::Value) -> anyhow::Result<Option<Self>> {
        let bits01_path = v["trace_bits01_path"].as_str();
        let jsonl_path = v["trace_jsonl_path"].as_str();
        if bits01_path.is_none() && jsonl_path.is_none() {
            return Ok(None);
        }

        let bits01 = if let Some(p) = bits01_path {
            let f = File::create(p)?;
            Some(BufWriter::new(f))
        } else {
            None
        };
        let jsonl = if let Some(p) = jsonl_path {
            let f = File::create(p)?;
            Some(BufWriter::new(f))
        } else {
            None
        };
        let flush_every = v["trace_flush_every"].as_u64().unwrap_or(1024) as usize;

        Ok(Some(Self {
            bits01,
            jsonl,
            flush_every,
            step: 0,
        }))
    }

    fn write_bits01(&mut self, bits: &[bool]) -> anyhow::Result<()> {
        if let Some(w) = self.bits01.as_mut() {
            for &b in bits {
                w.write_all(&[if b { 1u8 } else { 0u8 }])?;
            }
        }
        Ok(())
    }

    fn log_percept(
        &mut self,
        observations: &[u64],
        reward: i64,
        observation_bits: usize,
        reward_bits: usize,
        reward_offset: i64,
    ) -> anyhow::Result<()> {
        // Exact same bit encoding the agent uses internally.
        let mut bits = Vec::new();
        for &obs in observations {
            infotheory::aixi::common::encode(&mut bits, obs, observation_bits);
        }
        infotheory::aixi::common::encode_reward_offset(
            &mut bits,
            reward,
            reward_bits,
            reward_offset,
        );

        self.write_bits01(&bits)?;

        if let Some(w) = self.jsonl.as_mut() {
            let rec = serde_json::json!({
                "t": self.step,
                "kind": "percept",
                "observations": observations,
                "reward": reward,
            });
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    fn log_action(&mut self, action: u64, action_bits: usize) -> anyhow::Result<()> {
        let mut bits = Vec::new();
        infotheory::aixi::common::encode(&mut bits, action, action_bits);
        self.write_bits01(&bits)?;

        if let Some(w) = self.jsonl.as_mut() {
            let rec = serde_json::json!({
                "t": self.step,
                "kind": "action",
                "action": action,
            });
            writeln!(w, "{rec}")?;
        }
        Ok(())
    }

    fn next_step(&mut self) -> anyhow::Result<()> {
        self.step = self.step.saturating_add(1);
        if self.flush_every > 0 && self.step.is_multiple_of(self.flush_every) {
            if let Some(w) = self.bits01.as_mut() {
                w.flush()?;
            }
            if let Some(w) = self.jsonl.as_mut() {
                w.flush()?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "backend-rwkv")]
fn rwkv7_model_path_from_env() -> String {
    env::var("RWKV7_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("Error: RWKV7_MODEL_PATH env var must be set when using rwkv7 backends");
        std::process::exit(1);
    })
}

#[cfg(feature = "backend-mamba")]
fn mamba_model_path_from_env() -> String {
    env::var("MAMBA_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("Error: MAMBA_MODEL_PATH env var must be set when using mamba backends");
        std::process::exit(1);
    })
}

fn parse_rate_backend(v: &str) -> Option<&'static str> {
    match infotheory::backends::resolve_rate_backend_name(v) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => Some(name),
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            eprintln!(
                "Error: rate backend '{canonical}' requires infotheory built with feature '{feature}'"
            );
            std::process::exit(1);
        }
        None => None,
    }
}

fn parse_compression_backend(v: &str) -> Option<&'static str> {
    match infotheory::backends::resolve_compression_backend_name(v) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => Some(name),
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            eprintln!(
                "Error: compression backend '{canonical}' requires infotheory built with feature '{feature}'"
            );
            std::process::exit(1);
        }
        None => None,
    }
}

const MAX_MIXTURE_NESTING: usize = 8;

fn load_mixture_spec(path: &str) -> anyhow::Result<MixtureSpec> {
    load_mixture_spec_with_depth(path, MAX_MIXTURE_NESTING)
}

fn load_mixture_spec_with_depth(path: &str, depth: usize) -> anyhow::Result<MixtureSpec> {
    let raw = std::fs::read(path)?;
    let value: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => {
            #[cfg(feature = "backend-zpaq")]
            {
                let decompressed = zpaq_rs::decompress_to_vec(&raw)?;
                serde_json::from_slice(&decompressed)?
            }
            #[cfg(not(feature = "backend-zpaq"))]
            {
                return Err(anyhow::anyhow!(
                    "Failed to parse mixture JSON, and zpaq support is disabled at compile time"
                ));
            }
        }
    };
    let base_dir = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    parse_mixture_spec_value(&value, base_dir, depth)
}

fn load_particle_spec(path: &str) -> anyhow::Result<ParticleSpec> {
    let raw = std::fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&raw)?;
    let spec = parse_particle_spec_value(&value)?;
    spec.validate()
        .map_err(|e| anyhow::anyhow!("invalid particle spec: {e}"))?;
    Ok(spec)
}

fn load_calibrated_spec(path: &str) -> anyhow::Result<CalibratedSpec> {
    let raw = std::fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&raw)?;
    let base_dir = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    parse_calibrated_spec_value(&value, base_dir, 4)
}

fn load_expert_spec(path: &str) -> anyhow::Result<MixtureExpertSpec> {
    let raw = std::fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&raw)?;
    let base_dir = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    parse_mixture_expert_value(&value, base_dir, MAX_MIXTURE_NESTING)
}

fn parse_particle_spec_value(v: &serde_json::Value) -> anyhow::Result<ParticleSpec> {
    if v.get("experts").is_some() {
        return Err(anyhow::anyhow!(
            "looks like a mixture spec (found 'experts'); --rate-backend particle expects a ParticleSpec JSON"
        ));
    }
    if let Some(kind) = v.get("kind").and_then(|k| k.as_str()) {
        let k = kind.to_ascii_lowercase();
        if matches!(
            k.as_str(),
            "bayes"
                | "fading"
                | "fading-bayes"
                | "switch"
                | "switching"
                | "mdl"
                | "neural"
                | "mixture"
        ) {
            return Err(anyhow::anyhow!(
                "looks like a mixture spec (kind='{kind}'); --rate-backend particle expects a ParticleSpec JSON"
            ));
        }
    }
    let d = ParticleSpec::default();
    Ok(ParticleSpec {
        num_particles: v["num_particles"]
            .as_u64()
            .unwrap_or(d.num_particles as u64) as usize,
        context_window: v["context_window"]
            .as_u64()
            .unwrap_or(d.context_window as u64) as usize,
        unroll_steps: v["unroll_steps"].as_u64().unwrap_or(d.unroll_steps as u64) as usize,
        num_cells: v["num_cells"].as_u64().unwrap_or(d.num_cells as u64) as usize,
        cell_dim: v["cell_dim"].as_u64().unwrap_or(d.cell_dim as u64) as usize,
        num_rules: v["num_rules"].as_u64().unwrap_or(d.num_rules as u64) as usize,
        selector_hidden: v["selector_hidden"]
            .as_u64()
            .unwrap_or(d.selector_hidden as u64) as usize,
        rule_hidden: v["rule_hidden"].as_u64().unwrap_or(d.rule_hidden as u64) as usize,
        noise_dim: v["noise_dim"].as_u64().unwrap_or(d.noise_dim as u64) as usize,
        deterministic: v["deterministic"].as_bool().unwrap_or(d.deterministic),
        enable_noise: v["enable_noise"].as_bool().unwrap_or(d.enable_noise),
        noise_scale: v["noise_scale"].as_f64().unwrap_or(d.noise_scale),
        noise_anneal_steps: v["noise_anneal_steps"]
            .as_u64()
            .unwrap_or(d.noise_anneal_steps as u64) as usize,
        learning_rate_readout: v["learning_rate_readout"]
            .as_f64()
            .unwrap_or(d.learning_rate_readout),
        learning_rate_selector: v["learning_rate_selector"]
            .as_f64()
            .unwrap_or(d.learning_rate_selector),
        learning_rate_rule: v["learning_rate_rule"]
            .as_f64()
            .unwrap_or(d.learning_rate_rule),
        bptt_depth: v["bptt_depth"].as_u64().unwrap_or(d.bptt_depth as u64) as usize,
        optimizer_momentum: v["optimizer_momentum"]
            .as_f64()
            .unwrap_or(d.optimizer_momentum),
        grad_clip: v["grad_clip"].as_f64().unwrap_or(d.grad_clip),
        state_clip: v["state_clip"].as_f64().unwrap_or(d.state_clip),
        forget_lambda: v["forget_lambda"].as_f64().unwrap_or(d.forget_lambda),
        resample_threshold: v["resample_threshold"]
            .as_f64()
            .unwrap_or(d.resample_threshold),
        mutate_fraction: v["mutate_fraction"].as_f64().unwrap_or(d.mutate_fraction),
        mutate_scale: v["mutate_scale"].as_f64().unwrap_or(d.mutate_scale),
        mutate_model_params: v["mutate_model_params"]
            .as_bool()
            .unwrap_or(d.mutate_model_params),
        diagnostics_interval: v["diagnostics_interval"]
            .as_u64()
            .unwrap_or(d.diagnostics_interval as u64) as usize,
        min_prob: v["min_prob"].as_f64().unwrap_or(d.min_prob),
        seed: v["seed"].as_u64().unwrap_or(d.seed),
    })
}

fn parse_calibration_context_kind(value: Option<&str>) -> anyhow::Result<CalibrationContextKind> {
    match value.unwrap_or("text").trim().to_ascii_lowercase().as_str() {
        "global" => Ok(CalibrationContextKind::Global),
        "byteclass" | "byte-class" | "byte_class" => Ok(CalibrationContextKind::ByteClass),
        "text" => Ok(CalibrationContextKind::Text),
        "repeat" => Ok(CalibrationContextKind::Repeat),
        "textrepeat" | "text-repeat" | "text_repeat" => Ok(CalibrationContextKind::TextRepeat),
        other => Err(anyhow::anyhow!("unknown calibration context '{other}'")),
    }
}

fn parse_calibrated_spec_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> anyhow::Result<CalibratedSpec> {
    if depth == 0 {
        return Err(anyhow::anyhow!("calibrated spec nesting too deep"));
    }
    let base_backend = if let Some(base_v) = v.get("base") {
        parse_mixture_expert_value(base_v, base_dir, depth - 1)?.backend
    } else if let Some(path) = v["base_path"].as_str().or_else(|| v["path"].as_str()) {
        let full = base_dir.join(path);
        let raw = std::fs::read(&full)?;
        let value: serde_json::Value = serde_json::from_slice(&raw)?;
        parse_mixture_expert_value(&value, full.parent().unwrap_or(base_dir), depth - 1)?.backend
    } else {
        return Err(anyhow::anyhow!(
            "calibrated expert requires 'base' or 'base_path'"
        ));
    };
    Ok(CalibratedSpec {
        base: base_backend,
        context: parse_calibration_context_kind(v["context"].as_str())?,
        bins: v["bins"].as_u64().unwrap_or(33) as usize,
        learning_rate: v["learning_rate"].as_f64().unwrap_or(0.02),
        bias_clip: v["bias_clip"].as_f64().unwrap_or(4.0),
    })
}

fn parse_mixture_kind(kind: &str) -> anyhow::Result<MixtureKind> {
    match kind {
        "bayes" | "bayes-mix" | "bayes_mix" => Ok(MixtureKind::Bayes),
        "fading" | "fading-bayes" | "fading_bayes" => Ok(MixtureKind::FadingBayes),
        "switch" | "switching" | "switch-mix" | "switch_mix" => Ok(MixtureKind::Switching),
        "mdl" | "selector" | "mdr" => Ok(MixtureKind::Mdl),
        "neural" | "neural-mix" | "neural_mix" | "fx2" | "fx2-cmix" | "fx2_cmix" => {
            Ok(MixtureKind::Neural)
        }
        other => Err(anyhow::anyhow!("unknown mixture kind '{other}'")),
    }
}

fn parse_mixture_spec_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> anyhow::Result<MixtureSpec> {
    if depth == 0 {
        return Err(anyhow::anyhow!("mixture spec nesting too deep"));
    }
    let kind_str = v["kind"]
        .as_str()
        .or_else(|| v["mixture_kind"].as_str())
        .or_else(|| v["mix_kind"].as_str())
        .unwrap_or("bayes");
    let kind = parse_mixture_kind(kind_str)?;
    let alpha = v["alpha"].as_f64().unwrap_or(0.01);
    let decay = v["decay"].as_f64();
    let experts_v = v["experts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("mixture spec missing 'experts' array"))?;
    if experts_v.is_empty() {
        return Err(anyhow::anyhow!(
            "mixture spec must include at least one expert"
        ));
    }
    let mut experts = Vec::with_capacity(experts_v.len());
    for e in experts_v {
        experts.push(parse_mixture_expert_value(e, base_dir, depth - 1)?);
    }
    let mut spec = MixtureSpec::new(kind, experts).with_alpha(alpha);
    if let Some(decay) = decay {
        spec = spec.with_decay(decay);
    }
    if matches!(kind, MixtureKind::FadingBayes) && spec.decay.is_none() {
        return Err(anyhow::anyhow!(
            "fading Bayes mixture requires 'decay' in mixture spec"
        ));
    }
    Ok(spec)
}

fn parse_mixture_expert_value(
    v: &serde_json::Value,
    base_dir: &Path,
    depth: usize,
) -> anyhow::Result<MixtureExpertSpec> {
    if depth == 0 {
        return Err(anyhow::anyhow!("mixture spec nesting too deep"));
    }
    let raw_kind = v["kind"]
        .as_str()
        .or_else(|| v["type"].as_str())
        .or_else(|| v["backend"].as_str())
        .ok_or_else(|| anyhow::anyhow!("expert missing 'kind'"))?;
    let kind = match infotheory::backends::resolve_rate_backend_name(raw_kind) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => name,
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            return Err(anyhow::anyhow!(
                "expert backend '{canonical}' requires infotheory feature '{feature}'"
            ));
        }
        None => return Err(anyhow::anyhow!("unknown expert kind '{raw_kind}'")),
    };
    let name = v["name"].as_str().map(|s| s.to_string());
    let log_prior = v["log_prior"]
        .as_f64()
        .or_else(|| v["prior"].as_f64())
        .unwrap_or(0.0);

    match kind {
        "rosaplus" => {
            let max_order = v["max_order"]
                .as_i64()
                .or_else(|| v["order"].as_i64())
                .unwrap_or(8);
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order,
                backend: RateBackend::RosaPlus,
            })
        }
        "match" => Ok(MixtureExpertSpec {
            name,
            log_prior,
            max_order: -1,
            backend: RateBackend::Match {
                hash_bits: v["hash_bits"].as_u64().unwrap_or(20) as usize,
                min_len: v["min_len"].as_u64().unwrap_or(4) as usize,
                max_len: v["max_len"].as_u64().unwrap_or(255) as usize,
                base_mix: v["base_mix"].as_f64().unwrap_or(0.02),
                confidence_scale: v["confidence_scale"].as_f64().unwrap_or(1.0),
            },
        }),
        "sparse-match" => Ok(MixtureExpertSpec {
            name,
            log_prior,
            max_order: -1,
            backend: RateBackend::SparseMatch {
                hash_bits: v["hash_bits"].as_u64().unwrap_or(19) as usize,
                min_len: v["min_len"].as_u64().unwrap_or(3) as usize,
                max_len: v["max_len"].as_u64().unwrap_or(64) as usize,
                gap_min: v["gap_min"].as_u64().unwrap_or(1) as usize,
                gap_max: v["gap_max"].as_u64().unwrap_or(2) as usize,
                base_mix: v["base_mix"].as_f64().unwrap_or(0.05),
                confidence_scale: v["confidence_scale"].as_f64().unwrap_or(1.0),
            },
        }),
        "ppmd" => Ok(MixtureExpertSpec {
            name,
            log_prior,
            max_order: -1,
            backend: RateBackend::Ppmd {
                order: v["order"].as_u64().unwrap_or(10) as usize,
                memory_mb: v["memory_mb"].as_u64().unwrap_or(64) as usize,
            },
        }),
        "calibrated" => {
            let spec = parse_calibrated_spec_value(v, base_dir, depth - 1)?;
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::Calibrated {
                    spec: Arc::new(spec),
                },
            })
        }
        "ctw" => {
            let depth = v["depth"]
                .as_u64()
                .or_else(|| v["ct_depth"].as_u64())
                .unwrap_or(16) as usize;
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::Ctw { depth },
            })
        }
        "fac-ctw" => {
            let base_depth = v["base_depth"]
                .as_u64()
                .or_else(|| v["ct_depth"].as_u64())
                .unwrap_or(16) as usize;
            let encoding_bits = v["encoding_bits"].as_u64().unwrap_or(8) as usize;
            let num_percept_bits = v["num_percept_bits"]
                .as_u64()
                .unwrap_or(encoding_bits as u64) as usize;
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::FacCtw {
                    base_depth,
                    num_percept_bits,
                    encoding_bits,
                },
            })
        }
        "zpaq" => {
            let method = v["method"].as_str().unwrap_or("2").to_string();
            if let Err(err) = validate_zpaq_rate_method(&method) {
                return Err(anyhow::anyhow!(
                    "unsupported ZPAQ rate method '{method}': {err}"
                ));
            }
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::Zpaq { method },
            })
        }
        "mamba" => {
            #[cfg(feature = "backend-mamba")]
            {
                if let Some(method) = v["method"].as_str().or_else(|| v["mamba_method"].as_str()) {
                    Ok(MixtureExpertSpec {
                        name,
                        log_prior,
                        max_order: -1,
                        backend: RateBackend::MambaMethod {
                            method: method.to_string(),
                        },
                    })
                } else {
                    let model_path = v["model_path"]
                        .as_str()
                        .or_else(|| v["mamba_model_path"].as_str())
                        .ok_or_else(|| {
                            anyhow::anyhow!("mamba expert missing model_path or method")
                        })?;
                    let model_path = base_dir.join(model_path);
                    let model = load_mamba_model_from_path(model_path.to_string_lossy().as_ref());
                    Ok(MixtureExpertSpec {
                        name,
                        log_prior,
                        max_order: -1,
                        backend: RateBackend::Mamba { model },
                    })
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                let _ = (name, log_prior);
                Err(anyhow::anyhow!(
                    "mamba expert requires 'backend-mamba' feature in infotheory"
                ))
            }
        }
        "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            {
                if let Some(method) = v["method"].as_str().or_else(|| v["rwkv_method"].as_str()) {
                    Ok(MixtureExpertSpec {
                        name,
                        log_prior,
                        max_order: -1,
                        backend: RateBackend::Rwkv7Method {
                            method: method.to_string(),
                        },
                    })
                } else {
                    let model_path = v["model_path"]
                        .as_str()
                        .or_else(|| v["rwkv_model_path"].as_str())
                        .ok_or_else(|| {
                            anyhow::anyhow!("rwkv expert missing model_path or method")
                        })?;
                    let model_path = base_dir.join(model_path);
                    let model = load_rwkv7_model_from_path(model_path.to_string_lossy().as_ref());
                    Ok(MixtureExpertSpec {
                        name,
                        log_prior,
                        max_order: -1,
                        backend: RateBackend::Rwkv7 { model },
                    })
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                let _ = (name, log_prior);
                Err(anyhow::anyhow!(
                    "rwkv expert requires 'backend-rwkv' feature in infotheory"
                ))
            }
        }
        "mixture" => {
            let spec = if let Some(spec_v) = v.get("spec") {
                parse_mixture_spec_value(spec_v, base_dir, depth - 1)?
            } else if let Some(path) = v["spec_path"].as_str().or_else(|| v["path"].as_str()) {
                let full = base_dir.join(path);
                load_mixture_spec_with_depth(full.to_str().unwrap_or(path), depth - 1)?
            } else {
                return Err(anyhow::anyhow!(
                    "mixture expert requires 'spec' (inline) or 'spec_path'"
                ));
            };
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::Mixture {
                    spec: Arc::new(spec),
                },
            })
        }
        "particle" => {
            let spec = if let Some(spec_v) = v.get("spec") {
                parse_particle_spec_value(spec_v)?
            } else if let Some(path) = v["spec_path"].as_str().or_else(|| v["path"].as_str()) {
                let full = base_dir.join(path);
                load_particle_spec(full.to_str().unwrap_or(path))?
            } else {
                ParticleSpec::default()
            };
            spec.validate()
                .map_err(|e| anyhow::anyhow!("invalid particle spec: {e}"))?;
            Ok(MixtureExpertSpec {
                name,
                log_prior,
                max_order: -1,
                backend: RateBackend::Particle {
                    spec: Arc::new(spec),
                },
            })
        }
        other => Err(anyhow::anyhow!("unsupported expert kind '{other}'")),
    }
}

#[cfg(feature = "vm")]
fn parse_shared_memory_policy(v: Option<&str>) -> SharedMemoryPolicy {
    match v.unwrap_or("snapshot") {
        "preserve" | "keep" => SharedMemoryPolicy::Preserve,
        _ => SharedMemoryPolicy::Snapshot,
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_environment_config(
    v: &serde_json::Value,
    observation_bits: usize,
    reward_bits: usize,
    agent_horizon: usize,
    base_dir: &Path,
) -> anyhow::Result<NyxVmConfig> {
    let vm = &v["vm_config"];
    if vm.is_null() {
        return Err(anyhow::anyhow!("vm_config is required for environment=vm"));
    }

    let firecracker_config = vm["firecracker_config"]
        .as_str()
        .or_else(|| vm["config"].as_str())
        .or_else(|| v["firecracker_config"].as_str())
        .ok_or_else(|| anyhow::anyhow!("vm_config.firecracker_config is required"))?
        .to_string();

    let instance_id = vm["instance_id"].as_str().unwrap_or("aixi-nyx").to_string();
    let shared_region_name = vm["shared_region_name"]
        .as_str()
        .unwrap_or("shared")
        .to_string();
    let shared_region_size = vm["shared_region_size"].as_u64().unwrap_or(4096) as usize;
    let shared_memory_policy = parse_shared_memory_policy(
        vm["shared_memory_policy"]
            .as_str()
            .or_else(|| v["shared_memory_policy"].as_str()),
    );

    let step_timeout_ms = vm["step_timeout_ms"].as_u64().unwrap_or(100);
    let boot_timeout_ms = vm["boot_timeout_ms"].as_u64().unwrap_or(30_000);
    let episode_steps = vm["episode_steps"].as_u64().unwrap_or(agent_horizon as u64) as usize;
    let step_cost = vm["step_cost"].as_i64().unwrap_or(1);
    let debug_mode = vm["verbose"]
        .as_bool()
        .or_else(|| vm["debug"].as_bool())
        .unwrap_or(false);

    let protocol = parse_nyx_protocol_config(if !vm["protocol"].is_null() {
        &vm["protocol"]
    } else {
        &v["vm_protocol"]
    });
    let stats_backend = parse_vm_stats_backend(
        if !vm["stats_backend"].is_null() {
            &vm["stats_backend"]
        } else {
            &v["vm_stats_backend"]
        },
        v,
        base_dir,
    )?;
    let trace = parse_nyx_trace_config(if !vm["trace"].is_null() {
        &vm["trace"]
    } else {
        &v["vm_trace"]
    })?;
    let action_source = parse_nyx_actions(if !vm["actions"].is_null() {
        &vm["actions"]
    } else {
        &v["vm_actions"]
    })?;
    let observation_policy = parse_nyx_observation_policy(if !vm["observation"].is_null() {
        &vm["observation"]
    } else {
        &v["vm_observation"]
    });
    let observation_stream_len =
        parse_observation_stream_len_for_vm(if !vm["observation"].is_null() {
            &vm["observation"]
        } else {
            &v["vm_observation"]
        });
    let observation_stream_mode =
        parse_nyx_observation_stream_mode(if !vm["observation"].is_null() {
            &vm["observation"]
        } else {
            &v["vm_observation"]
        });
    let observation_stream_pad_byte =
        parse_nyx_observation_pad_byte(if !vm["observation"].is_null() {
            &vm["observation"]
        } else {
            &v["vm_observation"]
        });
    let reward_policy = parse_nyx_reward_policy(if !vm["reward"].is_null() {
        &vm["reward"]
    } else {
        &v["vm_reward"]
    })?;
    let reward_shaping = if !vm["reward_shaping"].is_null() {
        parse_nyx_reward_shaping(&vm["reward_shaping"])?
    } else if !v["vm_reward_shaping"].is_null() {
        parse_nyx_reward_shaping(&v["vm_reward_shaping"])?
    } else if !vm["reward"].is_null() && !vm["reward"]["shaping"].is_null() {
        parse_nyx_reward_shaping(&vm["reward"]["shaping"])?
    } else {
        None
    };
    let action_filter = parse_nyx_filter(
        if !vm["filter"].is_null() {
            &vm["filter"]
        } else {
            &v["vm_filter"]
        },
        step_cost,
    )?;

    Ok(NyxVmConfig {
        firecracker_config,
        instance_id,
        shared_region_name,
        shared_region_size,
        shared_memory_policy,
        step_timeout: Duration::from_millis(step_timeout_ms),
        boot_timeout: Duration::from_millis(boot_timeout_ms),
        episode_steps,
        step_cost,
        observation_policy,
        observation_bits,
        observation_stream_len,
        observation_stream_mode,
        observation_pad_byte: observation_stream_pad_byte,
        reward_bits,
        reward_policy,
        reward_shaping,
        action_source,
        action_filter,
        protocol,
        stats_backend,
        trace,
        debug_mode,
        crash_log: vm["crash_log"].as_str().map(|s| s.to_string()),
    })
}

#[cfg(feature = "vm")]
fn parse_vm_stats_backend(
    cfg: &serde_json::Value,
    root: &serde_json::Value,
    base_dir: &Path,
) -> anyhow::Result<RateBackend> {
    let fallback = default_vm_stats_backend(root)?;
    if cfg.is_null() {
        return Ok(fallback);
    }

    let name = cfg
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| cfg.get("rate_backend").and_then(|v| v.as_str()))
        .or_else(|| cfg.as_str())
        .unwrap_or("rosaplus");

    let resolved = match infotheory::backends::resolve_rate_backend_name(name) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => name,
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            return Err(anyhow::anyhow!(
                "rate backend '{canonical}' requires infotheory feature '{feature}'"
            ));
        }
        None => return Err(anyhow::anyhow!("unknown vm stats backend '{name}'")),
    };

    match resolved {
        "rosaplus" => Ok(RateBackend::RosaPlus),
        "ctw" => {
            let depth = cfg["ct_depth"]
                .as_u64()
                .or_else(|| cfg["depth"].as_u64())
                .unwrap_or(32) as usize;
            Ok(RateBackend::Ctw { depth })
        }
        "fac-ctw" => {
            let base_depth = cfg["base_depth"]
                .as_u64()
                .or_else(|| cfg["ct_depth"].as_u64())
                .unwrap_or(32) as usize;
            let encoding_bits = cfg["encoding_bits"].as_u64().unwrap_or(8) as usize;

            // Fix: Default num_percept_bits to observation_bits + reward_bits if available,
            // fallback to encoding_bits if not.
            let obs_bits = root["observation_bits"].as_u64().unwrap_or(16);
            let rew_bits = root["reward_bits"].as_u64().unwrap_or(8);
            let default_percept_bits = obs_bits + rew_bits;

            let num_percept_bits = cfg["num_percept_bits"]
                .as_u64()
                .unwrap_or(default_percept_bits) as usize;

            Ok(RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
            })
        }
        "mamba" => {
            #[cfg(feature = "backend-mamba")]
            {
                if let Some(method) = cfg["method"]
                    .as_str()
                    .or_else(|| cfg["mamba_method"].as_str())
                {
                    Ok(RateBackend::MambaMethod {
                        method: method.to_string(),
                    })
                } else {
                    let path = cfg["mamba_model_path"]
                        .as_str()
                        .or_else(|| cfg["model_path"].as_str())
                        .or_else(|| root["mamba_model_path"].as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(mamba_model_path_from_env);
                    let model = load_mamba_model_from_path(&path);
                    Ok(RateBackend::Mamba { model })
                }
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                Err(anyhow::anyhow!(
                    "mamba stats backend requires 'backend-mamba' feature in infotheory"
                ))
            }
        }
        "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            {
                if let Some(method) = cfg["method"]
                    .as_str()
                    .or_else(|| cfg["rwkv_method"].as_str())
                {
                    Ok(RateBackend::Rwkv7Method {
                        method: method.to_string(),
                    })
                } else {
                    let path = cfg["rwkv_model_path"]
                        .as_str()
                        .or_else(|| cfg["model_path"].as_str())
                        .or_else(|| root["rwkv_model_path"].as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(rwkv7_model_path_from_env);
                    let model = load_rwkv7_model_from_path(&path);
                    Ok(RateBackend::Rwkv7 { model })
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(anyhow::anyhow!(
                    "rwkv7 stats backend requires 'backend-rwkv' feature in infotheory"
                ))
            }
        }
        "zpaq" => {
            let method = cfg["method"]
                .as_str()
                .or_else(|| cfg["zpaq_method"].as_str())
                .or_else(|| root["method"].as_str())
                .unwrap_or("2")
                .to_string();
            if let Err(err) = validate_zpaq_rate_method(&method) {
                return Err(anyhow::anyhow!(
                    "unsupported ZPAQ rate method '{method}': {err}"
                ));
            }
            Ok(RateBackend::Zpaq { method })
        }
        "match" => Ok(RateBackend::Match {
            hash_bits: cfg["hash_bits"].as_u64().unwrap_or(20) as usize,
            min_len: cfg["min_len"].as_u64().unwrap_or(4) as usize,
            max_len: cfg["max_len"].as_u64().unwrap_or(255) as usize,
            base_mix: cfg["base_mix"].as_f64().unwrap_or(0.02),
            confidence_scale: cfg["confidence_scale"].as_f64().unwrap_or(1.0),
        }),
        "sparse-match" => Ok(RateBackend::SparseMatch {
            hash_bits: cfg["hash_bits"].as_u64().unwrap_or(19) as usize,
            min_len: cfg["min_len"].as_u64().unwrap_or(3) as usize,
            max_len: cfg["max_len"].as_u64().unwrap_or(64) as usize,
            gap_min: cfg["gap_min"].as_u64().unwrap_or(1) as usize,
            gap_max: cfg["gap_max"].as_u64().unwrap_or(2) as usize,
            base_mix: cfg["base_mix"].as_f64().unwrap_or(0.05),
            confidence_scale: cfg["confidence_scale"].as_f64().unwrap_or(1.0),
        }),
        "ppmd" => Ok(RateBackend::Ppmd {
            order: cfg["order"].as_u64().unwrap_or(10) as usize,
            memory_mb: cfg["memory_mb"].as_u64().unwrap_or(64) as usize,
        }),
        "mixture" => {
            let spec = if let Some(spec_v) = cfg.get("spec").filter(|value| value.is_object()) {
                parse_mixture_spec_value(spec_v, base_dir, MAX_MIXTURE_NESTING)?
            } else if let Some(path) = cfg["mixture_spec"]
                .as_str()
                .or_else(|| cfg["spec_path"].as_str())
                .or_else(|| cfg["spec"].as_str())
                .or_else(|| root["mixture_spec"].as_str())
            {
                let full = base_dir.join(path);
                load_mixture_spec(full.to_str().unwrap_or(path))?
            } else {
                return Err(anyhow::anyhow!(
                    "mixture stats backend requires inline 'spec' or 'mixture_spec' path"
                ));
            };
            Ok(RateBackend::Mixture {
                spec: Arc::new(spec),
            })
        }
        "particle" => {
            let spec = if let Some(spec_v) = cfg.get("spec").filter(|value| value.is_object()) {
                parse_particle_spec_value(spec_v)?
            } else if let Some(path) = cfg["particle_spec"]
                .as_str()
                .or_else(|| cfg["spec_path"].as_str())
                .or_else(|| cfg["spec"].as_str())
                .or_else(|| root["particle_spec"].as_str())
            {
                let full = base_dir.join(path);
                load_particle_spec(full.to_str().unwrap_or(path))?
            } else {
                parse_particle_spec_value(cfg)?
            };
            spec.validate()
                .map_err(|e| anyhow::anyhow!("invalid particle spec: {e}"))?;
            Ok(RateBackend::Particle {
                spec: Arc::new(spec),
            })
        }
        "calibrated" => {
            let spec = if let Some(spec_v) = cfg.get("spec").filter(|value| value.is_object()) {
                parse_calibrated_spec_value(spec_v, base_dir, 4)?
            } else if let Some(path) = cfg["calibrated_spec"]
                .as_str()
                .or_else(|| cfg["spec_path"].as_str())
                .or_else(|| cfg["spec"].as_str())
                .or_else(|| root["calibrated_spec"].as_str())
            {
                let full = base_dir.join(path);
                load_calibrated_spec(full.to_str().unwrap_or(path))?
            } else {
                parse_calibrated_spec_value(cfg, base_dir, 4)?
            };
            Ok(RateBackend::Calibrated {
                spec: Arc::new(spec),
            })
        }
        other => Err(anyhow::anyhow!("unsupported vm stats backend '{other}'")),
    }
}

#[cfg(feature = "vm")]
fn default_vm_stats_backend(root: &serde_json::Value) -> anyhow::Result<RateBackend> {
    let algo = root["algorithm"].as_str().unwrap_or("ctw");
    let ct_depth = root["ct_depth"].as_u64().unwrap_or(20) as usize;
    match algo {
        "ctw" | "ac-ctw" | "ctw-context-tree" => Ok(RateBackend::Ctw { depth: ct_depth }),
        "fac-ctw" => Ok(RateBackend::FacCtw {
            base_depth: ct_depth,
            num_percept_bits: 8,
            encoding_bits: 8,
        }),
        "mamba" | "mamba1" => {
            #[cfg(feature = "backend-mamba")]
            {
                let path = root["mamba_model_path"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(mamba_model_path_from_env);
                let model = load_mamba_model_from_path(&path);
                Ok(RateBackend::Mamba { model })
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                Err(anyhow::anyhow!(
                    "mamba default stats backend requires 'backend-mamba' feature in infotheory"
                ))
            }
        }
        "rosa" | "rosaplus" => Ok(RateBackend::RosaPlus),
        "rwkv" | "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            {
                let path = root["rwkv_model_path"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(rwkv7_model_path_from_env);
                let model = load_rwkv7_model_from_path(&path);
                Ok(RateBackend::Rwkv7 { model })
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                Err(anyhow::anyhow!(
                    "rwkv7 default stats backend requires 'backend-rwkv' feature in infotheory"
                ))
            }
        }
        "zpaq" => Ok(RateBackend::Zpaq {
            method: {
                let method = root["method"].as_str().unwrap_or("2").to_string();
                if let Err(err) = validate_zpaq_rate_method(&method) {
                    return Err(anyhow::anyhow!(
                        "unsupported ZPAQ rate method '{method}': {err}"
                    ));
                }
                method
            },
        }),
        "mixture" | "mix" => {
            let spec_path = root["mixture_spec"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("mixture stats backend requires mixture_spec"))?;
            let spec = load_mixture_spec(spec_path)?;
            Ok(RateBackend::Mixture {
                spec: Arc::new(spec),
            })
        }
        _ => Ok(RateBackend::RosaPlus),
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_trace_config(v: &serde_json::Value) -> anyhow::Result<Option<NyxTraceConfig>> {
    if v.is_null() {
        return Ok(None);
    }
    let max_bytes = v["max_bytes"].as_u64().unwrap_or(1_000_000) as usize;
    let reset_on_episode = v["reset_on_episode"].as_bool().unwrap_or(false);
    let shared_region_name = v["shared_region_name"]
        .as_str()
        .or_else(|| v["shared_region"].as_str())
        .or_else(|| v["name"].as_str())
        .or_else(|| {
            if v["mode"].as_str() == Some("shared-memory") {
                Some("trace")
            } else {
                None
            }
        })
        .map(|s| s.to_string())
        .or(Some("trace".to_string()));

    Ok(Some(NyxTraceConfig {
        shared_region_name,
        max_bytes,
        reset_on_episode,
    }))
}

#[cfg(feature = "vm")]
fn parse_nyx_protocol_config(v: &serde_json::Value) -> NyxProtocolConfig {
    let mut cfg = NyxProtocolConfig::default();
    if let Some(s) = v["action_prefix"].as_str() {
        cfg.action_prefix = s.to_string();
    }
    if let Some(s) = v["action_suffix"].as_str() {
        cfg.action_suffix = s.to_string();
    }
    if let Some(s) = v["obs_prefix"].as_str() {
        cfg.obs_prefix = s.to_string();
    }
    if let Some(s) = v["rew_prefix"].as_str() {
        cfg.rew_prefix = s.to_string();
    }
    if let Some(s) = v["done_prefix"].as_str() {
        cfg.done_prefix = s.to_string();
    }
    if let Some(s) = v["data_prefix"].as_str() {
        cfg.data_prefix = s.to_string();
    }
    if let Some(s) = v["wire_encoding"].as_str() {
        if let Some(enc) = NyxPayloadEncoding::parse(s) {
            cfg.wire_encoding = enc;
        }
    }
    cfg
}

#[cfg(feature = "vm")]
fn parse_nyx_actions(v: &serde_json::Value) -> anyhow::Result<NyxActionSource> {
    let mode = v["mode"].as_str().unwrap_or("literal");
    match mode {
        "fuzz" => {
            let fuzz = if v["fuzz"].is_null() { v } else { &v["fuzz"] };
            let seed_encoding =
                NyxPayloadEncoding::parse(fuzz["seed_encoding"].as_str().unwrap_or("utf8"))
                    .unwrap_or(NyxPayloadEncoding::Utf8);
            let mut seeds = Vec::new();
            if let Some(arr) = fuzz["seed_paths"].as_array() {
                for item in arr {
                    if let Some(path) = item.as_str() {
                        let data = std::fs::read(path)?;
                        seeds.push(data);
                    }
                }
            }
            if let Some(arr) = fuzz["seed_inputs"].as_array() {
                for item in arr {
                    if let Some(text) = item.as_str() {
                        seeds.push(seed_encoding.decode(text)?);
                    }
                }
            }

            let mut mutators = Vec::new();
            if let Some(arr) = fuzz["mutators"].as_array() {
                for item in arr {
                    if let Some(name) = item.as_str() {
                        if let Some(m) = parse_nyx_fuzz_mutator(name) {
                            mutators.push(m);
                        }
                    }
                }
            }
            let min_len = fuzz["min_len"].as_u64().unwrap_or(1) as usize;
            let max_len = fuzz["max_len"].as_u64().unwrap_or(4096) as usize;
            let dict_encoding =
                NyxPayloadEncoding::parse(fuzz["dict_encoding"].as_str().unwrap_or("utf8"))
                    .unwrap_or(NyxPayloadEncoding::Utf8);
            let mut dictionary = Vec::new();
            if let Some(arr) = fuzz["dictionary"].as_array() {
                for item in arr {
                    if let Some(text) = item.as_str() {
                        dictionary.push(dict_encoding.decode(text)?);
                    }
                }
            }
            let rng_seed = fuzz["rng_seed"].as_u64().unwrap_or(0);
            Ok(NyxActionSource::Fuzz(NyxFuzzConfig {
                seeds,
                mutators,
                min_len,
                max_len,
                dictionary,
                rng_seed,
            }))
        }
        _ => {
            let mut actions = Vec::new();
            if let Some(arr) = v["actions"].as_array() {
                for item in arr {
                    if let Some(text) = item.as_str() {
                        let payload = NyxPayloadEncoding::Utf8.decode(text)?;
                        actions.push(NyxActionSpec {
                            name: None,
                            payload,
                        });
                        continue;
                    }
                    let payload = item["payload"].as_str().unwrap_or_default();
                    let encoding =
                        NyxPayloadEncoding::parse(item["encoding"].as_str().unwrap_or("utf8"))
                            .unwrap_or(NyxPayloadEncoding::Utf8);
                    let payload = encoding.decode(payload)?;
                    let name = item["name"].as_str().map(|s| s.to_string());
                    actions.push(NyxActionSpec { name, payload });
                }
            }
            Ok(NyxActionSource::Literal(actions))
        }
    }
}

#[cfg(feature = "vm")]
#[cfg(feature = "vm")]
fn parse_nyx_fuzz_mutator(name: &str) -> Option<NyxFuzzMutator> {
    match name {
        "flip_bit" | "flipbit" => Some(NyxFuzzMutator::FlipBit),
        "flip_byte" | "flipbyte" => Some(NyxFuzzMutator::FlipByte),
        "insert" | "insert_byte" => Some(NyxFuzzMutator::InsertByte),
        "delete" | "delete_byte" => Some(NyxFuzzMutator::DeleteByte),
        "splice" | "splice_seed" => Some(NyxFuzzMutator::SpliceSeed),
        "reset" | "reset_seed" => Some(NyxFuzzMutator::ResetSeed),
        "havoc" => Some(NyxFuzzMutator::Havoc),
        _ => None,
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_observation_policy(v: &serde_json::Value) -> NyxObservationPolicy {
    match v["mode"].as_str().unwrap_or("guest") {
        "raw" | "raw-bytes" | "bytes" | "stream" => NyxObservationPolicy::RawOutput,
        "hash" | "output-hash" => NyxObservationPolicy::OutputHash,
        "shared-memory" | "shared_mem" | "shared" => NyxObservationPolicy::SharedMemory,
        _ => NyxObservationPolicy::FromGuest,
    }
}

fn parse_observation_stream_len(v: &serde_json::Value) -> usize {
    v["observation_stream_len"].as_u64().unwrap_or(1) as usize
}

fn parse_observation_key_mode(v: &serde_json::Value) -> ObservationKeyMode {
    parse_observation_key_mode_str(v["observation_key_mode"].as_str().unwrap_or("full"))
}

fn parse_observation_key_mode_str(s: &str) -> ObservationKeyMode {
    match s {
        "full" | "full-stream" | "stream" => ObservationKeyMode::FullStream,
        "last" => ObservationKeyMode::Last,
        "hash" | "stream-hash" => ObservationKeyMode::StreamHash,
        _ => ObservationKeyMode::First,
    }
}

fn parse_observation_stream_len_for_env(v: &serde_json::Value, env_name: &str) -> usize {
    if env_name == "vm" || env_name == "nyx" || env_name == "nyx-vm" {
        if v["vm_observation"].is_null() {
            parse_observation_stream_len(v)
        } else {
            parse_observation_stream_len_for_vm(&v["vm_observation"])
        }
    } else {
        parse_observation_stream_len(v)
    }
}

fn parse_observation_key_mode_for_env(v: &serde_json::Value, env_name: &str) -> ObservationKeyMode {
    if env_name == "vm" || env_name == "nyx" || env_name == "nyx-vm" {
        if v["vm_observation"].is_null() {
            parse_observation_key_mode(v)
        } else {
            parse_observation_key_mode_for_vm(&v["vm_observation"])
        }
    } else {
        parse_observation_key_mode(v)
    }
}

fn parse_observation_key_mode_for_vm(v: &serde_json::Value) -> ObservationKeyMode {
    if v.is_null() {
        return ObservationKeyMode::FullStream;
    }
    parse_observation_key_mode_str(
        v["key_mode"]
            .as_str()
            .unwrap_or_else(|| v["observation_key_mode"].as_str().unwrap_or("full")),
    )
}

fn parse_observation_stream_len_for_vm(v: &serde_json::Value) -> usize {
    if v.is_null() {
        return 1;
    }
    v["stream_len"]
        .as_u64()
        .or_else(|| v["observation_stream_len"].as_u64())
        .unwrap_or(1) as usize
}

#[cfg(feature = "vm")]
fn parse_nyx_observation_stream_mode(v: &serde_json::Value) -> NyxObservationStreamMode {
    match v["stream_mode"].as_str().unwrap_or("pad-truncate") {
        "pad" => NyxObservationStreamMode::Pad,
        "truncate" => NyxObservationStreamMode::Truncate,
        _ => NyxObservationStreamMode::PadTruncate,
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_observation_pad_byte(v: &serde_json::Value) -> u8 {
    v["pad_byte"].as_u64().unwrap_or(0) as u8
}

fn extract_observation_stream_len_raw(v: &serde_json::Value) -> Option<usize> {
    v["observation_stream_len"].as_u64().map(|n| n as usize)
}

fn extract_vm_observation_stream_len_raw(v: &serde_json::Value) -> Option<usize> {
    if v.is_null() {
        return None;
    }
    v["stream_len"]
        .as_u64()
        .or_else(|| v["observation_stream_len"].as_u64())
        .map(|n| n as usize)
}

fn extract_observation_key_mode_raw(v: &serde_json::Value) -> Option<ObservationKeyMode> {
    v["observation_key_mode"]
        .as_str()
        .map(parse_observation_key_mode_str)
}

fn extract_vm_observation_key_mode_raw(v: &serde_json::Value) -> Option<ObservationKeyMode> {
    if v.is_null() {
        return None;
    }
    v["key_mode"]
        .as_str()
        .or_else(|| v["observation_key_mode"].as_str())
        .map(parse_observation_key_mode_str)
}

fn validate_observation_config(
    env_name: &str,
    v: &serde_json::Value,
    observation_stream_len: usize,
    observation_key_mode: ObservationKeyMode,
) -> anyhow::Result<()> {
    if observation_stream_len == 0 {
        return Err(anyhow::anyhow!("observation_stream_len must be > 0"));
    }
    if env_name == "vm" || env_name == "nyx" || env_name == "nyx-vm" {
        if let (Some(top_len), Some(vm_len)) = (
            extract_observation_stream_len_raw(v),
            extract_vm_observation_stream_len_raw(&v["vm_observation"]),
        ) && top_len != vm_len
        {
            return Err(anyhow::anyhow!(
                "observation_stream_len ({}) conflicts with vm_observation.stream_len ({})",
                top_len,
                vm_len
            ));
        }
        if let (Some(top_mode), Some(vm_mode)) = (
            extract_observation_key_mode_raw(v),
            extract_vm_observation_key_mode_raw(&v["vm_observation"]),
        ) && top_mode != vm_mode
        {
            return Err(anyhow::anyhow!(
                "observation_key_mode ({:?}) conflicts with vm_observation.key_mode ({:?})",
                top_mode,
                vm_mode
            ));
        }
    }
    if observation_stream_len > 1 && matches!(observation_key_mode, ObservationKeyMode::First) {
        eprintln!(
            "Warning: observation_key_mode=first collapses multi-symbol observation streams; prefer \"full\" for paper-accurate expectimax."
        );
    }
    if observation_stream_len > 1 && !matches!(observation_key_mode, ObservationKeyMode::FullStream)
    {
        eprintln!(
            "Warning: observation_key_mode {:?} reduces multi-symbol observation streams and deviates from paper-accurate expectimax.",
            observation_key_mode
        );
    }
    Ok(())
}

/// Validates that the actual observation stream length matches the configured value.
///
/// This is a hard error to prevent FAC-CTW bit cycling desynchronization.
fn validate_obs_stream_len(expected: usize, actual: usize) -> anyhow::Result<()> {
    if actual != expected {
        return Err(anyhow::anyhow!(
            "Observation stream length mismatch: config expects {} symbols, but environment returned {}. \
            This causes FAC-CTW bit cycling desynchronization. \
            Fix your `observation_stream_len` config or environment implementation.",
            expected,
            actual
        ));
    }
    Ok(())
}

#[cfg(feature = "vm")]
fn parse_nyx_reward_policy(v: &serde_json::Value) -> anyhow::Result<NyxRewardPolicy> {
    match v["mode"].as_str().unwrap_or("guest") {
        "pattern" => {
            let pattern = v["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("vm_reward.pattern is required"))?
                .to_string();
            let base_reward = v["base_reward"].as_i64().unwrap_or(0);
            let bonus_reward = v["bonus_reward"].as_i64().unwrap_or(10);
            Ok(NyxRewardPolicy::Pattern {
                pattern,
                base_reward,
                bonus_reward,
            })
        }
        _ => Ok(NyxRewardPolicy::FromGuest),
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_reward_shaping(v: &serde_json::Value) -> anyhow::Result<Option<NyxRewardShaping>> {
    if v.is_null() {
        return Ok(None);
    }
    match v["mode"].as_str().unwrap_or("none") {
        "entropy-reduction" | "entropy_reduction" => {
            let baseline_path = v["baseline_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("vm_reward_shaping.baseline_path is required"))?;
            let baseline_bytes = std::fs::read(baseline_path)?;
            let max_order = v["max_order"].as_i64().unwrap_or(8);
            let scale = v["scale"].as_f64().unwrap_or(10.0);
            let crash_bonus = v["crash_bonus"].as_i64();
            let timeout_bonus = v["timeout_bonus"].as_i64();
            Ok(Some(NyxRewardShaping::EntropyReduction {
                baseline_bytes,
                max_order,
                scale,
                crash_bonus,
                timeout_bonus,
            }))
        }
        "trace-entropy" | "trace_entropy" => {
            let max_order = v["max_order"].as_i64().unwrap_or(8);
            let scale = v["scale"].as_f64().unwrap_or(1.0);
            let normalize = v["normalize"].as_bool().unwrap_or(false);
            Ok(Some(NyxRewardShaping::TraceEntropy {
                max_order,
                scale,
                normalize,
            }))
        }
        "none" | "off" => Ok(None),
        _ => Ok(None),
    }
}

#[cfg(feature = "vm")]
fn parse_nyx_filter(
    v: &serde_json::Value,
    step_cost: i64,
) -> anyhow::Result<Option<NyxActionFilter>> {
    if v.is_null() {
        return Ok(None);
    }
    let novelty_prior = if let Some(path) = v["novelty_prior_path"].as_str() {
        Some(std::fs::read(path)?)
    } else {
        None
    };
    let reject_reward = v["reject_reward"].as_i64().or_else(|| Some(-step_cost));
    Ok(Some(NyxActionFilter {
        min_entropy: v["min_entropy"].as_f64(),
        max_entropy: v["max_entropy"].as_f64(),
        min_intrinsic_dependence: v["min_intrinsic_dependence"].as_f64(),
        min_novelty: v["min_novelty"].as_f64(),
        novelty_prior,
        max_order: v["max_order"].as_i64().unwrap_or(8),
        reject_reward,
    }))
}

struct BuiltCtx {
    ctx: InfotheoryCtx,
    expert_spec_max_order: Option<i64>,
}

fn build_ctx(
    rate_backend: &str,
    compression_backend: &str,
    method: Option<&str>,
    expert_spec_path: Option<&str>,
) -> BuiltCtx {
    let (rate_backend, expert_spec_max_order) = if let Some(path) = expert_spec_path {
        let spec = load_expert_spec(path).unwrap_or_else(|e| {
            eprintln!("Error: failed to load expert spec '{path}': {e}");
            std::process::exit(1);
        });
        (spec.backend, Some(spec.max_order))
    } else {
        (
            match rate_backend {
                "mamba" => {
                    #[cfg(feature = "backend-mamba")]
                    {
                        if let Some(m) = method {
                            RateBackend::MambaMethod {
                                method: m.to_string(),
                            }
                        } else {
                            let p = mamba_model_path_from_env();
                            let model = load_mamba_model_from_path(&p);
                            RateBackend::Mamba { model }
                        }
                    }
                    #[cfg(not(feature = "backend-mamba"))]
                    {
                        eprintln!(
                            "Error: rate backend 'mamba' requires infotheory built with feature 'backend-mamba'"
                        );
                        std::process::exit(1);
                    }
                }
                "rwkv7" => {
                    #[cfg(feature = "backend-rwkv")]
                    {
                        if let Some(m) = method {
                            RateBackend::Rwkv7Method {
                                method: m.to_string(),
                            }
                        } else {
                            let p = rwkv7_model_path_from_env();
                            let model = load_rwkv7_model_from_path(&p);
                            RateBackend::Rwkv7 { model }
                        }
                    }
                    #[cfg(not(feature = "backend-rwkv"))]
                    {
                        eprintln!(
                            "Error: rate backend 'rwkv7' requires infotheory built with feature 'backend-rwkv'"
                        );
                        std::process::exit(1);
                    }
                }
                "match" => RateBackend::Match {
                    hash_bits: 20,
                    min_len: 4,
                    max_len: 255,
                    base_mix: 0.02,
                    confidence_scale: 1.0,
                },
                "sparse-match" => RateBackend::SparseMatch {
                    hash_bits: 19,
                    min_len: 3,
                    max_len: 64,
                    gap_min: 1,
                    gap_max: 2,
                    base_mix: 0.05,
                    confidence_scale: 1.0,
                },
                "ppmd" => RateBackend::Ppmd {
                    order: method.and_then(|m| m.parse::<usize>().ok()).unwrap_or(10),
                    memory_mb: 64,
                },
                "ctw" => {
                    let depth = if let Some(m) = method {
                        m.parse::<usize>().unwrap_or(20)
                    } else {
                        20
                    };
                    RateBackend::Ctw { depth }
                }
                "fac-ctw" => {
                    let depth = if let Some(m) = method {
                        m.parse::<usize>().unwrap_or(20)
                    } else {
                        20
                    };
                    RateBackend::FacCtw {
                        base_depth: depth,
                        num_percept_bits: 8, // Default for byte-oriented CLI
                        encoding_bits: 8,    // Default for byte-oriented CLI
                    }
                }
                "zpaq" => {
                    let m = method.unwrap_or("2").to_string();
                    if let Err(err) = validate_zpaq_rate_method(&m) {
                        eprintln!("Error: unsupported ZPAQ rate method '{m}': {err}");
                        std::process::exit(1);
                    }
                    RateBackend::Zpaq { method: m }
                }
                "mixture" => {
                    let path = method.unwrap_or_else(|| {
                        eprintln!("Error: --rate-backend mixture requires --method <spec.json>");
                        std::process::exit(1);
                    });
                    let spec = load_mixture_spec(path).unwrap_or_else(|e| {
                        eprintln!("Error: failed to load mixture spec '{path}': {e}");
                        std::process::exit(1);
                    });
                    RateBackend::Mixture {
                        spec: Arc::new(spec),
                    }
                }
                "particle" => {
                    let path = method.unwrap_or_else(|| {
                        eprintln!("Error: --rate-backend particle requires --method <spec.json>");
                        std::process::exit(1);
                    });
                    let spec = load_particle_spec(path).unwrap_or_else(|e| {
                        eprintln!("Error: failed to load particle spec '{path}': {e}");
                        std::process::exit(1);
                    });
                    RateBackend::Particle {
                        spec: Arc::new(spec),
                    }
                }
                "calibrated" => {
                    let path = method.unwrap_or_else(|| {
                        eprintln!("Error: --rate-backend calibrated requires --method <spec.json>");
                        std::process::exit(1);
                    });
                    let spec = load_calibrated_spec(path).unwrap_or_else(|e| {
                        eprintln!("Error: failed to load calibrated spec '{path}': {e}");
                        std::process::exit(1);
                    });
                    RateBackend::Calibrated {
                        spec: Arc::new(spec),
                    }
                }
                _ => RateBackend::RosaPlus,
            },
            None,
        )
    };

    let compression_backend = match compression_backend {
        "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            {
                match method {
                    // Legacy: allow passing only coder (ac/rans), model path from env.
                    Some(m) if infotheory::backends::parse_rwkv7_coder(m).is_some() => {
                        let model_path = rwkv7_model_path_from_env();
                        let model = load_rwkv7_model_from_path(&model_path);
                        let coder = infotheory::backends::parse_rwkv7_coder(m)
                            .unwrap_or(rwkvzip::CoderType::AC);
                        CompressionBackend::Rwkv7 { model, coder }
                    }
                    // Method-based RWKV config: file:/... or cfg:...
                    Some(m) => match rwkvzip::parse_method_spec(m) {
                        Ok(rwkvzip::MethodSpec::File { path, policy: None }) => {
                            let model = load_rwkv7_model_from_path(path.to_string_lossy().as_ref());
                            CompressionBackend::Rwkv7 {
                                model,
                                coder: rwkvzip::CoderType::AC,
                            }
                        }
                        Ok(rwkvzip::MethodSpec::File {
                            policy: Some(_), ..
                        })
                        | Ok(rwkvzip::MethodSpec::Online { .. }) => CompressionBackend::Rate {
                            rate_backend: RateBackend::Rwkv7Method {
                                method: m.to_string(),
                            },
                            coder: rwkvzip::CoderType::AC,
                            framing: infotheory::compression::FramingMode::Raw,
                        },
                        Err(err) => {
                            eprintln!(
                                "Error: invalid RWKV method for --compression-backend rwkv7: {err}"
                            );
                            std::process::exit(1);
                        }
                    },
                    // No method: model path from env.
                    None => {
                        let model_path = rwkv7_model_path_from_env();
                        let model = load_rwkv7_model_from_path(&model_path);
                        CompressionBackend::Rwkv7 {
                            model,
                            coder: rwkvzip::CoderType::AC,
                        }
                    }
                }
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                eprintln!(
                    "Error: compression backend 'rwkv7' requires infotheory built with feature 'backend-rwkv'"
                );
                std::process::exit(1);
            }
        }
        "rate-ac" => CompressionBackend::Rate {
            rate_backend: rate_backend.clone(),
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        },
        "rate-rans" => CompressionBackend::Rate {
            rate_backend: rate_backend.clone(),
            coder: infotheory::coders::CoderType::RANS,
            framing: infotheory::compression::FramingMode::Raw,
        },
        _ => {
            let m = method.unwrap_or("5").to_string();
            CompressionBackend::Zpaq { method: m }
        }
    };

    BuiltCtx {
        ctx: InfotheoryCtx::new(rate_backend, compression_backend),
        expert_spec_max_order,
    }
}

fn read_file(path: &str) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            std::process::exit(1);
        }
    }
}

fn read_stdin_all_for_generate() -> Vec<u8> {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        eprintln!("Error: 'generate' requires <input_file> or piped stdin");
        std::process::exit(1);
    }
    let mut data = Vec::new();
    if let Err(e) = stdin.lock().read_to_end(&mut data) {
        eprintln!("Error reading stdin: {e}");
        std::process::exit(1);
    }
    data
}

fn file_roundtrip_backend(backend: &CompressionBackend) -> CompressionBackend {
    match backend {
        CompressionBackend::Rate {
            rate_backend,
            coder,
            ..
        } => CompressionBackend::Rate {
            rate_backend: rate_backend.clone(),
            coder: *coder,
            framing: infotheory::compression::FramingMode::Framed,
        },
        _ => backend.clone(),
    }
}

fn maybe_export_online_model(
    export_path: Option<&str>,
    _ctx: &InfotheoryCtx,
    _parts: &[&[u8]],
) -> anyhow::Result<()> {
    let Some(_path) = export_path else {
        return Ok(());
    };

    #[cfg(feature = "backend-rwkv")]
    {
        let rwkv_method = match &_ctx.rate_backend {
            RateBackend::Rwkv7Method { method } => Some(method.as_str()),
            _ => match &_ctx.compression_backend {
                CompressionBackend::Rate {
                    rate_backend: RateBackend::Rwkv7Method { method },
                    ..
                } => Some(method.as_str()),
                _ => None,
            },
        };
        if let Some(method) = rwkv_method {
            let mut compressor = rwkvzip::Compressor::new_from_method(method)?;
            let _ = compressor.compress_size_chain(_parts, infotheory::coders::CoderType::AC)?;
            compressor.export_online(_path)?;
            return Ok(());
        }
        // Pre-loaded model Arc variants do not carry a method string and cannot
        // replay training for export.  Warn the user explicitly.
        let has_rwkv_model = matches!(&_ctx.rate_backend, RateBackend::Rwkv7 { .. })
            || matches!(
                &_ctx.compression_backend,
                CompressionBackend::Rate {
                    rate_backend: RateBackend::Rwkv7 { .. },
                    ..
                } | CompressionBackend::Rwkv7 { .. }
            );
        if has_rwkv_model {
            eprintln!(
                "Warning: --model-export is not supported for pre-loaded RWKV7 model backends. \
                 Use --method with a cfg:/file: spec to enable online model export."
            );
            return Ok(());
        }
    }

    #[cfg(feature = "backend-mamba")]
    {
        let mamba_method = match &_ctx.rate_backend {
            RateBackend::MambaMethod { method } => Some(method.as_str()),
            _ => match &_ctx.compression_backend {
                CompressionBackend::Rate {
                    rate_backend: RateBackend::MambaMethod { method },
                    ..
                } => Some(method.as_str()),
                _ => None,
            },
        };
        if let Some(method) = mamba_method {
            let mut compressor = mambazip::Compressor::new_from_method(method)?;
            let _ = compressor.compress_size_chain(_parts, infotheory::coders::CoderType::AC)?;
            compressor.export_online(_path)?;
            return Ok(());
        }
        let has_mamba_model = matches!(&_ctx.rate_backend, RateBackend::Mamba { .. })
            || matches!(
                &_ctx.compression_backend,
                CompressionBackend::Rate {
                    rate_backend: RateBackend::Mamba { .. },
                    ..
                }
            );
        if has_mamba_model {
            eprintln!(
                "Warning: --model-export is not supported for pre-loaded Mamba model backends. \
                 Use --method with a cfg: spec to enable online model export."
            );
            return Ok(());
        }
    }

    eprintln!(
        "Warning: --model-export was requested but the current backend does not support \
         online model export. Only RWKV7 and Mamba method-based backends support export."
    );
    Ok(())
}

// ============================================================
// Batch JSON Mode - For programmatic use from Python
// ============================================================

/// ROSA-based symmetric codelength distance (NCD-like but faster)
/// d_ROSA(x,y) = 0.5 * (H_y(x)/H_x(x) + H_x(y)/H_y(y)) - 1
/// Clamped to [0, 1]
fn rosa_distance(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }

    // Self-entropy rates (biased/plugin estimator for consistency)
    let h_x_x = biased_entropy_rate_bytes(x, max_order);
    let h_y_y = biased_entropy_rate_bytes(y, max_order);

    // Cross-entropy rates
    let h_y_x = cross_entropy_rate_bytes(x, y, max_order); // score x under model trained on y
    let h_x_y = cross_entropy_rate_bytes(y, x, max_order); // score y under model trained on x

    // Avoid division by zero
    if h_x_x < 1e-9 || h_y_y < 1e-9 {
        return 1.0;
    }

    let d = 0.5 * (h_y_x / h_x_x + h_x_y / h_y_y) - 1.0;
    d.clamp(0.0, 1.0)
}

/// Process a single JSON line and return result
fn process_json_line(line: &str) -> String {
    let line = line.trim();
    if line.is_empty() {
        return r#"{"error":"empty input"}"#.to_string();
    }

    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "error": format!("invalid json: {e}")
            })
            .to_string();
        }
    };
    let op = v.get("op").and_then(|x| x.as_str()).unwrap_or("");

    match op {
        "metrics" => {
            // Single text metrics: H0, H_rate, ID
            let text = v
                .get("text")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);
            let data = text.as_bytes();

            if data.is_empty() {
                return r#"{"error":"empty text"}"#.to_string();
            }

            let h0 = marginal_entropy_bytes(data);
            let h_rate = entropy_rate_bytes(data, max_order);
            let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };

            format!(
                r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                h0, h_rate, id, data.len()
            )
        }

        "metrics_file" => {
            // File-based metrics
            let path = v
                .get("path")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);

            match std::fs::read(&path) {
                Ok(data) => {
                    let h0 = marginal_entropy_bytes(&data);
                    let h_rate = entropy_rate_bytes(&data, max_order);
                    let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };

                    format!(
                        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                        h0, h_rate, id, data.len()
                    )
                }
                Err(e) => format!(r#"{{"error":"failed to read file: {}"}}"#, e),
            }
        }

        "ncd" => {
            // NCD between two texts
            let text1 = v
                .get("text1")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let text2 = v
                .get("text2")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let method = v
                .get("method")
                .and_then(|x| x.as_str())
                .unwrap_or("5")
                .to_string();
            let variant = v
                .get("variant")
                .and_then(|x| x.as_str())
                .unwrap_or("vitanyi")
                .to_string();

            let x = text1.as_bytes();
            let y = text2.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let ncd = ncd_bytes(x, y, &method, ncd_variant);
            format!(r#"{{"ncd":{:.6}}}"#, ncd)
        }
        "ncd_files" => {
            // NCD between two files
            let path1 = v
                .get("path1")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let path2 = v
                .get("path2")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let method = v
                .get("method")
                .and_then(|x| x.as_str())
                .unwrap_or("5")
                .to_string();
            let variant = v
                .get("variant")
                .and_then(|x| x.as_str())
                .unwrap_or("vitanyi")
                .to_string();

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let ncd = ncd_paths(&path1, &path2, &method, ncd_variant);
            format!(r#"{{"ncd":{:.6}}}"#, ncd)
        }

        "rosa_dist" => {
            // ROSA-based distance (faster than NCD)
            let text1 = v
                .get("text1")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let text2 = v
                .get("text2")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);

            let x = text1.as_bytes();
            let y = text2.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let dist = rosa_distance(x, y, max_order);
            format!(r#"{{"rosa_dist":{:.6}}}"#, dist)
        }

        "cross_entropy" => {
            // Cross-entropy H_y(x) - score x under model trained on y
            let text_x = v
                .get("text_x")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let text_y = v
                .get("text_y")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);

            let x = text_x.as_bytes();
            let y = text_y.as_bytes();

            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            let xe = cross_entropy_rate_bytes(x, y, max_order);
            format!(r#"{{"cross_entropy":{:.6}}}"#, xe)
        }
        "batch_metrics" => {
            // Batch metrics for multiple texts
            let texts: Vec<String> = v
                .get("texts")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);

            let results: Vec<String> = texts.iter().map(|text| {
                let data = text.as_bytes();
                if data.is_empty() {
                    r#"{"h0":0,"h_rate":0,"id":0,"len":0}"#.to_string()
                } else {
                    let h0 = marginal_entropy_bytes(data);
                    let h_rate = entropy_rate_bytes(data, max_order);
                    let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };
                    format!(
                        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                        h0, h_rate, id, data.len()
                    )
                }
            }).collect();

            format!(r#"{{"results":[{}]}}"#, results.join(","))
        }

        "ncd_matrix" => {
            // NCD matrix for multiple texts (for diversity/clustering)
            let texts: Vec<String> = v
                .get("texts")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let method = v
                .get("method")
                .and_then(|x| x.as_str())
                .unwrap_or("5")
                .to_string();
            let variant = v
                .get("variant")
                .and_then(|x| x.as_str())
                .unwrap_or("vitanyi")
                .to_string();

            let ncd_variant = match variant.as_str() {
                "sym" | "sym_vitanyi" => NcdVariant::SymVitanyi,
                "cons" => NcdVariant::Cons,
                "sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };

            let datas: Vec<Vec<u8>> = texts.iter().map(|t| t.as_bytes().to_vec()).collect();
            let matrix = ncd_matrix_bytes(&datas, &method, ncd_variant);
            let n = datas.len();

            // Format as row-major array of arrays
            let rows: Vec<String> = (0..n).map(|i| {
                let row: Vec<String> = (0..n).map(|j| format!("{:.6}", matrix[i * n + j])).collect();
                format!("[{}]", row.join(","))
            }).collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "rosa_matrix" => {
            // ROSA distance matrix (faster than NCD matrix)
            let texts: Vec<String> = v
                .get("texts")
                .and_then(|x| x.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let max_order = v.get("max_order").and_then(|x| x.as_i64()).unwrap_or(-1);

            let n = texts.len();
            let datas: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();

            // Compute matrix (symmetric)
            let mut matrix = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let d = if i == j {
                        0.0
                    } else {
                        rosa_distance(datas[i], datas[j], max_order)
                    };
                    matrix[i * n + j] = d;
                    matrix[j * n + i] = d;
                }
            }

            // Format as row-major array of arrays
            let rows: Vec<String> = (0..n).map(|i| {
                let row: Vec<String> = (0..n).map(|j| format!("{:.6}", matrix[i * n + j])).collect();
                format!("[{}]", row.join(","))
            }).collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "spam_check" => {
            // Quick spam/quality check for a single text
            let text = v
                .get("text")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let h0_threshold = v.get("h0_min").and_then(|x| x.as_f64()).unwrap_or(1.0);
            let h_rate_threshold = v.get("h_rate_min").and_then(|x| x.as_f64()).unwrap_or(0.5);
            let id_threshold = v.get("id_max").and_then(|x| x.as_f64()).unwrap_or(0.95);
            let min_len = v.get("min_len").and_then(|x| x.as_i64()).unwrap_or(10) as usize;

            let data = text.as_bytes();
            let len = data.len();

            if len < min_len {
                return format!(r#"{{"pass":false,"reason":"too_short","len":{}}}"#, len);
            }

            let h0 = marginal_entropy_bytes(data);
            if h0 < h0_threshold {
                return format!(r#"{{"pass":false,"reason":"low_entropy","h0":{:.4}}}"#, h0);
            }

            let h_rate = entropy_rate_bytes(data, -1);
            if h_rate < h_rate_threshold {
                return format!(r#"{{"pass":false,"reason":"low_entropy_rate","h_rate":{:.4}}}"#, h_rate);
            }

            let id = if h0 < 1e-9 { 0.0 } else { ((h0 - h_rate) / h0).clamp(0.0, 1.0) };
            if id > id_threshold {
                return format!(r#"{{"pass":false,"reason":"high_redundancy","id":{:.4}}}"#, id);
            }

            format!(r#"{{"pass":true,"h0":{:.4},"h_rate":{:.4},"id":{:.4},"len":{}}}"#, h0, h_rate, id, len)
        }
        "help" => {
            r#"{"ops":["metrics","metrics_file","ncd","ncd_files","rosa_dist","cross_entropy","batch_metrics","ncd_matrix","rosa_matrix","spam_check"]}"#.to_string()
        }
        _ => {
            format!(r#"{{"error":"unknown op: {}"}}"#, op)
        }
    }
}

fn run_batch_mode() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        match line {
            Ok(l) => println!("{}", process_json_line(&l)),
            Err(_) => continue,
        }
    }
}

fn run_aixi_mode(config_path: &str) -> anyhow::Result<()> {
    let mut file = File::open(config_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let v: serde_json::Value = serde_json::from_str(&content)?;

    let env_name = v["environment"].as_str().unwrap_or("coin-flip");
    let mut env: Box<dyn Environment> = match env_name {
        "coin-flip" => Box::new(CoinFlip::new(0.9)),
        "ctw-test" | "ctwtest" => Box::new(CtwTest::new()),
        "extended-tiger" => Box::new(ExtendedTiger::new()),
        "tictactoe" => Box::new(TicTacToe::new()),
        "biased-rock-paper-scissor" => Box::new(BiasedRockPaperScissor::new()),
        "kuhn-poker" => Box::new(KuhnPoker::new()),
        "external" => {
            return Err(anyhow::anyhow!(
                "environment=external (ProcessEnvironment) has been removed for security reasons. \
                Use NyxVmEnvironment with --features vm for secure sandboxed environments."
            ));
        }
        "vm" | "nyx" | "nyx-vm" => {
            #[cfg(not(feature = "vm"))]
            {
                return Err(anyhow::anyhow!(
                    "VM environments require the `vm` feature (enable with --features vm)"
                ));
            }
            #[cfg(feature = "vm")]
            {
                let observation_bits = v["observation_bits"].as_u64().unwrap_or(16) as usize;
                let reward_bits = v["reward_bits"].as_u64().unwrap_or(8) as usize;
                let agent_horizon = v["agent_horizon"].as_u64().unwrap_or(3) as usize;
                let config_dir = Path::new(config_path).parent().unwrap_or(Path::new("."));
                let vm_cfg = parse_nyx_environment_config(
                    &v,
                    observation_bits,
                    reward_bits,
                    agent_horizon,
                    config_dir,
                )?;
                Box::new(NyxVmEnvironment::new(vm_cfg)?)
            }
        }
        _ => return Err(anyhow::anyhow!("Unknown environment: {}", env_name)),
    };

    let log_every = v["log_every"].as_u64().unwrap_or(1) as usize;
    let perf = v["perf"].as_bool().unwrap_or(false);
    let vm_perf_only = v["vm_perf_only"].as_bool().unwrap_or(false);

    if vm_perf_only {
        let cycles = v["perf_cycles"]
            .as_u64()
            .or_else(|| v["terminate-lifetime"].as_u64())
            .unwrap_or(1000) as usize;
        let observation_stream_len = parse_observation_stream_len_for_env(&v, env_name);
        let mut obs_stream = env.drain_observations();
        validate_obs_stream_len(observation_stream_len, obs_stream.len())?;
        let mut obs = obs_stream.first().copied().unwrap_or(0);
        let mut rew = env.get_reward();
        let start = Instant::now();
        for t in 0..cycles {
            if log_every > 0 && t % log_every == 0 {
                println!("Cycle {}: Obs={}, Rew={}", t, obs, rew);
            }
            env.perform_action(0);
            obs_stream = env.drain_observations();
            validate_obs_stream_len(observation_stream_len, obs_stream.len())?;
            obs = obs_stream.first().copied().unwrap_or(0);
            rew = env.get_reward();
        }
        if perf {
            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            let cps = cycles as f64 / elapsed;
            println!("Perf cycles/s: {:.2}", cps);
        }
        return Ok(());
    }

    let observation_bits = v["observation_bits"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or_else(|| env.get_observation_bits());
    let observation_stream_len = parse_observation_stream_len_for_env(&v, env_name);
    let observation_key_mode = parse_observation_key_mode_for_env(&v, env_name);
    validate_observation_config(env_name, &v, observation_stream_len, observation_key_mode)?;
    let reward_bits = v["reward_bits"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or_else(|| env.get_reward_bits());
    let agent_actions = v["agent_actions"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or_else(|| env.get_num_actions());
    let min_reward = env.min_reward();
    let max_reward = env.max_reward();
    let reward_offset = v["reward_offset"]
        .as_i64()
        .unwrap_or_else(|| (-min_reward).max(0));
    let discount_gamma = v["discount_gamma"].as_f64().unwrap_or(1.0);
    if !(0.0..=1.0).contains(&discount_gamma) {
        return Err(anyhow::anyhow!(
            "discount_gamma must be in [0, 1] (got {})",
            discount_gamma
        ));
    }

    let config = AgentConfig {
        algorithm: v["algorithm"].as_str().unwrap_or("ctw").to_string(),
        ct_depth: v["ct_depth"].as_u64().unwrap_or(20) as usize,
        agent_horizon: v["agent_horizon"].as_u64().unwrap_or(3) as usize,
        observation_bits,
        observation_stream_len,
        observation_key_mode,
        reward_bits,
        agent_actions,
        num_simulations: v["num_simulations"].as_u64().unwrap_or(50) as usize,
        exploration_exploitation_ratio: v["exploration_exploitation_ratio"].as_f64().unwrap_or(1.4),
        discount_gamma,
        min_reward,
        max_reward,
        reward_offset,
        rwkv_model_path: v["rwkv_model_path"].as_str().map(|s| s.to_string()),
        rosa_max_order: v["rosa_max_order"].as_u64().map(|n| n as i64),
        zpaq_method: v["zpaq_method"].as_str().map(|s| s.to_string()),
    };

    let mut agent = Agent::new(config);
    println!(
        "Agent initialized with {} algorithm for {} environment.",
        v["algorithm"].as_str().unwrap_or("ctw"),
        env_name
    );

    let learn_cycles = v["learn_cycles"].as_u64().map(|n| n as usize);
    let eval_cycles = v["eval_cycles"].as_u64().map(|n| n as usize);
    let cycles = v["terminate-lifetime"].as_u64().unwrap_or(20) as usize;

    let (learn_cycles, eval_cycles) = match (learn_cycles, eval_cycles) {
        (Some(l), Some(e)) => (l, e),
        (Some(l), None) => (l, 0usize),
        (None, Some(e)) => (cycles, e),
        (None, None) => (cycles, 0usize),
    };

    let mut total_reward = 0;
    let mut prev_action = 0;
    let mut obs_stream = env.drain_observations();
    validate_obs_stream_len(observation_stream_len, obs_stream.len())?;
    let mut obs_repr = agent.observation_repr_from_stream(&obs_stream);
    let mut rew = env.get_reward();

    let explore_epsilon = v["explore_epsilon"].as_f64().unwrap_or(0.0);
    let explore_gamma = v["explore_gamma"].as_f64().unwrap_or(1.0);
    let mut explore_rng = RandomGenerator::new();

    // Optional trace logger: can emit a RWKV-friendly stream (0/1 bytes) and/or JSONL metadata.
    // NOTE: The bits are emitted in the same order the agent consumes them:
    // percept bits (obs stream + reward) first, then action bits.
    let mut trace_logger = AixiRunLogger::new(&v)?;

    let learn_start = Instant::now();
    for t in 0..learn_cycles {
        if log_every > 0 && t % log_every == 0 {
            println!("Cycle {}: Obs={:?}, Rew={}", t, obs_repr, rew);
        }
        if let Some(l) = trace_logger.as_mut() {
            l.log_percept(
                &obs_stream,
                rew,
                observation_bits,
                reward_bits,
                reward_offset,
            )?;
        }
        agent.model_update_percept_stream(&obs_stream, rew);
        total_reward += rew;

        let explore_p = if explore_epsilon > 0.0 {
            explore_epsilon * explore_gamma.powi(t as i32)
        } else {
            0.0
        };
        let action = if explore_p > 0.0 && explore_rng.gen_bool(explore_p.min(1.0)) {
            explore_rng.gen_range(agent_actions) as u64
        } else {
            agent.get_planned_action(&obs_stream, rew, prev_action)
        };
        if log_every > 0 && t % log_every == 0 {
            println!("Cycle {}: Planned Action={}", t, action);
        }

        if let Some(l) = trace_logger.as_mut() {
            // Mirror the agent's action encoding (same number of bits).
            let action_bits = env.get_action_bits();
            l.log_action(action, action_bits)?;
        }
        agent.model_update_action_external(action);
        env.perform_action(action);
        obs_stream = env.drain_observations();
        validate_obs_stream_len(observation_stream_len, obs_stream.len())?;
        obs_repr = agent.observation_repr_from_stream(&obs_stream);
        rew = env.get_reward();
        prev_action = action;

        if let Some(l) = trace_logger.as_mut() {
            l.next_step()?;
        }
    }

    if perf && learn_cycles > 0 {
        let elapsed = learn_start.elapsed().as_secs_f64().max(1e-9);
        let cps = learn_cycles as f64 / elapsed;
        println!("Learn cycles/s: {:.2}", cps);
    }

    if eval_cycles > 0 {
        let mut eval_total_reward: i64 = 0;
        let eval_start = Instant::now();
        for t in 0..eval_cycles {
            let step = learn_cycles + t;
            if log_every > 0 && step % log_every == 0 {
                println!("Cycle {}: Obs={:?}, Rew={}", step, obs_repr, rew);
            }
            if let Some(l) = trace_logger.as_mut() {
                l.log_percept(
                    &obs_stream,
                    rew,
                    observation_bits,
                    reward_bits,
                    reward_offset,
                )?;
            }
            agent.model_update_percept_stream(&obs_stream, rew);
            eval_total_reward += rew;

            let action = agent.get_planned_action(&obs_stream, rew, prev_action);
            if log_every > 0 && step % log_every == 0 {
                println!("Cycle {}: Planned Action={}", step, action);
            }

            if let Some(l) = trace_logger.as_mut() {
                let action_bits = env.get_action_bits();
                l.log_action(action, action_bits)?;
            }
            agent.model_update_action_external(action);
            env.perform_action(action);
            obs_stream = env.drain_observations();
            validate_obs_stream_len(observation_stream_len, obs_stream.len())?;
            obs_repr = agent.observation_repr_from_stream(&obs_stream);
            rew = env.get_reward();
            prev_action = action;

            if let Some(l) = trace_logger.as_mut() {
                l.next_step()?;
            }
        }

        if perf && eval_cycles > 0 {
            let elapsed = eval_start.elapsed().as_secs_f64().max(1e-9);
            let cps = eval_cycles as f64 / elapsed;
            println!("Eval cycles/s: {:.2}", cps);
        }

        let avg = (eval_total_reward as f64) / (eval_cycles as f64);
        println!("Eval Total Reward: {}", eval_total_reward);
        println!("Eval Average Reward per Cycle: {:.6}", avg);
    }

    println!("Total Reward: {}", total_reward);
    Ok(())
}

fn search_command(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Error: 'search' requires query and target path.");
        std::process::exit(1);
    }
    let query = &args[2];
    let target = &args[3];

    // Preserve the legacy behavior (and avoid extra parsing work) when no flags are given.
    if args.len() == 4 {
        search::run_search(query, target);
        return;
    }

    let mut opts = search::SearchOptions::default();
    let mut rate_backend = "rosaplus".to_string();
    let compression_backend = "zpaq".to_string();
    let mut method: Option<String> = None;
    let mut expert_spec_path: Option<String> = None;
    let mut stage2_prior_mode: Option<search::Stage2PriorMode> = None;

    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--level" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --level requires snippet|file");
                opts.granularity = if v == "snippet" {
                    search::SearchGranularity::Snippet
                } else {
                    search::SearchGranularity::File
                };
            }
            "--prior" => {
                i += 1;
                opts.universal_prior = args.get(i).cloned();
            }
            "--max-order" => {
                i += 1;
                opts.max_order = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--top-k" => {
                i += 1;
                opts.top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend = parse_rate_backend(v).unwrap_or("rosaplus").to_string();
            }
            "--method" => {
                i += 1;
                method = args.get(i).cloned();
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
            }
            "--stage2-prior-mode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    stage2_prior_mode = match v.as_str() {
                        "none" | "no-prior" => Some(search::Stage2PriorMode::Disable),
                        "summarize" | "summarize-prior" => Some(search::Stage2PriorMode::Summarize),
                        "use" | "use-prior" => Some(search::Stage2PriorMode::Use),
                        _ => Some(search::Stage2PriorMode::Use),
                    };
                }
            }
            _ => {
                i += 1;
            }
        }
        i += 1;
    }
    if let Some(mode) = stage2_prior_mode {
        opts.stage2_prior_mode = mode;
    }
    opts.ctx = build_ctx(
        &rate_backend,
        &compression_backend,
        method.as_deref(),
        expert_spec_path.as_deref(),
    )
    .ctx;
    search::run_search_with_options(query, target, &opts);
}

trait OptionExt<T> {
    fn unwrap_or_exit(self, msg: &str) -> T;
}
impl<T> OptionExt<T> for Option<T> {
    fn unwrap_or_exit(self, msg: &str) -> T {
        self.unwrap_or_else(|| {
            eprintln!("{}", msg);
            std::process::exit(1);
        })
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for help flag early
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_usage();
        return;
    }

    if args.len() < 2 {
        print_usage();
        return;
    }

    let primitive = &args[1];
    if primitive == "batch" {
        run_batch_mode();
        return;
    }

    // Common positional and flag parsing.
    // Collect positionals only up to the first flag token, then parse flags separately.
    let mut file1: Option<String> = None;
    let mut file2: Option<String> = None;
    let mut pos_arg3: Option<String> = None;
    let mut flags_start = 2usize;

    if primitive != "search" && primitive != "aixi" {
        let mut positionals: Vec<String> = Vec::new();
        let mut i = 2usize;
        while i < args.len() {
            let tok = &args[i];
            if tok.starts_with('-') {
                break;
            }
            positionals.push(tok.clone());
            i += 1;
        }
        flags_start = i;
        file1 = positionals.first().cloned();
        file2 = positionals.get(1).cloned();
        pos_arg3 = positionals.get(2).cloned();
    }

    let mut rate_backend_str = "rosaplus".to_string();
    let mut compression_backend_str = "zpaq".to_string();
    let mut method_str: Option<String> = None;
    let mut expert_spec_path: Option<String> = None;
    let mut model_export_path: Option<String> = None;
    let mut generate_len_bytes: usize = 8;
    let mut generate_config = GenerationConfig::default();
    let mut rate_backend_specified = false;

    let mut i = flags_start;
    while i < args.len() {
        match args[i].as_str() {
            "--rate-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --rate-backend requires a value");
                rate_backend_str = parse_rate_backend(v).unwrap_or("rosaplus").to_string();
                rate_backend_specified = true;
            }
            "--ncd-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend requires a value");
                compression_backend_str =
                    parse_compression_backend(v).unwrap_or("zpaq").to_string();
            }
            "--compression-backend" => {
                i += 1;
                let v = args
                    .get(i)
                    .unwrap_or_exit("Error: --compression-backend requires a value");
                compression_backend_str =
                    parse_compression_backend(v).unwrap_or("zpaq").to_string();
            }
            "--method" => {
                i += 1;
                method_str = args.get(i).cloned();
            }
            "--expert-spec" => {
                i += 1;
                expert_spec_path = args.get(i).cloned();
                rate_backend_specified = true;
            }
            "--model-export" | "--rwkv-export" => {
                i += 1;
                model_export_path = args.get(i).cloned();
            }
            "--bytes" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --bytes requires a non-negative integer");
                generate_len_bytes = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --bytes must be a non-negative integer, got '{raw}'");
                    std::process::exit(1);
                });
            }
            "--sample" => {
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--greedy" => {
                generate_config.strategy = GenerationStrategy::Greedy;
            }
            "--adaptive" => {
                generate_config.update_mode = GenerationUpdateMode::Adaptive;
            }
            "--seed" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --seed requires an unsigned integer");
                generate_config.seed = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("Error: --seed must be an unsigned integer, got '{raw}'");
                    std::process::exit(1);
                });
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--temperature" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --temperature requires a finite number");
                generate_config.temperature = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: --temperature must be a finite number, got '{raw}'");
                    std::process::exit(1);
                });
                if !generate_config.temperature.is_finite() || generate_config.temperature < 0.0 {
                    eprintln!(
                        "Error: --temperature must be finite and non-negative, got '{}'",
                        generate_config.temperature
                    );
                    std::process::exit(1);
                }
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--top-k" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --top-k requires a non-negative integer");
                generate_config.top_k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Error: --top-k must be a non-negative integer, got '{raw}'");
                    std::process::exit(1);
                });
                generate_config.strategy = GenerationStrategy::Sample;
            }
            "--top-p" => {
                i += 1;
                let raw = args
                    .get(i)
                    .unwrap_or_exit("Error: --top-p requires a number in (0, 1]");
                generate_config.top_p = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Error: --top-p must be a number in (0, 1], got '{raw}'");
                    std::process::exit(1);
                });
                if !generate_config.top_p.is_finite()
                    || generate_config.top_p <= 0.0
                    || generate_config.top_p > 1.0
                {
                    eprintln!(
                        "Error: --top-p must be in (0, 1], got '{}'",
                        generate_config.top_p
                    );
                    std::process::exit(1);
                }
                generate_config.strategy = GenerationStrategy::Sample;
            }
            _ => {}
        }
        i += 1;
    }

    let built_ctx = build_ctx(
        &rate_backend_str,
        &compression_backend_str,
        method_str.as_deref(),
        expert_spec_path.as_deref(),
    );
    let ctx = built_ctx.ctx;
    let expert_spec_max_order = built_ctx.expert_spec_max_order;
    set_default_ctx(ctx.clone());

    match primitive.as_str() {
        "aixi" => {
            if let Some(p) = args.get(2) {
                if let Err(e) = run_aixi_mode(p) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            } else {
                eprintln!("Error: 'aixi' requires config.json");
                std::process::exit(1);
            }
        }
        "search" => search_command(&args),
        "compress" => {
            let in_path = file1.unwrap_or_exit("Error: 'compress' requires <input> <output>");
            let out_path = file2.unwrap_or_exit("Error: 'compress' requires <input> <output>");
            let data = read_file(&in_path);
            let backend = file_roundtrip_backend(&ctx.compression_backend);
            let compressed = match compress_bytes_backend(&data, &backend) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: compression failed: {e}");
                    std::process::exit(1);
                }
            };
            if let Err(e) = std::fs::write(&out_path, &compressed) {
                eprintln!("Error: failed to write output '{}': {}", out_path, e);
                std::process::exit(1);
            }
            println!(
                "compressed {} bytes -> {} bytes",
                data.len(),
                compressed.len()
            );
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "decompress" => {
            let in_path = file1.unwrap_or_exit("Error: 'decompress' requires <input> <output>");
            let out_path = file2.unwrap_or_exit("Error: 'decompress' requires <input> <output>");
            let input = read_file(&in_path);
            let backend = file_roundtrip_backend(&ctx.compression_backend);
            let decoded = match decompress_bytes_backend(&input, &backend) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: decompression failed: {e}");
                    std::process::exit(1);
                }
            };
            if let Err(e) = std::fs::write(&out_path, &decoded) {
                eprintln!("Error: failed to write output '{}': {}", out_path, e);
                std::process::exit(1);
            }
            println!(
                "decompressed {} bytes -> {} bytes",
                input.len(),
                decoded.len()
            );
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&decoded])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "generate" => {
            // Disambiguate positional args for `generate [file] [max_order]`.
            // When stdin is piped and the first positional looks like an integer,
            // treat it as max_order (not a file path).
            let stdin_is_piped = !io::stdin().is_terminal();
            let (file_path, explicit_max_order) = match (file1.as_deref(), file2.as_deref()) {
                // `generate <file> <max_order>` — both present
                (Some(f), Some(mo)) => (Some(f), mo.parse::<i64>().ok()),
                // `generate <arg>` — single positional:
                //   if stdin is piped and it parses as an integer, it's max_order
                //   otherwise it's a file path
                (Some(arg), None) if stdin_is_piped && arg.parse::<i64>().is_ok() => {
                    (None, arg.parse::<i64>().ok())
                }
                (Some(f), None) => (Some(f), None),
                // No positionals at all
                (None, _) => (None, None),
            };
            let max_order = explicit_max_order
                .or(pos_arg3.as_deref().and_then(|s| s.parse().ok()))
                .or(expert_spec_max_order)
                .unwrap_or(-1);
            let input = if let Some(path) = file_path {
                read_file(path)
            } else {
                read_stdin_all_for_generate()
            };
            let generated = ctx.generate_bytes_with_config(
                &input,
                generate_len_bytes,
                max_order,
                generate_config,
            );
            if let Err(e) = io::stdout().write_all(&generated) {
                eprintln!("Error writing generated output: {e}");
                std::process::exit(1);
            }
            if let Err(e) = io::stdout().flush() {
                eprintln!("Error flushing generated output: {e}");
                std::process::exit(1);
            }
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&input])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "ncd" | "ncd_vitanyi" | "ncd_sym" | "ncd_sym_vitanyi" | "ncd_cons" | "ncd_sym_cons" => {
            let f1 = file1.unwrap_or_exit("Error: NCD requires two files");
            let f2 = file2.unwrap_or_exit("Error: NCD requires two files");
            let _method = pos_arg3.or(method_str).unwrap_or_else(|| "5".to_string());
            let variant = match primitive.as_str() {
                "ncd_sym" | "ncd_sym_vitanyi" => NcdVariant::SymVitanyi,
                "ncd_cons" => NcdVariant::Cons,
                "ncd_sym_cons" => NcdVariant::SymCons,
                _ => NcdVariant::Vitanyi,
            };
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            println!(
                "{}",
                ncd_bytes_backend(&b1, &b2, &ctx.compression_backend, variant)
            );
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&b1, &b2])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "entropy" | "h" | "entropy_rate" | "h_rate" => {
            let f1 = file1.unwrap_or_exit("Error: 'h' requires a file");
            let default_order = if primitive.contains("rate") || rate_backend_specified {
                expert_spec_max_order.unwrap_or(-1)
            } else {
                0
            };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let data = read_file(&f1);
            if max_order == 0 && !primitive.contains("rate") && !rate_backend_specified {
                println!("{}", marginal_entropy_bytes(&data));
            } else {
                println!("{}", ctx.entropy_rate_bytes(&data, max_order));
            }
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        "id" | "intrinsic_dep" => {
            let f1 = file1.unwrap_or_exit("Error: 'id' requires a file");
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(expert_spec_max_order.unwrap_or(-1));
            let data = read_file(&f1);
            println!("{:.6}", intrinsic_dependence_bytes(&data, max_order));
            if let Err(e) = maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&data])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
        other => {
            let f1 = file1.unwrap_or_exit("Error: requires two files");
            let f2 = file2.unwrap_or_exit("Error: requires two files");
            let default_order = if rate_backend_specified {
                expert_spec_max_order.unwrap_or(-1)
            } else {
                0
            };
            let max_order = pos_arg3
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_order);
            let b1 = read_file(&f1);
            let b2 = read_file(&f2);
            let res = match other {
                "ned" => ned_bytes(&b1, &b2, max_order),
                "ned_cons" => ned_cons_bytes(&b1, &b2, max_order),
                "nte" => nte_bytes(&b1, &b2, max_order),
                "mi" | "mutual_info" => mutual_information_bytes(&b1, &b2, max_order),
                "ce" | "conditional_entropy" => conditional_entropy_bytes(&b1, &b2, max_order),
                "xe" | "cross_entropy" => cross_entropy_bytes(&b1, &b2, max_order),
                "joint_entropy" | "h_xy" => {
                    if max_order == 0 {
                        joint_marginal_entropy_bytes(&b1, &b2)
                    } else {
                        joint_entropy_rate_bytes(&b1, &b2, max_order)
                    }
                }
                "rt" | "resistance" => resistance_to_transformation_bytes(&b1, &b2, max_order),
                "tvd" => tvd_paths(&f1, &f2, max_order),
                "nhd" => nhd_paths(&f1, &f2, max_order),
                "kl" | "kl_divergence" => kl_divergence_paths(&f1, &f2),
                "js" | "js_divergence" => js_divergence_paths(&f1, &f2),
                _ => {
                    eprintln!("Unknown primitive: {}", other);
                    print_usage();
                    return;
                }
            };
            println!("{}", res);
            if let Err(e) =
                maybe_export_online_model(model_export_path.as_deref(), &ctx, &[&b1, &b2])
            {
                eprintln!("Error exporting online model: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn print_usage() {
    let rate_backends = infotheory::backends::AVAILABLE_RATE_BACKENDS
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if idx == 0 {
                format!("'{name}' (default)")
            } else {
                format!("'{name}'")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let compression_backends = infotheory::backends::AVAILABLE_COMPRESSION_BACKENDS
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            if idx == 0 {
                format!("'{name}' (default)")
            } else {
                format!("'{name}'")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    eprintln!(
        r#"InfoTheory CLI
Usage: infotheory <primitive> [args...] [options]

Primitives:
  Entropy & Information:
    h, entropy <file> [max_order]           Entropy (marginal if order=0, rate if >0)
    h_rate, entropy_rate <file> [max_order] Force entropy rate estimation
    mi, mutual_info <f1> <f2> [max_order]   Mutual Information I(X;Y)
    xe, cross_entropy <f1> <f2> [max_order] Cross Entropy H(X,Y) - H(Y)? (Check def)
    ce, conditional_entropy <f1> <f2>       Conditional Entropy H(X|Y)
    joint_entropy, h_xy <f1> <f2>           Joint Entropy H(X,Y)
    id, intrinsic_dep <file> [max_order]    Intrinsic Dependence

  Distance & Divergence:
    ncd <f1> <f2> [method]                  Normalized Compression Distance (Vitanyi)
    ncd_sym, ncd_cons, ncd_sym_cons         NCD variants (Symmetric, Conservative, etc.)
    ned <f1> <f2> [max_order]               Normalized Entropy Distance
    nte <f1> <f2> [max_order]               Normalized Transform Effort
    kl, kl_divergence <f1> <f2>             Kullback-Leibler Divergence
    js, js_divergence <f1> <f2>             Jensen-Shannon Divergence
    tvd <f1> <f2>                           Total Variation Distance
    nhd <f1> <f2>                           Normalized Hellinger Distance
    rt, resistance <f1> <f2>                Resistance to Transformation

  Tools:
    search <query> <target> [options]       Search target using info-theoretic ranking
    aixi <config.json>                      Run AIXI agent
    batch                                   Run in JSON-L batch mode
    generate [file] [max_order]             Generate continuation from file or piped stdin
    compress <in> <out>                     Compress file using selected compression backend
    decompress <in> <out>                   Decompress file using selected compression backend

Options:
    --rate-backend <name>   Backend for rate estimation: {rate_backends}
  --compression-backend <name>
                          Backend for NCD/compression: {compression_backends}
  --ncd-backend <name>    Deprecated alias for --compression-backend
  --method <val>          Method/config (e.g. '5' for zpaq, '16' for ctw, mixture spec path,
                          model method: file:/path/model.safetensors[;policy:...] or cfg:key=value,...[;policy:...])
  --expert-spec <path>    Load one exact standalone expert JSON (same schema as a mixture 'experts' entry)
  --model-export <path>   Optional online model export path (.safetensors + .json sidecar)
  --rwkv-export <path>    Backward-compatible alias for --model-export
  --bytes <n>             Bytes to generate for 'generate' (default: 8)
  --sample                Use seeded sampling for generation
  --greedy                Force deterministic greedy generation
  --adaptive              Keep fitting on generated bytes instead of frozen continuation
  --seed <u64>            RNG seed for sampled generation
  --temperature <x>       Sampling temperature (default: 1.0)
  --top-k <n>             Sample only from the top-k bytes (0 disables)
  --top-p <p>             Nucleus sampling threshold in (0, 1]

Examples:
  infotheory ncd file1.txt file2.txt --compression-backend zpaq --method 5
  infotheory ncd file1.txt file2.txt --compression-backend rate-ac --rate-backend ctw
  infotheory h file.txt --expert-spec ./expert.json
  infotheory h file.txt --rate-backend mamba --method "cfg:hidden=128,layers=2,intermediate=256,state=16,conv=4,train=adam,lr=0.001;policy:schedule=0..100:train(scope=head+bias,opt=adam,lr=0.001,stride=1,bptt=1,clip=0,momentum=0.9)" --model-export ./mamba_online.safetensors
  infotheory h file.txt --rate-backend ctw --method 32
  infotheory h file.txt --rate-backend mixture --method mixture.json
  infotheory search "encryption" ./src --prior "codebase context"
  cat prompt.txt | infotheory generate --rate-backend ctw --method 32 --bytes 8
  infotheory generate prompt.txt --rate-backend match --bytes 16 --sample --seed 7
  infotheory compress in.bin out.itc --compression-backend rate-ac --rate-backend mixture --method mixture.json
  infotheory decompress out.itc restored.bin --compression-backend rate-ac --rate-backend mixture --method mixture.json
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::any::Any;
    use std::panic;
    use std::path::Path;

    #[test]
    fn file_roundtrip_backend_keeps_zpaq_unchanged() {
        let b = CompressionBackend::Zpaq {
            method: "5".to_string(),
        };
        let out = file_roundtrip_backend(&b);
        assert!(matches!(out, CompressionBackend::Zpaq { method } if method == "5"));
    }

    #[test]
    fn file_roundtrip_backend_forces_rate_framed() {
        let b = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        };
        let out = file_roundtrip_backend(&b);
        match out {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }

    #[test]
    fn process_json_line_rejects_invalid_json() {
        let out = process_json_line(r#"{"op":"metrics","text":"abc""#);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("output should be json");
        assert!(
            parsed
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("invalid json")
        );
    }

    #[test]
    fn process_json_line_parses_escaped_and_nested_json_correctly() {
        let line = r#"{
            "op":"metrics",
            "text":"hello\n\"json\"",
            "meta":{"op":"ncd"},
            "max_order":-1
        }"#;
        let out = process_json_line(line);
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("output should be json");
        assert!(parsed.get("h0").and_then(|v| v.as_f64()).unwrap_or(-1.0) >= 0.0);
        assert_eq!(parsed.get("len").and_then(|v| v.as_u64()), Some(12));
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn build_ctx_rwkv7_compression_accepts_cfg_method() {
        let ctx = build_ctx(
            "rosaplus",
            "rwkv7",
            Some(
                "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer",
            ),
            None,
        )
        .ctx;

        match ctx.compression_backend {
            CompressionBackend::Rate {
                rate_backend,
                coder,
                framing,
            } => {
                assert!(matches!(rate_backend, RateBackend::Rwkv7Method { .. }));
                assert_eq!(coder, rwkvzip::CoderType::AC);
                assert_eq!(framing, infotheory::compression::FramingMode::Raw);
            }
            _ => panic!("expected rate-coded RWKV backend for cfg: method"),
        }
    }

    #[test]
    fn parse_backend_aliases_and_unknowns() {
        assert_eq!(parse_rate_backend("rosa"), Some("rosaplus"));
        assert_eq!(parse_rate_backend("facctw"), Some("fac-ctw"));
        assert_eq!(parse_rate_backend("sparsematch"), Some("sparse-match"));
        assert_eq!(parse_rate_backend("ppm"), Some("ppmd"));
        assert_eq!(parse_rate_backend("cal"), Some("calibrated"));
        assert_eq!(parse_rate_backend("unknown"), None);

        assert_eq!(parse_compression_backend("unknown"), None);
        #[cfg(feature = "backend-zpaq")]
        assert_eq!(parse_compression_backend("zpaq"), Some("zpaq"));
        assert_eq!(parse_compression_backend("rate_ac"), Some("rate-ac"));
        assert_eq!(parse_compression_backend("raterans"), Some("rate-rans"));
        #[cfg(feature = "backend-rwkv")]
        {
            assert_eq!(parse_compression_backend("rwkv"), Some("rwkv7"));
        }
        #[cfg(feature = "backend-mamba")]
        {
            assert_eq!(parse_rate_backend("mamba1"), Some("mamba"));
        }
    }

    #[test]
    fn parse_mixture_expert_supports_calibrated_and_match_backends() {
        let base_dir = Path::new(".");
        let expert = json!({
            "name": "cal-ctw",
            "kind": "calibrated",
            "context": "text",
            "bins": 33,
            "learning_rate": 0.02,
            "bias_clip": 4.0,
            "base": {
                "kind": "match"
            }
        });
        let parsed = parse_mixture_expert_value(&expert, base_dir, 4).expect("expert should parse");
        match parsed.backend {
            RateBackend::Calibrated { spec } => match spec.base {
                RateBackend::Match { .. } => {}
                _ => panic!("unexpected calibrated base"),
            },
            _ => panic!("expected calibrated backend"),
        }
    }

    #[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
    fn panic_message(payload: Box<dyn Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<String>() {
            return s.clone();
        }
        if let Some(s) = payload.downcast_ref::<&str>() {
            return (*s).to_string();
        }
        "non-string panic payload".to_string()
    }

    #[cfg(feature = "backend-mamba")]
    #[test]
    fn parse_mixture_expert_resolves_mamba_model_path_relative_to_base_dir() {
        let base_dir =
            std::env::temp_dir().join(format!("infotheory-mamba-relpath-{}", std::process::id()));
        std::fs::create_dir_all(base_dir.join("weights")).expect("create temp dir");
        let rel_path = "weights/model.safetensors";
        let expected = base_dir.join(rel_path).to_string_lossy().to_string();
        let expert = json!({
            "name": "mamba-relative",
            "kind": "mamba",
            "model_path": rel_path
        });
        let panic = panic::catch_unwind(|| {
            let _ = parse_mixture_expert_value(&expert, &base_dir, 4);
        })
        .expect_err("missing model should panic during load");
        let msg = panic_message(panic);
        assert!(
            msg.contains(&expected),
            "panic should mention resolved absolute model path. expected substring: {expected}, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn parse_mixture_expert_resolves_rwkv_model_path_relative_to_base_dir() {
        let base_dir =
            std::env::temp_dir().join(format!("infotheory-rwkv-relpath-{}", std::process::id()));
        std::fs::create_dir_all(base_dir.join("weights")).expect("create temp dir");
        let rel_path = "weights/model.safetensors";
        let expected = base_dir.join(rel_path).to_string_lossy().to_string();
        let expert = json!({
            "name": "rwkv-relative",
            "kind": "rwkv7",
            "model_path": rel_path
        });
        let panic = panic::catch_unwind(|| {
            let _ = parse_mixture_expert_value(&expert, &base_dir, 4);
        })
        .expect_err("missing model should panic during load");
        let msg = panic_message(panic);
        assert!(
            msg.contains(&expected),
            "panic should mention resolved absolute model path. expected substring: {expected}, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn load_expert_spec_preserves_exact_ppmd_settings() {
        let expert_path = std::env::temp_dir().join(format!(
            "infotheory-expert-spec-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(
            &expert_path,
            serde_json::to_vec(&json!({
                "name": "ppmd",
                "kind": "ppmd",
                "order": 12,
                "memory_mb": 256
            }))
            .expect("expert json"),
        )
        .expect("write temp expert spec");

        let parsed = load_expert_spec(expert_path.to_str().expect("utf8 path"))
            .expect("ppmd expert should load");
        match parsed.backend {
            RateBackend::Ppmd { order, memory_mb } => {
                assert_eq!(order, 12);
                assert_eq!(memory_mb, 256);
            }
            _ => panic!("expected ppmd backend"),
        }

        let _ = std::fs::remove_file(&expert_path);
    }

    #[test]
    fn build_ctx_propagates_expert_spec_max_order_default() {
        let expert_path = std::env::temp_dir().join(format!(
            "infotheory-expert-spec-rosa-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(
            &expert_path,
            serde_json::to_vec(&json!({
                "name": "rosa",
                "kind": "rosaplus",
                "max_order": 32
            }))
            .expect("expert json"),
        )
        .expect("write temp expert spec");

        let built = build_ctx(
            "rosaplus",
            "zpaq",
            None,
            Some(expert_path.to_str().expect("utf8 path")),
        );
        assert_eq!(built.expert_spec_max_order, Some(32));
        assert!(matches!(built.ctx.rate_backend, RateBackend::RosaPlus));

        let _ = std::fs::remove_file(&expert_path);
    }

    #[test]
    fn parse_observation_helpers_cover_vm_and_non_vm_cases() {
        let base = json!({
            "observation_stream_len": 3,
            "observation_key_mode": "stream-hash"
        });
        assert_eq!(parse_observation_stream_len(&base), 3);
        assert_eq!(
            parse_observation_key_mode(&base),
            ObservationKeyMode::StreamHash
        );
        assert_eq!(
            parse_observation_key_mode_str("full"),
            ObservationKeyMode::FullStream
        );
        assert_eq!(
            parse_observation_key_mode_str("last"),
            ObservationKeyMode::Last
        );
        assert_eq!(
            parse_observation_key_mode_str("unknown"),
            ObservationKeyMode::First
        );

        let vm = json!({
            "observation_stream_len": 2,
            "observation_key_mode": "full",
            "vm_observation": {
                "stream_len": 2,
                "key_mode": "last"
            }
        });
        assert_eq!(
            parse_observation_stream_len_for_vm(&vm["vm_observation"]),
            2
        );
        assert_eq!(
            parse_observation_key_mode_for_vm(&vm["vm_observation"]),
            ObservationKeyMode::Last
        );
        assert_eq!(parse_observation_stream_len_for_env(&vm, "vm"), 2);
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "vm"),
            ObservationKeyMode::Last
        );
        assert_eq!(
            parse_observation_key_mode_for_env(&vm, "coin"),
            ObservationKeyMode::FullStream
        );

        let mismatch = json!({
            "observation_stream_len": 2,
            "vm_observation": {
                "stream_len": 3
            }
        });
        let err = validate_observation_config("vm", &mismatch, 2, ObservationKeyMode::FullStream)
            .expect_err("mismatched vm stream_len should fail");
        assert!(err.to_string().contains("conflicts"));

        let mismatch_mode = json!({
            "observation_key_mode": "full",
            "vm_observation": {
                "key_mode": "last"
            }
        });
        let err = validate_observation_config("nyx", &mismatch_mode, 1, ObservationKeyMode::Last)
            .expect_err("mismatched vm key mode should fail");
        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn parse_mixture_kind_and_spec_validation() {
        assert_eq!(
            parse_mixture_kind("bayes-mix").expect("bayes alias"),
            MixtureKind::Bayes
        );
        assert_eq!(
            parse_mixture_kind("switch").expect("switch alias"),
            MixtureKind::Switching
        );
        assert_eq!(
            parse_mixture_kind("neural").expect("neural kind"),
            MixtureKind::Neural
        );
        assert!(parse_mixture_kind("nonsense").is_err());

        let base_dir = Path::new(".");
        let missing_experts = json!({
            "kind": "bayes",
            "experts": []
        });
        assert!(parse_mixture_spec_value(&missing_experts, base_dir, 8).is_err());

        let fading_without_decay = json!({
            "kind": "fading",
            "experts": [
                {"name": "ctw-e", "kind": "ctw", "depth": 4}
            ]
        });
        assert!(parse_mixture_spec_value(&fading_without_decay, base_dir, 8).is_err());

        let valid = json!({
            "kind": "bayes",
            "alpha": 0.03,
            "experts": [
                {"name": "ctw-e", "kind": "ctw", "depth": 8},
                {"name": "fac-e", "kind": "fac-ctw", "base_depth": 8, "encoding_bits": 8}
            ]
        });
        let spec = parse_mixture_spec_value(&valid, base_dir, 8).expect("valid mixture");
        assert_eq!(spec.alpha, 0.03);
        assert_eq!(spec.experts.len(), 2);
        assert!(matches!(spec.kind, MixtureKind::Bayes));
    }

    #[cfg(feature = "vm")]
    #[test]
    fn parse_vm_stats_backend_supports_new_backends_and_rejects_unknowns() {
        let root = json!({
            "algorithm": "ctw",
            "ct_depth": 8,
            "observation_bits": 8,
            "reward_bits": 8
        });
        let base_dir = Path::new(".");

        let matched =
            parse_vm_stats_backend(&json!({"name":"match","hash_bits":18}), &root, base_dir)
                .expect("match backend should parse");
        assert!(matches!(matched, RateBackend::Match { hash_bits: 18, .. }));

        let sparse = parse_vm_stats_backend(
            &json!({"name":"sparse-match","gap_min":2,"gap_max":4}),
            &root,
            base_dir,
        )
        .expect("sparse-match backend should parse");
        assert!(matches!(
            sparse,
            RateBackend::SparseMatch {
                gap_min: 2,
                gap_max: 4,
                ..
            }
        ));

        let ppmd = parse_vm_stats_backend(&json!({"name":"ppmd","order":12}), &root, base_dir)
            .expect("ppmd backend should parse");
        assert!(matches!(ppmd, RateBackend::Ppmd { order: 12, .. }));

        let particle = parse_vm_stats_backend(
            &json!({
                "name":"particle",
                "spec":{"num_particles":4,"num_cells":4,"cell_dim":8}
            }),
            &root,
            base_dir,
        )
        .expect("particle backend should parse");
        assert!(matches!(particle, RateBackend::Particle { .. }));

        let mixture = parse_vm_stats_backend(
            &json!({
                "name":"mixture",
                "spec":{"kind":"bayes","experts":[{"kind":"match"}]}
            }),
            &root,
            base_dir,
        )
        .expect("mixture backend should parse");
        assert!(matches!(mixture, RateBackend::Mixture { .. }));

        let calibrated = parse_vm_stats_backend(
            &json!({
                "name":"calibrated",
                "base":{"kind":"ctw","depth":8},
                "context":"text",
                "bins":17,
                "learning_rate":0.05,
                "bias_clip":3.0
            }),
            &root,
            base_dir,
        )
        .expect("calibrated backend should parse");
        assert!(matches!(calibrated, RateBackend::Calibrated { .. }));

        let err = match parse_vm_stats_backend(&json!("unknown-backend"), &root, base_dir) {
            Ok(_) => panic!("unknown backend should not silently fall back"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("unknown vm stats backend"),
            "unexpected error: {err}"
        );
    }
}
