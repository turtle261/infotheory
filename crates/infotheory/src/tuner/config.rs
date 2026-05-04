use serde_json::Value;
use std::fs;

/// Executor-side tuning request assembled from CLI inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneCommandRequest {
    /// Canonical tune document path (`.json` or `.itsd`).
    pub spec_path: String,
    /// Non-canonical execution controls and theorem claim inputs.
    pub execution: TuneExecutionConfig,
}

/// Executor-side tuning controls (non-canonical).
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneExecutionConfig {
    pub max_evaluations: Option<usize>,
    pub annealer_kernel_profile: AnnealerKernelProfile,
    pub cpu_affinity: Option<String>,
    /// Optional evaluator-worker internal thread count.
    ///
    /// `None` means single-threaded evaluator workers.
    ///
    /// Determinism contract:
    /// when this is greater than 1, the selected backend/runtime path must
    /// still guarantee deterministic evaluator outputs under fixed `H`
    /// (including objective, target-loss, timeout/deployability decisions, and
    /// any reported metrics used for theorem-facing claims). If that guarantee
    /// is not established for the chosen backend path, do not enable
    /// multithreaded evaluator workers for theorem-facing runs.
    pub threads: Option<usize>,
    pub warmup_baseline_runs: usize,
    pub self_improvement_rounds: usize,
    pub stagnation_reset_evals: Option<usize>,
    pub log_path: Option<String>,
    pub diagnostic_chunk_bytes: Option<usize>,
    pub rss_mode: PeakMemoryMode,
    pub planner_deployable_model: bool,
    pub warmstart_trace_refresh: bool,
    pub theorem: TuneTheoremConfig,
}

/// Theorem-facing runtime claims and certification references (non-canonical).
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TuneTheoremConfig {
    pub claim_exact_finite_mdp: bool,
    pub claim_exact_observed_markov: bool,
    pub claim_planner_convergence: bool,
    pub timing_certification_tier: TimingCertificationTier,
    pub determinism_deadline_certificate: Option<String>,
    pub observation_adapter_spec_ref: Option<String>,
    pub exact_state_encoder_spec_ref: Option<String>,
    pub scalar_representation_ref: Option<String>,
    pub finite_planner_state_certificate: Option<String>,
    pub no_hidden_state_certificate: Option<String>,
    pub exact_reward_encoding_certificate: Option<String>,
    pub exact_state_observation_certificate: Option<String>,
    pub deterministic_evaluator_table: Option<String>,
}

/// Executor-selected annealer kernel profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AnnealerKernelProfile {
    ReversibleElementaryMetropolis,
    CompiledUniformMetropolisHastings,
}

/// Peak-memory accounting mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PeakMemoryMode {
    ProcessRssPeak,
    BackendReported,
    HybridStrictMax,
}

/// Timing certification tier declaration for theorem-facing claims.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TimingCertificationTier {
    BestEffort,
    Isolated,
    RealTime,
    DeterministicTable,
}

impl Default for TuneExecutionConfig {
    fn default() -> Self {
        Self {
            max_evaluations: None,
            annealer_kernel_profile: AnnealerKernelProfile::ReversibleElementaryMetropolis,
            cpu_affinity: None,
            threads: None,
            warmup_baseline_runs: 0,
            self_improvement_rounds: 1,
            stagnation_reset_evals: None,
            log_path: None,
            diagnostic_chunk_bytes: None,
            rss_mode: PeakMemoryMode::ProcessRssPeak,
            planner_deployable_model: false,
            warmstart_trace_refresh: false,
            theorem: TuneTheoremConfig::default(),
        }
    }
}

impl Default for TuneTheoremConfig {
    fn default() -> Self {
        Self {
            claim_exact_finite_mdp: false,
            claim_exact_observed_markov: false,
            claim_planner_convergence: false,
            timing_certification_tier: TimingCertificationTier::BestEffort,
            determinism_deadline_certificate: None,
            observation_adapter_spec_ref: None,
            exact_state_encoder_spec_ref: None,
            scalar_representation_ref: None,
            finite_planner_state_certificate: None,
            no_hidden_state_certificate: None,
            exact_reward_encoding_certificate: None,
            exact_state_observation_certificate: None,
            deterministic_evaluator_table: None,
        }
    }
}

const EXECUTION_CONFIG_FIELDS: &[&str] = &[
    "max_evaluations",
    "annealer_kernel_profile",
    "cpu_affinity",
    "threads",
    "warmup_baseline_runs",
    "self_improvement_rounds",
    "stagnation_reset_evals",
    "log_path",
    "diagnostic_chunk_bytes",
    "rss_mode",
    "planner_deployable_model",
    "warmstart_trace_refresh",
    "theorem",
    // Theorem keys are accepted at the top level for sidecar/CLI parity.
    "claim_exact_finite_mdp",
    "claim_exact_observed_markov",
    "claim_planner_convergence",
    "timing_certification_tier",
    "determinism_deadline_certificate",
    "observation_adapter_spec_ref",
    "exact_state_encoder_spec_ref",
    "scalar_representation_ref",
    "finite_planner_state_certificate",
    "no_hidden_state_certificate",
    "exact_reward_encoding_certificate",
    "exact_state_observation_certificate",
    "deterministic_evaluator_table",
];

const THEOREM_CONFIG_FIELDS: &[&str] = &[
    "claim_exact_finite_mdp",
    "claim_exact_observed_markov",
    "claim_planner_convergence",
    "timing_certification_tier",
    "determinism_deadline_certificate",
    "observation_adapter_spec_ref",
    "exact_state_encoder_spec_ref",
    "scalar_representation_ref",
    "finite_planner_state_certificate",
    "no_hidden_state_certificate",
    "exact_reward_encoding_certificate",
    "exact_state_observation_certificate",
    "deterministic_evaluator_table",
];

const EXECUTION_AND_THEOREM_CONFIG_FIELDS: &[&str] = EXECUTION_CONFIG_FIELDS;

fn ensure_known_execution_config_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown execution config field '{key}'"));
        }
    }
    Ok(())
}

impl TuneExecutionConfig {
    /// Load executor controls from JSON. This is intentionally distinct from
    /// canonical `SpecDocument::Tune`.
    pub fn from_json_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "execution config must be a JSON object".to_string())?;
        ensure_known_execution_config_fields(object, EXECUTION_CONFIG_FIELDS)?;
        let mut cfg = Self::default();
        apply_optional_usize(object.get("max_evaluations"), &mut cfg.max_evaluations)?;
        if let Some(raw) = object.get("annealer_kernel_profile") {
            cfg.annealer_kernel_profile =
                parse_annealer_kernel_profile(required_str(raw, "annealer_kernel_profile")?)?;
        }
        cfg.cpu_affinity = clean_optional_string(object.get("cpu_affinity"));
        apply_optional_usize(object.get("threads"), &mut cfg.threads)?;
        if let Some(raw) = object.get("warmup_baseline_runs") {
            cfg.warmup_baseline_runs = required_usize(raw, "warmup_baseline_runs")?;
        }
        if let Some(raw) = object.get("self_improvement_rounds") {
            cfg.self_improvement_rounds = required_usize(raw, "self_improvement_rounds")?;
        }
        apply_optional_usize(
            object.get("stagnation_reset_evals"),
            &mut cfg.stagnation_reset_evals,
        )?;
        cfg.log_path = clean_optional_string(object.get("log_path"));
        apply_optional_usize(
            object.get("diagnostic_chunk_bytes"),
            &mut cfg.diagnostic_chunk_bytes,
        )?;
        if let Some(raw) = object.get("rss_mode") {
            cfg.rss_mode = parse_peak_memory_mode(required_str(raw, "rss_mode")?)?;
        }
        if let Some(raw) = object.get("planner_deployable_model") {
            cfg.planner_deployable_model = required_bool(raw, "planner_deployable_model")?;
        }
        if let Some(raw) = object.get("warmstart_trace_refresh") {
            cfg.warmstart_trace_refresh = required_bool(raw, "warmstart_trace_refresh")?;
        }

        if let Some(theorem_obj) = object.get("theorem") {
            let theorem_map = theorem_obj
                .as_object()
                .ok_or_else(|| "execution config field 'theorem' must be an object".to_string())?;
            ensure_known_execution_config_fields(theorem_map, THEOREM_CONFIG_FIELDS)?;
            cfg.apply_theorem_object(theorem_map)?;
        }
        // Accept theorem keys at the top-level for CLI/sidecar ergonomics.
        cfg.apply_theorem_object(object)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load executor controls from a JSON file path.
    pub fn from_json_path(path: &str) -> Result<Self, String> {
        let raw = fs::read(path)
            .map_err(|err| format!("failed to read execution config '{}': {err}", path))?;
        let value: Value = serde_json::from_slice(&raw)
            .map_err(|err| format!("invalid execution config JSON '{}': {err}", path))?;
        Self::from_json_value(&value)
    }

    fn apply_theorem_object(
        &mut self,
        object: &serde_json::Map<String, Value>,
    ) -> Result<(), String> {
        ensure_known_execution_config_fields(object, EXECUTION_AND_THEOREM_CONFIG_FIELDS)?;
        if let Some(raw) = object.get("claim_exact_finite_mdp") {
            self.theorem.claim_exact_finite_mdp = required_bool(raw, "claim_exact_finite_mdp")?;
        }
        if let Some(raw) = object.get("claim_exact_observed_markov") {
            self.theorem.claim_exact_observed_markov =
                required_bool(raw, "claim_exact_observed_markov")?;
        }
        if let Some(raw) = object.get("claim_planner_convergence") {
            self.theorem.claim_planner_convergence =
                required_bool(raw, "claim_planner_convergence")?;
        }
        if let Some(raw) = object.get("timing_certification_tier") {
            self.theorem.timing_certification_tier =
                parse_timing_tier(required_str(raw, "timing_certification_tier")?)?;
        }
        if let Some(raw) = object.get("determinism_deadline_certificate") {
            self.theorem.determinism_deadline_certificate =
                parse_optional_non_empty_string(Some(raw), "determinism_deadline_certificate")?;
        }
        if let Some(raw) = object.get("observation_adapter_spec_ref") {
            self.theorem.observation_adapter_spec_ref =
                parse_optional_non_empty_string(Some(raw), "observation_adapter_spec_ref")?;
        }
        if let Some(raw) = object.get("exact_state_encoder_spec_ref") {
            self.theorem.exact_state_encoder_spec_ref =
                parse_optional_non_empty_string(Some(raw), "exact_state_encoder_spec_ref")?;
        }
        if let Some(raw) = object.get("scalar_representation_ref") {
            self.theorem.scalar_representation_ref =
                parse_optional_non_empty_string(Some(raw), "scalar_representation_ref")?;
        }
        if let Some(raw) = object.get("finite_planner_state_certificate") {
            self.theorem.finite_planner_state_certificate =
                parse_optional_non_empty_string(Some(raw), "finite_planner_state_certificate")?;
        }
        if let Some(raw) = object.get("no_hidden_state_certificate") {
            self.theorem.no_hidden_state_certificate =
                parse_optional_non_empty_string(Some(raw), "no_hidden_state_certificate")?;
        }
        if let Some(raw) = object.get("exact_reward_encoding_certificate") {
            self.theorem.exact_reward_encoding_certificate =
                parse_optional_non_empty_string(Some(raw), "exact_reward_encoding_certificate")?;
        }
        if let Some(raw) = object.get("exact_state_observation_certificate") {
            self.theorem.exact_state_observation_certificate =
                parse_optional_non_empty_string(Some(raw), "exact_state_observation_certificate")?;
        }
        if let Some(raw) = object.get("deterministic_evaluator_table") {
            self.theorem.deterministic_evaluator_table =
                parse_optional_non_empty_string(Some(raw), "deterministic_evaluator_table")?;
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if let Some(threads) = self.threads
            && threads == 0
        {
            return Err("threads must be >= 1 when set".to_string());
        }
        if let Some(max_evaluations) = self.max_evaluations
            && max_evaluations == 0
        {
            return Err("max_evaluations must be >= 1 when set; the normative baseline evaluation counts as the first non-warmup admitted candidate result".to_string());
        }
        if self.self_improvement_rounds == 0 {
            return Err("self_improvement_rounds must be >= 1".to_string());
        }
        if let Some(bytes) = self.diagnostic_chunk_bytes
            && bytes == 0
        {
            return Err("diagnostic_chunk_bytes must be >= 1 when set".to_string());
        }
        Ok(())
    }

    pub(crate) fn evaluator_threads(&self) -> usize {
        self.threads.unwrap_or(1)
    }

    /// Evaluator determinism declaration for report/provenance profile output.
    ///
    /// This is a declaration of the expected determinism contract, not a proof.
    pub(crate) fn evaluator_determinism(&self) -> &'static str {
        if self.evaluator_threads() == 1 {
            "deterministic_under_h"
        } else {
            "requires_backend_determinism_when_threaded"
        }
    }

    /// Serialize this executor profile for report/provenance output.
    pub fn to_json_value(&self) -> Value {
        serde_json::json!({
            "max_evaluations": self.max_evaluations,
            "annealer_kernel_profile": annealer_kernel_profile_name(self.annealer_kernel_profile),
            "cpu_affinity": self.cpu_affinity,
            "threads": self.threads,
            "evaluator_threads": self.evaluator_threads(),
            "parent_controller_threads": 1usize,
            "worker_isolation_mode": "spawn_exec_worker",
            "evaluator_determinism": self.evaluator_determinism(),
            "warmup_baseline_runs": self.warmup_baseline_runs,
            "self_improvement_rounds": self.self_improvement_rounds,
            "stagnation_reset_evals": self.stagnation_reset_evals,
            "log_path": self.log_path,
            "diagnostic_chunk_bytes": self.diagnostic_chunk_bytes,
            "rss_mode": peak_memory_mode_name(self.rss_mode),
            "planner_deployable_model": self.planner_deployable_model,
            "warmstart_trace_refresh": self.warmstart_trace_refresh,
            "theorem": {
                "claim_exact_finite_mdp": self.theorem.claim_exact_finite_mdp,
                "claim_exact_observed_markov": self.theorem.claim_exact_observed_markov,
                "claim_planner_convergence": self.theorem.claim_planner_convergence,
                "timing_certification_tier": timing_tier_name(self.theorem.timing_certification_tier),
                "determinism_deadline_certificate": self.theorem.determinism_deadline_certificate,
                "observation_adapter_spec_ref": self.theorem.observation_adapter_spec_ref,
                "exact_state_encoder_spec_ref": self.theorem.exact_state_encoder_spec_ref,
                "scalar_representation_ref": self.theorem.scalar_representation_ref,
                "finite_planner_state_certificate": self.theorem.finite_planner_state_certificate,
                "no_hidden_state_certificate": self.theorem.no_hidden_state_certificate,
                "exact_reward_encoding_certificate": self.theorem.exact_reward_encoding_certificate,
                "exact_state_observation_certificate": self.theorem.exact_state_observation_certificate,
                "deterministic_evaluator_table": self.theorem.deterministic_evaluator_table,
            }
        })
    }
}

/// Parse `infotheory tune ...` command arguments into canonical spec path and
/// executor config.
pub fn parse_tune_command_args(args: &[String]) -> Result<TuneCommandRequest, String> {
    if args.len() < 3 {
        return Err("Error: 'tune' requires <spec.json|spec.itsd>".to_string());
    }
    let mut request = TuneCommandRequest {
        spec_path: args[2].clone(),
        execution: TuneExecutionConfig::default(),
    };
    let mut exec_config_path = None::<String>;
    let mut i = 3usize;
    while i < args.len() {
        if args[i].as_str() == "--exec-config" {
            i += 1;
            exec_config_path = Some(
                args.get(i)
                    .ok_or_else(|| "Error: --exec-config requires a JSON path".to_string())?
                    .clone(),
            );
        }
        i += 1;
    }
    if let Some(path) = exec_config_path {
        request.execution = TuneExecutionConfig::from_json_path(&path)?;
    }
    i = 3usize;
    while i < args.len() {
        match args[i].as_str() {
            "--exec-config" => {
                i += 1;
                let _ = args
                    .get(i)
                    .ok_or_else(|| "Error: --exec-config requires a JSON path".to_string())?;
            }
            "--max-evaluations" => {
                i += 1;
                request.execution.max_evaluations =
                    Some(parse_cli_usize(args.get(i), "--max-evaluations")?);
            }
            "--annealer-kernel-profile" => {
                i += 1;
                request.execution.annealer_kernel_profile = parse_annealer_kernel_profile(
                    parse_cli_str(args.get(i), "--annealer-kernel-profile")?,
                )?;
            }
            "--cpu-affinity" => {
                i += 1;
                request.execution.cpu_affinity =
                    Some(parse_cli_str(args.get(i), "--cpu-affinity")?.to_string());
            }
            "--threads" => {
                i += 1;
                request.execution.threads = Some(parse_cli_usize(args.get(i), "--threads")?);
            }
            "--warmup-baseline-runs" => {
                i += 1;
                request.execution.warmup_baseline_runs =
                    parse_cli_usize(args.get(i), "--warmup-baseline-runs")?;
            }
            "--self-improvement-rounds" => {
                i += 1;
                request.execution.self_improvement_rounds =
                    parse_cli_usize(args.get(i), "--self-improvement-rounds")?;
            }
            "--stagnation-reset-evals" => {
                i += 1;
                request.execution.stagnation_reset_evals =
                    Some(parse_cli_usize(args.get(i), "--stagnation-reset-evals")?);
            }
            "--log-path" => {
                i += 1;
                request.execution.log_path =
                    Some(parse_cli_str(args.get(i), "--log-path")?.to_string());
            }
            "--diagnostic-chunk-bytes" => {
                i += 1;
                request.execution.diagnostic_chunk_bytes =
                    Some(parse_cli_usize(args.get(i), "--diagnostic-chunk-bytes")?);
            }
            "--rss-mode" => {
                i += 1;
                request.execution.rss_mode =
                    parse_peak_memory_mode(parse_cli_str(args.get(i), "--rss-mode")?)?;
            }
            "--planner-deployable-model" => {
                request.execution.planner_deployable_model = true;
            }
            "--warmstart-trace-refresh" => {
                request.execution.warmstart_trace_refresh = true;
            }
            "--timing-tier" => {
                i += 1;
                request.execution.theorem.timing_certification_tier =
                    parse_timing_tier(parse_cli_str(args.get(i), "--timing-tier")?)?;
            }
            "--determinism-deadline-certificate" => {
                i += 1;
                request.execution.theorem.determinism_deadline_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--determinism-deadline-certificate")?
                        .to_string(),
                );
            }
            "--deterministic-evaluator-table" => {
                i += 1;
                request.execution.theorem.deterministic_evaluator_table = Some(
                    parse_cli_non_empty_str(args.get(i), "--deterministic-evaluator-table")?
                        .to_string(),
                );
            }
            "--finite-planner-state-certificate" => {
                i += 1;
                request.execution.theorem.finite_planner_state_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--finite-planner-state-certificate")?
                        .to_string(),
                );
            }
            "--no-hidden-state-certificate" => {
                i += 1;
                request.execution.theorem.no_hidden_state_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--no-hidden-state-certificate")?
                        .to_string(),
                );
            }
            "--exact-reward-encoding-certificate" => {
                i += 1;
                request.execution.theorem.exact_reward_encoding_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-reward-encoding-certificate")?
                        .to_string(),
                );
            }
            "--exact-state-observation-certificate" => {
                i += 1;
                request
                    .execution
                    .theorem
                    .exact_state_observation_certificate = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-state-observation-certificate")?
                        .to_string(),
                );
            }
            "--observation-adapter-spec-ref" => {
                i += 1;
                request.execution.theorem.observation_adapter_spec_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--observation-adapter-spec-ref")?
                        .to_string(),
                );
            }
            "--exact-state-encoder-spec-ref" => {
                i += 1;
                request.execution.theorem.exact_state_encoder_spec_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--exact-state-encoder-spec-ref")?
                        .to_string(),
                );
            }
            "--scalar-representation-ref" => {
                i += 1;
                request.execution.theorem.scalar_representation_ref = Some(
                    parse_cli_non_empty_str(args.get(i), "--scalar-representation-ref")?
                        .to_string(),
                );
            }
            "--claim-exact-finite-mdp" => {
                request.execution.theorem.claim_exact_finite_mdp = true;
            }
            "--claim-exact-observed-markov" => {
                request.execution.theorem.claim_exact_observed_markov = true;
            }
            "--claim-planner-convergence" => {
                request.execution.theorem.claim_planner_convergence = true;
            }
            other => {
                return Err(format!("Error: unknown tune option '{other}'"));
            }
        }
        i += 1;
    }
    request.execution.validate()?;
    Ok(request)
}

fn required_str<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label} must be a string"))
}

fn required_bool(value: &Value, label: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label} must be a boolean"))
}

fn required_usize(value: &Value, label: &str) -> Result<usize, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{label} must be an unsigned integer"))?;
    usize::try_from(raw).map_err(|_| format!("{label} is too large"))
}

fn apply_optional_usize(value: Option<&Value>, output: &mut Option<usize>) -> Result<(), String> {
    if let Some(raw) = value {
        *output = Some(required_usize(raw, "value")?);
    }
    Ok(())
}

fn parse_annealer_kernel_profile(raw: &str) -> Result<AnnealerKernelProfile, String> {
    match raw {
        "reversible_elementary_metropolis" => {
            Ok(AnnealerKernelProfile::ReversibleElementaryMetropolis)
        }
        "compiled_uniform_metropolis_hastings" => {
            Ok(AnnealerKernelProfile::CompiledUniformMetropolisHastings)
        }
        other => Err(format!(
            "unknown annealer kernel profile '{other}', expected 'reversible_elementary_metropolis' or 'compiled_uniform_metropolis_hastings'"
        )),
    }
}

fn parse_peak_memory_mode(raw: &str) -> Result<PeakMemoryMode, String> {
    match raw {
        "process_rss_peak" => Ok(PeakMemoryMode::ProcessRssPeak),
        "backend_reported" => Ok(PeakMemoryMode::BackendReported),
        "hybrid_strict_max" => Ok(PeakMemoryMode::HybridStrictMax),
        other => Err(format!(
            "unknown rss mode '{other}', expected 'process_rss_peak', 'backend_reported', or 'hybrid_strict_max'"
        )),
    }
}

fn parse_timing_tier(raw: &str) -> Result<TimingCertificationTier, String> {
    match raw {
        "best_effort" => Ok(TimingCertificationTier::BestEffort),
        "isolated" => Ok(TimingCertificationTier::Isolated),
        "real_time" => Ok(TimingCertificationTier::RealTime),
        "deterministic_table" => Ok(TimingCertificationTier::DeterministicTable),
        other => Err(format!(
            "unknown timing tier '{other}', expected 'best_effort', 'isolated', 'real_time', or 'deterministic_table'"
        )),
    }
}

fn clean_optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_optional_non_empty_string(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(raw)) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Err(format!("{label} must be a non-empty string when set"))
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("{label} must be a string or null")),
    }
}

fn parse_cli_str<'a>(value: Option<&'a String>, label: &str) -> Result<&'a str, String> {
    value
        .map(String::as_str)
        .ok_or_else(|| format!("Error: {label} requires a value"))
}

fn parse_cli_non_empty_str<'a>(value: Option<&'a String>, label: &str) -> Result<&'a str, String> {
    let raw = parse_cli_str(value, label)?;
    if raw.trim().is_empty() {
        Err(format!("Error: {label} requires a non-empty value"))
    } else {
        Ok(raw)
    }
}

fn parse_cli_usize(value: Option<&String>, label: &str) -> Result<usize, String> {
    let raw = parse_cli_str(value, label)?;
    raw.parse::<usize>()
        .map_err(|_| format!("Error: {label} expects an unsigned integer"))
}

pub(super) fn compiled_feature_set() -> Vec<&'static str> {
    let mut features = Vec::<&'static str>::new();
    if cfg!(feature = "default-backends") {
        features.push("default-backends");
    }
    if cfg!(feature = "capability-default") {
        features.push("capability-default");
    }
    if cfg!(feature = "capability-statistical") {
        features.push("capability-statistical");
    }
    if cfg!(feature = "capability-neural") {
        features.push("capability-neural");
    }
    if cfg!(feature = "capability-archive") {
        features.push("capability-archive");
    }
    if cfg!(feature = "capability-vm") {
        features.push("capability-vm");
    }
    if cfg!(feature = "aixi") {
        features.push("aixi");
    }
    if cfg!(feature = "tuner") {
        features.push("tuner");
    }
    if cfg!(feature = "aixi-gameengine") {
        features.push("aixi-gameengine");
    }
    if cfg!(feature = "aixi-gameengine-physics") {
        features.push("aixi-gameengine-physics");
    }
    if cfg!(feature = "aixi-vm") {
        features.push("aixi-vm");
    }
    if cfg!(feature = "all-backends") {
        features.push("all-backends");
    }
    if cfg!(feature = "backend-rosa") {
        features.push("backend-rosa");
    }
    if cfg!(feature = "backend-ctw") {
        features.push("backend-ctw");
    }
    if cfg!(feature = "backend-match") {
        features.push("backend-match");
    }
    if cfg!(feature = "backend-ppmd") {
        features.push("backend-ppmd");
    }
    if cfg!(feature = "backend-sequitur") {
        features.push("backend-sequitur");
    }
    if cfg!(feature = "backend-mixture") {
        features.push("backend-mixture");
    }
    if cfg!(feature = "backend-particle") {
        features.push("backend-particle");
    }
    if cfg!(feature = "backend-calibrated") {
        features.push("backend-calibrated");
    }
    if cfg!(feature = "backend-mamba") {
        features.push("backend-mamba");
    }
    if cfg!(feature = "backend-rwkv") {
        features.push("backend-rwkv");
    }
    if cfg!(feature = "backend-zpaq") {
        features.push("backend-zpaq");
    }
    if cfg!(feature = "cli") {
        features.push("cli");
    }
    if cfg!(feature = "vm") {
        features.push("vm");
    }
    features
}

pub(super) fn annealer_kernel_profile_name(value: AnnealerKernelProfile) -> &'static str {
    match value {
        AnnealerKernelProfile::ReversibleElementaryMetropolis => "reversible_elementary_metropolis",
        AnnealerKernelProfile::CompiledUniformMetropolisHastings => {
            "compiled_uniform_metropolis_hastings"
        }
    }
}

pub(super) fn peak_memory_mode_name(value: PeakMemoryMode) -> &'static str {
    match value {
        PeakMemoryMode::ProcessRssPeak => "process_rss_peak",
        PeakMemoryMode::BackendReported => "backend_reported",
        PeakMemoryMode::HybridStrictMax => "hybrid_strict_max",
    }
}

pub(super) fn timing_tier_name(value: TimingCertificationTier) -> &'static str {
    match value {
        TimingCertificationTier::BestEffort => "best_effort",
        TimingCertificationTier::Isolated => "isolated",
        TimingCertificationTier::RealTime => "real_time",
        TimingCertificationTier::DeterministicTable => "deterministic_table",
    }
}
