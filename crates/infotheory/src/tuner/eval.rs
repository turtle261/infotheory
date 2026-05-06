use super::*;
use std::ffi::OsStr;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResolvedMemoryAccountingKind {
    DeterministicEvaluatorTable,
    StrictLinuxCgroupV2PeakMaxProcessRss,
    UnixProcessRssFallbackExplicit,
    UnixProcessRssWithBackendReportedDiagnosticOnly,
}

impl ResolvedMemoryAccountingKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::DeterministicEvaluatorTable => "deterministic_evaluator_table_row_peak_memory",
            Self::StrictLinuxCgroupV2PeakMaxProcessRss => {
                "strict_linux_max_process_rss_cgroup_v2_peak"
            }
            Self::UnixProcessRssFallbackExplicit => "unix_process_rss_fallback_explicit",
            Self::UnixProcessRssWithBackendReportedDiagnosticOnly => {
                "unix_process_rss_backend_reported_diagnostic_only"
            }
        }
    }

    pub(super) fn strict_theorem_memory_certified(self) -> bool {
        matches!(
            self,
            Self::DeterministicEvaluatorTable | Self::StrictLinuxCgroupV2PeakMaxProcessRss
        )
    }

    fn worker_rss_mode(self) -> PeakMemoryMode {
        match self {
            Self::DeterministicEvaluatorTable => PeakMemoryMode::ProcessRssPeak,
            Self::StrictLinuxCgroupV2PeakMaxProcessRss => PeakMemoryMode::HybridStrictMax,
            Self::UnixProcessRssFallbackExplicit => PeakMemoryMode::ProcessRssPeak,
            Self::UnixProcessRssWithBackendReportedDiagnosticOnly => PeakMemoryMode::ProcessRssPeak,
        }
    }

    fn requires_per_eval_cgroup(self) -> bool {
        matches!(self, Self::StrictLinuxCgroupV2PeakMaxProcessRss)
    }

    pub(super) fn backend_report_component_policy(self) -> &'static str {
        match self {
            Self::DeterministicEvaluatorTable => "none_deterministic_table_row",
            Self::StrictLinuxCgroupV2PeakMaxProcessRss => {
                "diagnostic_only_combined_with_os_controller_peak"
            }
            Self::UnixProcessRssFallbackExplicit => "none",
            Self::UnixProcessRssWithBackendReportedDiagnosticOnly => {
                "diagnostic_only_no_strict_os_controller_peak"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedEvaluatorRuntimeProfile {
    pub(super) worker_executable: Option<PathBuf>,
    pub(super) worker_executable_identity: Option<String>,
    pub(super) resolved_evaluator_cgroup_parent: Option<PathBuf>,
    pub(super) memory_accounting_kind: ResolvedMemoryAccountingKind,
}

impl ResolvedEvaluatorRuntimeProfile {
    pub(super) fn strict_theorem_memory_certified(&self) -> bool {
        self.memory_accounting_kind
            .strict_theorem_memory_certified()
    }

    pub(super) fn resolved_cgroup_parent_string(&self) -> Option<String> {
        self.resolved_evaluator_cgroup_parent
            .as_ref()
            .map(|path| path.to_string_lossy().to_string())
    }

    pub(super) fn to_provenance_value(&self) -> Value {
        serde_json::json!({
            "worker_executable_identity": self.worker_executable_identity.as_deref(),
            "memory_accounting_kind": self.memory_accounting_kind.name(),
            "strict_theorem_memory_certified": self.strict_theorem_memory_certified(),
            "resolved_evaluator_cgroup_parent": self.resolved_cgroup_parent_string(),
            "backend_report_component_policy": self.memory_accounting_kind.backend_report_component_policy(),
        })
    }
}

pub(super) fn resolve_evaluator_runtime_profile(
    execution: &TuneExecutionConfig,
    deterministic_table_requested: bool,
) -> Result<ResolvedEvaluatorRuntimeProfile, String> {
    if deterministic_table_requested {
        return Ok(ResolvedEvaluatorRuntimeProfile {
            worker_executable: None,
            worker_executable_identity: None,
            resolved_evaluator_cgroup_parent: None,
            memory_accounting_kind: ResolvedMemoryAccountingKind::DeterministicEvaluatorTable,
        });
    }
    #[cfg(not(unix))]
    {
        let _ = execution;
        return Err(
            "tuner requires a Unix target for process-isolated candidate evaluation".to_string(),
        );
    }
    #[cfg(unix)]
    {
        let worker_executable =
            resolve_tuner_eval_worker_executable(execution.evaluator_worker_executable.as_deref())?;
        let worker_identity = worker_executable_identity(&worker_executable)?;
        let memory_accounting_kind = resolve_memory_accounting_kind(execution.rss_mode)?;
        let resolved_evaluator_cgroup_parent = if memory_accounting_kind.requires_per_eval_cgroup()
        {
            Some(resolve_required_tuner_eval_cgroup_parent(
                execution.evaluator_cgroup_parent.as_deref(),
            )?)
        } else {
            reject_unix_fallback_cgroup_overrides(execution.evaluator_cgroup_parent.as_deref())?;
            None
        };
        Ok(ResolvedEvaluatorRuntimeProfile {
            worker_executable: Some(worker_executable),
            worker_executable_identity: Some(worker_identity),
            resolved_evaluator_cgroup_parent,
            memory_accounting_kind,
        })
    }
}

fn resolve_memory_accounting_kind(
    rss_mode: PeakMemoryMode,
) -> Result<ResolvedMemoryAccountingKind, String> {
    match rss_mode {
        PeakMemoryMode::ProcessRssPeak => {
            Ok(ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit)
        }
        PeakMemoryMode::BackendReported => {
            Ok(ResolvedMemoryAccountingKind::UnixProcessRssWithBackendReportedDiagnosticOnly)
        }
        PeakMemoryMode::HybridStrictMax => {
            #[cfg(target_os = "linux")]
            {
                Ok(ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss)
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err(
                    "strict memory-accounting mode (rss_mode=hybrid_strict_max) is Linux-only and requires delegated cgroup-v2 peak accounting"
                        .to_string(),
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_candidate(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    evaluator_threads: usize,
    runtime_profile: &ResolvedEvaluatorRuntimeProfile,
    deterministic_table: Option<&VerifiedDeterministicEvaluatorTable>,
) -> Result<CandidateEvalResult, String> {
    if let Some(table) = deterministic_table {
        let _ = evaluator_threads;
        let _ = runtime_profile;
        return table.evaluate(
            candidate,
            dataset,
            model_bytes,
            min_throughput_bytes_per_second,
            max_memory_bytes,
            effective_eval_time_limit_seconds,
        );
    }
    #[cfg(not(unix))]
    {
        let _ = candidate;
        let _ = dataset;
        let _ = model_bytes;
        let _ = min_throughput_bytes_per_second;
        let _ = max_memory_bytes;
        let _ = effective_eval_time_limit_seconds;
        let _ = evaluator_threads;
        let _ = runtime_profile;
        let _ = deterministic_table;
        return Err(
            "tuner requires a Unix target for process-isolated candidate evaluation".to_string(),
        );
    }

    #[cfg(unix)]
    {
        evaluate_candidate_unix_isolated(
            candidate,
            dataset,
            model_bytes,
            min_throughput_bytes_per_second,
            max_memory_bytes,
            effective_eval_time_limit_seconds,
            evaluator_threads,
            runtime_profile,
        )
    }
}

#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn evaluate_candidate_unix_isolated(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    evaluator_threads: usize,
    runtime_profile: &ResolvedEvaluatorRuntimeProfile,
) -> Result<CandidateEvalResult, String> {
    let worker_rss_mode = runtime_profile.memory_accounting_kind.worker_rss_mode();
    if effective_eval_time_limit_seconds <= 0.0 {
        let peak_memory_bytes = match runtime_profile.memory_accounting_kind {
            ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss => 0,
            ResolvedMemoryAccountingKind::DeterministicEvaluatorTable
            | ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit
            | ResolvedMemoryAccountingKind::UnixProcessRssWithBackendReportedDiagnosticOnly => {
                peak_memory_bytes(PeakMemoryMode::ProcessRssPeak)
            }
        };
        return Ok(timeout_eval_result(
            0.0,
            peak_memory_bytes,
            effective_eval_time_limit_seconds,
        ));
    }
    let temp_paths = EvaluatorWorkerTempPaths::new(dataset.resolved_path.as_str())?;
    let candidate_json = crate::spec::compression_backend_to_json_value(candidate.canonical_spec())
        .map_err(|err| format!("failed to serialize candidate for evaluator worker: {err}"))?;
    let request = serde_json::json!({
        "candidate": candidate_json,
        "candidate_base_dir": ".",
        "dataset_path": dataset.resolved_path,
        "model_bytes": model_bytes,
        "min_throughput_bytes_per_second": min_throughput_bytes_per_second,
        "max_memory_bytes": max_memory_bytes,
        "effective_eval_time_limit_seconds": effective_eval_time_limit_seconds,
        "rss_mode": peak_memory_mode_name(worker_rss_mode),
        "evaluator_threads": evaluator_threads,
    });
    fs::write(
        &temp_paths.request_path,
        serde_json::to_vec(&request)
            .map_err(|err| format!("failed to encode evaluator worker request: {err}"))?,
    )
    .map_err(|err| {
        format!(
            "failed to write evaluator worker request '{}': {err}",
            temp_paths.request_path.display()
        )
    })?;

    #[cfg(target_os = "linux")]
    let evaluation_cgroup = runtime_profile
        .resolved_evaluator_cgroup_parent
        .as_deref()
        .map(|parent| EvaluatorWorkerCgroup::create(parent, dataset.resolved_path.as_str()))
        .transpose()?;
    #[cfg(not(target_os = "linux"))]
    let evaluation_cgroup: Option<EvaluatorWorkerCgroup> = {
        let _ = runtime_profile;
        None
    };

    let mut command = evaluator_worker_command(runtime_profile.worker_executable.as_deref())?;
    #[cfg(target_os = "linux")]
    if let Some(cgroup) = evaluation_cgroup.as_ref() {
        cgroup.configure_worker_command(&mut command)?;
    }
    command
        .env(
            "INFOTHEORY_TUNER_EVAL_REQUEST_PATH",
            &temp_paths.request_path,
        )
        .env(
            "INFOTHEORY_TUNER_EVAL_RESPONSE_PATH",
            &temp_paths.response_path,
        )
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|err| format!("failed to spawn evaluator worker process: {err}"))?;
    let started = Instant::now();
    let timeout = Duration::from_secs_f64(effective_eval_time_limit_seconds);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("failed while waiting for evaluator worker: {err}"))?
        {
            if !status.success() {
                let stderr = child
                    .wait_with_output()
                    .ok()
                    .map(|output| String::from_utf8_lossy(&output.stderr).trim().to_string())
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| status.to_string());
                return Err(format!("evaluator worker exited unsuccessfully: {stderr}"));
            }
            break;
        }
        if started.elapsed() >= timeout {
            let peak_before_kill = peak_memory_bytes_for_live_worker(
                child.id() as libc::pid_t,
                runtime_profile.memory_accounting_kind,
                evaluation_cgroup.as_ref(),
            );
            #[cfg(target_os = "linux")]
            if let Some(cgroup) = evaluation_cgroup.as_ref() {
                let _ = cgroup.kill_all();
            }
            let _ = child.kill();
            let _ = child.wait();
            return Ok(timeout_eval_result(
                effective_eval_time_limit_seconds,
                peak_before_kill?,
                effective_eval_time_limit_seconds,
            ));
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    let payload_bytes = fs::read(&temp_paths.response_path).map_err(|err| {
        format!(
            "failed to read evaluator worker response '{}': {err}",
            temp_paths.response_path.display()
        )
    })?;
    if payload_bytes.is_empty() {
        return Err("candidate evaluation worker returned no payload".to_string());
    }
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .map_err(|err| format!("invalid evaluator payload from child process: {err}"))?;
    let mut result = parse_candidate_eval_payload(&payload)?;
    apply_authoritative_worker_peak_memory(
        &mut result,
        model_bytes,
        min_throughput_bytes_per_second,
        max_memory_bytes,
        runtime_profile.memory_accounting_kind,
        evaluation_cgroup.as_ref(),
    )?;
    Ok(result)
}

#[cfg(unix)]
fn parse_candidate_eval_payload(payload: &Value) -> Result<CandidateEvalResult, String> {
    let object = payload
        .as_object()
        .ok_or_else(|| "invalid evaluator payload shape".to_string())?;
    let ok = object
        .get("ok")
        .and_then(Value::as_bool)
        .ok_or_else(|| "evaluator payload missing boolean 'ok' field".to_string())?;
    if !ok {
        let err = object
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("candidate evaluator failed without error message");
        return Err(err.to_string());
    }
    let status = match object
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| "evaluator payload missing status".to_string())?
    {
        "success" => CandidateEvalStatus::Success,
        "timeout" => CandidateEvalStatus::Timeout,
        "invalid" => CandidateEvalStatus::Invalid,
        "error" => CandidateEvalStatus::Error,
        other => return Err(format!("unknown evaluator status '{other}'")),
    };
    let compressed_bytes = object
        .get("compressed_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "evaluator payload missing compressed_bytes".to_string())?;
    let compressed_bytes = usize::try_from(compressed_bytes)
        .map_err(|_| "evaluator payload compressed_bytes does not fit usize".to_string())?;
    let elapsed_seconds = object
        .get("elapsed_seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing elapsed_seconds".to_string())?;
    let effective_eval_time_limit_seconds = object
        .get("effective_eval_time_limit_seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "evaluator payload missing effective_eval_time_limit_seconds".to_string())?;
    if !effective_eval_time_limit_seconds.is_finite() || effective_eval_time_limit_seconds < 0.0 {
        return Err("evaluator payload has invalid effective_eval_time_limit_seconds".to_string());
    }
    let throughput_bytes_per_second = object
        .get("throughput_bytes_per_second")
        .and_then(|v| {
            if v.is_null() {
                Some(f64::INFINITY)
            } else {
                v.as_f64()
            }
        })
        .ok_or_else(|| "evaluator payload missing throughput_bytes_per_second".to_string())?;
    let peak_memory_bytes = object
        .get("peak_memory_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "evaluator payload missing peak_memory_bytes".to_string())?;
    let target_loss_bits = object
        .get("target_loss_bits")
        .and_then(|v| {
            if v.is_null() {
                Some(f64::INFINITY)
            } else {
                v.as_f64()
            }
        })
        .ok_or_else(|| "evaluator payload missing target_loss_bits".to_string())?;
    let objective_bits = object
        .get("objective_bits")
        .and_then(|v| {
            if v.is_null() {
                Some(f64::INFINITY)
            } else {
                v.as_f64()
            }
        })
        .ok_or_else(|| "evaluator payload missing objective_bits".to_string())?;
    let deployable = object
        .get("deployable")
        .and_then(Value::as_bool)
        .ok_or_else(|| "evaluator payload missing deployable".to_string())?;
    Ok(CandidateEvalResult {
        status,
        compressed_bytes,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second,
        peak_memory_bytes,
        target_loss_bits,
        objective_bits,
        deployable,
    })
}

/// Run the process-isolated tuner evaluator worker described by environment.
///
/// The parent executor sets `INFOTHEORY_TUNER_EVAL_REQUEST_PATH` to a JSON
/// request and `INFOTHEORY_TUNER_EVAL_RESPONSE_PATH` to the file where this
/// worker must write its JSON response. The worker performs exactly one
/// candidate evaluation, serializes either an `ok: true` result payload or an
/// `ok: false` error payload, and returns only after the response has been
/// written. This entrypoint is public so the CLI binary and libtest worker shim
/// can share the same evaluator contract; it is not a canonical tune-spec API.
pub fn run_tuner_eval_worker_from_env() -> Result<(), String> {
    if std::env::var_os("INFOTHEORY_TUNER_EVAL_WORKER_PING").as_deref() == Some(OsStr::new("1")) {
        return Ok(());
    }
    let request_path = std::env::var_os("INFOTHEORY_TUNER_EVAL_REQUEST_PATH")
        .ok_or_else(|| "missing INFOTHEORY_TUNER_EVAL_REQUEST_PATH".to_string())?;
    let response_path = std::env::var_os("INFOTHEORY_TUNER_EVAL_RESPONSE_PATH")
        .ok_or_else(|| "missing INFOTHEORY_TUNER_EVAL_RESPONSE_PATH".to_string())?;
    let request_bytes = fs::read(&request_path).map_err(|err| {
        format!(
            "failed to read evaluator worker request '{}': {err}",
            PathBuf::from(&request_path).display()
        )
    })?;
    let request: Value = serde_json::from_slice(&request_bytes)
        .map_err(|err| format!("invalid evaluator worker request JSON: {err}"))?;
    let payload = match run_tuner_eval_worker_request(&request) {
        Ok(result) => candidate_eval_result_payload(&result),
        Err(err) => serde_json::json!({
            "ok": false,
            "error": err,
        }),
    };
    fs::write(
        &response_path,
        serde_json::to_vec(&payload)
            .map_err(|err| format!("failed to encode evaluator worker response: {err}"))?,
    )
    .map_err(|err| {
        format!(
            "failed to write evaluator worker response '{}': {err}",
            PathBuf::from(&response_path).display()
        )
    })
}

fn run_tuner_eval_worker_request(payload: &Value) -> Result<CandidateEvalResult, String> {
    let object = payload
        .as_object()
        .ok_or_else(|| "evaluator worker request must be a JSON object".to_string())?;
    let candidate_value = object
        .get("candidate")
        .ok_or_else(|| "evaluator worker request missing candidate".to_string())?;
    let candidate_base_dir = object
        .get("candidate_base_dir")
        .and_then(Value::as_str)
        .unwrap_or(".");
    let candidate = crate::spec::parse_compression_backend_json(
        candidate_value,
        Path::new(candidate_base_dir),
        None,
        crate::compression::FramingMode::Framed,
    )
    .map_err(|err| format!("failed to parse evaluator worker candidate: {err}"))?
    .compile_in(&SpecEnvironment::new(candidate_base_dir))
    .map_err(|err| format!("failed to compile evaluator worker candidate: {err}"))?;
    let dataset_path = required_worker_str(object, "dataset_path")?;
    let dataset = load_dataset(Path::new(dataset_path))?;
    let model_bytes = required_worker_usize(object, "model_bytes")?;
    let min_throughput_bytes_per_second =
        required_worker_nonnegative_f64(object, "min_throughput_bytes_per_second")?;
    let max_memory_bytes = required_worker_u64(object, "max_memory_bytes")?;
    let effective_eval_time_limit_seconds =
        required_worker_nonnegative_f64(object, "effective_eval_time_limit_seconds")?;
    let rss_mode = parse_worker_peak_memory_mode(required_worker_str(object, "rss_mode")?)?;
    let evaluator_threads = required_worker_usize(object, "evaluator_threads")?;
    if evaluator_threads == 0 {
        return Err("evaluator_threads must be >= 1".to_string());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(evaluator_threads)
        .build_global()
        .map_err(|err| format!("failed to initialize evaluator worker thread pool: {err}"))?;
    evaluate_candidate_unbounded(
        &candidate,
        &dataset,
        model_bytes,
        min_throughput_bytes_per_second,
        max_memory_bytes,
        effective_eval_time_limit_seconds,
        rss_mode,
    )
}

fn candidate_eval_result_payload(value: &CandidateEvalResult) -> Value {
    serde_json::json!({
        "ok": true,
        "status": value.status.name(),
        "compressed_bytes": value.compressed_bytes,
        "elapsed_seconds": value.elapsed_seconds,
        "effective_eval_time_limit_seconds": value.effective_eval_time_limit_seconds,
        "throughput_bytes_per_second": value.throughput_bytes_per_second,
        "peak_memory_bytes": value.peak_memory_bytes,
        "target_loss_bits": value.target_loss_bits,
        "objective_bits": value.objective_bits,
        "deployable": value.deployable,
    })
}

fn required_worker_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("evaluator worker request field '{field}' must be a string"))
}

fn required_worker_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<u64, String> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        format!("evaluator worker request field '{field}' must be an unsigned integer")
    })
}

fn required_worker_usize(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<usize, String> {
    usize::try_from(required_worker_u64(object, field)?)
        .map_err(|_| format!("evaluator worker request field '{field}' is too large"))
}

fn required_worker_nonnegative_f64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<f64, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("evaluator worker request field '{field}' must be a number"))?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(format!(
            "evaluator worker request field '{field}' must be finite and nonnegative"
        ))
    }
}

fn parse_worker_peak_memory_mode(raw: &str) -> Result<PeakMemoryMode, String> {
    match raw {
        "process_rss_peak" => Ok(PeakMemoryMode::ProcessRssPeak),
        "backend_reported" => Ok(PeakMemoryMode::BackendReported),
        "hybrid_strict_max" => Ok(PeakMemoryMode::HybridStrictMax),
        other => Err(format!("unknown evaluator worker rss_mode '{other}'")),
    }
}

#[cfg(unix)]
struct EvaluatorWorkerTempPaths {
    request_path: PathBuf,
    response_path: PathBuf,
}

#[cfg(unix)]
impl EvaluatorWorkerTempPaths {
    fn new(label: &str) -> Result<Self, String> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_else(|_| Instant::now().elapsed().as_nanos());
        let digest = crc32_hex(label.as_bytes());
        let base = std::env::temp_dir().join(format!(
            "infotheory-tuner-eval-worker-{}-{nonce}-{digest}",
            std::process::id()
        ));
        fs::create_dir_all(&base).map_err(|err| {
            format!(
                "failed to create evaluator worker temp directory '{}': {err}",
                base.display()
            )
        })?;
        Ok(Self {
            request_path: base.join("request.json"),
            response_path: base.join("response.json"),
        })
    }

    fn cleanup(&self) -> Result<(), String> {
        let Some(parent) = self.request_path.parent() else {
            return Ok(());
        };
        fs::remove_dir_all(parent).map_err(|err| {
            format!(
                "failed to remove evaluator worker temp directory '{}': {err}",
                parent.display()
            )
        })
    }
}

#[cfg(unix)]
impl Drop for EvaluatorWorkerTempPaths {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[cfg(target_os = "linux")]
fn resolve_required_tuner_eval_cgroup_parent(explicit: Option<&str>) -> Result<PathBuf, String> {
    let explicit_path = explicit.map(PathBuf::from);
    let env_path = std::env::var_os("INFOTHEORY_TUNER_EVAL_CGROUP_PARENT").map(PathBuf::from);
    let path = if let Some(path) = explicit_path {
        path
    } else if let Some(path) = env_path {
        path
    } else {
        return Err(
            "strict memory-accounting mode (rss_mode=hybrid_strict_max) requires a delegated cgroup-v2 parent via execution.evaluator_cgroup_parent, --evaluator-cgroup-parent, or INFOTHEORY_TUNER_EVAL_CGROUP_PARENT"
                .to_string(),
        );
    };
    let canonical = validate_evaluator_cgroup_parent(&path)?;
    probe_evaluator_cgroup_parent(&canonical)?;
    Ok(canonical)
}

#[cfg(not(target_os = "linux"))]
fn resolve_required_tuner_eval_cgroup_parent(explicit: Option<&str>) -> Result<PathBuf, String> {
    let _ = explicit;
    Err(
        "strict memory-accounting mode (rss_mode=hybrid_strict_max) requires Linux cgroup-v2 per-evaluation accounting"
            .to_string(),
    )
}

#[cfg(not(target_os = "linux"))]
fn reject_unix_fallback_cgroup_overrides(explicit: Option<&str>) -> Result<(), String> {
    if explicit.is_some() || std::env::var_os("INFOTHEORY_TUNER_EVAL_CGROUP_PARENT").is_some() {
        return Err("evaluator_cgroup_parent requires Linux cgroup v2".to_string());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn reject_unix_fallback_cgroup_overrides(explicit: Option<&str>) -> Result<(), String> {
    if explicit.is_some() || std::env::var_os("INFOTHEORY_TUNER_EVAL_CGROUP_PARENT").is_some() {
        return Err(
            "evaluator_cgroup_parent is only valid for strict Linux cgroup-v2 memory accounting mode (rss_mode=hybrid_strict_max)"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn resolve_tuner_eval_worker_executable(
    explicit: Option<&str>,
) -> Result<PathBuf, String> {
    let explicit_path = explicit.map(PathBuf::from);
    let executable = evaluator_worker_executable(explicit_path.as_deref())?;
    probe_evaluator_worker_executable(&executable)?;
    Ok(executable)
}

#[cfg(not(unix))]
pub(super) fn resolve_tuner_eval_worker_executable(
    explicit: Option<&str>,
) -> Result<PathBuf, String> {
    let _ = explicit;
    Err("tuner requires a Unix target for process-isolated candidate evaluation".to_string())
}

#[cfg(unix)]
fn evaluator_worker_command(
    explicit_worker_executable: Option<&Path>,
) -> Result<std::process::Command, String> {
    let executable = evaluator_worker_executable(explicit_worker_executable)?;
    let mut command = std::process::Command::new(&executable);
    append_evaluator_worker_entrypoint(&mut command, &executable);
    Ok(command)
}

#[cfg(unix)]
fn evaluator_worker_executable(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return ensure_file_path(path.to_path_buf(), "execution.evaluator_worker_executable");
    }
    if let Some(path) = std::env::var_os("INFOTHEORY_TUNER_EVAL_WORKER_EXE") {
        return ensure_file_path(PathBuf::from(path), "INFOTHEORY_TUNER_EVAL_WORKER_EXE");
    }
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_infotheory") {
        let executable = PathBuf::from(path);
        if executable.is_file() {
            return Ok(executable);
        }
    }
    let current = std::env::current_exe().map_err(|err| {
        format!("failed to resolve evaluator worker executable from current_exe: {err}")
    })?;
    ensure_file_path(current, "current_exe")
}

#[cfg(unix)]
fn ensure_file_path(path: PathBuf, label: &str) -> Result<PathBuf, String> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "{label} '{}' does not resolve to a file",
            path.display()
        ))
    }
}

#[cfg(unix)]
fn worker_executable_identity(executable: &Path) -> Result<String, String> {
    let raw = fs::read(executable).map_err(|err| {
        format!(
            "failed to read evaluator worker executable '{}' for cache identity: {err}",
            executable.display()
        )
    })?;
    Ok(format!("crc32:{}:bytes:{}", crc32_hex(&raw), raw.len()))
}

#[cfg(unix)]
fn append_evaluator_worker_entrypoint(command: &mut std::process::Command, executable: &Path) {
    if evaluator_worker_executable_is_libtest(executable) {
        command
            .arg("__infotheory_tuner_eval_worker")
            .arg("--ignored")
            .arg("--nocapture");
    } else {
        command.arg("__infotheory-tuner-eval-worker");
    }
}

#[cfg(unix)]
fn probe_evaluator_worker_executable(executable: &Path) -> Result<(), String> {
    let mut command = std::process::Command::new(executable);
    append_evaluator_worker_entrypoint(&mut command, executable);
    let output = command
        .env("INFOTHEORY_TUNER_EVAL_WORKER_PING", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|err| {
            format!(
                "failed to probe evaluator worker executable '{}': {err}",
                executable.display()
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let detail = if stderr.is_empty() {
        output.status.to_string()
    } else {
        stderr
    };
    Err(format!(
        "evaluator worker executable '{}' is not compatible with tuner worker entrypoint: {detail}",
        executable.display()
    ))
}

#[cfg(unix)]
fn evaluator_worker_executable_is_libtest(path: &Path) -> bool {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        == Some("deps")
}

#[cfg(target_os = "linux")]
static EVALUATOR_CGROUP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(target_os = "linux")]
struct EvaluatorWorkerCgroup {
    path: PathBuf,
}

#[cfg(not(target_os = "linux"))]
struct EvaluatorWorkerCgroup;

#[cfg(target_os = "linux")]
impl EvaluatorWorkerCgroup {
    fn create(parent: &Path, label: &str) -> Result<Self, String> {
        let sequence = EVALUATOR_CGROUP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_else(|_| Instant::now().elapsed().as_nanos());
        let digest = crc32_hex(label.as_bytes());
        let path = parent.join(format!(
            "infotheory-eval-{}-{sequence}-{nonce}-{digest}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|err| {
            format!(
                "failed to create per-evaluation cgroup '{}': {err}",
                path.display()
            )
        })?;
        let cgroup = Self { path };
        cgroup.peak_memory_bytes().map_err(|err| {
            format!(
                "created cgroup '{}' but could not read cgroup-v2 memory.peak: {err}",
                cgroup.path.display()
            )
        })?;
        Ok(cgroup)
    }

    fn configure_worker_command(&self, command: &mut std::process::Command) -> Result<(), String> {
        let cgroup_procs_path = self.path.join("cgroup.procs");
        let cgroup_procs = std::fs::OpenOptions::new()
            .write(true)
            .open(&cgroup_procs_path)
            .map_err(|err| {
                format!(
                    "failed to open per-evaluation cgroup procs file '{}': {err}",
                    cgroup_procs_path.display()
                )
            })?;
        // SAFETY: `pre_exec` runs in the forked child immediately before exec.
        // The closure captures an already-open `cgroup.procs` file descriptor and
        // performs only async-signal-safe operations: `getpid`, in-bounds pointer
        // arithmetic over a stack buffer, and `write`. This moves the child into
        // the dedicated cgroup before the evaluator worker binary is exec'd, so
        // spec parsing, runtime construction, and compression are all accounted in
        // the candidate-local memory peak without running them as root.
        unsafe {
            command.pre_exec(move || {
                let pid = libc::getpid();
                let mut buffer = [0u8; 32];
                let (start, len) = decimal_pid_line(pid as u64, &mut buffer);
                let written = libc::write(
                    cgroup_procs.as_raw_fd(),
                    buffer.as_ptr().add(start).cast::<libc::c_void>(),
                    len,
                );
                if written == len as isize {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
        Ok(())
    }

    fn peak_memory_bytes(&self) -> Result<u64, String> {
        read_u64_from_file(&self.path.join("memory.peak"))
    }

    fn kill_all(&self) -> Result<(), String> {
        let kill_path = self.path.join("cgroup.kill");
        if !kill_path.exists() {
            return Ok(());
        }
        fs::write(&kill_path, b"1\n").map_err(|err| {
            format!(
                "failed to kill evaluator worker cgroup '{}': {err}",
                kill_path.display()
            )
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for EvaluatorWorkerCgroup {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.path);
    }
}

#[cfg(target_os = "linux")]
fn decimal_pid_line(mut value: u64, buffer: &mut [u8; 32]) -> (usize, usize) {
    let mut start = buffer.len() - 1;
    buffer[start] = b'\n';
    if value == 0 {
        start -= 1;
        buffer[start] = b'0';
    } else {
        while value > 0 {
            start -= 1;
            buffer[start] = b'0' + (value % 10) as u8;
            value /= 10;
        }
    }
    (start, buffer.len() - start)
}

#[cfg(target_os = "linux")]
fn validate_evaluator_cgroup_parent(path: &Path) -> Result<PathBuf, String> {
    let cgroup_root = Path::new("/sys/fs/cgroup")
        .canonicalize()
        .map_err(|err| format!("failed to resolve /sys/fs/cgroup: {err}"))?;
    let canonical = path.canonicalize().map_err(|err| {
        format!(
            "failed to resolve evaluator cgroup parent '{}': {err}",
            path.display()
        )
    })?;
    if !canonical.starts_with(&cgroup_root) {
        return Err(format!(
            "evaluator_cgroup_parent '{}' must be under '{}'",
            canonical.display(),
            cgroup_root.display()
        ));
    }
    if !canonical.is_dir() {
        return Err(format!(
            "evaluator_cgroup_parent '{}' is not a directory",
            canonical.display()
        ));
    }
    Ok(canonical)
}

#[cfg(target_os = "linux")]
fn probe_evaluator_cgroup_parent(parent: &Path) -> Result<(), String> {
    let probe = EvaluatorWorkerCgroup::create(parent, "probe")?;
    probe.peak_memory_bytes()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn peak_memory_bytes_for_live_worker(
    pid: libc::pid_t,
    accounting: ResolvedMemoryAccountingKind,
    cgroup: Option<&EvaluatorWorkerCgroup>,
) -> Result<u64, String> {
    let process = peak_rss_bytes_for_pid(pid).unwrap_or(0);
    let cgroup_peak = cgroup
        .map(EvaluatorWorkerCgroup::peak_memory_bytes)
        .transpose()?;
    Ok(match accounting {
        ResolvedMemoryAccountingKind::DeterministicEvaluatorTable => process,
        ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss => cgroup_peak
            .ok_or_else(|| {
                "strict cgroup-v2 accounting expected per-evaluation cgroup peak".to_string()
            })?
            .max(process),
        ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit
        | ResolvedMemoryAccountingKind::UnixProcessRssWithBackendReportedDiagnosticOnly => process,
    })
}

#[cfg(not(target_os = "linux"))]
fn peak_memory_bytes_for_live_worker(
    pid: libc::pid_t,
    accounting: ResolvedMemoryAccountingKind,
    _cgroup: Option<&EvaluatorWorkerCgroup>,
) -> Result<u64, String> {
    let mode = match accounting {
        ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit
        | ResolvedMemoryAccountingKind::UnixProcessRssWithBackendReportedDiagnosticOnly
        | ResolvedMemoryAccountingKind::DeterministicEvaluatorTable => {
            PeakMemoryMode::ProcessRssPeak
        }
        ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss => {
            return Err(
                "strict Linux cgroup-v2 memory accounting is unavailable on this platform"
                    .to_string(),
            );
        }
    };
    Ok(peak_memory_bytes_for_pid(pid, mode).unwrap_or(0))
}

fn apply_authoritative_worker_peak_memory(
    result: &mut CandidateEvalResult,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    accounting: ResolvedMemoryAccountingKind,
    cgroup: Option<&EvaluatorWorkerCgroup>,
) -> Result<(), String> {
    apply_authoritative_worker_peak_memory_inner(
        result,
        model_bytes,
        min_throughput_bytes_per_second,
        max_memory_bytes,
        accounting,
        cgroup,
    )
}

#[cfg(target_os = "linux")]
fn apply_authoritative_worker_peak_memory_inner(
    result: &mut CandidateEvalResult,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    accounting: ResolvedMemoryAccountingKind,
    cgroup: Option<&EvaluatorWorkerCgroup>,
) -> Result<(), String> {
    match accounting {
        ResolvedMemoryAccountingKind::DeterministicEvaluatorTable
        | ResolvedMemoryAccountingKind::UnixProcessRssFallbackExplicit
        | ResolvedMemoryAccountingKind::UnixProcessRssWithBackendReportedDiagnosticOnly => {}
        ResolvedMemoryAccountingKind::StrictLinuxCgroupV2PeakMaxProcessRss => {
            let cgroup_peak = cgroup
                .ok_or_else(|| {
                    "strict cgroup-v2 accounting expected per-evaluation cgroup handle".to_string()
                })?
                .peak_memory_bytes()?;
            result.peak_memory_bytes = result.peak_memory_bytes.max(cgroup_peak);
        }
    };
    if result.status == CandidateEvalStatus::Success {
        result.deployable = result.throughput_bytes_per_second >= min_throughput_bytes_per_second
            && result.peak_memory_bytes <= max_memory_bytes;
        result.objective_bits = if result.deployable {
            ((model_bytes as f64) * 8.0) + result.target_loss_bits
        } else {
            f64::INFINITY
        };
    } else {
        result.deployable = false;
        result.objective_bits = f64::INFINITY;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_authoritative_worker_peak_memory_inner(
    result: &mut CandidateEvalResult,
    _model_bytes: usize,
    _min_throughput_bytes_per_second: f64,
    _max_memory_bytes: u64,
    _accounting: ResolvedMemoryAccountingKind,
    _cgroup: Option<&EvaluatorWorkerCgroup>,
) -> Result<(), String> {
    let _ = result;
    Ok(())
}

fn evaluate_candidate_unbounded(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    model_bytes: usize,
    min_throughput_bytes_per_second: f64,
    max_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
    rss_mode: PeakMemoryMode,
) -> Result<CandidateEvalResult, String> {
    let before_peak = peak_memory_bytes(rss_mode);
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(effective_eval_time_limit_seconds);
    let (compressed_bytes, target_loss_bits) = match dataset.kind {
        DatasetKind::PassiveBytes => {
            let mut runtime = crate::runtime::build_compression_runtime(candidate)
                .map_err(|err| format!("failed to build candidate runtime: {err}"))?;
            let compressed_bytes_u64 = runtime
                .compress_size(&dataset.raw_bytes)
                .map_err(|err| format!("candidate evaluation failed: {err}"))?;
            let compressed_bytes = usize::try_from(compressed_bytes_u64)
                .map_err(|_| "compressed size does not fit usize on this platform".to_string())?;
            (compressed_bytes, (compressed_bytes as f64) * 8.0)
        }
        DatasetKind::InteractiveTrace | DatasetKind::CausalPrefixDataset => {
            evaluate_candidate_causal_loss(candidate, dataset, deadline)?
        }
    };
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let after_peak = peak_memory_bytes(rss_mode);
    let peak_memory_bytes = after_peak.max(before_peak);
    if elapsed_seconds >= effective_eval_time_limit_seconds || target_loss_bits.is_infinite() {
        return Ok(timeout_eval_result(
            elapsed_seconds,
            peak_memory_bytes,
            effective_eval_time_limit_seconds,
        ));
    }

    let throughput_bytes_per_second = if elapsed_seconds <= 0.0 {
        f64::INFINITY
    } else {
        dataset.dataset_units / elapsed_seconds
    };
    let deployable = throughput_bytes_per_second >= min_throughput_bytes_per_second
        && peak_memory_bytes <= max_memory_bytes;
    let objective_bits = if deployable {
        ((model_bytes as f64) * 8.0) + target_loss_bits
    } else {
        f64::INFINITY
    };

    Ok(CandidateEvalResult {
        status: CandidateEvalStatus::Success,
        compressed_bytes,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second,
        peak_memory_bytes,
        target_loss_bits,
        objective_bits,
        deployable,
    })
}

pub(super) fn timeout_eval_result(
    elapsed_seconds: f64,
    peak_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
) -> CandidateEvalResult {
    CandidateEvalResult {
        status: CandidateEvalStatus::Timeout,
        compressed_bytes: 0,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second: 0.0,
        peak_memory_bytes,
        target_loss_bits: f64::INFINITY,
        objective_bits: f64::INFINITY,
        deployable: false,
    }
}

pub(super) fn error_eval_result(
    elapsed_seconds: f64,
    peak_memory_bytes: u64,
    effective_eval_time_limit_seconds: f64,
) -> CandidateEvalResult {
    CandidateEvalResult {
        status: CandidateEvalStatus::Error,
        compressed_bytes: 0,
        elapsed_seconds,
        effective_eval_time_limit_seconds,
        throughput_bytes_per_second: 0.0,
        peak_memory_bytes,
        target_loss_bits: f64::INFINITY,
        objective_bits: f64::INFINITY,
        deployable: false,
    }
}

pub(super) fn evaluate_candidate_causal_loss(
    candidate: &crate::spec::CompiledCompressionBackend,
    dataset: &LoadedDataset,
    deadline: Instant,
) -> Result<(usize, f64), String> {
    if let crate::spec::core::CompressionBackendPlan::Rate { rate_backend, .. } = candidate.plan() {
        let causal_profile = dataset.causal_profile.as_ref().ok_or_else(|| {
            "causal dataset evaluation requires a typed causal profile".to_string()
        })?;
        let compiled_rate =
            crate::spec::core::compiled_rate_backend_from_plan(rate_backend.clone())
                .map_err(|err| format!("failed to compile causal evaluator rate backend: {err}"))?;
        let mut prefix_parts = Vec::<Vec<u8>>::new();
        let mut target_loss_bits = 0.0f64;
        for event in &dataset.events {
            if Instant::now() >= deadline {
                return Ok((0, f64::INFINITY));
            }
            match event {
                LoweredCausalEvent::Reset => prefix_parts.clear(),
                LoweredCausalEvent::Context { channel, bytes } => {
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "context", channel, None, bytes,
                    ));
                }
                LoweredCausalEvent::ObserveTargetNoScore {
                    channel,
                    domain,
                    bytes,
                } => {
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "observe_target_no_score",
                        channel,
                        Some(domain),
                        bytes,
                    ));
                }
                LoweredCausalEvent::Target {
                    channel,
                    domain,
                    bytes,
                    weight,
                } => {
                    let support = causal_profile.domains.get(domain).ok_or_else(|| {
                        format!("target event references undeclared domain '{domain}'")
                    })?;
                    let descriptor = causal_event_descriptor_bytes("target", channel, Some(domain));
                    let refs = prefix_parts
                        .iter()
                        .map(Vec::as_slice)
                        .collect::<Vec<&[u8]>>();
                    let loss = causal_target_loss_bits(
                        &refs,
                        &descriptor,
                        bytes,
                        support,
                        &compiled_rate,
                    )?;
                    target_loss_bits += (*weight) * loss;
                    prefix_parts.push(causal_event_conditioning_bytes(
                        "target",
                        channel,
                        Some(domain),
                        bytes,
                    ));
                }
            }
        }
        let compressed_bytes = (target_loss_bits / 8.0).ceil().max(0.0) as usize;
        Ok((compressed_bytes, target_loss_bits))
    } else {
        Err(
            "causal dataset evaluation requires a rate backend with conditional target-loss semantics"
                .to_string(),
        )
    }
}

fn causal_target_loss_bits(
    prefix_parts: &[&[u8]],
    descriptor: &[u8],
    target: &[u8],
    support: &CausalTargetDomain,
    compiled_rate: &crate::spec::CompiledRateBackend,
) -> Result<f64, String> {
    let mut descriptor_conditioned = Vec::<&[u8]>::with_capacity(prefix_parts.len() + 1);
    descriptor_conditioned.extend_from_slice(prefix_parts);
    descriptor_conditioned.push(descriptor);
    match support {
        CausalTargetDomain::ByteAlphabet => {
            if target.len() != 1 {
                return Err(
                    "byte_alphabet target payloads must be exactly one byte after lowering"
                        .to_string(),
                );
            }
            crate::runtime::try_cross_entropy_conditional_chain_backend(
                &descriptor_conditioned,
                target,
                compiled_rate,
            )
            .map_err(|err| format!("causal byte-domain target evaluation failed: {err}"))
        }
        CausalTargetDomain::EnumeratedPayloads { payloads } => {
            if !payloads.iter().any(|payload| payload == target) {
                return Err(
                    "target payload is outside enumerated target-domain support".to_string()
                );
            }
            let mut target_loss = None::<f64>;
            let mut log2_terms = Vec::<f64>::with_capacity(payloads.len());
            for payload in payloads {
                let loss = crate::runtime::try_cross_entropy_conditional_chain_backend(
                    &descriptor_conditioned,
                    payload,
                    compiled_rate,
                )
                .map_err(|err| {
                    format!("causal enumerated-domain target evaluation failed: {err}")
                })?;
                if payload == target {
                    target_loss = Some(loss);
                }
                log2_terms.push(-loss);
            }
            let log2_z = log2_sum_exp(&log2_terms);
            let loss = target_loss.ok_or_else(|| {
                "target payload is outside enumerated target-domain support".to_string()
            })?;
            Ok(loss + log2_z)
        }
    }
}

fn log2_sum_exp(log2_terms: &[f64]) -> f64 {
    let max_term = log2_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_term.is_finite() {
        return max_term;
    }
    let sum = log2_terms
        .iter()
        .map(|term| 2.0f64.powf(*term - max_term))
        .sum::<f64>();
    max_term + sum.log2()
}

fn causal_event_conditioning_bytes(
    kind: &str,
    channel: &str,
    domain: Option<&str>,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = causal_event_descriptor_bytes(kind, channel, domain);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn causal_event_descriptor_bytes(kind: &str, channel: &str, domain: Option<&str>) -> Vec<u8> {
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(b"infotheory:tuner:causal-event:v1\0");
    push_tag_component(&mut out, kind.as_bytes());
    push_tag_component(&mut out, channel.as_bytes());
    push_tag_component(&mut out, domain.unwrap_or("").as_bytes());
    out
}

fn push_tag_component(out: &mut Vec<u8>, component: &[u8]) {
    out.extend_from_slice(&(component.len() as u64).to_le_bytes());
    out.extend_from_slice(component);
}

pub(super) fn cache_key_for_candidate(
    candidate_bytes: &[u8],
    evaluator_profile: &EvaluatorProfile,
    dataset_hash: &str,
) -> Result<CandidateCacheKey, String> {
    let evaluator_profile_bytes = evaluator_profile.cache_identity_bytes()?;
    Ok(CandidateCacheKey {
        candidate_canonical_bytes: candidate_bytes.to_vec(),
        evaluator_profile_bytes,
        dataset_identity: dataset_hash.to_string(),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn parse_worker_payload_accepts_null_serialized_infinities() {
        let payload = serde_json::json!({
            "ok": true,
            "status": "success",
            "compressed_bytes": 0,
            "elapsed_seconds": 0.0,
            "effective_eval_time_limit_seconds": 1.0,
            "throughput_bytes_per_second": null,
            "peak_memory_bytes": 0,
            "target_loss_bits": 0.0,
            "objective_bits": 8.0,
            "deployable": true
        });

        let result = parse_candidate_eval_payload(&payload).expect("parse worker payload");

        assert_eq!(result.status, CandidateEvalStatus::Success);
        assert!(result.throughput_bytes_per_second.is_infinite());
        assert!(result.throughput_bytes_per_second.is_sign_positive());

        let payload = serde_json::json!({
            "ok": true,
            "status": "timeout",
            "compressed_bytes": 0,
            "elapsed_seconds": 1.0,
            "effective_eval_time_limit_seconds": 1.0,
            "throughput_bytes_per_second": 0.0,
            "peak_memory_bytes": 0,
            "target_loss_bits": null,
            "objective_bits": null,
            "deployable": false
        });

        let result = parse_candidate_eval_payload(&payload).expect("parse worker payload");

        assert_eq!(result.status, CandidateEvalStatus::Timeout);
        assert!(result.target_loss_bits.is_infinite());
        assert!(result.objective_bits.is_infinite());
    }
}
