#![allow(clippy::needless_pass_by_value)]

use infotheory::api::{
    self, CalibratedSpec, CalibrationContextKind, CompiledCompressionBackend, CompiledRateBackend,
    CompressionBackend, GenerationConfig, GenerationStrategy, GenerationUpdateMode, InfotheoryCtx,
    MixtureExpertSpec, MixtureKind, MixtureScheduleMode, MixtureSpec, NcdVariant, ParticleSpec,
    RateBackend, RateBackendSession,
};
use infotheory::error::InfotheoryError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn py_try<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => Err(PyRuntimeError::new_err(
            "infotheory internal panic while executing binding call",
        )),
    }
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Fatal policy for Python trait callbacks:
/// any unhandled callback exception aborts the process after traceback output.
///
/// This avoids silently continuing MCTS/planning with default fallback values.
fn fatal_python_callback_error(py: Python<'_>, where_: &'static str, err: PyErr) -> ! {
    eprintln!("fatal: unhandled Python callback exception in {where_}; terminating process");
    err.print(py);
    std::process::exit(1);
}

/// Extracts a Python result or terminates on callback exception.
fn py_result_or_fatal<T>(py: Python<'_>, where_: &'static str, result: PyResult<T>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => fatal_python_callback_error(py, where_, e),
    }
}

/// `hasattr` variant that follows the same fatal callback-exception policy.
fn py_hasattr_or_fatal(obj: &Bound<'_, PyAny>, name: &str, where_: &'static str) -> bool {
    match obj.hasattr(name) {
        Ok(v) => v,
        Err(e) => fatal_python_callback_error(obj.py(), where_, e),
    }
}

fn py_spec_value_error(err: infotheory::spec::SpecError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn compile_rate_backend(backend: RateBackend) -> PyResult<CompiledRateBackend> {
    backend.compile().map_err(py_spec_value_error)
}

fn compile_compression_backend(
    backend: CompressionBackend,
) -> PyResult<CompiledCompressionBackend> {
    backend.compile().map_err(py_spec_value_error)
}

fn py_infotheory_error(err: InfotheoryError) -> PyErr {
    match err {
        InfotheoryError::InvalidBackendConfig(_)
        | InfotheoryError::Unsupported(_)
        | InfotheoryError::Spec(_) => PyValueError::new_err(err.to_string()),
        InfotheoryError::Runtime(_) | InfotheoryError::Io(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
    }
}

fn parse_ncd_variant(s: &str) -> PyResult<NcdVariant> {
    match s.to_ascii_lowercase().as_str() {
        "vitanyi" | "v" => Ok(NcdVariant::Vitanyi),
        "sym_vitanyi" | "sym" | "s" => Ok(NcdVariant::SymVitanyi),
        "cons" | "c" => Ok(NcdVariant::Cons),
        "sym_cons" | "sc" => Ok(NcdVariant::SymCons),
        _ => Err(PyValueError::new_err(format!("unknown NcdVariant: {s}"))),
    }
}

fn parse_framing_mode(s: &str) -> PyResult<infotheory::compression::FramingMode> {
    match s.to_ascii_lowercase().as_str() {
        "raw" => Ok(infotheory::compression::FramingMode::Raw),
        "framed" | "frame" => Ok(infotheory::compression::FramingMode::Framed),
        _ => Err(PyValueError::new_err(format!(
            "unknown framing '{s}' (expected 'raw' or 'framed')"
        ))),
    }
}

fn parse_observation_key_mode(
    py_obj: &Bound<'_, PyAny>,
) -> PyResult<infotheory::aixi::common::ObservationKeyMode> {
    if let Ok(mode) = py_obj.extract::<PyRef<'_, PyObservationKeyMode>>() {
        return Ok(mode.inner);
    }

    if let Ok(s) = py_obj.extract::<String>() {
        match s.to_ascii_lowercase().as_str() {
            "first" => return Ok(infotheory::aixi::common::ObservationKeyMode::First),
            "last" => return Ok(infotheory::aixi::common::ObservationKeyMode::Last),
            "streamhash" | "stream_hash" | "stream-hash" | "hash" => {
                return Ok(infotheory::aixi::common::ObservationKeyMode::StreamHash);
            }
            "full" | "stream" | "fullstream" | "full_stream" | "full-stream" => {
                return Ok(infotheory::aixi::common::ObservationKeyMode::FullStream);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown ObservationKeyMode '{s}' (expected one of: first, last, hash/stream_hash/stream-hash, full/full-stream/full_stream/fullstream/stream)"
                )));
            }
        }
    }
    Err(PyValueError::new_err(
        "ObservationKeyMode must be an ObservationKeyMode enum value or string alias",
    ))
}

fn parse_generation_strategy_value(py_obj: &Bound<'_, PyAny>) -> PyResult<GenerationStrategy> {
    if let Ok(strategy) = py_obj.extract::<PyRef<'_, PyGenerationStrategy>>() {
        return Ok(strategy.inner);
    }
    if let Ok(s) = py_obj.extract::<String>() {
        return match s.to_ascii_lowercase().as_str() {
            "greedy" => Ok(GenerationStrategy::Greedy),
            "sample" | "sampled" => Ok(GenerationStrategy::Sample),
            _ => Err(PyValueError::new_err(format!(
                "unknown GenerationStrategy '{s}' (expected 'greedy' or 'sample')"
            ))),
        };
    }
    Err(PyValueError::new_err(
        "GenerationStrategy must be a GenerationStrategy enum value or string alias",
    ))
}

fn parse_generation_update_mode_value(py_obj: &Bound<'_, PyAny>) -> PyResult<GenerationUpdateMode> {
    if let Ok(mode) = py_obj.extract::<PyRef<'_, PyGenerationUpdateMode>>() {
        return Ok(mode.inner);
    }
    if let Ok(s) = py_obj.extract::<String>() {
        return match s.to_ascii_lowercase().as_str() {
            "adaptive" => Ok(GenerationUpdateMode::Adaptive),
            "frozen" => Ok(GenerationUpdateMode::Frozen),
            _ => Err(PyValueError::new_err(format!(
                "unknown GenerationUpdateMode '{s}' (expected 'adaptive' or 'frozen')"
            ))),
        };
    }
    Err(PyValueError::new_err(
        "GenerationUpdateMode must be a GenerationUpdateMode enum value or string alias",
    ))
}

fn generation_config_from_py(config: Option<&Bound<'_, PyAny>>) -> PyResult<GenerationConfig> {
    if let Some(obj) = config {
        if let Ok(cfg) = obj.extract::<PyRef<'_, PyGenerationConfig>>() {
            return Ok(cfg.inner);
        }
        return Err(PyValueError::new_err(
            "config must be a GenerationConfig instance",
        ));
    }
    Ok(GenerationConfig::default())
}

#[pyclass(name = "GenerationStrategy", from_py_object)]
#[derive(Clone, Copy)]
struct PyGenerationStrategy {
    inner: GenerationStrategy,
}

#[pymethods]
impl PyGenerationStrategy {
    #[classattr]
    #[pyo3(name = "Greedy")]
    fn greedy() -> Self {
        Self {
            inner: GenerationStrategy::Greedy,
        }
    }

    #[classattr]
    #[pyo3(name = "Sample")]
    fn sample() -> Self {
        Self {
            inner: GenerationStrategy::Sample,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            GenerationStrategy::Greedy => "GenerationStrategy.Greedy",
            GenerationStrategy::Sample => "GenerationStrategy.Sample",
        }
    }
}

#[pyclass(name = "GenerationUpdateMode", from_py_object)]
#[derive(Clone, Copy)]
struct PyGenerationUpdateMode {
    inner: GenerationUpdateMode,
}

#[pymethods]
impl PyGenerationUpdateMode {
    #[classattr]
    #[pyo3(name = "Adaptive")]
    fn adaptive() -> Self {
        Self {
            inner: GenerationUpdateMode::Adaptive,
        }
    }

    #[classattr]
    #[pyo3(name = "Frozen")]
    fn frozen() -> Self {
        Self {
            inner: GenerationUpdateMode::Frozen,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            GenerationUpdateMode::Adaptive => "GenerationUpdateMode.Adaptive",
            GenerationUpdateMode::Frozen => "GenerationUpdateMode.Frozen",
        }
    }
}

#[pyclass(name = "GenerationConfig", from_py_object)]
#[derive(Clone, Copy)]
struct PyGenerationConfig {
    inner: GenerationConfig,
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (
        strategy=None,
        update_mode=None,
        seed=42,
        temperature=1.0,
        top_k=0,
        top_p=1.0
    ))]
    fn new(
        strategy: Option<&Bound<'_, PyAny>>,
        update_mode: Option<&Bound<'_, PyAny>>,
        seed: u64,
        temperature: f64,
        top_k: usize,
        top_p: f64,
    ) -> PyResult<Self> {
        let mut inner = GenerationConfig::default();
        if let Some(strategy) = strategy {
            inner.strategy = parse_generation_strategy_value(strategy)?;
        }
        if let Some(update_mode) = update_mode {
            inner.update_mode = parse_generation_update_mode_value(update_mode)?;
        }
        inner.seed = seed;
        inner.temperature = temperature;
        inner.top_k = top_k;
        inner.top_p = top_p;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn greedy_frozen() -> Self {
        Self {
            inner: GenerationConfig::greedy_frozen(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (seed=42))]
    fn sampled_frozen(seed: u64) -> Self {
        Self {
            inner: GenerationConfig::sampled_frozen(seed),
        }
    }

    #[getter]
    fn strategy(&self) -> PyGenerationStrategy {
        PyGenerationStrategy {
            inner: self.inner.strategy,
        }
    }

    #[getter]
    fn update_mode(&self) -> PyGenerationUpdateMode {
        PyGenerationUpdateMode {
            inner: self.inner.update_mode,
        }
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    #[getter]
    fn top_k(&self) -> usize {
        self.inner.top_k
    }

    #[getter]
    fn top_p(&self) -> f64 {
        self.inner.top_p
    }
}

fn parse_calibration_context_kind_alias(s: &str) -> Option<CalibrationContextKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "global" => Some(CalibrationContextKind::Global),
        "byteclass" | "byte-class" | "byte_class" | "class" => {
            Some(CalibrationContextKind::ByteClass)
        }
        "text" => Some(CalibrationContextKind::Text),
        "repeat" => Some(CalibrationContextKind::Repeat),
        "textrepeat" | "text-repeat" | "text_repeat" => Some(CalibrationContextKind::TextRepeat),
        _ => None,
    }
}

fn parse_calibration_context_kind_value(
    py_obj: &Bound<'_, PyAny>,
) -> PyResult<CalibrationContextKind> {
    if let Ok(mode) = py_obj.extract::<PyRef<'_, PyCalibrationContextKind>>() {
        return Ok(mode.inner);
    }

    if let Ok(s) = py_obj.extract::<String>() {
        return parse_calibration_context_kind_alias(&s)
            .ok_or_else(|| PyValueError::new_err(format!("unknown calibration context '{s}'")));
    }

    Err(PyValueError::new_err(
        "CalibrationContextKind must be a CalibrationContextKind enum value or string alias",
    ))
}

fn parse_rate_backend(name: &str, method: Option<&str>) -> PyResult<RateBackend> {
    let opts = infotheory::spec::RateBackendShorthandOptions::default();
    infotheory::spec::parse_rate_backend_name_method(name, method, &opts)
        .map_err(py_spec_value_error)
}

fn parse_compression_backend(
    name: &str,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
) -> PyResult<CompressionBackend> {
    #[cfg(feature = "backend-rwkv")]
    let mut opts = infotheory::spec::CompressionBackendShorthandOptions {
        default_rate_backend: rate_backend.clone(),
        default_framing: infotheory::compression::FramingMode::Framed,
        ..Default::default()
    };
    #[cfg(not(feature = "backend-rwkv"))]
    let opts = infotheory::spec::CompressionBackendShorthandOptions {
        default_rate_backend: rate_backend,
        default_framing: infotheory::compression::FramingMode::Framed,
        ..Default::default()
    };
    #[cfg(feature = "backend-rwkv")]
    if name.trim().eq_ignore_ascii_case("rwkv7") || name.trim().eq_ignore_ascii_case("rwkv") {
        if let Ok(path) = std::env::var("RWKV7_MODEL_PATH") {
            opts.default_rwkv_model_path = Some(path);
        }
    }
    infotheory::spec::parse_compression_backend_name_method(name, method, None, &opts)
        .map_err(py_spec_value_error)
}

#[pyclass(name = "MixtureKind", from_py_object)]
#[derive(Clone)]
struct PyMixtureKind {
    inner: MixtureKind,
}

#[pymethods]
impl PyMixtureKind {
    #[classattr]
    #[pyo3(name = "Bayes")]
    fn bayes() -> Self {
        Self {
            inner: MixtureKind::Bayes,
        }
    }
    #[classattr]
    #[pyo3(name = "FadingBayes")]
    fn fading_bayes() -> Self {
        Self {
            inner: MixtureKind::FadingBayes,
        }
    }
    #[classattr]
    #[pyo3(name = "Switching")]
    fn switching() -> Self {
        Self {
            inner: MixtureKind::Switching,
        }
    }
    #[classattr]
    #[pyo3(name = "Convex")]
    fn convex() -> Self {
        Self {
            inner: MixtureKind::Convex,
        }
    }
    #[classattr]
    #[pyo3(name = "Mdl")]
    fn mdl() -> Self {
        Self {
            inner: MixtureKind::Mdl,
        }
    }
    #[classattr]
    #[pyo3(name = "Neural")]
    fn neural() -> Self {
        Self {
            inner: MixtureKind::Neural,
        }
    }
}

#[pyclass(name = "MixtureScheduleMode", from_py_object)]
#[derive(Clone)]
struct PyMixtureScheduleMode {
    inner: MixtureScheduleMode,
}

#[pymethods]
impl PyMixtureScheduleMode {
    #[classattr]
    #[pyo3(name = "Default")]
    fn default_mode() -> Self {
        Self {
            inner: MixtureScheduleMode::Default,
        }
    }

    #[classattr]
    #[pyo3(name = "Theorem")]
    fn theorem() -> Self {
        Self {
            inner: MixtureScheduleMode::Theorem,
        }
    }
}

#[pyclass(name = "MixtureExpertSpec", from_py_object)]
#[derive(Clone)]
struct PyMixtureExpertSpec {
    inner: MixtureExpertSpec,
}

#[pymethods]
impl PyMixtureExpertSpec {
    #[new]
    #[pyo3(signature = (backend, max_order=-1, log_prior=0.0, name=None))]
    fn new(backend: &PyRateBackend, max_order: i64, log_prior: f64, name: Option<String>) -> Self {
        Self {
            inner: MixtureExpertSpec {
                name,
                log_prior,
                max_order,
                backend: backend.inner.clone(),
            },
        }
    }
}

#[pyclass(name = "MixtureSpec", from_py_object)]
#[derive(Clone)]
struct PyMixtureSpec {
    inner: MixtureSpec,
}

#[pymethods]
impl PyMixtureSpec {
    #[new]
    #[pyo3(signature = (kind, experts, alpha=0.01, decay=None, schedule=None))]
    fn new(
        kind: &PyMixtureKind,
        experts: Vec<PyMixtureExpertSpec>,
        alpha: f64,
        decay: Option<f64>,
        schedule: Option<&PyMixtureScheduleMode>,
    ) -> PyResult<Self> {
        let mut spec = MixtureSpec::new(kind.inner, experts.into_iter().map(|e| e.inner).collect())
            .with_schedule(schedule.map(|mode| mode.inner).unwrap_or_default())
            .with_alpha(alpha);
        if let Some(d) = decay {
            spec = spec.with_decay(d);
        }
        spec.validate()
            .map_err(|e| PyValueError::new_err(format!("invalid MixtureSpec: {e}")))?;
        Ok(Self { inner: spec })
    }
}

#[pyclass(name = "ParticleSpec", from_py_object)]
#[derive(Clone)]
struct PyParticleSpec {
    inner: ParticleSpec,
}

#[pymethods]
impl PyParticleSpec {
    #[new]
    #[pyo3(signature = (
        num_particles=16,
        context_window=32,
        unroll_steps=2,
        num_cells=8,
        cell_dim=32,
        num_rules=4,
        selector_hidden=64,
        rule_hidden=64,
        noise_dim=8,
        deterministic=true,
        enable_noise=false,
        noise_scale=0.10,
        noise_anneal_steps=8192,
        learning_rate_readout=0.01,
        learning_rate_selector=1e-4,
        learning_rate_rule=3e-4,
        bptt_depth=3,
        optimizer_momentum=0.05,
        grad_clip=1.0,
        state_clip=8.0,
        forget_lambda=0.0,
        resample_threshold=0.5,
        mutate_fraction=0.1,
        mutate_scale=0.01,
        mutate_model_params=false,
        diagnostics_interval=0,
        min_prob=5.960_464_477_539_063e-8,
        seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_particles: usize,
        context_window: usize,
        unroll_steps: usize,
        num_cells: usize,
        cell_dim: usize,
        num_rules: usize,
        selector_hidden: usize,
        rule_hidden: usize,
        noise_dim: usize,
        deterministic: bool,
        enable_noise: bool,
        noise_scale: f64,
        noise_anneal_steps: usize,
        learning_rate_readout: f64,
        learning_rate_selector: f64,
        learning_rate_rule: f64,
        bptt_depth: usize,
        optimizer_momentum: f64,
        grad_clip: f64,
        state_clip: f64,
        forget_lambda: f64,
        resample_threshold: f64,
        mutate_fraction: f64,
        mutate_scale: f64,
        mutate_model_params: bool,
        diagnostics_interval: usize,
        min_prob: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let spec = ParticleSpec {
            num_particles,
            context_window,
            unroll_steps,
            num_cells,
            cell_dim,
            num_rules,
            selector_hidden,
            rule_hidden,
            noise_dim,
            deterministic,
            enable_noise,
            noise_scale,
            noise_anneal_steps,
            learning_rate_readout,
            learning_rate_selector,
            learning_rate_rule,
            bptt_depth,
            optimizer_momentum,
            grad_clip,
            state_clip,
            forget_lambda,
            resample_threshold,
            mutate_fraction,
            mutate_scale,
            mutate_model_params,
            diagnostics_interval,
            min_prob,
            seed,
        };
        spec.validate()
            .map_err(|e| PyValueError::new_err(format!("invalid ParticleSpec: {e}")))?;
        Ok(Self { inner: spec })
    }

    fn __repr__(&self) -> String {
        format!(
            "ParticleSpec(num_particles={}, num_cells={}, cell_dim={})",
            self.inner.num_particles, self.inner.num_cells, self.inner.cell_dim
        )
    }
}

#[pyclass(name = "CalibrationContextKind", from_py_object)]
#[derive(Clone, Copy)]
struct PyCalibrationContextKind {
    inner: CalibrationContextKind,
}

#[pymethods]
impl PyCalibrationContextKind {
    #[classattr]
    #[pyo3(name = "Global")]
    fn global() -> Self {
        Self {
            inner: CalibrationContextKind::Global,
        }
    }

    #[classattr]
    #[pyo3(name = "ByteClass")]
    fn byte_class() -> Self {
        Self {
            inner: CalibrationContextKind::ByteClass,
        }
    }

    #[classattr]
    #[pyo3(name = "Text")]
    fn text() -> Self {
        Self {
            inner: CalibrationContextKind::Text,
        }
    }

    #[classattr]
    #[pyo3(name = "Repeat")]
    fn repeat() -> Self {
        Self {
            inner: CalibrationContextKind::Repeat,
        }
    }

    #[classattr]
    #[pyo3(name = "TextRepeat")]
    fn text_repeat() -> Self {
        Self {
            inner: CalibrationContextKind::TextRepeat,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            CalibrationContextKind::Global => "CalibrationContextKind.Global",
            CalibrationContextKind::ByteClass => "CalibrationContextKind.ByteClass",
            CalibrationContextKind::Text => "CalibrationContextKind.Text",
            CalibrationContextKind::Repeat => "CalibrationContextKind.Repeat",
            CalibrationContextKind::TextRepeat => "CalibrationContextKind.TextRepeat",
        }
    }
}

#[pyclass(name = "RateBackend", from_py_object)]
#[derive(Clone)]
struct PyRateBackend {
    inner: RateBackend,
}

#[pymethods]
impl PyRateBackend {
    #[staticmethod]
    fn rosaplus() -> Self {
        Self {
            inner: RateBackend::RosaPlus,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (depth=16))]
    fn ctw(depth: usize) -> Self {
        Self {
            inner: RateBackend::Ctw { depth },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_depth=16, num_percept_bits=8, encoding_bits=8))]
    fn fac_ctw(base_depth: usize, num_percept_bits: usize, encoding_bits: usize) -> Self {
        Self {
            inner: RateBackend::FacCtw {
                base_depth,
                num_percept_bits,
                encoding_bits,
            },
        }
    }

    #[staticmethod]
    #[pyo3(name = "match")]
    #[pyo3(signature = (hash_bits=20, min_len=4, max_len=255, base_mix=0.02, confidence_scale=1.0))]
    fn match_backend(
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        base_mix: f64,
        confidence_scale: f64,
    ) -> Self {
        Self {
            inner: RateBackend::Match {
                hash_bits,
                min_len,
                max_len,
                base_mix,
                confidence_scale,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        hash_bits=19,
        min_len=3,
        max_len=64,
        gap_min=1,
        gap_max=2,
        base_mix=0.05,
        confidence_scale=1.0
    ))]
    fn sparse_match(
        hash_bits: usize,
        min_len: usize,
        max_len: usize,
        gap_min: usize,
        gap_max: usize,
        base_mix: f64,
        confidence_scale: f64,
    ) -> Self {
        Self {
            inner: RateBackend::SparseMatch {
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (order=10, memory_mb=64))]
    fn ppmd(order: usize, memory_mb: usize) -> Self {
        Self {
            inner: RateBackend::Ppmd { order, memory_mb },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (method=None))]
    fn zpaq(method: Option<String>) -> PyResult<Self> {
        infotheory::spec::resolve_enabled_rate_backend_name("zpaq").map_err(py_spec_value_error)?;
        Ok(Self {
            inner: RateBackend::Zpaq {
                method: infotheory::api::ZpaqMethodSpec::literal(
                    method.unwrap_or_else(|| "2".to_string()),
                ),
            },
        })
    }

    #[staticmethod]
    #[cfg(feature = "backend-mamba")]
    fn mamba(method: String) -> PyResult<Self> {
        parse_rate_backend("mamba", Some(method.as_str())).map(|inner| Self { inner })
    }

    #[staticmethod]
    #[cfg(feature = "backend-rwkv")]
    fn rwkv7(method: String) -> PyResult<Self> {
        parse_rate_backend("rwkv7", Some(method.as_str())).map(|inner| Self { inner })
    }

    #[staticmethod]
    fn mixture(spec: &PyMixtureSpec) -> PyResult<Self> {
        spec.inner
            .validate()
            .map_err(|e| PyValueError::new_err(format!("invalid MixtureSpec: {e}")))?;
        Ok(Self {
            inner: RateBackend::Mixture {
                spec: Arc::new(spec.inner.clone()),
            },
        })
    }

    #[staticmethod]
    fn particle(spec: &PyParticleSpec) -> Self {
        Self {
            inner: RateBackend::Particle {
                spec: Arc::new(spec.inner.clone()),
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (base_backend, context=None, bins=33, learning_rate=0.02, bias_clip=4.0))]
    fn calibrated(
        base_backend: &PyRateBackend,
        context: Option<&Bound<'_, PyAny>>,
        bins: usize,
        learning_rate: f64,
        bias_clip: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RateBackend::Calibrated {
                spec: Arc::new(CalibratedSpec {
                    base: base_backend.inner.clone(),
                    context: context
                        .map(parse_calibration_context_kind_value)
                        .transpose()?
                        .unwrap_or(CalibrationContextKind::Text),
                    bins,
                    learning_rate,
                    bias_clip,
                }),
            },
        })
    }

    fn __repr__(&self) -> String {
        "RateBackend(...)".to_string()
    }
}

#[pyclass(name = "CompressionBackend", from_py_object)]
#[derive(Clone)]
struct PyCompressionBackend {
    inner: CompressionBackend,
}

#[pymethods]
impl PyCompressionBackend {
    #[staticmethod]
    #[pyo3(signature = (method=None))]
    fn zpaq(method: Option<String>) -> PyResult<Self> {
        infotheory::spec::resolve_enabled_compression_backend_name("zpaq")
            .map_err(py_spec_value_error)?;
        Ok(Self {
            inner: CompressionBackend::Zpaq {
                method: infotheory::api::ZpaqMethodSpec::literal(
                    method.unwrap_or_else(|| "5".to_string()),
                ),
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (rate_backend, framing="framed"))]
    fn rate_ac(rate_backend: &PyRateBackend, framing: &str) -> PyResult<Self> {
        let framing = parse_framing_mode(framing)?;
        Ok(Self {
            inner: CompressionBackend::Rate {
                rate_backend: rate_backend.inner.clone(),
                coder: infotheory::coders::CoderType::AC,
                framing,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (rate_backend, framing="framed"))]
    fn rate_rans(rate_backend: &PyRateBackend, framing: &str) -> PyResult<Self> {
        let framing = parse_framing_mode(framing)?;
        Ok(Self {
            inner: CompressionBackend::Rate {
                rate_backend: rate_backend.inner.clone(),
                coder: infotheory::coders::CoderType::RANS,
                framing,
            },
        })
    }

    #[staticmethod]
    #[cfg(feature = "backend-rwkv")]
    #[pyo3(signature = (method=None, coder="ac"))]
    fn rwkv7(method: Option<String>, coder: &str) -> PyResult<Self> {
        let coder = infotheory::backends::parse_rwkv7_coder(coder)
            .ok_or_else(|| PyValueError::new_err("coder must be 'ac' or 'rans'"))?;
        let mut opts = infotheory::spec::CompressionBackendShorthandOptions {
            default_framing: infotheory::compression::FramingMode::Framed,
            ..Default::default()
        };
        if let Ok(path) = std::env::var("RWKV7_MODEL_PATH") {
            opts.default_rwkv_model_path = Some(path);
        }
        let inner = infotheory::spec::parse_rwkv7_compression_backend_method(
            method.as_deref(),
            coder,
            &opts,
        )
        .map_err(py_spec_value_error)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "CompressionBackend(...)".to_string()
    }
}

#[pyclass(name = "InfotheoryCtx", from_py_object)]
#[derive(Clone)]
struct PyInfotheoryCtx {
    inner: InfotheoryCtx,
}

#[pyclass(name = "RateBackendSession", from_py_object)]
#[derive(Clone)]
struct PyRateBackendSession {
    inner: Arc<Mutex<RateBackendSession>>,
}

#[pymethods]
impl PyInfotheoryCtx {
    #[new]
    #[pyo3(signature = (rate_backend=None, compression_backend=None))]
    fn new(
        rate_backend: Option<&PyRateBackend>,
        compression_backend: Option<&PyCompressionBackend>,
    ) -> PyResult<Self> {
        let rb = compile_rate_backend(match rate_backend {
            Some(backend) => backend.inner.clone(),
            None => RateBackend::try_default().map_err(py_infotheory_error)?,
        })?;
        let cb = compile_compression_backend(match compression_backend {
            Some(backend) => backend.inner.clone(),
            None => CompressionBackend::try_default().map_err(py_infotheory_error)?,
        })?;
        Ok(Self {
            inner: InfotheoryCtx::new(rb, cb),
        })
    }

    fn entropy_rate_bytes(&self, py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_entropy_rate_bytes(data, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn biased_entropy_rate_bytes(
        &self,
        py: Python<'_>,
        data: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_biased_entropy_rate_bytes(data, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn compress_size(&self, py: Python<'_>, data: &[u8]) -> PyResult<u64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_compress_size(data)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn compress_size_chain(&self, py: Python<'_>, parts: Vec<Vec<u8>>) -> PyResult<u64> {
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_compress_size_chain(&refs)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn cross_entropy_rate_bytes(
        &self,
        py: Python<'_>,
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_cross_entropy_rate_bytes(test_data, train_data, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn cross_entropy_bytes(
        &self,
        py: Python<'_>,
        test_data: &[u8],
        train_data: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_cross_entropy_bytes(test_data, train_data, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn joint_entropy_rate_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_joint_entropy_rate_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn conditional_entropy_rate_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_conditional_entropy_rate_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn cross_entropy_conditional_chain(
        &self,
        py: Python<'_>,
        prefix_parts: Vec<Vec<u8>>,
        data: &[u8],
    ) -> PyResult<f64> {
        let refs: Vec<&[u8]> = prefix_parts.iter().map(Vec::as_slice).collect();
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_cross_entropy_conditional_chain(&refs, data)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn mutual_information_rate_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_mutual_information_rate_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn mutual_information_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_mutual_information_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn conditional_entropy_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_conditional_entropy_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn ned_bytes(&self, py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_ned_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn ned_cons_bytes(&self, py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_ned_cons_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn nte_bytes(&self, py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_nte_bytes(x, y, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn intrinsic_dependence_bytes(
        &self,
        py: Python<'_>,
        data: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_intrinsic_dependence_bytes(data, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    fn resistance_to_transformation_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        tx: &[u8],
        max_order: i64,
    ) -> PyResult<f64> {
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_resistance_to_transformation_bytes(x, tx, max_order)
                    .map_err(py_infotheory_error)
            })
        })
    }

    #[pyo3(signature = (prompt, bytes, max_order=-1, config=None))]
    fn generate_bytes<'py>(
        &self,
        py: Python<'py>,
        prompt: &[u8],
        bytes: usize,
        max_order: i64,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = generation_config_from_py(config)?;
        let out: Vec<u8> = py.detach(|| {
            py_try(|| {
                self.inner
                    .try_generate_bytes_with_config(prompt, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?;
        Ok(PyBytes::new(py, &out))
    }

    #[pyo3(signature = (prefix_parts, bytes, max_order=-1, config=None))]
    fn generate_bytes_conditional_chain<'py>(
        &self,
        py: Python<'py>,
        prefix_parts: Vec<Vec<u8>>,
        bytes: usize,
        max_order: i64,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = generation_config_from_py(config)?;
        let refs: Vec<&[u8]> = prefix_parts.iter().map(Vec::as_slice).collect();
        let out: Vec<u8> = py.detach(|| {
            py_try(|| {
                self.inner
                    .try_generate_bytes_conditional_chain_with_config(&refs, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?;
        Ok(PyBytes::new(py, &out))
    }

    #[pyo3(signature = (max_order=-1, total_symbols=None))]
    fn rate_backend_session(
        &self,
        max_order: i64,
        total_symbols: Option<u64>,
    ) -> PyResult<PyRateBackendSession> {
        let inner = self
            .inner
            .rate_backend_session(max_order, total_symbols)
            .map_err(py_infotheory_error)?;
        Ok(PyRateBackendSession {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[pyo3(signature = (x, y, variant=None))]
    fn ncd_bytes(
        &self,
        py: Python<'_>,
        x: &[u8],
        y: &[u8],
        variant: Option<String>,
    ) -> PyResult<f64> {
        let v = parse_ncd_variant(variant.as_deref().unwrap_or("vitanyi"))?;
        py.detach(|| {
            py_try(|| {
                self.inner
                    .try_ncd_bytes(x, y, v)
                    .map_err(py_infotheory_error)
            })
        })
    }

    #[pyo3(signature = (x, y, variant=None))]
    fn ncd_paths(
        &self,
        py: Python<'_>,
        x: &str,
        y: &str,
        variant: Option<String>,
    ) -> PyResult<f64> {
        let v = parse_ncd_variant(variant.as_deref().unwrap_or("vitanyi"))?;
        py.detach(|| {
            py_try(|| {
                api::try_ncd_paths_backend(x, y, self.inner.compression_backend.canonical_spec(), v)
                    .map_err(py_infotheory_error)
            })
        })
    }
}

#[pymethods]
impl PyRateBackendSession {
    #[new]
    #[pyo3(signature = (backend, max_order=-1, total_symbols=None))]
    fn new(backend: &PyRateBackend, max_order: i64, total_symbols: Option<u64>) -> PyResult<Self> {
        py_try(|| {
            let inner = RateBackendSession::from_backend(
                compile_rate_backend(backend.inner.clone())?,
                max_order,
                total_symbols,
            )
            .map_err(py_infotheory_error)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            })
        })
    }

    fn observe(&self, data: &[u8]) {
        lock_recover(&self.inner).observe(data);
    }

    fn condition(&self, data: &[u8]) {
        lock_recover(&self.inner).condition(data);
    }

    #[pyo3(signature = (total_symbols=None))]
    fn reset_frozen(&self, total_symbols: Option<u64>) -> PyResult<()> {
        lock_recover(&self.inner)
            .reset_frozen(total_symbols)
            .map_err(py_infotheory_error)
    }

    fn fill_log_probs(&self) -> Vec<f64> {
        let mut out = [0.0f64; 256];
        lock_recover(&self.inner).fill_log_probs(&mut out);
        out.to_vec()
    }

    #[pyo3(signature = (bytes, config=None))]
    fn generate_bytes<'py>(
        &self,
        py: Python<'py>,
        bytes: usize,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = generation_config_from_py(config)?;
        let out = {
            let mut guard = lock_recover(&self.inner);
            guard.generate_bytes(bytes, cfg)
        };
        Ok(PyBytes::new(py, &out))
    }

    fn finish(&self) -> PyResult<()> {
        lock_recover(&self.inner)
            .finish()
            .map_err(py_infotheory_error)
    }
}

#[pyfunction]
fn get_default_ctx() -> PyResult<PyInfotheoryCtx> {
    Ok(PyInfotheoryCtx {
        inner: api::get_default_ctx().map_err(py_infotheory_error)?,
    })
}

#[pyfunction]
fn set_default_ctx(ctx: &PyInfotheoryCtx) {
    api::set_default_ctx(ctx.inner.clone());
}

#[pyfunction]
#[pyo3(signature = (x, y, tolerance=1e-9))]
fn verify_identity(x: &[u8], y: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_identity(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        x,
        tolerance,
    ) && infotheory::axioms::verify_identity(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        y,
        tolerance,
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, tolerance=1e-9))]
fn verify_symmetry(x: &[u8], y: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_symmetry(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        x,
        y,
        tolerance,
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, z, tolerance=1e-9))]
fn verify_triangle_inequality(x: &[u8], y: &[u8], z: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_triangle_inequality(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        x,
        y,
        z,
        tolerance,
    )
}

#[pyfunction]
fn verify_non_negativity(x: &[u8], y: &[u8]) -> bool {
    infotheory::axioms::verify_non_negativity(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        x,
        y,
    )
}

#[pyfunction]
fn verify_mi_nonnegative(x: &[u8], y: &[u8]) -> bool {
    infotheory::axioms::verify_mi_nonnegative(api::mutual_information_marg_bytes, x, y)
}

#[pyfunction]
#[pyo3(signature = (x, y, tolerance=1e-9))]
fn verify_subadditivity(x: &[u8], y: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_subadditivity(
        api::joint_marginal_entropy_bytes,
        api::marginal_entropy_bytes,
        x,
        y,
        tolerance,
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, tolerance=1e-9))]
fn verify_conditioning_reduces_entropy(x: &[u8], y: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_conditioning_reduces_entropy(
        |a, b| api::try_conditional_entropy_bytes(a, b, 6).unwrap_or(f64::NAN),
        api::marginal_entropy_bytes,
        x,
        y,
        tolerance,
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, tolerance=1e-9))]
fn verify_chain_rule(x: &[u8], y: &[u8], tolerance: f64) -> bool {
    infotheory::axioms::verify_chain_rule(
        |a, b| api::try_joint_entropy_rate_bytes(a, b, 6).unwrap_or(f64::NAN),
        |a| api::try_entropy_rate_bytes(a, 6).unwrap_or(f64::NAN),
        |a, b| api::try_conditional_entropy_rate_bytes(a, b, 6).unwrap_or(f64::NAN),
        x,
        y,
        tolerance,
    )
}

#[pyfunction]
fn verify_ncd_bounds(x: &[u8], y: &[u8]) -> bool {
    infotheory::axioms::verify_ncd_bounds(
        |a, b| api::try_ncd_bytes(a, b, "5", NcdVariant::Vitanyi).unwrap_or(f64::NAN),
        x,
        y,
    )
}

#[pyfunction]
fn verify_entropy_bounds(data: &[u8]) -> bool {
    infotheory::axioms::verify_entropy_bounds(api::marginal_entropy_bytes, data)
}

fn compression_backend_from_py(
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
) -> PyResult<CompressionBackend> {
    if let Some(b) = backend {
        if let Ok(typed) = b.extract::<PyRef<'_, PyCompressionBackend>>() {
            return Ok(typed.inner.clone());
        }
        if let Ok(name) = b.extract::<String>() {
            return parse_compression_backend(&name, method, rate_backend);
        }
        return Err(PyValueError::new_err(
            "compression backend must be CompressionBackend or string",
        ));
    }
    parse_compression_backend("zpaq", method, rate_backend)
}

fn file_roundtrip_backend(backend: &CompressionBackend) -> CompressionBackend {
    infotheory::backends::normalize_file_roundtrip_backend(backend)
}

fn compiled_compression_backend_from_py(
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
) -> PyResult<CompiledCompressionBackend> {
    compile_compression_backend(compression_backend_from_py(backend, method, rate_backend)?)
}

fn rate_backend_from_py(
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<RateBackend> {
    if let Some(b) = backend {
        if let Ok(typed) = b.extract::<PyRef<'_, PyRateBackend>>() {
            return Ok(typed.inner.clone());
        }
        if let Ok(name) = b.extract::<String>() {
            return parse_rate_backend(&name, method);
        }
        return Err(PyValueError::new_err(
            "rate backend must be RateBackend or string",
        ));
    }
    RateBackend::try_default().map_err(py_infotheory_error)
}

fn compiled_rate_backend_from_py(
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<CompiledRateBackend> {
    compile_rate_backend(rate_backend_from_py(backend, method)?)
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5", variant="vitanyi", backend=None))]
fn ncd_paths(
    py: Python<'_>,
    x: &str,
    y: &str,
    method: &str,
    variant: &str,
    backend: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    let cb = compression_backend_from_py(backend, Some(method), None)?;
    py.detach(|| py_try(|| api::try_ncd_paths_backend(x, y, &cb, v).map_err(py_infotheory_error)))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5", variant="vitanyi", backend=None))]
fn ncd_bytes(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    method: &str,
    variant: &str,
    backend: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    let cb = compiled_compression_backend_from_py(backend, Some(method), None)?;
    py.detach(|| py_try(|| api::try_ncd_bytes_backend(x, y, &cb, v).map_err(py_infotheory_error)))
}

#[pyfunction]
#[pyo3(signature = (x, y, backend=None, method=None, variant="vitanyi"))]
fn ncd_paths_with_backend(
    py: Python<'_>,
    x: &str,
    y: &str,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    variant: &str,
) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    let cb = compression_backend_from_py(backend, method, None)?;
    py.detach(|| py_try(|| api::try_ncd_paths_backend(x, y, &cb, v).map_err(py_infotheory_error)))
}

#[pyfunction]
#[pyo3(signature = (x, y, backend=None, method=None, variant="vitanyi"))]
fn ncd_bytes_with_backend(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    variant: &str,
) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    let cb = compiled_compression_backend_from_py(backend, method, None)?;
    py.detach(|| py_try(|| api::try_ncd_bytes_backend(x, y, &cb, v).map_err(py_infotheory_error)))
}

#[pyfunction]
#[pyo3(signature = (x, y, variant="vitanyi"))]
fn ncd_bytes_default(py: Python<'_>, x: &[u8], y: &[u8], variant: &str) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    py.detach(|| py_try(|| api::try_ncd_bytes_default(x, y, v).map_err(py_infotheory_error)))
}

#[pyfunction]
fn entropy_rate_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| api::try_entropy_rate_bytes(data, max_order).map_err(py_infotheory_error))
    })
}

#[pyfunction]
fn biased_entropy_rate_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| api::try_biased_entropy_rate_bytes(data, max_order).map_err(py_infotheory_error))
    })
}

#[pyfunction]
fn marginal_entropy_bytes(data: &[u8]) -> f64 {
    api::marginal_entropy_bytes(data)
}

#[pyfunction]
fn joint_marginal_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::joint_marginal_entropy_bytes(x, y)
}

macro_rules! py_metric_bytes_3 {
    ($fn_name:ident, $target:path) => {
        #[pyfunction]
        fn $fn_name(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
            py.detach(|| py_try(|| Ok($target(x, y, max_order))))
        }
    };
}

macro_rules! py_metric_bytes_3_try {
    ($fn_name:ident, $target:path) => {
        #[pyfunction]
        fn $fn_name(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
            py.detach(|| py_try(|| $target(x, y, max_order).map_err(py_infotheory_error)))
        }
    };
}

macro_rules! py_metric_paths_3_try {
    ($fn_name:ident, $target:path) => {
        #[pyfunction]
        fn $fn_name(py: Python<'_>, x: &str, y: &str, max_order: i64) -> PyResult<f64> {
            py.detach(|| {
                py_try(|| {
                    $target(x, y, max_order).map_err(|e| PyRuntimeError::new_err(e.to_string()))
                })
            })
        }
    };
}

py_metric_bytes_3_try!(joint_entropy_rate_bytes, api::try_joint_entropy_rate_bytes);
py_metric_bytes_3_try!(
    conditional_entropy_rate_bytes,
    api::try_conditional_entropy_rate_bytes
);
py_metric_bytes_3_try!(
    conditional_entropy_bytes,
    api::try_conditional_entropy_bytes
);
py_metric_bytes_3_try!(mutual_information_bytes, api::try_mutual_information_bytes);
py_metric_bytes_3_try!(
    mutual_information_rate_bytes,
    api::try_mutual_information_rate_bytes
);
py_metric_bytes_3_try!(ned_bytes, api::try_ned_bytes);
py_metric_bytes_3_try!(nte_bytes, api::try_nte_bytes);
py_metric_bytes_3!(tvd_bytes, api::tvd_bytes);
py_metric_bytes_3!(nhd_bytes, api::nhd_bytes);
py_metric_bytes_3_try!(cross_entropy_bytes, api::try_cross_entropy_bytes);
py_metric_bytes_3_try!(cross_entropy_rate_bytes, api::try_cross_entropy_rate_bytes);

py_metric_paths_3_try!(ned_paths, api::try_ned_paths);
py_metric_paths_3_try!(nte_paths, api::try_nte_paths);
py_metric_paths_3_try!(tvd_paths, api::try_tvd_paths);
py_metric_paths_3_try!(nhd_paths, api::try_nhd_paths);
py_metric_paths_3_try!(mutual_information_paths, api::try_mutual_information_paths);
py_metric_paths_3_try!(
    conditional_entropy_paths,
    api::try_conditional_entropy_paths
);
py_metric_paths_3_try!(cross_entropy_paths, api::try_cross_entropy_paths);

#[pyfunction]
fn d_kl_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::d_kl_bytes(x, y)
}

#[pyfunction]
fn js_div_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::js_div_bytes(x, y)
}

#[pyfunction]
fn kl_divergence_paths(py: Python<'_>, x: &str, y: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| api::try_kl_divergence_paths(x, y).map_err(py_infotheory_error)))
}

#[pyfunction]
fn js_divergence_paths(py: Python<'_>, x: &str, y: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| api::try_js_divergence_paths(x, y).map_err(py_infotheory_error)))
}

#[pyfunction]
fn intrinsic_dependence_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| api::try_intrinsic_dependence_bytes(data, max_order).map_err(py_infotheory_error))
    })
}

#[pyfunction]
fn resistance_to_transformation_bytes(
    py: Python<'_>,
    x: &[u8],
    tx: &[u8],
    max_order: i64,
) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| {
            api::try_resistance_to_transformation_bytes(x, tx, max_order)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
fn mutual_information_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::mutual_information_marg_bytes(x, y)
}

#[pyfunction]
fn ned_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::ned_marg_bytes(x, y)
}

#[pyfunction]
fn ned_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| api::try_ned_rate_bytes(x, y, max_order).map_err(py_infotheory_error)))
}

#[pyfunction]
fn ned_cons_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| api::try_ned_cons_bytes(x, y, max_order).map_err(py_infotheory_error)))
}

#[pyfunction]
fn ned_cons_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::ned_cons_marg_bytes(x, y)
}

#[pyfunction]
fn ned_cons_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| api::try_ned_cons_rate_bytes(x, y, max_order).map_err(py_infotheory_error))
    })
}

#[pyfunction]
fn nte_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    api::nte_marg_bytes(x, y)
}

#[pyfunction]
fn nte_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| api::try_nte_rate_bytes(x, y, max_order).map_err(py_infotheory_error)))
}

#[pyfunction]
fn validate_zpaq_rate_method(method: &str) -> PyResult<()> {
    infotheory::validate_zpaq_rate_method(method).map_err(py_infotheory_error)
}

#[pyfunction]
fn get_compressed_size(py: Python<'_>, path: &str, method: &str) -> PyResult<u64> {
    py.detach(|| py_try(|| api::try_get_compressed_size(path, method).map_err(py_infotheory_error)))
}

#[pyfunction]
fn get_compressed_size_parallel(
    py: Python<'_>,
    path: &str,
    method: &str,
    threads: usize,
) -> PyResult<u64> {
    py.detach(|| {
        py_try(|| {
            api::try_get_compressed_size_parallel(path, method, threads)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5"))]
fn get_compressed_sizes_from_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
) -> PyResult<Vec<u64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| {
        py_try(|| {
            api::try_get_compressed_sizes_from_paths(&refs, method).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5"))]
fn get_sequential_compressed_sizes_from_sequential_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
) -> PyResult<Vec<u64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| {
        py_try(|| {
            api::try_get_sequential_compressed_sizes_from_sequential_paths(&refs, method)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5", threads=1))]
fn get_parallel_compressed_sizes_from_sequential_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
    threads: usize,
) -> PyResult<Vec<u64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| {
        py_try(|| {
            api::try_get_parallel_compressed_sizes_from_sequential_paths(&refs, method, threads)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5"))]
fn get_sequential_compressed_sizes_from_parallel_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
) -> PyResult<Vec<u64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| {
        py_try(|| {
            api::try_get_sequential_compressed_sizes_from_parallel_paths(&refs, method)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5", threads=1))]
fn get_parallel_compressed_sizes_from_parallel_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
    threads: usize,
) -> PyResult<Vec<u64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| {
        py_try(|| {
            api::try_get_parallel_compressed_sizes_from_parallel_paths(&refs, method, threads)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
fn get_bytes_from_paths(py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<Vec<u8>>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| py_try(|| api::try_get_bytes_from_paths(&refs).map_err(py_infotheory_error)))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_vitanyi(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| {
            api::try_ncd_paths(x, y, method, NcdVariant::Vitanyi).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_sym_vitanyi(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| {
            api::try_ncd_paths(x, y, method, NcdVariant::SymVitanyi).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_cons(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| api::try_ncd_paths(x, y, method, NcdVariant::Cons).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_sym_cons(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| {
        py_try(|| {
            api::try_ncd_paths(x, y, method, NcdVariant::SymCons).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (data, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn compress_size_backend(
    py: Python<'_>,
    data: &[u8],
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<u64> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = compiled_compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    py.detach(|| {
        py_try(|| {
            infotheory::api::try_compress_size_backend(data, &cb)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    })
}

#[pyfunction]
#[pyo3(signature = (parts, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn compress_size_chain_backend(
    py: Python<'_>,
    parts: Vec<Vec<u8>>,
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<u64> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = compiled_compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    py.detach(|| {
        py_try(|| {
            infotheory::api::try_compress_size_chain_backend(&refs, &cb)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    })
}

#[pyfunction]
#[pyo3(signature = (data, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn compress_bytes_backend<'py>(
    py: Python<'py>,
    data: &[u8],
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = compiled_compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let out = py.detach(|| {
        py_try(|| {
            infotheory::api::try_compress_bytes_backend(data, &cb).map_err(py_infotheory_error)
        })
    })?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (input, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn decompress_bytes_backend<'py>(
    py: Python<'py>,
    input: &[u8],
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = compiled_compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let out = py.detach(|| {
        py_try(|| {
            infotheory::api::try_decompress_bytes_backend(input, &cb).map_err(py_infotheory_error)
        })
    })?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn compress_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<()> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = file_roundtrip_backend(&compression_backend_from_py(
        compression_backend,
        Some(method),
        Some(rb),
    )?);
    let cb = compile_compression_backend(cb)?;
    py.detach(|| {
        py_try(|| {
            let input = std::fs::read(input_path).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to read '{input_path}': {e}"))
            })?;
            let out = infotheory::api::try_compress_bytes_backend(&input, &cb)
                .map_err(py_infotheory_error)?;
            std::fs::write(output_path, &out).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to write '{output_path}': {e}"))
            })?;
            Ok(())
        })
    })
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, compression_backend=None, method="5", rate_backend=None, rate_method=None))]
fn decompress_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: &str,
    rate_backend: Option<&Bound<'_, PyAny>>,
    rate_method: Option<&str>,
) -> PyResult<()> {
    let rb = rate_backend_from_py(rate_backend, rate_method)?;
    let cb = file_roundtrip_backend(&compression_backend_from_py(
        compression_backend,
        Some(method),
        Some(rb),
    )?);
    let cb = compile_compression_backend(cb)?;
    py.detach(|| {
        py_try(|| {
            let input = std::fs::read(input_path).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to read '{input_path}': {e}"))
            })?;
            let out = infotheory::api::try_decompress_bytes_backend(&input, &cb)
                .map_err(py_infotheory_error)?;
            std::fs::write(output_path, &out).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to write '{output_path}': {e}"))
            })?;
            Ok(())
        })
    })
}

#[pyfunction]
#[pyo3(signature = (prompt, bytes, max_order=-1, backend=None, method=None, config=None))]
fn generate_bytes<'py>(
    py: Python<'py>,
    prompt: &[u8],
    bytes: usize,
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let cfg = generation_config_from_py(config)?;
    let out = if let Some(backend) = backend {
        let rb = compiled_rate_backend_from_py(Some(backend), method)?;
        let cb = compile_compression_backend(
            CompressionBackend::try_default().map_err(py_infotheory_error)?,
        )?;
        py.detach(|| {
            py_try(|| {
                let ctx = InfotheoryCtx::new(rb, cb);
                ctx.try_generate_bytes_with_config(prompt, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?
    } else {
        py.detach(|| {
            py_try(|| {
                api::try_generate_bytes_with_config(prompt, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?
    };
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (prefix_parts, bytes, max_order=-1, backend=None, method=None, config=None))]
fn generate_bytes_conditional_chain<'py>(
    py: Python<'py>,
    prefix_parts: Vec<Vec<u8>>,
    bytes: usize,
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let cfg = generation_config_from_py(config)?;
    let refs: Vec<&[u8]> = prefix_parts.iter().map(Vec::as_slice).collect();
    let out = if let Some(backend) = backend {
        let rb = compiled_rate_backend_from_py(Some(backend), method)?;
        let cb = compile_compression_backend(
            CompressionBackend::try_default().map_err(py_infotheory_error)?,
        )?;
        py.detach(|| {
            py_try(|| {
                let ctx = InfotheoryCtx::new(rb, cb);
                ctx.try_generate_bytes_conditional_chain_with_config(&refs, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?
    } else {
        py.detach(|| {
            py_try(|| {
                api::try_generate_bytes_conditional_chain_with_config(&refs, bytes, max_order, cfg)
                    .map_err(py_infotheory_error)
            })
        })?
    };
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (data, max_order, backend=None, method=None))]
fn entropy_rate_backend(
    py: Python<'_>,
    data: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| api::try_entropy_rate_backend(data, max_order, &rb).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (data, max_order, backend=None, method=None))]
fn biased_entropy_rate_backend(
    py: Python<'_>,
    data: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            api::try_biased_entropy_rate_backend(data, max_order, &rb).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (test_data, train_data, max_order, backend=None, method=None))]
fn cross_entropy_rate_backend(
    py: Python<'_>,
    test_data: &[u8],
    train_data: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            api::try_cross_entropy_rate_backend(test_data, train_data, max_order, &rb)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, max_order, backend=None, method=None))]
fn joint_entropy_rate_backend(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            api::try_joint_entropy_rate_backend(x, y, max_order, &rb).map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, max_order, backend=None, method=None))]
fn mutual_information_rate_backend(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            api::try_mutual_information_rate_backend(x, y, max_order, &rb)
                .map_err(py_infotheory_error)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, max_order, backend=None, method=None))]
fn ned_rate_backend(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| api::try_ned_rate_backend(x, y, max_order, &rb).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (x, y, max_order, backend=None, method=None))]
fn nte_rate_backend(
    py: Python<'_>,
    x: &[u8],
    y: &[u8],
    max_order: i64,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<f64> {
    let rb = compiled_rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| api::try_nte_rate_backend(x, y, max_order, &rb).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (paths, method="5", variant="vitanyi"))]
fn ncd_matrix_paths(
    py: Python<'_>,
    paths: Vec<String>,
    method: &str,
    variant: &str,
) -> PyResult<Vec<f64>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    let v = parse_ncd_variant(variant)?;
    py.detach(|| {
        py_try(|| api::try_ncd_matrix_paths(&refs, method, v).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (datas, method="5", variant="vitanyi"))]
fn ncd_matrix_bytes(
    py: Python<'_>,
    datas: Vec<Vec<u8>>,
    method: &str,
    variant: &str,
) -> PyResult<Vec<f64>> {
    let v = parse_ncd_variant(variant)?;
    py.detach(|| {
        py_try(|| api::try_ncd_matrix_bytes(&datas, method, v).map_err(py_infotheory_error))
    })
}

#[pyfunction]
#[pyo3(signature = (datas, backend=None, method=None, variant="vitanyi"))]
fn ncd_matrix_bytes_with_backend(
    py: Python<'_>,
    datas: Vec<Vec<u8>>,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    variant: &str,
) -> PyResult<Vec<f64>> {
    let v = parse_ncd_variant(variant)?;
    let cb = compiled_compression_backend_from_py(backend, method, None)?;
    py.detach(|| {
        py_try(|| {
            let n = datas.len();
            let mut out = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    out[i * n + j] = api::try_ncd_bytes_backend(&datas[i], &datas[j], &cb, v)
                        .map_err(py_infotheory_error)?;
                }
            }
            Ok(out)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (paths, backend=None, method=None, variant="vitanyi"))]
fn ncd_matrix_paths_with_backend(
    py: Python<'_>,
    paths: Vec<String>,
    backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
    variant: &str,
) -> PyResult<Vec<f64>> {
    let datas = get_bytes_from_paths(py, paths)?;
    ncd_matrix_bytes_with_backend(py, datas, backend, method, variant)
}

#[pyfunction]
#[pyo3(signature = (name, method=None))]
fn rate_backend(name: &str, method: Option<&str>) -> PyResult<PyRateBackend> {
    Ok(PyRateBackend {
        inner: parse_rate_backend(name, method)?,
    })
}

#[pyclass(name = "NcdVariant", from_py_object)]
#[derive(Clone)]
struct PyNcdVariant {
    inner: NcdVariant,
}

#[pymethods]
impl PyNcdVariant {
    #[classattr]
    #[pyo3(name = "Vitanyi")]
    fn vitanyi() -> Self {
        Self {
            inner: NcdVariant::Vitanyi,
        }
    }
    #[classattr]
    #[pyo3(name = "SymVitanyi")]
    fn sym_vitanyi() -> Self {
        Self {
            inner: NcdVariant::SymVitanyi,
        }
    }
    #[classattr]
    #[pyo3(name = "Cons")]
    fn cons() -> Self {
        Self {
            inner: NcdVariant::Cons,
        }
    }
    #[classattr]
    #[pyo3(name = "SymCons")]
    fn sym_cons() -> Self {
        Self {
            inner: NcdVariant::SymCons,
        }
    }

    fn __repr__(&self) -> String {
        let n = match self.inner {
            NcdVariant::Vitanyi => "Vitanyi",
            NcdVariant::SymVitanyi => "SymVitanyi",
            NcdVariant::Cons => "Cons",
            NcdVariant::SymCons => "SymCons",
        };
        format!("NcdVariant.{n}")
    }
}

#[pyfunction]
#[pyo3(signature = (mode, observations, observation_bits))]
fn observation_key_from_stream(
    mode: &Bound<'_, PyAny>,
    observations: Vec<u64>,
    observation_bits: usize,
) -> PyResult<u64> {
    Ok(infotheory::aixi::common::observation_key_from_stream(
        parse_observation_key_mode(mode)?,
        &observations,
        observation_bits,
    ))
}

#[pyfunction]
#[pyo3(signature = (mode, observations, observation_bits))]
fn observation_repr_from_stream(
    mode: &Bound<'_, PyAny>,
    observations: Vec<u64>,
    observation_bits: usize,
) -> PyResult<Vec<u64>> {
    Ok(infotheory::aixi::common::observation_repr_from_stream(
        parse_observation_key_mode(mode)?,
        &observations,
        observation_bits,
    ))
}

#[pyfunction]
fn encode_bits(value: u64, bits: usize) -> Vec<bool> {
    let mut out = Vec::new();
    infotheory::aixi::common::encode(&mut out, value, bits);
    out
}

#[pyfunction]
fn decode_bits(symbols: Vec<bool>, bits: usize) -> u64 {
    infotheory::aixi::common::decode(&symbols, bits)
}

#[pyfunction]
fn encode_reward_bits(value: i64, bits: usize) -> Vec<bool> {
    let mut out = Vec::new();
    infotheory::aixi::common::encode_reward(&mut out, value, bits);
    out
}

#[pyfunction]
fn decode_reward_bits(symbols: Vec<bool>, bits: usize) -> i64 {
    infotheory::aixi::common::decode_reward(&symbols, bits)
}

#[pyfunction]
fn encode_reward_offset_bits(value: i64, bits: usize, offset: i64) -> Vec<bool> {
    let mut out = Vec::new();
    infotheory::aixi::common::encode_reward_offset(&mut out, value, bits, offset);
    out
}

#[pyfunction]
fn decode_reward_offset_bits(symbols: Vec<bool>, bits: usize, offset: i64) -> i64 {
    infotheory::aixi::common::decode_reward_offset(&symbols, bits, offset)
}

#[pyclass(name = "RandomGenerator", from_py_object)]
#[derive(Clone, Copy)]
struct PyRandomGenerator {
    inner: infotheory::aixi::common::RandomGenerator,
}

#[pymethods]
impl PyRandomGenerator {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::common::RandomGenerator::new(),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
    fn gen_range(&mut self, end: usize) -> usize {
        self.inner.gen_range(end)
    }
    fn gen_bool(&mut self, p: f64) -> bool {
        self.inner.gen_bool(p)
    }
    fn gen_f64(&mut self) -> f64 {
        self.inner.gen_f64()
    }
    fn fork_with(&self, salt: u64) -> Self {
        Self {
            inner: self.inner.fork_with(salt),
        }
    }
}

struct PyPredictorShim {
    obj: Mutex<Py<PyAny>>,
}

impl PyPredictorShim {
    fn new(obj: Py<PyAny>) -> Self {
        Self {
            obj: Mutex::new(obj),
        }
    }

    fn clone_py_obj(obj: &Py<PyAny>) -> Py<PyAny> {
        Python::attach(|py| {
            let b = obj.bind(py);
            if py_hasattr_or_fatal(
                b,
                "boxed_clone_with_seed",
                "Predictor.boxed_clone_with_seed",
            ) {
                match b.call_method1("boxed_clone_with_seed", (0u64,)) {
                    Ok(v) => return v.unbind(),
                    Err(e) => fatal_python_callback_error(py, "Predictor.boxed_clone_with_seed", e),
                }
            }
            if py_hasattr_or_fatal(b, "boxed_clone", "Predictor.boxed_clone") {
                match b.call_method0("boxed_clone") {
                    Ok(v) => return v.unbind(),
                    Err(e) => fatal_python_callback_error(py, "Predictor.boxed_clone", e),
                }
            }
            match py.import("copy") {
                Ok(copy_mod) => match copy_mod.call_method1("deepcopy", (b,)) {
                    Ok(v) => v.unbind(),
                    Err(e) => fatal_python_callback_error(py, "Predictor.deepcopy", e),
                },
                Err(e) => fatal_python_callback_error(py, "Predictor.deepcopy_import", e),
            }
        })
    }
}

impl infotheory::aixi::model::Predictor for PyPredictorShim {
    fn update(&mut self, sym: bool) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if let Err(e) = guard.bind(py).call_method1("update", (sym,)) {
                fatal_python_callback_error(py, "Predictor.update", e);
            }
        });
    }

    fn update_history(&mut self, sym: bool) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(guard.bind(py), "update_history", "Predictor.update_history") {
                if let Err(e) = guard.bind(py).call_method1("update_history", (sym,)) {
                    fatal_python_callback_error(py, "Predictor.update_history", e);
                }
            } else if let Err(e) = guard.bind(py).call_method1("update", (sym,)) {
                fatal_python_callback_error(py, "Predictor.update", e);
            }
        });
    }

    fn revert(&mut self) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if let Err(e) = guard.bind(py).call_method0("revert") {
                fatal_python_callback_error(py, "Predictor.revert", e);
            }
        });
    }

    fn pop_history(&mut self) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(guard.bind(py), "pop_history", "Predictor.pop_history") {
                if let Err(e) = guard.bind(py).call_method0("pop_history") {
                    fatal_python_callback_error(py, "Predictor.pop_history", e);
                }
            } else if let Err(e) = guard.bind(py).call_method0("revert") {
                fatal_python_callback_error(py, "Predictor.revert", e);
            }
        });
    }

    fn predict_prob(&mut self, sym: bool) -> f64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Predictor.predict_prob",
                guard
                    .bind(py)
                    .call_method1("predict_prob", (sym,))
                    .and_then(|v| v.extract::<f64>()),
            )
        })
    }

    fn model_name(&self) -> String {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Predictor.model_name",
                guard
                    .bind(py)
                    .call_method0("model_name")
                    .and_then(|v| v.extract::<String>()),
            )
        })
    }

    fn boxed_clone(&self) -> Box<dyn infotheory::aixi::model::Predictor> {
        // Keep lock ordering consistent (GIL -> mutex) across all callbacks.
        // Clone a Py handle under lock, then perform callback-driven cloning
        // after releasing the mutex.
        let src = Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            guard.clone_ref(py)
        });
        let cloned = Self::clone_py_obj(&src);
        Box::new(Self::new(cloned))
    }
}

struct PyEnvironmentShim {
    obj: Mutex<Py<PyAny>>,
}

impl PyEnvironmentShim {
    fn new(obj: Py<PyAny>) -> Self {
        Self {
            obj: Mutex::new(obj),
        }
    }

    fn parse_step_result(result: &Bound<'_, PyAny>) -> Option<(Vec<u64>, i64, Option<bool>)> {
        if result.is_none() {
            return None;
        }

        if let Ok((obs_stream, reward, finished)) = result.extract::<(Vec<u64>, i64, bool)>() {
            return Some((obs_stream, reward, Some(finished)));
        }

        if let Ok((obs, reward, finished)) = result.extract::<(u64, i64, bool)>() {
            return Some((vec![obs], reward, Some(finished)));
        }

        if let Ok((obs_stream, reward)) = result.extract::<(Vec<u64>, i64)>() {
            return Some((obs_stream, reward, None));
        }

        if let Ok((obs, reward)) = result.extract::<(u64, i64)>() {
            return Some((vec![obs], reward, None));
        }

        None
    }

    fn perform_action_and_collect(
        &mut self,
        action: u64,
        include_finished: bool,
    ) -> (Vec<u64>, i64, Option<bool>) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            let obj = guard.bind(py);

            let step_result = match obj.call_method1("perform_action", (action,)) {
                Ok(v) => v,
                Err(e) => fatal_python_callback_error(py, "Environment.perform_action", e),
            };

            if let Some((obs_stream, reward, finished_opt)) = Self::parse_step_result(&step_result)
            {
                let finished = if include_finished {
                    match finished_opt {
                        Some(done) => Some(done),
                        None => Some(py_result_or_fatal(
                            py,
                            "Environment.is_finished",
                            obj.call_method0("is_finished")
                                .and_then(|v| v.extract::<bool>()),
                        )),
                    }
                } else {
                    finished_opt
                };

                return (obs_stream, reward, finished);
            }

            let observations =
                if py_hasattr_or_fatal(obj, "drain_observations", "Environment.drain_observations")
                {
                    match obj
                        .call_method0("drain_observations")
                        .and_then(|v| v.extract::<Vec<u64>>())
                    {
                        Ok(v) => v,
                        Err(e) => {
                            fatal_python_callback_error(py, "Environment.drain_observations", e)
                        }
                    }
                } else {
                    vec![py_result_or_fatal(
                        py,
                        "Environment.get_observation",
                        obj.call_method0("get_observation")
                            .and_then(|v| v.extract::<u64>()),
                    )]
                };

            let reward = py_result_or_fatal(
                py,
                "Environment.get_reward",
                obj.call_method0("get_reward")
                    .and_then(|v| v.extract::<i64>()),
            );

            let finished = if include_finished {
                Some(py_result_or_fatal(
                    py,
                    "Environment.is_finished",
                    obj.call_method0("is_finished")
                        .and_then(|v| v.extract::<bool>()),
                ))
            } else {
                None
            };

            (observations, reward, finished)
        })
    }
}

impl infotheory::aixi::environment::Environment for PyEnvironmentShim {
    fn perform_action(&mut self, action: u64) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if let Err(e) = guard.bind(py).call_method1("perform_action", (action,)) {
                fatal_python_callback_error(py, "Environment.perform_action", e);
            }
        });
    }

    fn get_observation(&self) -> u64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.get_observation",
                guard
                    .bind(py)
                    .call_method0("get_observation")
                    .and_then(|v| v.extract::<u64>()),
            )
        })
    }

    fn drain_observations(&mut self) -> Vec<u64> {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "drain_observations",
                "Environment.drain_observations",
            ) {
                match guard
                    .bind(py)
                    .call_method0("drain_observations")
                    .and_then(|v| v.extract::<Vec<u64>>())
                {
                    Ok(v) => v,
                    Err(e) => fatal_python_callback_error(py, "Environment.drain_observations", e),
                }
            } else {
                vec![py_result_or_fatal(
                    py,
                    "Environment.get_observation",
                    guard
                        .bind(py)
                        .call_method0("get_observation")
                        .and_then(|v| v.extract::<u64>()),
                )]
            }
        })
    }

    fn get_reward(&self) -> i64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.get_reward",
                guard
                    .bind(py)
                    .call_method0("get_reward")
                    .and_then(|v| v.extract::<i64>()),
            )
        })
    }

    fn is_finished(&self) -> bool {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.is_finished",
                guard
                    .bind(py)
                    .call_method0("is_finished")
                    .and_then(|v| v.extract::<bool>()),
            )
        })
    }

    fn get_observation_bits(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.get_observation_bits",
                guard
                    .bind(py)
                    .call_method0("get_observation_bits")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn get_reward_bits(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.get_reward_bits",
                guard
                    .bind(py)
                    .call_method0("get_reward_bits")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn get_action_bits(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "Environment.get_action_bits",
                guard
                    .bind(py)
                    .call_method0("get_action_bits")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn set_random_seed(&mut self, seed: u64) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            let obj = guard.bind(py);
            if py_hasattr_or_fatal(obj, "set_random_seed", "Environment.set_random_seed") {
                if let Err(e) = obj.call_method1("set_random_seed", (seed,)) {
                    fatal_python_callback_error(py, "Environment.set_random_seed", e);
                }
            }
        });
    }
}

struct PyAgentSimulatorShim {
    obj: Mutex<Py<PyAny>>,
}

impl PyAgentSimulatorShim {
    fn new(obj: Py<PyAny>) -> Self {
        Self {
            obj: Mutex::new(obj),
        }
    }

    fn parse_key_mode(py_obj: &Bound<'_, PyAny>) -> infotheory::aixi::common::ObservationKeyMode {
        py_result_or_fatal(
            py_obj.py(),
            "AgentSimulator.observation_key_mode",
            parse_observation_key_mode(py_obj),
        )
    }

    fn clone_py_obj(obj: &Py<PyAny>) -> Py<PyAny> {
        Python::attach(|py| {
            let b = obj.bind(py);
            if py_hasattr_or_fatal(
                b,
                "boxed_clone_with_seed",
                "AgentSimulator.boxed_clone_with_seed",
            ) {
                match b.call_method1("boxed_clone_with_seed", (0u64,)) {
                    Ok(v) => return v.unbind(),
                    Err(e) => {
                        fatal_python_callback_error(py, "AgentSimulator.boxed_clone_with_seed", e)
                    }
                }
            }
            if py_hasattr_or_fatal(b, "boxed_clone", "AgentSimulator.boxed_clone") {
                match b.call_method0("boxed_clone") {
                    Ok(v) => return v.unbind(),
                    Err(e) => fatal_python_callback_error(py, "AgentSimulator.boxed_clone", e),
                }
            }
            match py.import("copy") {
                Ok(copy_mod) => match copy_mod.call_method1("deepcopy", (b,)) {
                    Ok(v) => v.unbind(),
                    Err(e) => fatal_python_callback_error(py, "AgentSimulator.deepcopy", e),
                },
                Err(e) => fatal_python_callback_error(py, "AgentSimulator.deepcopy_import", e),
            }
        })
    }
}

impl infotheory::aixi::mcts::AgentSimulator for PyAgentSimulatorShim {
    fn get_num_actions(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.get_num_actions",
                guard
                    .bind(py)
                    .call_method0("get_num_actions")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn get_num_observation_bits(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.get_num_observation_bits",
                guard
                    .bind(py)
                    .call_method0("get_num_observation_bits")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn observation_stream_len(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "observation_stream_len",
                "AgentSimulator.observation_stream_len",
            ) {
                py_result_or_fatal(
                    py,
                    "AgentSimulator.observation_stream_len",
                    guard
                        .bind(py)
                        .call_method0("observation_stream_len")
                        .and_then(|v| v.extract::<usize>()),
                )
            } else {
                1
            }
        })
    }

    fn observation_key_mode(&self) -> infotheory::aixi::common::ObservationKeyMode {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "observation_key_mode",
                "AgentSimulator.observation_key_mode",
            ) {
                match guard.bind(py).call_method0("observation_key_mode") {
                    Ok(v) => Self::parse_key_mode(&v),
                    Err(e) => {
                        fatal_python_callback_error(py, "AgentSimulator.observation_key_mode", e)
                    }
                }
            } else {
                infotheory::aixi::common::ObservationKeyMode::FullStream
            }
        })
    }

    fn get_num_reward_bits(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.get_num_reward_bits",
                guard
                    .bind(py)
                    .call_method0("get_num_reward_bits")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn horizon(&self) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.horizon",
                guard
                    .bind(py)
                    .call_method0("horizon")
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn max_reward(&self) -> i64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.max_reward",
                guard
                    .bind(py)
                    .call_method0("max_reward")
                    .and_then(|v| v.extract::<i64>()),
            )
        })
    }

    fn min_reward(&self) -> i64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.min_reward",
                guard
                    .bind(py)
                    .call_method0("min_reward")
                    .and_then(|v| v.extract::<i64>()),
            )
        })
    }

    fn reward_offset(&self) -> i64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "reward_offset",
                "AgentSimulator.reward_offset",
            ) {
                py_result_or_fatal(
                    py,
                    "AgentSimulator.reward_offset",
                    guard
                        .bind(py)
                        .call_method0("reward_offset")
                        .and_then(|v| v.extract::<i64>()),
                )
            } else {
                0
            }
        })
    }

    fn get_explore_exploit_ratio(&self) -> f64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "get_explore_exploit_ratio",
                "AgentSimulator.get_explore_exploit_ratio",
            ) {
                py_result_or_fatal(
                    py,
                    "AgentSimulator.get_explore_exploit_ratio",
                    guard
                        .bind(py)
                        .call_method0("get_explore_exploit_ratio")
                        .and_then(|v| v.extract::<f64>()),
                )
            } else {
                1.0
            }
        })
    }

    fn discount_gamma(&self) -> f64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if py_hasattr_or_fatal(
                guard.bind(py),
                "discount_gamma",
                "AgentSimulator.discount_gamma",
            ) {
                py_result_or_fatal(
                    py,
                    "AgentSimulator.discount_gamma",
                    guard
                        .bind(py)
                        .call_method0("discount_gamma")
                        .and_then(|v| v.extract::<f64>()),
                )
            } else {
                1.0
            }
        })
    }

    fn model_update_action(&mut self, action: u64) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if let Err(e) = guard
                .bind(py)
                .call_method1("model_update_action", (action,))
            {
                fatal_python_callback_error(py, "AgentSimulator.model_update_action", e);
            }
        });
    }

    fn gen_percept_and_update(&mut self, bits: usize) -> u64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.gen_percept_and_update",
                guard
                    .bind(py)
                    .call_method1("gen_percept_and_update", (bits,))
                    .and_then(|v| v.extract::<u64>()),
            )
        })
    }

    fn model_revert(&mut self, steps: usize) {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            if let Err(e) = guard.bind(py).call_method1("model_revert", (steps,)) {
                fatal_python_callback_error(py, "AgentSimulator.model_revert", e);
            }
        });
    }

    fn gen_range(&mut self, end: usize) -> usize {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.gen_range",
                guard
                    .bind(py)
                    .call_method1("gen_range", (end,))
                    .and_then(|v| v.extract::<usize>()),
            )
        })
    }

    fn gen_f64(&mut self) -> f64 {
        Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            py_result_or_fatal(
                py,
                "AgentSimulator.gen_f64",
                guard
                    .bind(py)
                    .call_method0("gen_f64")
                    .and_then(|v| v.extract::<f64>()),
            )
        })
    }

    fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn infotheory::aixi::mcts::AgentSimulator> {
        enum CloneDecision {
            InvokeSeedClone(Py<PyAny>),
            Fallback(Py<PyAny>),
        }

        let decision = Python::attach(|py| {
            let guard = lock_recover(&self.obj);
            let src = guard.clone_ref(py);
            let b = guard.bind(py);
            let has_seed_clone = py_hasattr_or_fatal(
                b,
                "boxed_clone_with_seed",
                "AgentSimulator.boxed_clone_with_seed",
            );
            if has_seed_clone {
                CloneDecision::InvokeSeedClone(src)
            } else {
                CloneDecision::Fallback(src)
            }
        });

        let cloned = match decision {
            CloneDecision::InvokeSeedClone(src) => Python::attach(|py| {
                match src.bind(py).call_method1("boxed_clone_with_seed", (seed,)) {
                    Ok(v) => v.unbind(),
                    Err(e) => {
                        fatal_python_callback_error(py, "AgentSimulator.boxed_clone_with_seed", e)
                    }
                }
            }),
            CloneDecision::Fallback(src) => Self::clone_py_obj(&src),
        };
        Box::new(Self::new(cloned))
    }
}

#[pyfunction]
#[pyo3(signature = (predictor, steps=8))]
fn predictor_probe(
    py: Python<'_>,
    predictor: Py<PyAny>,
    steps: usize,
) -> PyResult<(Vec<f64>, String)> {
    let (probs, name) = py.detach(|| {
        py_try(|| {
            use infotheory::aixi::model::Predictor;
            let mut p = PyPredictorShim::new(predictor);
            let mut probs = Vec::with_capacity(steps);
            for _ in 0..steps {
                let q = p.predict_one();
                probs.push(q);
                p.update(q >= 0.5);
            }
            p.revert();
            Ok((probs, p.model_name()))
        })
    })?;
    Ok((probs, name))
}

#[pyfunction]
#[pyo3(signature = (environment, actions))]
fn environment_probe(
    py: Python<'_>,
    environment: Py<PyAny>,
    actions: Vec<u64>,
) -> PyResult<Vec<(u64, i64, bool)>> {
    py.detach(|| {
        py_try(|| {
            let mut env = PyEnvironmentShim::new(environment);
            let mut out = Vec::with_capacity(actions.len());
            for a in actions {
                let (obs_stream, reward, finished_opt) = env.perform_action_and_collect(a, true);
                let observation = obs_stream.first().copied().unwrap_or(0);
                out.push((observation, reward, finished_opt.unwrap_or(false)));
            }
            Ok(out)
        })
    })
}

fn validate_observation_stream_len(expected: usize, actual: usize) -> PyResult<()> {
    if expected != actual {
        return Err(PyValueError::new_err(format!(
            "observation stream length mismatch: AgentConfig expects {expected} symbols, but environment returned {actual}; update AgentConfig.observation_stream_len or environment.drain_observations"
        )));
    }
    Ok(())
}

struct AixiRunSummary {
    learn_total_reward: i64,
    eval_total_reward: i64,
    eval_average_reward: f64,
    learn_cycles_completed: usize,
    eval_cycles_completed: usize,
    learn_elapsed_seconds: f64,
    eval_elapsed_seconds: f64,
    learn_cycles_per_second: f64,
    eval_cycles_per_second: f64,
    last_action: u64,
    last_reward: i64,
    last_observation_stream: Vec<u64>,
}

#[pyfunction]
#[pyo3(signature = (
    environment,
    config,
    learn_cycles=None,
    eval_cycles=None,
    terminate_lifetime=20,
    explore_epsilon=0.0,
    explore_gamma=1.0,
    prev_action=0,
    check_finished=false
))]
fn run_agent_with_environment<'py>(
    py: Python<'py>,
    environment: Py<PyAny>,
    config: &PyAgentConfig,
    learn_cycles: Option<usize>,
    eval_cycles: Option<usize>,
    terminate_lifetime: usize,
    explore_epsilon: f64,
    explore_gamma: f64,
    prev_action: u64,
    check_finished: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if explore_epsilon < 0.0 {
        return Err(PyValueError::new_err(
            "explore_epsilon must be >= 0.0 for run_agent_with_environment",
        ));
    }
    if !(0.0..=1.0).contains(&explore_gamma) {
        return Err(PyValueError::new_err(
            "explore_gamma must be in [0, 1] for run_agent_with_environment",
        ));
    }
    if config.inner.agent_actions == 0 {
        return Err(PyValueError::new_err(
            "AgentConfig.agent_actions must be >= 1 for run_agent_with_environment",
        ));
    }
    config.inner.validate().map_err(PyValueError::new_err)?;

    let summary = py.detach(|| {
        py_try(|| {
            use infotheory::aixi::agent::Agent;
            use infotheory::aixi::common::RandomGenerator;
            use infotheory::aixi::environment::Environment;

            let mut env = PyEnvironmentShim::new(environment);
            if let Some(seed) = config.inner.random_seed {
                env.set_random_seed(seed);
            }
            let mut agent = Agent::try_new(config.inner.clone()).map_err(PyValueError::new_err)?;

            let observation_stream_len = config.inner.observation_stream_len.max(1);
            let (learn_cycles, eval_cycles) = match (learn_cycles, eval_cycles) {
                (Some(learn), Some(eval)) => (learn, eval),
                (Some(learn), None) => (learn, 0usize),
                (None, Some(eval)) => (terminate_lifetime, eval),
                (None, None) => (terminate_lifetime, 0usize),
            };

            let mut learn_total_reward: i64 = 0;
            let mut eval_total_reward: i64 = 0;
            let mut prev_action = prev_action;
            let mut obs_stream = env.drain_observations();
            validate_observation_stream_len(observation_stream_len, obs_stream.len())?;
            let mut reward = env.get_reward();
            let mut explore_rng = if let Some(seed) = config.inner.random_seed {
                RandomGenerator::from_seed(seed).fork_with(0x4558504c4f52455f)
            } else {
                RandomGenerator::new()
            };

            let learn_start = Instant::now();
            let mut learn_cycles_completed = 0usize;
            for t in 0..learn_cycles {
                agent.model_update_percept_stream(&obs_stream, reward);

                let explore_p = if explore_epsilon > 0.0 {
                    explore_epsilon * explore_gamma.powi(t as i32)
                } else {
                    0.0
                };

                let action = if explore_p > 0.0 && explore_rng.gen_bool(explore_p.min(1.0)) {
                    explore_rng.gen_range(config.inner.agent_actions) as u64
                } else {
                    agent.get_planned_action(&obs_stream, reward, prev_action)
                };

                agent.model_update_action_external(action);
                let (next_obs_stream, next_reward, finished_opt) =
                    env.perform_action_and_collect(action, check_finished);
                validate_observation_stream_len(observation_stream_len, next_obs_stream.len())?;

                obs_stream = next_obs_stream;
                reward = next_reward;
                prev_action = action;
                learn_total_reward += reward;
                learn_cycles_completed += 1;

                if check_finished && finished_opt.unwrap_or(false) {
                    break;
                }
            }
            let learn_elapsed_seconds = learn_start.elapsed().as_secs_f64();

            let eval_start = Instant::now();
            let mut eval_cycles_completed = 0usize;
            for _ in 0..eval_cycles {
                agent.model_update_percept_stream(&obs_stream, reward);

                let action = agent.get_planned_action(&obs_stream, reward, prev_action);
                agent.model_update_action_external(action);

                let (next_obs_stream, next_reward, finished_opt) =
                    env.perform_action_and_collect(action, check_finished);
                validate_observation_stream_len(observation_stream_len, next_obs_stream.len())?;

                obs_stream = next_obs_stream;
                reward = next_reward;
                prev_action = action;
                eval_total_reward += reward;
                eval_cycles_completed += 1;

                if check_finished && finished_opt.unwrap_or(false) {
                    break;
                }
            }
            let eval_elapsed_seconds = eval_start.elapsed().as_secs_f64();

            let eval_average_reward = if eval_cycles_completed > 0 {
                eval_total_reward as f64 / eval_cycles_completed as f64
            } else {
                0.0
            };

            let learn_cycles_per_second = if learn_elapsed_seconds > 0.0 {
                learn_cycles_completed as f64 / learn_elapsed_seconds
            } else {
                0.0
            };

            let eval_cycles_per_second = if eval_elapsed_seconds > 0.0 {
                eval_cycles_completed as f64 / eval_elapsed_seconds
            } else {
                0.0
            };

            Ok(AixiRunSummary {
                learn_total_reward,
                eval_total_reward,
                eval_average_reward,
                learn_cycles_completed,
                eval_cycles_completed,
                learn_elapsed_seconds,
                eval_elapsed_seconds,
                learn_cycles_per_second,
                eval_cycles_per_second,
                last_action: prev_action,
                last_reward: reward,
                last_observation_stream: obs_stream,
            })
        })
    })?;

    let out = PyDict::new(py);
    out.set_item("learn_total_reward", summary.learn_total_reward)?;
    out.set_item("eval_total_reward", summary.eval_total_reward)?;
    out.set_item("eval_average_reward", summary.eval_average_reward)?;
    out.set_item("learn_cycles_completed", summary.learn_cycles_completed)?;
    out.set_item("eval_cycles_completed", summary.eval_cycles_completed)?;
    out.set_item("learn_elapsed_seconds", summary.learn_elapsed_seconds)?;
    out.set_item("eval_elapsed_seconds", summary.eval_elapsed_seconds)?;
    out.set_item("learn_cycles_per_second", summary.learn_cycles_per_second)?;
    out.set_item("eval_cycles_per_second", summary.eval_cycles_per_second)?;
    out.set_item("last_action", summary.last_action)?;
    out.set_item("last_reward", summary.last_reward)?;
    out.set_item("last_observation_stream", summary.last_observation_stream)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    environment,
    config,
    learn_cycles=None,
    eval_cycles=None,
    terminate_lifetime=20,
    explore_epsilon=0.0,
    explore_gamma=1.0,
    check_finished=false
))]
fn run_aiqi_with_environment<'py>(
    py: Python<'py>,
    environment: Py<PyAny>,
    config: &PyAiqiConfig,
    learn_cycles: Option<usize>,
    eval_cycles: Option<usize>,
    terminate_lifetime: usize,
    explore_epsilon: f64,
    explore_gamma: f64,
    check_finished: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if explore_epsilon < 0.0 {
        return Err(PyValueError::new_err(
            "explore_epsilon must be >= 0.0 for run_aiqi_with_environment",
        ));
    }
    if !(0.0..=1.0).contains(&explore_gamma) {
        return Err(PyValueError::new_err(
            "explore_gamma must be in [0, 1] for run_aiqi_with_environment",
        ));
    }

    let summary = py.detach(|| {
        py_try(|| {
            use infotheory::aixi::aiqi::AiqiAgent;
            use infotheory::aixi::environment::Environment;

            let mut env = PyEnvironmentShim::new(environment);
            if let Some(seed) = config.inner.random_seed {
                env.set_random_seed(seed);
            }
            let mut agent = AiqiAgent::new(config.inner.clone()).map_err(PyValueError::new_err)?;

            let observation_stream_len = config.inner.observation_stream_len.max(1);
            let (learn_cycles, eval_cycles) = match (learn_cycles, eval_cycles) {
                (Some(learn), Some(eval)) => (learn, eval),
                (Some(learn), None) => (learn, 0usize),
                (None, Some(eval)) => (terminate_lifetime, eval),
                (None, None) => (terminate_lifetime, 0usize),
            };

            let mut learn_total_reward: i64 = 0;
            let mut eval_total_reward: i64 = 0;
            let mut last_action = 0u64;

            let mut obs_stream = env.drain_observations();
            validate_observation_stream_len(observation_stream_len, obs_stream.len())?;
            let mut reward = env.get_reward();

            let learn_start = Instant::now();
            let mut learn_cycles_completed = 0usize;
            for t in 0..learn_cycles {
                let extra_explore_p = if explore_epsilon > 0.0 {
                    (explore_epsilon * explore_gamma.powi(t as i32)).min(1.0)
                } else {
                    0.0
                };
                let action = agent.get_planned_action_with_extra_exploration(extra_explore_p);

                let (next_obs_stream, next_reward, finished_opt) =
                    env.perform_action_and_collect(action, check_finished);
                validate_observation_stream_len(observation_stream_len, next_obs_stream.len())?;

                agent
                    .observe_transition(action, &next_obs_stream, next_reward)
                    .map_err(PyValueError::new_err)?;

                obs_stream = next_obs_stream;
                reward = next_reward;
                last_action = action;
                learn_total_reward += reward;
                learn_cycles_completed += 1;

                if check_finished && finished_opt.unwrap_or(false) {
                    break;
                }
            }
            let learn_elapsed_seconds = learn_start.elapsed().as_secs_f64();

            let eval_start = Instant::now();
            let mut eval_cycles_completed = 0usize;
            for _ in 0..eval_cycles {
                let action = agent.get_planned_action();

                let (next_obs_stream, next_reward, finished_opt) =
                    env.perform_action_and_collect(action, check_finished);
                validate_observation_stream_len(observation_stream_len, next_obs_stream.len())?;

                agent
                    .observe_transition(action, &next_obs_stream, next_reward)
                    .map_err(PyValueError::new_err)?;

                obs_stream = next_obs_stream;
                reward = next_reward;
                last_action = action;
                eval_total_reward += reward;
                eval_cycles_completed += 1;

                if check_finished && finished_opt.unwrap_or(false) {
                    break;
                }
            }
            let eval_elapsed_seconds = eval_start.elapsed().as_secs_f64();

            let eval_average_reward = if eval_cycles_completed > 0 {
                eval_total_reward as f64 / eval_cycles_completed as f64
            } else {
                0.0
            };

            let learn_cycles_per_second = if learn_elapsed_seconds > 0.0 {
                learn_cycles_completed as f64 / learn_elapsed_seconds
            } else {
                0.0
            };

            let eval_cycles_per_second = if eval_elapsed_seconds > 0.0 {
                eval_cycles_completed as f64 / eval_elapsed_seconds
            } else {
                0.0
            };

            Ok(AixiRunSummary {
                learn_total_reward,
                eval_total_reward,
                eval_average_reward,
                learn_cycles_completed,
                eval_cycles_completed,
                learn_elapsed_seconds,
                eval_elapsed_seconds,
                learn_cycles_per_second,
                eval_cycles_per_second,
                last_action,
                last_reward: reward,
                last_observation_stream: obs_stream,
            })
        })
    })?;

    let out = PyDict::new(py);
    out.set_item("learn_total_reward", summary.learn_total_reward)?;
    out.set_item("eval_total_reward", summary.eval_total_reward)?;
    out.set_item("eval_average_reward", summary.eval_average_reward)?;
    out.set_item("learn_cycles_completed", summary.learn_cycles_completed)?;
    out.set_item("eval_cycles_completed", summary.eval_cycles_completed)?;
    out.set_item("learn_elapsed_seconds", summary.learn_elapsed_seconds)?;
    out.set_item("eval_elapsed_seconds", summary.eval_elapsed_seconds)?;
    out.set_item("learn_cycles_per_second", summary.learn_cycles_per_second)?;
    out.set_item("eval_cycles_per_second", summary.eval_cycles_per_second)?;
    out.set_item("last_action", summary.last_action)?;
    out.set_item("last_reward", summary.last_reward)?;
    out.set_item("last_observation_stream", summary.last_observation_stream)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (simulator, prev_obs_stream, prev_rew, prev_act, num_simulations))]
fn search_with_simulator(
    py: Python<'_>,
    simulator: Py<PyAny>,
    prev_obs_stream: Vec<u64>,
    prev_rew: i64,
    prev_act: u64,
    num_simulations: usize,
) -> PyResult<u64> {
    py.detach(|| {
        py_try(|| {
            let mut sim = PyAgentSimulatorShim::new(simulator);
            let mut tree = infotheory::aixi::mcts::SearchTree::new();
            Ok(tree.search(
                &mut sim,
                &prev_obs_stream,
                prev_rew,
                prev_act,
                num_simulations,
            ))
        })
    })
}

#[pyclass(name = "ObservationKeyMode", from_py_object)]
#[derive(Clone)]
struct PyObservationKeyMode {
    inner: infotheory::aixi::common::ObservationKeyMode,
}

#[pymethods]
impl PyObservationKeyMode {
    #[classattr]
    #[pyo3(name = "First")]
    fn first() -> Self {
        Self {
            inner: infotheory::aixi::common::ObservationKeyMode::First,
        }
    }
    #[classattr]
    #[pyo3(name = "Last")]
    fn last() -> Self {
        Self {
            inner: infotheory::aixi::common::ObservationKeyMode::Last,
        }
    }
    #[classattr]
    #[pyo3(name = "StreamHash")]
    fn stream_hash() -> Self {
        Self {
            inner: infotheory::aixi::common::ObservationKeyMode::StreamHash,
        }
    }
    #[classattr]
    #[pyo3(name = "FullStream")]
    fn full_stream() -> Self {
        Self {
            inner: infotheory::aixi::common::ObservationKeyMode::FullStream,
        }
    }
}

#[pyclass(name = "AgentConfig", from_py_object)]
#[derive(Clone)]
struct PyAgentConfig {
    inner: infotheory::aixi::agent::AgentConfig,
}

#[pyclass(name = "AiqiConfig", from_py_object)]
#[derive(Clone)]
struct PyAiqiConfig {
    inner: infotheory::aixi::aiqi::AiqiConfig,
}

#[pymethods]
impl PyAgentConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        algorithm="fac-ctw".to_string(),
        ct_depth=16,
        agent_horizon=6,
        observation_bits=8,
        observation_stream_len=1,
        observation_key_mode=None,
        reward_bits=8,
        agent_actions=2,
        num_simulations=256,
        exploration_exploitation_ratio=1.41,
        discount_gamma=1.0,
        min_reward=-128,
        max_reward=127,
        reward_offset=128,
        random_seed=None,
        rate_backend=None,
        rate_backend_max_order=20,
        rwkv_model_path=None,
        rosa_max_order=None,
        zpaq_method=None
    ))]
    fn new(
        algorithm: String,
        ct_depth: usize,
        agent_horizon: usize,
        observation_bits: usize,
        observation_stream_len: usize,
        observation_key_mode: Option<&PyObservationKeyMode>,
        reward_bits: usize,
        agent_actions: usize,
        num_simulations: usize,
        exploration_exploitation_ratio: f64,
        discount_gamma: f64,
        min_reward: i64,
        max_reward: i64,
        reward_offset: i64,
        random_seed: Option<u64>,
        rate_backend: Option<&PyRateBackend>,
        rate_backend_max_order: i64,
        rwkv_model_path: Option<String>,
        rosa_max_order: Option<i64>,
        zpaq_method: Option<String>,
    ) -> PyResult<Self> {
        let inner = infotheory::aixi::agent::AgentConfig {
            algorithm,
            ct_depth,
            agent_horizon,
            observation_bits,
            observation_stream_len,
            observation_key_mode: observation_key_mode
                .map(|m| m.inner)
                .unwrap_or(infotheory::aixi::common::ObservationKeyMode::FullStream),
            reward_bits,
            agent_actions,
            num_simulations,
            exploration_exploitation_ratio,
            discount_gamma,
            min_reward,
            max_reward,
            reward_offset,
            random_seed,
            rate_backend: rate_backend.map(|rb| rb.inner.clone()),
            rate_backend_max_order,
            rwkv_model_path,
            rwkv_method: None,
            mamba_model_path: None,
            mamba_method: None,
            rosa_max_order,
            zpaq_method,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyAiqiConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        algorithm="ac-ctw".to_string(),
        ct_depth=16,
        observation_bits=8,
        observation_stream_len=1,
        reward_bits=8,
        agent_actions=2,
        min_reward=-128,
        max_reward=127,
        reward_offset=128,
        discount_gamma=0.99,
        return_horizon=6,
        return_bins=32,
        augmentation_period=None,
        history_prune_keep_steps=None,
        baseline_exploration=0.01,
        random_seed=None,
        rate_backend=None,
        rate_backend_max_order=20,
        rwkv_model_path=None,
        rosa_max_order=None,
        zpaq_method=None
    ))]
    fn new(
        algorithm: String,
        ct_depth: usize,
        observation_bits: usize,
        observation_stream_len: usize,
        reward_bits: usize,
        agent_actions: usize,
        min_reward: i64,
        max_reward: i64,
        reward_offset: i64,
        discount_gamma: f64,
        return_horizon: usize,
        return_bins: usize,
        augmentation_period: Option<usize>,
        history_prune_keep_steps: Option<usize>,
        baseline_exploration: f64,
        random_seed: Option<u64>,
        rate_backend: Option<&PyRateBackend>,
        rate_backend_max_order: i64,
        rwkv_model_path: Option<String>,
        rosa_max_order: Option<i64>,
        zpaq_method: Option<String>,
    ) -> PyResult<Self> {
        let inner = infotheory::aixi::aiqi::AiqiConfig {
            algorithm,
            ct_depth,
            observation_bits,
            observation_stream_len,
            reward_bits,
            agent_actions,
            min_reward,
            max_reward,
            reward_offset,
            discount_gamma,
            return_horizon,
            return_bins,
            augmentation_period: augmentation_period.unwrap_or(return_horizon),
            history_prune_keep_steps,
            baseline_exploration,
            random_seed,
            rate_backend: rate_backend.map(|rb| rb.inner.clone()),
            rate_backend_max_order,
            rwkv_model_path,
            rosa_max_order,
            zpaq_method,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "Agent", unsendable)]
struct PyAgent {
    inner: infotheory::aixi::agent::Agent,
}

#[pymethods]
impl PyAgent {
    #[new]
    fn new(config: &PyAgentConfig) -> PyResult<Self> {
        py_try(|| {
            let inner = infotheory::aixi::agent::Agent::try_new(config.inner.clone())
                .map_err(PyValueError::new_err)?;
            Ok(Self { inner })
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_planned_action(
        &mut self,
        prev_obs_stream: Vec<u64>,
        prev_rew: i64,
        prev_act: u64,
    ) -> u64 {
        self.inner
            .get_planned_action(&prev_obs_stream, prev_rew, prev_act)
    }

    fn model_update_percept(&mut self, observation: u64, reward: i64) {
        self.inner.model_update_percept(observation, reward)
    }

    fn model_update_percept_stream(&mut self, observations: Vec<u64>, reward: i64) {
        self.inner
            .model_update_percept_stream(&observations, reward)
    }

    fn observation_repr_from_stream(&self, observations: Vec<u64>) -> Vec<u64> {
        self.inner.observation_repr_from_stream(&observations)
    }

    fn model_update_action_external(&mut self, action: u64) {
        self.inner.model_update_action_external(action)
    }
}

#[pyclass(name = "AiqiAgent", unsendable)]
struct PyAiqiAgent {
    inner: infotheory::aixi::aiqi::AiqiAgent,
}

#[pymethods]
impl PyAiqiAgent {
    #[new]
    fn new(config: &PyAiqiConfig) -> PyResult<Self> {
        py_try(|| {
            let inner = infotheory::aixi::aiqi::AiqiAgent::new(config.inner.clone())
                .map_err(PyValueError::new_err)?;
            Ok(Self { inner })
        })
    }

    fn steps_observed(&self) -> usize {
        self.inner.steps_observed()
    }

    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    fn get_planned_action(&mut self) -> u64 {
        self.inner.get_planned_action()
    }

    #[pyo3(signature = (extra_exploration=0.0))]
    fn get_planned_action_with_extra_exploration(&mut self, extra_exploration: f64) -> u64 {
        self.inner
            .get_planned_action_with_extra_exploration(extra_exploration)
    }

    fn observe_transition(
        &mut self,
        action: u64,
        observations: Vec<u64>,
        reward: i64,
    ) -> PyResult<()> {
        self.inner
            .observe_transition(action, &observations, reward)
            .map_err(PyValueError::new_err)
    }
}

#[cfg(feature = "backend-ctw")]
#[pyclass(name = "CtwPredictor")]
struct PyCtwPredictor {
    inner: infotheory::aixi::model::CtwPredictor,
}

#[cfg(feature = "backend-ctw")]
#[pymethods]
impl PyCtwPredictor {
    #[new]
    fn new(depth: usize) -> Self {
        Self {
            inner: infotheory::aixi::model::CtwPredictor::new(depth),
        }
    }
    fn update(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update_history(sym);
    }
    fn revert(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.revert();
    }
    fn pop_history(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.pop_history();
    }
    fn predict_prob(&mut self, sym: bool) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_prob(sym)
    }
    fn predict_one(&mut self) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_one()
    }
    fn model_name(&self) -> String {
        use infotheory::aixi::model::Predictor;
        self.inner.model_name()
    }
}

#[cfg(feature = "backend-ctw")]
#[pyclass(name = "FacCtwPredictor")]
struct PyFacCtwPredictor {
    inner: infotheory::aixi::model::FacCtwPredictor,
}

#[cfg(feature = "backend-ctw")]
#[pymethods]
impl PyFacCtwPredictor {
    #[new]
    fn new(base_depth: usize, num_percept_bits: usize) -> Self {
        Self {
            inner: infotheory::aixi::model::FacCtwPredictor::new(base_depth, num_percept_bits),
        }
    }
    fn update(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update_history(sym);
    }
    fn revert(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.revert();
    }
    fn pop_history(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.pop_history();
    }
    fn predict_prob(&mut self, sym: bool) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_prob(sym)
    }
    fn predict_one(&mut self) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_one()
    }
    fn model_name(&self) -> String {
        use infotheory::aixi::model::Predictor;
        self.inner.model_name()
    }
}

#[cfg(feature = "backend-rosa")]
#[pyclass(name = "RosaPredictor")]
struct PyRosaPredictor {
    inner: infotheory::aixi::model::RosaPredictor,
}

#[cfg(feature = "backend-rosa")]
#[pymethods]
impl PyRosaPredictor {
    #[new]
    #[pyo3(signature = (max_order=20))]
    fn new(max_order: i64) -> Self {
        Self {
            inner: infotheory::aixi::model::RosaPredictor::new(max_order),
        }
    }
    fn update(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update_history(sym);
    }
    fn revert(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.revert();
    }
    fn pop_history(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.pop_history();
    }
    fn predict_prob(&mut self, sym: bool) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_prob(sym)
    }
    fn predict_one(&mut self) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_one()
    }
    fn model_name(&self) -> String {
        use infotheory::aixi::model::Predictor;
        self.inner.model_name()
    }
}

#[cfg(feature = "backend-zpaq")]
#[pyclass(name = "ZpaqPredictor", unsendable)]
struct PyZpaqPredictor {
    inner: infotheory::aixi::model::ZpaqPredictor,
}

#[cfg(feature = "backend-zpaq")]
#[pymethods]
impl PyZpaqPredictor {
    #[new]
    #[pyo3(signature = (method=None, min_prob=5.960_464_477_539_063e-8))]
    fn new(method: Option<String>, min_prob: f64) -> Self {
        Self {
            inner: infotheory::aixi::model::ZpaqPredictor::new(
                method.unwrap_or_else(|| "1".to_string()),
                min_prob,
            ),
        }
    }
    fn update(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update_history(sym);
    }
    fn revert(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.revert();
    }
    fn pop_history(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.pop_history();
    }
    fn predict_prob(&mut self, sym: bool) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_prob(sym)
    }
    fn predict_one(&mut self) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_one()
    }
    fn model_name(&self) -> String {
        use infotheory::aixi::model::Predictor;
        self.inner.model_name()
    }
}

#[cfg(feature = "backend-rwkv")]
#[pyclass(name = "RwkvPredictor")]
struct PyRwkvPredictor {
    inner: infotheory::aixi::model::RwkvPredictor,
}

#[cfg(feature = "backend-rwkv")]
#[pymethods]
impl PyRwkvPredictor {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let model = infotheory::rwkvzip::Compressor::load_model(&model_path).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to load RWKV7 model: {err:#}"))
        })?;
        Ok(Self {
            inner: infotheory::aixi::model::RwkvPredictor::new(model),
        })
    }
    fn update(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update(sym);
    }
    fn update_history(&mut self, sym: bool) {
        use infotheory::aixi::model::Predictor;
        self.inner.update_history(sym);
    }
    fn revert(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.revert();
    }
    fn pop_history(&mut self) {
        use infotheory::aixi::model::Predictor;
        self.inner.pop_history();
    }
    fn predict_prob(&mut self, sym: bool) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_prob(sym)
    }
    fn predict_one(&mut self) -> f64 {
        use infotheory::aixi::model::Predictor;
        self.inner.predict_one()
    }
    fn model_name(&self) -> String {
        use infotheory::aixi::model::Predictor;
        self.inner.model_name()
    }
}

#[pyclass(name = "SearchNode")]
struct PySearchNode {
    inner: infotheory::aixi::mcts::SearchNode,
}

#[pymethods]
impl PySearchNode {
    #[new]
    #[pyo3(signature = (is_chance_node=false))]
    fn new(is_chance_node: bool) -> Self {
        Self {
            inner: infotheory::aixi::mcts::SearchNode::new(is_chance_node),
        }
    }

    fn best_action(&self, agent: &mut PyAgent) -> u64 {
        self.inner.best_action(&mut agent.inner)
    }
}

#[pyclass(name = "SearchTree")]
struct PySearchTree {
    inner: infotheory::aixi::mcts::SearchTree,
}

#[pymethods]
impl PySearchTree {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::mcts::SearchTree::new(),
        }
    }

    fn search(
        &mut self,
        agent: &mut PyAgent,
        prev_obs_stream: Vec<u64>,
        prev_rew: i64,
        prev_act: u64,
        num_simulations: usize,
    ) -> u64 {
        self.inner.search(
            &mut agent.inner,
            &prev_obs_stream,
            prev_rew,
            prev_act,
            num_simulations,
        )
    }
}

#[cfg(feature = "aixi-gameengine")]
fn new_gameengine_builtin(
    builtin: infotheory::spec::BuiltinEnvironmentSpec,
    random_seed: Option<u64>,
) -> PyResult<Box<dyn infotheory::aixi::environment::Environment>> {
    let mut env = infotheory::aixi::gameengine::build_builtin_environment(builtin)
        .map_err(PyRuntimeError::new_err)?;
    if let Some(seed) = random_seed {
        env.set_random_seed(seed);
    }
    Ok(env)
}

#[cfg(feature = "aixi-gameengine")]
fn coin_flip_probability_parts(p: f64) -> PyResult<(u64, u64)> {
    const DENOMINATOR: u64 = 1_000_000;
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(
            "CoinFlipEnv p must be a finite probability in [0.0, 1.0]",
        ));
    }
    Ok(((p * DENOMINATOR as f64).round() as u64, DENOMINATOR))
}

#[cfg(feature = "aixi-gameengine")]
macro_rules! define_gameengine_env_class {
    ($(#[$cfg:meta])* $name:ident, $py_name:literal, $builtin:expr) => {
        $(#[$cfg])*
        #[pyclass(name = $py_name, unsendable)]
        struct $name {
            inner: Box<dyn infotheory::aixi::environment::Environment>,
        }

        $(#[$cfg])*
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (random_seed=None))]
            fn new(random_seed: Option<u64>) -> PyResult<Self> {
                Ok(Self {
                    inner: new_gameengine_builtin($builtin, random_seed)?,
                })
            }

            fn set_random_seed(&mut self, seed: u64) {
                self.inner.set_random_seed(seed);
            }

            fn get_observation_bits(&self) -> usize {
                self.inner.get_observation_bits()
            }

            fn get_reward_bits(&self) -> usize {
                self.inner.get_reward_bits()
            }

            fn get_action_bits(&self) -> usize {
                self.inner.get_action_bits()
            }

            fn get_num_actions(&self) -> usize {
                self.inner.get_num_actions()
            }

            fn min_reward(&self) -> i64 {
                self.inner.min_reward()
            }

            fn max_reward(&self) -> i64 {
                self.inner.max_reward()
            }

            fn perform_action(&mut self, action: u64) {
                self.inner.perform_action(action);
            }

            fn get_observation(&self) -> u64 {
                self.inner.get_observation()
            }

            fn get_reward(&self) -> i64 {
                self.inner.get_reward()
            }

            fn is_finished(&self) -> bool {
                self.inner.is_finished()
            }

            fn drain_observations(&mut self) -> Vec<u64> {
                self.inner.drain_observations()
            }
        }
    };
}

#[cfg(feature = "aixi-gameengine")]
#[pyclass(name = "CoinFlipEnv", unsendable)]
struct CoinFlipEnv {
    inner: Box<dyn infotheory::aixi::environment::Environment>,
}

#[cfg(feature = "aixi-gameengine")]
#[pymethods]
impl CoinFlipEnv {
    #[new]
    #[pyo3(signature = (p=0.7, random_seed=None))]
    fn new(p: f64, random_seed: Option<u64>) -> PyResult<Self> {
        let seed = random_seed.unwrap_or(0);
        let (head_numerator, head_denominator) = coin_flip_probability_parts(p)?;
        Ok(Self {
            inner: infotheory::aixi::gameengine::build_coin_flip_environment(
                head_numerator,
                head_denominator,
                seed,
            )
            .map_err(PyRuntimeError::new_err)?,
        })
    }

    fn set_random_seed(&mut self, seed: u64) {
        self.inner.set_random_seed(seed);
    }

    fn get_observation_bits(&self) -> usize {
        self.inner.get_observation_bits()
    }

    fn get_reward_bits(&self) -> usize {
        self.inner.get_reward_bits()
    }

    fn get_action_bits(&self) -> usize {
        self.inner.get_action_bits()
    }

    fn get_num_actions(&self) -> usize {
        self.inner.get_num_actions()
    }

    fn min_reward(&self) -> i64 {
        self.inner.min_reward()
    }

    fn max_reward(&self) -> i64 {
        self.inner.max_reward()
    }

    fn perform_action(&mut self, action: u64) {
        self.inner.perform_action(action);
    }

    fn get_observation(&self) -> u64 {
        self.inner.get_observation()
    }

    fn get_reward(&self) -> i64 {
        self.inner.get_reward()
    }

    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    fn drain_observations(&mut self) -> Vec<u64> {
        self.inner.drain_observations()
    }
}

define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine")]
    BiasedRockPaperScissorEnv,
    "BiasedRockPaperScissorEnv",
    infotheory::spec::BuiltinEnvironmentSpec::BiasedRockPaperScissor
);
define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine")]
    KuhnPokerEnv,
    "KuhnPokerEnv",
    infotheory::spec::BuiltinEnvironmentSpec::KuhnPoker
);
define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine")]
    ExtendedTigerEnv,
    "ExtendedTigerEnv",
    infotheory::spec::BuiltinEnvironmentSpec::ExtendedTiger
);
define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine")]
    TicTacToeEnv,
    "TicTacToeEnv",
    infotheory::spec::BuiltinEnvironmentSpec::TicTacToe
);
define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine")]
    BlackjackEnv,
    "BlackjackEnv",
    infotheory::spec::BuiltinEnvironmentSpec::Blackjack
);
define_gameengine_env_class!(
    #[cfg(feature = "aixi-gameengine-physics")]
    PlatformerEnv,
    "PlatformerEnv",
    infotheory::spec::BuiltinEnvironmentSpec::Platformer
);

#[pyfunction]
fn vm_enabled() -> bool {
    cfg!(feature = "vm")
}

#[cfg(feature = "vm")]
#[pyclass(name = "NyxVmConfig", from_py_object)]
#[derive(Clone)]
struct PyNyxVmConfig {
    inner: infotheory::aixi::vm_nyx::NyxVmConfig,
}

#[cfg(feature = "vm")]
#[pymethods]
impl PyNyxVmConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::vm_nyx::NyxVmConfig::default(),
        }
    }

    fn set_firecracker_config(&mut self, path: String) {
        self.inner.firecracker_config = path;
    }

    fn set_instance_id(&mut self, id: String) {
        self.inner.instance_id = id;
    }

    fn set_episode_steps(&mut self, steps: usize) {
        self.inner.episode_steps = steps;
    }

    fn set_step_cost(&mut self, step_cost: i64) {
        self.inner.step_cost = step_cost;
    }

    fn set_observation_bits(&mut self, bits: usize) {
        self.inner.observation_bits = bits;
    }

    fn set_reward_bits(&mut self, bits: usize) {
        self.inner.reward_bits = bits;
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(PyValueError::new_err)
    }
}

#[cfg(feature = "vm")]
#[pyclass(name = "NyxVmEnvironment", unsendable)]
struct PyNyxVmEnvironment {
    inner: infotheory::aixi::vm_nyx::NyxVmEnvironment,
}

#[cfg(feature = "vm")]
#[pymethods]
impl PyNyxVmEnvironment {
    #[new]
    fn new(config: &PyNyxVmConfig) -> PyResult<Self> {
        config.inner.validate().map_err(PyValueError::new_err)?;
        let env = infotheory::aixi::vm_nyx::NyxVmEnvironment::new(config.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: env })
    }

    fn reset(&mut self) -> PyResult<()> {
        self.inner
            .reset()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn run_step<'py>(&mut self, py: Python<'py>, payload: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let step = self
            .inner
            .run_step(payload)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = PyDict::new(py);
        out.set_item("done", step.done)?;
        out.set_item("output", PyBytes::new(py, &step.output))?;
        out.set_item("parsed_obs", step.parsed_obs)?;
        out.set_item("parsed_rew", step.parsed_rew)?;
        out.set_item("trace_data", PyBytes::new(py, &step.trace_data))?;
        out.set_item("shared_memory", PyBytes::new(py, &step.shared_memory))?;
        Ok(out)
    }
}

#[cfg(feature = "backend-rosa")]
#[pyclass(name = "SearchGranularity", eq, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
struct PySearchGranularity {
    inner: infotheory::search::SearchGranularity,
}

#[cfg(feature = "backend-rosa")]
#[pymethods]
impl PySearchGranularity {
    #[classattr]
    #[pyo3(name = "Snippet")]
    fn snippet() -> Self {
        Self {
            inner: infotheory::search::SearchGranularity::Snippet,
        }
    }
    #[classattr]
    #[pyo3(name = "File")]
    fn file() -> Self {
        Self {
            inner: infotheory::search::SearchGranularity::File,
        }
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            infotheory::search::SearchGranularity::Snippet => "SearchGranularity.Snippet",
            infotheory::search::SearchGranularity::File => "SearchGranularity.File",
        }
    }
}

#[cfg(feature = "backend-rosa")]
#[pyclass(name = "Stage2PriorMode", eq, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
struct PyStage2PriorMode {
    inner: infotheory::search::Stage2PriorMode,
}

#[cfg(feature = "backend-rosa")]
#[pymethods]
impl PyStage2PriorMode {
    #[classattr]
    #[pyo3(name = "Use")]
    fn use_prior() -> Self {
        Self {
            inner: infotheory::search::Stage2PriorMode::Use,
        }
    }
    #[classattr]
    #[pyo3(name = "Disable")]
    fn disable() -> Self {
        Self {
            inner: infotheory::search::Stage2PriorMode::Disable,
        }
    }
    #[classattr]
    #[pyo3(name = "Summarize")]
    fn summarize() -> Self {
        Self {
            inner: infotheory::search::Stage2PriorMode::Summarize,
        }
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            infotheory::search::Stage2PriorMode::Use => "Stage2PriorMode.Use",
            infotheory::search::Stage2PriorMode::Disable => "Stage2PriorMode.Disable",
            infotheory::search::Stage2PriorMode::Summarize => "Stage2PriorMode.Summarize",
        }
    }
}

#[cfg(feature = "backend-rosa")]
#[pyfunction]
#[pyo3(signature = (
    query,
    target_path,
    granularity=None,
    universal_prior=None,
    stage2_prior_mode=None,
    max_order=8,
    top_k=50,
    stage0_keep_frac=0.2,
    rate_backend=None,
    compression_backend=None,
    method=None
))]
fn search(
    py: Python<'_>,
    query: &str,
    target_path: &str,
    granularity: Option<&PySearchGranularity>,
    universal_prior: Option<String>,
    stage2_prior_mode: Option<&PyStage2PriorMode>,
    max_order: i64,
    top_k: usize,
    stage0_keep_frac: f64,
    rate_backend: Option<&Bound<'_, PyAny>>,
    compression_backend: Option<&Bound<'_, PyAny>>,
    method: Option<&str>,
) -> PyResult<Vec<(String, usize, usize, f64)>> {
    let raw_rb = rate_backend_from_py(rate_backend, method)?;
    let rb = compile_rate_backend(raw_rb.clone())?;
    let cb = compiled_compression_backend_from_py(compression_backend, method, Some(raw_rb))?;
    let q = query.to_string();
    let tp = target_path.to_string();
    let gran = granularity
        .map(|g| g.inner)
        .unwrap_or(infotheory::search::SearchGranularity::Snippet);
    let s2pm = stage2_prior_mode
        .map(|m| m.inner)
        .unwrap_or(infotheory::search::Stage2PriorMode::Use);

    py.detach(|| {
        py_try(|| {
            let opts = infotheory::search::SearchOptions {
                granularity: gran,
                universal_prior,
                stage2_prior_mode: s2pm,
                max_order,
                top_k,
                stage0_keep_frac,
                ctx: InfotheoryCtx::new(rb, cb),
            };
            let results = infotheory::search::search_with_options(&q, &tp, &opts)
                .map_err(py_infotheory_error)?;
            Ok(results
                .into_iter()
                .map(|s| {
                    (
                        s.path.to_string_lossy().to_string(),
                        s.start_line,
                        s.end_line,
                        s.score,
                    )
                })
                .collect())
        })
    })
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRateBackend>()?;
    m.add_class::<PyCompressionBackend>()?;
    m.add_class::<PyInfotheoryCtx>()?;
    m.add_class::<PyGenerationStrategy>()?;
    m.add_class::<PyGenerationUpdateMode>()?;
    m.add_class::<PyGenerationConfig>()?;
    m.add_class::<PyRateBackendSession>()?;
    m.add_class::<PyMixtureKind>()?;
    m.add_class::<PyMixtureScheduleMode>()?;
    m.add_class::<PyMixtureExpertSpec>()?;
    m.add_class::<PyMixtureSpec>()?;
    m.add_class::<PyParticleSpec>()?;
    m.add_class::<PyCalibrationContextKind>()?;
    m.add_class::<PyNcdVariant>()?;
    m.add_class::<PyObservationKeyMode>()?;
    m.add_class::<PyRandomGenerator>()?;
    m.add_class::<PyAgentConfig>()?;
    m.add_class::<PyAiqiConfig>()?;
    m.add_class::<PyAgent>()?;
    m.add_class::<PyAiqiAgent>()?;
    #[cfg(feature = "backend-ctw")]
    m.add_class::<PyCtwPredictor>()?;
    #[cfg(feature = "backend-ctw")]
    m.add_class::<PyFacCtwPredictor>()?;
    #[cfg(feature = "backend-rosa")]
    m.add_class::<PyRosaPredictor>()?;
    #[cfg(feature = "backend-zpaq")]
    m.add_class::<PyZpaqPredictor>()?;
    #[cfg(feature = "backend-rwkv")]
    m.add_class::<PyRwkvPredictor>()?;
    m.add_class::<PySearchNode>()?;
    m.add_class::<PySearchTree>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<CoinFlipEnv>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<BiasedRockPaperScissorEnv>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<KuhnPokerEnv>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<ExtendedTigerEnv>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<TicTacToeEnv>()?;
    #[cfg(feature = "aixi-gameengine")]
    m.add_class::<BlackjackEnv>()?;
    #[cfg(feature = "aixi-gameengine-physics")]
    m.add_class::<PlatformerEnv>()?;
    #[cfg(feature = "backend-rosa")]
    m.add_class::<PySearchGranularity>()?;
    #[cfg(feature = "backend-rosa")]
    m.add_class::<PyStage2PriorMode>()?;

    #[cfg(feature = "vm")]
    {
        m.add_class::<PyNyxVmConfig>()?;
        m.add_class::<PyNyxVmEnvironment>()?;
    }

    m.add_function(wrap_pyfunction!(get_default_ctx, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_ctx, m)?)?;
    m.add_function(wrap_pyfunction!(rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(validate_zpaq_rate_method, m)?)?;
    m.add_function(wrap_pyfunction!(get_compressed_size, m)?)?;
    m.add_function(wrap_pyfunction!(get_compressed_size_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(get_bytes_from_paths, m)?)?;
    m.add_function(wrap_pyfunction!(get_compressed_sizes_from_paths, m)?)?;
    m.add_function(wrap_pyfunction!(
        get_sequential_compressed_sizes_from_sequential_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        get_parallel_compressed_sizes_from_sequential_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        get_sequential_compressed_sizes_from_parallel_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        get_parallel_compressed_sizes_from_parallel_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compress_size_backend, m)?)?;
    m.add_function(wrap_pyfunction!(compress_size_chain_backend, m)?)?;
    m.add_function(wrap_pyfunction!(compress_bytes_backend, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_bytes_backend, m)?)?;
    m.add_function(wrap_pyfunction!(compress_file, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_file, m)?)?;
    m.add_function(wrap_pyfunction!(generate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(generate_bytes_conditional_chain, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_paths, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_bytes_default, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_paths_with_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_bytes_with_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_vitanyi, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_sym_vitanyi, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_cons, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_sym_cons, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_matrix_paths, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_matrix_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_matrix_paths_with_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ncd_matrix_bytes_with_backend, m)?)?;
    m.add_function(wrap_pyfunction!(marginal_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(biased_entropy_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(biased_entropy_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(joint_marginal_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(joint_entropy_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(joint_entropy_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_entropy_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_marg_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ned_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_marg_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ned_cons_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_cons_marg_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_cons_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(nte_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(nte_marg_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(nte_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(nte_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(tvd_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(nhd_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_rate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_rate_backend, m)?)?;
    m.add_function(wrap_pyfunction!(d_kl_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(js_div_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ned_paths, m)?)?;
    m.add_function(wrap_pyfunction!(nte_paths, m)?)?;
    m.add_function(wrap_pyfunction!(tvd_paths, m)?)?;
    m.add_function(wrap_pyfunction!(nhd_paths, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information_paths, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_entropy_paths, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_paths, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence_paths, m)?)?;
    m.add_function(wrap_pyfunction!(js_divergence_paths, m)?)?;
    m.add_function(wrap_pyfunction!(intrinsic_dependence_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(resistance_to_transformation_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(verify_identity, m)?)?;
    m.add_function(wrap_pyfunction!(verify_symmetry, m)?)?;
    m.add_function(wrap_pyfunction!(verify_triangle_inequality, m)?)?;
    m.add_function(wrap_pyfunction!(verify_non_negativity, m)?)?;
    m.add_function(wrap_pyfunction!(verify_mi_nonnegative, m)?)?;
    m.add_function(wrap_pyfunction!(verify_subadditivity, m)?)?;
    m.add_function(wrap_pyfunction!(verify_conditioning_reduces_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(verify_chain_rule, m)?)?;
    m.add_function(wrap_pyfunction!(verify_ncd_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(verify_entropy_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(observation_key_from_stream, m)?)?;
    m.add_function(wrap_pyfunction!(observation_repr_from_stream, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bits, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bits, m)?)?;
    m.add_function(wrap_pyfunction!(encode_reward_bits, m)?)?;
    m.add_function(wrap_pyfunction!(decode_reward_bits, m)?)?;
    m.add_function(wrap_pyfunction!(encode_reward_offset_bits, m)?)?;
    m.add_function(wrap_pyfunction!(decode_reward_offset_bits, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_probe, m)?)?;
    m.add_function(wrap_pyfunction!(environment_probe, m)?)?;
    m.add_function(wrap_pyfunction!(run_agent_with_environment, m)?)?;
    m.add_function(wrap_pyfunction!(run_aiqi_with_environment, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_simulator, m)?)?;
    #[cfg(feature = "backend-rosa")]
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(vm_enabled, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    fn with_python_initialized<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(Python::initialize);
        Python::attach(f)
    }

    #[test]
    fn parse_observation_key_mode_accepts_pyclass_instance() {
        with_python_initialized(|py| {
            let mode_obj = Py::new(
                py,
                PyObservationKeyMode {
                    inner: infotheory::aixi::common::ObservationKeyMode::First,
                },
            )
            .expect("construct ObservationKeyMode pyclass");

            let parsed = PyAgentSimulatorShim::parse_key_mode(mode_obj.bind(py).as_any());
            assert_eq!(parsed, infotheory::aixi::common::ObservationKeyMode::First);
        });
    }

    #[test]
    fn parse_observation_key_mode_accepts_string_aliases() {
        with_python_initialized(|py| {
            let stream_hash = pyo3::types::PyString::new(py, "stream_hash");
            let parsed_hash = PyAgentSimulatorShim::parse_key_mode(stream_hash.as_any());
            assert_eq!(
                parsed_hash,
                infotheory::aixi::common::ObservationKeyMode::StreamHash
            );

            let stream_hash_hyphen = pyo3::types::PyString::new(py, "stream-hash");
            let parsed_hash_hyphen =
                PyAgentSimulatorShim::parse_key_mode(stream_hash_hyphen.as_any());
            assert_eq!(
                parsed_hash_hyphen,
                infotheory::aixi::common::ObservationKeyMode::StreamHash
            );

            let full_stream = pyo3::types::PyString::new(py, "fullstream");
            let parsed_full = PyAgentSimulatorShim::parse_key_mode(full_stream.as_any());
            assert_eq!(
                parsed_full,
                infotheory::aixi::common::ObservationKeyMode::FullStream
            );

            let full_stream_hyphen = pyo3::types::PyString::new(py, "full-stream");
            let parsed_full_hyphen =
                PyAgentSimulatorShim::parse_key_mode(full_stream_hyphen.as_any());
            assert_eq!(
                parsed_full_hyphen,
                infotheory::aixi::common::ObservationKeyMode::FullStream
            );

            let full = pyo3::types::PyString::new(py, "full");
            let parsed_full_alias = PyAgentSimulatorShim::parse_key_mode(full.as_any());
            assert_eq!(
                parsed_full_alias,
                infotheory::aixi::common::ObservationKeyMode::FullStream
            );
        });
    }

    #[test]
    fn parse_framing_mode_accepts_aliases() {
        let raw = parse_framing_mode("raw").expect("raw framing");
        let framed = parse_framing_mode("framed").expect("framed framing");
        let framed_alias = parse_framing_mode("frame").expect("frame alias");
        assert_eq!(raw, infotheory::compression::FramingMode::Raw);
        assert_eq!(framed, infotheory::compression::FramingMode::Framed);
        assert_eq!(framed_alias, infotheory::compression::FramingMode::Framed);
        assert!(parse_framing_mode("nope").is_err());
    }

    #[test]
    fn compression_backend_rate_methods_accept_framed() {
        let rb = PyRateBackend {
            inner: RateBackend::Ctw { depth: 8 },
        };
        let cb_ac = PyCompressionBackend::rate_ac(&rb, "framed").expect("rate_ac framed");
        let cb_rans = PyCompressionBackend::rate_rans(&rb, "framed").expect("rate_rans framed");
        match cb_ac.inner {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
        match cb_rans.inner {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }

    #[test]
    fn compression_backend_rate_methods_default_to_framed() {
        let rb = PyRateBackend {
            inner: RateBackend::Ctw { depth: 8 },
        };
        let cb_ac = PyCompressionBackend::rate_ac(&rb, "framed").expect("rate_ac default framed");
        let cb_rans =
            PyCompressionBackend::rate_rans(&rb, "framed").expect("rate_rans default framed");
        match cb_ac.inner {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
        match cb_rans.inner {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }

    #[test]
    fn parse_rate_compression_backend_defaults_to_framed() {
        let ac = parse_compression_backend("rate-ac", None, None).expect("parse rate-ac");
        let rans = parse_compression_backend("rate-rans", None, None).expect("parse rate-rans");
        match ac {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
        match rans {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }

    #[cfg(feature = "backend-rwkv")]
    #[test]
    fn compression_backend_rwkv7_constructor_uses_shared_cfg_lowering() {
        let backend = PyCompressionBackend::rwkv7(
            Some(
                "cfg:hidden=64,intermediate=64,layers=1,train=sgd,lr=0.01;policy:schedule=0..100:infer"
                    .to_string(),
            ),
            "ac",
        )
        .expect("rwkv7 constructor");

        match backend.inner {
            CompressionBackend::Rate {
                rate_backend: RateBackend::Rwkv7Method { .. },
                coder,
                framing,
            } => {
                assert_eq!(coder, infotheory::coders::CoderType::AC);
                assert_eq!(framing, infotheory::compression::FramingMode::Framed);
            }
            _ => panic!("expected rate-coded rwkv7 backend"),
        }
    }

    #[test]
    fn file_roundtrip_backend_forces_rate_framed() {
        let backend = CompressionBackend::Rate {
            rate_backend: RateBackend::Ctw { depth: 8 },
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        };
        let normalized = file_roundtrip_backend(&backend);
        match normalized {
            CompressionBackend::Rate { framing, .. } => {
                assert_eq!(framing, infotheory::compression::FramingMode::Framed)
            }
            _ => panic!("expected rate backend"),
        }
    }
}
