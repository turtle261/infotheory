#![allow(clippy::needless_pass_by_value)]

use infotheory::{
    CompressionBackend, InfotheoryCtx, MixtureExpertSpec, MixtureKind, MixtureSpec, NcdVariant,
    ParticleSpec, RateBackend,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
#[cfg(feature = "vm")]
use pyo3::types::PyDict;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

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
            "streamhash" | "stream_hash" | "hash" => {
                return Ok(infotheory::aixi::common::ObservationKeyMode::StreamHash);
            }
            "fullstream" | "full_stream" => {
                return Ok(infotheory::aixi::common::ObservationKeyMode::FullStream);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown ObservationKeyMode '{s}' (expected one of: first, last, stream_hash, full_stream)"
                )));
            }
        }
    }
    Err(PyValueError::new_err(
        "ObservationKeyMode must be an ObservationKeyMode enum value or string alias",
    ))
}

fn parse_particle_spec_json(v: &serde_json::Value) -> PyResult<ParticleSpec> {
    if v.get("experts").is_some() {
        return Err(PyValueError::new_err(
            "looks like a mixture spec (found 'experts'); expected ParticleSpec JSON",
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
            return Err(PyValueError::new_err(format!(
                "looks like a mixture spec (kind='{kind}'); expected ParticleSpec JSON"
            )));
        }
    }
    let d = ParticleSpec::default();
    Ok(ParticleSpec {
        num_particles: v
            .get("num_particles")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.num_particles),
        context_window: v
            .get("context_window")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.context_window),
        unroll_steps: v
            .get("unroll_steps")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.unroll_steps),
        num_cells: v
            .get("num_cells")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.num_cells),
        cell_dim: v
            .get("cell_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.cell_dim),
        num_rules: v
            .get("num_rules")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.num_rules),
        selector_hidden: v
            .get("selector_hidden")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.selector_hidden),
        rule_hidden: v
            .get("rule_hidden")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.rule_hidden),
        noise_dim: v
            .get("noise_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.noise_dim),
        deterministic: v
            .get("deterministic")
            .and_then(|x| x.as_bool())
            .unwrap_or(d.deterministic),
        enable_noise: v
            .get("enable_noise")
            .and_then(|x| x.as_bool())
            .unwrap_or(d.enable_noise),
        noise_scale: v
            .get("noise_scale")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.noise_scale),
        noise_anneal_steps: v
            .get("noise_anneal_steps")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.noise_anneal_steps),
        learning_rate_readout: v
            .get("learning_rate_readout")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.learning_rate_readout),
        learning_rate_selector: v
            .get("learning_rate_selector")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.learning_rate_selector),
        learning_rate_rule: v
            .get("learning_rate_rule")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.learning_rate_rule),
        bptt_depth: v
            .get("bptt_depth")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.bptt_depth),
        optimizer_momentum: v
            .get("optimizer_momentum")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.optimizer_momentum),
        grad_clip: v
            .get("grad_clip")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.grad_clip),
        state_clip: v
            .get("state_clip")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.state_clip),
        forget_lambda: v
            .get("forget_lambda")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.forget_lambda),
        resample_threshold: v
            .get("resample_threshold")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.resample_threshold),
        mutate_fraction: v
            .get("mutate_fraction")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.mutate_fraction),
        mutate_scale: v
            .get("mutate_scale")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.mutate_scale),
        mutate_model_params: v
            .get("mutate_model_params")
            .and_then(|x| x.as_bool())
            .unwrap_or(d.mutate_model_params),
        diagnostics_interval: v
            .get("diagnostics_interval")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.diagnostics_interval),
        min_prob: v
            .get("min_prob")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.min_prob),
        seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(d.seed),
    })
}

fn parse_rate_backend(name: &str, method: Option<&str>) -> PyResult<RateBackend> {
    let m = method.unwrap_or_default();
    match name.to_ascii_lowercase().as_str() {
        "rosa" | "rosaplus" => Ok(RateBackend::RosaPlus),
        "ctw" => Ok(RateBackend::Ctw {
            depth: if m.is_empty() {
                16
            } else {
                m.parse().unwrap_or(16)
            },
        }),
        "fac-ctw" | "facctw" => {
            let depth = if m.is_empty() {
                16
            } else {
                m.parse().unwrap_or(16)
            };
            Ok(RateBackend::FacCtw {
                base_depth: depth,
                num_percept_bits: 8,
                encoding_bits: 8,
            })
        }
        "zpaq" => Ok(RateBackend::Zpaq {
            method: if m.is_empty() {
                "1".to_string()
            } else {
                m.to_string()
            },
        }),
        #[cfg(feature = "backend-mamba")]
        "mamba" | "mamba1" => {
            if m.is_empty() {
                Err(PyValueError::new_err(
                    "mamba backend requires method string (cfg:...;policy:... or file:...)",
                ))
            } else {
                Ok(RateBackend::MambaMethod {
                    method: m.to_string(),
                })
            }
        }
        #[cfg(not(feature = "backend-mamba"))]
        "mamba" | "mamba1" => Err(PyValueError::new_err(
            "mamba backend disabled at compile time",
        )),
        #[cfg(feature = "backend-rwkv")]
        "rwkv" | "rwkv7" => {
            if m.is_empty() {
                Err(PyValueError::new_err(
                    "rwkv backend requires method string (cfg:...;policy:... or file:...)",
                ))
            } else {
                Ok(RateBackend::Rwkv7Method {
                    method: m.to_string(),
                })
            }
        }
        #[cfg(not(feature = "backend-rwkv"))]
        "rwkv" | "rwkv7" => Err(PyValueError::new_err(
            "rwkv backend disabled at compile time",
        )),
        "particle" | "particles" => {
            let spec = if m.is_empty() {
                ParticleSpec::default()
            } else {
                let raw = std::fs::read_to_string(m).map_err(|e| {
                    PyValueError::new_err(format!("failed to read particle spec '{m}': {e}"))
                })?;
                let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
                    PyValueError::new_err(format!("invalid particle spec JSON: {e}"))
                })?;
                parse_particle_spec_json(&v)?
            };
            spec.validate()
                .map_err(|e| PyValueError::new_err(format!("invalid particle spec: {e}")))?;
            Ok(RateBackend::Particle {
                spec: Arc::new(spec),
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown rate backend '{name}'"
        ))),
    }
}

fn parse_compression_backend(
    name: &str,
    method: Option<&str>,
    rate_backend: Option<RateBackend>,
) -> PyResult<CompressionBackend> {
    let m = method.unwrap_or_default();
    match name.to_ascii_lowercase().as_str() {
        "zpaq" => Ok(CompressionBackend::Zpaq {
            method: if m.is_empty() {
                "5".to_string()
            } else {
                m.to_string()
            },
        }),
        "rate-ac" | "rate_ac" | "rateac" => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend.unwrap_or_default(),
            coder: infotheory::coders::CoderType::AC,
            framing: infotheory::compression::FramingMode::Raw,
        }),
        "rate-rans" | "rate_rans" | "raterans" => Ok(CompressionBackend::Rate {
            rate_backend: rate_backend.unwrap_or_default(),
            coder: infotheory::coders::CoderType::RANS,
            framing: infotheory::compression::FramingMode::Raw,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unknown compression backend '{name}'"
        ))),
    }
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
    #[pyo3(signature = (kind, experts, alpha=0.01, decay=None))]
    fn new(
        kind: &PyMixtureKind,
        experts: Vec<PyMixtureExpertSpec>,
        alpha: f64,
        decay: Option<f64>,
    ) -> Self {
        let mut spec = MixtureSpec::new(kind.inner, experts.into_iter().map(|e| e.inner).collect())
            .with_alpha(alpha);
        if let Some(d) = decay {
            spec = spec.with_decay(d);
        }
        Self { inner: spec }
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
    #[pyo3(signature = (method=None))]
    fn zpaq(method: Option<String>) -> Self {
        Self {
            inner: RateBackend::Zpaq {
                method: method.unwrap_or_else(|| "1".to_string()),
            },
        }
    }

    #[staticmethod]
    #[cfg(feature = "backend-mamba")]
    fn mamba(method: String) -> Self {
        Self {
            inner: RateBackend::MambaMethod { method },
        }
    }

    #[staticmethod]
    #[cfg(feature = "backend-rwkv")]
    fn rwkv7(method: String) -> Self {
        Self {
            inner: RateBackend::Rwkv7Method { method },
        }
    }

    #[staticmethod]
    fn mixture(spec: &PyMixtureSpec) -> Self {
        Self {
            inner: RateBackend::Mixture {
                spec: Arc::new(spec.inner.clone()),
            },
        }
    }

    #[staticmethod]
    fn particle(spec: &PyParticleSpec) -> Self {
        Self {
            inner: RateBackend::Particle {
                spec: Arc::new(spec.inner.clone()),
            },
        }
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
    fn zpaq(method: Option<String>) -> Self {
        Self {
            inner: CompressionBackend::Zpaq {
                method: method.unwrap_or_else(|| "5".to_string()),
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (rate_backend, framing="raw"))]
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
    #[pyo3(signature = (rate_backend, framing="raw"))]
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
        match method {
            Some(m) => match infotheory::rwkvzip::parse_method_spec(&m) {
                Ok(infotheory::rwkvzip::MethodSpec::File { path, policy: None }) => {
                    let model =
                        infotheory::load_rwkv7_model_from_path(path.to_string_lossy().as_ref());
                    Ok(Self {
                        inner: CompressionBackend::Rwkv7 { model, coder },
                    })
                }
                Ok(infotheory::rwkvzip::MethodSpec::File {
                    policy: Some(_), ..
                })
                | Ok(infotheory::rwkvzip::MethodSpec::Online { .. }) => Ok(Self {
                    inner: CompressionBackend::Rate {
                        rate_backend: RateBackend::Rwkv7Method { method: m },
                        coder,
                        framing: infotheory::compression::FramingMode::Raw,
                    },
                }),
                Err(e) => Err(PyValueError::new_err(format!(
                    "invalid rwkv method string: {e}"
                ))),
            },
            None => {
                let path = std::env::var("RWKV7_MODEL_PATH")
                    .map_err(|_| PyValueError::new_err("RWKV7_MODEL_PATH not set"))?;
                let model = infotheory::load_rwkv7_model_from_path(&path);
                Ok(Self {
                    inner: CompressionBackend::Rwkv7 { model, coder },
                })
            }
        }
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

#[pymethods]
impl PyInfotheoryCtx {
    #[new]
    #[pyo3(signature = (rate_backend=None, compression_backend=None))]
    fn new(
        rate_backend: Option<&PyRateBackend>,
        compression_backend: Option<&PyCompressionBackend>,
    ) -> Self {
        let rb = rate_backend.map(|b| b.inner.clone()).unwrap_or_default();
        let cb = compression_backend
            .map(|b| b.inner.clone())
            .unwrap_or_default();
        Self {
            inner: InfotheoryCtx::new(rb, cb),
        }
    }

    fn entropy_rate_bytes(&self, py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
        py.detach(|| py_try(|| Ok(self.inner.entropy_rate_bytes(data, max_order))))
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
        py.detach(|| py_try(|| Ok(self.inner.ncd_bytes(x, y, v))))
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
                Ok(infotheory::ncd_paths_backend(
                    x,
                    y,
                    &self.inner.compression_backend,
                    v,
                ))
            })
        })
    }
}

#[pyfunction]
fn get_default_ctx() -> PyInfotheoryCtx {
    PyInfotheoryCtx {
        inner: infotheory::get_default_ctx(),
    }
}

#[pyfunction]
fn set_default_ctx(ctx: &PyInfotheoryCtx) {
    infotheory::set_default_ctx(ctx.inner.clone());
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
    Ok(RateBackend::default())
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
    py.detach(|| py_try(|| Ok(infotheory::ncd_paths_backend(x, y, &cb, v))))
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
    let cb = compression_backend_from_py(backend, Some(method), None)?;
    py.detach(|| py_try(|| Ok(infotheory::ncd_bytes_backend(x, y, &cb, v))))
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

    py.detach(|| py_try(|| Ok(infotheory::ncd_paths_backend(x, y, &cb, v))))
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
    let cb = compression_backend_from_py(backend, method, None)?;
    py.detach(|| py_try(|| Ok(infotheory::ncd_bytes_backend(x, y, &cb, v))))
}

#[pyfunction]
#[pyo3(signature = (x, y, variant="vitanyi"))]
fn ncd_bytes_default(py: Python<'_>, x: &[u8], y: &[u8], variant: &str) -> PyResult<f64> {
    let v = parse_ncd_variant(variant)?;
    py.detach(|| py_try(|| Ok(infotheory::ncd_bytes_default(x, y, v))))
}

#[pyfunction]
fn entropy_rate_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::entropy_rate_bytes(data, max_order))))
}

#[pyfunction]
fn biased_entropy_rate_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::biased_entropy_rate_bytes(data, max_order))))
}

#[pyfunction]
fn marginal_entropy_bytes(data: &[u8]) -> f64 {
    infotheory::marginal_entropy_bytes(data)
}

#[pyfunction]
fn joint_marginal_entropy_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::joint_marginal_entropy_bytes(x, y)
}

macro_rules! py_metric_bytes_3 {
    ($fn_name:ident, $target:path) => {
        #[pyfunction]
        fn $fn_name(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
            py.detach(|| py_try(|| Ok($target(x, y, max_order))))
        }
    };
}

macro_rules! py_metric_paths_3 {
    ($fn_name:ident, $target:path) => {
        #[pyfunction]
        fn $fn_name(py: Python<'_>, x: &str, y: &str, max_order: i64) -> PyResult<f64> {
            py.detach(|| py_try(|| Ok($target(x, y, max_order))))
        }
    };
}

py_metric_bytes_3!(
    joint_entropy_rate_bytes,
    infotheory::joint_entropy_rate_bytes
);
py_metric_bytes_3!(
    conditional_entropy_rate_bytes,
    infotheory::conditional_entropy_rate_bytes
);
py_metric_bytes_3!(
    conditional_entropy_bytes,
    infotheory::conditional_entropy_bytes
);
py_metric_bytes_3!(
    mutual_information_bytes,
    infotheory::mutual_information_bytes
);
py_metric_bytes_3!(
    mutual_information_rate_bytes,
    infotheory::mutual_information_rate_bytes
);
py_metric_bytes_3!(ned_bytes, infotheory::ned_bytes);
py_metric_bytes_3!(nte_bytes, infotheory::nte_bytes);
py_metric_bytes_3!(tvd_bytes, infotheory::tvd_bytes);
py_metric_bytes_3!(nhd_bytes, infotheory::nhd_bytes);
py_metric_bytes_3!(cross_entropy_bytes, infotheory::cross_entropy_bytes);
py_metric_bytes_3!(
    cross_entropy_rate_bytes,
    infotheory::cross_entropy_rate_bytes
);

py_metric_paths_3!(ned_paths, infotheory::ned_paths);
py_metric_paths_3!(nte_paths, infotheory::nte_paths);
py_metric_paths_3!(tvd_paths, infotheory::tvd_paths);
py_metric_paths_3!(nhd_paths, infotheory::nhd_paths);
py_metric_paths_3!(
    mutual_information_paths,
    infotheory::mutual_information_paths
);
py_metric_paths_3!(
    conditional_entropy_paths,
    infotheory::conditional_entropy_paths
);
py_metric_paths_3!(cross_entropy_paths, infotheory::cross_entropy_paths);

#[pyfunction]
fn d_kl_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::d_kl_bytes(x, y)
}

#[pyfunction]
fn js_div_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::js_div_bytes(x, y)
}

#[pyfunction]
fn kl_divergence_paths(py: Python<'_>, x: &str, y: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::kl_divergence_paths(x, y))))
}

#[pyfunction]
fn js_divergence_paths(py: Python<'_>, x: &str, y: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::js_divergence_paths(x, y))))
}

#[pyfunction]
fn intrinsic_dependence_bytes(py: Python<'_>, data: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::intrinsic_dependence_bytes(data, max_order))))
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
            Ok(infotheory::resistance_to_transformation_bytes(
                x, tx, max_order,
            ))
        })
    })
}

#[pyfunction]
fn mutual_information_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::mutual_information_marg_bytes(x, y)
}

#[pyfunction]
fn ned_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::ned_marg_bytes(x, y)
}

#[pyfunction]
fn ned_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ned_rate_bytes(x, y, max_order))))
}

#[pyfunction]
fn ned_cons_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ned_cons_bytes(x, y, max_order))))
}

#[pyfunction]
fn ned_cons_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::ned_cons_marg_bytes(x, y)
}

#[pyfunction]
fn ned_cons_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ned_cons_rate_bytes(x, y, max_order))))
}

#[pyfunction]
fn nte_marg_bytes(x: &[u8], y: &[u8]) -> f64 {
    infotheory::nte_marg_bytes(x, y)
}

#[pyfunction]
fn nte_rate_bytes(py: Python<'_>, x: &[u8], y: &[u8], max_order: i64) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::nte_rate_bytes(x, y, max_order))))
}

#[pyfunction]
fn validate_zpaq_rate_method(method: &str) -> PyResult<()> {
    match infotheory::validate_zpaq_rate_method(method) {
        Ok(()) => Ok(()),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

#[pyfunction]
fn get_compressed_size(py: Python<'_>, path: &str, method: &str) -> PyResult<u64> {
    py.detach(|| py_try(|| Ok(infotheory::get_compressed_size(path, method))))
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
            Ok(infotheory::get_compressed_size_parallel(
                path, method, threads,
            ))
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
    py.detach(|| py_try(|| Ok(infotheory::get_compressed_sizes_from_paths(&refs, method))))
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
            Ok(infotheory::get_sequential_compressed_sizes_from_sequential_paths(&refs, method))
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
            Ok(
                infotheory::get_parallel_compressed_sizes_from_sequential_paths(
                    &refs, method, threads,
                ),
            )
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
            Ok(infotheory::get_sequential_compressed_sizes_from_parallel_paths(&refs, method))
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
            Ok(
                infotheory::get_parallel_compressed_sizes_from_parallel_paths(
                    &refs, method, threads,
                ),
            )
        })
    })
}

#[pyfunction]
fn get_bytes_from_paths(py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<Vec<u8>>> {
    let refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    py.detach(|| py_try(|| Ok(infotheory::get_bytes_from_paths(&refs))))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_vitanyi(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ncd_vitanyi(x, y, method))))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_sym_vitanyi(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ncd_sym_vitanyi(x, y, method))))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_cons(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ncd_cons(x, y, method))))
}

#[pyfunction]
#[pyo3(signature = (x, y, method="5"))]
fn ncd_sym_cons(py: Python<'_>, x: &str, y: &str, method: &str) -> PyResult<f64> {
    py.detach(|| py_try(|| Ok(infotheory::ncd_sym_cons(x, y, method))))
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
    let cb = compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    py.detach(|| py_try(|| Ok(infotheory::compress_size_backend(data, &cb))))
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
    let cb = compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    py.detach(|| py_try(|| Ok(infotheory::compress_size_chain_backend(&refs, &cb))))
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
    let cb = compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let out = py.detach(|| {
        py_try(|| {
            infotheory::compress_bytes_backend(data, &cb)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    let cb = compression_backend_from_py(compression_backend, Some(method), Some(rb))?;
    let out = py.detach(|| {
        py_try(|| {
            infotheory::decompress_bytes_backend(input, &cb)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    })?;
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| py_try(|| Ok(infotheory::entropy_rate_backend(data, max_order, &rb))))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            Ok(infotheory::biased_entropy_rate_backend(
                data, max_order, &rb,
            ))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            Ok(infotheory::cross_entropy_rate_backend(
                test_data, train_data, max_order, &rb,
            ))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| py_try(|| Ok(infotheory::joint_entropy_rate_backend(x, y, max_order, &rb))))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| {
        py_try(|| {
            Ok(infotheory::mutual_information_rate_backend(
                x, y, max_order, &rb,
            ))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| py_try(|| Ok(infotheory::ned_rate_backend(x, y, max_order, &rb))))
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
    let rb = rate_backend_from_py(backend, method)?;
    py.detach(|| py_try(|| Ok(infotheory::nte_rate_backend(x, y, max_order, &rb))))
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
    py.detach(|| py_try(|| Ok(infotheory::ncd_matrix_paths(&refs, method, v))))
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
    py.detach(|| py_try(|| Ok(infotheory::ncd_matrix_bytes(&datas, method, v))))
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
            use infotheory::aixi::environment::Environment;
            let mut env = PyEnvironmentShim::new(environment);
            let mut out = Vec::with_capacity(actions.len());
            for a in actions {
                env.perform_action(a);
                out.push((env.get_observation(), env.get_reward(), env.is_finished()));
            }
            Ok(out)
        })
    })
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
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(pool.install(|| {
                tree.search(
                    &mut sim,
                    &prev_obs_stream,
                    prev_rew,
                    prev_act,
                    num_simulations,
                )
            }))
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
        rwkv_model_path: Option<String>,
        rosa_max_order: Option<i64>,
        zpaq_method: Option<String>,
    ) -> Self {
        Self {
            inner: infotheory::aixi::agent::AgentConfig {
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
                rwkv_model_path,
                rosa_max_order,
                zpaq_method,
            },
        }
    }
}

#[pyclass(name = "Agent")]
struct PyAgent {
    inner: infotheory::aixi::agent::Agent,
}

#[pymethods]
impl PyAgent {
    #[new]
    fn new(config: &PyAgentConfig) -> Self {
        Self {
            inner: infotheory::aixi::agent::Agent::new(config.inner.clone()),
        }
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

#[pyclass(name = "CtwPredictor")]
struct PyCtwPredictor {
    inner: infotheory::aixi::model::CtwPredictor,
}

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

#[pyclass(name = "FacCtwPredictor")]
struct PyFacCtwPredictor {
    inner: infotheory::aixi::model::FacCtwPredictor,
}

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

#[pyclass(name = "RosaPredictor")]
struct PyRosaPredictor {
    inner: infotheory::aixi::model::RosaPredictor,
}

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

#[pyclass(name = "ZpaqPredictor")]
struct PyZpaqPredictor {
    inner: infotheory::aixi::model::ZpaqPredictor,
}

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
    fn new(model_path: String) -> Self {
        let model = infotheory::load_rwkv7_model_from_path(&model_path);
        Self {
            inner: infotheory::aixi::model::RwkvPredictor::new(model),
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

#[pyclass(name = "CoinFlipEnv")]
struct CoinFlipEnv {
    inner: infotheory::aixi::environment::CoinFlip,
}

#[pymethods]
impl CoinFlipEnv {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        Self {
            inner: infotheory::aixi::environment::CoinFlip::new(p),
        }
    }

    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

#[pyclass(name = "CtwTestEnv")]
struct CtwTestEnv {
    inner: infotheory::aixi::environment::CtwTest,
}

#[pymethods]
impl CtwTestEnv {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::environment::CtwTest::new(),
        }
    }
    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

#[pyclass(name = "BiasedRockPaperScissorEnv")]
struct BiasedRockPaperScissorEnv {
    inner: infotheory::aixi::environment::BiasedRockPaperScissor,
}

#[pymethods]
impl BiasedRockPaperScissorEnv {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::environment::BiasedRockPaperScissor::new(),
        }
    }
    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

#[pyclass(name = "ExtendedTigerEnv")]
struct ExtendedTigerEnv {
    inner: infotheory::aixi::environment::ExtendedTiger,
}

#[pymethods]
impl ExtendedTigerEnv {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::environment::ExtendedTiger::new(),
        }
    }
    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

#[pyclass(name = "TicTacToeEnv")]
struct TicTacToeEnv {
    inner: infotheory::aixi::environment::TicTacToe,
}

#[pymethods]
impl TicTacToeEnv {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::environment::TicTacToe::new(),
        }
    }
    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

#[pyclass(name = "KuhnPokerEnv")]
struct KuhnPokerEnv {
    inner: infotheory::aixi::environment::KuhnPoker,
}

#[pymethods]
impl KuhnPokerEnv {
    #[new]
    fn new() -> Self {
        Self {
            inner: infotheory::aixi::environment::KuhnPoker::new(),
        }
    }
    fn perform_action(&mut self, action: u64) {
        use infotheory::aixi::environment::Environment;
        self.inner.perform_action(action);
    }
    fn get_observation(&self) -> u64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_observation()
    }
    fn get_reward(&self) -> i64 {
        use infotheory::aixi::environment::Environment;
        self.inner.get_reward()
    }
    fn is_finished(&self) -> bool {
        use infotheory::aixi::environment::Environment;
        self.inner.is_finished()
    }
    fn drain_observations(&mut self) -> Vec<u64> {
        use infotheory::aixi::environment::Environment;
        self.inner.drain_observations()
    }
}

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

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRateBackend>()?;
    m.add_class::<PyCompressionBackend>()?;
    m.add_class::<PyInfotheoryCtx>()?;
    m.add_class::<PyMixtureKind>()?;
    m.add_class::<PyMixtureExpertSpec>()?;
    m.add_class::<PyMixtureSpec>()?;
    m.add_class::<PyParticleSpec>()?;
    m.add_class::<PyNcdVariant>()?;
    m.add_class::<PyObservationKeyMode>()?;
    m.add_class::<PyRandomGenerator>()?;
    m.add_class::<PyAgentConfig>()?;
    m.add_class::<PyAgent>()?;
    m.add_class::<PyCtwPredictor>()?;
    m.add_class::<PyFacCtwPredictor>()?;
    m.add_class::<PyRosaPredictor>()?;
    m.add_class::<PyZpaqPredictor>()?;
    #[cfg(feature = "backend-rwkv")]
    m.add_class::<PyRwkvPredictor>()?;
    m.add_class::<PySearchNode>()?;
    m.add_class::<PySearchTree>()?;
    m.add_class::<CoinFlipEnv>()?;
    m.add_class::<CtwTestEnv>()?;
    m.add_class::<BiasedRockPaperScissorEnv>()?;
    m.add_class::<ExtendedTigerEnv>()?;
    m.add_class::<TicTacToeEnv>()?;
    m.add_class::<KuhnPokerEnv>()?;

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
    m.add_function(wrap_pyfunction!(search_with_simulator, m)?)?;
    m.add_function(wrap_pyfunction!(vm_enabled, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_observation_key_mode_accepts_pyclass_instance() {
        Python::attach(|py| {
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
        Python::attach(|py| {
            let stream_hash = pyo3::types::PyString::new(py, "stream_hash");
            let parsed_hash = PyAgentSimulatorShim::parse_key_mode(stream_hash.as_any());
            assert_eq!(
                parsed_hash,
                infotheory::aixi::common::ObservationKeyMode::StreamHash
            );

            let full_stream = pyo3::types::PyString::new(py, "fullstream");
            let parsed_full = PyAgentSimulatorShim::parse_key_mode(full_stream.as_any());
            assert_eq!(
                parsed_full,
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
}
