//! Binary envelope codec for canonical top-level spec documents.

use super::*;
use std::path::Path;
use std::sync::Arc;

pub(super) fn encode_spec_document_payload(doc: &SpecDocument) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DOCUMENT_MAGIC);
    out.push(DOCUMENT_BINARY_VERSION);
    match doc {
        SpecDocument::PlannerRun(spec) => {
            out.push(0);
            encode_planner_run(spec, &mut out);
        }
        #[cfg(feature = "tuner")]
        SpecDocument::Tune(spec) => {
            out.push(1);
            encode_tune_spec(spec, &mut out);
        }
        SpecDocument::RateBackend(backend) => {
            out.push(2);
            encode_rate_backend(&mut out, backend);
        }
        SpecDocument::CompressionBackend(backend) => {
            out.push(3);
            encode_compression_backend(&mut out, backend);
        }
    }
    out
}

pub(super) fn decode_spec_document(bytes: &[u8], base_dir: &Path) -> SpecResult<SpecDocument> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.read_exact(4)?;
    if magic != DOCUMENT_MAGIC {
        return Err(SpecError::new("invalid spec document magic"));
    }
    let version = cursor.read_u8()?;
    if version != DOCUMENT_BINARY_VERSION {
        return Err(SpecError::new(format!(
            "unsupported spec document binary version '{version}'"
        )));
    }
    let document = match cursor.read_u8()? {
        0 => Ok(SpecDocument::PlannerRun(decode_planner_run(
            &mut cursor,
            base_dir,
        )?)),
        #[cfg(feature = "tuner")]
        1 => Ok(SpecDocument::Tune(decode_tune_spec(&mut cursor, base_dir)?)),
        #[cfg(not(feature = "tuner"))]
        1 => Err(SpecError::new(
            "tune binary documents require infotheory built with feature 'tuner'",
        )),
        2 => Ok(SpecDocument::RateBackend(decode_rate_backend(
            &mut cursor,
            base_dir,
        )?)),
        3 => Ok(SpecDocument::CompressionBackend(
            decode_compression_backend(&mut cursor, base_dir)?,
        )),
        tag => Err(SpecError::new(format!("unknown spec document tag '{tag}'"))),
    }?;
    if cursor.has_remaining() {
        return Err(SpecError::new("unexpected trailing bytes in spec document"));
    }
    Ok(document)
}

fn encode_planner_run(spec: &PlannerRunSpec, out: &mut Vec<u8>) {
    encode_assets(&spec.assets, out);
    encode_environment_spec(&spec.environment, out);
    encode_interface_spec(&spec.interface, out);
    encode_controller_spec(&spec.controller, out);
    encode_runtime_spec(&spec.runtime, out);
}

fn decode_planner_run(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<PlannerRunSpec> {
    let spec = PlannerRunSpec {
        assets: decode_assets(cursor)?,
        environment: decode_environment_spec(cursor, base_dir)?,
        interface: decode_interface_spec(cursor)?,
        controller: decode_controller_spec(cursor, base_dir)?,
        runtime: decode_runtime_spec(cursor)?,
    };
    if matches!(
        spec.environment,
        EnvironmentSpec::Builtin {
            builtin: BuiltinEnvironmentSpec::TunerBridge
        }
    ) {
        return Err(SpecError::new(
            "builtin environment 'tuner_bridge' is an internal tuner planner bridge and is not accepted in public binary planner-run documents",
        ));
    }
    Ok(spec)
}

#[cfg(feature = "tuner")]
fn encode_tune_spec(spec: &TuneSpec, out: &mut Vec<u8>) {
    encode_assets(&spec.assets, out);
    push_string(out, &spec.input_asset);
    encode_compression_backend(out, &spec.baseline_candidate);
    encode_tune_controller(&spec.controller, out);
    encode_tune_bounds(&spec.bounds, out);
    push_f64(out, spec.eval_time_limit_seconds);
    push_f64(out, spec.time_budget_seconds);
    push_f64(out, spec.min_throughput_bytes_per_second);
    push_u64(out, spec.max_memory_bytes);
    push_string(out, &spec.output_config_path);
    push_u64(out, spec.seed);
    push_option_string(out, spec.report_path.as_deref());
}

#[cfg(feature = "tuner")]
fn decode_tune_spec(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<TuneSpec> {
    let assets = decode_assets(cursor)?;
    let input_asset = cursor.read_string()?;
    Ok(TuneSpec {
        assets,
        input_asset,
        baseline_candidate: decode_compression_backend(cursor, base_dir)?,
        controller: decode_tune_controller(cursor)?,
        bounds: decode_tune_bounds(cursor)?,
        eval_time_limit_seconds: cursor.read_f64()?,
        time_budget_seconds: cursor.read_f64()?,
        min_throughput_bytes_per_second: cursor.read_f64()?,
        max_memory_bytes: cursor.read_u64()?,
        output_config_path: cursor.read_string()?,
        seed: cursor.read_u64()?,
        report_path: cursor.read_option_string()?,
    })
}

fn encode_assets(assets: &[AssetBinding], out: &mut Vec<u8>) {
    push_u64(out, assets.len() as u64);
    for asset in assets {
        push_string(out, &asset.id);
        push_string(out, &asset.path);
    }
}

fn decode_assets(cursor: &mut Cursor<'_>) -> SpecResult<Vec<AssetBinding>> {
    let len = cursor.read_u64()? as usize;
    let mut assets = Vec::with_capacity(len);
    for _ in 0..len {
        assets.push(AssetBinding {
            id: cursor.read_string()?,
            path: cursor.read_string()?,
        });
    }
    Ok(assets)
}

fn encode_zpaq_method_spec(out: &mut Vec<u8>, method: &crate::api::ZpaqMethodSpec) {
    match method {
        crate::api::ZpaqMethodSpec::Literal { value } => {
            out.push(0);
            push_string(out, value);
        }
    }
}

fn decode_zpaq_method_spec(cursor: &mut Cursor<'_>) -> SpecResult<crate::api::ZpaqMethodSpec> {
    match cursor.read_u8()? {
        0 => Ok(crate::api::ZpaqMethodSpec::literal(cursor.read_string()?)),
        tag => Err(SpecError::new(format!(
            "unknown zpaq method spec tag '{tag}'"
        ))),
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn encode_llm_position_expr(out: &mut Vec<u8>, expr: &crate::backends::llm_policy::PositionExpr) {
    match expr {
        crate::backends::llm_policy::PositionExpr::Bytes(value) => {
            out.push(0);
            push_u64(out, *value);
        }
        crate::backends::llm_policy::PositionExpr::Percent(value) => {
            out.push(1);
            push_f64(out, *value);
        }
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn decode_llm_position_expr(
    cursor: &mut Cursor<'_>,
) -> SpecResult<crate::backends::llm_policy::PositionExpr> {
    match cursor.read_u8()? {
        0 => Ok(crate::backends::llm_policy::PositionExpr::Bytes(
            cursor.read_u64()?,
        )),
        1 => Ok(crate::backends::llm_policy::PositionExpr::Percent(
            cursor.read_f64()?,
        )),
        tag => Err(SpecError::new(format!(
            "unknown llm position expr tag '{tag}'"
        ))),
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn encode_optimizer_kind(out: &mut Vec<u8>, kind: crate::backends::llm_policy::OptimizerKind) {
    out.push(match kind {
        crate::backends::llm_policy::OptimizerKind::Sgd => 0,
        crate::backends::llm_policy::OptimizerKind::Adam => 1,
    });
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn decode_optimizer_kind(
    cursor: &mut Cursor<'_>,
) -> SpecResult<crate::backends::llm_policy::OptimizerKind> {
    match cursor.read_u8()? {
        0 => Ok(crate::backends::llm_policy::OptimizerKind::Sgd),
        1 => Ok(crate::backends::llm_policy::OptimizerKind::Adam),
        tag => Err(SpecError::new(format!(
            "unknown optimizer kind tag '{tag}'"
        ))),
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn encode_train_scope_set(out: &mut Vec<u8>, scope: &crate::backends::llm_policy::TrainScopeSet) {
    push_bool(out, scope.all);
    push_string_list(out, &scope.names);
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn decode_train_scope_set(
    cursor: &mut Cursor<'_>,
) -> SpecResult<crate::backends::llm_policy::TrainScopeSet> {
    Ok(crate::backends::llm_policy::TrainScopeSet {
        all: cursor.read_bool()?,
        names: cursor.read_string_list()?,
    })
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn encode_policy_action(out: &mut Vec<u8>, action: &crate::backends::llm_policy::PolicyAction) {
    match action {
        crate::backends::llm_policy::PolicyAction::Infer => out.push(0),
        crate::backends::llm_policy::PolicyAction::Train(train) => {
            out.push(1);
            encode_train_scope_set(out, &train.scope);
            encode_optimizer_kind(out, train.optimizer);
            push_f64(out, train.hyper.lr as f64);
            push_u64(out, train.hyper.stride as u64);
            push_u64(out, train.hyper.bptt as u64);
            push_f64(out, train.hyper.clip as f64);
            push_f64(out, train.hyper.momentum as f64);
        }
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn decode_policy_action(
    cursor: &mut Cursor<'_>,
) -> SpecResult<crate::backends::llm_policy::PolicyAction> {
    match cursor.read_u8()? {
        0 => Ok(crate::backends::llm_policy::PolicyAction::Infer),
        1 => Ok(crate::backends::llm_policy::PolicyAction::Train(
            crate::backends::llm_policy::TrainAction {
                scope: decode_train_scope_set(cursor)?,
                optimizer: decode_optimizer_kind(cursor)?,
                hyper: crate::backends::llm_policy::OptimizerHyperParams {
                    lr: cursor.read_f64()? as f32,
                    stride: cursor.read_u64()? as usize,
                    bptt: cursor.read_u64()? as usize,
                    clip: cursor.read_f64()? as f32,
                    momentum: cursor.read_f64()? as f32,
                },
            },
        )),
        tag => Err(SpecError::new(format!("unknown policy action tag '{tag}'"))),
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn encode_llm_policy(out: &mut Vec<u8>, policy: Option<&crate::backends::llm_policy::LlmPolicy>) {
    match policy {
        Some(policy) => {
            out.push(1);
            push_option_string(
                out,
                policy
                    .load_from
                    .as_ref()
                    .map(|path| path.to_string_lossy())
                    .as_deref(),
            );
            push_u64(out, policy.schedule.len() as u64);
            for rule in &policy.schedule {
                match rule {
                    crate::backends::llm_policy::ScheduleRule::Interval(rule) => {
                        out.push(0);
                        encode_llm_position_expr(out, &rule.start);
                        encode_llm_position_expr(out, &rule.end);
                        encode_policy_action(out, &rule.action);
                    }
                    crate::backends::llm_policy::ScheduleRule::Repeat(rule) => {
                        out.push(1);
                        encode_llm_position_expr(out, &rule.start);
                        encode_llm_position_expr(out, &rule.end);
                        encode_llm_position_expr(out, &rule.period);
                        push_u64(out, rule.pattern.len() as u64);
                        for segment in &rule.pattern {
                            encode_llm_position_expr(out, &segment.span);
                            encode_policy_action(out, &segment.action);
                        }
                    }
                }
            }
        }
        None => out.push(0),
    }
}

#[cfg(any(feature = "backend-rwkv", feature = "backend-mamba"))]
fn decode_llm_policy(
    cursor: &mut Cursor<'_>,
) -> SpecResult<Option<crate::backends::llm_policy::LlmPolicy>> {
    if cursor.read_u8()? == 0 {
        return Ok(None);
    }
    let load_from = cursor.read_option_string()?.map(std::path::PathBuf::from);
    let rule_len = cursor.read_u64()? as usize;
    let mut schedule = Vec::with_capacity(rule_len);
    for _ in 0..rule_len {
        match cursor.read_u8()? {
            0 => schedule.push(crate::backends::llm_policy::ScheduleRule::Interval(
                crate::backends::llm_policy::PolicyRule {
                    start: decode_llm_position_expr(cursor)?,
                    end: decode_llm_position_expr(cursor)?,
                    action: decode_policy_action(cursor)?,
                },
            )),
            1 => {
                let start = decode_llm_position_expr(cursor)?;
                let end = decode_llm_position_expr(cursor)?;
                let period = decode_llm_position_expr(cursor)?;
                let pattern_len = cursor.read_u64()? as usize;
                let mut pattern = Vec::with_capacity(pattern_len);
                for _ in 0..pattern_len {
                    pattern.push(crate::backends::llm_policy::RepeatSegment {
                        span: decode_llm_position_expr(cursor)?,
                        action: decode_policy_action(cursor)?,
                    });
                }
                schedule.push(crate::backends::llm_policy::ScheduleRule::Repeat(
                    crate::backends::llm_policy::RepeatRule {
                        start,
                        end,
                        period,
                        pattern,
                    },
                ));
            }
            tag => {
                return Err(SpecError::new(format!(
                    "unknown llm policy schedule tag '{tag}'"
                )));
            }
        }
    }
    Ok(Some(crate::backends::llm_policy::LlmPolicy {
        load_from,
        schedule,
    }))
}

#[cfg(feature = "backend-rwkv")]
fn encode_rwkv_method_spec(out: &mut Vec<u8>, method: &crate::rwkvzip::MethodSpec) {
    match method {
        crate::rwkvzip::MethodSpec::File { path, policy } => {
            out.push(0);
            push_string(out, &path.to_string_lossy());
            encode_llm_policy(out, policy.as_ref());
        }
        crate::rwkvzip::MethodSpec::Online { cfg, policy } => {
            out.push(1);
            push_u64(out, cfg.hidden as u64);
            push_u64(out, cfg.layers as u64);
            push_u64(out, cfg.intermediate as u64);
            push_u64(out, cfg.decay_rank as u64);
            push_u64(out, cfg.a_rank as u64);
            push_u64(out, cfg.v_rank as u64);
            push_u64(out, cfg.g_rank as u64);
            push_u64(out, cfg.seed);
            out.push(match cfg.train_mode {
                crate::rwkvzip::OnlineTrainMode::None => 0,
                crate::rwkvzip::OnlineTrainMode::Sgd => 1,
                crate::rwkvzip::OnlineTrainMode::Adam => 2,
            });
            push_f64(out, cfg.lr as f64);
            push_u64(out, cfg.stride as u64);
            encode_llm_policy(out, policy.as_ref());
        }
    }
}

#[cfg(feature = "backend-rwkv")]
fn decode_rwkv_method_spec(cursor: &mut Cursor<'_>) -> SpecResult<crate::rwkvzip::MethodSpec> {
    match cursor.read_u8()? {
        0 => Ok(crate::rwkvzip::MethodSpec::File {
            path: std::path::PathBuf::from(cursor.read_string()?),
            policy: decode_llm_policy(cursor)?,
        }),
        1 => Ok(crate::rwkvzip::MethodSpec::Online {
            cfg: crate::rwkvzip::OnlineConfig {
                hidden: cursor.read_u64()? as usize,
                layers: cursor.read_u64()? as usize,
                intermediate: cursor.read_u64()? as usize,
                decay_rank: cursor.read_u64()? as usize,
                a_rank: cursor.read_u64()? as usize,
                v_rank: cursor.read_u64()? as usize,
                g_rank: cursor.read_u64()? as usize,
                seed: cursor.read_u64()?,
                train_mode: match cursor.read_u8()? {
                    0 => crate::rwkvzip::OnlineTrainMode::None,
                    1 => crate::rwkvzip::OnlineTrainMode::Sgd,
                    2 => crate::rwkvzip::OnlineTrainMode::Adam,
                    tag => {
                        return Err(SpecError::new(format!(
                            "unknown rwkv online train mode tag '{tag}'"
                        )));
                    }
                },
                lr: cursor.read_f64()? as f32,
                stride: cursor.read_u64()? as usize,
            },
            policy: decode_llm_policy(cursor)?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown rwkv method spec tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "backend-mamba")]
fn encode_mamba_method_spec(out: &mut Vec<u8>, method: &crate::mambazip::MethodSpec) {
    match method {
        crate::mambazip::MethodSpec::File { path, policy } => {
            out.push(0);
            push_string(out, &path.to_string_lossy());
            encode_llm_policy(out, policy.as_ref());
        }
        crate::mambazip::MethodSpec::Online { cfg, policy } => {
            out.push(1);
            push_u64(out, cfg.hidden as u64);
            push_u64(out, cfg.layers as u64);
            push_u64(out, cfg.intermediate as u64);
            push_u64(out, cfg.state as u64);
            push_u64(out, cfg.conv as u64);
            push_u64(out, cfg.dt_rank as u64);
            push_u64(out, cfg.seed);
            out.push(match cfg.train_mode {
                crate::mambazip::OnlineTrainMode::None => 0,
                crate::mambazip::OnlineTrainMode::Sgd => 1,
                crate::mambazip::OnlineTrainMode::Adam => 2,
            });
            push_f64(out, cfg.lr as f64);
            push_u64(out, cfg.stride as u64);
            encode_llm_policy(out, policy.as_ref());
        }
    }
}

#[cfg(feature = "backend-mamba")]
fn decode_mamba_method_spec(cursor: &mut Cursor<'_>) -> SpecResult<crate::mambazip::MethodSpec> {
    match cursor.read_u8()? {
        0 => Ok(crate::mambazip::MethodSpec::File {
            path: std::path::PathBuf::from(cursor.read_string()?),
            policy: decode_llm_policy(cursor)?,
        }),
        1 => Ok(crate::mambazip::MethodSpec::Online {
            cfg: crate::mambazip::OnlineConfig {
                hidden: cursor.read_u64()? as usize,
                layers: cursor.read_u64()? as usize,
                intermediate: cursor.read_u64()? as usize,
                state: cursor.read_u64()? as usize,
                conv: cursor.read_u64()? as usize,
                dt_rank: cursor.read_u64()? as usize,
                seed: cursor.read_u64()?,
                train_mode: match cursor.read_u8()? {
                    0 => crate::mambazip::OnlineTrainMode::None,
                    1 => crate::mambazip::OnlineTrainMode::Sgd,
                    2 => crate::mambazip::OnlineTrainMode::Adam,
                    tag => {
                        return Err(SpecError::new(format!(
                            "unknown mamba online train mode tag '{tag}'"
                        )));
                    }
                },
                lr: cursor.read_f64()? as f32,
                stride: cursor.read_u64()? as usize,
            },
            policy: decode_llm_policy(cursor)?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown mamba method spec tag '{tag}'"
        ))),
    }
}

fn encode_rate_backend(out: &mut Vec<u8>, backend: &RateBackend) {
    match backend {
        RateBackend::RosaPlus { max_order } => {
            out.push(0);
            push_i64(out, *max_order);
        }
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => {
            out.push(1);
            push_u64(out, *hash_bits as u64);
            push_u64(out, *min_len as u64);
            push_u64(out, *max_len as u64);
            push_f64(out, *base_mix);
            push_f64(out, *confidence_scale);
        }
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => {
            out.push(2);
            push_u64(out, *hash_bits as u64);
            push_u64(out, *min_len as u64);
            push_u64(out, *max_len as u64);
            push_u64(out, *gap_min as u64);
            push_u64(out, *gap_max as u64);
            push_f64(out, *base_mix);
            push_f64(out, *confidence_scale);
        }
        RateBackend::Ppmd { order, memory_mb } => {
            out.push(3);
            push_u64(out, *order as u64);
            push_u64(out, *memory_mb as u64);
        }
        RateBackend::Sequitur { context_bytes } => {
            out.push(4);
            push_u64(out, *context_bytes as u64);
        }
        RateBackend::Ctw { depth } => {
            out.push(5);
            push_u64(out, *depth as u64);
        }
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
        } => {
            out.push(6);
            push_u64(out, *base_depth as u64);
            push_u64(out, *num_percept_bits as u64);
            push_u64(out, *encoding_bits as u64);
        }
        RateBackend::Zpaq { method } => {
            out.push(7);
            encode_zpaq_method_spec(out, method);
        }
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => {
            out.push(8);
            encode_mamba_method_spec(out, method);
        }
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => {
            out.push(9);
            encode_rwkv_method_spec(out, method);
        }
        RateBackend::Mixture { spec } => {
            out.push(10);
            out.push(mixture_kind_tag(spec.kind));
            out.push(mixture_schedule_tag(spec.schedule));
            push_f64(out, spec.alpha);
            push_option_f64(out, spec.decay);
            push_u64(out, spec.experts.len() as u64);
            for expert in &spec.experts {
                push_option_string(out, expert.name.as_deref());
                push_f64(out, expert.log_prior);
                encode_rate_backend(out, &expert.backend);
            }
        }
        RateBackend::Particle { spec } => {
            out.push(11);
            encode_particle_spec(out, spec.as_ref());
        }
        RateBackend::Calibrated { spec } => {
            out.push(12);
            out.push(calibration_context_tag(spec.context));
            push_u64(out, spec.bins as u64);
            push_f64(out, spec.learning_rate);
            push_f64(out, spec.bias_clip);
            encode_rate_backend(out, &spec.base);
        }
    }
}

fn decode_rate_backend(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<RateBackend> {
    let _ = base_dir;
    match cursor.read_u8()? {
        0 => Ok(RateBackend::RosaPlus {
            max_order: cursor.read_i64()?,
        }),
        1 => Ok(RateBackend::Match {
            hash_bits: cursor.read_u64()? as usize,
            min_len: cursor.read_u64()? as usize,
            max_len: cursor.read_u64()? as usize,
            base_mix: cursor.read_f64()?,
            confidence_scale: cursor.read_f64()?,
        }),
        2 => Ok(RateBackend::SparseMatch {
            hash_bits: cursor.read_u64()? as usize,
            min_len: cursor.read_u64()? as usize,
            max_len: cursor.read_u64()? as usize,
            gap_min: cursor.read_u64()? as usize,
            gap_max: cursor.read_u64()? as usize,
            base_mix: cursor.read_f64()?,
            confidence_scale: cursor.read_f64()?,
        }),
        3 => Ok(RateBackend::Ppmd {
            order: cursor.read_u64()? as usize,
            memory_mb: cursor.read_u64()? as usize,
        }),
        4 => Ok(RateBackend::Sequitur {
            context_bytes: cursor.read_u64()? as usize,
        }),
        5 => Ok(RateBackend::Ctw {
            depth: cursor.read_u64()? as usize,
        }),
        6 => Ok(RateBackend::FacCtw {
            base_depth: cursor.read_u64()? as usize,
            num_percept_bits: cursor.read_u64()? as usize,
            encoding_bits: cursor.read_u64()? as usize,
        }),
        7 => Ok(RateBackend::Zpaq {
            method: decode_zpaq_method_spec(cursor)?,
        }),
        #[cfg(feature = "backend-mamba")]
        8 => Ok(RateBackend::MambaMethod {
            method: decode_mamba_method_spec(cursor)?,
        }),
        #[cfg(not(feature = "backend-mamba"))]
        8 => Err(SpecError::new(
            "binary mamba backend requires the 'backend-mamba' feature",
        )),
        #[cfg(feature = "backend-rwkv")]
        9 => Ok(RateBackend::Rwkv7Method {
            method: decode_rwkv_method_spec(cursor)?,
        }),
        #[cfg(not(feature = "backend-rwkv"))]
        9 => Err(SpecError::new(
            "binary rwkv backend requires the 'backend-rwkv' feature",
        )),
        10 => {
            let kind = decode_mixture_kind(cursor.read_u8()?)?;
            let schedule = decode_mixture_schedule(cursor.read_u8()?)?;
            let alpha = cursor.read_f64()?;
            let decay = cursor.read_option_f64()?;
            let expert_len = cursor.read_u64()? as usize;
            let mut experts = Vec::with_capacity(expert_len);
            for _ in 0..expert_len {
                experts.push(crate::api::MixtureExpertSpec {
                    name: cursor.read_option_string()?,
                    log_prior: cursor.read_f64()?,
                    backend: decode_rate_backend(cursor, base_dir)?,
                });
            }
            Ok(RateBackend::Mixture {
                spec: Arc::new(crate::api::MixtureSpec {
                    kind,
                    schedule,
                    alpha,
                    decay,
                    experts,
                }),
            })
        }
        11 => Ok(RateBackend::Particle {
            spec: Arc::new(decode_particle_spec(cursor)?),
        }),
        12 => Ok(RateBackend::Calibrated {
            spec: Arc::new(crate::api::CalibratedSpec {
                context: decode_calibration_context(cursor.read_u8()?)?,
                bins: cursor.read_u64()? as usize,
                learning_rate: cursor.read_f64()?,
                bias_clip: cursor.read_f64()?,
                base: decode_rate_backend(cursor, base_dir)?,
            }),
        }),
        tag => Err(SpecError::new(format!("unknown rate backend tag '{tag}'"))),
    }
}

fn encode_compression_backend(out: &mut Vec<u8>, backend: &CompressionBackend) {
    match backend {
        CompressionBackend::Zpaq { method, threads } => {
            out.push(0);
            encode_zpaq_method_spec(out, method);
            push_u64(out, threads.get() as u64);
        }
        #[cfg(feature = "backend-rwkv")]
        CompressionBackend::Rwkv7 { method, coder } => {
            out.push(1);
            encode_rwkv_method_spec(out, method);
            out.push(coder_tag(*coder));
        }
        CompressionBackend::Rate {
            rate_backend,
            coder,
            framing,
        } => {
            out.push(2);
            out.push(coder_tag(*coder));
            out.push(framing_tag(*framing));
            encode_rate_backend(out, rate_backend);
        }
    }
}

fn decode_compression_backend(
    cursor: &mut Cursor<'_>,
    base_dir: &Path,
) -> SpecResult<CompressionBackend> {
    match cursor.read_u8()? {
        0 => Ok(CompressionBackend::Zpaq {
            method: decode_zpaq_method_spec(cursor)?,
            threads: std::num::NonZeroUsize::new(
                usize::try_from(cursor.read_u64()?)
                    .map_err(|_| SpecError::new("zpaq compression threads exceeds usize::MAX"))?,
            )
            .ok_or_else(|| SpecError::new("zpaq compression threads must be >= 1"))?,
        }),
        #[cfg(feature = "backend-rwkv")]
        1 => Ok(CompressionBackend::Rwkv7 {
            method: decode_rwkv_method_spec(cursor)?,
            coder: decode_coder(cursor.read_u8()?)?,
        }),
        #[cfg(not(feature = "backend-rwkv"))]
        1 => Err(SpecError::new(
            "binary rwkv compression backend requires the 'backend-rwkv' feature",
        )),
        2 => Ok(CompressionBackend::Rate {
            coder: decode_coder(cursor.read_u8()?)?,
            framing: decode_framing(cursor.read_u8()?)?,
            rate_backend: decode_rate_backend(cursor, base_dir)?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown compression backend tag '{tag}'"
        ))),
    }
}

fn encode_particle_spec(out: &mut Vec<u8>, spec: &crate::api::ParticleSpec) {
    push_u64(out, spec.num_particles as u64);
    push_u64(out, spec.context_window as u64);
    push_u64(out, spec.unroll_steps as u64);
    push_u64(out, spec.num_cells as u64);
    push_u64(out, spec.cell_dim as u64);
    push_u64(out, spec.num_rules as u64);
    push_u64(out, spec.selector_hidden as u64);
    push_u64(out, spec.rule_hidden as u64);
    push_u64(out, spec.noise_dim as u64);
    push_bool(out, spec.deterministic);
    push_bool(out, spec.enable_noise);
    push_f64(out, spec.noise_scale);
    push_u64(out, spec.noise_anneal_steps as u64);
    push_f64(out, spec.learning_rate_readout);
    push_f64(out, spec.learning_rate_selector);
    push_f64(out, spec.learning_rate_rule);
    push_u64(out, spec.bptt_depth as u64);
    push_f64(out, spec.optimizer_momentum);
    push_f64(out, spec.grad_clip);
    push_f64(out, spec.state_clip);
    push_f64(out, spec.forget_lambda);
    push_f64(out, spec.resample_threshold);
    push_f64(out, spec.mutate_fraction);
    push_f64(out, spec.mutate_scale);
    push_bool(out, spec.mutate_model_params);
    push_u64(out, spec.diagnostics_interval as u64);
    push_f64(out, spec.min_prob);
    push_u64(out, spec.seed);
}

fn decode_particle_spec(cursor: &mut Cursor<'_>) -> SpecResult<crate::api::ParticleSpec> {
    Ok(crate::api::ParticleSpec {
        num_particles: cursor.read_u64()? as usize,
        context_window: cursor.read_u64()? as usize,
        unroll_steps: cursor.read_u64()? as usize,
        num_cells: cursor.read_u64()? as usize,
        cell_dim: cursor.read_u64()? as usize,
        num_rules: cursor.read_u64()? as usize,
        selector_hidden: cursor.read_u64()? as usize,
        rule_hidden: cursor.read_u64()? as usize,
        noise_dim: cursor.read_u64()? as usize,
        deterministic: cursor.read_bool()?,
        enable_noise: cursor.read_bool()?,
        noise_scale: cursor.read_f64()?,
        noise_anneal_steps: cursor.read_u64()? as usize,
        learning_rate_readout: cursor.read_f64()?,
        learning_rate_selector: cursor.read_f64()?,
        learning_rate_rule: cursor.read_f64()?,
        bptt_depth: cursor.read_u64()? as usize,
        optimizer_momentum: cursor.read_f64()?,
        grad_clip: cursor.read_f64()?,
        state_clip: cursor.read_f64()?,
        forget_lambda: cursor.read_f64()?,
        resample_threshold: cursor.read_f64()?,
        mutate_fraction: cursor.read_f64()?,
        mutate_scale: cursor.read_f64()?,
        mutate_model_params: cursor.read_bool()?,
        diagnostics_interval: cursor.read_u64()? as usize,
        min_prob: cursor.read_f64()?,
        seed: cursor.read_u64()?,
    })
}

fn encode_environment_spec(spec: &EnvironmentSpec, out: &mut Vec<u8>) {
    match spec {
        EnvironmentSpec::Builtin { builtin } => {
            out.push(0);
            out.push(builtin_environment_tag(*builtin));
        }
        #[cfg(feature = "vm")]
        EnvironmentSpec::NyxVm(vm) => {
            out.push(1);
            push_string(out, &vm.firecracker_config_asset);
            push_string(out, &vm.instance_id);
            push_string(out, &vm.shared_region_name);
            push_u64(out, vm.shared_region_size as u64);
            out.push(shared_memory_policy_tag(vm.shared_memory_policy));
            push_u64(out, vm.step_timeout_ms);
            push_u64(out, vm.boot_timeout_ms);
            push_u64(out, vm.episode_steps as u64);
            push_i64(out, vm.step_cost);
            out.push(vm_observation_policy_tag(vm.observation_policy));
            push_u64(out, vm.observation_bits as u64);
            push_u64(out, vm.observation_stream_len as u64);
            out.push(vm_observation_stream_mode_tag(vm.observation_stream_mode));
            out.push(vm.observation_pad_byte);
            push_u64(out, vm.reward_bits as u64);
            encode_vm_reward_policy(&vm.reward_policy, out);
            match &vm.reward_shaping {
                Some(shape) => {
                    out.push(1);
                    encode_vm_reward_shaping(shape, out);
                }
                None => out.push(0),
            }
            encode_vm_action_source(&vm.action_source, out);
            match &vm.action_filter {
                Some(filter) => {
                    out.push(1);
                    encode_vm_action_filter(filter, out);
                }
                None => out.push(0),
            }
            push_string(out, &vm.action_prefix);
            push_string(out, &vm.action_suffix);
            push_string(out, &vm.obs_prefix);
            push_string(out, &vm.rew_prefix);
            push_string(out, &vm.done_prefix);
            push_string(out, &vm.data_prefix);
            out.push(vm_payload_encoding_tag(vm.wire_encoding));
            encode_rate_backend(out, &vm.stats_backend);
            match &vm.trace {
                Some(trace) => {
                    out.push(1);
                    encode_vm_trace(trace, out);
                }
                None => out.push(0),
            }
            push_bool(out, vm.debug_mode);
            push_option_string(out, vm.crash_log.as_deref());
        }
    }
}

fn decode_environment_spec(
    cursor: &mut Cursor<'_>,
    base_dir: &Path,
) -> SpecResult<EnvironmentSpec> {
    #[cfg(not(feature = "vm"))]
    let _ = base_dir;
    match cursor.read_u8()? {
        0 => Ok(EnvironmentSpec::Builtin {
            builtin: decode_builtin_environment(cursor.read_u8()?)?,
        }),
        #[cfg(feature = "vm")]
        1 => {
            let baseline = cursor.read_string()?;
            let instance_id = cursor.read_string()?;
            let shared_region_name = cursor.read_string()?;
            let shared_region_size = cursor.read_u64()? as usize;
            let shared_memory_policy = decode_shared_memory_policy(cursor.read_u8()?)?;
            let step_timeout_ms = cursor.read_u64()?;
            let boot_timeout_ms = cursor.read_u64()?;
            let episode_steps = cursor.read_u64()? as usize;
            let step_cost = cursor.read_i64()?;
            let observation_policy = decode_vm_observation_policy(cursor.read_u8()?)?;
            let observation_bits = cursor.read_u64()? as usize;
            let observation_stream_len = cursor.read_u64()? as usize;
            let observation_stream_mode = decode_vm_observation_stream_mode(cursor.read_u8()?)?;
            let observation_pad_byte = cursor.read_u8()?;
            let reward_bits = cursor.read_u64()? as usize;
            let reward_policy = decode_vm_reward_policy(cursor)?;
            let reward_shaping = if cursor.read_u8()? == 1 {
                Some(decode_vm_reward_shaping(cursor)?)
            } else {
                None
            };
            let action_source = decode_vm_action_source(cursor)?;
            let action_filter = if cursor.read_u8()? == 1 {
                Some(decode_vm_action_filter(cursor)?)
            } else {
                None
            };
            let action_prefix = cursor.read_string()?;
            let action_suffix = cursor.read_string()?;
            let obs_prefix = cursor.read_string()?;
            let rew_prefix = cursor.read_string()?;
            let done_prefix = cursor.read_string()?;
            let data_prefix = cursor.read_string()?;
            let wire_encoding = decode_vm_payload_encoding(cursor.read_u8()?)?;
            let stats_backend = decode_rate_backend(cursor, base_dir)?;
            let trace = if cursor.read_u8()? == 1 {
                Some(decode_vm_trace(cursor)?)
            } else {
                None
            };
            let debug_mode = cursor.read_bool()?;
            let crash_log = if cursor.has_remaining() {
                cursor.read_option_string()?
            } else {
                None
            };
            Ok(EnvironmentSpec::NyxVm(VmEnvironmentSpec {
                firecracker_config_asset: baseline,
                instance_id,
                shared_region_name,
                shared_region_size,
                shared_memory_policy,
                step_timeout_ms,
                boot_timeout_ms,
                episode_steps,
                step_cost,
                observation_policy,
                observation_bits,
                observation_stream_len,
                observation_stream_mode,
                observation_pad_byte,
                reward_bits,
                reward_policy,
                reward_shaping,
                action_source,
                action_filter,
                action_prefix,
                action_suffix,
                obs_prefix,
                rew_prefix,
                done_prefix,
                data_prefix,
                wire_encoding,
                stats_backend,
                trace,
                debug_mode,
                crash_log,
            }))
        }
        #[cfg(not(feature = "vm"))]
        1 => Err(SpecError::new(
            "binary nyx_vm environment requires the 'vm' feature",
        )),
        tag => Err(SpecError::new(format!("unknown environment tag '{tag}'"))),
    }
}

fn encode_interface_spec(spec: &PlannerInterfaceSpec, out: &mut Vec<u8>) {
    push_u64(out, spec.observation_bits as u64);
    push_u64(out, spec.observation_stream_len as u64);
    out.push(observation_key_mode_tag(spec.observation_key_mode));
    push_u64(out, spec.reward_bits as u64);
    push_u64(out, spec.agent_actions.get() as u64);
}

fn decode_interface_spec(cursor: &mut Cursor<'_>) -> SpecResult<PlannerInterfaceSpec> {
    let observation_bits = decode_usize_field(cursor, "interface.observation_bits")?;
    let observation_stream_len = decode_usize_field(cursor, "interface.observation_stream_len")?;
    let observation_key_mode = decode_observation_key_mode(cursor.read_u8()?)?;
    let reward_bits = decode_usize_field(cursor, "interface.reward_bits")?;
    let agent_actions_raw = decode_usize_field(cursor, "interface.agent_actions")?;
    let agent_actions = crate::aixi::common::ActionAlphabet::try_from_usize(agent_actions_raw)
        .map_err(|_| SpecError::new("binary interface.agent_actions must be >= 1"))?;
    Ok(PlannerInterfaceSpec {
        observation_bits,
        observation_stream_len,
        observation_key_mode,
        reward_bits,
        agent_actions,
    })
}

#[cfg(feature = "tuner")]
fn encode_tune_interface_spec(spec: &TunePlannerInterfaceSpec, out: &mut Vec<u8>) {
    push_u64(out, spec.observation_bits as u64);
    push_u64(out, spec.observation_stream_len as u64);
    out.push(observation_key_mode_tag(spec.observation_key_mode));
    push_u64(out, spec.reward_bits as u64);
    push_u64(out, spec.agent_actions.get() as u64);
}

#[cfg(feature = "tuner")]
fn decode_tune_interface_spec(cursor: &mut Cursor<'_>) -> SpecResult<TunePlannerInterfaceSpec> {
    let observation_bits = decode_usize_field(cursor, "interface.observation_bits")?;
    let observation_stream_len = decode_usize_field(cursor, "interface.observation_stream_len")?;
    let observation_key_mode = decode_observation_key_mode(cursor.read_u8()?)?;
    let reward_bits = decode_usize_field(cursor, "interface.reward_bits")?;
    let agent_actions_raw = decode_usize_field(cursor, "interface.agent_actions")?;
    let agent_actions = crate::aixi::common::ActionAlphabet::try_from_usize(agent_actions_raw)
        .map_err(|_| SpecError::new("binary interface.agent_actions must be >= 1"))?;
    Ok(TunePlannerInterfaceSpec {
        observation_bits,
        observation_stream_len,
        observation_key_mode,
        reward_bits,
        agent_actions,
    })
}

fn decode_usize_field(cursor: &mut Cursor<'_>, label: &str) -> SpecResult<usize> {
    let raw = cursor.read_u64()?;
    usize::try_from(raw).map_err(|_| SpecError::new(format!("{label} exceeds usize::MAX")))
}

fn encode_controller_spec(spec: &ControllerSpec, out: &mut Vec<u8>) {
    match spec {
        ControllerSpec::McAixi(inner) => {
            out.push(0);
            encode_rate_backend(out, &inner.predictor);
            push_u64(out, inner.agent_horizon as u64);
            push_u64(out, inner.num_simulations as u64);
            encode_mcts_strategy(out, inner.mcts_strategy);
            push_f64(out, inner.exploration_exploitation_ratio);
            push_f64(out, inner.discount_gamma);
        }
        ControllerSpec::AiqiDiscounted(inner) => {
            out.push(1);
            encode_rate_backend(out, &inner.predictor);
            push_f64(out, inner.discount_gamma);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_u64(out, inner.augmentation_period as u64);
            push_option_u64(out, inner.history_prune_keep_steps.map(|n| n as u64));
            push_f64(out, inner.baseline_exploration);
        }
        ControllerSpec::AiqiWarmstartExactJh(inner) => {
            out.push(2);
            encode_rate_backend(out, &inner.predictor);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_u64(out, inner.label_phase_period as u64);
            push_string(out, &inner.teacher_dataset_asset);
            push_u64(out, inner.planner_simulations_per_step as u64);
        }
    }
}

fn decode_controller_spec(cursor: &mut Cursor<'_>, base_dir: &Path) -> SpecResult<ControllerSpec> {
    match cursor.read_u8()? {
        0 => Ok(ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: decode_rate_backend(cursor, base_dir)?,
            agent_horizon: cursor.read_u64()? as usize,
            num_simulations: cursor.read_u64()? as usize,
            mcts_strategy: decode_mcts_strategy(cursor)?,
            exploration_exploitation_ratio: cursor.read_f64()?,
            discount_gamma: cursor.read_f64()?,
        })),
        1 => Ok(ControllerSpec::AiqiDiscounted(
            AiqiDiscountedControllerSpec {
                predictor: decode_rate_backend(cursor, base_dir)?,
                discount_gamma: cursor.read_f64()?,
                return_horizon: cursor.read_u64()? as usize,
                return_bins: cursor.read_u64()? as usize,
                augmentation_period: cursor.read_u64()? as usize,
                history_prune_keep_steps: cursor.read_option_u64()?.map(|n| n as usize),
                baseline_exploration: cursor.read_f64()?,
            },
        )),
        2 => Ok(ControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhControllerSpec {
                predictor: decode_rate_backend(cursor, base_dir)?,
                return_horizon: cursor.read_u64()? as usize,
                return_bins: cursor.read_u64()? as usize,
                label_phase_period: cursor.read_u64()? as usize,
                teacher_dataset_asset: cursor.read_string()?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
            },
        )),
        tag => Err(SpecError::new(format!("unknown controller tag '{tag}'"))),
    }
}

fn encode_mcts_strategy(out: &mut Vec<u8>, strategy: crate::aixi::common::MctsStrategy) {
    use crate::aixi::common::MctsStrategy;
    match strategy {
        MctsStrategy::RhoUct => out.push(0),
        MctsStrategy::ParallelUct {
            workers,
            bu_uct_m_max,
        } => {
            out.push(1);
            push_u64(out, workers.get() as u64);
            push_option_f64(out, bu_uct_m_max);
        }
    }
}

fn decode_mcts_strategy(cursor: &mut Cursor<'_>) -> SpecResult<crate::aixi::common::MctsStrategy> {
    use crate::aixi::common::MctsStrategy;
    use std::num::NonZeroUsize;
    match cursor.read_u8()? {
        0 => Ok(MctsStrategy::RhoUct),
        1 => {
            let workers_raw = cursor.read_u64()?;
            let workers = NonZeroUsize::new(workers_raw as usize).ok_or_else(|| {
                SpecError::new("binary mcts_strategy parallel_uct workers must be >= 1")
            })?;
            Ok(MctsStrategy::ParallelUct {
                workers,
                bu_uct_m_max: cursor.read_option_f64()?,
            })
        }
        tag => Err(SpecError::new(format!("unknown MCTS strategy tag '{tag}'"))),
    }
}

fn encode_runtime_spec(spec: &PlannerRuntimeSpec, out: &mut Vec<u8>) {
    push_option_u64(out, spec.random_seed);
    push_option_u64(out, spec.learn_cycles.map(|n| n as u64));
    push_option_u64(out, spec.eval_cycles.map(|n| n as u64));
    push_u64(out, spec.terminate_lifetime as u64);
    push_u64(out, spec.log_every as u64);
    push_bool(out, spec.perf);
    push_bool(out, spec.vm_perf_only);
    push_f64(out, spec.explore_epsilon);
    push_f64(out, spec.explore_gamma);
}

fn decode_runtime_spec(cursor: &mut Cursor<'_>) -> SpecResult<PlannerRuntimeSpec> {
    Ok(PlannerRuntimeSpec {
        random_seed: cursor.read_option_u64()?,
        learn_cycles: cursor.read_option_u64()?.map(|n| n as usize),
        eval_cycles: cursor.read_option_u64()?.map(|n| n as usize),
        terminate_lifetime: cursor.read_u64()? as usize,
        log_every: cursor.read_u64()? as usize,
        perf: cursor.read_bool()?,
        vm_perf_only: cursor.read_bool()?,
        explore_epsilon: cursor.read_f64()?,
        explore_gamma: cursor.read_f64()?,
    })
}

#[cfg(feature = "tuner")]
fn encode_tune_bounds(bounds: &TuneBoundsSpec, out: &mut Vec<u8>) {
    push_string_list(out, &bounds.allowed_backends);
    push_string_list(out, &bounds.forbidden_backends);
    push_u64(out, bounds.parameter_ranges.len() as u64);
    for range in &bounds.parameter_ranges {
        push_string(out, &range.parameter);
        push_f64(out, range.min);
        push_f64(out, range.max);
    }
    push_u64(out, bounds.max_experts as u64);
    push_u64(out, bounds.max_mixture_nesting_depth as u64);
    push_option_u64(out, bounds.min_experts.map(|n| n as u64));
    match bounds.allow_duplicate_experts {
        Some(value) => {
            out.push(1);
            push_bool(out, value);
        }
        None => out.push(0),
    }
    push_string_list(out, &bounds.required_experts);
    push_u64(out, bounds.forbidden_expert_pairs.len() as u64);
    for (left, right) in &bounds.forbidden_expert_pairs {
        push_string(out, left);
        push_string(out, right);
    }
}

#[cfg(feature = "tuner")]
fn decode_tune_bounds(cursor: &mut Cursor<'_>) -> SpecResult<TuneBoundsSpec> {
    let allowed_backends = cursor.read_string_list()?;
    let forbidden_backends = cursor.read_string_list()?;
    let range_len = cursor.read_u64()? as usize;
    let mut parameter_ranges = Vec::with_capacity(range_len);
    for _ in 0..range_len {
        parameter_ranges.push(TuneParameterRangeSpec {
            parameter: cursor.read_string()?,
            min: cursor.read_f64()?,
            max: cursor.read_f64()?,
        });
    }
    let max_experts = cursor.read_u64()? as usize;
    let max_mixture_nesting_depth = cursor.read_u64()? as usize;
    let min_experts = cursor.read_option_u64()?.map(|n| n as usize);
    let allow_duplicate_experts = if cursor.read_u8()? == 1 {
        Some(cursor.read_bool()?)
    } else {
        None
    };
    let required_experts = cursor.read_string_list()?;
    let pair_len = cursor.read_u64()? as usize;
    let mut forbidden_expert_pairs = Vec::with_capacity(pair_len);
    for _ in 0..pair_len {
        forbidden_expert_pairs.push((cursor.read_string()?, cursor.read_string()?));
    }
    Ok(TuneBoundsSpec {
        allowed_backends,
        forbidden_backends,
        parameter_ranges,
        max_experts,
        max_mixture_nesting_depth,
        min_experts,
        allow_duplicate_experts,
        required_experts,
        forbidden_expert_pairs,
    })
}

#[cfg(feature = "tuner")]
fn encode_tune_controller(spec: &TuneControllerSpec, out: &mut Vec<u8>) {
    match spec {
        TuneControllerSpec::AnnealedHillClimbing(inner) => {
            out.push(tune_controller_kind_tag(
                TuneControllerKind::AnnealedHillClimbing,
            ));
            push_u64(out, inner.max_mutation_radius as u64);
        }
        TuneControllerSpec::McAixiFacCtw(inner) => {
            out.push(tune_controller_kind_tag(TuneControllerKind::McAixiFacCtw));
            encode_tune_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
        }
        TuneControllerSpec::AiqiDiscounted(inner) => {
            out.push(tune_controller_kind_tag(TuneControllerKind::AiqiDiscounted));
            encode_tune_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
            push_u64(out, inner.return_horizon as u64);
            push_u64(out, inner.return_bins as u64);
            push_f64(out, inner.discount_factor);
            push_f64(out, inner.min_improvement);
            push_f64(out, inner.max_improvement);
        }
        TuneControllerSpec::AiqiWarmstartExactJh(inner) => {
            out.push(tune_controller_kind_tag(
                TuneControllerKind::AiqiWarmstartExactJh,
            ));
            encode_tune_interface_spec(&inner.interface, out);
            push_u64(out, inner.planner_simulations_per_step as u64);
            push_u64(out, inner.return_horizon as u64);
            push_string(out, &inner.warmstart_teacher_dataset_asset);
            push_u64(out, inner.label_phase_period as u64);
        }
    }
}

#[cfg(feature = "tuner")]
fn decode_tune_controller(cursor: &mut Cursor<'_>) -> SpecResult<TuneControllerSpec> {
    match decode_tune_controller_kind(cursor.read_u8()?)? {
        TuneControllerKind::AnnealedHillClimbing => Ok(TuneControllerSpec::AnnealedHillClimbing(
            AnnealedHillClimbingTuneControllerSpec {
                max_mutation_radius: cursor.read_u64()? as usize,
            },
        )),
        TuneControllerKind::McAixiFacCtw => Ok(TuneControllerSpec::McAixiFacCtw(
            McAixiFacCtwTuneControllerSpec {
                interface: decode_tune_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
            },
        )),
        TuneControllerKind::AiqiDiscounted => Ok(TuneControllerSpec::AiqiDiscounted(
            AiqiDiscountedTuneControllerSpec {
                interface: decode_tune_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
                return_horizon: cursor.read_u64()? as usize,
                return_bins: cursor.read_u64()? as usize,
                discount_factor: cursor.read_f64()?,
                min_improvement: cursor.read_f64()?,
                max_improvement: cursor.read_f64()?,
            },
        )),
        TuneControllerKind::AiqiWarmstartExactJh => Ok(TuneControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhTuneControllerSpec {
                interface: decode_tune_interface_spec(cursor)?,
                planner_simulations_per_step: cursor.read_u64()? as usize,
                return_horizon: cursor.read_u64()? as usize,
                warmstart_teacher_dataset_asset: cursor.read_string()?,
                label_phase_period: cursor.read_u64()? as usize,
            },
        )),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_reward_policy(policy: &VmRewardPolicySpec, out: &mut Vec<u8>) {
    match policy {
        VmRewardPolicySpec::FromGuest => out.push(0),
        VmRewardPolicySpec::Pattern {
            pattern,
            base_reward,
            bonus_reward,
        } => {
            out.push(1);
            push_string(out, pattern);
            push_i64(out, *base_reward);
            push_i64(out, *bonus_reward);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_reward_policy(cursor: &mut Cursor<'_>) -> SpecResult<VmRewardPolicySpec> {
    match cursor.read_u8()? {
        0 => Ok(VmRewardPolicySpec::FromGuest),
        1 => Ok(VmRewardPolicySpec::Pattern {
            pattern: cursor.read_string()?,
            base_reward: cursor.read_i64()?,
            bonus_reward: cursor.read_i64()?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown vm reward policy tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_reward_shaping(spec: &VmRewardShapingSpec, out: &mut Vec<u8>) {
    match spec {
        VmRewardShapingSpec::EntropyReduction {
            baseline_asset,
            scale,
            crash_bonus,
            timeout_bonus,
        } => {
            out.push(0);
            push_string(out, baseline_asset);
            push_f64(out, *scale);
            push_option_i64(out, *crash_bonus);
            push_option_i64(out, *timeout_bonus);
        }
        VmRewardShapingSpec::TraceEntropy { scale, normalize } => {
            out.push(1);
            push_f64(out, *scale);
            push_bool(out, *normalize);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_reward_shaping(cursor: &mut Cursor<'_>) -> SpecResult<VmRewardShapingSpec> {
    match cursor.read_u8()? {
        0 => Ok(VmRewardShapingSpec::EntropyReduction {
            baseline_asset: cursor.read_string()?,
            scale: cursor.read_f64()?,
            crash_bonus: cursor.read_option_i64()?,
            timeout_bonus: cursor.read_option_i64()?,
        }),
        1 => Ok(VmRewardShapingSpec::TraceEntropy {
            scale: cursor.read_f64()?,
            normalize: cursor.read_bool()?,
        }),
        tag => Err(SpecError::new(format!(
            "unknown vm reward shaping tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_action_source(spec: &VmRuntimeActionSourceSpec, out: &mut Vec<u8>) {
    match spec {
        VmRuntimeActionSourceSpec::Literal {
            names,
            payloads,
            encoding,
        } => {
            out.push(0);
            out.push(vm_payload_encoding_tag(*encoding));
            push_u64(out, payloads.len() as u64);
            for (idx, payload) in payloads.iter().enumerate() {
                push_option_string(out, names.get(idx).cloned().flatten().as_deref());
                push_string(out, payload);
            }
        }
        VmRuntimeActionSourceSpec::Fuzz {
            seeds,
            encoding,
            mutators,
            min_len,
            max_len,
            dictionary,
            rng_seed,
        } => {
            out.push(1);
            out.push(vm_payload_encoding_tag(*encoding));
            push_string_list(out, seeds);
            push_u64(out, mutators.len() as u64);
            for mutator in mutators {
                out.push(vm_fuzz_mutator_tag(*mutator));
            }
            push_u64(out, *min_len as u64);
            push_u64(out, *max_len as u64);
            push_string_list(out, dictionary);
            push_u64(out, *rng_seed);
        }
    }
}

#[cfg(feature = "vm")]
fn decode_vm_action_source(cursor: &mut Cursor<'_>) -> SpecResult<VmRuntimeActionSourceSpec> {
    match cursor.read_u8()? {
        0 => {
            let encoding = decode_vm_payload_encoding(cursor.read_u8()?)?;
            let len = cursor.read_u64()? as usize;
            let mut names = Vec::with_capacity(len);
            let mut payloads = Vec::with_capacity(len);
            for _ in 0..len {
                names.push(cursor.read_option_string()?);
                payloads.push(cursor.read_string()?);
            }
            Ok(VmRuntimeActionSourceSpec::Literal {
                names,
                payloads,
                encoding,
            })
        }
        1 => {
            let encoding = decode_vm_payload_encoding(cursor.read_u8()?)?;
            let seeds = cursor.read_string_list()?;
            let mutators_len = cursor.read_u64()? as usize;
            let mut mutators = Vec::with_capacity(mutators_len);
            for _ in 0..mutators_len {
                mutators.push(decode_vm_fuzz_mutator(cursor.read_u8()?)?);
            }
            Ok(VmRuntimeActionSourceSpec::Fuzz {
                seeds,
                encoding,
                mutators,
                min_len: cursor.read_u64()? as usize,
                max_len: cursor.read_u64()? as usize,
                dictionary: cursor.read_string_list()?,
                rng_seed: cursor.read_u64()?,
            })
        }
        tag => Err(SpecError::new(format!(
            "unknown vm action source tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn encode_vm_action_filter(spec: &VmActionFilterSpec, out: &mut Vec<u8>) {
    push_option_f64(out, spec.min_entropy);
    push_option_f64(out, spec.max_entropy);
    push_option_f64(out, spec.min_intrinsic_dependence);
    push_option_f64(out, spec.min_novelty);
    push_option_string(out, spec.novelty_prior_asset.as_deref());
    push_option_i64(out, spec.reject_reward);
}

#[cfg(feature = "vm")]
fn decode_vm_action_filter(cursor: &mut Cursor<'_>) -> SpecResult<VmActionFilterSpec> {
    Ok(VmActionFilterSpec {
        min_entropy: cursor.read_option_f64()?,
        max_entropy: cursor.read_option_f64()?,
        min_intrinsic_dependence: cursor.read_option_f64()?,
        min_novelty: cursor.read_option_f64()?,
        novelty_prior_asset: cursor.read_option_string()?,
        reject_reward: cursor.read_option_i64()?,
    })
}

#[cfg(feature = "vm")]
fn encode_vm_trace(spec: &VmTraceSpec, out: &mut Vec<u8>) {
    push_option_string(out, spec.shared_region_name.as_deref());
    push_u64(out, spec.max_bytes as u64);
    push_bool(out, spec.reset_on_episode);
}

#[cfg(feature = "vm")]
fn decode_vm_trace(cursor: &mut Cursor<'_>) -> SpecResult<VmTraceSpec> {
    Ok(VmTraceSpec {
        shared_region_name: cursor.read_option_string()?,
        max_bytes: cursor.read_u64()? as usize,
        reset_on_episode: cursor.read_bool()?,
    })
}

pub(super) fn builtin_environment_name(env: BuiltinEnvironmentSpec) -> &'static str {
    match env {
        BuiltinEnvironmentSpec::TunerBridge => "tuner_bridge",
        BuiltinEnvironmentSpec::CoinFlip => "coin_flip",
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => "biased_rock_paper_scissor",
        BuiltinEnvironmentSpec::KuhnPoker => "kuhn_poker",
        BuiltinEnvironmentSpec::ExtendedTiger => "extended_tiger",
        BuiltinEnvironmentSpec::TicTacToe => "tic_tac_toe",
        BuiltinEnvironmentSpec::Blackjack => "blackjack",
        BuiltinEnvironmentSpec::Platformer => "platformer",
    }
}

fn coder_tag(coder: crate::coders::CoderType) -> u8 {
    match coder {
        crate::coders::CoderType::AC => 0,
        crate::coders::CoderType::RANS => 1,
    }
}

fn decode_coder(tag: u8) -> SpecResult<crate::coders::CoderType> {
    match tag {
        0 => Ok(crate::coders::CoderType::AC),
        1 => Ok(crate::coders::CoderType::RANS),
        _ => Err(SpecError::new(format!("unknown coder tag '{tag}'"))),
    }
}

fn framing_tag(framing: crate::compression::FramingMode) -> u8 {
    match framing {
        crate::compression::FramingMode::Raw => 0,
        crate::compression::FramingMode::Framed => 1,
    }
}

fn decode_framing(tag: u8) -> SpecResult<crate::compression::FramingMode> {
    match tag {
        0 => Ok(crate::compression::FramingMode::Raw),
        1 => Ok(crate::compression::FramingMode::Framed),
        _ => Err(SpecError::new(format!("unknown framing tag '{tag}'"))),
    }
}

fn mixture_kind_tag(kind: crate::api::MixtureKind) -> u8 {
    match kind {
        crate::api::MixtureKind::Bayes => 0,
        crate::api::MixtureKind::FadingBayes => 1,
        crate::api::MixtureKind::Switching => 2,
        crate::api::MixtureKind::Convex => 3,
        crate::api::MixtureKind::Mdl => 4,
        crate::api::MixtureKind::Neural => 5,
    }
}

fn decode_mixture_kind(tag: u8) -> SpecResult<crate::api::MixtureKind> {
    match tag {
        0 => Ok(crate::api::MixtureKind::Bayes),
        1 => Ok(crate::api::MixtureKind::FadingBayes),
        2 => Ok(crate::api::MixtureKind::Switching),
        3 => Ok(crate::api::MixtureKind::Convex),
        4 => Ok(crate::api::MixtureKind::Mdl),
        5 => Ok(crate::api::MixtureKind::Neural),
        _ => Err(SpecError::new(format!("unknown mixture kind tag '{tag}'"))),
    }
}

fn mixture_schedule_tag(schedule: crate::api::MixtureScheduleMode) -> u8 {
    match schedule {
        crate::api::MixtureScheduleMode::Default => 0,
        crate::api::MixtureScheduleMode::Theorem => 1,
    }
}

fn decode_mixture_schedule(tag: u8) -> SpecResult<crate::api::MixtureScheduleMode> {
    match tag {
        0 => Ok(crate::api::MixtureScheduleMode::Default),
        1 => Ok(crate::api::MixtureScheduleMode::Theorem),
        _ => Err(SpecError::new(format!(
            "unknown mixture schedule tag '{tag}'"
        ))),
    }
}

fn calibration_context_tag(context: crate::api::CalibrationContextKind) -> u8 {
    match context {
        crate::api::CalibrationContextKind::Global => 0,
        crate::api::CalibrationContextKind::ByteClass => 1,
        crate::api::CalibrationContextKind::Text => 2,
        crate::api::CalibrationContextKind::Repeat => 3,
        crate::api::CalibrationContextKind::TextRepeat => 4,
    }
}

fn decode_calibration_context(tag: u8) -> SpecResult<crate::api::CalibrationContextKind> {
    match tag {
        0 => Ok(crate::api::CalibrationContextKind::Global),
        1 => Ok(crate::api::CalibrationContextKind::ByteClass),
        2 => Ok(crate::api::CalibrationContextKind::Text),
        3 => Ok(crate::api::CalibrationContextKind::Repeat),
        4 => Ok(crate::api::CalibrationContextKind::TextRepeat),
        _ => Err(SpecError::new(format!(
            "unknown calibration context tag '{tag}'"
        ))),
    }
}

fn builtin_environment_tag(env: BuiltinEnvironmentSpec) -> u8 {
    match env {
        BuiltinEnvironmentSpec::TunerBridge => 8,
        BuiltinEnvironmentSpec::CoinFlip => 0,
        BuiltinEnvironmentSpec::ExtendedTiger => 2,
        BuiltinEnvironmentSpec::TicTacToe => 3,
        BuiltinEnvironmentSpec::BiasedRockPaperScissor => 4,
        BuiltinEnvironmentSpec::KuhnPoker => 5,
        BuiltinEnvironmentSpec::Blackjack => 6,
        BuiltinEnvironmentSpec::Platformer => 7,
    }
}

fn decode_builtin_environment(tag: u8) -> SpecResult<BuiltinEnvironmentSpec> {
    match tag {
        8 => Ok(BuiltinEnvironmentSpec::TunerBridge),
        0 => Ok(BuiltinEnvironmentSpec::CoinFlip),
        1 => Err(SpecError::new(
            "builtin environment tag '1' (ctw_test) is no longer supported",
        )),
        2 => Ok(BuiltinEnvironmentSpec::ExtendedTiger),
        3 => Ok(BuiltinEnvironmentSpec::TicTacToe),
        4 => Ok(BuiltinEnvironmentSpec::BiasedRockPaperScissor),
        5 => Ok(BuiltinEnvironmentSpec::KuhnPoker),
        6 => Ok(BuiltinEnvironmentSpec::Blackjack),
        7 => Ok(BuiltinEnvironmentSpec::Platformer),
        _ => Err(SpecError::new(format!(
            "unknown builtin environment tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
pub(super) fn shared_memory_policy_name(policy: SharedMemoryPolicySpec) -> &'static str {
    match policy {
        SharedMemoryPolicySpec::Preserve => "preserve",
        SharedMemoryPolicySpec::Snapshot => "snapshot",
    }
}

#[cfg(feature = "vm")]
fn shared_memory_policy_tag(policy: SharedMemoryPolicySpec) -> u8 {
    match policy {
        SharedMemoryPolicySpec::Preserve => 0,
        SharedMemoryPolicySpec::Snapshot => 1,
    }
}

#[cfg(feature = "vm")]
fn decode_shared_memory_policy(tag: u8) -> SpecResult<SharedMemoryPolicySpec> {
    match tag {
        0 => Ok(SharedMemoryPolicySpec::Preserve),
        1 => Ok(SharedMemoryPolicySpec::Snapshot),
        _ => Err(SpecError::new(format!(
            "unknown shared memory policy tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
pub(super) fn vm_observation_policy_name(policy: VmObservationPolicySpec) -> &'static str {
    match policy {
        VmObservationPolicySpec::FromGuest => "from_guest",
        VmObservationPolicySpec::OutputHash => "output_hash",
        VmObservationPolicySpec::RawOutput => "raw_output",
        VmObservationPolicySpec::SharedMemory => "shared_memory",
    }
}

#[cfg(feature = "vm")]
fn vm_observation_policy_tag(policy: VmObservationPolicySpec) -> u8 {
    match policy {
        VmObservationPolicySpec::FromGuest => 0,
        VmObservationPolicySpec::OutputHash => 1,
        VmObservationPolicySpec::RawOutput => 2,
        VmObservationPolicySpec::SharedMemory => 3,
    }
}

#[cfg(feature = "vm")]
fn decode_vm_observation_policy(tag: u8) -> SpecResult<VmObservationPolicySpec> {
    match tag {
        0 => Ok(VmObservationPolicySpec::FromGuest),
        1 => Ok(VmObservationPolicySpec::OutputHash),
        2 => Ok(VmObservationPolicySpec::RawOutput),
        3 => Ok(VmObservationPolicySpec::SharedMemory),
        _ => Err(SpecError::new(format!(
            "unknown VM observation policy tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
pub(super) fn vm_observation_stream_mode_name(mode: VmObservationStreamModeSpec) -> &'static str {
    match mode {
        VmObservationStreamModeSpec::PadTruncate => "pad_truncate",
        VmObservationStreamModeSpec::Pad => "pad",
        VmObservationStreamModeSpec::Truncate => "truncate",
    }
}

#[cfg(feature = "vm")]
fn vm_observation_stream_mode_tag(mode: VmObservationStreamModeSpec) -> u8 {
    match mode {
        VmObservationStreamModeSpec::PadTruncate => 0,
        VmObservationStreamModeSpec::Pad => 1,
        VmObservationStreamModeSpec::Truncate => 2,
    }
}

#[cfg(feature = "vm")]
fn decode_vm_observation_stream_mode(tag: u8) -> SpecResult<VmObservationStreamModeSpec> {
    match tag {
        0 => Ok(VmObservationStreamModeSpec::PadTruncate),
        1 => Ok(VmObservationStreamModeSpec::Pad),
        2 => Ok(VmObservationStreamModeSpec::Truncate),
        _ => Err(SpecError::new(format!(
            "unknown VM observation stream mode tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
pub(super) fn vm_payload_encoding_name(encoding: VmPayloadEncodingSpec) -> &'static str {
    match encoding {
        VmPayloadEncodingSpec::Utf8 => "utf8",
        VmPayloadEncodingSpec::Hex => "hex",
    }
}

#[cfg(feature = "vm")]
fn vm_payload_encoding_tag(encoding: VmPayloadEncodingSpec) -> u8 {
    match encoding {
        VmPayloadEncodingSpec::Utf8 => 0,
        VmPayloadEncodingSpec::Hex => 1,
    }
}

#[cfg(feature = "vm")]
fn decode_vm_payload_encoding(tag: u8) -> SpecResult<VmPayloadEncodingSpec> {
    match tag {
        0 => Ok(VmPayloadEncodingSpec::Utf8),
        1 => Ok(VmPayloadEncodingSpec::Hex),
        _ => Err(SpecError::new(format!(
            "unknown VM payload encoding tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "vm")]
pub(super) fn vm_fuzz_mutator_name(mutator: VmFuzzMutatorSpec) -> &'static str {
    match mutator {
        VmFuzzMutatorSpec::FlipBit => "flip_bit",
        VmFuzzMutatorSpec::FlipByte => "flip_byte",
        VmFuzzMutatorSpec::InsertByte => "insert_byte",
        VmFuzzMutatorSpec::DeleteByte => "delete_byte",
        VmFuzzMutatorSpec::SpliceSeed => "splice_seed",
        VmFuzzMutatorSpec::ResetSeed => "reset_seed",
        VmFuzzMutatorSpec::Havoc => "havoc",
    }
}

#[cfg(feature = "vm")]
fn vm_fuzz_mutator_tag(mutator: VmFuzzMutatorSpec) -> u8 {
    match mutator {
        VmFuzzMutatorSpec::FlipBit => 0,
        VmFuzzMutatorSpec::FlipByte => 1,
        VmFuzzMutatorSpec::InsertByte => 2,
        VmFuzzMutatorSpec::DeleteByte => 3,
        VmFuzzMutatorSpec::SpliceSeed => 4,
        VmFuzzMutatorSpec::ResetSeed => 5,
        VmFuzzMutatorSpec::Havoc => 6,
    }
}

#[cfg(feature = "vm")]
fn decode_vm_fuzz_mutator(tag: u8) -> SpecResult<VmFuzzMutatorSpec> {
    match tag {
        0 => Ok(VmFuzzMutatorSpec::FlipBit),
        1 => Ok(VmFuzzMutatorSpec::FlipByte),
        2 => Ok(VmFuzzMutatorSpec::InsertByte),
        3 => Ok(VmFuzzMutatorSpec::DeleteByte),
        4 => Ok(VmFuzzMutatorSpec::SpliceSeed),
        5 => Ok(VmFuzzMutatorSpec::ResetSeed),
        6 => Ok(VmFuzzMutatorSpec::Havoc),
        _ => Err(SpecError::new(format!(
            "unknown VM fuzz mutator tag '{tag}'"
        ))),
    }
}

pub(super) fn observation_key_mode_name(mode: ObservationKeyMode) -> &'static str {
    match mode {
        ObservationKeyMode::First => "first",
        ObservationKeyMode::Last => "last",
        ObservationKeyMode::StreamHash => "stream_hash",
        ObservationKeyMode::FullStream => "full_stream",
    }
}

fn observation_key_mode_tag(mode: ObservationKeyMode) -> u8 {
    match mode {
        ObservationKeyMode::First => 0,
        ObservationKeyMode::Last => 1,
        ObservationKeyMode::StreamHash => 2,
        ObservationKeyMode::FullStream => 3,
    }
}

fn decode_observation_key_mode(tag: u8) -> SpecResult<ObservationKeyMode> {
    match tag {
        0 => Ok(ObservationKeyMode::First),
        1 => Ok(ObservationKeyMode::Last),
        2 => Ok(ObservationKeyMode::StreamHash),
        3 => Ok(ObservationKeyMode::FullStream),
        _ => Err(SpecError::new(format!(
            "unknown observation key mode tag '{tag}'"
        ))),
    }
}

#[cfg(feature = "tuner")]
fn tune_controller_kind_tag(kind: TuneControllerKind) -> u8 {
    match kind {
        TuneControllerKind::AnnealedHillClimbing => 0,
        TuneControllerKind::McAixiFacCtw => 1,
        TuneControllerKind::AiqiDiscounted => 2,
        TuneControllerKind::AiqiWarmstartExactJh => 3,
    }
}

#[cfg(feature = "tuner")]
fn decode_tune_controller_kind(tag: u8) -> SpecResult<TuneControllerKind> {
    match tag {
        0 => Ok(TuneControllerKind::AnnealedHillClimbing),
        1 => Ok(TuneControllerKind::McAixiFacCtw),
        2 => Ok(TuneControllerKind::AiqiDiscounted),
        3 => Ok(TuneControllerKind::AiqiWarmstartExactJh),
        _ => Err(SpecError::new(format!(
            "unknown tune controller tag '{tag}'"
        ))),
    }
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn push_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    push_u64(out, ((value << 1) ^ (value >> 63)) as u64);
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn push_option_string(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            push_string(out, value);
        }
        None => out.push(0),
    }
}

fn push_option_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_u64(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(feature = "vm")]
fn push_option_i64(out: &mut Vec<u8>, value: Option<i64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_i64(out, value);
        }
        None => out.push(0),
    }
}

fn push_option_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            out.push(1);
            push_f64(out, value);
        }
        None => out.push(0),
    }
}

#[cfg(any(
    feature = "tuner",
    feature = "vm",
    feature = "backend-rwkv",
    feature = "backend-mamba"
))]
fn push_string_list(out: &mut Vec<u8>, items: &[String]) {
    push_u64(out, items.len() as u64);
    for item in items {
        push_string(out, item);
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_exact(&mut self, len: usize) -> SpecResult<&'a [u8]> {
        if self.pos + len > self.bytes.len() {
            return Err(SpecError::new("unexpected end of spec document"));
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.bytes[start..self.pos])
    }

    fn read_u8(&mut self) -> SpecResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_bool(&mut self) -> SpecResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u64(&mut self) -> SpecResult<u64> {
        let mut shift = 0u32;
        let mut out = 0u64;
        loop {
            let byte = self.read_u8()?;
            out |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                return Err(SpecError::new("invalid varint in spec document"));
            }
        }
    }

    fn read_i64(&mut self) -> SpecResult<i64> {
        let value = self.read_u64()?;
        Ok(((value >> 1) as i64) ^ (-((value & 1) as i64)))
    }

    fn read_f64(&mut self) -> SpecResult<f64> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(f64::from_le_bytes(bytes))
    }

    fn read_string(&mut self) -> SpecResult<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|err| SpecError::new(err.to_string()))
    }

    fn read_option_string(&mut self) -> SpecResult<Option<String>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_string()?))
        } else {
            Ok(None)
        }
    }

    fn read_option_u64(&mut self) -> SpecResult<Option<u64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_u64()?))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "vm")]
    fn read_option_i64(&mut self) -> SpecResult<Option<i64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_i64()?))
        } else {
            Ok(None)
        }
    }

    fn read_option_f64(&mut self) -> SpecResult<Option<f64>> {
        if self.read_u8()? == 1 {
            Ok(Some(self.read_f64()?))
        } else {
            Ok(None)
        }
    }

    #[cfg(any(
        feature = "tuner",
        feature = "vm",
        feature = "backend-rwkv",
        feature = "backend-mamba"
    ))]
    fn read_string_list(&mut self) -> SpecResult<Vec<String>> {
        let len = self.read_u64()? as usize;
        let mut items = Vec::with_capacity(len);
        for _ in 0..len {
            items.push(self.read_string()?);
        }
        Ok(items)
    }

    fn has_remaining(&self) -> bool {
        self.pos < self.bytes.len()
    }
}

#[cfg(all(test, feature = "tuner"))]
mod tests {
    use super::*;

    fn encode_tune_interface(
        observation_bits: u64,
        observation_stream_len: u64,
        reward_bits: u64,
        agent_actions: u64,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u64(&mut bytes, observation_bits);
        push_u64(&mut bytes, observation_stream_len);
        bytes.push(observation_key_mode_tag(ObservationKeyMode::FullStream));
        push_u64(&mut bytes, reward_bits);
        push_u64(&mut bytes, agent_actions);
        bytes
    }

    #[test]
    fn decode_tune_interface_accepts_platform_usize_max() {
        let platform_max: u64 = usize::MAX as u64;
        let bytes = encode_tune_interface(platform_max, platform_max, platform_max, 2);
        let mut cursor = Cursor::new(&bytes);

        let decoded = decode_tune_interface_spec(&mut cursor).expect("decode tune interface");

        assert_eq!(decoded.observation_bits, usize::MAX);
        assert_eq!(decoded.observation_stream_len, usize::MAX);
        assert_eq!(decoded.reward_bits, usize::MAX);
        assert_eq!(decoded.agent_actions.get(), 2);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn decode_tune_interface_rejects_observation_bits_exceeding_usize() {
        let overflow: u64 = (u32::MAX as u64) + 1;
        let bytes = encode_tune_interface(overflow, 1, 1, 2);
        let mut cursor = Cursor::new(&bytes);

        let err = decode_tune_interface_spec(&mut cursor)
            .expect_err("overflowing observation_bits must be rejected");

        assert!(
            err.to_string()
                .contains("interface.observation_bits exceeds usize::MAX"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn decode_tune_interface_rejects_reward_bits_exceeding_usize() {
        let overflow: u64 = (u32::MAX as u64) + 1;
        let bytes = encode_tune_interface(1, 1, overflow, 2);
        let mut cursor = Cursor::new(&bytes);

        let err = decode_tune_interface_spec(&mut cursor)
            .expect_err("overflowing reward_bits must be rejected");

        assert!(
            err.to_string()
                .contains("interface.reward_bits exceeds usize::MAX"),
            "unexpected error: {err}"
        );
    }
}
