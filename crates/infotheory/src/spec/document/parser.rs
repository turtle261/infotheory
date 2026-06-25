//! JSON parsing for canonical top-level specification documents.

#[cfg(feature = "aixi")]
use super::WarmStartExactJhControllerSpec;
use super::{
    AiqiDiscountedControllerSpec, AssetBinding, ControllerSpec, EnvironmentSpec,
    McAixiControllerSpec, PlannerInterfaceSpec, PlannerRunSpec, PlannerRuntimeSpec,
    SPEC_DOCUMENT_SCHEMA_VERSION, SpecDocument, SpecError, SpecResult,
    parse_compression_backend_json, parse_rate_backend_json,
};
#[cfg(feature = "tuner")]
use super::{
    AiqiDiscountedTuneControllerSpec, AnnealedHillClimbingTuneControllerSpec,
    McAixiFacCtwTuneControllerSpec, TuneBoundsSpec, TuneControllerSpec, TuneInvalidReason,
    TuneParameterRangeSpec, TunePlannerInterfaceSpec, TuneSpec, WarmStartExactJhTuneControllerSpec,
    compression_backend_to_json_value,
};
use crate::aixi::common::{ActionAlphabet, MctsStrategy};
use crate::api::{BitOrder, BitStreamSemantics};
use std::num::NonZeroUsize;

#[cfg(feature = "vm")]
use super::{
    VmActionFilterSpec, VmEnvironmentSpec, VmRewardPolicySpec, VmRewardShapingSpec,
    VmRuntimeActionSourceSpec, VmTraceSpec,
};
use std::path::Path;

pub(super) fn parse_spec_document_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<SpecDocument> {
    let version = value["schema_version"].as_u64().unwrap_or(0);
    if version != SPEC_DOCUMENT_SCHEMA_VERSION as u64 {
        return Err(SpecError::new(format!(
            "unsupported spec document schema_version '{version}'"
        )));
    }
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("spec document kind is required"))?;
    match kind {
        "planner_run" => Ok(SpecDocument::PlannerRun(parse_planner_run_json_value(
            value, base_dir,
        )?)),
        #[cfg(feature = "tuner")]
        "tune" => Ok(SpecDocument::Tune(parse_tune_spec_json_value(
            value, base_dir,
        )?)),
        #[cfg(not(feature = "tuner"))]
        "tune" => Err(SpecError::new(
            "tune documents require infotheory built with feature 'tuner'",
        )),
        "rate_backend" => Ok(SpecDocument::RateBackend(parse_rate_backend_json(
            &value["backend"],
            base_dir,
            crate::api::MAX_MIXTURE_NESTING,
        )?)),
        "compression_backend" => Ok(SpecDocument::CompressionBackend(
            parse_compression_backend_json(
                &value["backend"],
                base_dir,
                None,
                crate::compression::FramingMode::Framed,
            )?,
        )),
        other => Err(SpecError::new(format!(
            "unknown spec document kind '{other}'"
        ))),
    }
}

fn parse_planner_run_json_value(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<PlannerRunSpec> {
    Ok(PlannerRunSpec {
        assets: parse_asset_bindings(&value["assets"])?,
        environment: parse_environment_spec(&value["environment"], base_dir)?,
        interface: parse_interface_spec(&value["interface"])?,
        controller: parse_controller_spec(&value["controller"], base_dir)?,
        runtime: parse_runtime_spec(&value["runtime"])?,
    })
}

#[cfg(feature = "tuner")]
fn parse_tune_spec_json_value(value: &serde_json::Value, base_dir: &Path) -> SpecResult<TuneSpec> {
    ensure_known_fields(
        value,
        &[
            "schema_version",
            "kind",
            "assets",
            "input_asset",
            "baseline_candidate",
            "controller",
            "bounds",
            "eval_time_limit_seconds",
            "time_budget_seconds",
            "min_throughput_bytes_per_second",
            "max_memory_bytes",
            "output_config_path",
            "seed",
            "report_path",
        ],
        "tune",
    )?;
    let baseline_candidate_value = value
        .get("baseline_candidate")
        .ok_or_else(|| SpecError::new("tune.baseline_candidate is required"))?;
    reject_tune_candidate_local_external_refs(baseline_candidate_value)?;
    let baseline_candidate = parse_compression_backend_json(
        baseline_candidate_value,
        base_dir,
        None,
        crate::compression::FramingMode::Framed,
    )?;
    ensure_tune_baseline_candidate_is_canonical_json(
        baseline_candidate_value,
        &baseline_candidate,
    )?;
    Ok(TuneSpec {
        assets: parse_tune_asset_bindings(
            value
                .get("assets")
                .ok_or_else(|| SpecError::new("tune.assets is required"))?,
        )?,
        input_asset: required_string(&value["input_asset"], "input_asset")?,
        baseline_candidate,
        controller: parse_tune_controller_spec(&value["controller"])?,
        bounds: parse_tune_bounds_spec(&value["bounds"])?,
        eval_time_limit_seconds: required_f64(
            &value["eval_time_limit_seconds"],
            "eval_time_limit_seconds",
        )?,
        time_budget_seconds: required_f64(&value["time_budget_seconds"], "time_budget_seconds")?,
        min_throughput_bytes_per_second: required_f64(
            &value["min_throughput_bytes_per_second"],
            "min_throughput_bytes_per_second",
        )?,
        max_memory_bytes: required_u64(&value["max_memory_bytes"], "max_memory_bytes")?,
        output_config_path: required_string(&value["output_config_path"], "output_config_path")?,
        seed: required_u64(&value["seed"], "seed")?,
        report_path: optional_string(&value["report_path"]),
    })
}

#[cfg(feature = "tuner")]
fn reject_tune_candidate_local_external_refs(value: &serde_json::Value) -> SpecResult<()> {
    fn external_asset_forbidden_error(detail: String) -> SpecError {
        SpecError::new(format!(
            "{}: {detail}",
            TuneInvalidReason::CandidateExternalAssetForbidden.as_str()
        ))
    }

    fn visit(value: &serde_json::Value, path: &str) -> SpecResult<()> {
        match value {
            serde_json::Value::Object(object) => {
                for (key, child) in object {
                    let next = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    if matches!(
                        key.as_str(),
                        "spec_path" | "base_path" | "model_path" | "path" | "load_from"
                    ) {
                        return Err(external_asset_forbidden_error(format!(
                            "tune baseline_candidate contains candidate-local external asset field '{next}'"
                        )));
                    }
                    visit(child, &next)?;
                }
            }
            serde_json::Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    visit(child, &format!("{path}[{index}]"))?;
                }
            }
            serde_json::Value::String(raw) => {
                let trimmed = raw.trim_start();
                if trimmed.starts_with("file:") || trimmed.contains("://") {
                    return Err(external_asset_forbidden_error(format!(
                        "tune baseline_candidate contains candidate-local external asset reference at '{path}'"
                    )));
                }
                if raw.split(';').any(|segment| {
                    segment
                        .trim_start()
                        .strip_prefix("policy:")
                        .is_some_and(|policy| {
                            policy
                                .split(',')
                                .any(|part| part.trim_start().starts_with("load_from="))
                        })
                }) {
                    return Err(external_asset_forbidden_error(format!(
                        "tune baseline_candidate contains candidate-local policy load_from at '{path}'"
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    visit(value, "baseline_candidate")
}

#[cfg(feature = "tuner")]
fn ensure_tune_baseline_candidate_is_canonical_json(
    source_value: &serde_json::Value,
    parsed: &crate::api::CompressionBackend,
) -> SpecResult<()> {
    let canonical_value =
        compression_backend_to_json_value(parsed).map_err(|err| SpecError::new(err.to_string()))?;
    if source_value != &canonical_value {
        let mismatch =
            first_json_mismatch_path(source_value, &canonical_value, "baseline_candidate");
        let mismatch_detail = mismatch
            .map(|path| format!("; first mismatch at '{path}'"))
            .unwrap_or_default();
        return Err(SpecError::new(format!(
            "tune.baseline_candidate must be canonical compression backend JSON with no unknown or alias fields{mismatch_detail}"
        )));
    }
    Ok(())
}

#[cfg(feature = "tuner")]
fn first_json_mismatch_path(
    observed: &serde_json::Value,
    canonical: &serde_json::Value,
    path: &str,
) -> Option<String> {
    match (observed, canonical) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            for key in left.keys() {
                if !right.contains_key(key) {
                    return Some(format!("{path}.{key}"));
                }
            }
            for key in right.keys() {
                let child_path = format!("{path}.{key}");
                match left.get(key) {
                    Some(left_child) => {
                        if let Some(mismatch) =
                            first_json_mismatch_path(left_child, &right[key], &child_path)
                        {
                            return Some(mismatch);
                        }
                    }
                    None => {
                        return Some(child_path);
                    }
                }
            }
            None
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            if left.len() != right.len() {
                return Some(format!("{path}.len"));
            }
            for (index, (left_child, right_child)) in left.iter().zip(right.iter()).enumerate() {
                let child_path = format!("{path}[{index}]");
                if let Some(mismatch) =
                    first_json_mismatch_path(left_child, right_child, &child_path)
                {
                    return Some(mismatch);
                }
            }
            None
        }
        _ => {
            if observed == canonical {
                None
            } else {
                Some(path.to_string())
            }
        }
    }
}

fn ensure_known_fields(value: &serde_json::Value, allowed: &[&str], label: &str) -> SpecResult<()> {
    let object = value
        .as_object()
        .ok_or_else(|| SpecError::new(format!("{label} document must be an object")))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(SpecError::new(format!("unknown {label} field '{key}'")));
        }
    }
    Ok(())
}

fn parse_asset_bindings(value: &serde_json::Value) -> SpecResult<Vec<AssetBinding>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            Ok(AssetBinding {
                id: required_string(&item["id"], "assets[].id")?,
                path: required_string(&item["path"], "assets[].path")?,
            })
        })
        .collect()
}

#[cfg(feature = "tuner")]
fn parse_tune_asset_bindings(value: &serde_json::Value) -> SpecResult<Vec<AssetBinding>> {
    let items = value
        .as_array()
        .ok_or_else(|| SpecError::new("tune.assets must be an array"))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            ensure_known_fields(item, &["id", "path"], &format!("tune.assets[{index}]"))?;
            Ok(AssetBinding {
                id: required_string(&item["id"], "assets[].id")?,
                path: required_string(&item["path"], "assets[].path")?,
            })
        })
        .collect()
}

fn parse_environment_spec(
    value: &serde_json::Value,
    base_dir: &Path,
) -> SpecResult<EnvironmentSpec> {
    #[cfg(not(feature = "vm"))]
    let _ = base_dir;
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("environment.kind is required"))?;
    match kind {
        "builtin" => Ok(EnvironmentSpec::Builtin {
            builtin: parse_builtin_environment(
                value["name"]
                    .as_str()
                    .ok_or_else(|| SpecError::new("environment.name is required"))?,
            )?,
        }),
        #[cfg(feature = "vm")]
        "nyx_vm" => Ok(EnvironmentSpec::NyxVm(VmEnvironmentSpec {
            firecracker_config_asset: required_string(
                &value["firecracker_config_asset"],
                "environment.firecracker_config_asset",
            )?,
            instance_id: optional_string(&value["instance_id"])
                .unwrap_or_else(|| "aixi-nyx".to_string()),
            shared_region_name: optional_string(&value["shared_region_name"])
                .unwrap_or_else(|| "shared".to_string()),
            shared_region_size: default_usize(
                &value["shared_region_size"],
                4096,
                "environment.shared_region_size",
            )?,
            shared_memory_policy: parse_shared_memory_policy(
                value["shared_memory_policy"].as_str().unwrap_or("snapshot"),
            )?,
            step_timeout_ms: value["step_timeout_ms"].as_u64().unwrap_or(100),
            boot_timeout_ms: value["boot_timeout_ms"].as_u64().unwrap_or(30_000),
            episode_steps: default_usize(
                &value["episode_steps"],
                100,
                "environment.episode_steps",
            )?,
            step_cost: value["step_cost"].as_i64().unwrap_or(0),
            observation_policy: super::canonicalize_vm_observation_policy_name(
                value["observation_policy"]
                    .as_str()
                    .unwrap_or("shared_memory"),
            )?,
            observation_bits: default_usize(
                &value["observation_bits"],
                8,
                "environment.observation_bits",
            )?,
            observation_stream_len: default_usize(
                &value["observation_stream_len"],
                64,
                "environment.observation_stream_len",
            )?,
            observation_stream_mode: super::canonicalize_vm_observation_stream_mode_name(
                value["observation_stream_mode"]
                    .as_str()
                    .unwrap_or("pad_truncate"),
            )?,
            observation_pad_byte: default_u8(
                &value["observation_pad_byte"],
                0,
                "environment.observation_pad_byte",
            )?,
            reward_bits: default_usize(&value["reward_bits"], 8, "environment.reward_bits")?,
            reward_policy: parse_vm_reward_policy(&value["reward_policy"])?,
            reward_shaping: parse_optional_vm_reward_shaping(&value["reward_shaping"])?,
            action_source: parse_vm_action_source(&value["action_source"])?,
            action_filter: parse_optional_vm_action_filter(&value["action_filter"])?,
            action_prefix: value["protocol"]["action_prefix"]
                .as_str()
                .unwrap_or("ACT ")
                .to_string(),
            action_suffix: value["protocol"]["action_suffix"]
                .as_str()
                .unwrap_or("\n")
                .to_string(),
            obs_prefix: value["protocol"]["obs_prefix"]
                .as_str()
                .unwrap_or("OBS ")
                .to_string(),
            rew_prefix: value["protocol"]["rew_prefix"]
                .as_str()
                .unwrap_or("REW ")
                .to_string(),
            done_prefix: value["protocol"]["done_prefix"]
                .as_str()
                .unwrap_or("DONE ")
                .to_string(),
            data_prefix: value["protocol"]["data_prefix"]
                .as_str()
                .unwrap_or("DATA ")
                .to_string(),
            wire_encoding: super::canonicalize_vm_payload_encoding(
                value["protocol"]["wire_encoding"].as_str().unwrap_or("hex"),
                "environment.protocol.wire_encoding",
            )?,
            stats_backend: parse_rate_backend_json(
                &value["stats_backend"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            trace: parse_optional_vm_trace(&value["trace"])?,
            debug_mode: value["debug_mode"].as_bool().unwrap_or(false),
            crash_log: optional_string(&value["crash_log"]),
        })),
        #[cfg(not(feature = "vm"))]
        "nyx_vm" => Err(SpecError::new(
            "nyx_vm environment requires the 'vm' feature",
        )),
        other => Err(SpecError::new(format!(
            "unknown environment kind '{other}'"
        ))),
    }
}

fn parse_interface_spec(value: &serde_json::Value) -> SpecResult<PlannerInterfaceSpec> {
    ensure_known_fields(
        value,
        &[
            "observation_bits",
            "observation_stream_len",
            "observation_key_mode",
            "reward_bits",
            "agent_actions",
        ],
        "interface",
    )?;
    let agent_actions_raw = required_usize(&value["agent_actions"], "interface.agent_actions")?;
    let agent_actions = ActionAlphabet::try_from_usize(agent_actions_raw)
        .map_err(|_| SpecError::new("interface.agent_actions must be >= 1"))?;
    Ok(PlannerInterfaceSpec {
        observation_bits: required_usize(&value["observation_bits"], "interface.observation_bits")?,
        observation_stream_len: required_usize(
            &value["observation_stream_len"],
            "interface.observation_stream_len",
        )?,
        observation_key_mode: parse_observation_key_mode(
            value["observation_key_mode"]
                .as_str()
                .unwrap_or("full_stream"),
        )?,
        reward_bits: required_usize(&value["reward_bits"], "interface.reward_bits")?,
        agent_actions,
    })
}

#[cfg(feature = "tuner")]
fn parse_tune_interface_spec(value: &serde_json::Value) -> SpecResult<TunePlannerInterfaceSpec> {
    ensure_known_fields(
        value,
        &[
            "observation_bits",
            "observation_stream_len",
            "observation_key_mode",
            "reward_bits",
            "agent_actions",
        ],
        "controller.interface",
    )?;
    let agent_actions_raw = required_usize(&value["agent_actions"], "interface.agent_actions")?;
    let agent_actions = ActionAlphabet::try_from_usize(agent_actions_raw)
        .map_err(|_| SpecError::new("interface.agent_actions must be >= 1"))?;
    Ok(TunePlannerInterfaceSpec {
        observation_bits: required_usize(&value["observation_bits"], "interface.observation_bits")?,
        observation_stream_len: required_usize(
            &value["observation_stream_len"],
            "interface.observation_stream_len",
        )?,
        observation_key_mode: parse_observation_key_mode(
            value["observation_key_mode"].as_str().ok_or_else(|| {
                SpecError::new("controller.interface.observation_key_mode is required")
            })?,
        )?,
        reward_bits: required_usize(&value["reward_bits"], "interface.reward_bits")?,
        agent_actions,
    })
}

fn parse_controller_spec(value: &serde_json::Value, base_dir: &Path) -> SpecResult<ControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "mc_aixi" => Ok(ControllerSpec::McAixi(McAixiControllerSpec {
            predictor: parse_rate_backend_json(
                &value["predictor"],
                base_dir,
                crate::api::MAX_MIXTURE_NESTING,
            )?,
            bit_stream_semantics: parse_bit_stream_semantics(
                value.get("bit_stream_semantics"),
                "controller.bit_stream_semantics",
            )?,
            agent_horizon: required_usize(&value["agent_horizon"], "controller.agent_horizon")?,
            num_simulations: required_usize(
                &value["num_simulations"],
                "controller.num_simulations",
            )?,
            mcts_strategy: parse_mcts_strategy(
                value.get("mcts_strategy"),
                "controller.mcts_strategy",
            )?,
            exploration_exploitation_ratio: required_f64(
                &value["exploration_exploitation_ratio"],
                "controller.exploration_exploitation_ratio",
            )?,
            discount_gamma: required_f64(&value["discount_gamma"], "controller.discount_gamma")?,
        })),
        "aiqi_discounted" => Ok(ControllerSpec::AiqiDiscounted(
            AiqiDiscountedControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                bit_stream_semantics: parse_bit_stream_semantics(
                    value.get("bit_stream_semantics"),
                    "controller.bit_stream_semantics",
                )?,
                discount_gamma: required_f64(
                    &value["discount_gamma"],
                    "controller.discount_gamma",
                )?,
                return_horizon: required_usize(
                    &value["return_horizon"],
                    "controller.return_horizon",
                )?,
                return_bins: required_usize(&value["return_bins"], "controller.return_bins")?,
                augmentation_period: required_usize(
                    &value["augmentation_period"],
                    "controller.augmentation_period",
                )?,
                history_prune_keep_steps: optional_usize(
                    &value["history_prune_keep_steps"],
                    "controller.history_prune_keep_steps",
                )?,
                baseline_exploration: required_f64(
                    &value["baseline_exploration"],
                    "controller.baseline_exploration",
                )?,
            },
        )),
        #[cfg(feature = "aixi")]
        "aiqi_warmstart_exact_jh" => Ok(ControllerSpec::AiqiWarmstartExactJh(
            WarmStartExactJhControllerSpec {
                predictor: parse_rate_backend_json(
                    &value["predictor"],
                    base_dir,
                    crate::api::MAX_MIXTURE_NESTING,
                )?,
                bit_stream_semantics: parse_bit_stream_semantics(
                    value.get("bit_stream_semantics"),
                    "controller.bit_stream_semantics",
                )?,
                return_horizon: required_usize(
                    &value["return_horizon"],
                    "controller.return_horizon",
                )?,
                return_bins: required_usize(&value["return_bins"], "controller.return_bins")?,
                label_phase_period: required_usize(
                    &value["label_phase_period"],
                    "controller.label_phase_period",
                )?,
                teacher_dataset_asset: required_string(
                    &value["teacher_dataset_asset"],
                    "controller.teacher_dataset_asset",
                )?,
                planner_simulations_per_step: required_usize(
                    &value["planner_simulations_per_step"],
                    "controller.planner_simulations_per_step",
                )?,
            },
        )),
        #[cfg(not(feature = "aixi"))]
        "aiqi_warmstart_exact_jh" => Err(SpecError::new(
            "aiqi_warmstart_exact_jh controller requires infotheory built with feature 'aixi'",
        )),
        other => Err(SpecError::new(format!("unknown controller kind '{other}'"))),
    }
}

fn parse_bit_stream_semantics(
    value: Option<&serde_json::Value>,
    label: &str,
) -> SpecResult<BitStreamSemantics> {
    let Some(value) = value else {
        // Default for absent bit_stream_semantics is BinaryTokens (AIXI planner
        // paths explicitly set their own default via aixi::model when needed).
        // This reference must remain feature-agnostic for parser hygiene.
        return Ok(BitStreamSemantics::BinaryTokens);
    };
    let Some(object) = value.as_object() else {
        return Err(SpecError::new(format!(
            "{label} must be an object with a 'kind' field"
        )));
    };
    let kind = object
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| SpecError::new(format!("{label}.kind is required")))?;
    match kind {
        "byte_packed" => {
            let order = match object.get("order").and_then(serde_json::Value::as_str) {
                Some("msb_first") | None => BitOrder::MsbFirst,
                Some("lsb_first") => BitOrder::LsbFirst,
                Some(other) => {
                    return Err(SpecError::new(format!("unknown {label}.order '{other}'")));
                }
            };
            Ok(BitStreamSemantics::BytePacked { order })
        }
        "binary_tokens" => Ok(BitStreamSemantics::BinaryTokens),
        other => Err(SpecError::new(format!(
            "unknown bit stream semantics kind '{other}'"
        ))),
    }
}

fn parse_mcts_strategy(value: Option<&serde_json::Value>, label: &str) -> SpecResult<MctsStrategy> {
    // Absent field defaults to the canonical sequential planner so older
    // documents that predate `mcts_strategy` continue to parse cleanly.
    let Some(value) = value else {
        return Ok(MctsStrategy::RhoUct);
    };
    // The canonical, serializer-emitted form is always an object with a
    // `kind` field. Strings are not accepted: there is exactly one way to
    // spell each strategy in the schema.
    let Some(object) = value.as_object() else {
        return Err(SpecError::new(format!(
            "{label} must be an object with a 'kind' field"
        )));
    };
    let kind = object
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| SpecError::new(format!("{label}.kind is required")))?;
    match kind {
        "rho_uct" => Ok(MctsStrategy::RhoUct),
        "parallel_uct" => {
            let workers_raw = required_usize(&value["workers"], &format!("{label}.workers"))?;
            let workers = NonZeroUsize::new(workers_raw)
                .ok_or_else(|| SpecError::new(format!("{label}.workers must be >= 1")))?;
            let bu_uct_m_max = match object.get("bu_uct_m_max") {
                Some(raw) if raw.is_null() => None,
                Some(raw) => Some(required_f64(raw, &format!("{label}.bu_uct_m_max"))?),
                None => None,
            };
            Ok(MctsStrategy::ParallelUct {
                workers,
                bu_uct_m_max,
            })
        }
        other => Err(SpecError::new(format!(
            "unknown MCTS strategy kind '{other}'"
        ))),
    }
}

fn parse_runtime_spec(value: &serde_json::Value) -> SpecResult<PlannerRuntimeSpec> {
    Ok(PlannerRuntimeSpec {
        random_seed: value["random_seed"].as_u64(),
        learn_cycles: optional_usize(&value["learn_cycles"], "runtime.learn_cycles")?,
        eval_cycles: optional_usize(&value["eval_cycles"], "runtime.eval_cycles")?,
        terminate_lifetime: default_usize(
            &value["terminate_lifetime"],
            20,
            "runtime.terminate_lifetime",
        )?,
        log_every: default_usize(&value["log_every"], 1, "runtime.log_every")?,
        perf: value["perf"].as_bool().unwrap_or(false),
        vm_perf_only: value["vm_perf_only"].as_bool().unwrap_or(false),
        explore_epsilon: value["explore_epsilon"].as_f64().unwrap_or(0.0),
        explore_gamma: value["explore_gamma"].as_f64().unwrap_or(1.0),
    })
}

#[cfg(feature = "tuner")]
fn parse_tune_bounds_spec(value: &serde_json::Value) -> SpecResult<TuneBoundsSpec> {
    ensure_known_fields(
        value,
        &[
            "allowed_backends",
            "forbidden_backends",
            "parameter_ranges",
            "max_experts",
            "max_mixture_nesting_depth",
            "min_experts",
            "allow_duplicate_experts",
            "required_experts",
            "forbidden_expert_pairs",
        ],
        "bounds",
    )?;
    Ok(TuneBoundsSpec {
        allowed_backends: optional_tune_string_list(
            value.get("allowed_backends"),
            "bounds.allowed_backends",
        )?,
        forbidden_backends: optional_tune_string_list(
            value.get("forbidden_backends"),
            "bounds.forbidden_backends",
        )?,
        parameter_ranges: parse_tune_parameter_ranges(value.get("parameter_ranges"))?,
        max_experts: required_usize(&value["max_experts"], "bounds.max_experts")?,
        max_mixture_nesting_depth: required_usize(
            &value["max_mixture_nesting_depth"],
            "bounds.max_mixture_nesting_depth",
        )?,
        min_experts: optional_usize(&value["min_experts"], "bounds.min_experts")?,
        allow_duplicate_experts: value["allow_duplicate_experts"].as_bool(),
        required_experts: optional_tune_string_list(
            value.get("required_experts"),
            "bounds.required_experts",
        )?,
        forbidden_expert_pairs: optional_tune_pair_list(
            value.get("forbidden_expert_pairs"),
            "bounds.forbidden_expert_pairs",
        )?,
    })
}

#[cfg(feature = "tuner")]
fn optional_tune_string_list(
    value: Option<&serde_json::Value>,
    label: &str,
) -> SpecResult<Vec<String>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    if !raw.is_array() {
        return Err(SpecError::new(format!("{label} must be an array")));
    }
    string_list(raw).map_err(|err| SpecError::new(format!("{label}: {err}")))
}

#[cfg(feature = "tuner")]
fn optional_tune_pair_list(
    value: Option<&serde_json::Value>,
    label: &str,
) -> SpecResult<Vec<(String, String)>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    if !raw.is_array() {
        return Err(SpecError::new(format!("{label} must be an array")));
    }
    pair_list(raw).map_err(|err| SpecError::new(format!("{label}: {err}")))
}

#[cfg(feature = "tuner")]
fn parse_tune_controller_spec(value: &serde_json::Value) -> SpecResult<TuneControllerSpec> {
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| SpecError::new("controller.kind is required"))?;
    match kind {
        "annealed_hill_climbing" => {
            ensure_known_fields(
                value,
                &["kind", "max_mutation_radius"],
                "controller.annealed_hill_climbing",
            )?;
            Ok(TuneControllerSpec::AnnealedHillClimbing(
                AnnealedHillClimbingTuneControllerSpec {
                    max_mutation_radius: required_usize(
                        &value["max_mutation_radius"],
                        "controller.max_mutation_radius",
                    )?,
                },
            ))
        }
        "mc_aixi_fac_ctw" => {
            ensure_known_fields(
                value,
                &["kind", "interface", "planner_simulations_per_step"],
                "controller.mc_aixi_fac_ctw",
            )?;
            Ok(TuneControllerSpec::McAixiFacCtw(
                McAixiFacCtwTuneControllerSpec {
                    interface: parse_tune_interface_spec(&value["interface"])?,
                    planner_simulations_per_step: required_usize(
                        &value["planner_simulations_per_step"],
                        "controller.planner_simulations_per_step",
                    )?,
                },
            ))
        }
        "aiqi_discounted" => {
            ensure_known_fields(
                value,
                &[
                    "kind",
                    "interface",
                    "planner_simulations_per_step",
                    "return_horizon",
                    "return_bins",
                    "discount_factor",
                    "min_improvement",
                    "max_improvement",
                ],
                "controller.aiqi_discounted",
            )?;
            Ok(TuneControllerSpec::AiqiDiscounted(
                AiqiDiscountedTuneControllerSpec {
                    interface: parse_tune_interface_spec(&value["interface"])?,
                    planner_simulations_per_step: required_usize(
                        &value["planner_simulations_per_step"],
                        "controller.planner_simulations_per_step",
                    )?,
                    return_horizon: required_usize(
                        &value["return_horizon"],
                        "controller.return_horizon",
                    )?,
                    return_bins: required_usize(&value["return_bins"], "controller.return_bins")?,
                    discount_factor: required_f64(
                        &value["discount_factor"],
                        "controller.discount_factor",
                    )?,
                    min_improvement: required_f64(
                        &value["min_improvement"],
                        "controller.min_improvement",
                    )?,
                    max_improvement: required_f64(
                        &value["max_improvement"],
                        "controller.max_improvement",
                    )?,
                },
            ))
        }
        #[cfg(feature = "aixi")]
        "aiqi_warmstart_exact_jh" => {
            ensure_known_fields(
                value,
                &[
                    "kind",
                    "interface",
                    "planner_simulations_per_step",
                    "return_horizon",
                    "warmstart_teacher_dataset_asset",
                    "label_phase_period",
                ],
                "controller.aiqi_warmstart_exact_jh",
            )?;
            Ok(TuneControllerSpec::AiqiWarmstartExactJh(
                WarmStartExactJhTuneControllerSpec {
                    interface: parse_tune_interface_spec(&value["interface"])?,
                    planner_simulations_per_step: required_usize(
                        &value["planner_simulations_per_step"],
                        "controller.planner_simulations_per_step",
                    )?,
                    return_horizon: required_usize(
                        &value["return_horizon"],
                        "controller.return_horizon",
                    )?,
                    warmstart_teacher_dataset_asset: required_string(
                        &value["warmstart_teacher_dataset_asset"],
                        "controller.warmstart_teacher_dataset_asset",
                    )?,
                    label_phase_period: required_usize(
                        &value["label_phase_period"],
                        "controller.label_phase_period",
                    )?,
                },
            ))
        }
        #[cfg(not(feature = "aixi"))]
        "aiqi_warmstart_exact_jh" => Err(SpecError::new(
            "aiqi_warmstart_exact_jh controller requires infotheory built with feature 'aixi'",
        )),
        other => Err(SpecError::new(format!(
            "unknown tune controller kind '{other}'"
        ))),
    }
}

#[cfg(feature = "tuner")]
fn parse_tune_parameter_ranges(
    value: Option<&serde_json::Value>,
) -> SpecResult<Vec<TuneParameterRangeSpec>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    let Some(items) = raw.as_array() else {
        return Err(SpecError::new("bounds.parameter_ranges must be an array"));
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            ensure_known_fields(
                item,
                &["parameter", "min", "max"],
                &format!("bounds.parameter_ranges[{index}]"),
            )?;
            Ok(TuneParameterRangeSpec {
                parameter: required_string(
                    &item["parameter"],
                    "bounds.parameter_ranges[].parameter",
                )?,
                min: required_f64(&item["min"], "bounds.parameter_ranges[].min")?,
                max: required_f64(&item["max"], "bounds.parameter_ranges[].max")?,
            })
        })
        .collect()
}

#[cfg(feature = "vm")]
fn parse_vm_reward_policy(value: &serde_json::Value) -> SpecResult<VmRewardPolicySpec> {
    match value["kind"].as_str().unwrap_or("from_guest") {
        "from_guest" => Ok(VmRewardPolicySpec::FromGuest),
        "pattern" => Ok(VmRewardPolicySpec::Pattern {
            pattern: required_string(&value["pattern"], "environment.reward_policy.pattern")?,
            base_reward: value["base_reward"].as_i64().unwrap_or(0),
            bonus_reward: value["bonus_reward"].as_i64().unwrap_or(10),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm reward policy '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_reward_shaping(
    value: &serde_json::Value,
) -> SpecResult<Option<VmRewardShapingSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    let spec = match value["kind"].as_str().unwrap_or("trace_entropy") {
        "entropy_reduction" => VmRewardShapingSpec::EntropyReduction {
            baseline_asset: required_string(
                &value["baseline_asset"],
                "environment.reward_shaping.baseline_asset",
            )?,
            scale: value["scale"].as_f64().unwrap_or(1.0),
            crash_bonus: value["crash_bonus"].as_i64(),
            timeout_bonus: value["timeout_bonus"].as_i64(),
        },
        "trace_entropy" => VmRewardShapingSpec::TraceEntropy {
            scale: value["scale"].as_f64().unwrap_or(1.0),
            normalize: value["normalize"].as_bool().unwrap_or(false),
        },
        other => {
            return Err(SpecError::new(format!(
                "unknown vm reward shaping '{other}'"
            )));
        }
    };
    Ok(Some(spec))
}

#[cfg(feature = "vm")]
fn parse_vm_action_source(value: &serde_json::Value) -> SpecResult<VmRuntimeActionSourceSpec> {
    match value["kind"].as_str().unwrap_or("literal") {
        "literal" => {
            let encoding = super::canonicalize_vm_payload_encoding(
                value["encoding"].as_str().unwrap_or("utf8"),
                "environment.action_source.encoding",
            )?;
            let actions = value["actions"]
                .as_array()
                .ok_or_else(|| SpecError::new("environment.action_source.actions is required"))?;
            let mut names = Vec::with_capacity(actions.len());
            let mut payloads = Vec::with_capacity(actions.len());
            for action in actions {
                names.push(optional_string(&action["name"]));
                payloads.push(required_string(
                    &action["payload"],
                    "environment.action_source.actions[].payload",
                )?);
            }
            Ok(VmRuntimeActionSourceSpec::Literal {
                names,
                payloads,
                encoding,
            })
        }
        "fuzz" => Ok(VmRuntimeActionSourceSpec::Fuzz {
            seeds: string_list(&value["seeds"])?,
            encoding: super::canonicalize_vm_payload_encoding(
                value["encoding"].as_str().unwrap_or("utf8"),
                "environment.action_source.encoding",
            )?,
            mutators: string_list(&value["mutators"])?
                .into_iter()
                .map(|name| super::canonicalize_vm_fuzz_mutator_name(&name))
                .collect::<SpecResult<Vec<_>>>()?,
            min_len: default_usize(&value["min_len"], 1, "environment.action_source.min_len")?,
            max_len: default_usize(&value["max_len"], 4096, "environment.action_source.max_len")?,
            dictionary: string_list(&value["dictionary"])?,
            rng_seed: value["rng_seed"].as_u64().unwrap_or(0),
        }),
        other => Err(SpecError::new(format!(
            "unknown vm action source '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_optional_vm_action_filter(
    value: &serde_json::Value,
) -> SpecResult<Option<VmActionFilterSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmActionFilterSpec {
        min_entropy: value["min_entropy"].as_f64(),
        max_entropy: value["max_entropy"].as_f64(),
        min_intrinsic_dependence: value["min_intrinsic_dependence"].as_f64(),
        min_novelty: value["min_novelty"].as_f64(),
        novelty_prior_asset: optional_string(&value["novelty_prior_asset"]),
        reject_reward: value["reject_reward"].as_i64(),
    }))
}

#[cfg(feature = "vm")]
fn parse_optional_vm_trace(value: &serde_json::Value) -> SpecResult<Option<VmTraceSpec>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(VmTraceSpec {
        shared_region_name: optional_string(&value["shared_region_name"]),
        max_bytes: default_usize(
            &value["max_bytes"],
            1_000_000,
            "environment.trace.max_bytes",
        )?,
        reset_on_episode: value["reset_on_episode"].as_bool().unwrap_or(false),
    }))
}

#[cfg(any(feature = "tuner", feature = "vm"))]
fn string_list(value: &serde_json::Value) -> SpecResult<Vec<String>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(|text| text.to_string())
                .ok_or_else(|| SpecError::new("expected string list item"))
        })
        .collect()
}

#[cfg(feature = "tuner")]
fn pair_list(value: &serde_json::Value) -> SpecResult<Vec<(String, String)>> {
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    let mut pairs = Vec::with_capacity(items.len());
    for item in items {
        let Some(pair) = item.as_array() else {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must be arrays",
            ));
        };
        if pair.len() != 2 {
            return Err(SpecError::new(
                "forbidden_expert_pairs entries must have length 2",
            ));
        }
        pairs.push((
            required_string(&pair[0], "forbidden_expert_pairs[][0]")?,
            required_string(&pair[1], "forbidden_expert_pairs[][1]")?,
        ));
    }
    Ok(pairs)
}

#[cfg(any(feature = "tuner", feature = "vm"))]
fn optional_string(value: &serde_json::Value) -> Option<String> {
    value.as_str().map(|text| text.to_string())
}

fn required_string(value: &serde_json::Value, label: &str) -> SpecResult<String> {
    value
        .as_str()
        .map(|text| text.to_string())
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_f64(value: &serde_json::Value, label: &str) -> SpecResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_u64(value: &serde_json::Value, label: &str) -> SpecResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| SpecError::new(format!("{label} is required")))
}

fn required_usize(value: &serde_json::Value, label: &str) -> SpecResult<usize> {
    usize::try_from(required_u64(value, label)?)
        .map_err(|_| SpecError::new(format!("{label} exceeds usize::MAX")))
}

fn optional_usize(value: &serde_json::Value, label: &str) -> SpecResult<Option<usize>> {
    if value.is_null() {
        Ok(None)
    } else {
        required_usize(value, label).map(Some)
    }
}

fn default_usize(value: &serde_json::Value, default: usize, label: &str) -> SpecResult<usize> {
    if value.is_null() {
        Ok(default)
    } else {
        required_usize(value, label)
    }
}

#[cfg(feature = "vm")]
fn default_u8(value: &serde_json::Value, default: u8, label: &str) -> SpecResult<u8> {
    if value.is_null() {
        Ok(default)
    } else {
        u8::try_from(required_u64(value, label)?)
            .map_err(|_| SpecError::new(format!("{label} exceeds u8::MAX")))
    }
}

fn parse_builtin_environment(name: &str) -> SpecResult<super::BuiltinEnvironmentSpec> {
    match name {
        "tuner_bridge" => Err(SpecError::new(
            "builtin environment 'tuner_bridge' is an internal tuner planner bridge and is not accepted in canonical planner-run JSON",
        )),
        "coin_flip" => Ok(super::BuiltinEnvironmentSpec::CoinFlip),
        "biased_rock_paper_scissor" => Ok(super::BuiltinEnvironmentSpec::BiasedRockPaperScissor),
        "kuhn_poker" => Ok(super::BuiltinEnvironmentSpec::KuhnPoker),
        "extended_tiger" => Ok(super::BuiltinEnvironmentSpec::ExtendedTiger),
        "ctw_test" => Err(SpecError::new(
            "builtin environment 'ctw_test' is no longer supported",
        )),
        "tic_tac_toe" => Ok(super::BuiltinEnvironmentSpec::TicTacToe),
        "blackjack" => Ok(super::BuiltinEnvironmentSpec::Blackjack),
        "platformer" => Ok(super::BuiltinEnvironmentSpec::Platformer),
        other => Err(SpecError::new(format!(
            "unknown builtin environment '{other}'"
        ))),
    }
}

#[cfg(feature = "vm")]
fn parse_shared_memory_policy(name: &str) -> SpecResult<super::SharedMemoryPolicySpec> {
    match name {
        "snapshot" => Ok(super::SharedMemoryPolicySpec::Snapshot),
        "preserve" => Ok(super::SharedMemoryPolicySpec::Preserve),
        other => Err(SpecError::new(format!(
            "unknown shared memory policy '{other}'"
        ))),
    }
}

fn parse_observation_key_mode(name: &str) -> SpecResult<crate::aixi::common::ObservationKeyMode> {
    match name {
        "first" => Ok(crate::aixi::common::ObservationKeyMode::First),
        "last" => Ok(crate::aixi::common::ObservationKeyMode::Last),
        "stream_hash" => Ok(crate::aixi::common::ObservationKeyMode::StreamHash),
        "full_stream" => Ok(crate::aixi::common::ObservationKeyMode::FullStream),
        other => Err(SpecError::new(format!(
            "unknown observation key mode '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spec_document_json_value_rejects_bad_schema_and_unknown_kind() {
        let err = match parse_spec_document_json_value(
            &serde_json::json!({
                "schema_version": 0,
                "kind": "planner_run",
            }),
            Path::new("."),
        ) {
            Ok(_) => panic!("unsupported schema version must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("unsupported spec document schema_version")
        );

        let err = match parse_spec_document_json_value(
            &serde_json::json!({
                "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
                "kind": "unknown",
            }),
            Path::new("."),
        ) {
            Ok(_) => panic!("unknown document kind must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("unknown spec document kind 'unknown'")
        );

        #[cfg(not(feature = "tuner"))]
        {
            let result = parse_spec_document_json_value(
                &serde_json::json!({
                    "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
                    "kind": "tune",
                }),
                Path::new("."),
            );
            match result {
                Ok(_) => panic!("tune must require tuner feature"),
                Err(err) => assert!(
                    err.to_string()
                        .contains("tune documents require infotheory built with feature 'tuner'")
                ),
            }
        }
    }

    #[test]
    fn parse_mcts_strategy_enforces_canonical_object_shape() {
        assert_eq!(
            parse_mcts_strategy(None, "controller.mcts_strategy")
                .expect("missing strategy should default"),
            MctsStrategy::RhoUct
        );

        let err = parse_mcts_strategy(
            Some(&serde_json::json!("rho_uct")),
            "controller.mcts_strategy",
        )
        .expect_err("string shorthand must fail");
        assert!(
            err.to_string()
                .contains("controller.mcts_strategy must be an object with a 'kind' field")
        );

        let err = parse_mcts_strategy(Some(&serde_json::json!({})), "controller.mcts_strategy")
            .expect_err("missing kind must fail");
        assert!(
            err.to_string()
                .contains("controller.mcts_strategy.kind is required")
        );

        let err = parse_mcts_strategy(
            Some(&serde_json::json!({
                "kind": "parallel_uct",
                "workers": 0,
            })),
            "controller.mcts_strategy",
        )
        .expect_err("zero workers must fail");
        assert!(
            err.to_string()
                .contains("controller.mcts_strategy.workers must be >= 1")
        );
    }

    #[test]
    fn parse_runtime_spec_applies_document_defaults() {
        let runtime = parse_runtime_spec(&serde_json::json!({})).expect("runtime defaults");
        assert_eq!(runtime.random_seed, None);
        assert_eq!(runtime.learn_cycles, None);
        assert_eq!(runtime.eval_cycles, None);
        assert_eq!(runtime.terminate_lifetime, 20);
        assert_eq!(runtime.log_every, 1);
        assert!(!runtime.perf);
        assert!(!runtime.vm_perf_only);
        assert_eq!(runtime.explore_epsilon, 0.0);
        assert_eq!(runtime.explore_gamma, 1.0);
    }

    #[cfg(feature = "tuner")]
    #[test]
    fn parse_tune_bounds_and_list_helpers_cover_optional_shape_contracts() {
        let parsed = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": ["ctw"],
            "forbidden_backends": ["zpaq"],
            "parameter_ranges": [{
                "parameter": "mixture.alpha",
                "min": 0.1,
                "max": 0.5,
            }],
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "min_experts": 1,
            "allow_duplicate_experts": false,
            "required_experts": ["ctw"],
            "forbidden_expert_pairs": [["ctw", "zpaq"]],
        }))
        .expect("valid tune bounds");
        assert_eq!(parsed.allowed_backends, vec!["ctw"]);
        assert_eq!(
            parsed.forbidden_expert_pairs,
            vec![("ctw".into(), "zpaq".into())]
        );

        let err =
            string_list(&serde_json::json!(["ctw", 7])).expect_err("mixed string list must fail");
        assert!(err.to_string().contains("expected string list item"));

        let err =
            pair_list(&serde_json::json!(["ctw"])).expect_err("non-array pair item must fail");
        assert!(
            err.to_string()
                .contains("forbidden_expert_pairs entries must be arrays")
        );

        let err = pair_list(&serde_json::json!([["ctw"]])).expect_err("short pair item must fail");
        assert!(
            err.to_string()
                .contains("forbidden_expert_pairs entries must have length 2")
        );

        let err = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": "ctw",
            "forbidden_backends": ["zpaq"],
            "parameter_ranges": [],
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "required_experts": [],
            "forbidden_expert_pairs": [],
        }))
        .expect_err("non-array allowed_backends must fail");
        assert!(
            err.to_string()
                .contains("bounds.allowed_backends must be an array")
        );

        let err = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": ["ctw"],
            "forbidden_backends": [],
            "parameter_ranges": [],
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "required_experts": "ctw",
            "forbidden_expert_pairs": [],
        }))
        .expect_err("non-array required_experts must fail");
        assert!(
            err.to_string()
                .contains("bounds.required_experts must be an array")
        );

        let err = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": ["ctw"],
            "forbidden_backends": [],
            "parameter_ranges": [],
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "required_experts": [],
            "forbidden_expert_pairs": "ctw,zpaq",
        }))
        .expect_err("non-array forbidden_expert_pairs must fail");
        assert!(err.to_string().contains("bounds.forbidden_expert_pairs"));

        let parsed_without_ranges = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": ["ctw"],
            "forbidden_backends": ["zpaq"],
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "required_experts": ["ctw"],
            "forbidden_expert_pairs": [["ctw", "zpaq"]],
        }))
        .expect("missing parameter_ranges should parse as empty optional list");
        assert!(parsed_without_ranges.parameter_ranges.is_empty());

        let err = parse_tune_bounds_spec(&serde_json::json!({
            "allowed_backends": ["ctw"],
            "forbidden_backends": ["zpaq"],
            "parameter_ranges": {"bad": "shape"},
            "max_experts": 4,
            "max_mixture_nesting_depth": 2,
            "required_experts": ["ctw"],
            "forbidden_expert_pairs": [["ctw", "zpaq"]],
        }))
        .expect_err("non-array parameter_ranges must fail");
        assert!(
            err.to_string()
                .contains("bounds.parameter_ranges must be an array")
        );
    }

    #[cfg(feature = "tuner")]
    #[test]
    fn parse_tune_controller_variants_cover_semantic_contracts() {
        let interface = serde_json::json!({
            "observation_bits": 1,
            "observation_stream_len": 1,
            "observation_key_mode": "full_stream",
            "reward_bits": 1,
            "agent_actions": 2,
        });

        let fac = parse_tune_controller_spec(&serde_json::json!({
            "kind": "mc_aixi_fac_ctw",
            "interface": interface.clone(),
            "planner_simulations_per_step": 16,
        }))
        .expect("mc_aixi_fac_ctw controller should parse");
        assert!(matches!(fac, TuneControllerSpec::McAixiFacCtw(_)));

        let discounted = parse_tune_controller_spec(&serde_json::json!({
            "kind": "aiqi_discounted",
            "interface": interface.clone(),
            "planner_simulations_per_step": 32,
            "return_horizon": 4,
            "return_bins": 8,
            "discount_factor": 0.95,
            "min_improvement": -1.0,
            "max_improvement": 1.0,
        }))
        .expect("aiqi_discounted controller should parse");
        assert!(matches!(discounted, TuneControllerSpec::AiqiDiscounted(_)));

        #[cfg(feature = "aixi")]
        {
            let warmstart = parse_tune_controller_spec(&serde_json::json!({
                "kind": "aiqi_warmstart_exact_jh",
                "interface": interface,
                "planner_simulations_per_step": 1,
                "return_horizon": 3,
                "warmstart_teacher_dataset_asset": "teacher",
                "label_phase_period": 2,
            }))
            .expect("aiqi_warmstart_exact_jh controller should parse");
            assert!(matches!(
                warmstart,
                TuneControllerSpec::AiqiWarmstartExactJh(_)
            ));
        }

        let err = parse_tune_controller_spec(&serde_json::json!({
            "kind": "definitely_unknown_tune_controller"
        }))
        .expect_err("unknown tune controller kind must fail");
        assert!(err.to_string().contains("unknown tune controller kind"));
    }

    #[cfg(feature = "tuner")]
    #[test]
    fn list_helpers_preserve_legacy_non_tune_defaults() {
        let empty_strings =
            string_list(&serde_json::json!("ctw")).expect("non-array string list should default");
        assert!(empty_strings.is_empty());

        let empty_pairs =
            pair_list(&serde_json::json!("ctw,zpaq")).expect("non-array pair list should default");
        assert!(empty_pairs.is_empty());
    }

    #[cfg(feature = "tuner")]
    #[test]
    fn parse_tune_reports_precise_baseline_canonical_mismatch_path() {
        let tune = serde_json::json!({
            "schema_version": SPEC_DOCUMENT_SCHEMA_VERSION,
            "kind": "tune",
            "assets": [{ "id": "dataset", "path": "dataset.bin" }],
            "input_asset": "dataset",
            "baseline_candidate": {
                "kind": "rate-ac",
                "rate_backend": {
                    "kind": "ctw",
                    "depth": 16,
                    "extra_alias_field": 7
                },
                "framing": "framed"
            },
            "controller": {
                "kind": "annealed_hill_climbing",
                "max_mutation_radius": 1
            },
            "bounds": {
                "allowed_backends": ["ctw"],
                "forbidden_backends": [],
                "parameter_ranges": [],
                "max_experts": 2,
                "max_mixture_nesting_depth": 1,
                "required_experts": [],
                "forbidden_expert_pairs": []
            },
            "eval_time_limit_seconds": 1.0,
            "time_budget_seconds": 1.0,
            "min_throughput_bytes_per_second": 1.0,
            "max_memory_bytes": 1024,
            "output_config_path": "out.json",
            "seed": 1
        });
        let err = match parse_spec_document_json_value(&tune, Path::new(".")) {
            Ok(_) => panic!("non-canonical baseline candidate must fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("must be canonical compression backend JSON"),
            "unexpected error: {message}"
        );
        assert!(
            message
                .contains("first mismatch at 'baseline_candidate.rate_backend.extra_alias_field'"),
            "mismatch path should be explicit: {message}"
        );
    }

    #[test]
    fn canonical_name_parsers_reject_removed_or_unknown_aliases() {
        let err =
            parse_builtin_environment("ctw_test").expect_err("removed builtin alias must fail");
        assert!(err.to_string().contains("no longer supported"));

        assert_eq!(
            parse_observation_key_mode("first").expect("first"),
            crate::aixi::common::ObservationKeyMode::First
        );
        assert_eq!(
            parse_observation_key_mode("full_stream").expect("full_stream"),
            crate::aixi::common::ObservationKeyMode::FullStream
        );

        let err = parse_observation_key_mode("streamhash")
            .expect_err("non-canonical observation alias must fail");
        assert!(err.to_string().contains("unknown observation key mode"));
    }
}
