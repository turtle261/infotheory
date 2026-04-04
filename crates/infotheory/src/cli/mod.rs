use super::*;

#[cfg(feature = "vm")]
pub(super) fn parse_shared_memory_policy(v: Option<&str>) -> SharedMemoryPolicy {
    match v.unwrap_or("snapshot") {
        "preserve" | "keep" => SharedMemoryPolicy::Preserve,
        _ => SharedMemoryPolicy::Snapshot,
    }
}

#[cfg(feature = "vm")]
pub(super) fn parse_nyx_environment_config(
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
        parse_nyx_reward_shaping(&vm["reward_shaping"], base_dir)?
    } else if !v["vm_reward_shaping"].is_null() {
        parse_nyx_reward_shaping(&v["vm_reward_shaping"], base_dir)?
    } else if !vm["reward"].is_null() && !vm["reward"]["shaping"].is_null() {
        parse_nyx_reward_shaping(&vm["reward"]["shaping"], base_dir)?
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

pub(super) fn parse_vm_stats_backend(
    cfg: &serde_json::Value,
    root: &serde_json::Value,
    base_dir: &Path,
) -> anyhow::Result<RateBackend> {
    let spec = normalize_vm_stats_backend_spec(cfg, root)?;
    infotheory::spec::parse_rate_backend_json(&spec, base_dir, MAX_MIXTURE_NESTING)
        .map_err(anyhow::Error::msg)
}

#[cfg(feature = "vm")]
pub(super) fn parse_nyx_trace_config(
    v: &serde_json::Value,
) -> anyhow::Result<Option<NyxTraceConfig>> {
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
pub(super) fn parse_nyx_protocol_config(v: &serde_json::Value) -> NyxProtocolConfig {
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
pub(super) fn parse_nyx_actions(v: &serde_json::Value) -> anyhow::Result<NyxActionSource> {
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
pub(super) fn parse_nyx_fuzz_mutator(name: &str) -> Option<NyxFuzzMutator> {
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
pub(super) fn parse_nyx_observation_policy(v: &serde_json::Value) -> NyxObservationPolicy {
    match v["mode"].as_str().unwrap_or("guest") {
        "raw" | "raw-bytes" | "bytes" | "stream" => NyxObservationPolicy::RawOutput,
        "hash" | "output-hash" => NyxObservationPolicy::OutputHash,
        "shared-memory" | "shared_mem" | "shared" => NyxObservationPolicy::SharedMemory,
        _ => NyxObservationPolicy::FromGuest,
    }
}

#[cfg(feature = "vm")]
pub(super) fn parse_nyx_observation_stream_mode(
    v: &serde_json::Value,
) -> NyxObservationStreamMode {
    match v["stream_mode"].as_str().unwrap_or("pad-truncate") {
        "pad" => NyxObservationStreamMode::Pad,
        "truncate" => NyxObservationStreamMode::Truncate,
        _ => NyxObservationStreamMode::PadTruncate,
    }
}

#[cfg(feature = "vm")]
pub(super) fn parse_nyx_observation_pad_byte(v: &serde_json::Value) -> u8 {
    v["pad_byte"].as_u64().unwrap_or(0) as u8
}

#[cfg(feature = "vm")]
pub(super) fn parse_nyx_reward_policy(v: &serde_json::Value) -> anyhow::Result<NyxRewardPolicy> {
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
pub(super) fn parse_nyx_reward_shaping(
    v: &serde_json::Value,
    base_dir: &Path,
) -> anyhow::Result<Option<NyxRewardShaping>> {
    if v.is_null() {
        return Ok(None);
    }
    match v["mode"].as_str().unwrap_or("none") {
        "entropy-reduction" | "entropy_reduction" => {
            let baseline_path = v["baseline_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("vm_reward_shaping.baseline_path is required"))?;
            let baseline_path = crate::spec::resolve_spec_path(base_dir, baseline_path);
            let baseline_bytes = std::fs::read(&baseline_path)?;
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
pub(super) fn parse_nyx_filter(
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

#[cfg(feature = "backend-rwkv")]
pub(super) fn rwkv7_model_path_from_env() -> String {
    env::var("RWKV7_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("Error: RWKV7_MODEL_PATH env var must be set when using rwkv7 backends");
        std::process::exit(1);
    })
}

#[cfg(feature = "backend-mamba")]
pub(super) fn mamba_model_path_from_env() -> String {
    env::var("MAMBA_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("Error: MAMBA_MODEL_PATH env var must be set when using mamba backends");
        std::process::exit(1);
    })
}

pub(super) fn parse_rate_backend(v: &str) -> Option<&'static str> {
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

pub(super) fn parse_compression_backend(v: &str) -> Option<&'static str> {
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

pub(super) fn load_mixture_spec(path: &str) -> anyhow::Result<MixtureSpec> {
    infotheory::spec::load_mixture_spec(path).map_err(anyhow::Error::msg)
}

pub(super) fn load_expert_spec(path: &str) -> anyhow::Result<MixtureExpertSpec> {
    infotheory::spec::load_expert_spec(path).map_err(anyhow::Error::msg)
}

pub(super) fn vm_stats_backend_spec_value(
    root: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let algo = root["algorithm"].as_str().unwrap_or("ctw");
    let ct_depth = root["ct_depth"].as_u64().unwrap_or(20) as usize;
    let spec = match algo {
        "ctw" | "ac-ctw" | "ctw-context-tree" => serde_json::json!({
            "kind": "ctw",
            "depth": ct_depth,
        }),
        "fac-ctw" => serde_json::json!({
            "kind": "fac-ctw",
            "base_depth": ct_depth,
            "num_percept_bits": 8,
            "encoding_bits": 8,
        }),
        "sequitur" => serde_json::json!({
            "kind": "sequitur",
            "context_bytes": root["context_bytes"].as_u64().unwrap_or(64) as usize,
        }),
        "mamba" | "mamba1" => {
            #[cfg(feature = "backend-mamba")]
            {
                serde_json::json!({
                    "kind": "mamba",
                    "model_path": root["mamba_model_path"]
                        .as_str()
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(mamba_model_path_from_env),
                })
            }
            #[cfg(not(feature = "backend-mamba"))]
            {
                return Err(anyhow::anyhow!(
                    "mamba default stats backend requires 'backend-mamba' feature in infotheory"
                ));
            }
        }
        "rosa" | "rosaplus" => serde_json::json!({ "kind": "rosaplus" }),
        "rwkv" | "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            {
                serde_json::json!({
                    "kind": "rwkv7",
                    "model_path": root["rwkv_model_path"]
                        .as_str()
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(rwkv7_model_path_from_env),
                })
            }
            #[cfg(not(feature = "backend-rwkv"))]
            {
                return Err(anyhow::anyhow!(
                    "rwkv7 default stats backend requires 'backend-rwkv' feature in infotheory"
                ));
            }
        }
        "zpaq" => serde_json::json!({
            "kind": "zpaq",
            "method": root["method"].as_str().unwrap_or("2"),
        }),
        "mixture" | "mix" => {
            let spec_path = root["mixture_spec"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("mixture stats backend requires mixture_spec"))?;
            serde_json::json!({
                "kind": "mixture",
                "spec_path": spec_path,
            })
        }
        _ => serde_json::json!({ "kind": "rosaplus" }),
    };
    Ok(spec)
}

pub(super) fn normalize_vm_stats_backend_spec(
    cfg: &serde_json::Value,
    root: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    if cfg.is_null() {
        return vm_stats_backend_spec_value(root);
    }

    let mut spec = if let Some(name) = cfg.as_str() {
        serde_json::json!({ "kind": name })
    } else if let Some(object) = cfg.as_object() {
        serde_json::Value::Object(object.clone())
    } else {
        return Err(anyhow::anyhow!(
            "vm stats backend must be a backend name string or JSON object"
        ));
    };

    let raw_kind = spec["kind"]
        .as_str()
        .or_else(|| spec["name"].as_str())
        .or_else(|| spec["rate_backend"].as_str())
        .unwrap_or("rosaplus");

    let resolved = match infotheory::backends::resolve_rate_backend_name(raw_kind) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => name,
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            return Err(anyhow::anyhow!(
                "rate backend '{canonical}' requires infotheory feature '{feature}'"
            ));
        }
        None => return Err(anyhow::anyhow!("unknown vm stats backend '{raw_kind}'")),
    };

    let obj = spec
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("vm stats backend must be a JSON object"))?;
    obj.insert(
        "kind".to_string(),
        serde_json::Value::String(resolved.to_string()),
    );
    obj.remove("name");
    obj.remove("rate_backend");

    match resolved {
        "ctw" => {
            if !obj.contains_key("depth") && !obj.contains_key("ct_depth") {
                obj.insert("ct_depth".to_string(), serde_json::json!(32usize));
            }
        }
        "fac-ctw" => {
            if !obj.contains_key("base_depth") && !obj.contains_key("ct_depth") {
                obj.insert("ct_depth".to_string(), serde_json::json!(32usize));
            }
            if !obj.contains_key("encoding_bits") {
                obj.insert("encoding_bits".to_string(), serde_json::json!(8usize));
            }
            if !obj.contains_key("num_percept_bits") {
                let obs_bits = root["observation_bits"].as_u64().unwrap_or(16);
                let rew_bits = root["reward_bits"].as_u64().unwrap_or(8);
                obj.insert(
                    "num_percept_bits".to_string(),
                    serde_json::json!(obs_bits + rew_bits),
                );
            }
        }
        "mamba" => {
            #[cfg(feature = "backend-mamba")]
            if !obj.contains_key("method")
                && !obj.contains_key("mamba_method")
                && !obj.contains_key("mamba_model_path")
                && !obj.contains_key("model_path")
            {
                obj.insert(
                    "model_path".to_string(),
                    serde_json::Value::String(
                        root["mamba_model_path"]
                            .as_str()
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(mamba_model_path_from_env),
                    ),
                );
            }
        }
        "rwkv7" => {
            #[cfg(feature = "backend-rwkv")]
            if !obj.contains_key("method")
                && !obj.contains_key("rwkv_method")
                && !obj.contains_key("rwkv_model_path")
                && !obj.contains_key("model_path")
            {
                obj.insert(
                    "model_path".to_string(),
                    serde_json::Value::String(
                        root["rwkv_model_path"]
                            .as_str()
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(rwkv7_model_path_from_env),
                    ),
                );
            }
        }
        "zpaq" => {
            if !obj.contains_key("method") && !obj.contains_key("zpaq_method") {
                obj.insert(
                    "method".to_string(),
                    serde_json::Value::String(root["method"].as_str().unwrap_or("2").to_string()),
                );
            }
        }
        "mixture" => {
            if !obj.contains_key("spec")
                && !obj.contains_key("spec_path")
                && !obj.contains_key("path")
                && !obj.contains_key("mixture_spec")
                && let Some(path) = root["mixture_spec"].as_str()
            {
                obj.insert(
                    "spec_path".to_string(),
                    serde_json::Value::String(path.to_string()),
                );
            }
        }
        "particle" => {
            if !obj.contains_key("spec")
                && !obj.contains_key("spec_path")
                && !obj.contains_key("path")
                && !obj.contains_key("particle_spec")
                && let Some(path) = root["particle_spec"].as_str()
            {
                obj.insert(
                    "spec_path".to_string(),
                    serde_json::Value::String(path.to_string()),
                );
            }
        }
        "calibrated" => {
            if !obj.contains_key("spec")
                && !obj.contains_key("spec_path")
                && !obj.contains_key("path")
                && !obj.contains_key("calibrated_spec")
                && let Some(path) = root["calibrated_spec"].as_str()
            {
                obj.insert(
                    "spec_path".to_string(),
                    serde_json::Value::String(path.to_string()),
                );
            }
        }
        _ => {}
    }

    Ok(spec)
}

pub(super) fn parse_observation_stream_len(v: &serde_json::Value) -> usize {
    v["observation_stream_len"].as_u64().unwrap_or(1) as usize
}

pub(super) fn parse_observation_key_mode(v: &serde_json::Value) -> ObservationKeyMode {
    parse_observation_key_mode_str(v["observation_key_mode"].as_str().unwrap_or("full"))
}

pub(super) fn parse_observation_key_mode_str(s: &str) -> ObservationKeyMode {
    match s {
        "full" | "full-stream" | "stream" => ObservationKeyMode::FullStream,
        "last" => ObservationKeyMode::Last,
        "hash" | "stream-hash" => ObservationKeyMode::StreamHash,
        _ => ObservationKeyMode::First,
    }
}

pub(super) fn parse_observation_stream_len_for_env(
    v: &serde_json::Value,
    env_name: &str,
) -> usize {
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

pub(super) fn parse_observation_key_mode_for_env(
    v: &serde_json::Value,
    env_name: &str,
) -> ObservationKeyMode {
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

pub(super) fn parse_observation_key_mode_for_vm(v: &serde_json::Value) -> ObservationKeyMode {
    if v.is_null() {
        return ObservationKeyMode::FullStream;
    }
    parse_observation_key_mode_str(
        v["key_mode"]
            .as_str()
            .unwrap_or_else(|| v["observation_key_mode"].as_str().unwrap_or("full")),
    )
}

pub(super) fn parse_observation_stream_len_for_vm(v: &serde_json::Value) -> usize {
    if v.is_null() {
        return 1;
    }
    v["stream_len"]
        .as_u64()
        .or_else(|| v["observation_stream_len"].as_u64())
        .unwrap_or(1) as usize
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

pub(super) fn validate_observation_config(
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
pub(super) fn validate_obs_stream_len(expected: usize, actual: usize) -> anyhow::Result<()> {
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

pub(super) fn aiqi_backend_label(config: &AiqiConfig) -> String {
    if let Some(rate_backend) = &config.rate_backend {
        let name = infotheory::mixture::RateBackendPredictor::default_name(
            rate_backend,
            config.rate_backend_max_order,
        );
        format!("rate_backend={name}")
    } else {
        format!("algorithm={}", config.algorithm)
    }
}

pub(super) struct BuiltCtx {
    pub(super) ctx: InfotheoryCtx,
    pub(super) expert_spec_max_order: Option<i64>,
}

pub(super) fn build_ctx(
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
        let shorthand = infotheory::spec::RateBackendShorthandOptions {
            base_dir: std::path::PathBuf::from("."),
            particle_default_if_missing_method: false,
            ..Default::default()
        };
        #[cfg(any(feature = "backend-mamba", feature = "backend-rwkv"))]
        let shorthand = {
            let mut shorthand = shorthand;
            #[cfg(feature = "backend-mamba")]
            if rate_backend == "mamba" && method.is_none() {
                shorthand.default_mamba_model_path = Some(mamba_model_path_from_env());
            }
            #[cfg(feature = "backend-rwkv")]
            if rate_backend == "rwkv7" && method.is_none() {
                shorthand.default_rwkv_model_path = Some(rwkv7_model_path_from_env());
            }
            shorthand
        };
        (
            infotheory::spec::parse_rate_backend_name_method(rate_backend, method, &shorthand)
                .unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }),
            None,
        )
    };

    #[allow(unused_mut)]
    let mut compression_opts = infotheory::spec::CompressionBackendShorthandOptions {
        default_rate_backend: Some(rate_backend.clone()),
        default_framing: infotheory::compression::FramingMode::Raw,
        ..Default::default()
    };
    #[cfg(feature = "backend-rwkv")]
    if compression_backend == "rwkv7"
        && (method.is_none()
            || method
                .map(|value| infotheory::backends::parse_rwkv7_coder(value).is_some())
                .unwrap_or(false))
    {
        compression_opts.default_rwkv_model_path = Some(rwkv7_model_path_from_env());
    }
    let compression_backend = infotheory::spec::parse_compression_backend_name_method(
        compression_backend,
        method,
        Some(rate_backend.clone()),
        &compression_opts,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    BuiltCtx {
        ctx: InfotheoryCtx::new(rate_backend, compression_backend),
        expert_spec_max_order,
    }
}

pub(super) fn read_file(path: &str) -> Vec<u8> {
    match std::fs::read(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            std::process::exit(1);
        }
    }
}

#[cfg_attr(not(feature = "backend-sequitur"), allow(dead_code))]
pub(super) fn parse_hex_bytes(raw: &str) -> anyhow::Result<Vec<u8>> {
    fn nibble(byte: u8) -> anyhow::Result<u8> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            b'A'..=b'F' => Ok(byte - b'A' + 10),
            _ => Err(anyhow::anyhow!("invalid hex digit '{}'", byte as char)),
        }
    }

    let cleaned: Vec<u8> = raw
        .bytes()
        .filter(|b| !matches!(b, b' ' | b'\n' | b'\r' | b'\t' | b'_'))
        .collect();
    if !cleaned.len().is_multiple_of(2) {
        return Err(anyhow::anyhow!(
            "hex input must have an even number of digits"
        ));
    }
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    let mut i = 0usize;
    while i < cleaned.len() {
        let hi = nibble(cleaned[i])?;
        let lo = nibble(cleaned[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

#[cfg_attr(not(feature = "backend-sequitur"), allow(dead_code))]
pub(super) fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0F) as usize] as char);
    }
    out
}

pub(super) fn read_stdin_all_for_generate() -> Vec<u8> {
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

pub(super) fn file_roundtrip_backend(backend: &CompressionBackend) -> CompressionBackend {
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

pub(super) fn maybe_export_online_model(
    export_path: Option<&str>,
    ctx: &InfotheoryCtx,
    parts: &[&[u8]],
) -> anyhow::Result<()> {
    let Some(path) = export_path else {
        return Ok(());
    };

    #[cfg(feature = "backend-rwkv")]
    {
        let rwkv_method = match &ctx.rate_backend {
            RateBackend::Rwkv7Method { method } => Some(method.as_str()),
            _ => match &ctx.compression_backend {
                CompressionBackend::Rate {
                    rate_backend: RateBackend::Rwkv7Method { method },
                    ..
                } => Some(method.as_str()),
                CompressionBackend::Rwkv7 { method, .. } => Some(method.as_str()),
                _ => None,
            },
        };
        if let Some(method) = rwkv_method {
            let mut compressor = rwkvzip::Compressor::new_from_method(method)?;
            let _ = compressor.compress_size_chain(parts, infotheory::coders::CoderType::AC)?;
            compressor.export_online(path)?;
            return Ok(());
        }
    }

    #[cfg(feature = "backend-mamba")]
    {
        let mamba_method = match &ctx.rate_backend {
            RateBackend::MambaMethod { method } => Some(method.as_str()),
            _ => match &ctx.compression_backend {
                CompressionBackend::Rate {
                    rate_backend: RateBackend::MambaMethod { method },
                    ..
                } => Some(method.as_str()),
                _ => None,
            },
        };
        if let Some(method) = mamba_method {
            let mut compressor = mambazip::Compressor::new_from_method(method)?;
            let _ = compressor.compress_size_chain(parts, infotheory::coders::CoderType::AC)?;
            compressor.export_online(path)?;
            return Ok(());
        }
    }

    eprintln!(
        "Warning: --model-export was requested but the current backend does not support \
         online model export. Only RWKV7 and Mamba method-based backends support export."
    );
    Ok(())
}

/// ROSA-based symmetric codelength distance (NCD-like but faster)
/// d_ROSA(x,y) = 0.5 * (H_y(x)/H_x(x) + H_x(y)/H_y(y)) - 1
/// Clamped to [0, 1]
pub(super) fn rosa_distance(x: &[u8], y: &[u8], max_order: i64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }

    let h_x_x = biased_entropy_rate_bytes(x, max_order);
    let h_y_y = biased_entropy_rate_bytes(y, max_order);
    let h_y_x = cross_entropy_rate_bytes(x, y, max_order);
    let h_x_y = cross_entropy_rate_bytes(y, x, max_order);

    if h_x_x < 1e-9 || h_y_y < 1e-9 {
        return 1.0;
    }

    let d = 0.5 * (h_y_x / h_x_x + h_x_y / h_y_y) - 1.0;
    d.clamp(0.0, 1.0)
}

/// Process a single JSON line and return result.
pub(super) fn process_json_line(line: &str) -> String {
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
            let id = if h0 < 1e-9 {
                0.0
            } else {
                ((h0 - h_rate) / h0).clamp(0.0, 1.0)
            };

            format!(
                r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                h0,
                h_rate,
                id,
                data.len()
            )
        }
        "metrics_file" => {
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
                    let id = if h0 < 1e-9 {
                        0.0
                    } else {
                        ((h0 - h_rate) / h0).clamp(0.0, 1.0)
                    };

                    format!(
                        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                        h0,
                        h_rate,
                        id,
                        data.len()
                    )
                }
                Err(e) => format!(r#"{{"error":"failed to read file: {}"}}"#, e),
            }
        }
        "ncd" => {
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

            let results: Vec<String> = texts
                .iter()
                .map(|text| {
                    let data = text.as_bytes();
                    if data.is_empty() {
                        r#"{"h0":0,"h_rate":0,"id":0,"len":0}"#.to_string()
                    } else {
                        let h0 = marginal_entropy_bytes(data);
                        let h_rate = entropy_rate_bytes(data, max_order);
                        let id = if h0 < 1e-9 {
                            0.0
                        } else {
                            ((h0 - h_rate) / h0).clamp(0.0, 1.0)
                        };
                        format!(
                            r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
                            h0,
                            h_rate,
                            id,
                            data.len()
                        )
                    }
                })
                .collect();

            format!(r#"{{"results":[{}]}}"#, results.join(","))
        }
        "ncd_matrix" => {
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
            let matrix = match try_ncd_matrix_bytes(&datas, &method, ncd_variant) {
                Ok(matrix) => matrix,
                Err(err) => return format!(r#"{{"error":"ncd_matrix failed: {}"}}"#, err),
            };
            let n = datas.len();

            let rows: Vec<String> = (0..n)
                .map(|i| {
                    let row: Vec<String> = (0..n)
                        .map(|j| format!("{:.6}", matrix[i * n + j]))
                        .collect();
                    format!("[{}]", row.join(","))
                })
                .collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "rosa_matrix" => {
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

            let rows: Vec<String> = (0..n)
                .map(|i| {
                    let row: Vec<String> = (0..n)
                        .map(|j| format!("{:.6}", matrix[i * n + j]))
                        .collect();
                    format!("[{}]", row.join(","))
                })
                .collect();

            format!(r#"{{"matrix":[{}],"n":{}}}"#, rows.join(","), n)
        }
        "spam_check" => {
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
                return format!(
                    r#"{{"pass":false,"reason":"low_entropy_rate","h_rate":{:.4}}}"#,
                    h_rate
                );
            }

            let id = if h0 < 1e-9 {
                0.0
            } else {
                ((h0 - h_rate) / h0).clamp(0.0, 1.0)
            };
            if id > id_threshold {
                return format!(
                    r#"{{"pass":false,"reason":"high_redundancy","id":{:.4}}}"#,
                    id
                );
            }

            format!(
                r#"{{"pass":true,"h0":{:.4},"h_rate":{:.4},"id":{:.4},"len":{}}}"#,
                h0, h_rate, id, len
            )
        }
        "help" => {
            r#"{"ops":["metrics","metrics_file","ncd","ncd_files","rosa_dist","cross_entropy","batch_metrics","ncd_matrix","rosa_matrix","spam_check"]}"#.to_string()
        }
        _ => format!(r#"{{"error":"unknown op: {}"}}"#, op),
    }
}

pub(super) fn run_batch_mode() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        match line {
            Ok(l) => println!("{}", process_json_line(&l)),
            Err(_) => continue,
        }
    }
}
