use super::*;
use infotheory::error::InfotheoryResult;
#[cfg(all(test, feature = "vm"))]
use std::time::Duration;

#[cfg(feature = "vm")]
#[cfg(test)]
#[allow(dead_code)]
pub(super) fn parse_shared_memory_policy(v: Option<&str>) -> SharedMemoryPolicy {
    match v.unwrap_or("snapshot") {
        "preserve" => SharedMemoryPolicy::Preserve,
        _ => SharedMemoryPolicy::Snapshot,
    }
}

#[cfg(feature = "vm")]
#[cfg(test)]
#[allow(dead_code)]
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
    })?;
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
    })?;
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
        })?;
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

    let mut cfg = NyxVmConfig::default();
    cfg.firecracker_config = firecracker_config;
    cfg.instance_id = instance_id;
    cfg.shared_region_name = shared_region_name;
    cfg.shared_region_size = shared_region_size;
    cfg.shared_memory_policy = shared_memory_policy;
    cfg.step_timeout = Duration::from_millis(step_timeout_ms);
    cfg.boot_timeout = Duration::from_millis(boot_timeout_ms);
    cfg.episode_steps = episode_steps;
    cfg.step_cost = step_cost;
    cfg.observation_policy = observation_policy;
    cfg.observation_bits = observation_bits;
    cfg.observation_stream_len = observation_stream_len;
    cfg.observation_stream_mode = observation_stream_mode;
    cfg.observation_pad_byte = observation_stream_pad_byte;
    cfg.reward_bits = reward_bits;
    cfg.reward_policy = reward_policy;
    cfg.reward_shaping = reward_shaping;
    cfg.action_source = action_source;
    cfg.action_filter = action_filter;
    cfg.protocol = protocol;
    cfg.stats_backend = stats_backend;
    cfg.trace = trace;
    cfg.debug_mode = debug_mode;
    cfg.crash_log = vm["crash_log"].as_str().map(|s| s.to_string());
    Ok(cfg)
}

#[cfg(all(test, feature = "vm"))]
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
#[allow(dead_code)]
#[cfg(test)]
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
        .or_else(|| {
            if v["mode"].as_str() == Some("shared_memory") {
                Some("trace")
            } else {
                None
            }
        })
        .map(|s| s.to_string())
        .or(Some("trace".to_string()));

    let mut trace = NyxTraceConfig::new();
    trace.shared_region_name = shared_region_name;
    trace.max_bytes = max_bytes;
    trace.reset_on_episode = reset_on_episode;
    Ok(Some(trace))
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_protocol_config(
    v: &serde_json::Value,
) -> anyhow::Result<NyxProtocolConfig> {
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
        cfg.wire_encoding = s
            .parse::<NyxPayloadEncoding>()
            .map_err(|_| anyhow::anyhow!("unknown VM wire_encoding '{s}'"))?;
    }
    Ok(cfg)
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_actions(v: &serde_json::Value) -> anyhow::Result<NyxActionSource> {
    let mode = v["mode"].as_str().unwrap_or("literal");
    match mode {
        "fuzz" => {
            let fuzz = if v["fuzz"].is_null() { v } else { &v["fuzz"] };
            let seed_encoding_label = fuzz["seed_encoding"].as_str().unwrap_or("utf8");
            let seed_encoding =
                seed_encoding_label
                    .parse::<NyxPayloadEncoding>()
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "unknown vm_actions.fuzz.seed_encoding '{seed_encoding_label}'"
                        )
                    })?;
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
            let dict_encoding_label = fuzz["dict_encoding"].as_str().unwrap_or("utf8");
            let dict_encoding =
                dict_encoding_label
                    .parse::<NyxPayloadEncoding>()
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "unknown vm_actions.fuzz.dict_encoding '{dict_encoding_label}'"
                        )
                    })?;
            let mut dictionary = Vec::new();
            if let Some(arr) = fuzz["dictionary"].as_array() {
                for item in arr {
                    if let Some(text) = item.as_str() {
                        dictionary.push(dict_encoding.decode(text)?);
                    }
                }
            }
            let rng_seed = fuzz["rng_seed"].as_u64().unwrap_or(0);
            let mut fuzz_cfg = NyxFuzzConfig::new(seeds);
            fuzz_cfg.mutators = mutators;
            fuzz_cfg.min_len = min_len;
            fuzz_cfg.max_len = max_len;
            fuzz_cfg.dictionary = dictionary;
            fuzz_cfg.rng_seed = rng_seed;
            Ok(NyxActionSource::Fuzz(fuzz_cfg))
        }
        _ => {
            let mut actions = Vec::new();
            if let Some(arr) = v["actions"].as_array() {
                for item in arr {
                    if let Some(text) = item.as_str() {
                        let payload = NyxPayloadEncoding::Utf8.decode(text)?;
                        actions.push(NyxActionSpec::new(payload));
                        continue;
                    }
                    let payload = item["payload"].as_str().unwrap_or_default();
                    let encoding_label = item["encoding"].as_str().unwrap_or("utf8");
                    let encoding = encoding_label.parse::<NyxPayloadEncoding>().map_err(|_| {
                        anyhow::anyhow!("unknown vm_actions[].encoding '{encoding_label}'")
                    })?;
                    let payload = encoding.decode(payload)?;
                    let mut spec = NyxActionSpec::new(payload);
                    spec.name = item["name"].as_str().map(|s| s.to_string());
                    actions.push(spec);
                }
            }
            Ok(NyxActionSource::Literal(actions))
        }
    }
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_fuzz_mutator(name: &str) -> Option<NyxFuzzMutator> {
    match name {
        "flip_bit" => Some(NyxFuzzMutator::FlipBit),
        "flip_byte" => Some(NyxFuzzMutator::FlipByte),
        "insert_byte" => Some(NyxFuzzMutator::InsertByte),
        "delete_byte" => Some(NyxFuzzMutator::DeleteByte),
        "splice_seed" => Some(NyxFuzzMutator::SpliceSeed),
        "reset_seed" => Some(NyxFuzzMutator::ResetSeed),
        "havoc" => Some(NyxFuzzMutator::Havoc),
        _ => None,
    }
}

#[cfg(feature = "vm")]
#[cfg(test)]
fn parse_nyx_observation_policy_str(mode: &str) -> anyhow::Result<NyxObservationPolicy> {
    match mode {
        "from_guest" => Ok(NyxObservationPolicy::FromGuest),
        "raw_output" => Ok(NyxObservationPolicy::RawOutput),
        "output_hash" => Ok(NyxObservationPolicy::OutputHash),
        "shared_memory" => Ok(NyxObservationPolicy::SharedMemory),
        other => Err(anyhow::anyhow!("unknown VM observation policy '{other}'")),
    }
}

#[cfg(feature = "vm")]
#[cfg(test)]
pub(super) fn parse_nyx_observation_policy(
    v: &serde_json::Value,
) -> anyhow::Result<NyxObservationPolicy> {
    parse_nyx_observation_policy_str(v["mode"].as_str().unwrap_or("from_guest"))
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_observation_stream_mode(
    v: &serde_json::Value,
) -> anyhow::Result<NyxObservationStreamMode> {
    match v["stream_mode"].as_str().unwrap_or("pad_truncate") {
        "pad" => Ok(NyxObservationStreamMode::Pad),
        "truncate" => Ok(NyxObservationStreamMode::Truncate),
        "pad_truncate" => Ok(NyxObservationStreamMode::PadTruncate),
        other => Err(anyhow::anyhow!("unknown observation stream mode '{other}'")),
    }
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_observation_pad_byte(v: &serde_json::Value) -> u8 {
    v["pad_byte"].as_u64().unwrap_or(0) as u8
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
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
#[allow(dead_code)]
#[cfg(test)]
pub(super) fn parse_nyx_reward_shaping(
    v: &serde_json::Value,
    base_dir: &Path,
) -> anyhow::Result<Option<NyxRewardShaping>> {
    if v.is_null() {
        return Ok(None);
    }
    match v["mode"].as_str().unwrap_or("none") {
        "entropy_reduction" => {
            let baseline_path = v["baseline_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("vm_reward_shaping.baseline_path is required"))?;
            let baseline_path = infotheory::spec::resolve_spec_path(base_dir, baseline_path);
            let baseline_bytes = std::fs::read(&baseline_path)?;
            let scale = v["scale"].as_f64().unwrap_or(10.0);
            let crash_bonus = v["crash_bonus"].as_i64();
            let timeout_bonus = v["timeout_bonus"].as_i64();
            Ok(Some(NyxRewardShaping::EntropyReduction {
                baseline_bytes,
                scale,
                crash_bonus,
                timeout_bonus,
            }))
        }
        "trace_entropy" => {
            let scale = v["scale"].as_f64().unwrap_or(1.0);
            let normalize = v["normalize"].as_bool().unwrap_or(false);
            Ok(Some(NyxRewardShaping::TraceEntropy { scale, normalize }))
        }
        "none" => Ok(None),
        other => Err(anyhow::anyhow!("unknown vm_reward_shaping.mode '{other}'")),
    }
}

#[cfg(feature = "vm")]
#[allow(dead_code)]
#[cfg(test)]
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
    let mut filter = NyxActionFilter::new();
    filter.min_entropy = v["min_entropy"].as_f64();
    filter.max_entropy = v["max_entropy"].as_f64();
    filter.min_intrinsic_dependence = v["min_intrinsic_dependence"].as_f64();
    filter.min_novelty = v["min_novelty"].as_f64();
    filter.novelty_prior = novelty_prior;
    filter.reject_reward = reject_reward;
    Ok(Some(filter))
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

#[cfg(all(test, feature = "vm"))]
pub(super) fn vm_stats_backend_spec_value(
    root: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let raw_kind = root["algorithm"].as_str().unwrap_or("ctw");
    let resolved = match infotheory::backends::resolve_rate_backend_name(raw_kind) {
        Some(infotheory::backends::BackendAvailability::Enabled(name)) => name,
        Some(infotheory::backends::BackendAvailability::Disabled { canonical, feature }) => {
            return Err(anyhow::anyhow!(
                "rate backend '{canonical}' requires infotheory feature '{feature}'"
            ));
        }
        None => {
            return Err(anyhow::anyhow!(
                "unknown stats backend algorithm '{raw_kind}'"
            ));
        }
    };
    let ct_depth = root["ct_depth"].as_u64().unwrap_or(20) as usize;
    let spec = match resolved {
        "ctw" => serde_json::json!({
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
        "mamba" => {
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
        "rosaplus" => serde_json::json!({ "kind": "rosaplus" }),
        "rwkv7" => {
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
        "mixture" => {
            let spec_path = root["mixture_spec"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("mixture stats backend requires mixture_spec"))?;
            serde_json::json!({
                "kind": "mixture",
                "spec_path": spec_path,
            })
        }
        other => return Err(anyhow::anyhow!("unknown stats backend algorithm '{other}'")),
    };
    Ok(spec)
}

#[cfg(all(test, feature = "vm"))]
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

    let raw_kind = spec["kind"].as_str().unwrap_or("rosaplus");

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

    match resolved {
        "ctw" => {
            if !obj.contains_key("depth") {
                obj.insert("depth".to_string(), serde_json::json!(32usize));
            }
        }
        "fac-ctw" => {
            if !obj.contains_key("base_depth") {
                obj.insert("base_depth".to_string(), serde_json::json!(32usize));
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
        "mamba" =>
        {
            #[cfg(feature = "backend-mamba")]
            if !obj.contains_key("method") && !obj.contains_key("model_path") {
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
        "rwkv7" =>
        {
            #[cfg(feature = "backend-rwkv")]
            if !obj.contains_key("method") && !obj.contains_key("model_path") {
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
            if !obj.contains_key("method") {
                obj.insert(
                    "method".to_string(),
                    serde_json::Value::String(root["method"].as_str().unwrap_or("2").to_string()),
                );
            }
        }
        "mixture" => {
            if !obj.contains_key("spec")
                && !obj.contains_key("spec_path")
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

#[cfg(test)]
pub(super) fn parse_observation_stream_len(v: &serde_json::Value) -> usize {
    v["observation_stream_len"].as_u64().unwrap_or(1) as usize
}

#[cfg(test)]
pub(super) fn parse_observation_key_mode(
    v: &serde_json::Value,
) -> anyhow::Result<ObservationKeyMode> {
    parse_observation_key_mode_str(v["observation_key_mode"].as_str().unwrap_or("full_stream"))
}

#[cfg(test)]
pub(super) fn parse_observation_key_mode_str(s: &str) -> anyhow::Result<ObservationKeyMode> {
    match s {
        "first" => Ok(ObservationKeyMode::First),
        "full_stream" => Ok(ObservationKeyMode::FullStream),
        "last" => Ok(ObservationKeyMode::Last),
        "stream_hash" => Ok(ObservationKeyMode::StreamHash),
        other => Err(anyhow::anyhow!("unknown observation key mode '{other}'")),
    }
}

#[cfg(test)]
pub(super) fn parse_observation_stream_len_for_env(v: &serde_json::Value, env_name: &str) -> usize {
    if env_name == "vm" {
        if v["vm_observation"].is_null() {
            parse_observation_stream_len(v)
        } else {
            parse_observation_stream_len_for_vm(&v["vm_observation"])
        }
    } else {
        parse_observation_stream_len(v)
    }
}

#[cfg(test)]
pub(super) fn parse_observation_key_mode_for_env(
    v: &serde_json::Value,
    env_name: &str,
) -> anyhow::Result<ObservationKeyMode> {
    if env_name == "vm" {
        if v["vm_observation"].is_null() {
            parse_observation_key_mode(v)
        } else {
            parse_observation_key_mode_for_vm(&v["vm_observation"])
        }
    } else {
        parse_observation_key_mode(v)
    }
}

#[cfg(test)]
pub(super) fn parse_observation_key_mode_for_vm(
    v: &serde_json::Value,
) -> anyhow::Result<ObservationKeyMode> {
    if v.is_null() {
        return Ok(ObservationKeyMode::FullStream);
    }
    parse_observation_key_mode_str(
        v["key_mode"]
            .as_str()
            .unwrap_or_else(|| v["observation_key_mode"].as_str().unwrap_or("full_stream")),
    )
}

#[cfg(test)]
pub(super) fn parse_observation_stream_len_for_vm(v: &serde_json::Value) -> usize {
    if v.is_null() {
        return 1;
    }
    v["stream_len"]
        .as_u64()
        .or_else(|| v["observation_stream_len"].as_u64())
        .unwrap_or(1) as usize
}

#[cfg(test)]
fn extract_observation_stream_len_raw(v: &serde_json::Value) -> Option<usize> {
    v["observation_stream_len"].as_u64().map(|n| n as usize)
}

#[cfg(test)]
fn extract_vm_observation_stream_len_raw(v: &serde_json::Value) -> Option<usize> {
    if v.is_null() {
        return None;
    }
    v["stream_len"]
        .as_u64()
        .or_else(|| v["observation_stream_len"].as_u64())
        .map(|n| n as usize)
}

#[cfg(test)]
fn extract_observation_key_mode_raw(v: &serde_json::Value) -> Option<ObservationKeyMode> {
    v["observation_key_mode"]
        .as_str()
        .map(parse_observation_key_mode_str)
        .transpose()
        .ok()
        .flatten()
}

#[cfg(test)]
fn extract_vm_observation_key_mode_raw(v: &serde_json::Value) -> Option<ObservationKeyMode> {
    if v.is_null() {
        return None;
    }
    v["key_mode"]
        .as_str()
        .or_else(|| v["observation_key_mode"].as_str())
        .map(parse_observation_key_mode_str)
        .transpose()
        .ok()
        .flatten()
}

#[cfg(test)]
pub(super) fn validate_observation_config(
    env_name: &str,
    v: &serde_json::Value,
    observation_stream_len: usize,
    observation_key_mode: ObservationKeyMode,
) -> anyhow::Result<()> {
    if observation_stream_len == 0 {
        return Err(anyhow::anyhow!("observation_stream_len must be > 0"));
    }
    if env_name == "vm" {
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
            "Warning: observation_key_mode=first collapses multi-symbol observation streams; prefer \"full_stream\" for paper-accurate expectimax."
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

pub(super) struct BuiltCtx {
    pub(super) ctx: InfotheoryCtx,
}

pub(super) fn build_ctx(
    rate_backend: &str,
    compression_backend: &str,
    method: Option<&str>,
    expert_spec_path: Option<&str>,
) -> BuiltCtx {
    let rate_backend = if let Some(path) = expert_spec_path {
        let spec = load_expert_spec(path).unwrap_or_else(|e| {
            eprintln!("Error: failed to load expert spec '{path}': {e}");
            std::process::exit(1);
        });
        spec.backend
    } else {
        let mut shorthand = infotheory::spec::RateBackendShorthandOptions::default();
        shorthand.base_dir = std::path::PathBuf::from(".");
        shorthand.particle_default_if_missing_method = false;
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
        infotheory::spec::parse_rate_backend_name_method(rate_backend, method, &shorthand)
            .unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            })
    };

    #[allow(unused_mut)]
    let mut compression_opts = infotheory::spec::CompressionBackendShorthandOptions::default();
    compression_opts.default_rate_backend = Some(rate_backend.clone());
    compression_opts.default_framing = infotheory::compression::FramingMode::Raw;
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
        ctx: InfotheoryCtx::from_specs(rate_backend, compression_backend).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }),
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

#[cfg(test)]
pub(super) fn file_roundtrip_backend(backend: &CompressionBackend) -> CompressionBackend {
    infotheory::backends::normalize_file_roundtrip_backend(backend)
}

pub(super) fn file_roundtrip_compiled_backend(
    backend: &infotheory::spec::CompiledCompressionBackend,
) -> infotheory::spec::CompiledCompressionBackend {
    infotheory::backends::normalize_file_roundtrip_compiled_backend(backend)
}

pub(super) fn maybe_export_online_model(
    export_path: Option<&str>,
    ctx: &InfotheoryCtx,
    parts: &[&[u8]],
) -> anyhow::Result<()> {
    let Some(path) = export_path else {
        return Ok(());
    };

    #[cfg(not(any(feature = "backend-rwkv", feature = "backend-mamba")))]
    let _ = (ctx, parts, path);

    #[cfg(feature = "backend-rwkv")]
    {
        let rwkv_method = infotheory::backends::rate_backend_method_string_compiled(
            &ctx.rate_backend,
            infotheory::backends::MethodBackendFamily::Rwkv7,
        )
        .or_else(|| {
            infotheory::backends::compression_backend_method_string_compiled(
                &ctx.compression_backend,
                infotheory::backends::MethodBackendFamily::Rwkv7,
            )
        });
        if let Some(method) = rwkv_method {
            let mut compressor = rwkvzip::Compressor::new_from_method(method)?;
            let _ = compressor.compress_size_chain(parts, infotheory::coders::CoderType::AC)?;
            compressor.export_online(path)?;
            return Ok(());
        }
    }

    #[cfg(feature = "backend-mamba")]
    {
        let mamba_method = infotheory::backends::rate_backend_method_string_compiled(
            &ctx.rate_backend,
            infotheory::backends::MethodBackendFamily::Mamba,
        )
        .or_else(|| {
            infotheory::backends::compression_backend_method_string_compiled(
                &ctx.compression_backend,
                infotheory::backends::MethodBackendFamily::Mamba,
            )
        });
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
fn json_error(message: impl std::fmt::Display) -> String {
    serde_json::json!({
        "error": message.to_string(),
    })
    .to_string()
}

fn try_metrics_summary(data: &[u8]) -> InfotheoryResult<(f64, f64, f64, usize)> {
    let h0 = empirical_entropy_bytes(data);
    let h_rate = try_entropy_rate_bytes(data)?;
    let id = if h0 < 1e-9 {
        0.0
    } else {
        ((h0 - h_rate) / h0).clamp(0.0, 1.0)
    };
    Ok((h0, h_rate, id, data.len()))
}

fn format_metrics_json(h0: f64, h_rate: f64, id: f64, len: usize) -> String {
    format!(
        r#"{{"h0":{:.6},"h_rate":{:.6},"id":{:.6},"len":{}}}"#,
        h0, h_rate, id, len
    )
}

fn rosa_distance(x: &[u8], y: &[u8]) -> InfotheoryResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Ok(1.0);
    }

    let h_x_x = try_biased_entropy_rate_bytes(x)?;
    let h_y_y = try_biased_entropy_rate_bytes(y)?;
    let h_y_x = try_cross_entropy_rate_bytes(x, y)?;
    let h_x_y = try_cross_entropy_rate_bytes(y, x)?;

    if h_x_x < 1e-9 || h_y_y < 1e-9 {
        return Ok(1.0);
    }

    Ok((0.5 * (h_y_x / h_x_x + h_x_y / h_y_y) - 1.0).clamp(0.0, 1.0))
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
            let data = text.as_bytes();

            if data.is_empty() {
                return r#"{"error":"empty text"}"#.to_string();
            }

            match try_metrics_summary(data) {
                Ok((h0, h_rate, id, len)) => format_metrics_json(h0, h_rate, id, len),
                Err(err) => json_error(format!("metrics failed: {err}")),
            }
        }
        "metrics_file" => {
            let path = v
                .get("path")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();

            match std::fs::read(&path) {
                Ok(data) => match try_metrics_summary(&data) {
                    Ok((h0, h_rate, id, len)) => format_metrics_json(h0, h_rate, id, len),
                    Err(err) => json_error(format!("metrics_file failed: {err}")),
                },
                Err(err) => json_error(format!("failed to read file: {err}")),
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

            match try_ncd_bytes(x, y, &method, ncd_variant) {
                Ok(ncd) => format!(r#"{{"ncd":{:.6}}}"#, ncd),
                Err(err) => json_error(format!("ncd failed: {err}")),
            }
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

            match try_ncd_paths(&path1, &path2, &method, ncd_variant) {
                Ok(ncd) => format!(r#"{{"ncd":{:.6}}}"#, ncd),
                Err(err) => json_error(format!("ncd_files failed: {err}")),
            }
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

            let x = text1.as_bytes();
            let y = text2.as_bytes();
            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            match rosa_distance(x, y) {
                Ok(dist) => format!(r#"{{"rosa_dist":{:.6}}}"#, dist),
                Err(err) => json_error(format!("rosa_dist failed: {err}")),
            }
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

            let x = text_x.as_bytes();
            let y = text_y.as_bytes();
            if x.is_empty() || y.is_empty() {
                return r#"{"error":"empty text(s)"}"#.to_string();
            }

            match try_cross_entropy_rate_bytes(x, y) {
                Ok(xe) => format!(r#"{{"cross_entropy":{:.6}}}"#, xe),
                Err(err) => json_error(format!("cross_entropy failed: {err}")),
            }
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

            let results: Vec<String> = texts
                .iter()
                .map(|text| {
                    let data = text.as_bytes();
                    if data.is_empty() {
                        r#"{"h0":0,"h_rate":0,"id":0,"len":0}"#.to_string()
                    } else {
                        match try_metrics_summary(data) {
                            Ok((h0, h_rate, id, len)) => format_metrics_json(h0, h_rate, id, len),
                            Err(err) => serde_json::json!({
                                "error": format!("{err}"),
                                "len": data.len(),
                            })
                            .to_string(),
                        }
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
                Err(err) => return json_error(format!("ncd_matrix failed: {err}")),
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

            let n = texts.len();
            let datas: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
            let mut matrix = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let d = if i == j {
                        0.0
                    } else {
                        match rosa_distance(datas[i], datas[j]) {
                            Ok(dist) => dist,
                            Err(err) => {
                                return json_error(format!("rosa_matrix failed: {err}"));
                            }
                        }
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

            let h0 = empirical_entropy_bytes(data);
            if h0 < h0_threshold {
                return format!(r#"{{"pass":false,"reason":"low_entropy","h0":{:.4}}}"#, h0);
            }

            let h_rate = match try_entropy_rate_bytes(data) {
                Ok(h_rate) => h_rate,
                Err(err) => return json_error(format!("spam_check failed: {err}")),
            };
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

#[cfg(test)]
mod non_vm_tests {
    use super::*;
    use serde_json::Value;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(label: &str, ext: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "infotheory-cli-tests-{label}-{}-{nonce}.{ext}",
            std::process::id()
        ))
    }

    fn parse_json_output(line: &str) -> Value {
        serde_json::from_str(line).expect("output should be valid json")
    }

    #[test]
    fn hex_helpers_roundtrip_and_reject_invalid_inputs() {
        let parsed = parse_hex_bytes("00 ff_10\n7A").expect("hex string should parse");
        assert_eq!(parsed, vec![0x00, 0xff, 0x10, 0x7a]);
        assert_eq!(bytes_to_hex(&parsed), "00ff107a");

        let err = parse_hex_bytes("abc").expect_err("odd hex digit count must fail");
        assert!(
            err.to_string()
                .contains("hex input must have an even number of digits"),
            "unexpected error: {err}"
        );

        let err = parse_hex_bytes("0g").expect_err("invalid digit must fail");
        assert!(
            err.to_string().contains("invalid hex digit 'g'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn process_json_line_reports_help_and_input_errors() {
        let help = parse_json_output(&process_json_line(r#"{ "op": "help" }"#));
        let ops = help["ops"].as_array().expect("help ops array");
        assert!(ops.iter().any(|value| value == "metrics"));
        assert!(ops.iter().any(|value| value == "spam_check"));

        let empty = parse_json_output(&process_json_line("   "));
        assert_eq!(empty["error"], "empty input");

        let invalid = parse_json_output(&process_json_line("{ invalid"));
        assert!(
            invalid["error"]
                .as_str()
                .expect("error string")
                .contains("invalid json"),
            "unexpected invalid-json output: {invalid}"
        );

        let unknown = parse_json_output(&process_json_line(r#"{ "op": "nope" }"#));
        assert_eq!(unknown["error"], "unknown op: nope");
    }

    #[test]
    fn process_json_line_handles_metrics_batch_and_spam_semantics() {
        let metrics = parse_json_output(&process_json_line(
            r#"{ "op": "metrics", "text": "banana bandana" }"#,
        ));
        assert_eq!(metrics["len"], 14);
        assert!(metrics["h0"].as_f64().expect("h0") >= 0.0);
        assert!(metrics["h_rate"].as_f64().expect("h_rate") >= 0.0);

        let batch = parse_json_output(&process_json_line(
            r#"{ "op": "batch_metrics", "texts": ["abcabcabc", ""] }"#,
        ));
        let results = batch["results"].as_array().expect("batch results");
        assert_eq!(results.len(), 2);
        assert_eq!(results[1]["len"], 0);
        assert_eq!(results[1]["h_rate"], 0);

        let too_short = parse_json_output(&process_json_line(
            r#"{ "op": "spam_check", "text": "tiny", "min_len": 10 }"#,
        ));
        assert_eq!(too_short["pass"], false);
        assert_eq!(too_short["reason"], "too_short");
        assert_eq!(too_short["len"], 4);

        let pass = parse_json_output(&process_json_line(
            r#"{ "op": "spam_check", "text": "bananas foster waffle cartography", "min_len": 8, "h0_min": 0.0, "h_rate_min": 0.0, "id_max": 1.0 }"#,
        ));
        assert_eq!(pass["pass"], true);
        assert_eq!(pass["len"], 33);
    }

    #[test]
    fn process_json_line_emits_structured_matrix_and_file_results() {
        let file_path = unique_temp_path("metrics-file", "txt");
        fs::write(&file_path, b"structured metrics fixture").expect("write metrics file");

        let metrics_file = parse_json_output(&process_json_line(&format!(
            r#"{{ "op": "metrics_file", "path": "{}" }}"#,
            file_path.display()
        )));
        assert_eq!(metrics_file["len"], 26);
        assert!(metrics_file["id"].as_f64().expect("id") >= 0.0);

        let ncd_matrix = parse_json_output(&process_json_line(
            r#"{ "op": "ncd_matrix", "texts": ["aaaa", "aaab"] }"#,
        ));
        assert_eq!(ncd_matrix["n"], 2);
        let matrix = ncd_matrix["matrix"].as_array().expect("matrix rows");
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0][0], 0.0);
        assert_eq!(matrix[1][1], 0.0);

        let rosa_matrix = parse_json_output(&process_json_line(
            r#"{ "op": "rosa_matrix", "texts": ["alpha alpha", "alpha beta"] }"#,
        ));
        assert_eq!(rosa_matrix["n"], 2);
        let rosa_rows = rosa_matrix["matrix"].as_array().expect("rosa matrix rows");
        assert_eq!(rosa_rows.len(), 2);
        assert_eq!(rosa_rows[0][0], 0.0);
        assert_eq!(rosa_rows[1][1], 0.0);

        let _ = fs::remove_file(file_path);
    }

    #[test]
    fn process_json_line_covers_pairwise_ops_and_contract_errors() {
        let ncd = parse_json_output(&process_json_line(
            r#"{ "op": "ncd", "text1": "abracadabra", "text2": "alakazam", "method": "5", "variant": "sym_cons" }"#,
        ));
        assert!(ncd["ncd"].as_f64().expect("ncd value").is_finite());

        let cross = parse_json_output(&process_json_line(
            r#"{ "op": "cross_entropy", "text_x": "abracadabra", "text_y": "alakazam" }"#,
        ));
        assert!(
            cross["cross_entropy"]
                .as_f64()
                .expect("cross entropy value")
                .is_finite()
        );

        let rosa = parse_json_output(&process_json_line(
            r#"{ "op": "rosa_dist", "text1": "alpha alpha alpha", "text2": "alpha beta alpha" }"#,
        ));
        let rosa_dist = rosa["rosa_dist"].as_f64().expect("rosa distance value");
        assert!(rosa_dist.is_finite());
        assert!((0.0..=1.0).contains(&rosa_dist));

        let left_path = unique_temp_path("ncd-left", "txt");
        let right_path = unique_temp_path("ncd-right", "txt");
        fs::write(&left_path, b"left fixture bytes").expect("write left fixture");
        fs::write(&right_path, b"right fixture bytes").expect("write right fixture");

        let ncd_files = parse_json_output(&process_json_line(&format!(
            r#"{{ "op": "ncd_files", "path1": "{}", "path2": "{}", "method": "5", "variant": "cons" }}"#,
            left_path.display(),
            right_path.display()
        )));
        assert!(
            ncd_files["ncd"]
                .as_f64()
                .expect("ncd file value")
                .is_finite()
        );

        let empty_ncd = parse_json_output(&process_json_line(
            r#"{ "op": "ncd", "text1": "", "text2": "non-empty" }"#,
        ));
        assert_eq!(empty_ncd["error"], "empty text(s)");

        let empty_cross = parse_json_output(&process_json_line(
            r#"{ "op": "cross_entropy", "text_x": "", "text_y": "non-empty" }"#,
        ));
        assert_eq!(empty_cross["error"], "empty text(s)");

        let empty_rosa = parse_json_output(&process_json_line(
            r#"{ "op": "rosa_dist", "text1": "", "text2": "non-empty" }"#,
        ));
        assert_eq!(empty_rosa["error"], "empty text(s)");

        let empty_metrics =
            parse_json_output(&process_json_line(r#"{ "op": "metrics", "text": "" }"#));
        assert_eq!(empty_metrics["error"], "empty text");

        let missing_metrics_file = parse_json_output(&process_json_line(
            r#"{ "op": "metrics_file", "path": "/definitely/missing/file/path" }"#,
        ));
        assert!(
            missing_metrics_file["error"]
                .as_str()
                .expect("metrics_file error string")
                .contains("failed to read file")
        );

        let _ = fs::remove_file(left_path);
        let _ = fs::remove_file(right_path);
    }

    #[test]
    fn process_json_line_spam_check_reasons_cover_entropy_guards() {
        let low_entropy = parse_json_output(&process_json_line(
            r#"{ "op": "spam_check", "text": "aaaaaaaaaaaa", "min_len": 4, "h0_min": 3.0, "h_rate_min": 0.0, "id_max": 1.0 }"#,
        ));
        assert_eq!(low_entropy["pass"], false);
        assert_eq!(low_entropy["reason"], "low_entropy");

        let low_entropy_rate = parse_json_output(&process_json_line(
            r#"{ "op": "spam_check", "text": "abcdefghijklmno", "min_len": 4, "h0_min": 0.0, "h_rate_min": 1000.0, "id_max": 1.0 }"#,
        ));
        assert_eq!(low_entropy_rate["pass"], false);
        assert_eq!(low_entropy_rate["reason"], "low_entropy_rate");

        let high_redundancy = parse_json_output(&process_json_line(
            r#"{ "op": "spam_check", "text": "abababababababab", "min_len": 4, "h0_min": 0.0, "h_rate_min": 0.0, "id_max": -1.0 }"#,
        ));
        assert_eq!(high_redundancy["pass"], false);
        assert_eq!(high_redundancy["reason"], "high_redundancy");
    }
}

#[cfg(all(test, feature = "vm"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_nyx_observation_policy_accepts_canonical_names() {
        let cases = [
            ("from_guest", NyxObservationPolicy::FromGuest),
            ("output_hash", NyxObservationPolicy::OutputHash),
            ("raw_output", NyxObservationPolicy::RawOutput),
            ("shared_memory", NyxObservationPolicy::SharedMemory),
        ];

        for (mode, expected) in cases {
            let parsed =
                parse_nyx_observation_policy(&json!({ "mode": mode })).expect("parse mode");
            assert!(
                std::mem::discriminant(&parsed) == std::mem::discriminant(&expected),
                "mode {mode} parsed as {parsed:?}"
            );
        }

        let err = parse_nyx_observation_policy(&json!({ "mode": "guest" }))
            .expect_err("legacy alias should be rejected");
        assert!(
            err.to_string().contains("unknown VM observation policy"),
            "unexpected error: {err}"
        );
    }
}
