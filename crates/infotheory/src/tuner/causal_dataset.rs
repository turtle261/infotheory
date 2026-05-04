use super::*;

pub(super) fn load_dataset(path: &Path) -> Result<LoadedDataset, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read input_asset '{}': {err}", path.display()))?;
    if let Ok(Value::Object(object)) = serde_json::from_slice::<Value>(&bytes) {
        if object.contains_key("events") {
            let value = Value::Object(object);
            return lower_interactive_trace_dataset(path, bytes.len(), &value);
        }
        if object.contains_key("examples") || object.contains_key("prefixes") {
            let value = Value::Object(object);
            return lower_causal_prefix_dataset(path, bytes.len(), &value);
        }
        return Err(
            "JSON object input_asset must match a canonical tuner causal dataset kind: provide 'events' or 'examples'/'prefixes'"
                .to_string(),
        );
    }
    let hash = crc32_hex(&bytes);
    Ok(LoadedDataset {
        kind: DatasetKind::PassiveBytes,
        objective_target: ObjectiveTarget::PassiveAc,
        lowering_version: PASSIVE_DATASET_LOWERING_VERSION,
        codec_hash: "passive-identity-bytes".to_string(),
        event_grammar_hash: "passive-target-only-byte-stream".to_string(),
        target_domain_support_hash: crc32_hex(b"passive-byte-alphabet"),
        causal_header_profile_hash: crc32_hex(b"passive-none"),
        target_size_function: "passive-bytes-len",
        canonical_content_hash: hash,
        lowered_skeleton_hash: crc32_hex(b"passive-bytes-target-only"),
        resolved_path: path.to_string_lossy().to_string(),
        source_size_bytes: bytes.len(),
        dataset_units: bytes.len() as f64,
        target_events: usize::from(!bytes.is_empty()),
        events: Vec::new(),
        causal_profile: None,
        raw_bytes: bytes,
    })
}

fn lower_interactive_trace_dataset(
    path: &Path,
    source_size_bytes: usize,
    value: &Value,
) -> Result<LoadedDataset, String> {
    let (header_profile, domain_supports) = parse_causal_header_profile(
        value,
        "interactive trace dataset",
        CausalPayloadKind::Events,
    )?;
    let events_value = value
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "interactive trace dataset requires an array field 'events'".to_string())?;
    let events = events_value
        .iter()
        .enumerate()
        .map(|(index, event)| parse_lowered_event(event, &format!("events[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    lowered_dataset_from_events(
        DatasetKind::InteractiveTrace,
        ObjectiveTarget::InteractiveCausalAc,
        INTERACTIVE_TRACE_LOWERING_VERSION,
        path,
        source_size_bytes,
        value,
        header_profile,
        domain_supports,
        events,
        "interactive-trace-target-bytes",
    )
}

fn lower_causal_prefix_dataset(
    path: &Path,
    source_size_bytes: usize,
    value: &Value,
) -> Result<LoadedDataset, String> {
    let (header_profile, domain_supports) = parse_causal_header_profile(
        value,
        "causal-prefix dataset",
        CausalPayloadKind::ExamplesOrPrefixes,
    )?;
    let examples_value = value
        .get("examples")
        .or_else(|| value.get("prefixes"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "causal-prefix dataset requires an array field 'examples' or 'prefixes'".to_string()
        })?;
    let mut events = Vec::<LoweredCausalEvent>::new();
    for (index, example) in examples_value.iter().enumerate() {
        let object = example
            .as_object()
            .ok_or_else(|| format!("examples[{index}] must be an object"))?;
        events.push(LoweredCausalEvent::Reset);
        if let Some(history) = object.get("history").and_then(Value::as_array) {
            for (history_index, event) in history.iter().enumerate() {
                let replay = parse_lowered_event(
                    event,
                    &format!("examples[{index}].history[{history_index}]"),
                )?;
                if matches!(replay, LoweredCausalEvent::Target { .. }) {
                    return Err(format!(
                        "examples[{index}].history[{history_index}] must replay targets with observe_target_no_score, not charged target events"
                    ));
                }
                events.push(replay);
            }
        }
        if let Some(action) = object.get("action") {
            events.push(LoweredCausalEvent::Context {
                channel: "action".to_string(),
                bytes: payload_bytes(action, &format!("examples[{index}].action"))?,
            });
        }
        let target = object
            .get("target")
            .or_else(|| object.get("percept"))
            .ok_or_else(|| format!("examples[{index}] requires 'target' or 'percept'"))?;
        let weight = object
            .get("weight")
            .map(|raw| {
                raw.as_f64()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| format!("examples[{index}].weight must be finite and > 0"))
            })
            .transpose()?
            .unwrap_or(1.0);
        events.push(LoweredCausalEvent::Target {
            channel: required_nonempty_string_field(
                object.get("channel"),
                &format!("examples[{index}].channel"),
            )?,
            domain: required_nonempty_string_field(
                object.get("domain"),
                &format!("examples[{index}].domain"),
            )?,
            bytes: payload_bytes(target, &format!("examples[{index}].target"))?,
            weight,
        });
    }
    lowered_dataset_from_events(
        DatasetKind::CausalPrefixDataset,
        ObjectiveTarget::InteractiveCausalAc,
        CAUSAL_PREFIX_LOWERING_VERSION,
        path,
        source_size_bytes,
        value,
        header_profile,
        domain_supports,
        events,
        "weighted-target-bytes-sum",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CausalPayloadKind {
    Events,
    ExamplesOrPrefixes,
}

fn parse_causal_header_profile(
    value: &Value,
    label: &str,
    payload_kind: CausalPayloadKind,
) -> Result<(CausalHeaderProfile, BTreeMap<String, CausalTargetDomain>), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let allowed = match payload_kind {
        CausalPayloadKind::Events => [
            "schema_version",
            "environment_id",
            "environment_config_crc32",
            "codec_hash",
            "reset_convention",
            "action_alphabet",
            "percept_schema",
            "reward_encoding",
            "terminal_encoding",
            "collection_policy",
            "target_domains",
            "event_grammar",
            "events",
        ]
        .as_slice(),
        CausalPayloadKind::ExamplesOrPrefixes => [
            "schema_version",
            "environment_id",
            "environment_config_crc32",
            "codec_hash",
            "reset_convention",
            "action_alphabet",
            "percept_schema",
            "reward_encoding",
            "terminal_encoding",
            "collection_policy",
            "target_domains",
            "event_grammar",
            "examples",
            "prefixes",
        ]
        .as_slice(),
    };
    ensure_known_fields_in_object(object, allowed, label)?;

    let schema_version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} requires schema_version: 1"))?;
    if schema_version != 1 {
        return Err(format!("{label} schema_version must be 1"));
    }
    let _environment_id =
        required_nonempty_string_field(object.get("environment_id"), "environment_id")?;
    let env_crc32 = required_nonempty_string_field(
        object.get("environment_config_crc32"),
        "environment_config_crc32",
    )?;
    if !is_lower_hex_crc32(&env_crc32) {
        return Err(format!(
            "{label}.environment_config_crc32 must be an 8-character lowercase hex CRC32"
        ));
    }
    let _codec_hash = required_nonempty_string_field(object.get("codec_hash"), "codec_hash")?;
    let reset_convention =
        required_nonempty_string_field(object.get("reset_convention"), "reset_convention")?;
    if reset_convention != "reset-before-episode" {
        return Err(format!(
            "{label}.reset_convention must be 'reset-before-episode'"
        ));
    }

    let domains = parse_causal_target_domains(
        object
            .get("target_domains")
            .ok_or_else(|| format!("{label} requires target_domains"))?,
    )?;
    let action_alphabet_size = parse_action_alphabet_header(
        object
            .get("action_alphabet")
            .ok_or_else(|| format!("{label} requires action_alphabet"))?,
    )?;
    let percept_channels = parse_percept_schema_header(
        object
            .get("percept_schema")
            .ok_or_else(|| format!("{label} requires percept_schema"))?,
    )?;
    let reward_channel = parse_single_encoding_channel_header(
        object
            .get("reward_encoding")
            .ok_or_else(|| format!("{label} requires reward_encoding"))?,
        "reward_encoding",
    )?;
    let terminal_channel = parse_single_encoding_channel_header(
        object
            .get("terminal_encoding")
            .ok_or_else(|| format!("{label} requires terminal_encoding"))?,
        "terminal_encoding",
    )?;
    let collection_policy =
        required_nonempty_string_field(object.get("collection_policy"), "collection_policy")?;
    let event_grammar = parse_event_grammar_header(
        object
            .get("event_grammar")
            .ok_or_else(|| format!("{label} requires event_grammar"))?,
    )?;

    validate_header_profile_consistency(
        &domains,
        &event_grammar,
        &percept_channels,
        &reward_channel,
        &terminal_channel,
    )?;

    if payload_kind == CausalPayloadKind::ExamplesOrPrefixes
        && object.get("examples").is_some()
        && object.get("prefixes").is_some()
    {
        return Err(format!(
            "{label} must declare exactly one of 'examples' or 'prefixes', not both"
        ));
    }

    let profile_hash = causal_header_profile_hash(
        action_alphabet_size,
        &collection_policy,
        &percept_channels,
        &reward_channel,
        &terminal_channel,
        &event_grammar,
    )?;
    Ok((
        CausalHeaderProfile {
            action_alphabet_size,
            collection_policy,
            percept_channels,
            reward_channel,
            terminal_channel,
            event_grammar,
            profile_hash,
        },
        domains,
    ))
}

fn ensure_known_fields_in_object(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field '{key}'"));
        }
    }
    Ok(())
}

fn parse_action_alphabet_header(value: &Value) -> Result<usize, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "action_alphabet must be an object".to_string())?;
    ensure_known_fields_in_object(object, &["size"], "action_alphabet")?;
    let size = object
        .get("size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "action_alphabet.size must be an integer".to_string())?;
    let size = usize::try_from(size).map_err(|_| "action_alphabet.size does not fit usize")?;
    if size == 0 {
        return Err("action_alphabet.size must be >= 1".to_string());
    }
    Ok(size)
}

fn parse_percept_schema_header(value: &Value) -> Result<BTreeSet<CausalChannelDomain>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "percept_schema must be an object".to_string())?;
    ensure_known_fields_in_object(object, &["encoding", "channels"], "percept_schema")?;
    let encoding = object
        .get("encoding")
        .and_then(Value::as_str)
        .ok_or_else(|| "percept_schema.encoding is required".to_string())?;
    if encoding != "bytes" {
        return Err("percept_schema.encoding must be 'bytes'".to_string());
    }
    let channels = object
        .get("channels")
        .and_then(Value::as_array)
        .ok_or_else(|| "percept_schema.channels must be an array".to_string())?;
    if channels.is_empty() {
        return Err("percept_schema.channels must contain at least one entry".to_string());
    }
    let mut out = BTreeSet::<CausalChannelDomain>::new();
    for (index, value) in channels.iter().enumerate() {
        let pair = parse_channel_domain_pair(value, &format!("percept_schema.channels[{index}]"))?;
        if !out.insert(pair.clone()) {
            return Err(format!(
                "percept_schema.channels[{index}] duplicates ({}, {})",
                pair.channel, pair.domain
            ));
        }
    }
    Ok(out)
}

fn parse_single_encoding_channel_header(
    value: &Value,
    label: &str,
) -> Result<CausalChannelDomain, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    ensure_known_fields_in_object(object, &["encoding", "channel", "domain"], label)?;
    let encoding = object
        .get("encoding")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.encoding is required"))?;
    if encoding != "bytes" {
        return Err(format!("{label}.encoding must be 'bytes'"));
    }
    Ok(CausalChannelDomain {
        channel: required_nonempty_string_field(
            object.get("channel"),
            &format!("{label}.channel"),
        )?,
        domain: required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?,
    })
}

fn parse_event_grammar_header(value: &Value) -> Result<CausalEventGrammar, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "event_grammar must be an object".to_string())?;
    ensure_known_fields_in_object(
        object,
        &["context_channels", "observe_target_no_score", "target"],
        "event_grammar",
    )?;
    let context_channels = object
        .get("context_channels")
        .and_then(Value::as_array)
        .ok_or_else(|| "event_grammar.context_channels must be an array".to_string())?;
    let mut context = BTreeSet::<String>::new();
    for (index, raw) in context_channels.iter().enumerate() {
        let channel = raw
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                format!(
                    "event_grammar.context_channels[{index}] must be a non-empty channel string"
                )
            })?
            .to_string();
        if !context.insert(channel.clone()) {
            return Err(format!(
                "event_grammar.context_channels[{index}] duplicates '{channel}'"
            ));
        }
    }
    let observe = parse_grammar_channel_domains(
        object
            .get("observe_target_no_score")
            .ok_or_else(|| "event_grammar.observe_target_no_score is required".to_string())?,
        "event_grammar.observe_target_no_score",
    )?;
    let target = parse_grammar_channel_domains(
        object
            .get("target")
            .ok_or_else(|| "event_grammar.target is required".to_string())?,
        "event_grammar.target",
    )?;
    Ok(CausalEventGrammar {
        context_channels: context,
        observe_target_no_score: observe,
        target,
    })
}

fn parse_grammar_channel_domains(
    value: &Value,
    label: &str,
) -> Result<BTreeSet<CausalChannelDomain>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    let mut out = BTreeSet::<CausalChannelDomain>::new();
    for (index, entry) in array.iter().enumerate() {
        let pair = parse_channel_domain_pair(entry, &format!("{label}[{index}]"))?;
        if !out.insert(pair.clone()) {
            return Err(format!(
                "{label}[{index}] duplicates ({}, {})",
                pair.channel, pair.domain
            ));
        }
    }
    Ok(out)
}

fn parse_channel_domain_pair(value: &Value, label: &str) -> Result<CausalChannelDomain, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    ensure_known_fields_in_object(object, &["channel", "domain"], label)?;
    Ok(CausalChannelDomain {
        channel: required_nonempty_string_field(
            object.get("channel"),
            &format!("{label}.channel"),
        )?,
        domain: required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?,
    })
}

fn validate_header_profile_consistency(
    domains: &BTreeMap<String, CausalTargetDomain>,
    event_grammar: &CausalEventGrammar,
    percept_channels: &BTreeSet<CausalChannelDomain>,
    reward_channel: &CausalChannelDomain,
    terminal_channel: &CausalChannelDomain,
) -> Result<(), String> {
    for pair in event_grammar
        .observe_target_no_score
        .iter()
        .chain(event_grammar.target.iter())
    {
        if !domains.contains_key(&pair.domain) {
            return Err(format!(
                "event_grammar references undeclared target domain '{}'",
                pair.domain
            ));
        }
    }
    for pair in percept_channels {
        if !event_grammar.target.contains(pair) {
            return Err(format!(
                "percept_schema channel/domain ({}, {}) must appear in event_grammar.target",
                pair.channel, pair.domain
            ));
        }
    }
    if !event_grammar.target.contains(reward_channel) {
        return Err(format!(
            "reward_encoding channel/domain ({}, {}) must appear in event_grammar.target",
            reward_channel.channel, reward_channel.domain
        ));
    }
    if !event_grammar.target.contains(terminal_channel) {
        return Err(format!(
            "terminal_encoding channel/domain ({}, {}) must appear in event_grammar.target",
            terminal_channel.channel, terminal_channel.domain
        ));
    }
    Ok(())
}

fn causal_header_profile_hash(
    action_alphabet_size: usize,
    collection_policy: &str,
    percept_channels: &BTreeSet<CausalChannelDomain>,
    reward_channel: &CausalChannelDomain,
    terminal_channel: &CausalChannelDomain,
    event_grammar: &CausalEventGrammar,
) -> Result<String, String> {
    let value = serde_json::json!({
        "action_alphabet_size": action_alphabet_size,
        "collection_policy": collection_policy,
        "percept_channels": percept_channels
            .iter()
            .map(|pair| {
                serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
            })
            .collect::<Vec<_>>(),
        "reward_channel": {"channel": reward_channel.channel.as_str(), "domain": reward_channel.domain.as_str()},
        "terminal_channel": {"channel": terminal_channel.channel.as_str(), "domain": terminal_channel.domain.as_str()},
        "event_grammar": {
            "context_channels": event_grammar.context_channels.iter().collect::<Vec<_>>(),
            "observe_target_no_score": event_grammar.observe_target_no_score
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
            "target": event_grammar.target
                .iter()
                .map(|pair| {
                    serde_json::json!({"channel": pair.channel.as_str(), "domain": pair.domain.as_str()})
                })
                .collect::<Vec<_>>(),
        }
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|err| format!("failed to encode causal header profile hash: {err}"))?;
    Ok(crc32_hex(&bytes))
}

fn parse_causal_target_domains(
    value: &Value,
) -> Result<BTreeMap<String, CausalTargetDomain>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "target_domains must be an object".to_string())?;
    if object.is_empty() {
        return Err("target_domains must declare at least one target domain".to_string());
    }
    object
        .iter()
        .map(|(domain, spec)| {
            if domain.trim().is_empty() {
                return Err("target_domains contains an empty domain tag".to_string());
            }
            let spec_object = spec
                .as_object()
                .ok_or_else(|| format!("target_domains.{domain} must be an object"))?;
            let kind = spec_object
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("target_domains.{domain}.kind is required"))?;
            match kind {
                "byte_alphabet" => {
                    ensure_known_causal_domain_fields(
                        spec_object,
                        &["kind"],
                        &format!("target_domains.{domain}"),
                    )?;
                    Ok((domain.clone(), CausalTargetDomain::ByteAlphabet))
                }
                "enumerated_payloads" => {
                    ensure_known_causal_domain_fields(
                        spec_object,
                        &["kind", "payloads"],
                        &format!("target_domains.{domain}"),
                    )?;
                    let payloads_value = spec_object
                        .get("payloads")
                        .and_then(Value::as_array)
                        .ok_or_else(|| {
                            format!("target_domains.{domain}.payloads must be an array")
                        })?;
                    if payloads_value.is_empty() {
                        return Err(format!(
                            "target_domains.{domain}.payloads must contain at least one payload"
                        ));
                    }
                    let mut seen = BTreeSet::<Vec<u8>>::new();
                    let mut payloads = Vec::<Vec<u8>>::with_capacity(payloads_value.len());
                    for (index, payload) in payloads_value.iter().enumerate() {
                        let bytes = payload_bytes(
                            payload,
                            &format!("target_domains.{domain}.payloads[{index}]"),
                        )?;
                        if !seen.insert(bytes.clone()) {
                            return Err(format!(
                                "target_domains.{domain}.payloads[{index}] duplicates an enumerated payload"
                            ));
                        }
                        payloads.push(bytes);
                    }
                    Ok((domain.clone(), CausalTargetDomain::EnumeratedPayloads { payloads }))
                }
                other => Err(format!(
                    "target_domains.{domain}.kind has unknown target-domain support kind '{other}'"
                )),
            }
        })
        .collect()
}

fn ensure_known_causal_domain_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!(
                "{label} contains unknown target-domain field '{key}'"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lowered_dataset_from_events(
    kind: DatasetKind,
    objective_target: ObjectiveTarget,
    lowering_version: &'static str,
    path: &Path,
    source_size_bytes: usize,
    source_value: &Value,
    header_profile: CausalHeaderProfile,
    domain_supports: BTreeMap<String, CausalTargetDomain>,
    events: Vec<LoweredCausalEvent>,
    target_size_function: &'static str,
) -> Result<LoadedDataset, String> {
    let canonical_bytes = serde_json::to_vec(source_value)
        .map_err(|err| format!("failed to canonicalize causal dataset JSON: {err}"))?;
    let canonical_content_hash = crc32_hex(&canonical_bytes);
    let normalized_events = expand_byte_alphabet_events(events, &domain_supports)?;
    validate_events_against_causal_profile(&normalized_events, &domain_supports, &header_profile)?;
    let mut charged = Vec::<u8>::new();
    let mut target_events = 0usize;
    let mut dataset_units = 0.0f64;
    for event in &normalized_events {
        if let LoweredCausalEvent::Target { bytes, weight, .. } = event {
            charged.extend_from_slice(bytes);
            target_events = target_events.saturating_add(1);
            dataset_units += (*weight) * (bytes.len() as f64);
        }
    }
    let skeleton_bytes = lowered_event_skeleton_bytes(&normalized_events)?;
    let domain_support_bytes = causal_domain_support_bytes(&domain_supports)?;
    let domain_support_hash = crc32_hex(&domain_support_bytes);
    let channel_set = causal_channel_set(&normalized_events);
    Ok(LoadedDataset {
        kind,
        objective_target,
        lowering_version,
        codec_hash: causal_dataset_string_field(source_value, "codec_hash")
            .unwrap_or_else(|| "json-causal-byte-events-v1".to_string()),
        event_grammar_hash: crc32_hex(&skeleton_bytes),
        target_domain_support_hash: domain_support_hash.clone(),
        causal_header_profile_hash: header_profile.profile_hash.clone(),
        target_size_function,
        canonical_content_hash,
        lowered_skeleton_hash: crc32_hex(&skeleton_bytes),
        resolved_path: path.to_string_lossy().to_string(),
        source_size_bytes,
        raw_bytes: charged,
        events: normalized_events,
        causal_profile: Some(CausalEvaluationProfile {
            domains: domain_supports,
            channel_set,
            domain_support_hash,
            byte_alphabet_symbol_width: 1,
            header_profile_hash: header_profile.profile_hash,
            event_grammar: header_profile.event_grammar,
            action_alphabet_size: header_profile.action_alphabet_size,
            collection_policy: header_profile.collection_policy,
            percept_channels: header_profile.percept_channels,
            reward_channel: header_profile.reward_channel,
            terminal_channel: header_profile.terminal_channel,
        }),
        dataset_units,
        target_events,
    })
}

fn parse_lowered_event(value: &Value, label: &str) -> Result<LoweredCausalEvent, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.kind is required"))?;
    match kind {
        "reset" => {
            ensure_known_event_fields(object, &["kind"], label)?;
            Ok(LoweredCausalEvent::Reset)
        }
        "context" => {
            ensure_known_event_fields(object, &["kind", "channel", "bytes"], label)?;
            Ok(LoweredCausalEvent::Context {
                channel: required_nonempty_string_field(
                    object.get("channel"),
                    &format!("{label}.channel"),
                )?,
                bytes: event_payload_bytes(value, label)?,
            })
        }
        "observe_target_no_score" => {
            ensure_known_event_fields(object, &["kind", "channel", "domain", "bytes"], label)?;
            Ok(LoweredCausalEvent::ObserveTargetNoScore {
                channel: required_nonempty_string_field(
                    object.get("channel"),
                    &format!("{label}.channel"),
                )?,
                domain: required_nonempty_string_field(
                    object.get("domain"),
                    &format!("{label}.domain"),
                )?,
                bytes: event_payload_bytes(value, label)?,
            })
        }
        "target" => {
            ensure_known_event_fields(
                object,
                &["kind", "channel", "domain", "bytes", "weight"],
                label,
            )?;
            let channel =
                required_nonempty_string_field(object.get("channel"), &format!("{label}.channel"))?;
            let domain =
                required_nonempty_string_field(object.get("domain"), &format!("{label}.domain"))?;
            let weight = object
                .get("weight")
                .map(|raw| {
                    raw.as_f64()
                        .filter(|value| value.is_finite() && *value > 0.0)
                        .ok_or_else(|| format!("{label}.weight must be finite and > 0"))
                })
                .transpose()?
                .unwrap_or(1.0);
            Ok(LoweredCausalEvent::Target {
                channel,
                domain,
                bytes: event_payload_bytes(value, label)?,
                weight,
            })
        }
        other => Err(format!(
            "{label}.kind has unknown causal event kind '{other}'"
        )),
    }
}

fn event_payload_bytes(value: &Value, label: &str) -> Result<Vec<u8>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let payload = object
        .get("bytes")
        .ok_or_else(|| format!("{label}.bytes is required"))?;
    payload_bytes(payload, &format!("{label}.bytes"))
}

fn payload_bytes(value: &Value, label: &str) -> Result<Vec<u8>, String> {
    match value {
        Value::String(text) => Ok(text.as_bytes().to_vec()),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let byte = item
                    .as_u64()
                    .ok_or_else(|| format!("{label}[{index}] must be an integer byte"))?;
                u8::try_from(byte).map_err(|_| format!("{label}[{index}] must be in 0..=255"))
            })
            .collect(),
        _ => Err(format!(
            "{label} must be either a byte string or an array of integer bytes"
        )),
    }
}

fn required_nonempty_string_field(value: Option<&Value>, label: &str) -> Result<String, String> {
    let text = value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label} is required"))?;
    Ok(text.to_string())
}

fn ensure_known_event_fields(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown event field '{key}'"));
        }
    }
    Ok(())
}

fn is_lower_hex_crc32(value: &str) -> bool {
    value.len() == 8
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn lowered_event_skeleton_bytes(events: &[LoweredCausalEvent]) -> Result<Vec<u8>, String> {
    let skeleton = events
        .iter()
        .map(|event| match event {
            LoweredCausalEvent::Reset => serde_json::json!({"kind": "reset"}),
            LoweredCausalEvent::Context { channel, bytes } => serde_json::json!({
                "kind": "context",
                "channel": channel,
                "bytes_len": bytes.len(),
            }),
            LoweredCausalEvent::ObserveTargetNoScore {
                channel,
                domain,
                bytes,
            } => serde_json::json!({
                "kind": "observe_target_no_score",
                "channel": channel,
                "domain": domain,
                "bytes_len": bytes.len(),
            }),
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } => serde_json::json!({
                "kind": "target",
                "channel": channel,
                "domain": domain,
                "bytes_len": bytes.len(),
                "weight_bits": weight.to_bits(),
            }),
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&skeleton)
        .map_err(|err| format!("failed to serialize lowered event skeleton: {err}"))
}

fn causal_domain_support_bytes(
    domains: &BTreeMap<String, CausalTargetDomain>,
) -> Result<Vec<u8>, String> {
    let value = domains
        .iter()
        .map(|(domain, support)| match support {
            CausalTargetDomain::ByteAlphabet => serde_json::json!({
                "domain": domain,
                "kind": "byte_alphabet",
                "symbol_width_bytes": 1usize,
                "symbols": 256usize,
            }),
            CausalTargetDomain::EnumeratedPayloads { payloads } => serde_json::json!({
                "domain": domain,
                "kind": "enumerated_payloads",
                "payloads": payloads,
            }),
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&value)
        .map_err(|err| format!("failed to serialize causal target-domain supports: {err}"))
}

fn causal_channel_set(events: &[LoweredCausalEvent]) -> BTreeSet<String> {
    let mut channels = BTreeSet::<String>::new();
    for event in events {
        match event {
            LoweredCausalEvent::Reset => {}
            LoweredCausalEvent::Context { channel, .. }
            | LoweredCausalEvent::ObserveTargetNoScore { channel, .. }
            | LoweredCausalEvent::Target { channel, .. } => {
                channels.insert(channel.clone());
            }
        }
    }
    channels
}

fn validate_events_against_causal_profile(
    events: &[LoweredCausalEvent],
    domains: &BTreeMap<String, CausalTargetDomain>,
    header: &CausalHeaderProfile,
) -> Result<(), String> {
    if header.action_alphabet_size > 256 && header.event_grammar.context_channels.contains("action")
    {
        return Err(
            "event_grammar contains action context but action_alphabet.size exceeds byte encoding capacity (must be <= 256)"
                .to_string(),
        );
    }
    for event in events {
        match event {
            LoweredCausalEvent::Reset => {}
            LoweredCausalEvent::Context { channel, bytes } => {
                if !header.event_grammar.context_channels.contains(channel) {
                    return Err(format!(
                        "causal context event channel '{channel}' is not declared in event_grammar.context_channels"
                    ));
                }
                if channel == "action" {
                    if bytes.len() != 1 {
                        return Err(
                            "action context payload must encode exactly one byte".to_string()
                        );
                    }
                    if bytes[0] as usize >= header.action_alphabet_size {
                        return Err(format!(
                            "action context value {} is outside action_alphabet.size={}",
                            bytes[0], header.action_alphabet_size
                        ));
                    }
                }
            }
            LoweredCausalEvent::ObserveTargetNoScore { domain, bytes, .. }
            | LoweredCausalEvent::Target { domain, bytes, .. } => {
                let support = domains.get(domain).ok_or_else(|| {
                    format!("causal event references undeclared target domain '{domain}'")
                })?;
                let descriptor = match event {
                    LoweredCausalEvent::ObserveTargetNoScore {
                        channel, domain, ..
                    } => CausalChannelDomain {
                        channel: channel.clone(),
                        domain: domain.clone(),
                    },
                    LoweredCausalEvent::Target {
                        channel, domain, ..
                    } => CausalChannelDomain {
                        channel: channel.clone(),
                        domain: domain.clone(),
                    },
                    _ => unreachable!(),
                };
                match event {
                    LoweredCausalEvent::ObserveTargetNoScore { .. } => {
                        if !header
                            .event_grammar
                            .observe_target_no_score
                            .contains(&descriptor)
                        {
                            return Err(format!(
                                "observe_target_no_score ({}, {}) is not declared in event_grammar.observe_target_no_score",
                                descriptor.channel, descriptor.domain
                            ));
                        }
                    }
                    LoweredCausalEvent::Target { .. } => {
                        if !header.event_grammar.target.contains(&descriptor) {
                            return Err(format!(
                                "target ({}, {}) is not declared in event_grammar.target",
                                descriptor.channel, descriptor.domain
                            ));
                        }
                    }
                    _ => {}
                }
                if !causal_support_contains(support, bytes) {
                    return Err(format!(
                        "causal event payload is outside target-domain support '{domain}'"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn causal_support_contains(support: &CausalTargetDomain, bytes: &[u8]) -> bool {
    match support {
        CausalTargetDomain::ByteAlphabet => bytes.len() == 1,
        CausalTargetDomain::EnumeratedPayloads { payloads } => {
            payloads.iter().any(|payload| payload == bytes)
        }
    }
}

fn expand_byte_alphabet_events(
    events: Vec<LoweredCausalEvent>,
    domains: &BTreeMap<String, CausalTargetDomain>,
) -> Result<Vec<LoweredCausalEvent>, String> {
    let mut expanded = Vec::<LoweredCausalEvent>::new();
    for event in events {
        match event {
            LoweredCausalEvent::Reset | LoweredCausalEvent::Context { .. } => expanded.push(event),
            LoweredCausalEvent::ObserveTargetNoScore {
                channel,
                domain,
                bytes,
            } => {
                let Some(support) = domains.get(&domain) else {
                    return Err(format!(
                        "causal event references undeclared target domain '{domain}'"
                    ));
                };
                match support {
                    CausalTargetDomain::ByteAlphabet => {
                        if bytes.is_empty() {
                            return Err(
                                "byte_alphabet payloads must contain at least one byte".to_string()
                            );
                        }
                        for byte in bytes {
                            expanded.push(LoweredCausalEvent::ObserveTargetNoScore {
                                channel: channel.clone(),
                                domain: domain.clone(),
                                bytes: vec![byte],
                            });
                        }
                    }
                    CausalTargetDomain::EnumeratedPayloads { .. } => {
                        expanded.push(LoweredCausalEvent::ObserveTargetNoScore {
                            channel,
                            domain,
                            bytes,
                        });
                    }
                }
            }
            LoweredCausalEvent::Target {
                channel,
                domain,
                bytes,
                weight,
            } => {
                let Some(support) = domains.get(&domain) else {
                    return Err(format!(
                        "causal event references undeclared target domain '{domain}'"
                    ));
                };
                match support {
                    CausalTargetDomain::ByteAlphabet => {
                        if bytes.is_empty() {
                            return Err(
                                "byte_alphabet payloads must contain at least one byte".to_string()
                            );
                        }
                        for byte in bytes {
                            expanded.push(LoweredCausalEvent::Target {
                                channel: channel.clone(),
                                domain: domain.clone(),
                                bytes: vec![byte],
                                weight,
                            });
                        }
                    }
                    CausalTargetDomain::EnumeratedPayloads { .. } => {
                        expanded.push(LoweredCausalEvent::Target {
                            channel,
                            domain,
                            bytes,
                            weight,
                        });
                    }
                }
            }
        }
    }
    Ok(expanded)
}

fn causal_dataset_string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}
