use super::*;

impl VerifiedTheoremInputs {
    pub(super) fn load(
        theorem: &TuneTheoremConfig,
        compiled: &crate::spec::CompiledTuneSpec,
        dataset: &LoadedDataset,
        evaluator_profile: &EvaluatorProfile,
        config_dir: &Path,
    ) -> Result<Self, String> {
        let bounds_digest = bounds_hash(&compiled.canonical_spec().bounds)?;
        let controller_kind = controller_kind_name(compiled.controller());
        let scalar_representation = theorem
            .scalar_representation_ref
            .as_deref()
            .unwrap_or(SCALAR_REPRESENTATION_DECLARATION);
        let mut verified = Self::default();

        verified.finite_planner_state = load_generic_certificate(
            theorem.finite_planner_state_certificate.as_deref(),
            "finite_planner_state",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.no_hidden_state = load_generic_certificate(
            theorem.no_hidden_state_certificate.as_deref(),
            "no_hidden_state",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.determinism_deadline = load_generic_certificate(
            theorem.determinism_deadline_certificate.as_deref(),
            "determinism_deadline",
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.exact_state_observation = load_exact_state_observation_certificate(
            theorem,
            verified.finite_planner_state.as_ref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        verified.exact_reward_encoding = load_exact_reward_certificate(
            theorem.exact_reward_encoding_certificate.as_deref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
            scalar_representation,
        )?;
        verified.deterministic_table = load_deterministic_evaluator_table(
            theorem.deterministic_evaluator_table.as_deref(),
            compiled,
            dataset,
            evaluator_profile,
            config_dir,
            &bounds_digest,
            controller_kind,
        )?;
        Ok(verified)
    }

    pub(super) fn timing_certified(&self, theorem: &TuneTheoremConfig) -> bool {
        match theorem.timing_certification_tier {
            TimingCertificationTier::RealTime => self.determinism_deadline.is_some(),
            TimingCertificationTier::DeterministicTable => self.deterministic_table.is_some(),
            TimingCertificationTier::BestEffort | TimingCertificationTier::Isolated => false,
        }
    }

    pub(super) fn to_json_value(&self) -> Value {
        serde_json::json!({
            "finite_planner_state": self.finite_planner_state.as_ref().map(VerifiedCertificate::to_json_value),
            "no_hidden_state": self.no_hidden_state.as_ref().map(VerifiedCertificate::to_json_value),
            "exact_reward_encoding": self.exact_reward_encoding.as_ref().map(VerifiedExactRewardEncodingCertificate::to_json_value),
            "exact_state_observation": self.exact_state_observation.as_ref().map(VerifiedExactStateObservationCertificate::to_json_value),
            "determinism_deadline": self.determinism_deadline.as_ref().map(VerifiedCertificate::to_json_value),
            "deterministic_evaluator_table": self.deterministic_table.as_ref().map(VerifiedDeterministicEvaluatorTable::to_json_value),
        })
    }
}

impl VerifiedCertificate {
    pub(super) fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.ref_value,
            "content_crc32": self.content_hash,
            "verified": true,
        })
    }
}

impl VerifiedExactRewardEncodingCertificate {
    pub(super) fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "scalar_representation": self.scalar_representation,
            "reward_bits": self.reward_bits,
            "max_reward": self.max_reward,
            "encoding": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => "integer_objective_difference",
                VerifiedRewardEncodingMode::FiniteRewardMap { .. } => "finite_reward_map",
            },
            "finite_reward_values": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => None,
                VerifiedRewardEncodingMode::FiniteRewardMap { map } => Some(map.objective_difference_to_symbol.len()),
            },
            "complete_nonnegative_interval_max": match &self.mode {
                VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => None,
                VerifiedRewardEncodingMode::FiniteRewardMap { map } => map.complete_nonnegative_interval_max,
            },
        })
    }

    pub(super) fn is_identity_or_interval_encoding(&self) -> bool {
        match &self.mode {
            VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval => true,
            VerifiedRewardEncodingMode::FiniteRewardMap { map } => map
                .objective_difference_to_symbol
                .iter()
                .all(|(objective_difference, symbol)| objective_difference == symbol),
        }
    }
}

impl VerifiedExactStateObservationCertificate {
    pub(super) fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "observation_key_mode": self.observation_key_mode,
            "exact_state_encoder_spec_ref": self.exact_state_encoder_spec_ref,
            "observation_adapter_spec_ref": self.observation_adapter_spec_ref,
            "observation_adapter_content_crc32": self.observation_adapter_content_hash,
            "finite_planner_state_certificate_crc32": self.finite_planner_state_certificate_hash,
            "finite_state_count": self.finite_state_count,
        })
    }
}

impl VerifiedDeterministicEvaluatorTable {
    pub(super) fn to_json_value(&self) -> Value {
        serde_json::json!({
            "ref": self.base.ref_value,
            "content_crc32": self.base.content_hash,
            "verified": true,
            "rows": self.rows.len(),
        })
    }

    pub(super) fn evaluate(
        &self,
        candidate: &crate::spec::CompiledCompressionBackend,
        dataset: &LoadedDataset,
        model_bytes: usize,
        min_throughput_bytes_per_second: f64,
        max_memory_bytes: u64,
        effective_eval_time_limit_seconds: f64,
    ) -> Result<CandidateEvalResult, String> {
        let candidate_crc32 = crc32_hex(candidate.canonical_bytes().as_slice());
        let row = self.rows.get(&candidate_crc32).ok_or_else(|| {
            format!(
                "deterministic evaluator table missing row for candidate_crc32 '{candidate_crc32}'"
            )
        })?;
        if row.status == CandidateEvalStatus::Timeout {
            return Ok(timeout_eval_result(
                row.elapsed_seconds,
                row.peak_memory_bytes,
                effective_eval_time_limit_seconds,
            ));
        }
        if row.status != CandidateEvalStatus::Success {
            return Ok(CandidateEvalResult {
                status: row.status,
                compressed_bytes: row.compressed_bytes,
                elapsed_seconds: row.elapsed_seconds,
                effective_eval_time_limit_seconds,
                throughput_bytes_per_second: 0.0,
                peak_memory_bytes: row.peak_memory_bytes,
                target_loss_bits: f64::INFINITY,
                objective_bits: f64::INFINITY,
                deployable: false,
            });
        }
        if row.elapsed_seconds >= effective_eval_time_limit_seconds {
            return Ok(timeout_eval_result(
                row.elapsed_seconds,
                row.peak_memory_bytes,
                effective_eval_time_limit_seconds,
            ));
        }
        let throughput_bytes_per_second = if row.elapsed_seconds <= 0.0 {
            f64::INFINITY
        } else {
            dataset.dataset_units / row.elapsed_seconds
        };
        let deployable = throughput_bytes_per_second >= min_throughput_bytes_per_second
            && row.peak_memory_bytes <= max_memory_bytes;
        let objective_bits = if deployable {
            ((model_bytes as f64) * 8.0) + row.target_loss_bits
        } else {
            f64::INFINITY
        };

        Ok(CandidateEvalResult {
            status: row.status,
            compressed_bytes: row.compressed_bytes,
            elapsed_seconds: row.elapsed_seconds,
            effective_eval_time_limit_seconds,
            throughput_bytes_per_second,
            peak_memory_bytes: row.peak_memory_bytes,
            target_loss_bits: row.target_loss_bits,
            objective_bits,
            deployable,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_generic_certificate(
    reference: Option<&str>,
    expected_kind: &str,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        expected_kind,
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    Ok(Some(VerifiedCertificate {
        ref_value,
        content_hash,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_exact_reward_certificate(
    reference: Option<&str>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
    scalar_representation: &str,
) -> Result<Option<VerifiedExactRewardEncodingCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "exact_reward_encoding",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "exact_reward_encoding")?;
    let encoding = required_cert_str(object, "encoding", "exact_reward_encoding")?;
    if encoding != "integer_objective_difference" && encoding != "finite_reward_map" {
        return Err(format!(
            "exact_reward_encoding certificate uses unsupported encoding '{encoding}'"
        ));
    }
    let cert_scalar = required_cert_str(object, "scalar_representation", "exact_reward_encoding")?;
    if cert_scalar != scalar_representation {
        return Err(format!(
            "exact_reward_encoding certificate scalar_representation '{cert_scalar}' does not match requested '{scalar_representation}'"
        ));
    }
    let reward_bits = required_cert_u64(object, "reward_bits", "exact_reward_encoding")?;
    let reward_bits = usize::try_from(reward_bits)
        .map_err(|_| "exact_reward_encoding.reward_bits does not fit usize".to_string())?;
    let max_reward = required_cert_u64(object, "max_reward", "exact_reward_encoding")?;
    let max_reward = Reward::try_from(max_reward)
        .map_err(|_| "exact_reward_encoding.max_reward does not fit Reward".to_string())?;
    let max_encoded = max_nonnegative_reward_for_bits(reward_bits)?;
    if max_reward > max_encoded {
        return Err(format!(
            "exact_reward_encoding.max_reward {max_reward} exceeds reward_bits={reward_bits} maximum {max_encoded}"
        ));
    }
    let mode = if encoding == "integer_objective_difference" {
        VerifiedRewardEncodingMode::IntegerObjectiveDifferenceInterval
    } else {
        let map = parse_finite_reward_map(object, reward_bits, max_reward)?;
        if controller_requires_exact_objective_difference(controller_kind)
            && map.complete_nonnegative_interval_max.is_none()
        {
            return Err(
                "exact_reward_encoding finite_reward_map certificates for exact-objective controllers must declare complete_nonnegative_interval_max"
                    .to_string(),
            );
        }
        VerifiedRewardEncodingMode::FiniteRewardMap { map }
    };
    Ok(Some(VerifiedExactRewardEncodingCertificate {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        max_reward,
        reward_bits,
        scalar_representation: cert_scalar.to_string(),
        mode,
    }))
}

pub(super) fn controller_requires_exact_objective_difference(controller_kind: &str) -> bool {
    matches!(
        controller_kind,
        "mc_aixi_fac_ctw" | "aiqi_warmstart_exact_jh"
    )
}

pub(super) fn parse_finite_reward_map(
    object: &serde_json::Map<String, Value>,
    reward_bits: usize,
    declared_max_reward: Reward,
) -> Result<VerifiedFiniteRewardMap, String> {
    let entries = object
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "exact_reward_encoding finite_reward_map requires a 'values' array".to_string()
        })?;
    if entries.is_empty() {
        return Err("exact_reward_encoding finite_reward_map values must be nonempty".to_string());
    }
    let max_encoded = max_nonnegative_reward_for_bits(reward_bits)?;
    let mut by_difference = BTreeMap::<Reward, Reward>::new();
    let mut seen_symbols = BTreeSet::<Reward>::new();
    for (index, entry) in entries.iter().enumerate() {
        let entry_object = entry
            .as_object()
            .ok_or_else(|| format!("exact_reward_encoding.values[{index}] must be an object"))?;
        let difference = required_cert_u64(
            entry_object,
            "objective_difference",
            &format!("exact_reward_encoding.values[{index}]"),
        )?;
        let difference = Reward::try_from(difference).map_err(|_| {
            format!(
                "exact_reward_encoding.values[{index}].objective_difference does not fit Reward"
            )
        })?;
        let symbol = required_cert_u64(
            entry_object,
            "symbol",
            &format!("exact_reward_encoding.values[{index}]"),
        )?;
        let symbol = Reward::try_from(symbol).map_err(|_| {
            format!("exact_reward_encoding.values[{index}].symbol does not fit Reward")
        })?;
        if symbol > max_encoded {
            return Err(format!(
                "exact_reward_encoding.values[{index}].symbol {symbol} exceeds reward_bits={reward_bits} maximum {max_encoded}"
            ));
        }
        if symbol > declared_max_reward {
            return Err(format!(
                "exact_reward_encoding.values[{index}].symbol {symbol} exceeds declared max_reward {declared_max_reward}"
            ));
        }
        if by_difference.insert(difference, symbol).is_some() {
            return Err(format!(
                "exact_reward_encoding finite_reward_map duplicates objective_difference {difference}"
            ));
        }
        if !seen_symbols.insert(symbol) {
            return Err(format!(
                "exact_reward_encoding finite_reward_map duplicates reward symbol {symbol}"
            ));
        }
    }
    let complete_nonnegative_interval_max = object
        .get("complete_nonnegative_interval_max")
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                "exact_reward_encoding.complete_nonnegative_interval_max must be an unsigned integer"
                    .to_string()
            })
        })
        .transpose()?
        .map(|max| {
            Reward::try_from(max).map_err(|_| {
                "exact_reward_encoding.complete_nonnegative_interval_max does not fit Reward"
                    .to_string()
            })
        })
        .transpose()?;
    if let Some(complete_max) = complete_nonnegative_interval_max {
        if complete_max > declared_max_reward {
            return Err(format!(
                "exact_reward_encoding.complete_nonnegative_interval_max {complete_max} exceeds declared max_reward {declared_max_reward}"
            ));
        }
        validate_complete_finite_reward_interval(&by_difference, complete_max)?;
    }
    Ok(VerifiedFiniteRewardMap {
        objective_difference_to_symbol: by_difference,
        complete_nonnegative_interval_max,
    })
}

pub(super) fn validate_complete_finite_reward_interval(
    objective_difference_to_symbol: &BTreeMap<Reward, Reward>,
    complete_max: Reward,
) -> Result<(), String> {
    for objective_difference in 0..=complete_max {
        if !objective_difference_to_symbol.contains_key(&objective_difference) {
            return Err(format!(
                "exact_reward_encoding finite_reward_map complete interval is missing objective_difference {objective_difference}"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_exact_state_observation_certificate(
    theorem: &TuneTheoremConfig,
    finite_planner_state: Option<&VerifiedCertificate>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedExactStateObservationCertificate>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(
        theorem.exact_state_observation_certificate.as_deref(),
        config_dir,
    )?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "exact_state_observation",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "exact_state_observation")?;
    let observation_key_mode =
        required_cert_str(object, "observation_key_mode", "exact_state_observation")?;
    let requested_adapter = theorem
        .observation_adapter_spec_ref
        .as_deref()
        .unwrap_or(OBSERVATION_ADAPTER_DECLARATION);
    let observed_adapter = required_cert_str(
        object,
        "observation_adapter_spec_ref",
        "exact_state_observation",
    )?;
    if observed_adapter != requested_adapter {
        return Err(format!(
            "exact_state_observation certificate observation_adapter_spec_ref '{observed_adapter}' does not match requested '{requested_adapter}'"
        ));
    }
    require_cert_string_match(
        object,
        "observation_adapter_content_crc32",
        &observation_adapter_content_hash()?,
        "exact_state_observation",
    )?;
    let finite_planner_state = finite_planner_state.ok_or_else(|| {
        "exact_state_observation certificate requires a verified finite_planner_state_certificate"
            .to_string()
    })?;
    require_cert_string_match(
        object,
        "finite_planner_state_certificate_crc32",
        &finite_planner_state.content_hash,
        "exact_state_observation",
    )?;
    let observed_encoder = required_cert_str(
        object,
        "exact_state_encoder_spec_ref",
        "exact_state_observation",
    )?;
    if let Some(expected) = theorem.exact_state_encoder_spec_ref.as_deref()
        && observed_encoder != expected
    {
        return Err(format!(
            "exact_state_observation certificate encoder ref '{observed_encoder}' does not match requested '{expected}'"
        ));
    }
    let interface = planner_interface_for_controller(compiled.controller()).ok_or_else(|| {
        "exact_state_observation certificate can only be validated for planner-family controllers"
            .to_string()
    })?;
    let finite_state_count =
        validate_exact_state_observation_artifact(object, observation_key_mode, interface)?;
    Ok(Some(VerifiedExactStateObservationCertificate {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        observation_key_mode: observation_key_mode.to_string(),
        exact_state_encoder_spec_ref: observed_encoder.to_string(),
        observation_adapter_spec_ref: observed_adapter.to_string(),
        observation_adapter_content_hash: observation_adapter_content_hash()?,
        finite_planner_state_certificate_hash: finite_planner_state.content_hash.clone(),
        finite_state_count,
    }))
}

fn planner_interface_for_controller(
    controller: &crate::spec::CompiledTuneController,
) -> Option<&crate::spec::TunePlannerInterfaceSpec> {
    match controller {
        crate::spec::CompiledTuneController::McAixiFacCtw(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AiqiDiscounted(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AiqiWarmstartExactJh(inner) => Some(&inner.interface),
        crate::spec::CompiledTuneController::AnnealedHillClimbing(_) => None,
    }
}

pub(super) fn validate_exact_state_observation_artifact(
    object: &serde_json::Map<String, Value>,
    observation_key_mode: &str,
    interface: &crate::spec::TunePlannerInterfaceSpec,
) -> Result<usize, String> {
    let states = object
        .get("psi_h_outputs")
        .or_else(|| object.get("state_outputs"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "exact_state_observation certificate requires a checkable psi_h_outputs array"
                .to_string()
        })?;
    if states.is_empty() {
        return Err("exact_state_observation psi_h_outputs must be nonempty".to_string());
    }
    let stream_len = interface.observation_stream_len.max(1);
    let max_symbol = max_observation_symbol_for_bits(interface.observation_bits)?;
    let mut seen_state_ids = BTreeSet::<String>::new();
    let mut seen_projected_outputs = BTreeSet::<Vec<PerceptVal>>::new();
    for (index, state_value) in states.iter().enumerate() {
        let state_object = state_value.as_object().ok_or_else(|| {
            format!("exact_state_observation.psi_h_outputs[{index}] must be an object")
        })?;
        let state_id = required_cert_str(
            state_object,
            "state_id",
            &format!("exact_state_observation.psi_h_outputs[{index}]"),
        )?;
        if !seen_state_ids.insert(state_id.to_string()) {
            return Err(format!(
                "exact_state_observation duplicate state_id '{state_id}'"
            ));
        }
        let observations = state_object
            .get("observations")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations must be an array"
                )
            })?;
        if observations.len() != stream_len {
            return Err(format!(
                "exact_state_observation.psi_h_outputs[{index}].observations length {} does not match observation_stream_len {stream_len}",
                observations.len()
            ));
        }
        let mut output = Vec::<PerceptVal>::with_capacity(stream_len);
        for (symbol_index, symbol_value) in observations.iter().enumerate() {
            let symbol = symbol_value.as_u64().ok_or_else(|| {
                format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations[{symbol_index}] must be an integer"
                )
            })?;
            if symbol > max_symbol {
                return Err(format!(
                    "exact_state_observation.psi_h_outputs[{index}].observations[{symbol_index}]={symbol} exceeds observation_bits={} maximum {max_symbol}",
                    interface.observation_bits
                ));
            }
            output.push(symbol);
        }
        let projected =
            project_observation_output(observation_key_mode, &output, interface.observation_bits)?;
        if !seen_projected_outputs.insert(projected) {
            return Err(format!(
                "exact_state_observation psi_h_outputs are not injective under observation_key_mode '{observation_key_mode}'"
            ));
        }
    }
    Ok(states.len())
}

pub(super) fn project_observation_output(
    observation_key_mode: &str,
    output: &[PerceptVal],
    observation_bits: usize,
) -> Result<Vec<PerceptVal>, String> {
    match observation_key_mode {
        "full_stream" => Ok(output.to_vec()),
        "first_symbol" | "first" => output
            .first()
            .copied()
            .map(|value| vec![value])
            .ok_or_else(|| "observation stream cannot be empty".to_string()),
        "last_symbol" | "last" => output
            .last()
            .copied()
            .map(|value| vec![value])
            .ok_or_else(|| "observation stream cannot be empty".to_string()),
        "stream_hash" => Ok(vec![crate::aixi::common::observation_key_from_stream(
            ObservationKeyMode::StreamHash,
            output,
            observation_bits,
        )]),
        other => Err(format!(
            "exact_state_observation certificate uses unsupported observation_key_mode '{other}'"
        )),
    }
}

pub(super) fn max_observation_symbol_for_bits(
    observation_bits: usize,
) -> Result<PerceptVal, String> {
    if observation_bits == 0 {
        Ok(0)
    } else if observation_bits >= 64 {
        Ok(PerceptVal::MAX)
    } else {
        Ok((1_u64 << observation_bits) - 1)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_deterministic_evaluator_table(
    reference: Option<&str>,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    config_dir: &Path,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<Option<VerifiedDeterministicEvaluatorTable>, String> {
    let Some((ref_value, value, content_hash)) = load_certificate_value(reference, config_dir)?
    else {
        return Ok(None);
    };
    validate_certificate_common(
        &value,
        "deterministic_evaluator_table",
        compiled,
        dataset,
        evaluator_profile,
        bounds_digest,
        controller_kind,
    )?;
    let object = certificate_object(&value, "deterministic_evaluator_table")?;
    let rows_value = object
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| "deterministic_evaluator_table.rows must be an array".to_string())?;
    let mut rows = HashMap::<String, DeterministicEvaluatorRow>::new();
    for (index, row_value) in rows_value.iter().enumerate() {
        let row_object = row_value.as_object().ok_or_else(|| {
            format!("deterministic_evaluator_table.rows[{index}] must be an object")
        })?;
        let candidate_crc32 = required_cert_str(
            row_object,
            "candidate_crc32",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let status = match required_cert_str(
            row_object,
            "status",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )? {
            "success" => CandidateEvalStatus::Success,
            "timeout" => CandidateEvalStatus::Timeout,
            "invalid" => CandidateEvalStatus::Invalid,
            "error" => CandidateEvalStatus::Error,
            other => {
                return Err(format!(
                    "deterministic_evaluator_table.rows[{index}].status has unknown status '{other}'"
                ));
            }
        };
        let compressed_bytes = required_cert_u64(
            row_object,
            "compressed_bytes",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let compressed_bytes = usize::try_from(compressed_bytes).map_err(|_| {
            format!(
                "deterministic_evaluator_table.rows[{index}].compressed_bytes does not fit usize"
            )
        })?;
        let target_loss_bits = required_cert_f64(
            row_object,
            "target_loss_bits",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let elapsed_seconds = required_cert_f64(
            row_object,
            "elapsed_seconds",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        let peak_memory_bytes = required_cert_u64(
            row_object,
            "peak_memory_bytes",
            &format!("deterministic_evaluator_table.rows[{index}]"),
        )?;
        if rows
            .insert(
                candidate_crc32.to_string(),
                DeterministicEvaluatorRow {
                    status,
                    compressed_bytes,
                    target_loss_bits,
                    elapsed_seconds,
                    peak_memory_bytes,
                },
            )
            .is_some()
        {
            return Err(format!(
                "deterministic_evaluator_table duplicate candidate_crc32 '{candidate_crc32}'"
            ));
        }
    }
    if rows.is_empty() {
        return Err("deterministic_evaluator_table.rows must not be empty".to_string());
    }
    Ok(Some(VerifiedDeterministicEvaluatorTable {
        base: VerifiedCertificate {
            ref_value,
            content_hash,
        },
        rows,
    }))
}

fn load_certificate_value(
    reference: Option<&str>,
    config_dir: &Path,
) -> Result<Option<(String, Value, String)>, String> {
    let Some(raw_reference) = reference else {
        return Ok(None);
    };
    let raw_ref = raw_reference.trim();
    if raw_ref.is_empty() {
        return Err("theorem certificate reference must be non-empty when set".to_string());
    };
    if raw_ref.contains("://") && !raw_ref.starts_with("file://") {
        return Err(format!(
            "unsupported theorem certificate reference scheme in '{raw_ref}'; use a filesystem path or file:// URI"
        ));
    }
    let path_text = raw_ref.strip_prefix("file://").unwrap_or(raw_ref);
    let path = Path::new(path_text);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_dir.join(path)
    };
    let raw = fs::read(&resolved).map_err(|err| {
        format!(
            "failed to read theorem certificate '{}': {err}",
            resolved.display()
        )
    })?;
    let value: Value = serde_json::from_slice(&raw).map_err(|err| {
        format!(
            "invalid theorem certificate JSON '{}': {err}",
            resolved.display()
        )
    })?;
    Ok(Some((raw_ref.to_string(), value, crc32_hex(&raw))))
}

#[allow(clippy::too_many_arguments)]
fn validate_certificate_common(
    value: &Value,
    expected_kind: &str,
    compiled: &crate::spec::CompiledTuneSpec,
    dataset: &LoadedDataset,
    evaluator_profile: &EvaluatorProfile,
    bounds_digest: &str,
    controller_kind: &str,
) -> Result<(), String> {
    let object = certificate_object(value, expected_kind)?;
    let schema_version = required_cert_u64(object, "schema_version", expected_kind)?;
    if schema_version != 1 {
        return Err(format!(
            "{expected_kind} certificate schema_version must be 1, got {schema_version}"
        ));
    }
    let kind = required_cert_str(object, "kind", expected_kind)?;
    if kind != expected_kind {
        return Err(format!("{expected_kind} certificate has kind '{kind}'"));
    }
    require_cert_string_match(
        object,
        "dataset_crc32",
        &dataset.canonical_content_hash,
        expected_kind,
    )?;
    require_cert_string_match(object, "bounds_crc32", bounds_digest, expected_kind)?;
    require_cert_string_match(
        object,
        "evaluator_profile_crc32",
        &evaluator_profile.hash()?,
        expected_kind,
    )?;
    require_cert_string_match(object, "controller_kind", controller_kind, expected_kind)?;
    if let Some(action_alphabet_size) = object.get("action_alphabet_size") {
        let expected = planner_action_count(compiled)?;
        let observed = action_alphabet_size
            .as_u64()
            .ok_or_else(|| format!("{expected_kind}.action_alphabet_size must be an integer"))?;
        if observed != expected as u64 {
            return Err(format!(
                "{expected_kind}.action_alphabet_size {observed} does not match compiled {expected}"
            ));
        }
    }
    Ok(())
}

fn certificate_object<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} certificate must be a JSON object"))
}

fn required_cert_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be a string"))
}

fn required_cert_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn required_cert_f64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<f64, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{label}.{field} must be a finite number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{label}.{field} must be finite and nonnegative"));
    }
    Ok(value)
}

fn require_cert_string_match(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let observed = required_cert_str(object, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} '{observed}' does not match expected '{expected}'"
        ));
    }
    Ok(())
}
