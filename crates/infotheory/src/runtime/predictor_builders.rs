#[cfg(any(
    feature = "backend-rosa",
    feature = "backend-match",
    feature = "backend-ppmd",
    feature = "backend-sequitur",
    feature = "backend-ctw",
    feature = "backend-zpaq",
    feature = "backend-mixture",
    feature = "backend-particle",
    feature = "backend-calibrated",
    feature = "backend-bit-reservoir",
    feature = "backend-mamba",
    feature = "backend-rwkv"
))]
use super::*;

macro_rules! feature_gated_rate_predictor_builder {
    (
        feature: $feature:literal,
        fn $name:ident($backend:ident, $min_prob:ident) $body:block
    ) => {
        #[cfg(feature = $feature)]
        pub(super) fn $name(
            $backend: &CompiledRateBackend,
            $min_prob: f64,
        ) -> Result<crate::mixture::RateBackendPredictor, String> $body
    };
}

feature_gated_rate_predictor_builder! {
    feature: "backend-rosa",
    fn build_predictor_rosa(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::RosaPlus { max_order },
            "rosa kernel used with non-rosa plan"
        );
        let mut model = RosaPlus::new(*max_order, false, 0, 42);
        model.build_lm_full_bytes_no_finalize_endpos();
        Ok(crate::mixture::RateBackendPredictor::Rosa {
            model,
            min_prob,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-match",
    fn build_predictor_match(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Match {
                hash_bits,
                min_len,
                max_len,
                base_mix,
                confidence_scale,
            },
            "match kernel used with non-match plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Match {
            model: MatchModel::new_contiguous(
                *hash_bits,
                *min_len,
                *max_len,
                *base_mix,
                *confidence_scale,
            ),
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-match",
    fn build_predictor_sparse_match(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::SparseMatch {
                hash_bits,
                min_len,
                max_len,
                gap_min,
                gap_max,
                base_mix,
                confidence_scale,
            },
            "sparse-match kernel used with non-sparse-match plan"
        );
        Ok(crate::mixture::RateBackendPredictor::SparseMatch {
            model: SparseMatchModel::new(
                *hash_bits,
                *min_len,
                *max_len,
                *gap_min,
                *gap_max,
                *base_mix,
                *confidence_scale,
            ),
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-ppmd",
    fn build_predictor_ppmd(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb },
            "ppmd kernel used with non-ppmd plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Ppmd {
            model: PpmdModel::new(*order, *memory_mb),
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-sequitur",
    fn build_predictor_sequitur(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Sequitur { context_bytes },
            "sequitur kernel used with non-sequitur plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Sequitur {
            model: SequiturModel::new(*context_bytes),
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-ctw",
    fn build_predictor_ctw(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Ctw { depth },
            "ctw kernel used with non-ctw plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Ctw {
            tree: ContextTree::new(*depth),
            bits_per_symbol: 8,
            min_prob,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
            native_prefix_progress: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-ctw",
    fn build_predictor_binary_tokens_ctw(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Ctw { depth },
            "ctw binary-token kernel used with non-ctw plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Ctw {
            tree: ContextTree::new(*depth),
            bits_per_symbol: 1,
            min_prob,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
            native_prefix_progress: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-ctw",
    fn build_predictor_fac_ctw(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
                msb_first,
            },
            "fac-ctw kernel used with non-fac-ctw plan"
        );
        let bits_per_symbol = *encoding_bits;
        Ok(crate::mixture::RateBackendPredictor::FacCtw {
            tree: FacContextTree::new(*base_depth, bits_per_symbol),
            bits_per_symbol,
            msb_first: *msb_first,
            min_prob,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
            native_prefix_progress: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-ctw",
    fn build_predictor_binary_tokens_fac_ctw(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::FacCtw { base_depth, .. },
            "fac-ctw binary-token kernel used with non-fac-ctw plan"
        );
        Ok(crate::mixture::RateBackendPredictor::FacCtw {
            tree: FacContextTree::new(*base_depth, 1),
            bits_per_symbol: 1,
            msb_first: false,
            min_prob,
            checkpoint_journal: Vec::new(),
            checkpoint_depth: 0,
            native_prefix_progress: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-bit-reservoir",
    fn build_predictor_bit_reservoir(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::BitReservoir { config },
            "bit-reservoir kernel used with non-bit-reservoir plan"
        );
        Ok(crate::mixture::RateBackendPredictor::BitReservoir {
            model: BitReservoirModel::new(config.clone())?,
            symbol_mode: crate::mixture::BitReservoirSymbolMode::Byte,
            min_prob,
            native_prefix_progress: None,
            native_prediction: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-bit-reservoir",
    fn build_predictor_binary_tokens_bit_reservoir(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::BitReservoir { config },
            "bit-reservoir binary-token kernel used with non-bit-reservoir plan"
        );
        Ok(crate::mixture::RateBackendPredictor::BitReservoir {
            model: BitReservoirModel::new(config.clone())?,
            symbol_mode: crate::mixture::BitReservoirSymbolMode::BitToken,
            min_prob,
            native_prefix_progress: None,
            native_prediction: None,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-rwkv",
    fn build_predictor_rwkv(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. },
            "rwkv kernel used with non-rwkv plan"
        );
        let mut compressor = rwkvzip::Compressor::new_from_method_spec(parsed_method)
            .map_err(|e| format!("invalid rwkv method: {e}"))?;
        compressor.reset_and_prime();
        Ok(crate::mixture::RateBackendPredictor::Rwkv7 {
            pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
            compressor,
            primed: true,
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-mamba",
    fn build_predictor_mamba(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. },
            "mamba kernel used with non-mamba plan"
        );
        let mut compressor = mambazip::Compressor::new_from_method_spec(parsed_method)
            .map_err(|e| format!("invalid mamba method: {e}"))?;
        let bias = compressor.online_bias_snapshot();
        let logits = compressor
            .model
            .forward(&mut compressor.scratch, 0, &mut compressor.state);
        mambazip::Compressor::logits_to_pdf(logits, bias.as_deref(), &mut compressor.pdf_buffer);
        Ok(crate::mixture::RateBackendPredictor::Mamba {
            pdf_scratch: vec![0.0; compressor.pdf_buffer.len()],
            compressor,
            primed: true,
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-zpaq",
    fn build_predictor_zpaq(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Zpaq { method },
            "zpaq kernel used with non-zpaq plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Zpaq {
            model: ZpaqRateModel::new(method.clone(), min_prob),
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-mixture",
    fn build_predictor_mixture(backend, _min_prob) {
        let experts = crate::mixture::expert_configs_from_compiled_mixture(backend)?;
        let runtime = crate::mixture::build_mixture_runtime_from_compiled(backend, &experts)
            .map_err(|e| format!("MixtureSpec invalid: {e}"))?;
        Ok(crate::mixture::RateBackendPredictor::Mixture { runtime })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-mixture",
    fn build_predictor_binary_tokens_mixture(backend, min_prob) {
        let experts = crate::mixture::expert_configs_from_compiled_mixture_with_builder(
            backend,
            crate::runtime::build_rate_backend_binary_token_predictor,
            min_prob,
        )?;
        let runtime = crate::mixture::build_mixture_runtime_from_compiled(backend, &experts)
            .map_err(|e| format!("MixtureSpec invalid: {e}"))?;
        Ok(crate::mixture::RateBackendPredictor::Mixture { runtime })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-particle",
    fn build_predictor_particle(backend, _min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Particle { spec },
            "particle kernel used with non-particle plan"
        );
        Ok(crate::mixture::RateBackendPredictor::Particle {
            runtime: ParticleRuntime::new(spec),
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-calibrated",
    fn build_predictor_calibrated(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Calibrated {
                context,
                bins,
                learning_rate,
                bias_clip,
                base,
            },
            "calibrated kernel used with non-calibrated plan"
        );
        let base_backend = compile_calibrated_base_backend(base)?;
        Ok(crate::mixture::RateBackendPredictor::Calibrated {
            base: Box::new(build_rate_backend_predictor_via_kernel(
                &base_backend,
                min_prob,
            )?),
            core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
            bitwise: crate::mixture::BytePrefixStepState::new(),
            pdf: [1.0 / 256.0; 256],
            valid: false,
            min_prob,
        })
    }
}

feature_gated_rate_predictor_builder! {
    feature: "backend-calibrated",
    fn build_predictor_binary_tokens_calibrated(backend, min_prob) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Calibrated {
                context,
                bins,
                learning_rate,
                bias_clip,
                base,
            },
            "calibrated binary-token kernel used with non-calibrated plan"
        );
        let base_backend = compile_calibrated_base_backend(base)?;
        Ok(crate::mixture::RateBackendPredictor::Calibrated {
            base: Box::new(crate::runtime::build_rate_backend_binary_token_predictor(
                &base_backend,
                min_prob,
            )?),
            core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
            bitwise: crate::mixture::BytePrefixStepState::new(),
            pdf: [1.0 / 256.0; 256],
            valid: false,
            min_prob,
        })
    }
}
