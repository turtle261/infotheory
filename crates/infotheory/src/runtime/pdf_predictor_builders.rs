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

macro_rules! feature_gated_rate_pdf_predictor_builder {
    (
        feature: $feature:literal,
        fn $name:ident($backend:ident) $body:block
    ) => {
        #[cfg(feature = $feature)]
        pub(super) fn $name(
            $backend: &CompiledRateBackend,
        ) -> anyhow::Result<crate::compression::RatePdfPredictor> $body
    };
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-rosa",
    fn build_pdf_predictor_rosa(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::RosaPlus { max_order },
            "rosa kernel used with non-rosa plan"
        );
        Ok(crate::compression::RatePdfPredictor::Rosa(
            crate::compression::RosaPredictor::new(*max_order),
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-match",
    fn build_pdf_predictor_match(backend) {
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
        Ok(crate::compression::RatePdfPredictor::Match {
            model: MatchModel::new_contiguous(
                *hash_bits,
                *min_len,
                *max_len,
                *base_mix,
                *confidence_scale,
            ),
        })
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-match",
    fn build_pdf_predictor_sparse_match(backend) {
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
        Ok(crate::compression::RatePdfPredictor::SparseMatch {
            model: SparseMatchModel::new(
                *hash_bits,
                *min_len,
                *max_len,
                *gap_min,
                *gap_max,
                *base_mix,
                *confidence_scale,
            ),
        })
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-ppmd",
    fn build_pdf_predictor_ppmd(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Ppmd { order, memory_mb },
            "ppmd kernel used with non-ppmd plan"
        );
        Ok(crate::compression::RatePdfPredictor::Ppmd {
            model: PpmdModel::new(*order, *memory_mb),
        })
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-sequitur",
    fn build_pdf_predictor_sequitur(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Sequitur { context_bytes },
            "sequitur kernel used with non-sequitur plan"
        );
        Ok(crate::compression::RatePdfPredictor::Sequitur {
            model: SequiturModel::new(*context_bytes),
        })
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-ctw",
    fn build_pdf_predictor_ctw(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Ctw { depth },
            "ctw kernel used with non-ctw plan"
        );
        Ok(crate::compression::RatePdfPredictor::Ctw(
            crate::compression::CtwPredictor::new_ctw(*depth),
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-ctw",
    fn build_pdf_predictor_fac_ctw(backend) {
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
        Ok(crate::compression::RatePdfPredictor::FacCtw(
            crate::compression::CtwPredictor::new_fac(
                *base_depth,
                *encoding_bits,
                Some(*msb_first),
            ),
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-bit-reservoir",
    fn build_pdf_predictor_bit_reservoir(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::BitReservoir { config },
            "bit-reservoir kernel used with non-bit-reservoir plan"
        );
        Ok(crate::compression::RatePdfPredictor::BitReservoir {
            model: BitReservoirModel::new(config.clone()).map_err(anyhow::Error::msg)?,
            pdf: vec![1.0 / 256.0; 256],
            valid: false,
            native_prefix_progress: None,
            native_prediction: None,
        })
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-mamba",
    fn build_pdf_predictor_mamba(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Mamba { parsed_method, .. },
            "mamba kernel used with non-mamba plan"
        );
        Ok(crate::compression::RatePdfPredictor::Mamba(
            crate::compression::MambaPredictor::from_method_spec(parsed_method)?,
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-rwkv",
    fn build_pdf_predictor_rwkv(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Rwkv7 { parsed_method, .. },
            "rwkv kernel used with non-rwkv plan"
        );
        Ok(crate::compression::RatePdfPredictor::Rwkv(
            crate::compression::RwkvPredictor::from_method_spec(parsed_method)?,
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-zpaq",
    fn build_pdf_predictor_zpaq(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Zpaq { method },
            "zpaq kernel used with non-zpaq plan"
        );
        Ok(crate::compression::RatePdfPredictor::Zpaq(
            crate::compression::ZpaqPredictor::new(method.clone()),
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-mixture",
    fn build_pdf_predictor_mixture(backend) {
        Ok(crate::compression::RatePdfPredictor::Mixture(
            crate::compression::MixturePredictor::new_from_compiled(backend)?,
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-particle",
    fn build_pdf_predictor_particle(backend) {
        expect_plan_ref!(
            backend.plan(),
            crate::spec::core::RateBackendPlan::Particle { spec },
            "particle kernel used with non-particle plan"
        );
        Ok(crate::compression::RatePdfPredictor::Particle(
            ParticleRuntime::new(spec),
        ))
    }
}

feature_gated_rate_pdf_predictor_builder! {
    feature: "backend-calibrated",
    fn build_pdf_predictor_calibrated(backend) {
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
        let base_backend =
            compile_calibrated_base_backend(base).map_err(|err| anyhow::anyhow!("{err}"))?;
        Ok(crate::compression::RatePdfPredictor::Calibrated {
            base: Box::new(build_rate_pdf_predictor_via_kernel(&base_backend)?),
            core: CalibratorCore::new(*context, *bins, *learning_rate, *bias_clip),
            bitwise: Default::default(),
            pdf: vec![1.0 / 256.0; 256],
            valid: false,
        })
    }
}
