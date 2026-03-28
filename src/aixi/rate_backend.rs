use crate::{CalibratedSpec, MixtureSpec, RateBackend};
use std::sync::Arc;

pub(crate) fn adapt_rate_backend_for_bit_tokens(backend: RateBackend) -> RateBackend {
    match backend {
        RateBackend::Ctw { depth } => RateBackend::FacCtw {
            base_depth: depth,
            num_percept_bits: 1,
            encoding_bits: 1,
        },
        RateBackend::FacCtw { base_depth, .. } => RateBackend::FacCtw {
            base_depth,
            num_percept_bits: 1,
            encoding_bits: 1,
        },
        RateBackend::Mixture { spec } => {
            let experts = spec
                .experts
                .iter()
                .map(|expert| crate::MixtureExpertSpec {
                    name: expert.name.clone(),
                    log_prior: expert.log_prior,
                    max_order: expert.max_order,
                    backend: adapt_rate_backend_for_bit_tokens(expert.backend.clone()),
                })
                .collect();

            let mut adapted = MixtureSpec::new(spec.kind, experts)
                .with_schedule(spec.schedule)
                .with_alpha(spec.alpha);
            if let Some(decay) = spec.decay {
                adapted = adapted.with_decay(decay);
            }
            RateBackend::Mixture {
                spec: Arc::new(adapted),
            }
        }
        RateBackend::Calibrated { spec } => RateBackend::Calibrated {
            spec: Arc::new(CalibratedSpec {
                base: adapt_rate_backend_for_bit_tokens(spec.base.clone()),
                context: spec.context,
                bins: spec.bins,
                learning_rate: spec.learning_rate,
                bias_clip: spec.bias_clip,
            }),
        },
        other => other,
    }
}

pub(crate) fn rate_backend_contains_zpaq(backend: &RateBackend) -> bool {
    match backend {
        RateBackend::Zpaq { .. } => true,
        RateBackend::Mixture { spec } => spec
            .experts
            .iter()
            .any(|expert| rate_backend_contains_zpaq(&expert.backend)),
        RateBackend::Calibrated { spec } => rate_backend_contains_zpaq(&spec.base),
        _ => false,
    }
}
