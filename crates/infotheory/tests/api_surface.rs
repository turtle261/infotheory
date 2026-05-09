use infotheory::api::{
    MixtureExpertSpec, MixtureKind, MixtureSpec, ParticleSpec, RateBackend, RateBackendSession,
};
use infotheory::spec::CanonicalJson;
use std::sync::Arc;

#[test]
fn api_surface_rate_backend_session_rejects_invalid_programmatic_mixture() {
    let backend = RateBackend::Mixture {
        spec: Arc::new(MixtureSpec::new(MixtureKind::Bayes, vec![])),
    };
    let err = match RateBackendSession::from_spec(backend, None) {
        Ok(_) => panic!("invalid mixture backend should be rejected before runtime construction"),
        Err(err) => err,
    };
    let message = err.to_string();
    if cfg!(feature = "backend-mixture") {
        assert!(message.contains("must include at least one expert"));
    } else {
        assert!(message.contains("requires infotheory feature 'backend-mixture'"));
    }
}

#[test]
fn api_surface_spec_types_serialize_canonically() {
    let backend = RateBackend::Ctw { depth: 9 };
    let backend_json = backend.to_canonical_json().expect("backend json");
    assert!(backend_json.contains("\"kind\": \"ctw\""));

    let mixture = MixtureSpec::new(
        MixtureKind::Bayes,
        vec![{
            let mut expert = MixtureExpertSpec::new(backend.clone());
            expert.name = Some("ctw".to_string());
            expert
        }],
    );
    let mix_json = mixture.to_canonical_json().expect("mixture json");
    assert!(mix_json.contains("\"kind\": \"bayes\""));

    let particle_json = ParticleSpec::default().to_canonical_json();
    assert!(
        particle_json
            .expect("particle json")
            .contains("\"num_particles\"")
    );
}

#[cfg(feature = "backend-ctw")]
mod ctw_surface {
    use infotheory::api::{
        CompressionBackend, InfotheoryCtx, RateBackend, d_kl_bytes, empirical_cross_entropy_bytes,
        empirical_entropy_bytes, empirical_joint_entropy_bytes, empirical_mutual_information_bytes,
        empirical_ned_bytes, empirical_ned_cons_bytes, empirical_nte_bytes, get_default_ctx,
        js_div_bytes, nhd_bytes, set_default_ctx, try_biased_entropy_rate_backend,
        try_biased_entropy_rate_bytes, try_conditional_entropy_bytes,
        try_conditional_entropy_rate_bytes, try_cross_entropy_bytes,
        try_cross_entropy_rate_backend, try_cross_entropy_rate_bytes, try_entropy_rate_backend,
        try_entropy_rate_bytes, try_intrinsic_dependence_bytes, try_joint_entropy_rate_backend,
        try_joint_entropy_rate_bytes, try_mutual_information_bytes,
        try_mutual_information_rate_backend, try_mutual_information_rate_bytes, try_ned_bytes,
        try_ned_cons_bytes, try_ned_cons_rate_bytes, try_ned_rate_backend, try_ned_rate_bytes,
        try_nte_bytes, try_nte_rate_backend, try_nte_rate_bytes,
        try_resistance_to_transformation_bytes, tvd_bytes,
    };

    #[test]
    fn api_surface_entropy_and_distance_wrappers_are_callable() {
        let x = b"alpha beta alpha beta alpha";
        let y = b"alpha gamma alpha gamma alpha";
        let backend = RateBackend::Ctw { depth: 8 };
        let compiled = backend.compile().expect("compiled ctw backend");

        let prev = get_default_ctx().expect("default ctx");
        set_default_ctx(
            InfotheoryCtx::from_specs(
                backend.clone(),
                CompressionBackend::try_default().expect("default compression backend"),
            )
            .expect("ctw context"),
        );

        assert!(try_entropy_rate_backend(x, &compiled).expect("entropy rate") >= 0.0);
        assert!(try_biased_entropy_rate_backend(x, &compiled).expect("biased entropy rate") >= 0.0);
        assert!(
            try_cross_entropy_rate_backend(x, y, &compiled).expect("cross entropy rate") >= 0.0
        );
        assert!(
            try_joint_entropy_rate_backend(x, y, &compiled).expect("joint entropy rate") >= 0.0
        );
        assert!(try_mutual_information_rate_backend(x, y, &compiled).expect("mi rate") >= 0.0);
        assert!((0.0..=1.0).contains(&try_ned_rate_backend(x, y, &compiled).expect("ned rate")));
        assert!((0.0..=2.0).contains(&try_nte_rate_backend(x, y, &compiled).expect("nte rate")));

        assert!(empirical_entropy_bytes(x) >= 0.0);
        assert!(empirical_joint_entropy_bytes(x, y) >= 0.0);
        assert!(try_entropy_rate_bytes(x).expect("entropy rate bytes") >= 0.0);
        assert!(try_biased_entropy_rate_bytes(x).expect("biased entropy rate bytes") >= 0.0);
        assert!(try_joint_entropy_rate_bytes(x, y).expect("joint entropy rate bytes") >= 0.0);
        assert!(
            try_conditional_entropy_rate_bytes(x, y).expect("conditional entropy rate bytes")
                >= 0.0
        );
        assert!(try_conditional_entropy_bytes(x, y).expect("conditional entropy bytes") >= 0.0);
        assert!(try_mutual_information_bytes(x, y).expect("mutual information bytes") >= 0.0);
        assert!(empirical_mutual_information_bytes(x, y) >= 0.0);
        assert!(
            try_mutual_information_rate_bytes(x, y).expect("mutual information rate bytes") >= 0.0
        );
        assert!((0.0..=1.0).contains(&try_ned_bytes(x, y).expect("ned bytes")));
        assert!((0.0..=1.0).contains(&empirical_ned_bytes(x, y)));
        assert!((0.0..=1.0).contains(&try_ned_rate_bytes(x, y).expect("ned rate bytes")));
        assert!((0.0..=1.0).contains(&try_ned_cons_bytes(x, y).expect("ned cons bytes")));
        assert!((0.0..=1.0).contains(&empirical_ned_cons_bytes(x, y)));
        assert!((0.0..=1.0).contains(&try_ned_cons_rate_bytes(x, y).expect("ned cons rate bytes")));
        assert!((0.0..=2.0).contains(&try_nte_bytes(x, y).expect("nte bytes")));
        assert!((0.0..=2.0).contains(&empirical_nte_bytes(x, y)));
        assert!((0.0..=2.0).contains(&try_nte_rate_bytes(x, y).expect("nte rate bytes")));
        assert!((0.0..=1.0).contains(&tvd_bytes(x, y)));
        assert!((0.0..=1.0).contains(&nhd_bytes(x, y)));
        assert!(try_cross_entropy_bytes(x, y).expect("cross entropy bytes") >= 0.0);
        assert!(empirical_cross_entropy_bytes(x, y) >= 0.0);
        assert!(try_cross_entropy_rate_bytes(x, y).expect("cross entropy rate bytes") >= 0.0);
        assert!(d_kl_bytes(x, y) >= 0.0);
        assert!(js_div_bytes(x, y) >= 0.0);
        assert!(
            (0.0..=1.0).contains(&try_intrinsic_dependence_bytes(x).expect("intrinsic dependence"))
        );
        assert!((0.0..=1.0).contains(
            &try_resistance_to_transformation_bytes(x, y).expect("resistance to transformation")
        ));

        set_default_ctx(prev);
    }
}

#[cfg(feature = "backend-rosa")]
mod rosa_surface {
    use infotheory::api::{
        CompressionBackend, GenerationConfig, InfotheoryCtx, RateBackend, RateBackendSession,
    };

    #[test]
    fn api_surface_generation_session_and_config_are_callable() {
        let prompt = b"If a frog is green, dogs are red.\nIf a toad is green, cats are red.\nIf a dog is green, frogs are red.\nIf a cat is green, toads are red.\nIf a frog is red, dogs are green.\nIf a toad is red, cats are green.\nIf a dog is red, frogs are green.\nIf a cat is red, toads are ";
        let backend = RateBackend::RosaPlus { max_order: -1 };
        let ctx = InfotheoryCtx::from_specs(
            backend.clone(),
            CompressionBackend::try_default().expect("default compression backend"),
        )
        .expect("ctx");
        let cfg = GenerationConfig::sampled_frozen(42);

        let direct = ctx
            .try_generate_bytes_with_config(prompt, 8, cfg)
            .expect("direct generation");
        assert_eq!(direct.len(), 8);

        let mut session =
            RateBackendSession::from_spec(backend, Some((prompt.len() + direct.len()) as u64))
                .expect("session init");
        session.observe(prompt);
        let from_session = session.generate_bytes(8, cfg);
        session.finish().expect("session finish");

        assert_eq!(from_session, direct);
    }
}

#[cfg(feature = "backend-zpaq")]
mod zpaq_surface {
    use infotheory::api::{
        CompressionBackend, CompressionPathBatchOptions, NcdVariant, OperationParallelism,
        try_compress_bytes_backend, try_compress_size_backend, try_compress_size_chain_backend,
        try_conditional_entropy_paths, try_cross_entropy_paths, try_decompress_bytes_backend,
        try_get_bytes_from_paths, try_get_compressed_size_path_backend,
        try_get_compressed_sizes_from_paths_backend,
        try_get_compressed_sizes_from_paths_backend_with_options, try_js_divergence_paths,
        try_kl_divergence_paths, try_mutual_information_paths, try_ncd_bytes_backend,
        try_ncd_bytes_default, try_ncd_matrix_bytes_backend, try_ncd_matrix_paths_backend,
        try_ncd_paths_backend, try_ncd_paths_compiled_backend, try_ned_paths, try_nhd_paths,
        try_nte_paths, try_tvd_paths,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn has_not_found_io_error(err: &(dyn std::error::Error + 'static)) -> bool {
        let mut current = Some(err);
        while let Some(err) = current {
            if let Some(io_err) = err.downcast_ref::<std::io::Error>()
                && io_err.kind() == std::io::ErrorKind::NotFound
            {
                return true;
            }
            current = err.source();
        }
        false
    }

    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("infotheory_api_{name}_{ts}.bin"));
        fs::write(&path, contents).expect("temp fixture write should succeed");
        path
    }

    #[cfg(not(target_env = "musl"))]
    #[test]
    fn api_surface_path_and_compression_helpers_are_callable() {
        let x = b"lorem ipsum dolor sit amet";
        let y = b"lorem ipsum dolor";
        let px = temp_file("x", x);
        let py = temp_file("y", y);
        let sx = px.to_string_lossy().to_string();
        let sy = py.to_string_lossy().to_string();
        let paths = [sx.as_str(), sy.as_str()];

        let backend = CompressionBackend::zpaq("1");
        let compiled = backend.compile().expect("compiled zpaq backend");

        assert!(try_compress_size_backend(x, &compiled).expect("fallible zpaq size") > 0);
        assert!(
            try_compress_size_chain_backend(&[x.as_slice(), y.as_slice()], &compiled)
                .expect("fallible chain size")
                > 0
        );
        let c = try_compress_bytes_backend(x, &compiled).expect("zpaq compress");
        let d = try_decompress_bytes_backend(&c, &compiled).expect("zpaq decompress");
        assert_eq!(d, x);

        assert!(
            try_get_compressed_size_path_backend(&sx, &compiled).expect("fallible file size") > 0
        );

        let bytes_try = try_get_bytes_from_paths(&paths).expect("fallible bytes from paths");
        assert_eq!(bytes_try.len(), 2);
        assert_eq!(bytes_try[0], x);
        assert_eq!(bytes_try[1], y);

        let s_serial = try_get_compressed_sizes_from_paths_backend_with_options(
            &paths,
            &compiled,
            CompressionPathBatchOptions {
                parallelism: OperationParallelism::Serial,
            },
        )
        .expect("serial sizes");
        let s_auto =
            try_get_compressed_sizes_from_paths_backend(&paths, &compiled).expect("auto sizes");
        let s_pool = try_get_compressed_sizes_from_paths_backend_with_options(
            &paths,
            &compiled,
            CompressionPathBatchOptions {
                parallelism: OperationParallelism::Threads(2),
            },
        )
        .expect("pool sizes");
        for sizes in [s_serial, s_auto, s_pool] {
            assert_eq!(sizes.len(), 2);
            assert!(sizes[0] > 0);
            assert!(sizes[1] > 0);
        }

        assert!(
            try_ncd_bytes_backend(x, y, &compiled, NcdVariant::Vitanyi).expect("fallible ncd")
                >= 0.0
        );
        assert!(
            try_ncd_bytes_default(x, y, NcdVariant::SymVitanyi).expect("ncd bytes default") >= 0.0
        );
        assert!(
            try_ncd_bytes_backend(x, y, &compiled, NcdVariant::Cons).expect("ncd bytes backend")
                >= 0.0
        );
        assert!(
            try_ncd_paths_backend(&sx, &sy, &backend, NcdVariant::Vitanyi)
                .expect("fallible file ncd")
                >= 0.0
        );
        assert!(
            try_ncd_paths_compiled_backend(&sx, &sy, &compiled, NcdVariant::SymCons)
                .expect("ncd paths compiled")
                >= 0.0
        );
        let m =
            try_ncd_matrix_bytes_backend(&[x.to_vec(), y.to_vec()], &compiled, NcdVariant::Vitanyi)
                .expect("matrix ncd bytes");
        assert_eq!(m.len(), 4);
        let mp = try_ncd_matrix_paths_backend(&paths, &compiled, NcdVariant::Cons)
            .expect("matrix ncd paths");
        assert_eq!(mp.len(), 4);

        assert!(try_ned_paths(&sx, &sy).expect("ned paths") >= 0.0);
        assert!(try_nte_paths(&sx, &sy).expect("nte paths") >= 0.0);
        assert!(try_tvd_paths(&sx, &sy).expect("tvd paths") >= 0.0);
        assert!(try_nhd_paths(&sx, &sy).expect("nhd paths") >= 0.0);
        assert!(try_mutual_information_paths(&sx, &sy).expect("mi paths") >= 0.0);
        assert!(try_conditional_entropy_paths(&sx, &sy).expect("conditional entropy paths") >= 0.0);
        assert!(try_cross_entropy_paths(&sx, &sy).expect("cross entropy paths") >= 0.0);
        assert!(try_kl_divergence_paths(&sx, &sy).expect("kl paths") >= 0.0);
        assert!(try_js_divergence_paths(&sx, &sy).expect("js paths") >= 0.0);

        let _ = fs::remove_file(px);
        let _ = fs::remove_file(py);
    }

    #[test]
    fn api_surface_fallible_path_helpers_report_missing_files() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let missing_path = std::env::temp_dir().join(format!(
            "infotheory_api_missing_does_not_exist_{unique}.bin"
        ));
        let missing = missing_path.to_string_lossy().to_string();
        let compiled = CompressionBackend::zpaq("1")
            .compile()
            .expect("compile zpaq backend");

        let err = try_get_compressed_size_path_backend(&missing, &compiled)
            .expect_err("missing file should error");
        assert!(
            has_not_found_io_error(&err),
            "expected not-found io error, got: {err}"
        );

        let err =
            try_get_bytes_from_paths(&[&missing]).expect_err("missing bytes path should error");
        assert!(
            has_not_found_io_error(&err),
            "expected not-found io error, got: {err}"
        );

        for err in [
            try_ned_paths(&missing, &missing).expect_err("ned paths should error"),
            try_nte_paths(&missing, &missing).expect_err("nte paths should error"),
            try_nhd_paths(&missing, &missing).expect_err("nhd paths should error"),
            try_mutual_information_paths(&missing, &missing).expect_err("mi paths should error"),
            try_conditional_entropy_paths(&missing, &missing)
                .expect_err("conditional entropy paths should error"),
            try_cross_entropy_paths(&missing, &missing)
                .expect_err("cross entropy paths should error"),
            try_kl_divergence_paths(&missing, &missing).expect_err("kl paths should error"),
            try_js_divergence_paths(&missing, &missing).expect_err("jsd paths should error"),
        ] {
            assert!(
                has_not_found_io_error(&err),
                "expected not-found io error, got: {err}"
            );
        }
    }
}
