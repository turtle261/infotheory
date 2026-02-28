use infotheory::{
    CompressionBackend, InfotheoryCtx, RateBackend, biased_entropy_rate_backend,
    biased_entropy_rate_bytes, conditional_entropy_bytes, conditional_entropy_rate_bytes,
    cross_entropy_bytes, cross_entropy_rate_backend, cross_entropy_rate_bytes, d_kl_bytes,
    entropy_rate_backend, entropy_rate_bytes, get_default_ctx, intrinsic_dependence_bytes,
    joint_entropy_rate_backend, joint_entropy_rate_bytes, joint_marginal_entropy_bytes,
    js_div_bytes, marginal_entropy_bytes, mutual_information_bytes, mutual_information_marg_bytes,
    mutual_information_rate_backend, mutual_information_rate_bytes, ned_bytes, ned_cons_bytes,
    ned_cons_marg_bytes, ned_cons_rate_bytes, ned_marg_bytes, ned_rate_backend, ned_rate_bytes,
    nhd_bytes, nte_bytes, nte_marg_bytes, nte_rate_backend, nte_rate_bytes,
    resistance_to_transformation_bytes, set_default_ctx, tvd_bytes,
};
#[cfg(feature = "backend-zpaq")]
use infotheory::{
    NcdVariant, compress_bytes_backend, compress_size_backend, compress_size_chain_backend,
    conditional_entropy_paths, cross_entropy_paths, decompress_bytes_backend, get_bytes_from_paths,
    get_compressed_size, get_compressed_size_parallel, get_compressed_sizes_from_paths,
    get_parallel_compressed_sizes_from_parallel_paths,
    get_parallel_compressed_sizes_from_sequential_paths,
    get_sequential_compressed_sizes_from_parallel_paths,
    get_sequential_compressed_sizes_from_sequential_paths, js_divergence_paths,
    kl_divergence_paths, mutual_information_paths, ncd_bytes, ncd_bytes_backend, ncd_bytes_default,
    ncd_cons, ncd_matrix_bytes, ncd_matrix_paths, ncd_paths, ncd_paths_backend, ncd_sym_cons,
    ncd_sym_vitanyi, ncd_vitanyi, ned_paths, nhd_paths, nte_paths, tvd_paths,
};
#[cfg(feature = "backend-zpaq")]
use std::fs;
#[cfg(feature = "backend-zpaq")]
use std::path::PathBuf;
#[cfg(feature = "backend-zpaq")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "backend-zpaq")]
fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be monotonic")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("infotheory_api_{name}_{ts}.bin"));
    fs::write(&path, contents).expect("temp fixture write should succeed");
    path
}

#[test]
fn api_surface_entropy_and_distance_wrappers_are_callable() {
    let x = b"alpha beta alpha beta alpha";
    let y = b"alpha gamma alpha gamma alpha";
    let backend = RateBackend::Ctw { depth: 8 };

    let prev = get_default_ctx();
    set_default_ctx(InfotheoryCtx::new(
        backend.clone(),
        CompressionBackend::default(),
    ));

    assert!(entropy_rate_backend(x, -1, &backend) >= 0.0);
    assert!(biased_entropy_rate_backend(x, -1, &backend) >= 0.0);
    assert!(cross_entropy_rate_backend(x, y, -1, &backend) >= 0.0);
    assert!(joint_entropy_rate_backend(x, y, -1, &backend) >= 0.0);
    assert!(mutual_information_rate_backend(x, y, -1, &backend) >= 0.0);
    assert!((0.0..=1.0).contains(&ned_rate_backend(x, y, -1, &backend)));
    assert!((0.0..=2.0).contains(&nte_rate_backend(x, y, -1, &backend)));

    assert!(marginal_entropy_bytes(x) >= 0.0);
    assert!(joint_marginal_entropy_bytes(x, y) >= 0.0);
    assert!(entropy_rate_bytes(x, -1) >= 0.0);
    assert!(biased_entropy_rate_bytes(x, -1) >= 0.0);
    assert!(joint_entropy_rate_bytes(x, y, -1) >= 0.0);
    assert!(conditional_entropy_rate_bytes(x, y, -1) >= 0.0);
    assert!(conditional_entropy_bytes(x, y, 0) >= 0.0);
    assert!(mutual_information_bytes(x, y, 0) >= 0.0);
    assert!(mutual_information_marg_bytes(x, y) >= 0.0);
    assert!(mutual_information_rate_bytes(x, y, -1) >= 0.0);
    assert!((0.0..=1.0).contains(&ned_bytes(x, y, 0)));
    assert!((0.0..=1.0).contains(&ned_marg_bytes(x, y)));
    assert!((0.0..=1.0).contains(&ned_rate_bytes(x, y, -1)));
    assert!((0.0..=1.0).contains(&ned_cons_bytes(x, y, 0)));
    assert!((0.0..=1.0).contains(&ned_cons_marg_bytes(x, y)));
    assert!((0.0..=1.0).contains(&ned_cons_rate_bytes(x, y, -1)));
    assert!((0.0..=2.0).contains(&nte_bytes(x, y, 0)));
    assert!((0.0..=2.0).contains(&nte_marg_bytes(x, y)));
    assert!((0.0..=2.0).contains(&nte_rate_bytes(x, y, -1)));
    assert!((0.0..=1.0).contains(&tvd_bytes(x, y, 0)));
    assert!((0.0..=1.0).contains(&nhd_bytes(x, y, 0)));
    assert!(cross_entropy_bytes(x, y, 0) >= 0.0);
    assert!(cross_entropy_rate_bytes(x, y, -1) >= 0.0);
    assert!(d_kl_bytes(x, y) >= 0.0);
    assert!(js_div_bytes(x, y) >= 0.0);
    assert!((0.0..=1.0).contains(&intrinsic_dependence_bytes(x, -1)));
    assert!((0.0..=1.0).contains(&resistance_to_transformation_bytes(x, y, -1)));

    set_default_ctx(prev);
}

#[cfg(feature = "backend-zpaq")]
#[test]
fn api_surface_path_and_compression_helpers_are_callable() {
    let x = b"lorem ipsum dolor sit amet";
    let y = b"lorem ipsum dolor";
    let px = temp_file("x", x);
    let py = temp_file("y", y);
    let sx = px.to_string_lossy().to_string();
    let sy = py.to_string_lossy().to_string();
    let paths = [sx.as_str(), sy.as_str()];

    let backend = CompressionBackend::Zpaq {
        method: "1".to_string(),
    };

    assert!(compress_size_backend(x, &backend) > 0);
    assert!(compress_size_chain_backend(&[x.as_slice(), y.as_slice()], &backend) > 0);
    let c = compress_bytes_backend(x, &backend).expect("zpaq compress");
    let d = decompress_bytes_backend(&c, &backend).expect("zpaq decompress");
    assert_eq!(d, x);

    assert!(get_compressed_size(&sx, "1") > 0);
    assert!(get_compressed_size_parallel(&sx, "1", 2) > 0);

    let bytes = get_bytes_from_paths(&paths);
    assert_eq!(bytes.len(), 2);
    assert_eq!(bytes[0], x);
    assert_eq!(bytes[1], y);

    let s1 = get_sequential_compressed_sizes_from_sequential_paths(&paths, "1");
    let s2 = get_parallel_compressed_sizes_from_sequential_paths(&paths, "1", 2);
    let s3 = get_sequential_compressed_sizes_from_parallel_paths(&paths, "1");
    let s4 = get_parallel_compressed_sizes_from_parallel_paths(&paths, "1", 2);
    let s5 = get_compressed_sizes_from_paths(&paths, "1");
    for sizes in [s1, s2, s3, s4, s5] {
        assert_eq!(sizes.len(), 2);
        assert!(sizes[0] > 0);
        assert!(sizes[1] > 0);
    }

    assert!(ncd_bytes(x, y, "1", NcdVariant::Vitanyi) >= 0.0);
    assert!(ncd_bytes_default(x, y, NcdVariant::SymVitanyi) >= 0.0);
    assert!(ncd_bytes_backend(x, y, &backend, NcdVariant::Cons) >= 0.0);
    assert!(ncd_paths(&sx, &sy, "1", NcdVariant::SymCons) >= 0.0);
    assert!(ncd_paths_backend(&sx, &sy, &backend, NcdVariant::Vitanyi) >= 0.0);
    assert!(ncd_vitanyi(&sx, &sy, "1") >= 0.0);
    assert!(ncd_sym_vitanyi(&sx, &sy, "1") >= 0.0);
    assert!(ncd_cons(&sx, &sy, "1") >= 0.0);
    assert!(ncd_sym_cons(&sx, &sy, "1") >= 0.0);

    let m = ncd_matrix_bytes(&[x.to_vec(), y.to_vec()], "1", NcdVariant::Vitanyi);
    assert_eq!(m.len(), 4);
    let mp = ncd_matrix_paths(&paths, "1", NcdVariant::Cons);
    assert_eq!(mp.len(), 4);

    assert!(ned_paths(&sx, &sy, 0) >= 0.0);
    assert!(nte_paths(&sx, &sy, 0) >= 0.0);
    assert!(tvd_paths(&sx, &sy, 0) >= 0.0);
    assert!(nhd_paths(&sx, &sy, 0) >= 0.0);
    assert!(mutual_information_paths(&sx, &sy, 0) >= 0.0);
    assert!(conditional_entropy_paths(&sx, &sy, 0) >= 0.0);
    assert!(cross_entropy_paths(&sx, &sy, 0) >= 0.0);
    assert!(kl_divergence_paths(&sx, &sy) >= 0.0);
    assert!(js_divergence_paths(&sx, &sy) >= 0.0);

    let _ = fs::remove_file(px);
    let _ = fs::remove_file(py);
}
