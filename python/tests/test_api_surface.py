import math

import infotheory_rs as ait


def _is_finite_nonnegative(x: float) -> bool:
    return math.isfinite(x) and x >= 0.0


def test_expected_public_surface_symbols_present():
    names = set(n for n in dir(ait) if not n.startswith("_"))
    expected = {
        "RateBackend",
        "CompressionBackend",
        "InfotheoryCtx",
        "MixtureKind",
        "MixtureExpertSpec",
        "MixtureSpec",
        "NcdVariant",
        "ObservationKeyMode",
        "RandomGenerator",
        "ncd_paths",
        "ncd_bytes",
        "compress_bytes_backend",
        "decompress_bytes_backend",
        "mutual_information_bytes",
        "search_with_simulator",
        "vm_enabled",
    }
    missing = expected - names
    assert not missing, f"missing exported symbols: {sorted(missing)}"


def test_functional_metrics_surface_bytes_and_matrix():
    x = b"abracadabra"
    y = b"alakazam"
    assert _is_finite_nonnegative(ait.marginal_entropy_bytes(x))
    assert _is_finite_nonnegative(ait.entropy_rate_bytes(x, 4))
    assert _is_finite_nonnegative(ait.biased_entropy_rate_bytes(x, 4))
    assert _is_finite_nonnegative(ait.joint_marginal_entropy_bytes(x, y))
    assert _is_finite_nonnegative(ait.joint_entropy_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.conditional_entropy_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.conditional_entropy_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.mutual_information_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.mutual_information_marg_bytes(x, y))
    assert _is_finite_nonnegative(ait.mutual_information_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.ned_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.ned_marg_bytes(x, y))
    assert _is_finite_nonnegative(ait.ned_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.ned_cons_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.ned_cons_marg_bytes(x, y))
    assert _is_finite_nonnegative(ait.ned_cons_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.nte_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.nte_marg_bytes(x, y))
    assert _is_finite_nonnegative(ait.nte_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.tvd_bytes(x, y, 0))
    assert _is_finite_nonnegative(ait.nhd_bytes(x, y, 0))
    assert _is_finite_nonnegative(ait.cross_entropy_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.cross_entropy_rate_bytes(x, y, 4))
    assert _is_finite_nonnegative(ait.d_kl_bytes(x, y))
    assert _is_finite_nonnegative(ait.js_div_bytes(x, y))
    assert _is_finite_nonnegative(ait.intrinsic_dependence_bytes(x, 4))
    assert _is_finite_nonnegative(ait.resistance_to_transformation_bytes(x, x, 4))
    matrix = ait.ncd_matrix_bytes([x, y, b"xyzxyz"], method="5", variant="sym")
    assert len(matrix) == 9


def test_backend_objects_context_and_helpers(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello there")

    rb = ait.RateBackend.ctw(8)
    cb = ait.CompressionBackend.zpaq("5")
    ctx = ait.InfotheoryCtx(rb, cb)
    prev = ait.get_default_ctx()
    try:
        ait.set_default_ctx(ctx)
        assert isinstance(ait.get_default_ctx(), ait.InfotheoryCtx)
        assert _is_finite_nonnegative(ctx.entropy_rate_bytes(b"abcabcabc", 4))
        assert _is_finite_nonnegative(ctx.ncd_bytes(b"abc", b"abd", "vitanyi"))
        assert _is_finite_nonnegative(
            ait.ncd_paths(str(a), str(b), backend="zpaq", method="5", variant="vitanyi")
        )
        assert _is_finite_nonnegative(
            ait.ncd_bytes(b"hello world", b"hello there", method="5", variant="vitanyi")
        )
        assert _is_finite_nonnegative(
            ait.ncd_paths_with_backend(str(a), str(b), backend=cb, variant="vitanyi")
        )
        assert _is_finite_nonnegative(
            ait.ncd_bytes_with_backend(b"abc", b"abd", backend=cb, variant="vitanyi")
        )
        assert _is_finite_nonnegative(ait.ncd_bytes_default(b"abc", b"abd", "vitanyi"))
        assert _is_finite_nonnegative(ait.ncd_vitanyi(str(a), str(b), "5"))
        assert _is_finite_nonnegative(ait.ncd_sym_vitanyi(str(a), str(b), "5"))
        assert _is_finite_nonnegative(ait.ncd_cons(str(a), str(b), "5"))
        assert _is_finite_nonnegative(ait.ncd_sym_cons(str(a), str(b), "5"))
        assert _is_finite_nonnegative(ait.get_compressed_size(str(a), "5"))
        assert len(ait.get_compressed_sizes_from_paths([str(a), str(b)], "5")) == 2
        assert len(ait.get_bytes_from_paths([str(a), str(b)])) == 2
        assert _is_finite_nonnegative(ait.compress_size_backend(b"payload", "zpaq", "5"))
        assert _is_finite_nonnegative(
            ait.compress_size_chain_backend([b"pay", b"load"], "zpaq", "5")
        )
        compressed = ait.compress_bytes_backend(b"payload", "zpaq", "5")
        assert ait.decompress_bytes_backend(compressed, "zpaq", "5") == b"payload"
        assert ait.validate_zpaq_rate_method("1") is None
        assert _is_finite_nonnegative(ait.entropy_rate_backend(b"abc", 4, backend=rb))
        assert _is_finite_nonnegative(ait.biased_entropy_rate_backend(b"abc", 4, backend=rb))
        assert _is_finite_nonnegative(
            ait.joint_entropy_rate_backend(b"abc", b"abd", 4, backend=rb)
        )
        assert _is_finite_nonnegative(
            ait.mutual_information_rate_backend(b"abc", b"abd", 4, backend=rb)
        )
        assert _is_finite_nonnegative(ait.ned_rate_backend(b"abc", b"abd", 4, backend=rb))
        assert _is_finite_nonnegative(ait.nte_rate_backend(b"abc", b"abd", 4, backend=rb))
        assert _is_finite_nonnegative(
            ait.cross_entropy_rate_backend(b"abc", b"abd", 4, backend=rb)
        )
    finally:
        ait.set_default_ctx(prev)


def test_bit_and_observation_helpers():
    bits = ait.encode_bits(7, 3)
    assert bits == [True, True, True]
    assert ait.decode_bits(bits, 3) == 7
    rb = ait.encode_reward_bits(-1, 8)
    assert ait.decode_reward_bits(rb, 8) == -1
    orb = ait.encode_reward_offset_bits(5, 8, 128)
    assert ait.decode_reward_offset_bits(orb, 8, 128) == 5
    stream = [3, 9, 12]
    assert isinstance(ait.observation_key_from_stream("first", stream, 8), int)
    assert isinstance(
        ait.observation_key_from_stream(ait.ObservationKeyMode.StreamHash, stream, 8), int
    )
    assert isinstance(ait.observation_repr_from_stream("last", stream, 8), list)
