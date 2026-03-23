import json
import math

import infotheory_rs as ait
import pytest


def _is_finite_nonnegative(x: float) -> bool:
    return math.isfinite(x) and x >= 0.0


def _rwkv7_cfg_method() -> str:
    return (
        "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,"
        "v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;"
        "policy:schedule=0..100:infer"
    )


def _mamba_cfg_method() -> str:
    return (
        "cfg:hidden=64,layers=1,intermediate=128,state=16,conv=4,dt_rank=16,"
        "seed=13,train=none,lr=0.0,stride=1;policy:schedule=0..100:infer"
    )


def test_expected_public_surface_symbols_present():
    names = set(n for n in dir(ait) if not n.startswith("_"))
    expected = {
        "RateBackend",
        "CompressionBackend",
        "InfotheoryCtx",
        "GenerationStrategy",
        "GenerationUpdateMode",
        "GenerationConfig",
        "RateBackendSession",
        "MixtureKind",
        "MixtureExpertSpec",
        "MixtureSpec",
        "ParticleSpec",
        "CalibrationContextKind",
        "NcdVariant",
        "ObservationKeyMode",
        "RandomGenerator",
        "ncd_paths",
        "ncd_bytes",
        "generate_bytes",
        "generate_bytes_conditional_chain",
        "compress_bytes_backend",
        "decompress_bytes_backend",
        "compress_file",
        "decompress_file",
        "ncd_matrix_bytes_with_backend",
        "ncd_matrix_paths_with_backend",
        "mutual_information_bytes",
        "verify_chain_rule",
        "verify_ncd_bounds",
        "search_with_simulator",
        "search",
        "SearchGranularity",
        "Stage2PriorMode",
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
        assert _is_finite_nonnegative(ctx.biased_entropy_rate_bytes(b"abcabcabc", 4))
        assert _is_finite_nonnegative(ctx.compress_size(b"payload"))
        assert _is_finite_nonnegative(ctx.compress_size_chain([b"pay", b"load"]))
        assert _is_finite_nonnegative(
            ctx.cross_entropy_rate_bytes(b"abcabc", b"abcabd", 4)
        )
        assert _is_finite_nonnegative(ctx.cross_entropy_bytes(b"abcabc", b"abcabd", 4))
        assert _is_finite_nonnegative(ctx.joint_entropy_rate_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(
            ctx.conditional_entropy_rate_bytes(b"abc", b"abd", 4)
        )
        assert _is_finite_nonnegative(
            ctx.cross_entropy_conditional_chain([b"ab", b"ca"], b"bc")
        )
        assert _is_finite_nonnegative(ctx.mutual_information_rate_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.mutual_information_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.conditional_entropy_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.ned_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.ned_cons_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.nte_bytes(b"abc", b"abd", 4))
        assert _is_finite_nonnegative(ctx.intrinsic_dependence_bytes(b"abcabc", 4))
        assert _is_finite_nonnegative(
            ctx.resistance_to_transformation_bytes(b"abc", b"abc", 4)
        )
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
        generated = ctx.generate_bytes(
            b"abcabcabc",
            4,
            config=ait.GenerationConfig.greedy_frozen(),
        )
        assert len(generated) == 4
        session = ctx.rate_backend_session(total_symbols=16)
        session.observe(b"abcabc")
        assert len(session.fill_log_probs()) == 256
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


def test_new_rate_backends_parse_and_execute(tmp_path):
    mixture_path = tmp_path / "mixture.json"
    mixture_path.write_text(
        json.dumps(
            {
                "kind": "bayes",
                "experts": [
                    {"name": "match-expert", "kind": "match"},
                    {"name": "ctw-expert", "kind": "ctw", "depth": 8},
                ],
            }
        )
    )
    particle_path = tmp_path / "particle.json"
    particle_path.write_text(
        json.dumps({"num_particles": 4, "num_cells": 4, "cell_dim": 8})
    )
    calibrated_path = tmp_path / "calibrated.json"
    calibrated_path.write_text(
        json.dumps(
            {
                "base": {"kind": "ctw", "depth": 8},
                "context": "text",
                "bins": 17,
                "learning_rate": 0.05,
                "bias_clip": 3.0,
            }
        )
    )

    parsed_backends = [
        ait.rate_backend("match"),
        ait.rate_backend("sparse-match"),
        ait.rate_backend("ppmd", "12"),
        ait.rate_backend("mixture", str(mixture_path)),
        ait.rate_backend("particle", str(particle_path)),
        ait.rate_backend("calibrated", str(calibrated_path)),
    ]

    particle_spec = ait.ParticleSpec(num_particles=4, num_cells=4, cell_dim=8)
    mixture_spec = ait.MixtureSpec(
        ait.MixtureKind.Bayes,
        [
            ait.MixtureExpertSpec(
                ait.RateBackend.match(), max_order=-1, log_prior=0.0, name="match"
            )
        ],
        alpha=0.02,
    )
    constructed_backends = [
        ait.RateBackend.match(hash_bits=18, min_len=3, max_len=96),
        ait.RateBackend.sparse_match(gap_min=2, gap_max=4),
        ait.RateBackend.ppmd(order=8, memory_mb=8),
        ait.RateBackend.mixture(mixture_spec),
        ait.RateBackend.particle(particle_spec),
        ait.RateBackend.calibrated(
            ait.RateBackend.ctw(8),
            ait.CalibrationContextKind.Text,
            bins=17,
            learning_rate=0.05,
            bias_clip=3.0,
        ),
        ait.RateBackend.calibrated(ait.RateBackend.match(), "repeat", bins=9),
    ]

    payload = b"abracadabra abracadabra"
    peer = b"alakazam alakazam"
    for backend in parsed_backends + constructed_backends:
        assert _is_finite_nonnegative(ait.entropy_rate_backend(payload, 4, backend=backend))
        assert _is_finite_nonnegative(
            ait.cross_entropy_rate_backend(payload, peer, 4, backend=backend)
        )

    with pytest.raises(ValueError):
        ait.rate_backend("unknown-backend")


def test_rate_coded_roundtrips_cover_ac_and_rans():
    payload = b"rate coded roundtrip payload"

    match_backend = ait.RateBackend.match()
    cb_ac = ait.CompressionBackend.rate_ac(match_backend, "framed")
    enc_ac = ait.compress_bytes_backend(payload, compression_backend=cb_ac)
    assert ait.decompress_bytes_backend(enc_ac, compression_backend=cb_ac) == payload
    assert (
        ait.compress_size_backend(
            payload,
            compression_backend="rate-ac",
            rate_backend=match_backend,
        )
        > 0
    )

    particle_backend = ait.RateBackend.particle(
        ait.ParticleSpec(num_particles=4, num_cells=4, cell_dim=8)
    )
    cb_rans = ait.CompressionBackend.rate_rans(particle_backend, "framed")
    enc_rans = ait.compress_bytes_backend(payload, compression_backend=cb_rans)
    assert ait.decompress_bytes_backend(enc_rans, compression_backend=cb_rans) == payload
    assert (
        ait.compress_size_backend(
            payload,
            compression_backend="rate-rans",
            rate_backend=particle_backend,
        )
        > 0
    )


def test_rwkv7_string_compression_backend_matches_object_backend():
    method = _rwkv7_cfg_method()
    payload = b"rwkv parity payload"

    string_encoded = ait.compress_bytes_backend(
        payload,
        compression_backend="rwkv7",
        method=method,
    )
    object_backend = ait.CompressionBackend.rwkv7(method)
    object_encoded = ait.compress_bytes_backend(payload, compression_backend=object_backend)

    assert string_encoded == object_encoded
    assert len(string_encoded) > 0


def test_mamba_rate_backend_parse_construct_metrics_and_roundtrip_parity():
    method = _mamba_cfg_method()
    payload = b"mamba parity payload"
    peer = b"mamba parity peer"

    parsed_backend = ait.rate_backend("mamba", method)
    object_backend = ait.RateBackend.mamba(method)

    for backend in (parsed_backend, object_backend):
        assert _is_finite_nonnegative(ait.entropy_rate_backend(payload, 4, backend=backend))
        assert _is_finite_nonnegative(
            ait.cross_entropy_rate_backend(payload, peer, 4, backend=backend)
        )

    framed_from_parsed = ait.CompressionBackend.rate_ac(parsed_backend, "framed")
    framed_from_object = ait.CompressionBackend.rate_ac(object_backend, "framed")
    encoded_from_string = ait.compress_bytes_backend(
        payload,
        compression_backend=framed_from_parsed,
    )
    encoded_from_object = ait.compress_bytes_backend(
        payload,
        compression_backend=framed_from_object,
    )

    assert encoded_from_string == encoded_from_object
    decoded = ait.decompress_bytes_backend(
        encoded_from_string,
        compression_backend=framed_from_object,
    )
    assert decoded == payload


def test_search_pipeline_returns_results(tmp_path):
    # Create a small directory with a few text files
    (tmp_path / "algorithm.txt").write_text(
        "Kolmogorov complexity is the length of the shortest program that "
        "produces a given string. It is a fundamental concept in algorithmic "
        "information theory.\n" * 5
    )
    (tmp_path / "entropy.txt").write_text(
        "Shannon entropy measures the average amount of information produced "
        "by a stochastic source of data. It is measured in bits.\n" * 5
    )
    (tmp_path / "unrelated.txt").write_text(
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs.\n" * 5
    )

    # Verify enum types
    assert ait.SearchGranularity.Snippet != ait.SearchGranularity.File
    assert ait.Stage2PriorMode.Use != ait.Stage2PriorMode.Disable
    assert ait.Stage2PriorMode.Summarize != ait.Stage2PriorMode.Use

    # File-level search
    results = ait.search(
        "Kolmogorov complexity algorithmic information",
        str(tmp_path),
        granularity=ait.SearchGranularity.File,
        top_k=3,
    )
    assert len(results) >= 1, "search should return at least one result"
    # Each result is (path, start_line, end_line, score)
    path, start, end, score = results[0]
    assert isinstance(path, str)
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(score, float)
    assert "algorithm" in path, f"top result should be algorithm.txt, got {path}"
