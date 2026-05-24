import math

import infotheory_rs as ait
import pytest


PROMPT = (
    b"If a frog is green, dogs are red.\n"
    b"If a toad is green, cats are red.\n"
    b"If a dog is green, frogs are red.\n"
    b"If a cat is green, toads are red.\n"
    b"If a frog is red, dogs are green.\n"
    b"If a toad is red, cats are green.\n"
    b"If a dog is red, frogs are green.\n"
    b"If a cat is red, toads are \n"
)
EXPECTED_ROSA_CONTINUATION = b" green.\n"


def _rwkv7_cfg_method() -> str:
    return (
        "cfg:hidden=64,layers=1,intermediate=64,decay_rank=8,a_rank=8,"
        "v_rank=8,g_rank=8,seed=11,train=none,lr=0.0,stride=1;"
        "policy:schedule=0..100:infer"
    )


def _finite_log_probs(logps):
    return len(logps) == 256 and all(math.isfinite(v) for v in logps)


def test_rosa_generation_regression_matches_rust_behavior():
    cfg = ait.GenerationConfig.sampled_frozen(42)
    rosa = ait.RateBackend.rosaplus()
    ctx = ait.InfotheoryCtx(rosa, ait.CompressionBackend.zpaq("5"))

    top_level = ait.generate_bytes(PROMPT, 8, backend=rosa, config=cfg)
    from_ctx = ctx.generate_bytes(PROMPT, 8, config=cfg)

    session = ctx.rate_backend_session(total_symbols=len(PROMPT) + 8)
    session.observe(PROMPT)
    from_session = session.generate_bytes(8, cfg)

    assert top_level == EXPECTED_ROSA_CONTINUATION
    assert from_ctx == EXPECTED_ROSA_CONTINUATION
    assert from_session == EXPECTED_ROSA_CONTINUATION


def test_generation_conditional_chain_matches_flat_prompt():
    cfg = ait.GenerationConfig.sampled_frozen(42)
    backend = ait.RateBackend.rosaplus()
    split = len(PROMPT) // 2
    parts = [PROMPT[:split], PROMPT[split:]]

    flat = ait.generate_bytes(PROMPT, 8, backend=backend, config=cfg)
    chained = ait.generate_bytes_conditional_chain(parts, 8, backend=backend, config=cfg)

    assert flat == chained == EXPECTED_ROSA_CONTINUATION


def test_generation_session_fill_log_probs_and_reset_frozen():
    cfg = ait.GenerationConfig.sampled_frozen(7)
    session = ait.RateBackendSession(
        ait.RateBackend.ctw(32),
        total_symbols=len(PROMPT) + 8,
    )

    session.observe(PROMPT[:64])
    assert _finite_log_probs(session.fill_log_probs())

    session.reset_frozen(len(PROMPT) + 8)
    session.condition(PROMPT)
    first = session.generate_bytes(8, cfg)

    twin = ait.RateBackendSession(
        ait.RateBackend.ctw(32),
        total_symbols=len(PROMPT) + 8,
    )
    twin.observe(PROMPT[:64])
    twin.reset_frozen(len(PROMPT) + 8)
    twin.condition(PROMPT)
    second = twin.generate_bytes(8, cfg)

    assert len(first) == 8
    assert first == second


def test_generation_session_begin_stream_restarts_zpaq_without_frozen_reset():
    session = ait.RateBackendSession(ait.RateBackend.zpaq("1"), total_symbols=9)

    with pytest.raises(RuntimeError, match="plugin entropy"):
        session.reset_frozen(9)

    session.begin_stream(9)
    initial = session.fill_log_probs()
    assert _finite_log_probs(initial)

    session.observe(bytes([0, 1, 0, 1, 1, 0, 1, 0, 1]))
    session.begin_stream(9)
    restarted = session.fill_log_probs()
    assert _finite_log_probs(restarted)

    fresh = ait.RateBackendSession(ait.RateBackend.zpaq("1"), total_symbols=9)
    expected = fresh.fill_log_probs()
    assert _finite_log_probs(expected)
    for idx in range(256):
        assert abs(restarted[idx] - expected[idx]) < 1e-12
        assert abs(initial[idx] - restarted[idx]) < 1e-12

    session.finish()
    fresh.finish()


def test_generation_session_begin_stream_restarts_mixture_with_zpaq_without_frozen_reset():
    backend = ait.RateBackend.mixture(
        ait.MixtureSpec(
            ait.MixtureKind.Bayes,
            [
                ait.MixtureExpertSpec(ait.RateBackend.ctw(6)),
                ait.MixtureExpertSpec(ait.RateBackend.zpaq("1")),
            ],
        )
    )
    session = ait.RateBackendSession(backend, total_symbols=9)

    with pytest.raises(RuntimeError, match="plugin entropy"):
        session.reset_frozen(9)

    session.begin_stream(9)
    initial = session.fill_log_probs()
    assert _finite_log_probs(initial)

    session.observe(bytes([1, 0, 1, 0, 1, 1, 0, 0, 1]))
    session.begin_stream(9)
    restarted = session.fill_log_probs()
    assert _finite_log_probs(restarted)

    fresh = ait.RateBackendSession(backend, total_symbols=9)
    expected = fresh.fill_log_probs()
    assert _finite_log_probs(expected)
    any_changed = any(abs(restarted[idx] - expected[idx]) > 1e-12 for idx in range(256))
    assert any_changed, "mixture+zpaq restart should preserve fitted state from resettable experts"

    session.finish()
    fresh.finish()


def test_generation_is_deterministic_across_core_backends():
    cfg = ait.GenerationConfig.sampled_frozen(42)
    cases = [
        ("ctw", ait.RateBackend.ctw(32)),
        ("rosaplus", ait.RateBackend.rosaplus()),
        ("match", ait.RateBackend.match()),
        ("ppmd", ait.RateBackend.ppmd(order=10, memory_mb=8)),
        ("rwkv7", ait.RateBackend.rwkv7(_rwkv7_cfg_method())),
    ]

    for name, backend in cases:
        first = ait.generate_bytes(
            PROMPT,
            8,
            backend=backend,
            config=cfg,
        )
        second = ait.generate_bytes(
            PROMPT,
            8,
            backend=backend,
            config=cfg,
        )
        assert len(first) == 8, name
        assert first == second, name


def test_file_roundtrip_backend_matrices_and_axiom_helpers(tmp_path):
    payload = b"generation python binding payload"
    input_path = tmp_path / "input.bin"
    compressed_path = tmp_path / "payload.it"
    output_path = tmp_path / "output.bin"
    input_path.write_bytes(payload)

    match_backend = ait.RateBackend.match()
    rate_ac = ait.CompressionBackend.rate_ac(match_backend, "framed")
    ait.compress_file(str(input_path), str(compressed_path), compression_backend=rate_ac)
    ait.decompress_file(str(compressed_path), str(output_path), compression_backend=rate_ac)
    assert output_path.read_bytes() == payload

    datas = [b"abcabc", b"abcabd", b"xyzxyz"]
    matrix_bytes = ait.ncd_matrix_bytes_with_backend(datas, backend=rate_ac, variant="sym")
    assert len(matrix_bytes) == 9

    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    path_c = tmp_path / "c.txt"
    path_a.write_bytes(datas[0])
    path_b.write_bytes(datas[1])
    path_c.write_bytes(datas[2])
    matrix_paths = ait.ncd_matrix_paths_with_backend(
        [str(path_a), str(path_b), str(path_c)],
        backend=rate_ac,
        variant="sym",
    )
    assert len(matrix_paths) == 9

    assert ait.verify_identity(datas[0], datas[1], tolerance=1.0)
    assert ait.verify_symmetry(datas[0], datas[1], tolerance=1.0)
    assert ait.verify_non_negativity(datas[0], datas[1])
    assert ait.verify_mi_nonnegative(datas[0], datas[1])
    assert ait.verify_subadditivity(datas[0], datas[1], tolerance=1.0)
    assert ait.verify_conditioning_reduces_entropy(datas[0], datas[1], tolerance=1.0)
    assert ait.verify_chain_rule(datas[0], datas[1], tolerance=2.0)
    assert ait.verify_ncd_bounds(datas[0], datas[1])
    assert ait.verify_entropy_bounds(datas[0])


def test_file_roundtrip_with_string_rate_backend_defaults_to_framed(tmp_path):
    payload = b"string rate backend file roundtrip payload"
    input_path = tmp_path / "input.bin"
    compressed_path = tmp_path / "payload.it"
    output_path = tmp_path / "output.bin"
    input_path.write_bytes(payload)

    match_backend = ait.RateBackend.match()
    ait.compress_file(
        str(input_path),
        str(compressed_path),
        compression_backend="rate-ac",
        rate_backend=match_backend,
    )
    ait.decompress_file(
        str(compressed_path),
        str(output_path),
        compression_backend="rate-ac",
        rate_backend=match_backend,
    )

    assert output_path.read_bytes() == payload
