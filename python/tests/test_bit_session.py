import math
import pytest

import infotheory_rs as ait


def _normalize_pdf(weights):
    total = sum(weights)
    return [weight / total for weight in weights]


def test_bit_order():
    assert hasattr(ait.BitOrder, "MsbFirst")
    assert hasattr(ait.BitOrder, "LsbFirst")

    msb = ait.BitOrder.MsbFirst
    lsb = ait.BitOrder.LsbFirst

    assert repr(msb) == "BitOrder.MsbFirst"
    assert repr(lsb) == "BitOrder.LsbFirst"


def test_bit_stream_semantics():
    msb = ait.BitOrder.MsbFirst
    lsb = ait.BitOrder.LsbFirst

    bp_default = ait.BitStreamSemantics.byte_packed()
    assert bp_default.kind == "byte_packed"
    assert bp_default.order == msb

    bp_lsb = ait.BitStreamSemantics.byte_packed(lsb)
    assert bp_lsb.kind == "byte_packed"
    assert bp_lsb.order == lsb

    # Check that we can also pass a string alias for the order
    bp_lsb_str = ait.BitStreamSemantics.byte_packed("lsb_first")
    assert bp_lsb_str.kind == "byte_packed"
    assert bp_lsb_str.order == lsb

    bt = ait.BitStreamSemantics.binary_tokens()
    assert bt.kind == "binary_tokens"
    assert bt.order is None

    assert "byte_packed" in repr(bp_default)
    assert "binary_tokens" in repr(bt)


def test_binary_prediction():
    pred = ait.BinaryPrediction(0.3, 0.7)
    assert pred == ait.BinaryPrediction.from_prob_one_exact(0.7)
    assert pred.p0 == 1.0 - pred.p1
    assert pred.p1 == 0.7
    assert pred.prob(False) == pred.p0
    assert pred.prob(True) == pred.p1
    assert "BinaryPrediction" in repr(pred)

    exact = ait.BinaryPrediction.from_prob_one_exact(0.9)
    assert exact.p1 == 0.9
    assert abs(exact.p0 - 0.1) < 1e-12

    floored = ait.BinaryPrediction.from_prob_one(1.2, 0.01)
    assert floored.p1 == 0.99
    assert abs(floored.p0 - 0.01) < 1e-12

    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must be finite"):
            ait.BinaryPrediction.from_prob_one(invalid)
        with pytest.raises(ValueError, match="must be finite"):
            ait.BinaryPrediction.from_prob_one_exact(invalid)

    # Accept tiny floating-point drift, but canonicalize back to an exact complement.
    pred_near_one = ait.BinaryPrediction(math.nextafter(1.0, 0.0), 0.0)
    assert pred_near_one.p0 == 1.0
    assert pred_near_one.p1 == 0.0

    # Even an exact rounded sum of 1.0 must still canonicalize through P(1).
    pathological_p0 = 0.9990000000000001
    pathological_p1 = 0.0009999999999999979
    pathological = ait.BinaryPrediction(pathological_p0, pathological_p1)
    assert pathological.p1 == pathological_p1
    assert pathological.p0 == 1.0 - pathological_p1
    assert pathological.p0 != pathological_p0

    ulp = math.ulp(1.0)
    pred_boundary = ait.BinaryPrediction(0.5 - 4.0 * ulp, 0.5)
    assert pred_boundary.p0 + pred_boundary.p1 == 1.0

    with pytest.raises(ValueError, match="must sum to 1"):
        ait.BinaryPrediction(0.5 - 5.0 * ulp, 0.5)
    with pytest.raises(ValueError, match="must sum to 1"):
        ait.BinaryPrediction(0.3, 0.4)
    with pytest.raises(ValueError, match="must be finite and >= 0"):
        ait.BinaryPrediction(2.0, -1.0)
    with pytest.raises(ValueError, match="must sum to 1"):
        ait.BinaryPrediction(0.99999, 0.0)

    pred_norm = ait.BinaryPrediction(0.33333333, 0.66666667)
    assert pred_norm.p0 + pred_norm.p1 == 1.0


def test_byte_prefix_mass_from_pdf_tracks_symbol_probability():
    pdf = _normalize_pdf([idx + 1 for idx in range(256)])
    symbol = 0b1010_0110
    prefix = ait.BytePrefixMass.from_pdf(pdf, ait.BitOrder.MsbFirst)

    assert not prefix.is_complete()
    assert not prefix.has_partial_bits()
    with pytest.raises(RuntimeError, match="only meaningful after a full byte"):
        prefix.symbol()

    product = 1.0
    for bit_idx in range(8):
        bit = ((symbol >> (7 - bit_idx)) & 1) == 1
        pred = prefix.prediction()
        product *= pred.prob(bit)
        prefix.observe(bit)
        if bit_idx < 7:
            assert prefix.has_partial_bits()

    assert prefix.is_complete()
    assert not prefix.has_partial_bits()
    assert prefix.symbol() == symbol
    assert abs(product - pdf[symbol]) < 1e-12
    assert "complete=True" in repr(prefix)


def test_byte_prefix_mass_from_log_probs_matches_backend_row():
    backend = ait.RateBackend.ctw(6)
    byte_session = ait.RateBackendSession(backend, total_symbols=16)

    for symbol in b"bit-session":
        log_probs = byte_session.fill_log_probs()
        prefix = ait.BytePrefixMass.from_log_probs(log_probs, "msb_first")
        expected = math.exp(log_probs[symbol])

        product = 1.0
        for bit_idx in range(8):
            bit = ((symbol >> (7 - bit_idx)) & 1) == 1
            pred = prefix.prediction()
            product *= pred.prob(bit)
            prefix.observe(bit)

        assert prefix.is_complete()
        assert prefix.symbol() == symbol
        assert abs(product - expected) < 1e-9
        byte_session.observe(bytes([symbol]))

    byte_session.finish()


def test_byte_prefix_mass_rejects_wrong_row_lengths():
    with pytest.raises(ValueError, match="expects exactly 256 entries"):
        ait.BytePrefixMass.from_pdf([0.5, 0.5])
    with pytest.raises(ValueError, match="expects exactly 256 entries"):
        ait.BytePrefixMass.from_log_probs([0.0] * 255)


def test_byte_packed_bit_session_matches_byte_prediction_chain():
    backend = ait.RateBackend.ctw(6)

    # 16 symbols * 8 bits = 128 bits
    byte_session = ait.RateBackendSession(backend, total_symbols=16)
    bit_session = ait.RateBackendBitSession(
        backend,
        total_bits=128,
        semantics=ait.BitStreamSemantics.byte_packed(ait.BitOrder.MsbFirst),
    )

    data = b"bit-session"
    for symbol in data:
        row = byte_session.fill_log_probs()
        expected = math.exp(row[symbol])

        product = 1.0
        for bit_idx in range(8):
            bit = ((symbol >> (7 - bit_idx)) & 1) == 1
            pred = bit_session.step_bit(bit)
            product *= pred.prob(bit)

        byte_session.observe(bytes([symbol]))
        assert abs(product - expected) < 1e-9

    byte_session.finish()
    bit_session.finish()


def test_bit_session_string_aliases():
    # Verify string alias parsing
    backend = ait.RateBackend.ctw(6)

    # "byte" or "byte_packed" as string alias
    sess_byte = ait.RateBackendBitSession(backend, total_bits=8, semantics="byte")
    sess_byte.finish()

    # "binary" or "binary_tokens" as string alias
    sess_bit = ait.RateBackendBitSession(backend, total_bits=9, semantics="binary")
    sess_bit.finish()


def test_byte_packed_bit_session_rejects_non_byte_aligned_lengths():
    backend = ait.RateBackend.ctw(6)

    # Try non-byte aligned total_bits
    with pytest.raises(RuntimeError, match="whole number of bytes"):
        ait.RateBackendBitSession(backend, total_bits=9, semantics="byte")

    sess = ait.RateBackendBitSession(backend, total_bits=8, semantics="byte")

    with pytest.raises(RuntimeError, match="whole number of bytes"):
        sess.reset_frozen(total_bits=9)


def test_bit_session_predict_and_condition():
    backend = ait.RateBackend.ctw(6)
    sess = ait.RateBackendBitSession(backend, total_bits=8, semantics="binary")

    pred1 = sess.predict_bit()
    p1_val = sess.predict_one()
    assert abs(pred1.p1 - p1_val) < 1e-12

    sess.condition_bit(True)
    sess.observe_bit(False)
    sess.finish()


def test_byte_packed_bit_session_rejects_mixed_update_modes_without_panicking_python():
    backend = ait.RateBackend.ctw(6)
    sess = ait.RateBackendBitSession(backend, total_bits=8, semantics="byte")

    sess.condition_bit(True)

    with pytest.raises(RuntimeError, match="cannot mix conditioning-only and adaptive updates"):
        sess.observe_bit(False)


def test_zpaq_bit_session_begin_bit_stream_restarts_without_frozen_reset():
    sess = ait.RateBackendBitSession(ait.RateBackend.zpaq("1"), total_bits=9, semantics="binary")

    with pytest.raises(RuntimeError, match="plugin entropy"):
        sess.reset_frozen(total_bits=9)

    sess.begin_bit_stream(total_bits=9)

    for bit in [True, False, True, True, False, False, True, False, True]:
        pred = sess.step_bit(bit)
        assert abs((pred.p0 + pred.p1) - 1.0) < 1e-12

    with pytest.raises(RuntimeError, match="semantics are fixed"):
        sess.begin_bit_stream(total_bits=9, semantics="byte")

    sess.finish()


def test_mixture_with_zpaq_bit_session_begin_bit_stream_restarts_without_frozen_reset():
    mixture_spec = ait.MixtureSpec(
        ait.MixtureKind.Bayes,
        [
            ait.MixtureExpertSpec(ait.RateBackend.ctw(6)),
            ait.MixtureExpertSpec(ait.RateBackend.zpaq("1")),
        ],
    )
    backend = ait.RateBackend.mixture(mixture_spec)
    sess = ait.RateBackendBitSession(backend, total_bits=9, semantics="binary")

    with pytest.raises(RuntimeError, match="plugin entropy"):
        sess.reset_frozen(total_bits=9)

    sess.begin_bit_stream(total_bits=9)

    for bit in [True, False, True, False, True, True, False, False, True]:
        pred = sess.step_bit(bit)
        assert abs((pred.p0 + pred.p1) - 1.0) < 1e-12

    sess.finish()


def test_ctx_rate_backend_bit_session_delegates_to_default_backend():
    rb = ait.RateBackend.ctw(8)
    cb = ait.CompressionBackend.zpaq("5")
    ctx = ait.InfotheoryCtx(rb, cb)

    sess = ctx.rate_backend_bit_session(total_bits=8, semantics="binary")

    pred = sess.predict_bit()
    assert isinstance(pred, ait.BinaryPrediction)

    sess.observe_bit(True)
    sess.finish()


def test_bit_session_checkpoint_restore_binary_tokens_roundtrip():
    backend = ait.RateBackend.ctw(6)
    sess = ait.RateBackendBitSession(backend, total_bits=12, semantics="binary")

    for bit in [True, False, True]:
        sess.observe_bit(bit)

    pred_before = sess.predict_bit()
    checkpoint = sess.checkpoint()

    for bit in [False, False, True, True]:
        sess.observe_bit(bit)

    sess.restore_checkpoint(checkpoint)
    pred_after = sess.predict_bit()
    assert abs(pred_before.p1 - pred_after.p1) < 1e-12

    for bit in [True, False, False, True, True, False, True, False, False]:
        sess.observe_bit(bit)
    sess.finish()


def test_bit_session_checkpoint_restore_mid_prefix_byte_packed():
    backend = ait.RateBackend.ctw(6)
    sess = ait.RateBackendBitSession(
        backend,
        total_bits=8,
        semantics=ait.BitStreamSemantics.byte_packed(ait.BitOrder.MsbFirst),
    )

    sess.condition_bit(True)
    sess.condition_bit(False)
    pred_before = sess.predict_one()
    checkpoint = sess.checkpoint()

    sess.condition_bit(True)
    sess.condition_bit(True)
    sess.condition_bit(False)

    sess.restore_checkpoint(checkpoint)
    pred_after = sess.predict_one()
    assert abs(pred_before - pred_after) < 1e-12

    sess.clear_checkpoints_if_supported()
    for bit in [True, False, True, False, True, False]:
        sess.condition_bit(bit)
    sess.finish()


def test_bit_session_checkpoint_rejects_mismatched_semantics():
    backend = ait.RateBackend.ctw(6)
    binary = ait.RateBackendBitSession(backend, total_bits=8, semantics="binary")
    byte = ait.RateBackendBitSession(backend, total_bits=8, semantics="byte")

    checkpoint = binary.checkpoint()
    with pytest.raises(RuntimeError, match="different backend or bit semantics"):
        byte.restore_checkpoint(checkpoint)
