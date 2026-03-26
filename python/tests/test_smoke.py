import infotheory_rs as ait


def test_import_and_core_metrics():
    assert isinstance(ait.vm_enabled(), bool)
    x = b"abracadabra"
    y = b"alakazam"
    assert ait.marginal_entropy_bytes(x) >= 0.0
    assert ait.mutual_information_bytes(x, y, 0) >= 0.0


def test_backend_and_ctx_usage():
    rb = ait.RateBackend.ctw(8)
    cb = ait.CompressionBackend.zpaq("5")
    ctx = ait.InfotheoryCtx(rb, cb)
    d = ctx.ncd_bytes(b"hello", b"hello", "vitanyi")
    assert d >= 0.0


def test_ncd_paths_with_backend(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello world")
    b.write_text("hello world")
    d = ait.ncd_paths(str(a), str(b), backend="zpaq", method="5", variant="vitanyi")
    assert d >= 0.0
