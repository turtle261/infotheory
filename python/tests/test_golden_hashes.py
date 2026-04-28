import hashlib
import json
import pathlib
import subprocess

import infotheory_rs as ait


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _batch_line(payload: dict) -> str:
    proc = subprocess.run(
        ["cargo", "run", "-q", "--features", "cli", "--bin", "infotheory", "--", "batch"],
        cwd=_repo_root(),
        input=(json.dumps(payload) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode("utf-8").strip().splitlines()[0]


def test_zpaq_roundtrip_and_hash_fixture_a():
    data = (
        b"The quick brown fox jumps over the lazy dog.\n"
        b"Sphinx of black quartz, judge my vow.\n"
    )
    enc = ait.compress_bytes_backend(data, compression_backend="zpaq", method="5")
    dec = ait.decompress_bytes_backend(enc, compression_backend="zpaq", method="5")
    assert dec == data
    assert (
        hashlib.sha256(enc).hexdigest()
        == "26ad22d35f5f014d7b99a403af46a0c2b172986352ffee21a03d1f7a39d67498"
    )


def test_zpaq_roundtrip_and_hash_fixture_b():
    data = b"Pack my box with five dozen liquor jugs.\n0123456789abcdef\n"
    enc = ait.compress_bytes_backend(data, compression_backend="zpaq", method="5")
    dec = ait.decompress_bytes_backend(enc, compression_backend="zpaq", method="5")
    assert dec == data
    assert (
        hashlib.sha256(enc).hexdigest()
        == "df691b88c9c1a9791b57f3e7d70fc05c6bb7a324f71e9b45900696472befb837"
    )


def test_batch_metrics_output_hash_stability():
    line = _batch_line({"op": "metrics", "text": "abracadabra"})
    assert line == '{"h0":2.040373,"h_rate":1.763318,"id":0.135787,"len":11}'
    assert (
        hashlib.sha256(line.encode("utf-8")).hexdigest()
        == "11c696ec8f63b2b95df1ed16d6e3360fb3788788a4bde80c2510d45a75cd8f88"
    )

