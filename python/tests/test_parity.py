import json
import pathlib
import subprocess

import infotheory_rs as ait


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _cargo_batch(payload: dict) -> dict:
    cmd = [
        "cargo",
        "run",
        "-q",
        "--features",
        "cli",
        "--bin",
        "infotheory",
        "--",
        "batch",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_repo_root(),
        input=(json.dumps(payload) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    line = proc.stdout.decode("utf-8").strip().splitlines()[0]
    return json.loads(line)


def test_ncd_paths_backend_keyword(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello world")
    b.write_text("hello world")
    d = ait.ncd_paths(str(a), str(b), backend="zpaq", method="5", variant="vitanyi")
    assert d >= 0.0


def test_compress_decompress_roundtrip_zpaq():
    data = b"roundtrip-data" * 16
    enc = ait.compress_bytes_backend(data, compression_backend="zpaq", method="5")
    dec = ait.decompress_bytes_backend(enc, compression_backend="zpaq", method="5")
    assert dec == data


def test_python_rust_cli_parity_ncd_file(tmp_path):
    a = tmp_path / "x.txt"
    b = tmp_path / "y.txt"
    a.write_text("abracadabra")
    b.write_text("alakazam")

    py_val = ait.ncd_paths(str(a), str(b), method="5", variant="vitanyi")
    rust_out = _cargo_batch(
        {
            "op": "ncd_files",
            "path1": str(a),
            "path2": str(b),
            "method": "5",
            "variant": "vitanyi",
        }
    )
    rust_val = float(rust_out["ncd"])
    assert abs(py_val - rust_val) < 1e-5
