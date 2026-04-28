import json
import pathlib
import subprocess

import infotheory_rs as ait


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _batch(payload: dict) -> dict:
    proc = subprocess.run(
        ["cargo", "run", "-q", "--features", "cli", "--bin", "infotheory", "--", "batch"],
        cwd=_repo_root(),
        input=(json.dumps(payload) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout.decode("utf-8").strip().splitlines()[0])


def _close(a: float, b: float, tol: float = 1e-6) -> None:
    assert abs(a - b) <= tol


def test_metrics_and_cross_entropy_parity():
    x = "abracadabra"
    y = "alakazam"
    metrics = _batch({"op": "metrics", "text": x})
    _close(metrics["h0"], ait.empirical_entropy_bytes(x.encode()))
    _close(metrics["h_rate"], ait.entropy_rate_bytes(x.encode()))
    cross = _batch({"op": "cross_entropy", "text_x": x, "text_y": y})
    _close(cross["cross_entropy"], ait.cross_entropy_rate_bytes(x.encode(), y.encode()))


def test_ncd_file_and_matrix_parity(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_text("abracadabra")
    b.write_text("alakazam")
    c.write_text("xyzxyz")

    rust = _batch(
        {
            "op": "ncd_files",
            "path1": str(a),
            "path2": str(b),
            "method": "5",
            "variant": "vitanyi",
        }
    )
    py = ait.ncd_paths(str(a), str(b), method="5", variant="vitanyi")
    _close(rust["ncd"], py)

    rust_matrix = _batch(
        {
            "op": "ncd_matrix",
            "texts": ["abracadabra", "alakazam", "xyzxyz"],
            "method": "5",
            "variant": "sym",
        }
    )
    py_matrix = ait.ncd_matrix_bytes(
        [b"abracadabra", b"alakazam", b"xyzxyz"], method="5", variant="sym"
    )
    n = rust_matrix["n"]
    assert n == 3
    flat = [v for row in rust_matrix["matrix"] for v in row]
    assert len(py_matrix) == len(flat)
    for rv, pv in zip(flat, py_matrix):
        _close(rv, pv)

