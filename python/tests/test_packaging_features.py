import pathlib
import re


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _quoted_tokens(csv_like: str) -> list[str]:
    return re.findall(r'"([^"]+)"', csv_like)


def test_pyproject_maturin_features_include_mamba():
    pyproject = (_repo_root() / "pyproject.toml").read_text()
    match = re.search(r"(?m)^\s*features\s*=\s*\[(?P<body>[^\]]+)\]", pyproject)
    assert match is not None, "missing [tool.maturin].features in pyproject.toml"
    features = _quoted_tokens(match.group("body"))
    assert "backend-mamba" in features


def test_python_release_wheel_build_features_include_mamba():
    workflow = (_repo_root() / ".github/workflows/python-release.yml").read_text()
    feature_args = re.findall(r"maturin build --release --features ([^\n]+)", workflow)
    assert feature_args, "no maturin build commands found in python-release.yml"
    for args in feature_args:
        features = [part.strip() for part in args.strip().split(",")]
        assert "backend-mamba" in features


def test_python_ci_explicit_feature_builds_include_mamba():
    workflow = (_repo_root() / ".github/workflows/python.yml").read_text()
    feature_args = re.findall(r"maturin develop --release --features ([^\n]+)", workflow)
    assert feature_args, "no explicit maturin develop feature commands found in python.yml"
    for args in feature_args:
        features = [part.strip() for part in args.strip().split(",")]
        assert "backend-mamba" in features


def test_python_release_linux_build_targets_manylinux2014():
    workflow = (_repo_root() / ".github/workflows/python-release.yml").read_text()
    assert "PyO3/maturin-action@v1" in workflow
    assert "manylinux: 2014" in workflow


def test_python_release_linux_build_overrides_local_linker_and_uses_py310_abi3_base():
    workflow = (_repo_root() / ".github/workflows/python-release.yml").read_text()
    assert "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER: gcc" in workflow
    assert "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS: -C target-cpu=generic" in workflow
    assert "--interpreter /opt/python/cp310-cp310/bin/python3" in workflow
