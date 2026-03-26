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
    feature_args = re.findall(r"maturin develop --profile python-release --features ([^\n]+)", workflow)
    assert feature_args, "no explicit maturin develop feature commands found in python.yml"
    for args in feature_args:
        features = [part.strip() for part in args.strip().split(",")]
        assert "backend-mamba" in features


def test_python_ci_linux_uses_clang_and_lld_for_python_release_builds():
    workflow = (_repo_root() / ".github/workflows/python.yml").read_text()
    assert "CC: clang" in workflow
    assert "CXX: clang++" in workflow
    assert "RUSTFLAGS: -C link-arg=-fuse-ld=lld -C target-cpu=x86-64" in workflow
    assert "uv run --no-sync pytest" in workflow
    assert "uv run --no-sync pip install" in workflow


def test_python_release_linux_build_targets_manylinux2014():
    workflow = (_repo_root() / ".github/workflows/python-release.yml").read_text()
    assert "--compatibility manylinux2014" in workflow
    assert "--zig" in workflow


def test_python_release_linux_build_overrides_local_linker_and_uses_py310_abi3_base():
    workflow = (_repo_root() / ".github/workflows/python-release.yml").read_text()
    assert "cargo check --release --manifest-path infotheory_py/Cargo.toml" in workflow
    assert "RUSTFLAGS: -C target-cpu=x86-64" in workflow
    assert "CC: clang" in workflow
    assert "CXX: clang++" in workflow
    assert "--interpreter python3" in workflow


def test_infotheory_py_does_not_enable_pyo3_auto_initialize_for_extension_builds():
    cargo_toml = (_repo_root() / "infotheory_py/Cargo.toml").read_text()
    assert 'features = ["abi3-py310"]' in cargo_toml
    assert "auto-initialize" not in cargo_toml


def test_pyproject_uses_python_release_profile_for_wheel_and_editable_builds():
    pyproject = (_repo_root() / "pyproject.toml").read_text()
    assert 'profile = "python-release"' in pyproject
    assert 'editable-profile = "python-release"' in pyproject


def test_zpaq_build_disables_cpp_lto_for_python_extension_builds():
    build_rs = (_repo_root() / "zpaq_rs" / "build.rs").read_text()
    assert "fn building_python_extension()" in build_rs
    assert 'env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some()' in build_rs
    assert "!building_python_extension()" in build_rs
