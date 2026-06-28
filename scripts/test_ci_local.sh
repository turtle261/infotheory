#!/bin/sh
set -eu

export LC_ALL=C

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"; }

build_mode() {
  printf '%s' "${INFOTHEORY_BUILD_MODE:-native}"
}

validate_build_mode() {
  case "$(build_mode)" in
    native|portable) ;;
    *) fail "INFOTHEORY_BUILD_MODE must be one of: native, portable" ;;
  esac
}

portable_rustflags() {
  case "$(uname -s)" in
    Linux|FreeBSD|OpenBSD) printf '%s' "-C target-cpu=generic -C link-arg=-fuse-ld=lld" ;;
    *) printf '%s' "-C target-cpu=generic" ;;
  esac
}

run_cargo_mode() {
  validate_build_mode
  mode=$(build_mode)
  ci_jobs=${CARGO_BUILD_JOBS:-4}
  ci_dev_lto=${CARGO_PROFILE_DEV_LTO:-false}
  ci_test_lto=${CARGO_PROFILE_TEST_LTO:-false}
  ci_dev_cgu=${CARGO_PROFILE_DEV_CODEGEN_UNITS:-16}
  ci_test_cgu=${CARGO_PROFILE_TEST_CODEGEN_UNITS:-16}
  if [ "$mode" = "portable" ]; then
    mode_flags=$(portable_rustflags)
    CARGO_BUILD_JOBS="$ci_jobs" \
    CARGO_PROFILE_DEV_LTO="$ci_dev_lto" \
    CARGO_PROFILE_TEST_LTO="$ci_test_lto" \
    CARGO_PROFILE_DEV_CODEGEN_UNITS="$ci_dev_cgu" \
    CARGO_PROFILE_TEST_CODEGEN_UNITS="$ci_test_cgu" \
    CARGO_BUILD_RUSTFLAGS="$mode_flags" \
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-$mode_flags}" \
    cargo "$@"
  else
    CARGO_BUILD_JOBS="$ci_jobs" \
    CARGO_PROFILE_DEV_LTO="$ci_dev_lto" \
    CARGO_PROFILE_TEST_LTO="$ci_test_lto" \
    CARGO_PROFILE_DEV_CODEGEN_UNITS="$ci_dev_cgu" \
    CARGO_PROFILE_TEST_CODEGEN_UNITS="$ci_test_cgu" \
    cargo "$@"
  fi
}

cargo_check_warn() {
  (
    cd "$ROOT_DIR" && \
    RUSTFLAGS="-D warnings" \
    run_cargo_mode check "$@"
  )
}

ci_rustdoc_tmp=${TMPDIR:-/tmp}/infotheory-rustdoc-cov.$$.json
cleanup() {
  rm -f "$ci_rustdoc_tmp" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

cmd_rust_line_coverage() {
  say "[test_ci] Rust line coverage gate (>= ${INFOTHEORY_CI_FAIL_UNDER_LINES:-85}%)..."
  if ! cargo llvm-cov --version >/dev/null 2>&1; then
    fail "cargo-llvm-cov is required. Install it with: cargo install cargo-llvm-cov --locked"
  fi
  (
    cd "$ROOT_DIR" && \
    run_cargo_mode llvm-cov -p infotheory --tests --features "cli all-backends" --locked --summary-only --fail-under-lines "${INFOTHEORY_CI_FAIL_UNDER_LINES:-85}"
  )
}

cmd_rustdoc_coverage() {
  say "[test_ci] Rustdoc coverage gate (must remain 100%)..."
  if ! cargo +nightly --version >/dev/null 2>&1; then
    fail "nightly toolchain is required for rustdoc coverage. Install it with: rustup toolchain install nightly"
  fi
  (
    cd "$ROOT_DIR" && \
    run_cargo_mode +nightly rustdoc -p infotheory --all-features -- -Z unstable-options --show-coverage --output-format json > "$ci_rustdoc_tmp"
  )
  python3 - "$ci_rustdoc_tmp" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
files = data.get("files", data)
total = sum(v.get("total", 0) for v in files.values())
with_docs = sum(v.get("with_docs", 0) for v in files.values())
pct = 100.0 if total == 0 else (with_docs * 100.0 / total)
print(f"Rustdoc documented items: {with_docs}/{total} ({pct:.2f}%)")
if with_docs != total:
    print("Rustdoc coverage gate failed (<100.0%).", file=sys.stderr)
    sys.exit(1)
PY
}

cmd_feature_gates() {
  say "[test_ci] Feature-gate compile matrix (curated fast subset)..."

  cargo_check_warn -p zpaq_rs --locked
  cargo_check_warn -p benchman --locked
  cargo_check_warn --manifest-path vendor/gameengine/Cargo.toml --features builtin --locked

  cargo_check_warn -p infotheory --locked
  cargo_check_warn -p infotheory --no-default-features --locked
  cargo_check_warn -p infotheory --no-default-features --features backend-ctw --locked
  cargo_check_warn -p infotheory --no-default-features --features all-backends --locked
  cargo_check_warn -p infotheory --no-default-features --features aixi-gameengine --locked
  cargo_check_warn -p infotheory --no-default-features --features "tuner backend-ctw" --locked
  cargo_check_warn -p infotheory --features cli --locked
  cargo_check_warn -p infotheory --no-default-features --features cli --locked
  cargo_check_warn -p infotheory --no-default-features --features "cli all-backends" --locked
  (
    cd "$ROOT_DIR" && \
    PYO3_BUILD_EXTENSION_MODULE=1 \
    RUSTFLAGS="-D warnings" \
    run_cargo_mode check -p infotheory_py --no-default-features --features "tuner backend-ctw" --locked
  )

  say "[test_ci] Test-harness compile slices..."
  (
    cd "$ROOT_DIR" && \
    run_cargo_mode test -p infotheory --no-run --locked && \
    run_cargo_mode test -p infotheory --no-default-features --features "cli all-backends" --no-run --locked && \
    run_cargo_mode test -p infotheory --no-default-features --features "tuner backend-ctw" --no-run --locked
  )

  if [ "${INFOTHEORY_CI_INCLUDE_VM:-0}" = "1" ]; then
    say "[test_ci] VM compile slices enabled (INFOTHEORY_CI_INCLUDE_VM=1)..."
    (
      cd "$ROOT_DIR" && \
      run_cargo_mode test -p infotheory --no-default-features --features vm --no-run --locked && \
      run_cargo_mode test -p infotheory --no-default-features --features "vm backend-ctw" --no-run --locked
    )
  else
    say "[test_ci] Skipping VM compile slices (set INFOTHEORY_CI_INCLUDE_VM=1 to include)."
  fi
}

cmd_python_gates() {
  say "[test_ci] Python extension + coverage gate..."
  need_cmd uv

  venv_dir=${INFOTHEORY_CI_PYTHON_VENV:-"$ROOT_DIR/.venv"}
  venv_py=$venv_dir/bin/python

  if [ ! -x "$venv_py" ]; then
    (cd "$ROOT_DIR" && uv venv "$venv_dir")
  fi

  (cd "$ROOT_DIR" && uv pip install --python "$venv_py" "maturin>=1.7,<2" "pytest>=8.0" "pytest-cov>=7.0.0")

  (
    cd "$ROOT_DIR" && \
    VIRTUAL_ENV="$venv_dir" \
    PATH="$venv_dir/bin:$PATH" \
    "$venv_py" -m maturin develop --profile python-release --manifest-path crates/infotheory_py/Cargo.toml
  )

  (
    cd "$ROOT_DIR" && \
    "$venv_py" -m pytest --cov=infotheory_rs --cov-report=term-missing --cov-report=xml:target/python-coverage.xml --cov-fail-under=100 python/tests
  )

  say "[test_ci] Python aixi-gameengine smoke gate..."
  (
    cd "$ROOT_DIR" && \
    VIRTUAL_ENV="$venv_dir" \
    PATH="$venv_dir/bin:$PATH" \
    "$venv_py" -m maturin develop --profile python-release --manifest-path crates/infotheory_py/Cargo.toml --features python-extension,all-backends,aixi-gameengine && \
    "$venv_py" -m pytest -q python/tests/test_aixi_gameengine.py
  )

  if [ "${INFOTHEORY_CI_INCLUDE_VM:-0}" = "1" ]; then
    say "[test_ci] Python VM smoke gate enabled (INFOTHEORY_CI_INCLUDE_VM=1)..."
    (
      cd "$ROOT_DIR" && \
      VIRTUAL_ENV="$venv_dir" \
      PATH="$venv_dir/bin:$PATH" \
      "$venv_py" -m maturin develop --profile python-release --manifest-path crates/infotheory_py/Cargo.toml --features python-extension,all-backends,vm && \
      "$venv_py" -m pytest -q python/tests/test_vm.py
    )
  else
    say "[test_ci] Skipping Python VM smoke (set INFOTHEORY_CI_INCLUDE_VM=1 to include)."
  fi
}

cmd_main() {
  say "[test_ci] Running local CI preflight (fast comprehensive gates)..."
  need_cmd cargo
  need_cmd python3
  validate_build_mode
  say "[test_ci] Build mode: $(build_mode)"

  if [ "${INFOTHEORY_CI_SKIP_RUST_LINE_COVERAGE:-0}" = "1" ]; then
    say "[test_ci] Skipping Rust line coverage gate (INFOTHEORY_CI_SKIP_RUST_LINE_COVERAGE=1)."
  else
    cmd_rust_line_coverage
  fi

  if [ "${INFOTHEORY_CI_SKIP_RUSTDOC_COVERAGE:-0}" = "1" ]; then
    say "[test_ci] Skipping rustdoc coverage gate (INFOTHEORY_CI_SKIP_RUSTDOC_COVERAGE=1)."
  else
    cmd_rustdoc_coverage
  fi

  if [ "${INFOTHEORY_CI_SKIP_FEATURE_GATES:-0}" = "1" ]; then
    say "[test_ci] Skipping feature-gate compile matrix (INFOTHEORY_CI_SKIP_FEATURE_GATES=1)."
  else
    cmd_feature_gates
  fi

  if [ "${INFOTHEORY_CI_SKIP_PYTHON:-0}" = "1" ]; then
    say "[test_ci] Skipping Python coverage/smoke gates (INFOTHEORY_CI_SKIP_PYTHON=1)."
  else
    cmd_python_gates
  fi

  say "[test_ci] All local CI preflight gates passed."
}

cmd_main "$@"
