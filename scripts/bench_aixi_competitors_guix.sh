#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

guix_bin="guix"
guix_channels=""
out_dir=""
rayon_threads="1"
trials="1"
seed="0"
coin_flip_p="0.9"
profile="default"

pyaixi_url="https://github.com/sgkasselau/pyaixi"
pyaixi_commit="39afd4c"
cpp_url="https://github.com/moridinamael/mc-aixi.git"
cpp_commit="b9e2cd4"

usage() {
  cat <<'EOF'
Usage: scripts/bench_aixi_competitors_guix.sh [options]

Deterministic Guix time-machine benchmark of MC-AIXI implementations:
- Infotheory (Rust CLI)
- Infotheory (Python bindings)
- PyAIXI (sgkasselau/pyaixi @ 39afd4c)
- C++ MC-AIXI (moridinamael/mc-aixi @ b9e2cd4)

Outputs plot-ready TSV files under target/aixi-competitors/<timestamp>/.

Options:
  --out-dir <path>            Output directory (default: target/aixi-competitors/<timestamp>)
  --guix-bin <path>           guix executable (default: guix)
  --guix-channels <path>      Channels file for guix time-machine (default: lock from guix describe)
  --rayon-threads <n>         Fixed Rayon threads for infotheory runs (default: 1)
  --trials <n>                Trials per scenario (default: 1)
  --seed <n>                  Deterministic random seed across implementations (default: 0)
  --coin-flip-p <p>           CoinFlip probability for competitor envs (default: 0.9)
  --profile <default|quick>   Scenario profile (default: default)
  --pyaixi-url <url>          Override PyAIXI repository URL
  --pyaixi-commit <sha>       Override PyAIXI commit
  --cpp-url <url>             Override C++ MC-AIXI repository URL
  --cpp-commit <sha>          Override C++ MC-AIXI commit
  --help                      Show help

Notes:
- This script fails fast if Guix is unavailable.
- The benchmark executes inside a `guix time-machine` container sandbox for reproducible tooling.
- A channels lock used for the run is written to <out-dir>/channels.lock.scm.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --guix-bin)
      guix_bin="$2"
      shift 2
      ;;
    --guix-channels)
      guix_channels="$2"
      shift 2
      ;;
    --rayon-threads)
      rayon_threads="$2"
      shift 2
      ;;
    --trials)
      trials="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --coin-flip-p)
      coin_flip_p="$2"
      shift 2
      ;;
    --profile)
      profile="$2"
      shift 2
      ;;
    --pyaixi-url)
      pyaixi_url="$2"
      shift 2
      ;;
    --pyaixi-commit)
      pyaixi_commit="$2"
      shift 2
      ;;
    --cpp-url)
      cpp_url="$2"
      shift 2
      ;;
    --cpp-commit)
      cpp_commit="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$guix_bin" >/dev/null 2>&1; then
  echo "ERROR: Guix is required but not found (expected command: $guix_bin)." >&2
  exit 1
fi

if [[ "$profile" != "default" && "$profile" != "quick" ]]; then
  echo "ERROR: --profile must be one of: default, quick" >&2
  exit 1
fi

if [[ -z "$out_dir" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  out_dir="$repo_root/target/aixi-competitors/$stamp"
fi

mkdir -p "$out_dir"

if [[ -z "$guix_channels" ]]; then
  guix_channels="$out_dir/channels.lock.scm"
  "$guix_bin" describe --format=channels > "$guix_channels"
else
  cp "$guix_channels" "$out_dir/channels.lock.scm"
  guix_channels="$out_dir/channels.lock.scm"
fi

inner_script="$out_dir/run_inside_guix.sh"
cat > "$inner_script" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

repo_root="$1"
out_dir="$2"
pyaixi_url="$3"
pyaixi_commit="$4"
cpp_url="$5"
cpp_commit="$6"
rayon_threads="$7"
trials="$8"
profile="$9"
coin_flip_p="${10}"
seed="${11}"

export CARGO_TARGET_DIR="$out_dir/cargo-target"

clone_checkout() {
  local url="$1"
  local commit="$2"
  local dst="$3"

  if [[ ! -d "$dst/.git" ]]; then
    git clone "$url" "$dst"
  fi

  (
    cd "$dst"
    git fetch --all --tags --prune
    git checkout "$commit"
  )
}

sources_dir="$out_dir/sources"
mkdir -p "$sources_dir"

pyaixi_root="$sources_dir/pyaixi"
cpp_root="$sources_dir/mc-aixi-cpp"

clone_checkout "$pyaixi_url" "$pyaixi_commit" "$pyaixi_root"
clone_checkout "$cpp_url" "$cpp_commit" "$cpp_root"

# In pure Guix shells, ensure Rust build scripts can find libgcc_s at runtime.
gcc_lib_dir="$(dirname "$(gcc -print-file-name=libgcc_s.so.1)")"
export LD_LIBRARY_PATH="$gcc_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

(
  cd "$cpp_root"
  make -j"$(nproc)"
)

(
  cd "$repo_root"
  cargo build --release --no-default-features --features cli --bin infotheory
)

bench_venv="$out_dir/venv"
bench_python="$bench_venv/bin/python"

PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" -m venv "$bench_venv"
"$bench_python" -m pip install --upgrade pip setuptools wheel
"$bench_python" -m pip install "six>=1.16,<2"
"$bench_python" -m pip install "maturin==1.8.7"

unset PYTHONHOME
unset PYTHONPATH
export VIRTUAL_ENV="$bench_venv"
export PATH="$bench_venv/bin:$PATH"
export PYO3_PYTHON="$bench_python"

(
  cd "$repo_root/crates/infotheory_py"
  "$bench_python" -m maturin develop --release --no-default-features --features "python-extension" --pip-path "$bench_venv/bin/pip"
)

TIME_BIN="$(command -v time)"
INFOTHEORY_BIN="$CARGO_TARGET_DIR/release/infotheory"

"$bench_python" "$repo_root/scripts/bench_aixi_competitors_runner.py" \
  --repo-root "$repo_root" \
  --out-dir "$out_dir" \
  --bench-python "$bench_python" \
  --infotheory-bin "$INFOTHEORY_BIN" \
  --pyaixi-root "$pyaixi_root" \
  --cpp-root "$cpp_root" \
  --time-bin "$TIME_BIN" \
  --rayon-threads "$rayon_threads" \
  --trials "$trials" \
  --seed "$seed" \
  --coin-flip-p "$coin_flip_p" \
  --profile "$profile"
INNER
chmod +x "$inner_script"

echo "[bench__aixi_competitors] Running inside Guix time-machine..."
"$guix_bin" time-machine -C "$guix_channels" -- shell --pure --container --network --no-cwd \
  --share="$repo_root=$repo_root" \
  bash coreutils findutils grep sed gawk git make gcc-toolchain \
  clang-toolchain lld \
  python python-pip python-setuptools rust pkg-config time nss-certs \
  -- bash "$inner_script" \
    "$repo_root" \
    "$out_dir" \
    "$pyaixi_url" \
    "$pyaixi_commit" \
    "$cpp_url" \
    "$cpp_commit" \
    "$rayon_threads" \
    "$trials" \
    "$profile" \
    "$coin_flip_p" \
    "$seed"

echo "Benchmark complete."
echo "Output directory: $out_dir"
echo "Channels lock: $out_dir/channels.lock.scm"
echo "Raw TSV: $out_dir/raw.tsv"
echo "Summary TSV: $out_dir/summary.tsv"
echo "Report: $out_dir/report.txt"
