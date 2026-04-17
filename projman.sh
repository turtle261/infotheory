#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

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
  if [ "$mode" = "portable" ]; then
    CARGO_BUILD_RUSTFLAGS="$(portable_rustflags)" \
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-$(portable_rustflags)}" \
    cargo "$@"
  else
    cargo "$@"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

has_kvm() {
  [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]
}

vm_artifacts_present() {
  [ -f "$ROOT_DIR/vmlinux-6.1.58" ] && [ -f "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage/rootfs.ext4" ] && [ -f "$ROOT_DIR/vendor/nyx-lite/guest/aixi_initramfs.cpio" ]
}

cmd_check_nyx_lite_standalone() {
  say "[check-vm-builder] Checking standalone nyx-lite build_rootfs compile..."
  need_cmd cargo
  (cd "$ROOT_DIR" && \
    cargo check -q --manifest-path "$ROOT_DIR/vendor/nyx-lite/Cargo.toml" --bin build_rootfs)
  say "[check-vm-builder] Done"
}

cmd_init_vm() {
  say "[init-vm] Fetching/building VM artifacts..."
  need_cmd cargo

  # Kernel (cached)
  if [ ! -f "$ROOT_DIR/vmlinux-6.1.58" ]; then
    if command -v wget >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && sh "$ROOT_DIR/vendor/nyx-lite/vm_image/download_kernel.sh")
    elif command -v curl >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && curl -L -o vmlinux-6.1.58 'https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.6/x86_64/vmlinux-6.1.58')
    else
      fail "Need wget or curl to download vmlinux-6.1.58"
    fi
  else
    say "[init-vm] Kernel already present: vmlinux-6.1.58"
  fi

  # Minimal initramfs (cpio) for nyx-lite guest
  need_cmd cc
  need_cmd cpio
  say "[init-vm] Building vendor/nyx-lite/guest/aixi_initramfs.cpio"
  (cd "$ROOT_DIR" && \
    cc -O2 -static -s vendor/nyx-lite/guest/aixi_guest.c -o vendor/nyx-lite/guest/aixi_guest && \
    mkdir -p vendor/nyx-lite/guest/initramfs && \
    cp -f vendor/nyx-lite/guest/aixi_guest vendor/nyx-lite/guest/initramfs/init && \
    (cd vendor/nyx-lite/guest/initramfs && find . -print | cpio -o -H newc > ../aixi_initramfs.cpio))

  # Docker rootfs build (ext4)
  if [ "${SKIP_DOCKER:-}" = "1" ]; then
    say "[init-vm] SKIP_DOCKER=1 set; skipping docker rootfs build"
    return 0
  fi

  # Prefer the library implementation (RootfsBuilder) which avoids sudo/mount and
  # directly produces an ext4 image via `mke2fs -d`.
  need_cmd docker
  need_cmd tar
  need_cmd mke2fs
  say "[init-vm] Building vendor/nyx-lite/vm_image/dockerimage/rootfs.ext4 via RootfsBuilder (no sudo)"
  (cd "$ROOT_DIR" && \
    cargo run -q --manifest-path "$ROOT_DIR/vendor/nyx-lite/Cargo.toml" --bin build_rootfs -- \
      "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage/Dockerfile" \
      "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage" \
      "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage/rootfs.ext4" \
      --size-mib 512 \
      --work-dir "$ROOT_DIR/target/tmp/rootfs_work")

  say "[init-vm] Done"
}

cmd_code_test() {
  say "[code_test] Building + testing Rust (release)..."
  need_cmd cargo
  validate_build_mode
  say "[code_test] Build mode: $(build_mode)"

  (cd "$ROOT_DIR" && run_cargo_mode build --release)
  if [ "${BUILD_CLI:-0}" = "1" ]; then
    say "[code_test] BUILD_CLI=1 set; checking optional CLI binary"
    (cd "$ROOT_DIR" && run_cargo_mode build --release --features cli)
  fi

  # If docker is available, enable the nyx-lite rootfs builder test.
  DOCKER_TEST=0
  if command -v docker >/dev/null 2>&1 && command -v tar >/dev/null 2>&1 && command -v mke2fs >/dev/null 2>&1; then
    DOCKER_TEST=1
  fi

  if vm_artifacts_present && has_kvm; then
    say "[code_test] VM artifacts present and /dev/kvm accessible; running with --features vm"
    if [ "$DOCKER_TEST" -eq 1 ]; then
      (cd "$ROOT_DIR" && NYX_TEST_DOCKER=1 run_cargo_mode test --release --features vm)
    else
      (cd "$ROOT_DIR" && run_cargo_mode test --release --features vm)
    fi
  else
    if vm_artifacts_present; then
      say "[code_test] VM artifacts present but /dev/kvm not accessible; running without vm feature"
    else
      say "[code_test] VM artifacts not initialized; running without vm feature"
    fi
    if [ "$DOCKER_TEST" -eq 1 ]; then
      (cd "$ROOT_DIR" && NYX_TEST_DOCKER=1 run_cargo_mode test --release)
    else
      (cd "$ROOT_DIR" && run_cargo_mode test --release)
    fi
  fi

  say "[code_test] Done"
}

cmd_lean_test() {
  say "[lean_test] Building + running Lean validation (ite-bench)..."
  need_cmd lake

  (cd "$ROOT_DIR/ite-bench" && lake build)
  (cd "$ROOT_DIR/ite-bench" && lake exe runner)

  say "[lean_test] Done"
}

cmd_test_full() {
  cmd_check_nyx_lite_standalone
  cmd_init_vm
  cmd_code_test
  cmd_lean_test
}

cmd_test_all() {
  cmd_test_full
}

cmd_bench() {
  if [ "${1:-}" = "cli" ]; then
    shift
    cmd_bench_cli "$@"
    return 0
  fi

  suite=${INFOTHEORY_BENCH_SUITE:-two-json}
  case "${1:-}" in
    two-json|two_json|two|core|full)
      suite=two-json
      shift
      ;;
    extra)
      suite=extra
      shift
      ;;
  esac
  case "${suite}" in
    extra) suite_display="configs/bench/extra.json" ;;
    *) suite=two-json; suite_display="configs/bench/two.json" ;;
  esac
  say "[bench] Running ${suite_display} benchmark suite..."
  need_cmd sh
  (cd "$ROOT_DIR" && INFOTHEORY_BENCH_SUITE="$suite" sh "$ROOT_DIR/scripts/bench_two_json.sh" "$@")
  say "[bench] Done"
}

cmd_bench_cli() {
  [ $# -ge 1 ] || fail "Usage: ./projman.sh bench cli <baseline-commit> [preset]"
  need_cmd bash
  validate_build_mode
  case "${1:-}" in
    -h|--help)
      (cd "$ROOT_DIR" && bash "$ROOT_DIR/scripts/bench_cli_hyperfine.sh" "$@")
      return 0
      ;;
  esac
  say "[bench_cli] Running hyperfine CLI comparison against baseline '$1' (build mode: $(build_mode))..."
  (cd "$ROOT_DIR" && bash "$ROOT_DIR/scripts/bench_cli_hyperfine.sh" "$@" && "$ROOT_DIR/scripts/summarize_interpret.sh")
  say "[bench_cli] Done"
}

cmd_bench_aixi_competitors() {
  say "[bench_aixi_competitors] Running reproducible Guix benchmark (Infotheory Rust/Python vs PyAIXI vs C++ MC-AIXI)..."
  need_cmd guix
  need_cmd bash
  (cd "$ROOT_DIR" && bash "$ROOT_DIR/scripts/bench_aixi_competitors_guix.sh" "$@")
  say "[bench_aixi_competitors] Done"
}

cmd_plot() {
  suite=${INFOTHEORY_PLOT_SUITE:-${INFOTHEORY_BENCH_SUITE:-two-json}}
  case "${1:-}" in
    two-json|two_json|two|core|full)
      suite=two-json
      shift
      ;;
    extra)
      suite=extra
      shift
      ;;
  esac
  case "${suite}" in
    extra) suite_display="configs/bench/extra.json" ;;
    *) suite=two-json; suite_display="configs/bench/two.json" ;;
  esac
  say "[plot] Legacy plot generation is superseded by the benchman TUI."
  say "[plot] Use './projman.sh tui ${suite}' to inspect ${suite_display} benchmarks."
  cmd_tui "$suite" "$@"
}

cmd_tui_man() {
  need_cmd man
  need_cmd nvim
  export MANPAGER='nvim +Man!'
  man -l "$ROOT_DIR/docs/benchman.1"
}

cmd_tui() {
  if [ "${1:-}" = "man" ]; then
    shift
    cmd_tui_man "$@"
    return 0
  fi

  if [ "${1:-}" = "log-loss" ]; then
    shift
    [ $# -ge 1 ] || fail "Usage: ./projman.sh tui log-loss <prefix>"
    prefix=$1
    shift

    say "[tui] Building benchman (release)..."
    need_cmd cargo
    (cd "$ROOT_DIR" && CARGO_INCREMENTAL=0 cargo build --release --locked -p benchman)

    say "[tui] Launching benchman log-loss mode..."
    (cd "$ROOT_DIR" && "$ROOT_DIR/target/release/benchman" log-loss "$prefix" "$@")
    say "[tui] Done"
    return 0
  fi

  suite=${INFOTHEORY_PLOT_SUITE:-${INFOTHEORY_BENCH_SUITE:-two-json}}
  case "${1:-}" in
    two-json|two_json|two|core|full)
      suite=two-json
      shift
      ;;
    extra)
      suite=extra
      shift
      ;;
  esac

  say "[tui] Building benchman (release)..."
  need_cmd cargo
  (cd "$ROOT_DIR" && CARGO_INCREMENTAL=0 cargo build --release --locked -p benchman)

  say "[tui] Launching benchman..."
  (cd "$ROOT_DIR" && INFOTHEORY_PLOT_SUITE="$suite" INFOTHEORY_BENCH_SUITE="$suite" "$ROOT_DIR/target/release/benchman" --suite "$suite" "$@")
  say "[tui] Done"
}

cmd_clean() {
  say "[clean] Cleaning build artifacts (keeps kernel)..."
  need_cmd cargo

  (cd "$ROOT_DIR" && cargo clean)

  # Lean build outputs
  if command -v lake >/dev/null 2>&1; then
    (cd "$ROOT_DIR/ite-bench" && lake clean || true)
  fi
  rm -rf "$ROOT_DIR/ite-bench/.lake/build" || true

  # nyx-lite guest artifacts
  rm -f "$ROOT_DIR/vendor/nyx-lite/guest/aixi_guest" || true
  rm -rf "$ROOT_DIR/vendor/nyx-lite/guest/initramfs" || true
  rm -f "$ROOT_DIR/vendor/nyx-lite/guest/aixi_initramfs.cpio" || true

  # docker rootfs artifact (rebuildable)
  rm -f "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage/rootfs.ext4" || true
  rm -rf "$ROOT_DIR/vendor/nyx-lite/vm_image/dockerimage/mnt" || true

  say "[clean] Done"
}

cmd_legacy_aixi_convert() {
  [ $# -eq 1 ] || fail "Usage: ./projman.sh legacy_aixi_convert <input_file>"

  lua_cmd=""
  if command -v luajit >/dev/null 2>&1; then
    lua_cmd="luajit"
  elif command -v lua >/dev/null 2>&1; then
    lua_cmd="lua"
  else
    fail "Missing required command: luajit or lua"
  fi

  (cd "$ROOT_DIR" && "$lua_cmd" "$ROOT_DIR/scripts/legacy_aixi_convert.lua" "$1")
}

usage() {
  cat <<'EOF'
Usage: ./projman.sh <command>

Commands:
  bench [suite]  Run benchmark suite (`two-json` default, or `extra`). Requires /tmp/enwik7 to exist and be exactly 10000000 bytes. Resumes the newest raw TSV for the selected suite by default; set INFOTHEORY_BENCH_FRESH=1 for a new run. Not included in test_all.
  bench cli <baseline-commit> [preset]  Build baseline vs dirty current trees and compare CLI workloads with hyperfine. Presets: `default` (signal-focused defaults) and `quick` (same matrix with lighter defaults). Writes artifacts under /var/tmp/infotheory_bench/.
  bench_aixi_competitors  Run reproducible Guix time-machine benchmark for Infotheory MC-AIXI (Rust+Python) vs PyAIXI and C++ MC-AIXI. Fails fast if Guix is unavailable.
  plot [suite]   Open benchmark results in the benchman TUI for the selected suite (`two-json` default, or `extra`). Not included in test_all.
  tui [suite]    Build and launch the interactive benchmark TUI (`benchman`) for the selected suite (`two-json` default, or `extra`). Supports --summary-tsv/--baseline-summary-tsv/--raw-tsv/--subjects and manages /tmp/plotimgs.
  tui log-loss <prefix>  Build and launch the log-loss diagnostic TUI for <prefix>.trace.tsv / .nodes.tsv / .summary.tsv.
  tui man     Open the local benchman manual via nvim man pager (MANPAGER='nvim +Man!').
  code_test   Build (release) and run Rust tests (release). Uses --features vm iff VM artifacts exist and /dev/kvm is accessible.
  init-vm     Download/build VM artifacts needed for VM tests (kernel, initramfs, docker rootfs).
  lean_test   Run Lean validation suite (ite-bench). Requires lake.
  test_full   Run init-vm, code_test, and lean_test.
  test_all    Alias for test_full.
  clean       Clean build artifacts (cargo clean, lake clean, VM images/initramfs). Keeps vmlinux-6.1.58.
  legacy_aixi_convert <input_file>  Convert legacy AIXI JSON config to canonical planner_run JSON and write to stdout. External configs print: External configs were deprecated.

Environment variables:
  INFOTHEORY_BUILD_MODE=native|portable  Controls local cargo invocations in projman. `native` uses the repository's default target-cpu=native configuration; `portable` overrides local builds/tests to use generic CPU codegen like CI/release builds.
  INFOTHEORY_BENCH_*  Passed through to scripts/bench_two_json.sh for benchmark tuning/output paths, including INFOTHEORY_BENCH_SUBJECTS=rwkv, INFOTHEORY_BENCH_SUITE=extra, and INFOTHEORY_BENCH_BUILD_MODE=native|portable.
  INFOTHEORY_CLI_BENCH_*  Passed through to scripts/bench_cli_hyperfine.sh for baseline/current CLI benchmark tuning and input selection, including INFOTHEORY_CLI_BENCH_BUILD_MODE=native|portable.
  INFOTHEORY_PLOT_*   Passed through to scripts/plot_two_json.sh, including INFOTHEORY_PLOT_SUBJECTS=rwkv, INFOTHEORY_PLOT_SUMMARY_TSV=..., and INFOTHEORY_PLOT_SUITE=extra.
  INFOTHEORY_BASELINE_SUMMARY_TSV / INFOTHEORY_BENCH_RAW_TSV  Also read by benchman for baseline overlays and raw inspector detail.
  SKIP_DOCKER=1   Skip docker rootfs.ext4 build during init-vm.
  BUILD_CLI=1     Also build optional infotheory CLI binary (feature: cli) during code_test.
EOF
}

cmd=${1:-}
case "$cmd" in
  bench) shift; cmd_bench "$@" ;;
  bench_aixi_competitors) shift; cmd_bench_aixi_competitors "$@" ;;
  plot) shift; cmd_plot "$@" ;;
  tui) shift; cmd_tui "$@" ;;
  code_test) shift; cmd_code_test "$@" ;;
  init-vm) shift; cmd_init_vm "$@" ;;
  lean_test) shift; cmd_lean_test "$@" ;;
  test_full) shift; cmd_test_full "$@" ;;
  test_all) shift; cmd_test_all "$@" ;;
  clean) shift; cmd_clean "$@" ;;
  legacy_aixi_convert) shift; cmd_legacy_aixi_convert "$@" ;;
  -h|--help|help|'') usage ;;
  *) usage; fail "Unknown command: $cmd" ;;
esac
