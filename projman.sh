#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

has_kvm() {
  [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]
}

vm_artifacts_present() {
  [ -f "$ROOT_DIR/vmlinux-6.1.58" ] && [ -f "$ROOT_DIR/nyx-lite/vm_image/dockerimage/rootfs.ext4" ] && [ -f "$ROOT_DIR/nyx-lite/guest/aixi_initramfs.cpio" ]
}

cmd_init_vm() {
  say "[init-vm] Fetching/building VM artifacts..."
  need_cmd cargo

  # Kernel (cached)
  if [ ! -f "$ROOT_DIR/vmlinux-6.1.58" ]; then
    if command -v wget >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && sh "$ROOT_DIR/nyx-lite/vm_image/download_kernel.sh")
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
  say "[init-vm] Building nyx-lite/guest/aixi_initramfs.cpio"
  (cd "$ROOT_DIR" && \
    cc -O2 -static -s nyx-lite/guest/aixi_guest.c -o nyx-lite/guest/aixi_guest && \
    mkdir -p nyx-lite/guest/initramfs && \
    cp -f nyx-lite/guest/aixi_guest nyx-lite/guest/initramfs/init && \
    (cd nyx-lite/guest/initramfs && find . -print | cpio -o -H newc > ../aixi_initramfs.cpio))

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
  say "[init-vm] Building nyx-lite/vm_image/dockerimage/rootfs.ext4 via RootfsBuilder (no sudo)"
  (cd "$ROOT_DIR" && \
    cargo run -q -p nyx-lite --bin build_rootfs -- \
      "$ROOT_DIR/nyx-lite/vm_image/dockerimage/Dockerfile" \
      "$ROOT_DIR/nyx-lite/vm_image/dockerimage" \
      "$ROOT_DIR/nyx-lite/vm_image/dockerimage/rootfs.ext4" \
      --size-mib 512 \
      --work-dir "$ROOT_DIR/target/tmp/rootfs_work")

  say "[init-vm] Done"
}

cmd_code_test() {
  say "[code_test] Building + testing Rust (release)..."
  need_cmd cargo

  (cd "$ROOT_DIR" && cargo build --release)

  # If docker is available, enable the nyx-lite rootfs builder test.
  DOCKER_TEST=0
  if command -v docker >/dev/null 2>&1 && command -v tar >/dev/null 2>&1 && command -v mke2fs >/dev/null 2>&1; then
    DOCKER_TEST=1
  fi

  if vm_artifacts_present && has_kvm; then
    say "[code_test] VM artifacts present and /dev/kvm accessible; running with --features vm"
    if [ "$DOCKER_TEST" -eq 1 ]; then
      (cd "$ROOT_DIR" && NYX_TEST_DOCKER=1 cargo test --release --features vm)
    else
      (cd "$ROOT_DIR" && cargo test --release --features vm)
    fi
  else
    if vm_artifacts_present; then
      say "[code_test] VM artifacts present but /dev/kvm not accessible; running without vm feature"
    else
      say "[code_test] VM artifacts not initialized; running without vm feature"
    fi
    if [ "$DOCKER_TEST" -eq 1 ]; then
      (cd "$ROOT_DIR" && NYX_TEST_DOCKER=1 cargo test --release)
    else
      (cd "$ROOT_DIR" && cargo test --release)
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
  cmd_init_vm
  cmd_code_test
  cmd_lean_test
}

cmd_test_all() {
  cmd_test_full
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
  rm -f "$ROOT_DIR/nyx-lite/guest/aixi_guest" || true
  rm -rf "$ROOT_DIR/nyx-lite/guest/initramfs" || true
  rm -f "$ROOT_DIR/nyx-lite/guest/aixi_initramfs.cpio" || true

  # docker rootfs artifact (rebuildable)
  rm -f "$ROOT_DIR/nyx-lite/vm_image/dockerimage/rootfs.ext4" || true
  rm -rf "$ROOT_DIR/nyx-lite/vm_image/dockerimage/mnt" || true

  say "[clean] Done"
}

usage() {
  cat <<EOF
Usage: ./projman.sh <command>

Commands:
  code_test   Build (release) and run Rust tests (release). Uses --features vm iff VM artifacts exist and /dev/kvm is accessible.
  init-vm     Download/build VM artifacts needed for VM tests (kernel, initramfs, docker rootfs).
  lean_test   Run Lean validation suite (ite-bench). Requires lake.
  test_full   Run init-vm, code_test, and lean_test.
  test_all    Alias for test_full.
  clean       Clean build artifacts (cargo clean, lake clean, VM images/initramfs). Keeps vmlinux-6.1.58.

Environment variables:
  SKIP_DOCKER=1   Skip docker rootfs.ext4 build during init-vm.
EOF
}

cmd=${1:-}
case "$cmd" in
  code_test) shift; cmd_code_test "$@" ;;
  init-vm) shift; cmd_init_vm "$@" ;;
  lean_test) shift; cmd_lean_test "$@" ;;
  test_full) shift; cmd_test_full "$@" ;;
  test_all) shift; cmd_test_all "$@" ;;
  clean) shift; cmd_clean "$@" ;;
  -h|--help|help|'') usage ;;
  *) usage; fail "Unknown command: $cmd" ;;
esac
