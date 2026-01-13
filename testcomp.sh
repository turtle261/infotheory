#!/bin/sh
# POSIX sh compilation tester + deploy (uses host's default `cargo` only)

set -eu

KEY=${KEY:-"$HOME/.ssh/id_ed25519"}
ARCHIVE_BASE="it.tar"
ARCHIVE_GZ="${ARCHIVE_BASE}.gz"
REMOTE_DIR="infotheory"

HOSTS="192.168.122.177 192.168.122.2 192.168.122.63"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

cleanup() {
  if [ "${SSH_AGENT_PID:-}" ]; then
    ssh-agent -k >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT HUP TERM

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Start agent for this script only ---
eval "$(ssh-agent -s)" >/dev/null
ssh-add "$KEY"

# --- Build archive locally ---
# (no Makefile check anymore)

rm -f "$ARCHIVE_BASE" "$ARCHIVE_GZ"

echo "==> Creating archive: $ARCHIVE_GZ"
tar cf "$ARCHIVE_BASE" --exclude=.git --exclude=target . && gzip -f "$ARCHIVE_BASE"


# Remote script (POSIX sh). It MUST use the host's default `cargo`.
REMOTE_SH='
set -e

cd "$HOME/'"$REMOTE_DIR"'"

# clear directory contents safely
find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

# extract uploaded archive from home into this dir
tar xfz "../'"$ARCHIVE_GZ"'"

echo "==> Using cargo: $(command -v cargo)"
cargo build --release -q
'

overall_fail=0

for host in $HOSTS; do
  echo
  echo "=============================="
  echo "==> (*.177=Free, .63=Open, .2=Net BSD) Host: $host"
  echo "=============================="

  echo "-> Copying $ARCHIVE_GZ to $host:~/"
  if ! scp $SSH_OPTS "$ARCHIVE_GZ" "$host:~/"; then
    echo "!! SCP failed for $host" >&2
    overall_fail=1
    continue
  fi

  echo "-> Remote extract + build in ~/$REMOTE_DIR"
  # Feed the script via stdin; run it with /bin/sh explicitly for portability
  if ! ssh $SSH_OPTS "$host" /bin/sh -s <<EOF
$REMOTE_SH
EOF
  then
    echo "!! Build failed on $host" >&2
    overall_fail=1
    continue
  fi

  echo "-> OK: $host"
done

echo
echo "Done."
exit "$overall_fail"
