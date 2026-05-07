#!/bin/sh
set -eu

die() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  delegate_tuner_cgroup_v2.sh setup <tuner-user> [cgroup-name]
  delegate_tuner_cgroup_v2.sh run-in-session <tuner-user> [cgroup-name] -- <command> [args...]

Commands:
  setup
    Root-only. Create/delegate a strict-mode cgroup-v2 subtree with this layout:
      /sys/fs/cgroup/<cgroup-name>/
        session/   (for the long-lived tuner parent process)
        evals/     (pass this to --evaluator-cgroup-parent; tuner creates per-eval children here)

  run-in-session
    Root-only launcher helper. Runs an arbitrary command as <tuner-user>, places
    that process into /sys/fs/cgroup/<cgroup-name>/session, then resumes it.
    Use this when host delegation containment rules block unprivileged placement
    of the initial process into the delegated subtree.

Examples:
  sudo ./scripts/delegate_tuner_cgroup_v2.sh setup theo infotheory-tuner

  sudo ./scripts/delegate_tuner_cgroup_v2.sh run-in-session theo infotheory-tuner -- \
    env INFOTHEORY_TUNER_EVAL_CGROUP_PARENT=/sys/fs/cgroup/infotheory-tuner/evals \
    cargo run -p infotheory --features "tuner cli backend-ctw" -- \
      tune spec.json --rss-mode hybrid_strict_max --max-evaluations 1

Security model:
  - Root operations are limited to cgroup subtree setup/delegation and initial
    process placement into the delegated session cgroup.
  - Spec parsing/compression/scoring logic remains in the unprivileged
    <tuner-user> process.
EOF
}

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        die "this helper must be run as root"
    fi
}

validate_user() {
    user=$1
    if ! id "$user" >/dev/null 2>&1; then
        die "user '$user' does not exist"
    fi
}

validate_name() {
    name=$1
    case $name in
        */*|.*|*..*|*:*|"")
            die "cgroup-name must be a simple directory name"
            ;;
    esac
}

ensure_cgroup_v2_root() {
    root=$1
    [ -f "$root/cgroup.controllers" ] || die "$root is not a cgroup-v2 unified hierarchy"
    grep -qw memory "$root/cgroup.controllers" ||
        die "cgroup-v2 memory controller is not available in $root/cgroup.controllers"
}

enable_memory_subtree_control() {
    node=$1
    subtree="$node/cgroup.subtree_control"
    [ -f "$subtree" ] || die "missing $subtree"
    if ! grep -qw memory "$subtree"; then
        [ -w "$subtree" ] || die "cannot write $subtree to enable +memory"
        printf '+memory\n' > "$subtree" || die "failed to enable +memory in $subtree"
    fi
    grep -qw memory "$subtree" || die "memory controller is not enabled in $subtree"
}

delegate_node_to_user() {
    node=$1
    owner_group=$2
    chown "$owner_group" "$node" || die "failed to chown directory '$node' to $owner_group"
    chmod 0750 "$node" || die "failed to chmod 0750 '$node'"
    # Delegate only the files required for cgroup-v2 subtree management.
    for file in cgroup.procs cgroup.subtree_control cgroup.threads; do
        if [ -e "$node/$file" ]; then
            chown "$owner_group" "$node/$file" || die "failed to chown '$node/$file' to $owner_group"
        fi
    done
}

run_as_user() {
    user=$1
    shift
    if command -v runuser >/dev/null 2>&1; then
        runuser -u "$user" -- "$@"
    else
        su -s /bin/sh "$user" -c 'exec "$@"' sh "$@"
    fi
}

cmd_setup() {
    user=$1
    name=$2
    root=/sys/fs/cgroup
    base="$root/$name"
    session="$base/session"
    evals="$base/evals"

    require_root
    validate_user "$user"
    validate_name "$name"
    ensure_cgroup_v2_root "$root"

    group="$(id -gn "$user")" || die "failed to resolve primary group for user '$user'"
    owner_group="$user:$group"

    # Root operation: enable memory controller for children of /sys/fs/cgroup.
    enable_memory_subtree_control "$root"

    # Root operation: create delegated subtree layout.
    mkdir -p "$session" "$evals"

    # Root operation: enable memory controller where strict-mode evaluator child
    # cgroups will be created.
    enable_memory_subtree_control "$base"
    enable_memory_subtree_control "$evals"

    # Root operation: grant delegatee ownership only on delegation files + dirs.
    delegate_node_to_user "$base" "$owner_group"
    delegate_node_to_user "$session" "$owner_group"
    delegate_node_to_user "$evals" "$owner_group"

    cat <<EOF
Delegated Infotheory tuner cgroup subtree created:
  base:    $base
  session: $session
  evals:   $evals

Strict-mode evaluator parent path (pass to tune):
  --evaluator-cgroup-parent $evals

If unprivileged placement into delegated subtree is blocked by host delegation
containment rules, launch via:
  sudo $(basename "$0") run-in-session $user $name -- <command> [args...]
EOF
}

cmd_run_in_session() {
    user=$1
    name=$2
    shift 2
    [ "${1:-}" = "--" ] || die "run-in-session requires '-- <command> [args...]'"
    shift
    [ "$#" -gt 0 ] || die "run-in-session requires a command after '--'"

    require_root
    validate_user "$user"
    validate_name "$name"

    root=/sys/fs/cgroup
    base="$root/$name"
    session="$base/session"
    evals="$base/evals"
    [ -d "$base" ] || die "missing $base; run setup first"
    [ -d "$session" ] || die "missing $session; run setup first"
    [ -d "$evals" ] || die "missing $evals; run setup first"
    [ -w "$session/cgroup.procs" ] || die "session cgroup is not writable: $session/cgroup.procs"

    tmp_root=${TMPDIR:-/tmp}
    pid_file="$(mktemp "$tmp_root/infotheory-tuner-cgroup-pid.XXXXXX")"
    wrapper_file=""
    trap 'rm -f "$pid_file"; [ -n "$wrapper_file" ] && rm -f "$wrapper_file"' EXIT HUP INT TERM
    wrapper_file="$(mktemp "$tmp_root/infotheory-tuner-cgroup-wrapper.XXXXXX")"
    cat > "$wrapper_file" <<'WRAP'
#!/bin/sh
set -eu
pid_file=$1
shift
printf '%s\n' "$$" > "$pid_file"
kill -s STOP "$$"
exec "$@"
WRAP
    chmod 0700 "$wrapper_file"
    chown "$user:$(id -gn "$user")" "$wrapper_file" "$pid_file"

    # Root operation: start unprivileged command and place it into delegated
    # session cgroup before resuming. Invoke via /bin/sh so this works even on
    # hosts that mount /tmp with noexec.
    run_as_user "$user" /bin/sh "$wrapper_file" "$pid_file" "$@" &
    launcher_pid=$!

    i=0
    while [ ! -s "$pid_file" ]; do
        if ! kill -0 "$launcher_pid" 2>/dev/null; then
            wait "$launcher_pid" || true
            die "delegated command exited before PID capture (check command path/permissions)"
        fi
        i=$((i + 1))
        [ "$i" -le 200 ] || die "timed out waiting for delegated command PID capture"
        sleep 0.05
    done
    target_pid="$(cat "$pid_file")"
    case $target_pid in
        ""|*[!0-9]*)
            die "captured invalid target PID: '$target_pid'"
            ;;
    esac
    echo "$target_pid" > "$session/cgroup.procs" ||
        die "failed to place PID $target_pid into $session/cgroup.procs"
    kill -s CONT "$target_pid" || die "failed to resume PID $target_pid"
    wait "$launcher_pid"
}

main() {
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        usage
        exit 0
    fi

    subcommand=${1:-setup}
    case "$subcommand" in
        setup)
            shift || true
            user=${1:-}
            [ -n "$user" ] || {
                usage >&2
                exit 2
            }
            name=${2:-infotheory-tuner}
            cmd_setup "$user" "$name"
            ;;
        run-in-session)
            shift || true
            user=${1:-}
            [ -n "$user" ] || {
                usage >&2
                exit 2
            }
            shift || true
            name=infotheory-tuner
            if [ "${1:-}" != "--" ]; then
                name=${1:-}
                shift || true
            fi
            cmd_run_in_session "$user" "$name" "$@"
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
