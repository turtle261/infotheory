#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
SIMDIR=".sim"
mkdir -p "$SIMDIR"

seed_file="$SIMDIR/seed"
t_file="$SIMDIR/t"
fault_file="$SIMDIR/fault"
health_file="$SIMDIR/health"
power_file="$SIMDIR/power"
fixed_file="$SIMDIR/fixed"

die() { echo "ERR $*" >&2; exit 2; }

# ---------- deterministic RNG (xorshift32) ----------
rand_u32() {
  [[ -f "$seed_file" ]] || die "seed missing (init not run?)"
  local x
  x="$(cat "$seed_file")"
  # make sure it's an integer
  [[ "$x" =~ ^[0-9]+$ ]] || die "bad seed: $x"

  # xorshift32
  x=$(( (x ^ (x << 13)) & 0xFFFFFFFF ))
  x=$(( (x ^ (x >> 17)) & 0xFFFFFFFF ))
  x=$(( (x ^ (x << 5))  & 0xFFFFFFFF ))

  echo "$x" > "$seed_file"
  echo "$x"
}

rng_mod() {
  local m="${1:-}"
  [[ "$m" =~ ^[0-9]+$ ]] || die "rng_mod needs integer modulus, got: ${m:-<empty>}"
  (( m > 0 )) || die "rng_mod modulus must be > 0"
  local r
  r="$(rand_u32)"
  echo $(( r % m ))
}

init_if_needed() {
  if [[ ! -f "$seed_file" ]]; then
    # Seed from time + PID
    echo $(( ( $(date +%s) ^ $$ ) & 0xFFFFFFFF )) > "$seed_file"
  fi

  if [[ ! -f "$t_file" ]]; then
    echo 0 > "$t_file"

    # hidden fault in {A,B,C}
    case "$(rng_mod 3)" in
      0) echo "A" > "$fault_file" ;;
      1) echo "B" > "$fault_file" ;;
      2) echo "C" > "$fault_file" ;;
      *) die "unreachable fault selection" ;;
    esac

    echo 0  > "$fixed_file"
    echo 7000 > "$health_file"
    echo 50 > "$power_file"
  fi
}

clamp() {
  local v="$1" lo="$2" hi="$3"
  if (( v < lo )); then v="$lo"; fi
  if (( v > hi )); then v="$hi"; fi
  echo "$v"
}

read_int() {
  local path="$1" name="$2"
  [[ -f "$path" ]] || die "missing $name file: $path"
  local v
  v="$(cat "$path")"
  [[ "$v" =~ ^-?[0-9]+$ ]] || die "bad $name value: $v"
  echo "$v"
}

tick_dynamics() {
  local t health power fixed decay noise
  t="$(read_int "$t_file" t)"
  health="$(read_int "$health_file" health)"
  power="$(read_int "$power_file" power)"
  fixed="$(cat "$fixed_file" 2>/dev/null || true)"
  [[ "$fixed" == "0" || "$fixed" == "1" ]] || die "bad fixed value: ${fixed:-<empty>}"

  # Base decay: worse when not fixed; reduced by higher power
  decay=4
  if [[ "$fixed" == "0" ]]; then decay=$((decay + 3)); fi
  decay=$((decay - (power / 25)))  # power 0..100 => subtract 0..4
  if (( decay < 1 )); then decay=1; fi

  # Stochastic noise: 0..3
  noise="$(rng_mod 4)"
  decay=$((decay + noise))

  health=$((health - decay))
  health="$(clamp "$health" 0 1000)"

  # power naturally drifts down a bit: -2..-4
  power=$((power - 2 - $(rng_mod 3)))
  power="$(clamp "$power" 0 100)"

  t=$((t + 1))

  echo "$t" > "$t_file"
  echo "$health" > "$health_file"
  echo "$power" > "$power_file"
}

obs_scan() {
  local fault health power hint_noise hint health_est
  fault="$(cat "$fault_file")"
  health="$(read_int "$health_file" health)"
  power="$(read_int "$power_file" power)"

  # noisy hint: correct 2/3 of the time
  hint_noise="$(rng_mod 3)"
  if [[ "$hint_noise" == "0" ]]; then
    hint="$fault"
  else
    # wrong hint: pick one of the other two
    case "$fault" in
      A) hint=$([[ "$(rng_mod 2)" == "0" ]] && echo "B" || echo "C") ;;
      B) hint=$([[ "$(rng_mod 2)" == "0" ]] && echo "A" || echo "C") ;;
      C) hint=$([[ "$(rng_mod 2)" == "0" ]] && echo "A" || echo "B") ;;
      *) die "bad fault value: $fault" ;;
    esac
  fi

  # coarse health estimate (bucketed + noisy)
  health_est=$(( (health / 10) * 10 ))
  health_est=$((health_est + ( $(rng_mod 3) - 1 ) * 10 )) # -10/0/+10
  health_est="$(clamp "$health_est" 0 100)"

  echo "OBS scan t=$(cat "$t_file") hint=${hint} health~=${health_est} power=${power}"
}

do_reroute_power() {
  local power surge health
  power="$(read_int "$power_file" power)"
  power=$((power + 18))
  power="$(clamp "$power" 0 100)"
  echo "$power" > "$power_file"

  # 1/4 chance of surge damages health
  surge="$(rng_mod 4)"
  if [[ "$surge" == "0" ]]; then
    health="$(read_int "$health_file" health)"
    health=$((health - 8))
    health="$(clamp "$health" 0 1000)"
    echo "$health" > "$health_file"
    echo "EVENT power_surge=-8"
  else
    echo "EVENT power_ok"
  fi
}

do_repair() {
  local target="$1"
  local fault fixed health

  fault="$(cat "$fault_file")"
  fixed="$(cat "$fixed_file")"
  health="$(read_int "$health_file" health)"

  if [[ "$fixed" == "1" ]]; then
    echo "EVENT already_fixed"
    return
  fi

  if [[ "$target" == "$fault" ]]; then
    echo 1 > "$fixed_file"
    health=$((health + 1000))
    health="$(clamp "$health" 0 1000)"
    echo "$health" > "$health_file"
    echo "EVENT repair_${target}=SUCCESS"
  else
    # wrong repair damages health a lot
    health=$((health - 2))
    health="$(clamp "$health" 0 1000)"
    echo "$health" > "$health_file"
    echo "EVENT repair_${target}=FAIL"
  fi
}

do_commit() {
  local fixed health
  fixed="$(cat "$fixed_file")"
  health="$(read_int "$health_file" health)"

  if [[ "$fixed" == "1" && "$health" -ge 60 ]]; then
    echo "MISSION_SUCCESS fixed=1 health=${health}"
  else
    echo "MISSION_FAIL fixed=${fixed} health=${health}"
  fi
}

# ---------- main ----------
init_if_needed

case "$ACTION" in
  scan)
    tick_dynamics
    obs_scan
    ;;
  reroute_power)
    tick_dynamics
    do_reroute_power
    echo "OBS power t=$(cat "$t_file") health=$(cat "$health_file") power=$(cat "$power_file")"
    ;;
  repair_A|repair_B|repair_C)
    tick_dynamics
    do_repair "${ACTION#repair_}"
    echo "OBS repair t=$(cat "$t_file") health=$(cat "$health_file") power=$(cat "$power_file") fixed=$(cat "$fixed_file")"
    ;;
  commit)
    tick_dynamics
    do_commit
    echo "OBS commit t=$(cat "$t_file") health=$(cat "$health_file") power=$(cat "$power_file") fixed=$(cat "$fixed_file")"
    ;;
  *)
    die "unknown_action=$ACTION"
    ;;
esac

