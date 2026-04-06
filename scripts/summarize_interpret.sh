#!/usr/bin/env bash
set -euo pipefail

find_latest_summary() {
  local root="/var/tmp/infotheory_bench"
  local latest=""

  [[ -d "${root}" ]] || return 1

  latest="$(
    find "${root}" -mindepth 2 -maxdepth 2 -type f -name summary.tsv 2>/dev/null \
      | awk -F/ '
          {
            stamp=$(NF-1)
            if (stamp ~ /^[0-9]{8}-[0-9]{6}$/) {
              print stamp "\t" $0
            }
          }
        ' \
      | sort -r \
      | head -n1 \
      | cut -f2-
  )"

  [[ -n "${latest}" ]] || return 1
  printf '%s\n' "${latest}"
}

summary="${1:-}"
if [[ -z "${summary}" ]]; then
  if ! summary="$(find_latest_summary)"; then
    echo "error: no summary.tsv found under /var/tmp/infotheory_bench" >&2
    exit 1
  fi
fi

if [[ ! -f "${summary}" ]]; then
  echo "usage: $0 [/path/to/summary.tsv]" >&2
  echo "error: file not found: ${summary}" >&2
  exit 1
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

awk -F'\t' '
function abs(x) { return x < 0 ? -x : x }

NR == 1 { next }

NF >= 6 {
  label = $1
  bmean = $2 + 0
  bstd  = ($3 == "" ? 0 : $3 + 0)
  cmean = $4 + 0
  cstd  = ($5 == "" ? 0 : $5 + 0)
  ratio = $6 + 0

  pct = (ratio - 1.0) * 100.0
  diff = cmean - bmean

  noise = bstd + cstd
  signal = (noise > 0 ? abs(diff) / noise : 999999)

  if (pct > 0.5) {
    dir = "slower"
  } else if (pct < -0.5) {
    dir = "faster"
  } else {
    dir = "flat"
  }

  apct = abs(pct)
  if (apct < 1.0) {
    mag = "tiny"
  } else if (apct < 3.0) {
    mag = "small"
  } else if (apct < 10.0) {
    mag = "moderate"
  } else {
    mag = "large"
  }

  if (signal < 1.0) {
    conf = "low"
  } else if (signal < 2.0) {
    conf = "medium"
  } else {
    conf = "high"
  }

  printf "%.6f\t%s\t%+.2f%%\t%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.9f\t%.3f\n",
         apct, label, pct, dir, mag, conf, bmean, cmean, bstd, cstd, signal
}
' "${summary}" | sort -t $'\t' -k1,1nr > "${tmp}"

total=$(wc -l < "${tmp}" | tr -d ' ')
slower=$(awk -F'\t' '$4=="slower"{c++} END{print c+0}' "${tmp}")
faster=$(awk -F'\t' '$4=="faster"{c++} END{print c+0}' "${tmp}")
flat=$(awk -F'\t' '$4=="flat"{c++} END{print c+0}' "${tmp}")

strong_reg=$(awk -F'\t' '$4=="slower" && ($5=="large" || $6=="high"){c++} END{print c+0}' "${tmp}")
strong_imp=$(awk -F'\t' '$4=="faster" && ($5=="large" || $6=="high"){c++} END{print c+0}' "${tmp}")

echo "Summary for: ${summary}"
echo
printf 'Cases: %d | Slower: %d | Faster: %d | Flat: %d\n' \
  "${total}" "${slower}" "${faster}" "${flat}"

if (( strong_reg > 0 && strong_imp > 0 )); then
  echo "Interpretation: mixed result; there are both strong regressions and strong improvements."
elif (( strong_reg > 0 )); then
  echo "Interpretation: regression-leaning; inspect the top slower cases first."
elif (( strong_imp > 0 )); then
  echo "Interpretation: improvement-leaning; no strong regression stands out."
else
  echo "Interpretation: mostly flat/noisy; nothing stands out strongly from this summary alone."
fi

echo
printf '%-24s %12s  %-7s %-9s %-9s %14s %14s %8s\n' \
  "label" "change" "dir" "size" "confidence" "baseline(s)" "current(s)" "signal"
echo "---------------------------------------------------------------------------------------------------------------"

awk -F'\t' '
{
  printf "%-24s %12s  %-7s %-9s %-9s %14.9f %14.9f %8.3f\n",
         $2, $3, $4, $5, $6, $7, $8, $11
}
' "${tmp}"

echo
echo "Legend:"
echo "  change     = (current / baseline - 1) * 100"
echo "  signal     = |current_mean - baseline_mean| / (baseline_stddev + current_stddev)"
echo "  confidence = heuristic only, not a formal statistical test"
echo "  size       = tiny <1%, small <3%, moderate <10%, large >=10%"
