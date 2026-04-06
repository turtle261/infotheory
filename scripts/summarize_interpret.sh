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

python3 - "${summary}" <<'PY'
import csv
import math
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])


def parse_float(raw):
    if raw is None:
        return math.nan
    text = str(raw).strip()
    if not text:
        return math.nan
    lower = text.lower()
    if lower in {"inf", "+inf"}:
        return math.inf
    if lower == "-inf":
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_int(raw):
    value = parse_float(raw)
    if math.isfinite(value):
        return int(value)
    return None


def fmt(value, digits=9):
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    if value > 0:
        return "inf"
    if value < 0:
        return "-inf"
    return "nan"


rows = []
with summary_path.open("r", encoding="utf-8") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    if reader.fieldnames is None:
        print(f"Summary for: {summary_path}")
        print()
        print("No tabular header detected in summary.tsv.")
        raise SystemExit(0)
    for row in reader:
        if not any((value or "").strip() for value in row.values()):
            continue
        rows.append(row)

records = []
for row in rows:
    label = (row.get("label") or "").strip()
    if not label:
        continue

    bmean = parse_float(row.get("baseline_mean_s"))
    cmean = parse_float(row.get("current_mean_s"))
    bstd = parse_float(row.get("baseline_stddev_s"))
    cstd = parse_float(row.get("current_stddev_s"))
    ratio = parse_float(row.get("ratio_current_over_baseline"))

    if not math.isfinite(bstd):
        bstd = 0.0
    if not math.isfinite(cstd):
        cstd = 0.0

    if not math.isfinite(ratio):
        if bmean == 0:
            ratio = math.inf
        else:
            ratio = cmean / bmean

    b_n = parse_int(row.get("baseline_n"))
    c_n = parse_int(row.get("current_n"))

    bsem = parse_float(row.get("baseline_sem_s"))
    csem = parse_float(row.get("current_sem_s"))
    if not math.isfinite(bsem):
        if b_n is not None and b_n > 0:
            bsem = bstd / math.sqrt(b_n)
        else:
            bsem = bstd
    if not math.isfinite(csem):
        if c_n is not None and c_n > 0:
            csem = cstd / math.sqrt(c_n)
        else:
            csem = cstd

    delta = parse_float(row.get("delta_s"))
    if not math.isfinite(delta):
        delta = cmean - bmean

    se_delta = parse_float(row.get("se_delta_s"))
    if not math.isfinite(se_delta):
        se_delta = math.sqrt((bsem * bsem) + (csem * csem))

    t_like = parse_float(row.get("t_like"))
    if not math.isfinite(t_like):
        if se_delta == 0:
            t_like = math.inf if delta != 0 else 0.0
        else:
            t_like = abs(delta) / se_delta

    ci_low = parse_float(row.get("ci95_ratio_low"))
    ci_high = parse_float(row.get("ci95_ratio_high"))
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        if (
            bmean > 0
            and cmean > 0
            and math.isfinite(bsem)
            and math.isfinite(csem)
        ):
            log_ratio = math.log(cmean / bmean)
            se_log_ratio = math.sqrt((bsem / bmean) ** 2 + (csem / cmean) ** 2)
            ci_low = math.exp(log_ratio - 1.96 * se_log_ratio)
            ci_high = math.exp(log_ratio + 1.96 * se_log_ratio)
        else:
            ci_low = ratio
            ci_high = ratio

    residual_bits = parse_float(row.get("residual_bits_gaussian"))

    pct = (ratio - 1.0) * 100.0
    if pct > 0.5:
        direction = "slower"
    elif pct < -0.5:
        direction = "faster"
    else:
        direction = "flat"

    apct = abs(pct)
    if apct < 1.0:
        size = "tiny"
    elif apct < 3.0:
        size = "small"
    elif apct < 10.0:
        size = "moderate"
    else:
        size = "large"

    if t_like < 1.0:
        confidence = "low"
    elif t_like < 2.0:
        confidence = "medium"
    else:
        confidence = "high"

    ci_text = "n/a"
    ci_support = False
    if math.isfinite(ci_low) and math.isfinite(ci_high):
        lo_pct = (ci_low - 1.0) * 100.0
        hi_pct = (ci_high - 1.0) * 100.0
        ci_text = f"[{lo_pct:+.2f}%, {hi_pct:+.2f}%]"
        if direction == "slower" and ci_low > 1.0:
            ci_support = True
        if direction == "faster" and ci_high < 1.0:
            ci_support = True

    records.append(
        {
            "label": label,
            "pct": pct,
            "change_text": f"{pct:+.2f}%",
            "dir": direction,
            "size": size,
            "confidence": confidence,
            "bmean": bmean,
            "cmean": cmean,
            "se_delta": se_delta,
            "ci_text": ci_text,
            "residual_bits": residual_bits,
            "ci_support": ci_support,
            "sort_key": apct,
        }
    )

records.sort(key=lambda item: float(item["sort_key"]), reverse=True)

total = len(records)
slower = sum(1 for row in records if row["dir"] == "slower")
faster = sum(1 for row in records if row["dir"] == "faster")
flat = sum(1 for row in records if row["dir"] == "flat")

strong_reg = sum(
    1
    for row in records
    if row["dir"] == "slower"
    and (row["size"] == "large" or row["confidence"] == "high" or row["ci_support"])
)
strong_imp = sum(
    1
    for row in records
    if row["dir"] == "faster"
    and (row["size"] == "large" or row["confidence"] == "high" or row["ci_support"])
)

roundtrip_path = summary_path.with_name("roundtrip.tsv")
roundtrip_total = 0
roundtrip_pass = 0
roundtrip_fail = 0
if roundtrip_path.is_file():
    with roundtrip_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if not status:
                continue
            roundtrip_total += 1
            if status == "pass":
                roundtrip_pass += 1
            else:
                roundtrip_fail += 1

print(f"Summary for: {summary_path}")
print()
print(f"Cases: {total} | Slower: {slower} | Faster: {faster} | Flat: {flat}")
if roundtrip_total > 0:
    print(f"Roundtrip checks: {roundtrip_pass} passed / {roundtrip_total} total")

if roundtrip_fail > 0:
    print("Interpretation: invalid benchmark run; roundtrip verification reported failures.")
elif strong_reg > 0 and strong_imp > 0:
    print("Interpretation: mixed result; there are both strong regressions and strong improvements.")
elif strong_reg > 0:
    print("Interpretation: regression-leaning; inspect the top slower cases first.")
elif strong_imp > 0:
    print("Interpretation: improvement-leaning; no strong regression stands out.")
else:
    print("Interpretation: mostly flat/noisy; nothing stands out strongly from this summary alone.")

print()
print(
    f"{'label':<24} {'change':>12}  {'dir':<7} {'size':<9} {'confidence':<9} "
    f"{'baseline(s)':>14} {'current(s)':>14} {'se_delta(s)':>12} {'95% CI(change)':>20} {'resid_bits':>10}"
)
print("-" * 170)

for row in records:
    se_delta_text = fmt(float(row["se_delta"]), 9)
    resid_bits = float(row["residual_bits"])
    resid_text = fmt(resid_bits, 3) if math.isfinite(resid_bits) else "n/a"
    print(
        f"{str(row['label']):<24} {str(row['change_text']):>12}  {str(row['dir']):<7} "
        f"{str(row['size']):<9} {str(row['confidence']):<9} "
        f"{fmt(float(row['bmean']), 9):>14} {fmt(float(row['cmean']), 9):>14} "
        f"{se_delta_text:>12} {str(row['ci_text']):>20} {resid_text:>10}"
    )

print()
print("Legend:")
print("  change          = (current / baseline - 1) * 100")
print("  confidence      = |current_mean - baseline_mean| / se_delta")
print("  se_delta        = sqrt(baseline_sem^2 + current_sem^2)")
print("  95% CI(change)  = delta-method interval from log(current/baseline)")
print("  resid_bits      = 0.5 * log2(2*pi*e*pooled_residual_var_s2)")
print("  size            = tiny <1%, small <3%, moderate <10%, large >=10%")
PY
