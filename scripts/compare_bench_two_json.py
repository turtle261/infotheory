#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (
    REPO_ROOT / "benchmarks" / "baselines" / "infotheory-two-json-summary-20260310-212017.tsv"
)
CORE_OPERATIONS = {"h", "compress", "decompress"}
CORE_SUBJECTS = {"ppmd", "ctw", "rosa", "rwkv", "neural_mixture"}
CORE_SIZES = {"1048576", "4194304", "10000000"}


def latest_candidate_summary() -> Path | None:
    candidates = sorted(
        Path("/tmp").glob("infotheory-two-json-summary-*.tsv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a bench_two_json summary TSV against the checked-in baseline and "
            "hard-fail only on the core performance matrix."
        )
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        type=Path,
        help="Candidate summary TSV. Defaults to the newest /tmp/infotheory-two-json-summary-*.tsv.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline summary TSV. Defaults to {DEFAULT_BASELINE}.",
    )
    return parser.parse_args()


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["operation"],
        row["subject"],
        row["size_bytes"],
        row["compression_backend"],
    )


def load_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = row_key(row)
            if key in rows:
                raise SystemExit(f"duplicate summary row in {path}: {key!r}")
            rows[key] = row
    return rows


def parse_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "").strip()
    if not value:
        return None
    return float(value)


def parse_int(row: dict[str, str], field: str) -> int | None:
    value = row.get(field, "").strip()
    if not value:
        return None
    return int(float(value))


def is_core_key(key: tuple[str, str, str, str]) -> bool:
    operation, subject, size_bytes, _compression_backend = key
    return (
        operation in CORE_OPERATIONS
        and subject in CORE_SUBJECTS
        and size_bytes in CORE_SIZES
    )


def compare_rows(
    baseline: dict[str, str],
    candidate: dict[str, str],
) -> list[str]:
    reasons: list[str] = []

    if candidate.get("verified_all") != "1":
        reasons.append("verified_all != 1")

    baseline_real = parse_float(baseline, "real_seconds_median")
    candidate_real = parse_float(candidate, "real_seconds_median")
    if baseline_real is not None and candidate_real is not None:
        real_limit = max(baseline_real * 1.05, baseline_real + 0.02)
        if candidate_real > real_limit:
            reasons.append(
                f"real_seconds_median {candidate_real:.6g} > {real_limit:.6g}"
            )

    baseline_rss = parse_float(baseline, "rss_kib_median")
    candidate_rss = parse_float(candidate, "rss_kib_median")
    if baseline_rss is not None and candidate_rss is not None:
        rss_limit = max(baseline_rss * 1.03, baseline_rss + 4096.0)
        if candidate_rss > rss_limit:
            reasons.append(f"rss_kib_median {candidate_rss:.6g} > {rss_limit:.6g}")

    baseline_archive = parse_int(baseline, "archive_bytes_median")
    candidate_archive = parse_int(candidate, "archive_bytes_median")
    if baseline_archive is not None and candidate_archive is not None:
        if candidate_archive > baseline_archive + 1:
            reasons.append(
                f"archive_bytes_median {candidate_archive} > {baseline_archive + 1}"
            )

    baseline_entropy = parse_float(baseline, "entropy_bpb_median")
    candidate_entropy = parse_float(candidate, "entropy_bpb_median")
    if baseline_entropy is not None and candidate_entropy is not None:
        if candidate_entropy > baseline_entropy + 1e-9:
            reasons.append(
                f"entropy_bpb_median {candidate_entropy:.12g} > {baseline_entropy + 1e-9:.12g}"
            )

    return reasons


def sort_key(key: tuple[str, str, str, str]) -> tuple[int, str, int, str]:
    op_order = {"h": 0, "compress": 1, "decompress": 2}
    operation, subject, size_bytes, compression_backend = key
    return (
        op_order.get(operation, 99),
        subject,
        int(size_bytes),
        compression_backend,
    )


def main() -> int:
    args = parse_args()
    candidate_path = args.candidate or latest_candidate_summary()
    baseline_path = args.baseline

    if candidate_path is None:
        raise SystemExit("no candidate summary TSV provided and no /tmp summary TSV was found")
    if not baseline_path.is_file():
        raise SystemExit(f"baseline summary TSV not found: {baseline_path}")
    if not candidate_path.is_file():
        raise SystemExit(f"candidate summary TSV not found: {candidate_path}")

    baseline_rows = load_rows(baseline_path)
    candidate_rows = load_rows(candidate_path)
    keys = sorted(set(baseline_rows) | set(candidate_rows), key=sort_key)

    full_issues = 0
    core_failures = 0

    print(f"baseline\t{baseline_path}")
    print(f"candidate\t{candidate_path}")
    print(
        "scope\tstatus\toperation\tsubject\tsize_bytes\tcompression_backend\treasons"
    )
    for key in keys:
        scope = "core" if is_core_key(key) else "full"
        baseline = baseline_rows.get(key)
        candidate = candidate_rows.get(key)
        reasons: list[str]
        if baseline is None:
            reasons = ["missing baseline row"]
        elif candidate is None:
            reasons = ["missing candidate row"]
        else:
            reasons = compare_rows(baseline, candidate)

        if not reasons:
            status = "OK"
        elif scope == "core":
            status = "FAIL"
            core_failures += 1
        else:
            status = "WARN"
            full_issues += 1

        print(
            "\t".join(
                [
                    scope,
                    status,
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    "; ".join(reasons) if reasons else "-",
                ]
            )
        )

    print(
        f"summary\tcore_failures={core_failures}\tfull_warnings={full_issues}\trows={len(keys)}"
    )
    return 1 if core_failures else 0


if __name__ == "__main__":
    sys.exit(main())
