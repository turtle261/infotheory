import json
import functools
import os
import pathlib
import shutil
import subprocess

import pytest


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _resolve_bash_executable() -> str:
    if os.name != "nt":
        return "bash"

    candidates: list[str] = []
    for env_var in ("ProgramW6432", "ProgramFiles"):
        root = os.environ.get(env_var)
        if root:
            candidates.append(str(pathlib.Path(root) / "Git" / "bin" / "bash.exe"))
            candidates.append(str(pathlib.Path(root) / "Git" / "usr" / "bin" / "bash.exe"))

    which_bash = shutil.which("bash")
    if which_bash:
        candidates.append(which_bash)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(pathlib.Path(candidate))
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            probe = subprocess.run(
                [normalized, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            continue
        if probe.returncode == 0 and "GNU bash" in probe.stdout:
            return normalized

    pytest.skip("GNU bash executable is required on Windows for benchmark script tests")


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if cmd and cmd[0] == "bash":
        cmd = [_resolve_bash_executable(), *cmd[1:]]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    )


def _parse_plan(output: str) -> tuple[dict[str, str], list[str]]:
    plan: dict[str, str] = {}
    cases: list[str] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("PLAN"):
            fields = raw_line.split("\t")
            if len(fields) >= 3:
                _, key, value = fields[:3]
                plan[key.strip()] = value.strip()
                continue
            _, key, value = line.split(None, 2)
            plan[key.strip()] = value.strip()
            continue

        if line.startswith("CASE"):
            fields = raw_line.split("\t")
            if len(fields) >= 2:
                _, label = fields[:2]
                cases.append(label.strip())
                continue
            _, label = line.split(None, 1)
            cases.append(label.strip())

    return plan, cases


def _write_compare_summary(
    path: pathlib.Path,
    *,
    suite_spec_path: str,
    suite_spec_sha256: str,
    build_mode: str = "native",
    build_features: str = "cli",
) -> None:
    path.write_text(
        "\n".join(
            [
                "\t".join(
                    [
                        "operation",
                        "subject",
                        "size_bytes",
                        "compression_backend",
                        "suite_spec_path",
                        "suite_spec_sha256",
                        "build_mode",
                        "build_features",
                    ]
                ),
                "\t".join(
                    [
                        "h",
                        "neural_mixture",
                        "2097152",
                        "-",
                        suite_spec_path,
                        suite_spec_sha256,
                        build_mode,
                        build_features,
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_bench_plan_default_matrix_and_defaults():
    proc = _run(["bash", "scripts/bench_cli_hyperfine.sh", "--plan", "default"])
    assert proc.returncode == 0, proc.stderr

    plan, cases = _parse_plan(proc.stdout)

    expected_rate_backends = {
        "rosaplus",
        "ctw",
        "fac-ctw",
        "match",
        "sparse-match",
        "ppmd",
        "sequitur",
        "calibrated",
        "mixture",
        "particle",
        "mamba",
        "rwkv7",
    }

    assert plan["preset"] == "default"
    assert plan["runs"] == "10"
    assert plan["warmups"] == "3"
    assert plan["bytes"] == "32768"
    assert set(plan["rate_backends"].split(",")) == expected_rate_backends

    assert int(plan["cases"]) == 40
    assert int(plan["roundtrip_cases"]) == 14
    assert len(cases) == 40
    assert len(set(cases)) == 40

    assert sum(name.startswith("h_") for name in cases) == 12
    assert sum(name.startswith("compress_rate_ac_") for name in cases) == 12
    assert sum(name.startswith("decompress_rate_ac_") for name in cases) == 12
    assert sum(name.startswith("compress_rate_rans_") for name in cases) == 2
    assert sum(name.startswith("decompress_rate_rans_") for name in cases) == 2


def test_cli_bench_plan_quick_keeps_full_matrix_with_faster_defaults():
    proc = _run(["bash", "scripts/bench_cli_hyperfine.sh", "--plan", "quick"])
    assert proc.returncode == 0, proc.stderr

    plan, cases = _parse_plan(proc.stdout)

    assert plan["preset"] == "quick"
    assert plan["runs"] == "5"
    assert plan["warmups"] == "1"
    assert plan["bytes"] == "16384"
    assert int(plan["cases"]) == 40
    assert int(plan["roundtrip_cases"]) == 14
    assert len(cases) == 40


def test_projman_cli_plan_mode_skips_summary_lookup_error():
    proc = _run(
        ["bash", "projman.sh", "bench", "cli", "--plan", "quick"],
        env={"INFOTHEORY_BUILD_MODE": "portable"},
    )
    assert proc.returncode == 0, proc.stderr

    plan, _ = _parse_plan(proc.stdout)
    assert plan["preset"] == "quick"
    assert "no summary.tsv found under /var/tmp/infotheory_bench" not in proc.stderr


def test_projman_cli_plan_uses_canonical_build_mode_knob():
    proc = _run(
        ["bash", "projman.sh", "bench", "cli", "--plan", "quick"],
        env={
            "INFOTHEORY_BUILD_MODE": "portable",
            "INFOTHEORY_CLI_BENCH_BUILD_MODE": "invalid",
        },
    )
    assert proc.returncode == 0, proc.stderr

    plan, _ = _parse_plan(proc.stdout)
    assert plan["preset"] == "quick"


def test_summarize_interpret_supports_extended_summary_schema(tmp_path: pathlib.Path):
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "\n".join(
            [
                "label\tbaseline_mean_s\tbaseline_stddev_s\tcurrent_mean_s\tcurrent_stddev_s\tratio_current_over_baseline\tbaseline_n\tcurrent_n\tbaseline_sem_s\tcurrent_sem_s\tdelta_s\tse_delta_s\tt_like\tci95_ratio_low\tci95_ratio_high\tpooled_residual_var_s2\tresidual_bits_gaussian",
                "h_match\t0.010000000\t0.001000000\t0.011000000\t0.001200000\t1.100000\t20\t20\t0.000223607\t0.000268328\t0.001000000\t0.000349602\t2.860000\t1.030000\t1.170000\t0.000001220\t-5.891000",
                "h_ppmd\t0.020000000\t0.002000000\t0.019000000\t0.001700000\t0.950000\t20\t20\t0.000447214\t0.000380789\t-0.001000000\t0.000587724\t1.701000\t0.900000\t1.010000\t0.000003500\t-4.730000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    roundtrip = tmp_path / "roundtrip.tsv"
    roundtrip.write_text(
        "\n".join(
            [
                "label\tsubject\tstatus",
                "rate_ac_match\tbaseline\tpass",
                "rate_ac_match\tcurrent\tpass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run(["bash", "scripts/summarize_interpret.sh", str(summary)])
    assert proc.returncode == 0, proc.stderr

    out = proc.stdout
    assert "Roundtrip checks: 2 passed / 2 total" in out
    assert "se_delta(s)" in out
    assert "95% CI(change)" in out
    assert "h_match" in out


def test_summarize_interpret_supports_legacy_summary_schema(tmp_path: pathlib.Path):
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "\n".join(
            [
                "label\tbaseline_mean_s\tbaseline_stddev_s\tcurrent_mean_s\tcurrent_stddev_s\tratio_current_over_baseline",
                "h_ctw\t0.050000000\t0.001000000\t0.051000000\t0.001100000\t1.020000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run(["bash", "scripts/summarize_interpret.sh", str(summary)])
    assert proc.returncode == 0, proc.stderr

    out = proc.stdout
    assert "Cases: 1" in out
    assert "h_ctw" in out
    assert "Legend:" in out


def test_benchmark_two_json_specs_are_kept_in_sync():
    repo = _repo_root()
    configs_text = (repo / "configs/bench/two.json").read_text(encoding="utf-8")
    examples_text = (repo / "examples/two.json").read_text(encoding="utf-8")
    assert configs_text == examples_text
    assert json.loads(configs_text)["alpha"] == 0.03


def test_compare_bench_two_json_accepts_matching_suite_spec_digest(
    tmp_path: pathlib.Path,
):
    if shutil.which("luajit") is None:
        pytest.skip("luajit not installed")

    baseline = tmp_path / "baseline.tsv"
    candidate = tmp_path / "candidate.tsv"
    suite_spec_sha256 = "c" * 64
    _write_compare_summary(
        baseline,
        suite_spec_path="examples/two.json",
        suite_spec_sha256=suite_spec_sha256,
    )
    _write_compare_summary(
        candidate,
        suite_spec_path="configs/bench/two.json",
        suite_spec_sha256=suite_spec_sha256,
    )

    proc = _run(
        [
            "luajit",
            "scripts/compare_bench_two_json.lua",
            "--baseline",
            str(baseline),
            str(candidate),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "baseline_suite_spec_sha256" in proc.stdout
    assert "candidate_suite_spec_sha256" in proc.stdout


def test_compare_bench_two_json_rejects_mismatched_suite_spec_digest(
    tmp_path: pathlib.Path,
):
    if shutil.which("luajit") is None:
        pytest.skip("luajit not installed")

    baseline = tmp_path / "baseline.tsv"
    candidate = tmp_path / "candidate.tsv"
    _write_compare_summary(
        baseline,
        suite_spec_path="examples/two.json",
        suite_spec_sha256="a" * 64,
    )
    _write_compare_summary(
        candidate,
        suite_spec_path="configs/bench/two.json",
        suite_spec_sha256="b" * 64,
    )

    proc = _run(
        [
            "luajit",
            "scripts/compare_bench_two_json.lua",
            "--baseline",
            str(baseline),
            str(candidate),
        ]
    )
    assert proc.returncode != 0
    assert "suite spec digest mismatch" in proc.stderr


def test_compare_bench_two_json_explains_duplicate_summary_keys(
    tmp_path: pathlib.Path,
):
    if shutil.which("luajit") is None:
        pytest.skip("luajit not installed")

    baseline = tmp_path / "baseline.tsv"
    candidate = tmp_path / "candidate.tsv"
    baseline.write_text(
        "\n".join(
            [
                "\t".join(
                    [
                        "operation",
                        "subject",
                        "size_bytes",
                        "cpu",
                        "compression_backend",
                        "suite_spec_path",
                        "suite_spec_sha256",
                        "build_mode",
                        "build_features",
                    ]
                ),
                "\t".join(
                    [
                        "h",
                        "ctw",
                        "4096",
                        "0",
                        "-",
                        "configs/bench/two.json",
                        "a" * 64,
                        "native",
                        "cli",
                    ]
                ),
                "\t".join(
                    [
                        "h",
                        "ctw",
                        "4096",
                        "11",
                        "-",
                        "configs/bench/two.json",
                        "a" * 64,
                        "native",
                        "cli",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_compare_summary(
        candidate,
        suite_spec_path="configs/bench/two.json",
        suite_spec_sha256="a" * 64,
    )

    proc = _run(
        [
            "luajit",
            "scripts/compare_bench_two_json.lua",
            "--baseline",
            str(baseline),
            str(candidate),
        ]
    )
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert "duplicate comparison row" in proc.stderr
    assert "operation=h, subject=ctw, size_bytes=4096, compression_backend=-" in proc.stderr
    assert "differing columns: cpu: 0 != 11" in proc.stderr
    assert "mix CPU affinities" in proc.stderr


def test_bench_two_json_build_mode_namespace_is_bench_scoped():
    script_text = (_repo_root() / "scripts/bench_two_json.sh").read_text(encoding="utf-8")

    assert "INFOTHEORY_CLI_BENCH_BUILD_MODE" not in script_text
    assert "INFOTHEORY_BENCH_BUILD_MODE" in script_text
    assert "CARGO_BUILD_RUSTFLAGS" in script_text


def test_bench_two_json_compare_hint_uses_current_baseline_resolver():
    script_text = (_repo_root() / "scripts/bench_two_json.sh").read_text(encoding="utf-8")

    assert "current_two_json_baseline_tsv()" in script_text
    assert 'benchmarks/current/infotheory-two-json-summary"*.tsv' in script_text
    assert "--baseline '${CURRENT_BASELINE_TSV}'" in script_text
    assert "infotheory-two-json-summary-20260322-120428.tsv" not in script_text
