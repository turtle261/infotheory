#![cfg(feature = "cli")]

//! CLI help-topic coverage.
//!
//! Every user-facing command name and alias accepted by `main.rs`'s dispatch
//! arm must route through the universal `--help` trigger and print the
//! corresponding topic help page (or, for topics that are aliases of one
//! canonical page, that canonical page). This file enumerates the full set
//! of accepted command names once and asserts each one behaves that way.
//!
//! Keeping the list local to the test makes the gap immediately visible at
//! CI time: if a new alias is added to `main.rs` without a corresponding
//! help-trigger entry, the binary will instead attempt to execute the
//! command without real file arguments -- producing a file-not-found error,
//! garbled output, or a stdin stall -- all of which fail these assertions.

use std::process::{Command, Output, Stdio};

const HELP_USAGE_MARKER: &str = "Usage:";
const HELP_BINARY_MARKER: &str = "infotheory";

const ALL_CLI_COMMANDS: &[&str] = &[
    "help",
    "batch",
    "tune",
    "ctw-profile",
    "ctw_profile",
    "warmstart",
    "ac-log-loss",
    "ac_log_loss",
    "sequitur-debug",
    "sequitur_debug",
    "aixi",
    "search",
    "compress",
    "decompress",
    "generate",
    "ncd",
    "ncd_vitanyi",
    "ncd_sym",
    "ncd_sym_vitanyi",
    "ncd_cons",
    "ncd_sym_cons",
    "entropy",
    "h",
    "entropy_rate",
    "h_rate",
    "entropy_bits",
    "h_bits",
    "entropy_rate_per_bit",
    "h_rate_per_bit",
    "biased_entropy_rate_per_bit",
    "id",
    "id_bits",
    "intrinsic_dependence_bits",
    "ned",
    "ned_bits",
    "ned_cons",
    "ned_cons_bits",
    "nte",
    "nte_bits",
    "mi",
    "mutual_info",
    "mi_bits",
    "ce",
    "conditional_entropy",
    "xe",
    "cross_entropy",
    "xe_bits",
    "cross_entropy_bits",
    "joint_entropy",
    "h_xy",
    "joint_entropy_bits",
    "h_xy_bits",
    "rt",
    "resistance",
    "rt_bits",
    "resistance_bits",
    "tvd",
    "tvd_bits",
    "nhd",
    "nhd_bits",
    "kl",
    "kl_divergence",
    "kl_bits",
    "kl_divergence_bits",
    "js",
    "js_divergence",
    "js_bits",
    "js_divergence_bits",
    "joint_entropy_rate_per_bit",
    "h_xy_rate_per_bit",
    "mi_rate_per_bit",
    "mutual_information_rate_per_bit",
    "xe_rate_per_bit",
    "cross_entropy_rate_per_bit",
    "ce_rate_per_bit",
    "conditional_entropy_rate_per_bit",
    "ned_rate_per_bit",
    "ned_cons_rate_per_bit",
    "nte_rate_per_bit",
    "rt_per_bit",
    "resistance_per_bit",
];

fn run_cli(args: &[&str]) -> Output {
    let bin = env!("CARGO_BIN_EXE_infotheory");
    Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn infotheory")
}

fn stderr_string(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr utf8")
}

#[test]
fn cli_help_runs_for_every_known_command() {
    for cmd in ALL_CLI_COMMANDS {
        let output = run_cli(&[cmd, "--help"]);
        assert!(
            output.status.success(),
            "`infotheory {cmd} --help` did not exit 0; stderr={}",
            stderr_string(&output)
        );
        let help_text = stderr_string(&output);
        assert!(
            help_text.contains(HELP_USAGE_MARKER),
            "`infotheory {cmd} --help` did not print the usage marker `{HELP_USAGE_MARKER}`; stderr={help_text}"
        );
        assert!(
            help_text.contains(HELP_BINARY_MARKER),
            "`infotheory {cmd} --help` did not print the binary marker `{HELP_BINARY_MARKER}`; stderr={help_text}"
        );
    }
}
