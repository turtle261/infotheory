//! Integration tests for the nyx-lite VM environment.
#![cfg(feature = "vm")]
//!
//! These tests validate the NyxVmEnvironment implementation by testing:
//! - Configuration validation
//! - Action handling (literal and fuzz modes)
//! - Observation policies
//! - Reward policies
//! - Information-theoretic filtering
//! - Environment trait compliance
//!
//! Note: Full VM tests require a running Firecracker environment with
//! the appropriate guest image. Unit tests for non-VM components can
//! run without the VM.

use infotheory::aixi::vm_nyx::*;
use infotheory::api::RateBackend;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

fn vm_test_stats_backend() -> RateBackend {
    RateBackend::Ctw { depth: 20 }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_default_config() {
    let config = NyxVmConfig::default();

    assert_eq!(config.observation_bits, 8);
    assert_eq!(config.reward_bits, 8);
    assert_eq!(config.episode_steps, 100);
    assert_eq!(config.observation_stream_len, 64);
    assert!(config.firecracker_config.is_empty());
    assert!(matches!(
        config.stats_backend,
        RateBackend::Ctw { depth: 20 }
    ));
}

#[test]
fn test_protocol_config_default() {
    let protocol = NyxProtocolConfig::default();

    assert_eq!(protocol.action_prefix, "ACT ");
    assert_eq!(protocol.action_suffix, "\n");
    assert_eq!(protocol.obs_prefix, "OBS ");
    assert_eq!(protocol.rew_prefix, "REW ");
    assert_eq!(protocol.done_prefix, "DONE ");
    assert!(matches!(protocol.wire_encoding, PayloadEncoding::Hex));
}

// ============================================================================
// Encoding Tests
// ============================================================================

#[test]
fn test_hex_encoding_roundtrip() {
    let test_cases = vec![
        b"hello world".to_vec(),
        b"\x00\x01\x02\xff\xfe".to_vec(),
        Vec::new(),
        vec![0u8; 256],
    ];

    for original in test_cases {
        let encoded = PayloadEncoding::Hex.encode(&original);
        let decoded = PayloadEncoding::Hex.decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }
}

#[test]
fn test_utf8_encoding() {
    let data = b"test string";
    let encoded = PayloadEncoding::Utf8.encode(data);
    assert_eq!(encoded, "test string");

    let decoded = PayloadEncoding::Utf8.decode("test string").unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn test_hex_decode_with_whitespace() {
    let hex = "48 65 6c 6c\n6f";
    let decoded = PayloadEncoding::Hex.decode(hex).unwrap();
    assert_eq!(decoded, b"Hello");
}

#[test]
fn test_hex_decode_error_odd_length() {
    let result = PayloadEncoding::Hex.decode("123");
    assert!(result.is_err());
}

#[test]
fn test_hex_decode_error_invalid_char() {
    let result = PayloadEncoding::Hex.decode("zz");
    assert!(result.is_err());
}

// ============================================================================
// Action Source Tests
// ============================================================================

#[test]
fn test_literal_action_source() {
    let actions = vec![
        NyxActionSpec::named("ping", b"PING".to_vec()),
        NyxActionSpec::named("pong", b"PONG".to_vec()),
    ];

    let source = NyxActionSource::Literal(actions.clone());

    if let NyxActionSource::Literal(a) = source {
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].payload, b"PING");
        assert_eq!(a[1].payload, b"PONG");
    } else {
        panic!("Expected Literal action source");
    }
}

#[test]
fn test_fuzz_config() {
    let mut config = NyxFuzzConfig::new(vec![b"seed1".to_vec(), b"seed2".to_vec()]);
    config.mutators = vec![FuzzMutator::FlipBit, FuzzMutator::FlipByte];
    config.min_len = 1;
    config.max_len = 1024;
    config.dictionary = vec![b"dict1".to_vec()];
    config.rng_seed = 42;

    assert_eq!(config.seeds.len(), 2);
    assert_eq!(config.mutators.len(), 2);
    assert_eq!(config.min_len, 1);
    assert_eq!(config.max_len, 1024);
}

// ============================================================================
// Observation Policy Tests
// ============================================================================

#[test]
fn test_observation_stream_modes() {
    // These are just enum variant existence tests
    let _ = NyxObservationStreamMode::PadTruncate;
    let _ = NyxObservationStreamMode::Pad;
    let _ = NyxObservationStreamMode::Truncate;
}

#[test]
fn test_observation_policies() {
    // Ensure all observation policies can be instantiated
    let policies = [
        NyxObservationPolicy::FromGuest,
        NyxObservationPolicy::OutputHash,
        NyxObservationPolicy::RawOutput,
        NyxObservationPolicy::SharedMemory,
    ];

    assert_eq!(policies.len(), 4);
}

// ============================================================================
// Reward Policy Tests
// ============================================================================

#[test]
fn test_reward_policy_from_guest() {
    let policy = NyxRewardPolicy::FromGuest;
    let debug_str = format!("{:?}", policy);
    assert!(debug_str.contains("FromGuest"));
}

#[test]
fn test_reward_policy_pattern() {
    let policy = NyxRewardPolicy::Pattern {
        pattern: "SUCCESS".to_string(),
        base_reward: 0,
        bonus_reward: 100,
    };
    let debug_str = format!("{:?}", policy);
    assert!(debug_str.contains("Pattern"));
    assert!(debug_str.contains("SUCCESS"));
}

#[test]
fn test_reward_shaping_entropy() {
    let shaping = NyxRewardShaping::EntropyReduction {
        baseline_bytes: vec![0u8; 100],
        max_order: 8,
        scale: 1.0,
        crash_bonus: None,
        timeout_bonus: None,
    };
    let debug_str = format!("{:?}", shaping);
    assert!(debug_str.contains("EntropyReduction"));
}

#[test]
fn test_reward_shaping_trace() {
    let shaping = NyxRewardShaping::TraceEntropy {
        max_order: 4,
        scale: 2.0,
        normalize: true,
    };
    let debug_str = format!("{:?}", shaping);
    assert!(debug_str.contains("TraceEntropy"));
}

#[test]
fn test_reward_policy_custom() {
    let custom_fn = Arc::new(|result: &NyxStepResult| -> i64 {
        if result.done { 100 } else { 0 }
    });
    let policy = NyxRewardPolicy::Custom(custom_fn);
    let debug_str = format!("{:?}", policy);
    assert!(debug_str.contains("Custom"));
}

// ============================================================================
// Action Filter Tests
// ============================================================================

#[test]
fn test_action_filter() {
    let mut filter = NyxActionFilter::new();
    filter.min_entropy = Some(1.0);
    filter.max_entropy = Some(7.5);
    filter.min_intrinsic_dependence = Some(0.1);
    filter.min_novelty = Some(0.5);
    filter.novelty_prior = Some(vec![0, 1, 2, 3]);
    filter.max_order = 8;
    filter.reject_reward = Some(-10);

    assert_eq!(filter.min_entropy, Some(1.0));
    assert_eq!(filter.max_entropy, Some(7.5));
    assert_eq!(filter.reject_reward, Some(-10));
}

// ============================================================================
// Exit Kind Tests
// ============================================================================

#[test]
fn test_exit_kind_from_exit_reason() {
    use nyx_lite::ExitReason;

    let exit_done = ExitReason::ExecDone(42);
    let kind: NyxExitKind = exit_done.into();
    if let NyxExitKind::ExecDone(code) = kind {
        assert_eq!(code, 42);
    } else {
        panic!("Expected ExecDone");
    }

    let exit_timeout = ExitReason::Timeout;
    let kind: NyxExitKind = exit_timeout.into();
    assert!(matches!(kind, NyxExitKind::Timeout));

    let exit_shutdown = ExitReason::Shutdown;
    let kind: NyxExitKind = exit_shutdown.into();
    assert!(matches!(kind, NyxExitKind::Shutdown));
}

// ============================================================================
// Step Result Tests
// ============================================================================

#[test]
fn test_step_result_default() {
    let result = NyxStepResult {
        exit_reason: NyxExitKind::Timeout,
        output: Vec::new(),
        parsed_obs: None,
        parsed_rew: None,
        done: false,
        trace_data: Vec::new(),
        shared_memory: Vec::new(),
    };

    assert!(!result.done);
    assert!(result.parsed_obs.is_none());
    assert!(result.parsed_rew.is_none());
}

// ============================================================================
// Hypercall Constants Tests
// ============================================================================

#[test]
fn test_hypercall_constants() {
    // Verify the hypercall magic numbers are correct
    // These decode to ASCII strings
    assert_eq!(HYPERCALL_EXECDONE, 0x656e6f6463657865);
    assert_eq!(HYPERCALL_SNAPSHOT, 0x746f687370616e73);
    assert_eq!(HYPERCALL_NYX_LITE, 0x6574696c2d78796e);
    assert_eq!(HYPERCALL_SHAREMEM, 0x6d656d6572616873);
    assert_eq!(HYPERCALL_DBGPRINT, 0x746e697270676264);
}

// ============================================================================
// Trace Config Tests
// ============================================================================

#[test]
fn test_trace_config() {
    let mut config = NyxTraceConfig::new();
    config.shared_region_name = Some("trace_buffer".to_string());
    config.max_bytes = 4096;
    config.reset_on_episode = true;

    assert_eq!(config.shared_region_name, Some("trace_buffer".to_string()));
    assert_eq!(config.max_bytes, 4096);
    assert!(config.reset_on_episode);
}

// ============================================================================
// Payload Encoding Tests
// ============================================================================

#[test]
fn test_payload_encoding_from_str() {
    // Canonical parser is `std::str::FromStr`.
    assert!(matches!(
        PayloadEncoding::from_str("utf8"),
        Ok(PayloadEncoding::Utf8)
    ));
    assert!(matches!(
        PayloadEncoding::from_str("hex"),
        Ok(PayloadEncoding::Hex)
    ));
    assert!(PayloadEncoding::from_str("unknown").is_err());
    assert!(PayloadEncoding::from_str("text").is_err());
}

// ============================================================================
// Fuzz Mutator Tests (without actual mutation - that requires RandomGenerator)
// ============================================================================

#[test]
fn test_fuzz_mutator_variants() {
    let mutators = [
        FuzzMutator::FlipBit,
        FuzzMutator::FlipByte,
        FuzzMutator::InsertByte,
        FuzzMutator::DeleteByte,
        FuzzMutator::SpliceSeed,
        FuzzMutator::ResetSeed,
        FuzzMutator::Havoc,
    ];

    assert_eq!(mutators.len(), 7);
}

// ============================================================================
// Information-Theoretic Properties Tests
// ============================================================================

/// Tests that verify information-theoretic properties hold.
mod info_theory_properties {
    #[allow(unused_imports)]
    use super::*;
    use infotheory::api::marginal_entropy_bytes;
    #[cfg(feature = "backend-rosa")]
    use infotheory::api::try_entropy_rate_bytes;

    #[test]
    fn test_entropy_bounds() {
        // Maximum entropy for bytes is 8 bits
        let uniform_data: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let h = marginal_entropy_bytes(&uniform_data);
        assert!(h <= 8.0 + 1e-6, "Entropy should not exceed 8 bits per byte");
        assert!(h >= 0.0, "Entropy should be non-negative");
    }

    #[test]
    fn test_constant_data_low_entropy() {
        let constant_data = vec![42u8; 1000];
        let h = marginal_entropy_bytes(&constant_data);
        assert!(
            h < 0.01,
            "Constant data should have near-zero marginal entropy"
        );
    }

    #[cfg(feature = "backend-rosa")]
    #[test]
    fn test_rate_entropy_less_than_marginal() {
        // For structured data, H_rate <= H_marginal
        let pattern = b"ABCABCABCABCABCABC";
        let h_marg = marginal_entropy_bytes(pattern);
        let h_rate = try_entropy_rate_bytes(pattern, 8).expect("entropy rate");

        assert!(
            h_rate <= h_marg + 1e-6,
            "Entropy rate should not exceed marginal entropy for patterned data"
        );
    }
}

// ============================================================================
// Integration Test Markers (require VM)
// ============================================================================

/// These tests require a running Firecracker VM with proper setup.
/// They are marked with #[ignore] and can be run with:
/// cargo test -- --ignored
#[cfg(all(test, feature = "backend-ctw"))]
mod vm_integration_tests {
    #[allow(unused_imports)]
    use super::*;
    use infotheory::aixi::environment::Environment;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn get_project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    fn check_kvm_available() -> bool {
        std::path::Path::new("/dev/kvm").exists()
    }

    fn create_firecracker_config(kernel: &Path, initrd: &Path, test_name: &str) -> PathBuf {
        let config_json = format!(
            r#"{{
  "boot-source": {{
    "kernel_image_path": "{}",
    "initrd_path": "{}",
    "boot_args": "ro console=ttyS0 noapic reboot=k panic=1 pci=off nomodules random.trust_cpu=on init=/init quiet loglevel=3"
  }},
  "drives": [],
  "machine-config": {{
    "vcpu_count": 1,
    "mem_size_mib": 1024
  }}
}}"#,
            kernel.to_string_lossy(),
            initrd.to_string_lossy()
        );

        let root = get_project_root();
        let config_path = root.join("target").join(format!(
            "test_vm_config_{}_{}.json",
            std::process::id(),
            test_name
        ));
        // Ensure target dir exists
        let _ = std::fs::create_dir_all(root.join("target"));

        let mut file = File::create(&config_path).expect("Failed to create temp config");
        file.write_all(config_json.as_bytes())
            .expect("Failed to write config");

        config_path
    }

    #[cfg(feature = "backend-ctw")]
    fn get_test_vm_config(test_name: &str) -> Option<NyxVmConfig> {
        let root = get_project_root();
        let kernel_path = root.join("vmlinux-6.1.58");
        let initrd_path = root.join("vendor/nyx-lite/guest/aixi_initramfs.cpio");

        if !kernel_path.exists() {
            eprintln!("Skipping VM test: Kernel not found at {:?}", kernel_path);
            return None;
        }
        if !initrd_path.exists() {
            eprintln!("Skipping VM test: Initrd not found at {:?}", initrd_path);
            return None;
        }

        let fc_config_path = create_firecracker_config(&kernel_path, &initrd_path, test_name);

        let mut config = NyxVmConfig::default();
        config.firecracker_config = fc_config_path.to_string_lossy().to_string();
        config.instance_id = format!("test-vm-{}-{}", std::process::id(), test_name);
        config.shared_region_name = "shared".to_string();
        config.shared_region_size = 4096;
        config.shared_memory_policy = SharedMemoryPolicy::Snapshot;
        config.step_timeout = Duration::from_millis(500);
        config.boot_timeout = Duration::from_secs(30);
        config.episode_steps = 10;
        config.step_cost = 1;
        config.observation_policy = NyxObservationPolicy::SharedMemory;
        config.observation_bits = 8;
        config.observation_stream_len = 1;
        config.observation_stream_mode = NyxObservationStreamMode::PadTruncate;
        config.observation_pad_byte = 0;
        config.reward_bits = 8;
        config.reward_policy = NyxRewardPolicy::FromGuest;
        config.reward_shaping = None;
        config.action_source = NyxActionSource::Literal(vec![
            NyxActionSpec::named("nop", vec![]),
            NyxActionSpec::named("act", vec![0x01]),
        ]);
        config.action_filter = None;
        config.protocol = NyxProtocolConfig::default();
        config.stats_backend = vm_test_stats_backend();
        config.trace = None;
        config.debug_mode = true;
        config.crash_log = None;
        Some(config)
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn test_vm_boot_and_snapshot() {
        if !check_kvm_available() {
            eprintln!("Skipping VM test: /dev/kvm not found");
            return;
        }
        let Some(config) = get_test_vm_config("boot") else {
            return;
        };
        // Clean up config file on return ideally, but fine for test

        let result = NyxVmEnvironment::new(config);

        match result {
            Ok(mut env) => {
                assert!(!env.is_finished());
                env.perform_action(0);

                let obs = env.get_observation();
                let rew = env.get_reward();

                println!("VM Boot Success. Obs={}, Rew={}", obs, rew);
                assert!(!env.is_finished());
            }
            Err(e) => {
                panic!("Failed to create NyxVmEnvironment: {:?}", e);
            }
        }
    }

    #[cfg(feature = "backend-ctw")]
    #[test]
    fn test_vm_action_execution() {
        if !check_kvm_available() {
            return;
        }
        let Some(config) = get_test_vm_config("action") else {
            return;
        };

        let mut env = NyxVmEnvironment::new(config).expect("Failed to init VM");

        for _ in 0..5 {
            env.perform_action(1);
            let _ = env.get_observation();
            // In RawOutput mode, we might get 0 bytes if VM is quiet, or console logs.
            // Just verifying we don't crash.
        }
    }
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

/// Test that configuration can be built incrementally
#[test]
fn test_config_builder_pattern() {
    let mut config = NyxVmConfig::default();

    config.instance_id = "test-instance".to_string();
    config.episode_steps = 50;
    config.observation_bits = 16;
    config.reward_bits = 16;
    config.debug_mode = true;

    assert_eq!(config.instance_id, "test-instance");
    assert_eq!(config.episode_steps, 50);
    assert_eq!(config.observation_bits, 16);
    assert_eq!(config.reward_bits, 16);
    assert!(config.debug_mode);
}

/// Test complete configuration for a typical experiment
#[test]
fn test_complete_experiment_config() {
    let mut config = NyxVmConfig::default();
    config.firecracker_config = "/path/to/config.json".to_string();
    config.instance_id = "experiment-1".to_string();
    config.shared_region_name = "shared".to_string();
    config.shared_region_size = 4096;
    config.shared_memory_policy = SharedMemoryPolicy::Snapshot;
    config.step_timeout = Duration::from_millis(100);
    config.boot_timeout = Duration::from_secs(30);
    config.episode_steps = 100;
    config.step_cost = 1;
    config.observation_policy = NyxObservationPolicy::SharedMemory;
    config.observation_bits = 8;
    config.observation_stream_len = 64;
    config.observation_stream_mode = NyxObservationStreamMode::PadTruncate;
    config.observation_pad_byte = 0;
    config.reward_bits = 8;
    config.reward_policy = NyxRewardPolicy::FromGuest;
    config.reward_shaping = Some(NyxRewardShaping::TraceEntropy {
        max_order: 8,
        scale: 1.0,
        normalize: true,
    });
    config.action_source = NyxActionSource::Literal(vec![
        NyxActionSpec::named("nop", vec![]),
        NyxActionSpec::named("action1", b"A".to_vec()),
    ]);
    let mut action_filter = NyxActionFilter::new();
    action_filter.min_entropy = Some(0.5);
    action_filter.max_order = 4;
    action_filter.reject_reward = Some(-1);
    config.action_filter = Some(action_filter);
    config.protocol = NyxProtocolConfig::default();
    config.stats_backend = vm_test_stats_backend();
    let mut trace = NyxTraceConfig::new();
    trace.shared_region_name = Some("trace".to_string());
    trace.max_bytes = 1024;
    trace.reset_on_episode = true;
    config.trace = Some(trace);
    config.debug_mode = false;
    config.crash_log = None;

    assert_eq!(config.episode_steps, 100);
    assert_eq!(config.step_cost, 1);
    assert!(config.action_filter.is_some());
    assert!(config.trace.is_some());
}
