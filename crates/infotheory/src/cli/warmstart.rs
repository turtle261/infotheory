use infotheory::aixi::planner_agent::{
    NullPlannerObserver, PlannerControllerAgent, PlannerRunSession, build_planner_environment,
    compile_planner_run_document,
};
use infotheory::aixi::warmstart::{
    WarmStartExactJhTeacherDataset, WarmStartExactJhTeacherTrace, WarmStartExactJhTransition,
    merge_warmstart_teacher_traces_deterministic, read_warmstart_teacher_dataset_path,
    standalone_warmstart_teacher_contract_for_compiled_planner_run,
    validate_warmstart_teacher_dataset_for_compiled_planner_run, warmstart_target_return_horizon,
    warmstart_teacher_trace_from_jsonl_path, write_warmstart_teacher_dataset_path,
};
use infotheory::spec::{CanonicalJson, CompiledPlannerRunSpec};

/// Request for exporting a warm-start teacher dataset from a planner run.
pub(crate) struct WarmStartTeacherPlannerRunRequest {
    /// Target warm-start planner-run path.
    pub target_path: String,
    /// Teacher planner-run path.
    pub teacher_path: String,
    /// Output teacher dataset path.
    pub out_path: String,
}

/// Request for converting planner JSONL to a warm-start teacher dataset.
pub(crate) struct WarmStartTeacherJsonlRequest {
    /// Target warm-start planner-run path.
    pub target_path: String,
    /// Input JSONL path.
    pub jsonl_path: String,
    /// Output teacher dataset path.
    pub out_path: String,
}

/// Request for merging warm-start teacher datasets.
pub(crate) struct WarmStartTeacherMergeRequest {
    /// Target warm-start planner-run path.
    pub target_path: String,
    /// Input teacher dataset paths.
    pub teacher_paths: Vec<String>,
    /// Output teacher dataset path.
    pub out_path: String,
}

pub(crate) enum WarmStartCommand {
    PlannerRun(WarmStartTeacherPlannerRunRequest),
    FromJsonl(WarmStartTeacherJsonlRequest),
    Merge(WarmStartTeacherMergeRequest),
}

fn option_value(
    args: &[String],
    index: &mut usize,
    option: &str,
    expected: &str,
) -> anyhow::Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{option} requires {expected}"))
}

fn set_once(slot: &mut Option<String>, value: String, option: &str) -> anyhow::Result<()> {
    if slot.replace(value).is_some() {
        return Err(anyhow::anyhow!("duplicate {option}"));
    }
    Ok(())
}

fn parse_teacher_planner_run(args: &[String]) -> anyhow::Result<WarmStartTeacherPlannerRunRequest> {
    let mut target_path = None;
    let mut teacher_path = None;
    let mut out_path = None;
    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => set_once(
                &mut target_path,
                option_value(args, &mut i, "--target", "a warmstart planner_run path")?,
                "--target",
            )?,
            "--teacher" => set_once(
                &mut teacher_path,
                option_value(args, &mut i, "--teacher", "a teacher planner_run path")?,
                "--teacher",
            )?,
            "--out" => set_once(
                &mut out_path,
                option_value(args, &mut i, "--out", "an output teacher JSON path")?,
                "--out",
            )?,
            other => {
                return Err(anyhow::anyhow!(
                    "unknown warmstart teacher planner-run option '{other}'"
                ));
            }
        }
        i += 1;
    }
    Ok(WarmStartTeacherPlannerRunRequest {
        target_path: target_path.ok_or_else(|| anyhow::anyhow!("missing required --target"))?,
        teacher_path: teacher_path.ok_or_else(|| anyhow::anyhow!("missing required --teacher"))?,
        out_path: out_path.ok_or_else(|| anyhow::anyhow!("missing required --out"))?,
    })
}

fn parse_teacher_from_jsonl(args: &[String]) -> anyhow::Result<WarmStartTeacherJsonlRequest> {
    let mut target_path = None;
    let mut jsonl_path = None;
    let mut out_path = None;
    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => set_once(
                &mut target_path,
                option_value(args, &mut i, "--target", "a warmstart planner_run path")?,
                "--target",
            )?,
            "--jsonl" => set_once(
                &mut jsonl_path,
                option_value(args, &mut i, "--jsonl", "a JSONL trace path")?,
                "--jsonl",
            )?,
            "--out" => set_once(
                &mut out_path,
                option_value(args, &mut i, "--out", "an output teacher JSON path")?,
                "--out",
            )?,
            other => {
                return Err(anyhow::anyhow!(
                    "unknown warmstart teacher from-jsonl option '{other}'"
                ));
            }
        }
        i += 1;
    }
    Ok(WarmStartTeacherJsonlRequest {
        target_path: target_path.ok_or_else(|| anyhow::anyhow!("missing required --target"))?,
        jsonl_path: jsonl_path.ok_or_else(|| anyhow::anyhow!("missing required --jsonl"))?,
        out_path: out_path.ok_or_else(|| anyhow::anyhow!("missing required --out"))?,
    })
}

fn parse_teacher_merge(args: &[String]) -> anyhow::Result<WarmStartTeacherMergeRequest> {
    let mut target_path = None;
    let mut teacher_paths = Vec::new();
    let mut out_path = None;
    let mut i = 4usize;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => set_once(
                &mut target_path,
                option_value(args, &mut i, "--target", "a warmstart planner_run path")?,
                "--target",
            )?,
            "--teacher" => teacher_paths.push(option_value(
                args,
                &mut i,
                "--teacher",
                "a teacher JSON path",
            )?),
            "--out" => set_once(
                &mut out_path,
                option_value(args, &mut i, "--out", "an output teacher JSON path")?,
                "--out",
            )?,
            other => {
                return Err(anyhow::anyhow!(
                    "unknown warmstart teacher merge option '{other}'"
                ));
            }
        }
        i += 1;
    }
    if teacher_paths.is_empty() {
        return Err(anyhow::anyhow!("missing required --teacher"));
    }
    Ok(WarmStartTeacherMergeRequest {
        target_path: target_path.ok_or_else(|| anyhow::anyhow!("missing required --target"))?,
        teacher_paths,
        out_path: out_path.ok_or_else(|| anyhow::anyhow!("missing required --out"))?,
    })
}

pub(crate) fn parse_warmstart_command(args: &[String]) -> anyhow::Result<WarmStartCommand> {
    match (
        args.get(1).map(String::as_str),
        args.get(2).map(String::as_str),
        args.get(3).map(String::as_str),
    ) {
        (Some("warmstart"), Some("teacher"), Some("planner-run")) => Ok(
            WarmStartCommand::PlannerRun(parse_teacher_planner_run(args)?),
        ),
        (Some("warmstart"), Some("teacher"), Some("from-jsonl")) => {
            Ok(WarmStartCommand::FromJsonl(parse_teacher_from_jsonl(args)?))
        }
        (Some("warmstart"), Some("teacher"), Some("merge")) => {
            Ok(WarmStartCommand::Merge(parse_teacher_merge(args)?))
        }
        _ => Err(anyhow::anyhow!(
            "usage: infotheory warmstart teacher <planner-run|from-jsonl|merge> ..."
        )),
    }
}

fn compile_target(path: &str) -> anyhow::Result<CompiledPlannerRunSpec> {
    compile_planner_run_document(path, "warmstart teacher --target")
}

fn ensure_teacher_interface_matches_target(
    target: &CompiledPlannerRunSpec,
    teacher: &CompiledPlannerRunSpec,
) -> anyhow::Result<()> {
    if target.interface().agent_actions != teacher.interface().agent_actions
        || target.interface().observation_bits != teacher.interface().observation_bits
        || target.interface().observation_stream_len != teacher.interface().observation_stream_len
        || target.interface().reward_bits != teacher.interface().reward_bits
        || target.interface().observation_key_mode != teacher.interface().observation_key_mode
    {
        return Err(anyhow::anyhow!(
            "teacher planner interface does not match target warmstart planner interface"
        ));
    }
    Ok(())
}

fn compiled_environment_value(
    compiled: &CompiledPlannerRunSpec,
) -> anyhow::Result<serde_json::Value> {
    let value = compiled
        .canonical_spec()
        .to_canonical_json_value()
        .map_err(anyhow::Error::msg)?;
    value
        .get("environment")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("compiled planner canonical JSON is missing environment"))
}

fn validate_warmstart_teacher_export_compatibility(
    target: &CompiledPlannerRunSpec,
    teacher: &CompiledPlannerRunSpec,
) -> anyhow::Result<()> {
    ensure_teacher_interface_matches_target(target, teacher)?;
    if target.runtime().vm_perf_only || teacher.runtime().vm_perf_only {
        return Err(anyhow::anyhow!(
            "warmstart teacher planner-run export does not support vm_perf_only planner specs"
        ));
    }
    let target_environment = compiled_environment_value(target)?;
    let teacher_environment = compiled_environment_value(teacher)?;
    if target_environment != teacher_environment {
        return Err(anyhow::anyhow!(
            "teacher planner environment does not match target warmstart planner environment"
        ));
    }
    Ok(())
}

pub(crate) fn run_warmstart_teacher_planner_run_export(
    request: &WarmStartTeacherPlannerRunRequest,
) -> anyhow::Result<()> {
    let target = compile_target(&request.target_path)?;
    let teacher = compile_planner_run_document(
        &request.teacher_path,
        "warmstart teacher planner-run --teacher",
    )?;
    validate_warmstart_teacher_export_compatibility(&target, &teacher)?;
    let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(&target)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let (env, _) = build_planner_environment(&teacher)?;
    let agent = PlannerControllerAgent::from_compiled(&teacher)
        .map_err(|err| anyhow::anyhow!("failed to construct teacher planner: {err}"))?;
    let mut session = PlannerRunSession::new(&teacher, agent, env)
        .map_err(|err| anyhow::anyhow!("failed to initialize teacher planner session: {err}"))?;
    let mut observer = NullPlannerObserver;
    let mut transitions = Vec::new();
    while let Some(outcome) = session
        .run_next_cycle(&mut observer)
        .map_err(|err| anyhow::anyhow!("{err}"))?
    {
        transitions.push(WarmStartExactJhTransition::new(
            outcome.action,
            outcome.observations,
            outcome.reward,
        ));
    }
    let trace = WarmStartExactJhTeacherTrace::new(transitions);
    let dataset = WarmStartExactJhTeacherDataset::new(contract, vec![trace]);
    validate_warmstart_teacher_dataset_for_compiled_planner_run(&target, &dataset)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    write_warmstart_teacher_dataset_path(&request.out_path, &dataset)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!(
        "Warm-start teacher dataset written to {} ({} trace).",
        request.out_path,
        dataset.traces.len()
    );
    Ok(())
}

pub(crate) fn run_warmstart_teacher_from_jsonl_export(
    request: &WarmStartTeacherJsonlRequest,
) -> anyhow::Result<()> {
    let target = compile_target(&request.target_path)?;
    let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(&target)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let return_horizon =
        warmstart_target_return_horizon(&target).map_err(|err| anyhow::anyhow!("{err}"))?;
    let trace =
        warmstart_teacher_trace_from_jsonl_path(&request.jsonl_path, &contract, return_horizon)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
    let dataset = WarmStartExactJhTeacherDataset::new(contract, vec![trace]);
    validate_warmstart_teacher_dataset_for_compiled_planner_run(&target, &dataset)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    write_warmstart_teacher_dataset_path(&request.out_path, &dataset)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!(
        "Warm-start teacher dataset written to {} from {}.",
        request.out_path, request.jsonl_path
    );
    Ok(())
}

pub(crate) fn run_warmstart_teacher_merge(
    request: &WarmStartTeacherMergeRequest,
) -> anyhow::Result<()> {
    let target = compile_target(&request.target_path)?;
    let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(&target)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let mut merged = WarmStartExactJhTeacherDataset::new(contract, Vec::new());
    let mut expected_contract = None;
    let mut inserted_total = 0usize;
    for path in &request.teacher_paths {
        let dataset =
            read_warmstart_teacher_dataset_path(path).map_err(|err| anyhow::anyhow!("{err}"))?;
        validate_warmstart_teacher_dataset_for_compiled_planner_run(&target, &dataset)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        if let Some(contract) = &expected_contract {
            if contract != &dataset.contract {
                return Err(anyhow::anyhow!(
                    "warmstart teacher merge input '{}' has a contract different from earlier inputs",
                    path
                ));
            }
        } else {
            expected_contract = Some(dataset.contract.clone());
        }
        let (inserted, _) =
            merge_warmstart_teacher_traces_deterministic(&mut merged.traces, dataset.traces);
        inserted_total = inserted_total.saturating_add(inserted);
    }
    if merged.traces.is_empty() {
        return Err(anyhow::anyhow!(
            "merged teacher dataset would contain no traces"
        ));
    }
    validate_warmstart_teacher_dataset_for_compiled_planner_run(&target, &merged)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    write_warmstart_teacher_dataset_path(&request.out_path, &merged)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!(
        "Warm-start teacher dataset written to {} ({} traces, {} inserted).",
        request.out_path,
        merged.traces.len(),
        inserted_total
    );
    Ok(())
}

pub(crate) fn run_warmstart_command(command: WarmStartCommand) -> anyhow::Result<()> {
    match command {
        WarmStartCommand::PlannerRun(request) => run_warmstart_teacher_planner_run_export(&request),
        WarmStartCommand::FromJsonl(request) => run_warmstart_teacher_from_jsonl_export(&request),
        WarmStartCommand::Merge(request) => run_warmstart_teacher_merge(&request),
    }
}

#[cfg(all(test, feature = "backend-ctw"))]
mod tests {
    use super::*;
    use infotheory::aixi::planner_agent::PlannerActionProvenance;
    use infotheory::aixi::warmstart::{
        WarmStartExactJhTeacherTrace, WarmStartExactJhTransition, warmstart_jsonl_action_record,
        warmstart_jsonl_percept_record,
    };
    use infotheory::aixi::warmstart_contract::{
        TaskFingerprint, warmstart_exact_jh_planner_task_fingerprint,
    };
    use infotheory::spec::SpecDocument;
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_TEST_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
        let counter = TEMP_TEST_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{nanos}-{counter}{suffix}",
            std::process::id()
        ))
    }

    fn planner_run_value(
        environment_name: &str,
        teacher_path: &Path,
        return_horizon: usize,
        vm_perf_only: bool,
    ) -> serde_json::Value {
        json!({
            "schema_version": 1,
            "kind": "planner_run",
            "assets": [{
                "id": "teacher",
                "path": teacher_path.to_string_lossy()
            }],
            "environment": {
                "kind": "builtin",
                "name": environment_name
            },
            "interface": {
                "observation_bits": 2,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "reward_bits": 2,
                "agent_actions": 2
            },
            "controller": {
                "kind": "aiqi_warmstart_exact_jh",
                "predictor": {
                    "kind": "ctw",
                    "depth": 4
                },
                "return_horizon": return_horizon,
                "return_bins": return_horizon + 1,
                "label_phase_period": return_horizon,
                "teacher_dataset_asset": "teacher",
                "planner_simulations_per_step": 1
            },
            "runtime": {
                "random_seed": 7,
                "learn_cycles": 1,
                "eval_cycles": 0,
                "terminate_lifetime": 1,
                "log_every": 1,
                "perf": false,
                "vm_perf_only": vm_perf_only,
                "explore_epsilon": 0.0,
                "explore_gamma": 1.0
            }
        })
    }

    fn write_planner_run(
        path: &Path,
        environment_name: &str,
        teacher_path: &Path,
        return_horizon: usize,
        vm_perf_only: bool,
    ) {
        std::fs::write(
            path,
            serde_json::to_vec(&planner_run_value(
                environment_name,
                teacher_path,
                return_horizon,
                vm_perf_only,
            ))
            .expect("serialize planner_run"),
        )
        .expect("write planner_run");
    }

    fn compile_planner_value(
        environment_name: &str,
        teacher_path: &Path,
        return_horizon: usize,
        vm_perf_only: bool,
    ) -> CompiledPlannerRunSpec {
        let document = SpecDocument::parse_json_value(
            &planner_run_value(environment_name, teacher_path, return_horizon, vm_perf_only),
            Path::new("."),
        )
        .expect("parse planner_run");
        let SpecDocument::PlannerRun(spec) = document else {
            panic!("expected planner_run document");
        };
        spec.compile().expect("compile planner_run")
    }

    fn write_teacher_dataset_for_target(
        path: &Path,
        compiled: &CompiledPlannerRunSpec,
        task_fingerprint_override: Option<&str>,
    ) {
        let mut contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(compiled)
            .expect("standalone contract");
        if let Some(task_fingerprint) = task_fingerprint_override {
            contract.task_fingerprint = TaskFingerprint::parse_hex(task_fingerprint)
                .expect("test task fingerprint override must be canonical hex");
        }
        let dataset = WarmStartExactJhTeacherDataset::new(
            contract,
            vec![WarmStartExactJhTeacherTrace::new(vec![
                WarmStartExactJhTransition::new(0, vec![1], 1),
            ])],
        );
        write_warmstart_teacher_dataset_path(path, &dataset).expect("write teacher dataset");
    }

    #[test]
    fn planner_run_export_compatibility_rejects_environment_and_vm_perf_mismatch() {
        let teacher_path = unique_temp_path("warmstart-cli-teacher", ".json");
        let target = compile_planner_value("coin_flip", &teacher_path, 1, false);
        let other_environment =
            compile_planner_value("biased_rock_paper_scissor", &teacher_path, 1, false);
        let err = validate_warmstart_teacher_export_compatibility(&target, &other_environment)
            .expect_err("environment mismatch must fail before export");
        assert!(
            err.to_string().contains("environment does not match"),
            "{err}"
        );

        let vm_perf_only = compile_planner_value("coin_flip", &teacher_path, 1, true);
        let err = validate_warmstart_teacher_export_compatibility(&target, &vm_perf_only)
            .expect_err("vm_perf_only teacher must fail before export");
        assert!(err.to_string().contains("vm_perf_only"), "{err}");
    }

    #[test]
    fn from_jsonl_export_rejects_short_trace_before_writing() {
        let target_path = unique_temp_path("warmstart-cli-target", ".json");
        let teacher_asset_path = unique_temp_path("warmstart-cli-target-teacher", ".json");
        let jsonl_path = unique_temp_path("warmstart-cli-short", ".jsonl");
        let out_path = unique_temp_path("warmstart-cli-short-out", ".json");
        write_planner_run(&target_path, "coin_flip", &teacher_asset_path, 2, false);
        let jsonl = [
            warmstart_jsonl_action_record(0, 0, PlannerActionProvenance::Greedy).to_string(),
            warmstart_jsonl_percept_record(0, &[1], 1).to_string(),
        ]
        .join("\n");
        std::fs::write(&jsonl_path, jsonl).expect("write jsonl");

        let err = run_warmstart_teacher_from_jsonl_export(&WarmStartTeacherJsonlRequest {
            target_path: target_path.to_string_lossy().into_owned(),
            jsonl_path: jsonl_path.to_string_lossy().into_owned(),
            out_path: out_path.to_string_lossy().into_owned(),
        })
        .expect_err("short JSONL trace must fail before write");
        assert!(err.to_string().contains("return_horizon is 2"), "{err}");
        assert!(
            !out_path.exists(),
            "failed export must not leave a teacher output file"
        );

        let _ = std::fs::remove_file(target_path);
        let _ = std::fs::remove_file(jsonl_path);
    }

    #[test]
    fn merge_rejects_contract_mismatch_before_writing() {
        let target_path = unique_temp_path("warmstart-cli-merge-target", ".json");
        let target_teacher_asset = unique_temp_path("warmstart-cli-merge-target-teacher", ".json");
        let teacher_a_path = unique_temp_path("warmstart-cli-merge-a", ".json");
        let teacher_b_path = unique_temp_path("warmstart-cli-merge-b", ".json");
        let out_path = unique_temp_path("warmstart-cli-merge-out", ".json");
        write_planner_run(&target_path, "coin_flip", &target_teacher_asset, 1, false);
        let target = compile_target(target_path.to_str().expect("utf-8 target path"))
            .expect("compile target");
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&target).expect("task fingerprint");
        write_teacher_dataset_for_target(&teacher_a_path, &target, None);
        write_teacher_dataset_for_target(
            &teacher_b_path,
            &target,
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
        );

        let err = run_warmstart_teacher_merge(&WarmStartTeacherMergeRequest {
            target_path: target_path.to_string_lossy().into_owned(),
            teacher_paths: vec![
                teacher_a_path.to_string_lossy().into_owned(),
                teacher_b_path.to_string_lossy().into_owned(),
            ],
            out_path: out_path.to_string_lossy().into_owned(),
        })
        .expect_err("mismatched merge input contract must fail before write");
        assert!(err.to_string().contains("task_fingerprint"), "{err}");
        assert!(
            err.to_string().contains(&task_fingerprint.to_string()),
            "{err}"
        );
        assert!(
            !out_path.exists(),
            "failed merge must not leave a teacher output file"
        );

        let _ = std::fs::remove_file(target_path);
        let _ = std::fs::remove_file(teacher_a_path);
        let _ = std::fs::remove_file(teacher_b_path);
    }
}
