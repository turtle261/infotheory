use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use infotheory::aixi::warmstart::{WarmStartExactJhAgent, WarmStartExactJhTeacherDataset};
use infotheory::aixi::warmstart_contract::{
    TaskFingerprint, WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
    WARMSTART_STANDALONE_SCALAR_REPRESENTATION, WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
    standalone_exact_reward_encoding_certificate_hash,
    standalone_observation_adapter_content_crc32, warmstart_exact_jh_planner_task_fingerprint,
};
use infotheory::spec::{CompiledPlannerRunSpec, SpecDocument};
use serde_json::json;

#[derive(Clone, Copy)]
struct WarmstartBenchCase {
    name: &'static str,
    actions: usize,
    observation_bits: usize,
    reward_bits: usize,
    return_horizon: usize,
    return_bins: usize,
    label_phase_period: usize,
    teacher_steps: usize,
    live_steps: usize,
}

fn bench_cases() -> [WarmstartBenchCase; 2] {
    [
        WarmstartBenchCase {
            name: "small",
            actions: 2,
            observation_bits: 1,
            reward_bits: 2,
            return_horizon: 1,
            return_bins: 4,
            label_phase_period: 1,
            teacher_steps: 16,
            live_steps: 4,
        },
        WarmstartBenchCase {
            name: "medium",
            actions: 4,
            observation_bits: 2,
            reward_bits: 3,
            return_horizon: 3,
            return_bins: 16,
            label_phase_period: 3,
            teacher_steps: 64,
            live_steps: 12,
        },
    ]
}

fn compiled_planner_run(case: WarmstartBenchCase) -> CompiledPlannerRunSpec {
    let document = SpecDocument::parse_json_value(
        &json!({
            "schema_version": 1,
            "kind": "planner_run",
            "assets": [{
                "id": "teacher",
                "path": "warmstart-bench-teacher.json"
            }],
            "environment": {
                "kind": "builtin",
                "name": "coin_flip"
            },
            "interface": {
                "observation_bits": case.observation_bits,
                "observation_stream_len": 1,
                "observation_key_mode": "full_stream",
                "reward_bits": case.reward_bits,
                "agent_actions": case.actions
            },
            "controller": {
                "kind": "aiqi_warmstart_exact_jh",
                "predictor": {
                    "kind": "ctw",
                    "depth": 8
                },
                "bit_stream_semantics": { "kind": "binary_tokens" },
                "return_horizon": case.return_horizon,
                "return_bins": case.return_bins,
                "label_phase_period": case.label_phase_period,
                "teacher_dataset_asset": "teacher",
                "planner_simulations_per_step": 1
            },
            "runtime": {
                "random_seed": 7,
                "learn_cycles": 1,
                "eval_cycles": 1,
                "terminate_lifetime": 2,
                "log_every": 1,
                "perf": false,
                "vm_perf_only": false,
                "explore_epsilon": 0.0,
                "explore_gamma": 1.0
            }
        }),
        std::path::Path::new("."),
    )
    .expect("benchmark planner document");
    let SpecDocument::PlannerRun(spec) = document else {
        panic!("benchmark document must be planner_run");
    };
    spec.compile().expect("benchmark planner run must compile")
}

fn teacher_dataset(
    case: WarmstartBenchCase,
    task_fingerprint: TaskFingerprint,
) -> WarmStartExactJhTeacherDataset {
    let max_reward = ((case.return_bins - 1) / case.return_horizon) as i64;
    let observation_mod = 1u64 << case.observation_bits;
    let action_mod = u64::try_from(case.actions).expect("benchmark action count must fit u64");
    let transitions = (0..case.teacher_steps)
        .map(|step| {
            json!({
                "action": (step as u64) % action_mod,
                "observations": [(step as u64) % observation_mod],
                "reward": (step as i64) % (max_reward + 1),
            })
        })
        .collect::<Vec<_>>();
    WarmStartExactJhTeacherDataset::from_json_value(&json!({
        "schema_version": WARMSTART_TEACHER_CONTRACT_SCHEMA_VERSION,
        "contract": {
            "task_fingerprint": task_fingerprint.to_string(),
            "action_alphabet_size": case.actions,
            "observation_bits": case.observation_bits,
            "observation_stream_len": 1,
            "observation_key_mode": "full_stream",
            "observation_adapter_spec_ref": WARMSTART_STANDALONE_OBSERVATION_ADAPTER_SPEC_REF,
            "observation_adapter_content_crc32": standalone_observation_adapter_content_crc32(
                case.observation_bits,
                1,
                case.reward_bits,
            )
            .expect("benchmark observation adapter digest"),
            "reward_bits": case.reward_bits,
            "return_horizon": case.return_horizon,
            "label_phase_period": case.label_phase_period,
            "scalar_representation": WARMSTART_STANDALONE_SCALAR_REPRESENTATION,
            "exact_reward_encoding_certificate": standalone_exact_reward_encoding_certificate_hash(
                case.reward_bits,
            )
            .expect("benchmark reward certificate digest"),
        },
        "traces": [{
            "transitions": transitions,
        }],
    }))
    .expect("benchmark teacher dataset")
}

fn seeded_agent(
    case: WarmstartBenchCase,
    compiled: &CompiledPlannerRunSpec,
    teacher: &WarmStartExactJhTeacherDataset,
) -> WarmStartExactJhAgent {
    let mut agent =
        WarmStartExactJhAgent::from_compiled_planner_run(compiled, teacher.clone()).expect("agent");
    let observation_mod = 1u64 << case.observation_bits;
    let action_mod = u64::try_from(case.actions).expect("benchmark action count must fit u64");
    let max_reward = ((case.return_bins - 1) / case.return_horizon) as i64;
    for step in 0..case.live_steps {
        agent
            .observe_transition(
                (step as u64) % action_mod,
                &[(step as u64) % observation_mod],
                (step as i64) % (max_reward + 1),
            )
            .expect("benchmark live transition");
    }
    agent
}

fn bench_warmstart(c: &mut Criterion) {
    for case in bench_cases() {
        let compiled = compiled_planner_run(case);
        let task_fingerprint =
            warmstart_exact_jh_planner_task_fingerprint(&compiled).expect("task fingerprint");
        let teacher = teacher_dataset(case, task_fingerprint);

        let mut values_group = c.benchmark_group("warmstart_action_values");
        values_group.bench_with_input(
            BenchmarkId::new("estimate_action_values", case.name),
            &case,
            |b, &case| {
                b.iter_batched(
                    || seeded_agent(case, &compiled, &teacher),
                    |mut agent| black_box(agent.estimate_action_values().expect("action values")),
                    BatchSize::SmallInput,
                );
            },
        );
        values_group.finish();

        let mut plan_group = c.benchmark_group("warmstart_action_selection");
        plan_group.bench_with_input(
            BenchmarkId::new("get_planned_action", case.name),
            &case,
            |b, &case| {
                b.iter_batched(
                    || seeded_agent(case, &compiled, &teacher),
                    |mut agent| black_box(agent.get_planned_action()),
                    BatchSize::SmallInput,
                );
            },
        );
        plan_group.finish();

        let mut observe_group = c.benchmark_group("warmstart_observe_transition");
        observe_group.bench_with_input(
            BenchmarkId::new("observe_transition", case.name),
            &case,
            |b, &case| {
                b.iter_batched(
                    || seeded_agent(case, &compiled, &teacher),
                    |mut agent| {
                        agent
                            .observe_transition(0, &[0], 0)
                            .expect("observe transition");
                        black_box(agent)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        observe_group.finish();

        let mut construct_group = c.benchmark_group("warmstart_construction");
        construct_group.bench_with_input(
            BenchmarkId::new("from_teacher_dataset", case.name),
            &case,
            |b, _| {
                b.iter(|| {
                    black_box(
                        WarmStartExactJhAgent::from_compiled_planner_run(
                            &compiled,
                            teacher.clone(),
                        )
                        .expect("agent"),
                    )
                });
            },
        );
        construct_group.finish();
    }
}

criterion_group!(warmstart, bench_warmstart);
criterion_main!(warmstart);
