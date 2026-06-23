use infotheory::aixi::warmstart::{
    WarmStartExactJhAgent, WarmStartExactJhTeacherDataset, WarmStartExactJhTeacherTrace,
    WarmStartExactJhTransition, standalone_warmstart_teacher_contract_for_compiled_planner_run,
};
use infotheory::spec::SpecDocument;
use serde_json::json;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn compiled_warmstart_spec() -> infotheory::spec::CompiledPlannerRunSpec {
    let value = json!({
        "schema_version": 1,
        "kind": "planner_run",
        "assets": [{
            "id": "teacher",
            "path": "teacher.json"
        }],
        "environment": {"kind": "builtin", "name": "coin_flip"},
        "interface": {
            "observation_bits": 1,
            "observation_stream_len": 1,
            "observation_key_mode": "full_stream",
            "reward_bits": 1,
            "agent_actions": 2
        },
        "controller": {
            "kind": "aiqi_warmstart_exact_jh",
            "predictor": {"kind": "ctw", "depth": 8},
            "bit_stream_semantics": {"kind": "binary_tokens"},
            "return_horizon": 4,
            "return_bins": 5,
            "label_phase_period": 4,
            "teacher_dataset_asset": "teacher",
            "planner_simulations_per_step": 1
        },
        "runtime": {
            "random_seed": 7,
            "learn_cycles": 16,
            "eval_cycles": 0,
            "terminate_lifetime": 16,
            "log_every": 1,
            "perf": false,
            "vm_perf_only": false,
            "explore_epsilon": 0.0,
            "explore_gamma": 1.0
        }
    });
    let document =
        SpecDocument::parse_json_value(&value, Path::new(".")).expect("benchmark spec parses");
    let SpecDocument::PlannerRun(spec) = document else {
        panic!("expected planner_run benchmark spec");
    };
    spec.compile()
        .expect("benchmark planner spec should compile")
}

fn teacher_dataset(
    compiled: &infotheory::spec::CompiledPlannerRunSpec,
) -> WarmStartExactJhTeacherDataset {
    let contract = standalone_warmstart_teacher_contract_for_compiled_planner_run(compiled)
        .expect("benchmark contract should be valid");
    let transitions = (0..32usize)
        .map(|step| {
            WarmStartExactJhTransition::new(
                (step % 2) as u64,
                vec![(step % 2) as u64],
                (step % 2) as i64,
            )
        })
        .collect::<Vec<_>>();
    WarmStartExactJhTeacherDataset::new(
        contract,
        vec![WarmStartExactJhTeacherTrace::new(transitions)],
    )
}

fn seeded_agent() -> WarmStartExactJhAgent {
    let compiled = compiled_warmstart_spec();
    let teacher = teacher_dataset(&compiled);
    let mut agent = WarmStartExactJhAgent::from_compiled_planner_run(&compiled, teacher)
        .expect("benchmark warmstart agent should construct");
    for step in 0..16usize {
        agent
            .observe_transition((step % 2) as u64, &[(step % 2) as u64], (step % 2) as i64)
            .expect("benchmark transition should be accepted");
    }
    agent
}

fn bench_operation(
    name: &str,
    iterations: usize,
    mut operation: impl FnMut(&mut WarmStartExactJhAgent),
) {
    let mut agent = seeded_agent();
    let start = Instant::now();
    for _ in 0..iterations {
        operation(&mut agent);
    }
    let elapsed = start.elapsed();
    let ns_per_iter = (elapsed.as_nanos() as f64) / (iterations as f64);
    let iters_per_s = (iterations as f64) / elapsed.as_secs_f64().max(1e-12);
    println!(
        "{:>24}: {:>8.3} ms total | {:>10.1} ns/iter | {:>10.1} iters/s",
        name,
        elapsed.as_secs_f64() * 1e3,
        ns_per_iter,
        iters_per_s
    );
}

fn main() {
    let iterations = env_usize("WARMSTART_BENCH_ITERS", 1_000);
    println!("Warm-start exact-J_H benchmark (iterations={iterations})");

    let build_iterations = env_usize("WARMSTART_BENCH_BUILD_ITERS", 100);
    let compiled = compiled_warmstart_spec();
    let teacher = teacher_dataset(&compiled);
    let start = Instant::now();
    for _ in 0..build_iterations {
        black_box(
            WarmStartExactJhAgent::from_compiled_planner_run(&compiled, teacher.clone())
                .expect("benchmark warmstart agent should construct"),
        );
    }
    let elapsed = start.elapsed();
    println!(
        "{:>24}: {:>8.3} ms total | {:>10.1} ns/build | {:>10.1} builds/s",
        "construct",
        elapsed.as_secs_f64() * 1e3,
        (elapsed.as_nanos() as f64) / (build_iterations as f64),
        (build_iterations as f64) / elapsed.as_secs_f64().max(1e-12)
    );

    bench_operation("observe_transition", iterations, |agent| {
        let _: () = agent
            .observe_transition(1, &[1], 1)
            .expect("benchmark transition should be accepted");
        black_box(());
    });
    bench_operation("estimate_action_values", iterations, |agent| {
        black_box(
            agent
                .estimate_action_values()
                .expect("benchmark action values should compute"),
        );
    });
    bench_operation("get_planned_action", iterations, |agent| {
        black_box(agent.get_planned_action());
    });
}
