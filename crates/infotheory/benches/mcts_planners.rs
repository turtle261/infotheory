use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use infotheory::aixi::common::{Action, Reward};
use infotheory::aixi::mcts::{AgentSimulator, ParallelUctPlanner, RhoUctPlanner};
use rayon::ThreadPool;
use std::num::NonZeroUsize;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("benchmark worker count must be non-zero")
}

#[derive(Clone)]
struct BenchAgent {
    num_actions: usize,
    horizon: usize,
    min_reward: Reward,
    max_reward: Reward,
    discount_gamma: f64,
    explore_exploit_ratio: f64,
    step_work: usize,
    clone_work: usize,
    rng_state: u64,
    last_action: Action,
    emit_reward: bool,
}

impl BenchAgent {
    fn planner_stub(num_actions: usize, horizon: usize, step_work: usize) -> Self {
        Self {
            num_actions,
            horizon,
            min_reward: 0,
            max_reward: 1,
            discount_gamma: 0.95,
            explore_exploit_ratio: 1.0,
            step_work,
            clone_work: step_work / 2,
            rng_state: 1,
            last_action: 0,
            emit_reward: false,
        }
    }

    fn burn(&mut self, rounds: usize) -> u64 {
        let mut acc = self.rng_state ^ 0x9E37_79B9_7F4A_7C15;
        for _ in 0..rounds {
            acc ^= acc << 7;
            acc ^= acc >> 9;
            acc = acc.wrapping_mul(0xA24B_AED4_963E_E407);
        }
        self.rng_state = acc;
        acc
    }
}

impl AgentSimulator for BenchAgent {
    fn get_num_actions(&self) -> usize {
        self.num_actions
    }

    fn get_num_observation_bits(&self) -> usize {
        1
    }

    fn get_num_reward_bits(&self) -> usize {
        1
    }

    fn horizon(&self) -> usize {
        self.horizon
    }

    fn max_reward(&self) -> Reward {
        self.max_reward
    }

    fn min_reward(&self) -> Reward {
        self.min_reward
    }

    fn get_explore_exploit_ratio(&self) -> f64 {
        self.explore_exploit_ratio
    }

    fn discount_gamma(&self) -> f64 {
        self.discount_gamma
    }

    fn model_update_action(&mut self, action: Action) {
        self.last_action = action;
        self.emit_reward = false;
        let _ = self.burn(self.step_work / 4);
    }

    fn gen_percept_and_update(&mut self, _bits: usize) -> u64 {
        let mix = self.burn(self.step_work.max(1));
        if self.emit_reward {
            self.emit_reward = false;
            ((self.last_action ^ mix) & 1) as u64
        } else {
            self.emit_reward = true;
            ((mix >> 5) & 1) as u64
        }
    }

    fn begin_simulation(&mut self) {
        self.emit_reward = false;
    }

    fn model_revert(&mut self, _steps: usize) {
        self.emit_reward = false;
    }

    fn gen_range(&mut self, end: usize) -> usize {
        if end <= 1 {
            return 0;
        }
        (self.burn(1) as usize) % end
    }

    fn gen_f64(&mut self) -> f64 {
        const SCALE: f64 = (u64::MAX as f64) + 1.0;
        (self.burn(1) as f64) / SCALE
    }

    fn boxed_clone_with_seed(&self, seed: u64) -> Box<dyn AgentSimulator> {
        let mut clone = self.clone();
        clone.rng_state ^= seed.wrapping_mul(0xD6E8_FD50_5D2B_9B7D);
        let _ = clone.burn(clone.clone_work.max(1));
        Box::new(clone)
    }
}

fn planner_pool(threads: usize) -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("benchmark thread pool")
}

fn bench_planner_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_planner_throughput");
    let samples = 256usize;
    group.throughput(Throughput::Elements(samples as u64));

    let seq_agent = BenchAgent::planner_stub(8, 6, 8);
    group.bench_with_input(
        BenchmarkId::new("rho_uct", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                let mut planner = RhoUctPlanner::new();
                let mut agent = seq_agent.clone();
                black_box(planner.search(&mut agent, &[0], 0, 0, samples));
            });
        },
    );

    let wu_pool = planner_pool(1);
    let wu_agent = BenchAgent::planner_stub(8, 6, 8);
    group.bench_with_input(
        BenchmarkId::new("parallel_uct_wu_workers1", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                wu_pool.install(|| {
                    let mut planner =
                        ParallelUctPlanner::new(nz(1), None).expect("valid parallel_uct planner");
                    let mut agent = wu_agent.clone();
                    black_box(
                        planner
                            .search(&mut agent, &[0], 0, 0, samples)
                            .expect("benchmark planner stub uses positive horizon"),
                    );
                });
            });
        },
    );

    let par_pool = planner_pool(4);
    let par_agent = BenchAgent::planner_stub(8, 6, 8);
    group.bench_with_input(
        BenchmarkId::new("parallel_uct_wu_workers4", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                par_pool.install(|| {
                    let mut planner =
                        ParallelUctPlanner::new(nz(4), None).expect("valid parallel_uct planner");
                    let mut agent = par_agent.clone();
                    black_box(
                        planner
                            .search(&mut agent, &[0], 0, 0, samples)
                            .expect("benchmark planner stub uses positive horizon"),
                    );
                });
            });
        },
    );

    let bu_pool = planner_pool(4);
    let bu_agent = BenchAgent::planner_stub(8, 6, 8);
    group.bench_with_input(
        BenchmarkId::new("parallel_uct_bu_core_workers4", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                bu_pool.install(|| {
                    let mut planner = ParallelUctPlanner::new(nz(4), Some(0.8))
                        .expect("valid parallel_uct planner");
                    let mut agent = bu_agent.clone();
                    black_box(
                        planner
                            .search(&mut agent, &[0], 0, 0, samples)
                            .expect("benchmark planner stub uses positive horizon"),
                    );
                });
            });
        },
    );
    group.finish();
}

fn bench_tuner_shaped_mcts(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_tuner_shaped");
    let samples = 192usize;
    group.throughput(Throughput::Elements(samples as u64));

    let seq_agent = BenchAgent::planner_stub(64, 4, 48);
    group.bench_with_input(
        BenchmarkId::new("rho_uct_actions64_h4", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                let mut planner = RhoUctPlanner::new();
                let mut agent = seq_agent.clone();
                black_box(planner.search(&mut agent, &[0], 0, 0, samples));
            });
        },
    );

    let wu_pool = planner_pool(4);
    let wu_agent = BenchAgent::planner_stub(64, 4, 48);
    group.bench_with_input(
        BenchmarkId::new("parallel_uct_wu_actions64_h4", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                wu_pool.install(|| {
                    let mut planner =
                        ParallelUctPlanner::new(nz(4), None).expect("valid parallel_uct planner");
                    let mut agent = wu_agent.clone();
                    black_box(
                        planner
                            .search(&mut agent, &[0], 0, 0, samples)
                            .expect("benchmark planner stub uses positive horizon"),
                    );
                });
            });
        },
    );

    let bu_pool = planner_pool(4);
    let bu_agent = BenchAgent::planner_stub(64, 4, 48);
    group.bench_with_input(
        BenchmarkId::new("parallel_uct_bu_core_actions64_h4", samples),
        &samples,
        |b, &samples| {
            b.iter(|| {
                bu_pool.install(|| {
                    let mut planner = ParallelUctPlanner::new(nz(4), Some(0.8))
                        .expect("valid parallel_uct planner");
                    let mut agent = bu_agent.clone();
                    black_box(
                        planner
                            .search(&mut agent, &[0], 0, 0, samples)
                            .expect("benchmark planner stub uses positive horizon"),
                    );
                });
            });
        },
    );
    group.finish();
}

criterion_group!(
    mcts_planners,
    bench_planner_throughput,
    bench_tuner_shaped_mcts
);
criterion_main!(mcts_planners);
