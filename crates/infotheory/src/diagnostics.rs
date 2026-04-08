//! Exact AC/log-loss diagnostics for mixture compression.

use anyhow::{Context, Result, bail};

use crate::api::{CompiledRateBackend, MixtureSpec, RateBackend};
#[cfg(test)]
use crate::api::{MixtureExpertSpec, MixtureKind};
use crate::compression::{AcLogLossNodeValue, DiagnosticRatePredictor};
use crate::spec::core::{RateBackendPlan, RateBackendPlanExpert, compiled_rate_backend_from_plan};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct FlatNodeMeta {
    id: usize,
    parent_id: Option<usize>,
    depth: usize,
    path: String,
    display_name: String,
    backend_label: String,
    is_mixture: bool,
    is_leaf: bool,
    is_root_child: bool,
}

#[derive(Clone, Debug)]
struct FlatSchema {
    nodes: Vec<FlatNodeMeta>,
    non_root_ids: Vec<usize>,
    root_child_ids: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NodeSummaryAccum {
    total_bits: f64,
    total_local_weight: f64,
    total_effective_weight: f64,
    oracle_win_count: u64,
}

#[derive(Clone, Debug)]
/// Output summary for an AC/log-loss diagnostic run.
///
/// The diagnostic writer emits three TSV files sharing a common prefix:
///
/// - `*.trace.tsv`: per-position mixture/expert probabilities and weights
/// - `*.nodes.tsv`: flattened mixture-node schema used by trace columns
/// - `*.summary.tsv`: aggregate totals/averages over the full sequence
pub struct AcLogLossRunSummary {
    /// Path to the generated per-position trace TSV.
    pub trace_path: PathBuf,
    /// Path to the generated node-schema TSV.
    pub nodes_path: PathBuf,
    /// Path to the generated aggregate summary TSV.
    pub summary_path: PathBuf,
    /// Number of processed input positions.
    pub positions: usize,
    /// Total mixture code length in bits, computed from mixture probabilities.
    pub mix_total_bits: f64,
    /// Total oracle code length in bits, using best per-step expert in hindsight.
    pub oracle_total_bits: f64,
    /// Raw arithmetic-coder payload size in bits.
    pub ac_payload_bits_raw: u64,
}

#[derive(Default)]
struct CountingWriter {
    bytes_written: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes_written = self.bytes_written.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn parse_diagnostic_threads_from_env() -> Result<usize> {
    match env::var("RAYON_NUM_THREADS") {
        Ok(raw) => {
            let threads = raw
                .parse::<usize>()
                .with_context(|| format!("invalid RAYON_NUM_THREADS value '{raw}'"))?;
            if threads == 0 {
                bail!("RAYON_NUM_THREADS must be >= 1");
            }
            Ok(threads)
        }
        Err(env::VarError::NotPresent) => Ok(1),
        Err(err) => Err(err).context("failed to read RAYON_NUM_THREADS"),
    }
}

fn sanitize_tsv_text(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            '\t' | '\n' | '\r' => ' ',
            _ => ch,
        })
        .collect()
}

fn sanitize_path_segment(input: &str) -> String {
    let clean = sanitize_tsv_text(input);
    let mut out = String::with_capacity(clean.len());
    for ch in clean.chars() {
        match ch {
            '/' | '\\' => out.push('_'),
            _ => out.push(ch),
        }
    }
    if out.is_empty() {
        "node".to_string()
    } else {
        out
    }
}

fn format_f64(value: f64) -> String {
    format!("{value:.17e}")
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn bool_flag(value: bool) -> &'static str {
    if value { "1" } else { "0" }
}

fn bits_from_prob(prob: f64) -> f64 {
    -prob.max(crate::mixture::DEFAULT_MIN_PROB).log2()
}

fn flatten_compiled_mixture(backend: &CompiledRateBackend) -> FlatSchema {
    let RateBackendPlan::Mixture { experts, .. } = backend.plan() else {
        unreachable!("compiled diagnostic root must be a mixture backend");
    };
    let mut schema = FlatSchema {
        nodes: vec![FlatNodeMeta {
            id: 0,
            parent_id: None,
            depth: 0,
            path: "0:root".to_string(),
            display_name: "root".to_string(),
            backend_label: backend.display_label(-1),
            is_mixture: true,
            is_leaf: false,
            is_root_child: false,
        }],
        non_root_ids: Vec::new(),
        root_child_ids: Vec::new(),
    };
    flatten_experts(&mut schema, experts.as_ref(), 0, 1, "0:root", true);
    schema
}

#[cfg(test)]
fn flatten_mixture_spec(spec: &MixtureSpec) -> FlatSchema {
    let backend = RateBackend::Mixture {
        spec: Arc::new(spec.clone()),
    }
    .compile()
    .unwrap_or_else(|err| panic!("failed to compile diagnostic mixture schema: {err}"));
    flatten_compiled_mixture(&backend)
}

fn flatten_experts(
    schema: &mut FlatSchema,
    experts: &[RateBackendPlanExpert],
    parent_id: usize,
    depth: usize,
    parent_path: &str,
    root_level: bool,
) {
    for expert in experts {
        let backend =
            compiled_rate_backend_from_plan(expert.backend.clone()).unwrap_or_else(|err| {
                panic!(
                    "failed to compile diagnostic mixture expert '{}': {err}",
                    expert.name.as_deref().unwrap_or("<unnamed>")
                )
            });
        let raw_display_name = expert
            .name
            .clone()
            .unwrap_or_else(|| backend.default_name(expert.max_order));
        let display_name = sanitize_tsv_text(&raw_display_name);
        let node_id = schema.nodes.len();
        let path = format!(
            "{parent_path}/{}:{}",
            node_id,
            sanitize_path_segment(&display_name)
        );
        let is_mixture = matches!(expert.backend.as_ref(), RateBackendPlan::Mixture { .. });
        let meta = FlatNodeMeta {
            id: node_id,
            parent_id: Some(parent_id),
            depth,
            path,
            display_name,
            backend_label: sanitize_tsv_text(&backend.display_label(expert.max_order)),
            is_mixture,
            is_leaf: !is_mixture,
            is_root_child: root_level,
        };
        schema.nodes.push(meta);
        schema.non_root_ids.push(node_id);
        if root_level {
            schema.root_child_ids.push(node_id);
        }
        if let RateBackendPlan::Mixture { experts, .. } = expert.backend.as_ref() {
            let node_path = schema.nodes[node_id].path.clone();
            flatten_experts(
                schema,
                experts.as_ref(),
                node_id,
                depth + 1,
                &node_path,
                false,
            );
        }
    }
}

fn write_nodes_tsv(path: &Path, schema: &FlatSchema) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create nodes TSV {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "node_id\tparent_id\tdepth\tpath\tdisplay_name\tbackend_label\tis_mixture\tis_leaf\tis_root_child"
    )?;
    for node in &schema.nodes {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            node.id,
            format_optional_usize(node.parent_id),
            node.depth,
            node.path,
            node.display_name,
            node.backend_label,
            bool_flag(node.is_mixture),
            bool_flag(node.is_leaf),
            bool_flag(node.is_root_child),
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// Run exact AC/log-loss diagnostics for a byte sequence under a mixture spec.
///
/// This function validates `spec`, evaluates mixture and expert probabilities at
/// each input position, and writes three TSV artifacts using `out_prefix`:
///
/// - `out_prefix.trace.tsv`: per-position diagnostics and expert rows
/// - `out_prefix.nodes.tsv`: flattened node metadata for trace column mapping
/// - `out_prefix.summary.tsv`: aggregate totals and averages
///
/// The returned summary includes key scalar metrics and the concrete output
/// paths.
pub fn run_ac_log_loss_mixture_bytes(
    data: &[u8],
    spec: &MixtureSpec,
    out_prefix: impl AsRef<Path>,
) -> Result<AcLogLossRunSummary> {
    spec.validate().map_err(anyhow::Error::msg)?;

    let out_prefix = out_prefix.as_ref();
    if let Some(parent) = out_prefix.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let trace_path = out_prefix.with_extension("trace.tsv");
    let nodes_path = out_prefix.with_extension("nodes.tsv");
    let summary_path = out_prefix.with_extension("summary.tsv");
    let compiled_backend = RateBackend::Mixture {
        spec: Arc::new(spec.clone()),
    }
    .compile()
    .map_err(anyhow::Error::msg)?;
    let schema = flatten_compiled_mixture(&compiled_backend);
    write_nodes_tsv(&nodes_path, &schema)?;

    let threads = parse_diagnostic_threads_from_env()?;
    let pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .context("failed to build dedicated AC diagnostic Rayon pool")?,
        )
    } else {
        None
    };

    let mut predictor = DiagnosticRatePredictor::from_compiled(&compiled_backend, -1)?;
    predictor.begin_stream(data.len())?;

    let trace_file = File::create(&trace_path)
        .with_context(|| format!("failed to create trace TSV {}", trace_path.display()))?;
    let mut trace_writer = BufWriter::new(trace_file);

    let mut trace_header = vec![
        "t".to_string(),
        "byte_u8".to_string(),
        "byte_hex".to_string(),
        "mix_prob".to_string(),
        "mix_bits".to_string(),
        "root_weight_entropy_bits".to_string(),
        "root_top1_id".to_string(),
        "root_top1_weight".to_string(),
        "root_top2_id".to_string(),
        "root_top2_weight".to_string(),
        "root_top12_margin".to_string(),
        "root_top1_switched".to_string(),
        "oracle_best_id".to_string(),
        "oracle_best_bits".to_string(),
        "oracle_regret_bits".to_string(),
        "oracle_best_switched".to_string(),
    ];
    for &node_id in &schema.non_root_ids {
        trace_header.push(format!("n{node_id}__prob"));
        trace_header.push(format!("n{node_id}__bits"));
        trace_header.push(format!("n{node_id}__local_weight"));
        trace_header.push(format!("n{node_id}__effective_weight"));
    }
    writeln!(trace_writer, "{}", trace_header.join("\t"))?;

    let mut row_values = Vec::<AcLogLossNodeValue>::with_capacity(schema.non_root_ids.len());
    let mut node_accum = vec![NodeSummaryAccum::default(); schema.non_root_ids.len()];
    let mut mix_total_bits = 0.0;
    let mut oracle_total_bits = 0.0;
    let mut root_weight_entropy_sum = 0.0;
    let mut root_top12_margin_sum = 0.0;
    let mut root_top1_switch_count = 0u64;
    let mut oracle_switch_count = 0u64;
    let mut prev_root_top1_id = None;
    let mut prev_oracle_id = None;

    let mut counter = CountingWriter::default();
    {
        let mut encoder = crate::coders::ArithmeticEncoder::new(&mut counter);
        for (t, &byte) in data.iter().enumerate() {
            let root_snapshot =
                predictor.diagnostic_root_snapshot(byte, pool.as_ref(), &mut row_values)?;
            if row_values.len() != schema.non_root_ids.len() {
                bail!(
                    "diagnostic row width mismatch: got {}, expected {}",
                    row_values.len(),
                    schema.non_root_ids.len()
                );
            }

            let mix_bits = bits_from_prob(root_snapshot.mix_prob);
            mix_total_bits += mix_bits;
            root_weight_entropy_sum += root_snapshot.root_weight_entropy_bits;
            let root_top12_margin = root_snapshot.root_top1_weight - root_snapshot.root_top2_weight;
            root_top12_margin_sum += root_top12_margin;

            let root_top1_id = root_snapshot
                .root_top1_child_index
                .and_then(|idx| schema.root_child_ids.get(idx).copied());
            let root_top2_id = root_snapshot
                .root_top2_child_index
                .and_then(|idx| schema.root_child_ids.get(idx).copied());
            let root_top1_switched = prev_root_top1_id
                .zip(root_top1_id)
                .map(|(prev, curr)| prev != curr)
                .unwrap_or(false);
            if root_top1_switched {
                root_top1_switch_count = root_top1_switch_count.saturating_add(1);
            }
            prev_root_top1_id = root_top1_id;

            let mut oracle_best_row_index = 0usize;
            let mut oracle_best_id = schema.non_root_ids[0];
            let mut oracle_best_bits = bits_from_prob(row_values[0].prob);
            for (index, (&node_id, value)) in schema
                .non_root_ids
                .iter()
                .zip(row_values.iter())
                .enumerate()
            {
                let node_bits = bits_from_prob(value.prob);
                if node_bits < oracle_best_bits {
                    oracle_best_bits = node_bits;
                    oracle_best_id = node_id;
                    oracle_best_row_index = index;
                }
                let acc = &mut node_accum[index];
                acc.total_bits += node_bits;
                acc.total_local_weight += value.local_weight;
                acc.total_effective_weight += value.effective_weight;
            }
            node_accum[oracle_best_row_index].oracle_win_count = node_accum[oracle_best_row_index]
                .oracle_win_count
                .saturating_add(1);

            let oracle_best_switched = prev_oracle_id
                .zip(Some(oracle_best_id))
                .map(|(prev, curr)| prev != curr)
                .unwrap_or(false);
            if oracle_best_switched {
                oracle_switch_count = oracle_switch_count.saturating_add(1);
            }
            prev_oracle_id = Some(oracle_best_id);
            oracle_total_bits += oracle_best_bits;

            let mut row = Vec::with_capacity(trace_header.len());
            row.push(t.to_string());
            row.push(byte.to_string());
            row.push(format!("{byte:02X}"));
            row.push(format_f64(root_snapshot.mix_prob));
            row.push(format_f64(mix_bits));
            row.push(format_f64(root_snapshot.root_weight_entropy_bits));
            row.push(format_optional_usize(root_top1_id));
            row.push(format_f64(root_snapshot.root_top1_weight));
            row.push(format_optional_usize(root_top2_id));
            row.push(format_f64(root_snapshot.root_top2_weight));
            row.push(format_f64(root_top12_margin));
            row.push(bool_flag(root_top1_switched).to_string());
            row.push(oracle_best_id.to_string());
            row.push(format_f64(oracle_best_bits));
            row.push(format_f64(mix_bits - oracle_best_bits));
            row.push(bool_flag(oracle_best_switched).to_string());
            for value in &row_values {
                row.push(format_f64(value.prob));
                row.push(format_f64(bits_from_prob(value.prob)));
                row.push(format_f64(value.local_weight));
                row.push(format_f64(value.effective_weight));
            }
            writeln!(trace_writer, "{}", row.join("\t"))?;

            predictor.encode_symbol_ac_step(byte, &mut encoder)?;
        }
        let _ = encoder.finish()?;
    }
    predictor.finish_stream()?;
    trace_writer.flush()?;

    let positions = data.len();
    let ac_payload_bits_raw = counter.bytes_written.saturating_mul(8);
    let oracle_regret_bits = mix_total_bits - oracle_total_bits;
    let root_weight_entropy_bits_avg = if positions > 0 {
        root_weight_entropy_sum / (positions as f64)
    } else {
        0.0
    };
    let root_top12_margin_avg = if positions > 0 {
        root_top12_margin_sum / (positions as f64)
    } else {
        0.0
    };
    let coder_overhead_bits = (ac_payload_bits_raw as f64) - mix_total_bits;

    let summary_file = File::create(&summary_path)
        .with_context(|| format!("failed to create summary TSV {}", summary_path.display()))?;
    let mut summary_writer = BufWriter::new(summary_file);
    let mut summary_header = vec![
        "positions".to_string(),
        "input_bytes".to_string(),
        "mix_total_bits".to_string(),
        "oracle_total_bits".to_string(),
        "oracle_regret_bits".to_string(),
        "root_top1_switch_count".to_string(),
        "oracle_switch_count".to_string(),
        "root_weight_entropy_bits_avg".to_string(),
        "root_top12_margin_avg".to_string(),
        "ac_payload_bits_raw".to_string(),
        "coder_overhead_bits".to_string(),
    ];
    for &node_id in &schema.non_root_ids {
        summary_header.push(format!("n{node_id}__total_bits"));
        summary_header.push(format!("n{node_id}__regret_bits"));
        summary_header.push(format!("n{node_id}__oracle_win_count"));
        summary_header.push(format!("n{node_id}__avg_local_weight"));
        summary_header.push(format!("n{node_id}__avg_effective_weight"));
    }
    writeln!(summary_writer, "{}", summary_header.join("\t"))?;

    let mut summary_row = vec![
        positions.to_string(),
        data.len().to_string(),
        format_f64(mix_total_bits),
        format_f64(oracle_total_bits),
        format_f64(oracle_regret_bits),
        root_top1_switch_count.to_string(),
        oracle_switch_count.to_string(),
        format_f64(root_weight_entropy_bits_avg),
        format_f64(root_top12_margin_avg),
        ac_payload_bits_raw.to_string(),
        format_f64(coder_overhead_bits),
    ];
    for acc in &node_accum {
        summary_row.push(format_f64(acc.total_bits));
        summary_row.push(format_f64(mix_total_bits - acc.total_bits));
        summary_row.push(acc.oracle_win_count.to_string());
        summary_row.push(format_f64(if positions > 0 {
            acc.total_local_weight / (positions as f64)
        } else {
            0.0
        }));
        summary_row.push(format_f64(if positions > 0 {
            acc.total_effective_weight / (positions as f64)
        } else {
            0.0
        }));
    }
    writeln!(summary_writer, "{}", summary_row.join("\t"))?;
    summary_writer.flush()?;

    Ok(AcLogLossRunSummary {
        trace_path,
        nodes_path,
        summary_path,
        positions,
        mix_total_bits,
        oracle_total_bits,
        ac_payload_bits_raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_nested_spec() -> MixtureSpec {
        MixtureSpec::new(
            MixtureKind::Switching,
            vec![
                MixtureExpertSpec {
                    name: Some("ctw".to_string()),
                    log_prior: 0.0,
                    max_order: -1,
                    backend: RateBackend::Ctw { depth: 6 },
                },
                MixtureExpertSpec {
                    name: Some("nested".to_string()),
                    log_prior: -0.1,
                    max_order: -1,
                    backend: RateBackend::Mixture {
                        spec: Arc::new(MixtureSpec::new(
                            MixtureKind::Bayes,
                            vec![
                                MixtureExpertSpec {
                                    name: Some("fac".to_string()),
                                    log_prior: 0.0,
                                    max_order: -1,
                                    backend: RateBackend::FacCtw {
                                        base_depth: 5,
                                        num_percept_bits: 8,
                                        encoding_bits: 8,
                                    },
                                },
                                MixtureExpertSpec {
                                    name: Some("ppmd".to_string()),
                                    log_prior: 0.0,
                                    max_order: -1,
                                    backend: RateBackend::Ppmd {
                                        order: 4,
                                        memory_mb: 8,
                                    },
                                },
                            ],
                        )),
                    },
                },
            ],
        )
        .with_alpha(0.2)
    }

    #[test]
    fn flatten_schema_includes_submixtures_and_descendants_in_preorder() {
        let schema = flatten_mixture_spec(&test_nested_spec());
        assert_eq!(schema.nodes.len(), 5);
        assert_eq!(schema.nodes[0].display_name, "root");
        assert_eq!(schema.root_child_ids, vec![1, 2]);
        assert_eq!(schema.non_root_ids, vec![1, 2, 3, 4]);
        assert_eq!(schema.nodes[2].display_name, "nested");
        assert!(schema.nodes[2].is_mixture);
        assert_eq!(schema.nodes[3].parent_id, Some(2));
        assert_eq!(schema.nodes[4].parent_id, Some(2));
    }

    #[cfg(any(feature = "default-backends", feature = "all-backends"))]
    #[test]
    fn diagnostic_snapshot_matches_root_pdf_and_oracle_minimum() {
        let spec = test_nested_spec();
        let mut predictor = DiagnosticRatePredictor::from_rate_backend(
            RateBackend::Mixture {
                spec: Arc::new(spec.clone()),
            },
            -1,
        )
        .expect("predictor");
        let data = b"nested diagnostic payload";
        predictor.begin_stream(data.len()).expect("begin stream");
        let mut row_values = Vec::new();

        for &symbol in data {
            let snapshot = predictor
                .diagnostic_root_snapshot(symbol, None, &mut row_values)
                .expect("root snapshot");
            let root_pdf = predictor.pdf_next().expect("root pdf");
            let root_prob = root_pdf[symbol as usize];
            assert!(
                (snapshot.mix_prob - root_prob).abs() < 1e-8,
                "snapshot={} root_pdf={}",
                snapshot.mix_prob,
                root_prob
            );
            let mut oracle_bits = f64::INFINITY;
            for row in &row_values {
                let bits = bits_from_prob(row.prob);
                assert!(
                    (bits + row.prob.log2()).abs() < 1e-8,
                    "bits={} prob={}",
                    bits,
                    row.prob
                );
                oracle_bits = oracle_bits.min(bits);
            }
            assert!(
                oracle_bits <= bits_from_prob(snapshot.mix_prob) + 1e-8,
                "oracle_bits={} mix_bits={}",
                oracle_bits,
                bits_from_prob(snapshot.mix_prob)
            );
            predictor.update(symbol).expect("update");
        }
        predictor.finish_stream().expect("finish stream");
    }

    #[cfg(any(feature = "default-backends", feature = "all-backends"))]
    #[test]
    fn diagnostic_ac_payload_matches_raw_ac_compression_size() {
        let spec = test_nested_spec();
        let data = b"payload bits raw diagnostic parity";
        let stamp = format!(
            "infotheory_ac_diag_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        );
        let prefix = std::env::temp_dir().join(stamp);
        let summary =
            run_ac_log_loss_mixture_bytes(data, &spec, &prefix).expect("diagnostic run succeeds");
        let backend = RateBackend::Mixture {
            spec: Arc::new(spec),
        };
        let backend = backend.compile().expect("compiled mixture backend");
        let encoded = crate::compression::compress_rate_bytes(
            data,
            &backend,
            -1,
            crate::coders::CoderType::AC,
            crate::compression::FramingMode::Raw,
        )
        .expect("raw ac compression");
        assert_eq!(summary.ac_payload_bits_raw, (encoded.len() as u64) * 8);
        let _ = std::fs::remove_file(summary.trace_path);
        let _ = std::fs::remove_file(summary.nodes_path);
        let _ = std::fs::remove_file(summary.summary_path);
    }
}
