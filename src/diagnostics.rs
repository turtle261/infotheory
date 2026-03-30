//! Exact AC/log-loss diagnostics for mixture compression.

use anyhow::{Context, Result, bail};

use crate::compression::{AcLogLossNodeValue, DiagnosticRatePredictor};
use crate::mixture::RateBackendPredictor;
use crate::{MixtureExpertSpec, MixtureKind, MixtureSpec, RateBackend};
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
pub struct AcLogLossRunSummary {
    pub trace_path: PathBuf,
    pub nodes_path: PathBuf,
    pub summary_path: PathBuf,
    pub positions: usize,
    pub mix_total_bits: f64,
    pub oracle_total_bits: f64,
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

fn mixture_kind_label(kind: MixtureKind) -> &'static str {
    match kind {
        MixtureKind::Bayes => "mixture:bayes",
        MixtureKind::FadingBayes => "mixture:fading-bayes",
        MixtureKind::Switching => "mixture:switching",
        MixtureKind::Convex => "mixture:convex",
        MixtureKind::Mdl => "mixture:mdl",
        MixtureKind::Neural => "mixture:neural",
    }
}

fn backend_label(expert: &MixtureExpertSpec) -> String {
    match &expert.backend {
        RateBackend::RosaPlus => format!("rosaplus(max_order={})", expert.max_order),
        RateBackend::Match {
            hash_bits,
            min_len,
            max_len,
            base_mix,
            confidence_scale,
        } => format!(
            "match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},base_mix={base_mix},confidence_scale={confidence_scale})"
        ),
        RateBackend::SparseMatch {
            hash_bits,
            min_len,
            max_len,
            gap_min,
            gap_max,
            base_mix,
            confidence_scale,
        } => format!(
            "sparse-match(hash_bits={hash_bits},min_len={min_len},max_len={max_len},gap_min={gap_min},gap_max={gap_max},base_mix={base_mix},confidence_scale={confidence_scale})"
        ),
        RateBackend::Ppmd { order, memory_mb } => {
            format!("ppmd(order={order},memory_mb={memory_mb})")
        }
        RateBackend::Sequitur { context_bytes } => {
            format!("sequitur(context_bytes={context_bytes})")
        }
        RateBackend::Ctw { depth } => format!("ctw(depth={depth})"),
        RateBackend::FacCtw {
            base_depth,
            num_percept_bits,
            encoding_bits,
        } => format!(
            "fac-ctw(base_depth={base_depth},num_percept_bits={num_percept_bits},encoding_bits={encoding_bits})"
        ),
        #[cfg(feature = "backend-mamba")]
        RateBackend::Mamba { .. } => "mamba".to_string(),
        #[cfg(feature = "backend-mamba")]
        RateBackend::MambaMethod { method } => format!("mamba(method={method})"),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7 { .. } => "rwkv7".to_string(),
        #[cfg(feature = "backend-rwkv")]
        RateBackend::Rwkv7Method { method } => format!("rwkv7(method={method})"),
        RateBackend::Zpaq { method } => format!("zpaq(method={method})"),
        RateBackend::Mixture { spec } => mixture_kind_label(spec.kind).to_string(),
        RateBackend::Particle { spec } => format!(
            "particle(num_particles={},num_cells={})",
            spec.num_particles, spec.num_cells
        ),
        RateBackend::Calibrated { spec } => format!(
            "calibrated(context={:?},bins={},learning_rate={},bias_clip={})",
            spec.context, spec.bins, spec.learning_rate, spec.bias_clip
        ),
    }
}

fn flatten_mixture_spec(spec: &MixtureSpec) -> FlatSchema {
    let mut schema = FlatSchema {
        nodes: vec![FlatNodeMeta {
            id: 0,
            parent_id: None,
            depth: 0,
            path: "0:root".to_string(),
            display_name: "root".to_string(),
            backend_label: mixture_kind_label(spec.kind).to_string(),
            is_mixture: true,
            is_leaf: false,
            is_root_child: false,
        }],
        non_root_ids: Vec::new(),
        root_child_ids: Vec::new(),
    };
    flatten_experts(&mut schema, &spec.experts, 0, 1, "0:root", true);
    schema
}

fn flatten_experts(
    schema: &mut FlatSchema,
    experts: &[MixtureExpertSpec],
    parent_id: usize,
    depth: usize,
    parent_path: &str,
    root_level: bool,
) {
    for expert in experts {
        let raw_display_name = expert.name.clone().unwrap_or_else(|| {
            RateBackendPredictor::default_name(&expert.backend, expert.max_order)
        });
        let display_name = sanitize_tsv_text(&raw_display_name);
        let node_id = schema.nodes.len();
        let path = format!(
            "{parent_path}/{}:{}",
            node_id,
            sanitize_path_segment(&display_name)
        );
        let is_mixture = matches!(expert.backend, RateBackend::Mixture { .. });
        let meta = FlatNodeMeta {
            id: node_id,
            parent_id: Some(parent_id),
            depth,
            path,
            display_name,
            backend_label: sanitize_tsv_text(&backend_label(expert)),
            is_mixture,
            is_leaf: !is_mixture,
            is_root_child: root_level,
        };
        schema.nodes.push(meta);
        schema.non_root_ids.push(node_id);
        if root_level {
            schema.root_child_ids.push(node_id);
        }
        if let RateBackend::Mixture { spec } = &expert.backend {
            let node_path = schema.nodes[node_id].path.clone();
            flatten_experts(schema, &spec.experts, node_id, depth + 1, &node_path, false);
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
    let schema = flatten_mixture_spec(spec);
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

    let mut predictor = DiagnosticRatePredictor::from_rate_backend(
        RateBackend::Mixture {
            spec: Arc::new(spec.clone()),
        },
        -1,
    )?;
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
