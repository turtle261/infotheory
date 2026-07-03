use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use csv::ReaderBuilder;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, CrosstermBackend, Line, Modifier, Span, Style};
use ratatui::symbols::Marker;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, GraphType, List, ListItem, ListState, Paragraph,
    Wrap,
};
use ratatui::{Frame, Terminal};

mod log_loss;

use log_loss::{LogLossApp, LogLossCli};

const DEFAULT_PLOT_DIR: &str = "/tmp/plotimgs";
const DEFAULT_BENCH_SUITE: &str = "two-json";
const PLOT_DIR_MARKER_FILE: &str = ".benchman-managed-plot-dir";
const LEGACY_PLOT_SVG_SUFFIX: &str = ".svg";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchSuite {
    TwoJson,
    OneSse,
    Extra,
}

impl BenchSuite {
    fn parse(raw: &str) -> Result<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "two-json" | "two_json" | "two" | "core" | "full" => Ok(Self::TwoJson),
            "one-sse" | "one_sse" | "one" => Ok(Self::OneSse),
            "extra" => Ok(Self::Extra),
            _ => bail!(
                "unknown benchmark suite '{raw}' (expected 'two-json', 'one-sse', or 'extra')"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::TwoJson => "two-json",
            Self::OneSse => "one-sse",
            Self::Extra => "extra",
        }
    }

    fn summary_prefix(self) -> &'static str {
        match self {
            Self::TwoJson => "infotheory-two-json",
            Self::OneSse => "infotheory-one-sse",
            Self::Extra => "infotheory-extra",
        }
    }

    fn spec_label(self) -> &'static str {
        match self {
            Self::TwoJson => "configs/bench/two.json",
            Self::OneSse => "configs/bench/one_sse.json",
            Self::Extra => "configs/bench/extra.json",
        }
    }

    fn focus_subjects(self) -> &'static [&'static str] {
        match self {
            Self::TwoJson => &["neural_mixture", "rwkv7"],
            Self::OneSse => &["calibrated_mixture", "bit-reservoir"],
            Self::Extra => &["neural_mixture", "mamba"],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlotDirDeletionPolicy {
    ManagedMarker,
    LegacyArtifacts,
    RequiresExplicitConfirmation,
}

pub(crate) const COLOR_PALETTE: [Color; 12] = [
    Color::Rgb(0, 114, 178),
    Color::Rgb(213, 94, 0),
    Color::Rgb(0, 158, 115),
    Color::Rgb(204, 121, 167),
    Color::Rgb(230, 159, 0),
    Color::Rgb(86, 180, 233),
    Color::Rgb(240, 228, 66),
    Color::Rgb(102, 166, 30),
    Color::Rgb(231, 41, 138),
    Color::Rgb(166, 118, 29),
    Color::Rgb(117, 112, 179),
    Color::Rgb(102, 194, 165),
];

#[derive(Parser, Debug)]
#[command(
    name = "benchman",
    version,
    about = "Interactive InfoTheory TUI for benchmarks and AC log-loss diagnostics"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<BenchmanCommand>,

    #[command(flatten)]
    bench: BenchCli,
}

#[derive(Subcommand, Debug, Clone)]
enum BenchmanCommand {
    /// Inspect AC log-loss TSV diagnostics produced by `infotheory ac-log-loss`
    LogLoss(LogLossCli),
}

#[derive(Args, Debug, Clone)]
struct BenchCli {
    #[arg(
        long,
        env = "INFOTHEORY_PLOT_SUITE",
        default_value = DEFAULT_BENCH_SUITE,
        help = "Benchmark suite (`two-json`, `one-sse`, or `extra`) used for latest-summary discovery and plot regeneration"
    )]
    suite: String,

    #[arg(
        long,
        env = "INFOTHEORY_PLOT_SUMMARY_TSV",
        help = "Path to summary TSV. Defaults to latest /tmp/<suite>-summary-*.tsv for the selected suite"
    )]
    summary_tsv: Option<PathBuf>,

    #[arg(
        long,
        env = "INFOTHEORY_BASELINE_SUMMARY_TSV",
        help = "Optional baseline summary TSV for current-vs-baseline overlays"
    )]
    baseline_summary_tsv: Option<PathBuf>,

    #[arg(
        long,
        env = "INFOTHEORY_BENCH_RAW_TSV",
        help = "Optional raw TSV (used for per-repetition detail in Enter inspector)"
    )]
    raw_tsv: Option<PathBuf>,

    #[arg(
        long,
        env = "INFOTHEORY_PLOT_SUBJECTS",
        help = "Optional comma/whitespace-separated subject filter"
    )]
    subjects: Option<String>,

    #[arg(
        long,
        env = "INFOTHEORY_PLOT_OUTPUT_DIR",
        default_value = DEFAULT_PLOT_DIR,
        help = "Plot artifact directory; benchman enforces A/B reuse-or-rebuild prompt when it exists"
    )]
    plot_dir: PathBuf,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip running scripts/plot_two_json.sh even when plot dir is missing or rebuilt"
    )]
    no_replot: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum SummarySource {
    Current,
    Baseline,
}

impl SummarySource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Baseline => "baseline",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::enum_variant_names)]
enum Metric {
    RssKibMedian,
    RealSecondsMedian,
    EntropyBpbMedian,
    ArchiveRatioMedian,
}

impl Metric {
    fn y_label(self) -> &'static str {
        match self {
            Self::RssKibMedian => "peak RSS (KiB)",
            Self::RealSecondsMedian => "seconds",
            Self::EntropyBpbMedian => "bits per byte",
            Self::ArchiveRatioMedian => "archive/input ratio",
        }
    }

    fn key_name(self) -> &'static str {
        match self {
            Self::RssKibMedian => "rss_kib_median",
            Self::RealSecondsMedian => "real_seconds_median",
            Self::EntropyBpbMedian => "entropy_bpb_median",
            Self::ArchiveRatioMedian => "archive_ratio_median",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DataView {
    Current,
    Combined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeriesField {
    Subject,
    Series,
    SubjectOverlay,
    SeriesOverlay,
    SummarySource,
}

#[derive(Clone, Debug)]
struct RenderRow {
    source: SummarySource,
    operation: String,
    subject: String,
    series: String,
    size_bytes: u64,
    compression_backend: String,
    real_seconds_median: Option<f64>,
    rss_kib_median: Option<f64>,
    entropy_bpb_median: Option<f64>,
    archive_ratio_median: Option<f64>,
}

impl RenderRow {
    fn subject_overlay(&self) -> String {
        format!("{} ({})", self.subject, self.source.as_str())
    }

    fn series_overlay(&self) -> String {
        format!("{} ({})", self.series, self.source.as_str())
    }

    fn metric_value(&self, metric: Metric) -> Option<f64> {
        match metric {
            Metric::RssKibMedian => self.rss_kib_median,
            Metric::RealSecondsMedian => self.real_seconds_median,
            Metric::EntropyBpbMedian => self.entropy_bpb_median,
            Metric::ArchiveRatioMedian => self.archive_ratio_median,
        }
    }
}

#[derive(Clone, Debug)]
struct RawSample {
    repetition: u32,
    real_seconds: f64,
    user_seconds: f64,
    sys_seconds: f64,
    rss_kib: f64,
    archive_bytes: Option<u64>,
    entropy_bpb: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct RawKey {
    operation: String,
    subject: String,
    size_bytes: u64,
    compression_backend: String,
}

#[derive(Clone, Debug)]
struct GraphSpec {
    id: String,
    title: String,
    operation_filter: Option<String>,
    subject_filter: Option<String>,
    metric: Metric,
    series_field: SeriesField,
    data_view: DataView,
}

#[derive(Clone, Debug)]
struct PointMeta {
    x_bytes: u64,
    x_log10: f64,
    y: f64,
    operation: String,
    subject: String,
    compression_backend: String,
    source: SummarySource,
}

#[derive(Clone, Debug)]
struct SeriesData {
    name: String,
    color: Color,
    points: Vec<PointMeta>,
}

#[derive(Clone, Debug)]
struct GraphModel {
    spec: GraphSpec,
    series: Vec<SeriesData>,
}

#[derive(Clone, Debug)]
struct GraphState {
    visible_series: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct InspectionEntry {
    overlap_group: Option<usize>,
    series: String,
    y: f64,
    source: SummarySource,
    raw_samples: Vec<RawSample>,
}

#[derive(Clone, Debug)]
struct Inspection {
    x_bytes: u64,
    metric: Metric,
    entries: Vec<InspectionEntry>,
}

#[derive(Clone, Debug)]
struct FilterPopup {
    series_names: Vec<String>,
    selected_idx: usize,
    visible_series: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct BenchData {
    suite: BenchSuite,
    current_rows: Vec<RenderRow>,
    combined_rows: Vec<RenderRow>,
    baseline_paths_mismatch: Option<(usize, usize)>,
    raw_index: HashMap<RawKey, Vec<RawSample>>,
    summary_path: PathBuf,
    baseline_path: Option<PathBuf>,
    raw_path: Option<PathBuf>,
    plot_dir: PathBuf,
}

#[derive(Clone, Debug)]
struct ResolvedInputs {
    suite: BenchSuite,
    summary_path: PathBuf,
    baseline_path: Option<PathBuf>,
    raw_path: Option<PathBuf>,
    subjects_raw: Option<String>,
    subject_filter: Option<BTreeSet<String>>,
    plot_dir: PathBuf,
    no_replot: bool,
}

struct App {
    models: Vec<GraphModel>,
    states: Vec<GraphState>,
    current_graph: usize,
    cursor_x_idx: usize,
    cursor_series_idx: usize,
    inspection: Option<Inspection>,
    filter_popup: Option<FilterPopup>,
    help_popup: bool,
    raw_index: HashMap<RawKey, Vec<RawSample>>,
    summary_path: PathBuf,
    baseline_path: Option<PathBuf>,
    raw_path: Option<PathBuf>,
    plot_dir: PathBuf,
    warning_message: Option<String>,
    should_quit: bool,
}

impl App {
    fn new(data: BenchData) -> Result<Self> {
        let mut subjects_present = HashSet::new();
        for row in &data.combined_rows {
            subjects_present.insert(row.subject.clone());
        }

        let specs = build_graph_specs(data.suite, data.baseline_path.is_some(), &subjects_present);
        let mut models = Vec::with_capacity(specs.len());
        for spec in specs {
            let source = if spec.data_view == DataView::Current {
                &data.current_rows
            } else {
                &data.combined_rows
            };
            models.push(build_graph_model(spec, source));
        }
        if models.is_empty() {
            bail!("no graph models were generated from the provided data");
        }

        let mut states = Vec::with_capacity(models.len());
        for model in &models {
            let visible_series = model
                .series
                .iter()
                .map(|s| s.name.clone())
                .collect::<BTreeSet<_>>();
            states.push(GraphState { visible_series });
        }

        let warning_message = data
            .baseline_paths_mismatch
            .map(|(current_only, baseline_only)| {
                format!(
                    "Baseline key mismatch: current_only={}, baseline_only={}",
                    current_only, baseline_only
                )
            });

        let mut app = Self {
            models,
            states,
            current_graph: 0,
            cursor_x_idx: 0,
            cursor_series_idx: 0,
            inspection: None,
            filter_popup: None,
            help_popup: false,
            raw_index: data.raw_index,
            summary_path: data.summary_path,
            baseline_path: data.baseline_path,
            raw_path: data.raw_path,
            plot_dir: data.plot_dir,
            warning_message,
            should_quit: false,
        };
        app.clamp_cursor();
        Ok(app)
    }

    fn current_model(&self) -> &GraphModel {
        &self.models[self.current_graph]
    }

    fn current_state(&self) -> &GraphState {
        &self.states[self.current_graph]
    }

    fn current_state_mut(&mut self) -> &mut GraphState {
        &mut self.states[self.current_graph]
    }

    fn visible_series_indices(&self) -> Vec<usize> {
        let state = self.current_state();
        self.ordered_series_indices()
            .into_iter()
            .filter(|idx| {
                let series = &self.current_model().series[*idx];
                state.visible_series.contains(&series.name)
            })
            .collect()
    }

    fn ordered_series_indices(&self) -> Vec<usize> {
        let model = self.current_model();
        let mut indices = (0..model.series.len()).collect::<Vec<_>>();
        indices.sort_by(|a, b| {
            let a_series = &model.series[*a];
            let b_series = &model.series[*b];

            let a_last = a_series.points.last();
            let b_last = b_series.points.last();

            let a_y = a_last.map(|p| p.y).unwrap_or(f64::NEG_INFINITY);
            let b_y = b_last.map(|p| p.y).unwrap_or(f64::NEG_INFINITY);
            let a_x = a_last.map(|p| p.x_bytes).unwrap_or(0);
            let b_x = b_last.map(|p| p.x_bytes).unwrap_or(0);

            b_y.total_cmp(&a_y)
                .then_with(|| b_x.cmp(&a_x))
                .then_with(|| a_series.name.cmp(&b_series.name))
                .then_with(|| a.cmp(b))
        });
        indices
    }

    fn visible_x_values(&self) -> Vec<u64> {
        let mut xs = Vec::new();
        let indices = self.visible_series_indices();
        for idx in indices {
            for point in &self.current_model().series[idx].points {
                xs.push(point.x_bytes);
            }
        }
        xs.sort_unstable();
        xs.dedup();
        xs
    }

    fn clamp_cursor(&mut self) {
        let x_values = self.visible_x_values();
        if x_values.is_empty() {
            self.cursor_x_idx = 0;
        } else {
            self.cursor_x_idx = self.cursor_x_idx.min(x_values.len().saturating_sub(1));
        }

        let visible = self.visible_series_indices();
        if visible.is_empty() {
            self.cursor_series_idx = 0;
        } else {
            self.cursor_series_idx = self.cursor_series_idx.min(visible.len().saturating_sub(1));
        }
    }

    fn move_graph(&mut self, delta: i32) {
        if self.models.is_empty() {
            return;
        }
        let len = self.models.len() as i32;
        let mut next = self.current_graph as i32 + delta;
        if next < 0 {
            next = 0;
        }
        if next >= len {
            next = len - 1;
        }
        self.current_graph = next as usize;
        self.cursor_x_idx = 0;
        self.cursor_series_idx = 0;
        self.inspection = None;
        self.filter_popup = None;
        self.help_popup = false;
        self.clamp_cursor();
    }

    fn move_x_cursor(&mut self, delta: i32) {
        let x_values = self.visible_x_values();
        if x_values.is_empty() {
            return;
        }
        let mut next = self.cursor_x_idx as i32 + delta;
        if next < 0 {
            next = 0;
        }
        let max_idx = x_values.len().saturating_sub(1) as i32;
        if next > max_idx {
            next = max_idx;
        }
        self.cursor_x_idx = next as usize;
    }

    fn move_series_cursor(&mut self, delta: i32) {
        let visible = self.visible_series_indices();
        if visible.is_empty() {
            return;
        }
        let mut next = self.cursor_series_idx as i32 + delta;
        if next < 0 {
            next = 0;
        }
        let max_idx = visible.len().saturating_sub(1) as i32;
        if next > max_idx {
            next = max_idx;
        }
        self.cursor_series_idx = next as usize;
    }

    fn current_x_value(&self) -> Option<u64> {
        let xs = self.visible_x_values();
        xs.get(self.cursor_x_idx).copied()
    }

    fn selected_series_index(&self) -> Option<usize> {
        let visible = self.visible_series_indices();
        visible.get(self.cursor_series_idx).copied()
    }

    fn selected_cursor_point(&self) -> Option<(String, PointMeta)> {
        let x = self.current_x_value()?;
        let series_idx = self.selected_series_index()?;
        let series = &self.current_model().series[series_idx];
        let point = series.points.iter().find(|p| p.x_bytes == x)?.clone();
        Some((series.name.clone(), point))
    }

    fn open_filter_popup(&mut self) {
        let series_names = self
            .ordered_series_indices()
            .into_iter()
            .map(|idx| self.current_model().series[idx].name.clone())
            .collect::<Vec<_>>();
        if series_names.is_empty() {
            return;
        }
        self.filter_popup = Some(FilterPopup {
            series_names,
            selected_idx: 0,
            visible_series: self.current_state().visible_series.clone(),
        });
    }

    fn enter_inspection(&mut self) {
        let Some(x) = self.current_x_value() else {
            self.inspection = None;
            return;
        };

        let mut grouped: BTreeMap<String, Vec<InspectionEntry>> = BTreeMap::new();
        for idx in self.visible_series_indices() {
            let series = &self.current_model().series[idx];
            let Some(point) = series.points.iter().find(|p| p.x_bytes == x) else {
                continue;
            };

            let raw_samples = if point.source == SummarySource::Current {
                self.raw_index
                    .get(&RawKey {
                        operation: point.operation.clone(),
                        subject: point.subject.clone(),
                        size_bytes: point.x_bytes,
                        compression_backend: point.compression_backend.clone(),
                    })
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            grouped
                .entry(format_float(point.y))
                .or_default()
                .push(InspectionEntry {
                    overlap_group: None,
                    series: series.name.clone(),
                    y: point.y,
                    source: point.source,
                    raw_samples,
                });
        }

        if grouped.is_empty() {
            self.inspection = None;
            return;
        }

        let mut entries = Vec::new();
        let mut overlap_id = 1usize;
        for group in grouped.values() {
            let overlap_group = if group.len() > 1 {
                let id = overlap_id;
                overlap_id += 1;
                Some(id)
            } else {
                None
            };
            for entry in group {
                let mut item = entry.clone();
                item.overlap_group = overlap_group;
                entries.push(item);
            }
        }

        entries.sort_by(|a, b| {
            a.y.total_cmp(&b.y)
                .then_with(|| a.series.cmp(&b.series))
                .then_with(|| a.source.as_str().cmp(b.source.as_str()))
        });

        self.inspection = Some(Inspection {
            x_bytes: x,
            metric: self.current_model().spec.metric,
            entries,
        });
    }

    fn clear_inspection(&mut self) {
        self.inspection = None;
    }

    fn apply_filter_popup(&mut self) {
        if let Some(popup) = self.filter_popup.take() {
            self.current_state_mut().visible_series = popup.visible_series;
            self.clamp_cursor();
            self.inspection = None;
        }
    }

    fn handle_filter_key(&mut self, key: KeyEvent) {
        let Some(popup) = self.filter_popup.as_mut() else {
            return;
        };

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.filter_popup = None;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                popup.selected_idx = popup.selected_idx.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if popup.selected_idx + 1 < popup.series_names.len() {
                    popup.selected_idx += 1;
                }
            }
            KeyCode::Char(' ') => {
                if let Some(name) = popup.series_names.get(popup.selected_idx) {
                    if popup.visible_series.contains(name) {
                        popup.visible_series.remove(name);
                    } else {
                        popup.visible_series.insert(name.clone());
                    }
                }
            }
            KeyCode::Char('a') => {
                popup.visible_series = popup.series_names.iter().cloned().collect();
            }
            KeyCode::Char('n') => {
                popup.visible_series.clear();
            }
            KeyCode::Enter => {
                self.apply_filter_popup();
            }
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return;
        }

        if self.filter_popup.is_some() {
            self.handle_filter_key(key);
            return;
        }

        if self.help_popup {
            match key.code {
                KeyCode::Char('q') => self.should_quit = true,
                KeyCode::Char('c') | KeyCode::Esc => self.help_popup = false,
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('[') => self.move_graph(-1),
            KeyCode::Char(']') => self.move_graph(1),
            KeyCode::Char('g') => {
                self.current_graph = 0;
                self.cursor_x_idx = 0;
                self.cursor_series_idx = 0;
                self.inspection = None;
                self.clamp_cursor();
            }
            KeyCode::Char('G') => {
                self.current_graph = self.models.len().saturating_sub(1);
                self.cursor_x_idx = 0;
                self.cursor_series_idx = 0;
                self.inspection = None;
                self.clamp_cursor();
            }
            KeyCode::Left | KeyCode::Char('h') => self.move_x_cursor(-1),
            KeyCode::Right | KeyCode::Char('l') => self.move_x_cursor(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_series_cursor(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_series_cursor(1),
            KeyCode::Char('f') => self.open_filter_popup(),
            KeyCode::Char('?') => self.help_popup = true,
            KeyCode::Enter => self.enter_inspection(),
            KeyCode::Char('c') | KeyCode::Esc => self.clear_inspection(),
            _ => {}
        }
    }
}

enum TuiApp {
    Bench(App),
    LogLoss(LogLossApp),
}

impl TuiApp {
    fn should_quit(&self) -> bool {
        match self {
            Self::Bench(app) => app.should_quit,
            Self::LogLoss(app) => app.should_quit(),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match self {
            Self::Bench(app) => app.handle_key(key),
            Self::LogLoss(app) => app.handle_key(key),
        }
    }

    fn render(&self, frame: &mut Frame<'_>) {
        match self {
            Self::Bench(app) => render_bench(frame, app),
            Self::LogLoss(app) => app.render(frame),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(BenchmanCommand::LogLoss(log_loss_cli)) => {
            let mut app = TuiApp::LogLoss(LogLossApp::from_cli(&log_loss_cli)?);
            run_tui(&mut app)
        }
        None => {
            let inputs = resolve_inputs(cli.bench)?;
            ensure_plot_artifacts(&inputs)?;

            let data = load_bench_data(&inputs)?;
            let mut app = TuiApp::Bench(App::new(data)?);
            run_tui(&mut app)
        }
    }
}

fn resolve_inputs(cli: BenchCli) -> Result<ResolvedInputs> {
    let suite = BenchSuite::parse(&cli.suite)?;
    let summary_path = match cli.summary_tsv {
        Some(path) => path,
        None => latest_summary_tsv(suite)?.context(format!(
            "no summary TSV provided and no /tmp/{}-summary-*.tsv found",
            suite.summary_prefix()
        ))?,
    };
    if !summary_path.is_file() {
        bail!("summary TSV not found: {}", summary_path.display());
    }

    let baseline_path = match cli.baseline_summary_tsv {
        Some(path) => {
            if !path.is_file() {
                bail!("baseline summary TSV not found: {}", path.display());
            }
            Some(path)
        }
        None => None,
    };

    let derived_raw = derive_raw_path(&summary_path);
    let raw_path = match cli.raw_tsv {
        Some(path) => {
            if !path.is_file() {
                bail!("raw TSV not found: {}", path.display());
            }
            Some(path)
        }
        None => derived_raw.filter(|path| path.is_file()),
    };

    let subject_filter = parse_subject_filter(cli.subjects.as_deref())?;

    Ok(ResolvedInputs {
        suite,
        summary_path,
        baseline_path,
        raw_path,
        subjects_raw: cli.subjects,
        subject_filter,
        plot_dir: cli.plot_dir,
        no_replot: cli.no_replot,
    })
}

fn latest_summary_tsv(suite: BenchSuite) -> Result<Option<PathBuf>> {
    let expected_prefix = format!("{}-summary-", suite.summary_prefix());
    let mut candidates = Vec::new();
    for entry in fs::read_dir("/tmp").context("failed to list /tmp")? {
        let entry = entry?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if !(file_name.starts_with(&expected_prefix) && file_name.ends_with(".tsv")) {
            continue;
        }
        let modified = entry.metadata()?.modified()?;
        candidates.push((modified, path));
    }

    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(candidates.into_iter().next().map(|(_, path)| path))
}

fn derive_raw_path(summary_path: &Path) -> Option<PathBuf> {
    let file_name = summary_path.file_name()?.to_string_lossy();
    let parent = summary_path.parent()?;
    if file_name.contains("-summary-") {
        Some(parent.join(file_name.replace("-summary-", "-raw-")))
    } else if file_name.ends_with(".tsv") {
        Some(parent.join(file_name.replace(".tsv", ".raw.tsv")))
    } else {
        None
    }
}

fn parse_subject_filter(raw: Option<&str>) -> Result<Option<BTreeSet<String>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let mut selected = BTreeSet::new();
    for token in raw
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        selected.insert(canonicalize_subject(token).to_string());
    }
    if selected.is_empty() {
        bail!("subject filter was provided but contained no subjects");
    }
    Ok(Some(selected))
}

fn canonicalize_subject(subject: &str) -> &str {
    if subject == "rwkv" { "rwkv7" } else { subject }
}

fn ensure_plot_artifacts(inputs: &ResolvedInputs) -> Result<()> {
    let plot_dir = &inputs.plot_dir;
    let should_rebuild = if plot_dir.exists() {
        prompt_plot_dir_decision(plot_dir)?
    } else {
        true
    };

    if should_rebuild {
        clear_plot_dir_for_rebuild(
            plot_dir,
            if inputs.no_replot {
                "remove existing plot dir while --no-replot is set"
            } else {
                "clear plot dir before regenerating plots"
            },
        )?;

        if inputs.no_replot {
            fs::create_dir_all(plot_dir)
                .with_context(|| format!("failed to create {}", plot_dir.display()))?;
            write_plot_dir_marker(plot_dir)?;
        } else {
            run_plot_script(inputs)?;
            write_plot_dir_marker(plot_dir)?;
        }
    } else if !plot_dir.exists() {
        fs::create_dir_all(plot_dir)
            .with_context(|| format!("failed to create {}", plot_dir.display()))?;
        write_plot_dir_marker(plot_dir)?;
    }

    Ok(())
}

fn clear_plot_dir_for_rebuild(plot_dir: &Path, context: &str) -> Result<()> {
    if !plot_dir.exists() {
        return Ok(());
    }
    if !plot_dir.is_dir() {
        bail!("plot dir path is not a directory: {}", plot_dir.display());
    }
    if plot_dir.parent().is_none() {
        bail!(
            "refusing to delete filesystem root as plot dir: {}",
            plot_dir.display()
        );
    }

    match plot_dir_deletion_policy(plot_dir)? {
        PlotDirDeletionPolicy::ManagedMarker | PlotDirDeletionPolicy::LegacyArtifacts => {}
        PlotDirDeletionPolicy::RequiresExplicitConfirmation => {
            confirm_unmanaged_plot_dir_deletion(plot_dir)?;
        }
    }

    fs::remove_dir_all(plot_dir)
        .with_context(|| format!("failed to {context}: {}", plot_dir.display()))?;
    Ok(())
}

fn plot_dir_deletion_policy(plot_dir: &Path) -> Result<PlotDirDeletionPolicy> {
    if plot_dir_marker_path(plot_dir).is_file() {
        return Ok(PlotDirDeletionPolicy::ManagedMarker);
    }

    let mut saw_legacy_artifact = false;
    for entry in fs::read_dir(plot_dir)
        .with_context(|| format!("failed to inspect {}", plot_dir.display()))?
    {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            return Ok(PlotDirDeletionPolicy::RequiresExplicitConfirmation);
        };
        if file_name == PLOT_DIR_MARKER_FILE {
            continue;
        }

        if entry.file_type()?.is_file() && is_legacy_plot_svg_name(file_name) {
            saw_legacy_artifact = true;
            continue;
        }
        return Ok(PlotDirDeletionPolicy::RequiresExplicitConfirmation);
    }

    if saw_legacy_artifact {
        Ok(PlotDirDeletionPolicy::LegacyArtifacts)
    } else {
        Ok(PlotDirDeletionPolicy::RequiresExplicitConfirmation)
    }
}

fn is_legacy_plot_svg_name(file_name: &str) -> bool {
    file_name.ends_with(LEGACY_PLOT_SVG_SUFFIX)
        && [BenchSuite::TwoJson, BenchSuite::OneSse, BenchSuite::Extra]
            .into_iter()
            .any(|suite| {
                let expected_prefix = format!("{}-", suite.summary_prefix());
                file_name.starts_with(&expected_prefix)
            })
}

fn confirm_unmanaged_plot_dir_deletion(plot_dir: &Path) -> Result<()> {
    println!(
        "[benchman] Plot directory {} is not benchman-managed.",
        plot_dir.display()
    );
    println!("It does not match legacy plot artifacts and might contain unrelated files.");
    println!("Type the full path to confirm deletion, or press Enter to cancel rebuild.");

    print!("Confirm deletion of {}: ", plot_dir.display());
    io::stdout().flush().context("failed to flush stdout")?;

    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .context("failed to read deletion confirmation")?;

    if line.trim() == plot_dir.to_string_lossy() {
        return Ok(());
    }

    bail!(
        "aborted rebuild because deletion confirmation did not match {}",
        plot_dir.display()
    );
}

fn plot_dir_marker_path(plot_dir: &Path) -> PathBuf {
    plot_dir.join(PLOT_DIR_MARKER_FILE)
}

fn write_plot_dir_marker(plot_dir: &Path) -> Result<()> {
    let marker_path = plot_dir_marker_path(plot_dir);
    fs::write(&marker_path, "managed by benchman\n")
        .with_context(|| format!("failed to write marker {}", marker_path.display()))
}

fn prompt_plot_dir_decision(plot_dir: &Path) -> Result<bool> {
    println!(
        "[benchman] Plot directory already exists: {}",
        plot_dir.display()
    );
    println!("A) Use it (graphs are already up to date)");
    println!("B) Remove it and reproduce graphs");

    loop {
        print!("Choose [A/B] (default A): ");
        io::stdout().flush().context("failed to flush stdout")?;
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .context("failed to read A/B choice")?;
        let choice = line.trim().to_ascii_lowercase();
        match choice.as_str() {
            "" | "a" => return Ok(false),
            "b" => return Ok(true),
            _ => {
                println!("Please enter A or B.");
            }
        }
    }
}

fn run_plot_script(inputs: &ResolvedInputs) -> Result<()> {
    let repo_root = find_repo_root()?;
    let script = repo_root.join("scripts/plot_two_json.sh");
    if !script.is_file() {
        bail!("plot script not found at {}", script.display());
    }

    println!(
        "[benchman] Reproducing plot artifacts via {}",
        script.display()
    );

    let mut cmd = Command::new("sh");
    cmd.arg(&script)
        .env("INFOTHEORY_PLOT_SUMMARY_TSV", &inputs.summary_path)
        .env("INFOTHEORY_PLOT_SUITE", inputs.suite.as_str())
        .env("INFOTHEORY_PLOT_OUTPUT_DIR", &inputs.plot_dir);

    if let Some(path) = &inputs.baseline_path {
        cmd.env("INFOTHEORY_BASELINE_SUMMARY_TSV", path);
    }
    if let Some(subjects) = &inputs.subjects_raw {
        cmd.env("INFOTHEORY_PLOT_SUBJECTS", subjects);
    }

    let status = cmd
        .status()
        .with_context(|| format!("failed to run {}", script.display()))?;
    if !status.success() {
        bail!("plot regeneration failed with exit status {status}");
    }

    Ok(())
}

fn find_repo_root() -> Result<PathBuf> {
    let from_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in from_manifest.ancestors().skip(1) {
        if candidate.join("scripts/plot_two_json.sh").is_file()
            && candidate.join("projman.sh").is_file()
        {
            return Ok(candidate.to_path_buf());
        }
    }

    let mut cursor = std::env::current_dir().context("failed to resolve current directory")?;
    loop {
        if cursor.join("scripts/plot_two_json.sh").is_file() && cursor.join("projman.sh").is_file()
        {
            return Ok(cursor);
        }
        if !cursor.pop() {
            break;
        }
    }

    bail!("failed to locate repository root (expected scripts/plot_two_json.sh)")
}

fn load_bench_data(inputs: &ResolvedInputs) -> Result<BenchData> {
    let current_rows = load_summary_rows(
        &inputs.summary_path,
        SummarySource::Current,
        inputs.subject_filter.as_ref(),
    )?;
    if current_rows.is_empty() {
        bail!(
            "summary TSV {} produced no rows (check subject filters)",
            inputs.summary_path.display()
        );
    }

    let baseline_rows = if let Some(path) = &inputs.baseline_path {
        load_summary_rows(
            path,
            SummarySource::Baseline,
            inputs.subject_filter.as_ref(),
        )?
    } else {
        Vec::new()
    };

    let baseline_paths_mismatch = if !baseline_rows.is_empty() {
        let current_keys = current_rows.iter().map(summary_key).collect::<HashSet<_>>();
        let baseline_keys = baseline_rows
            .iter()
            .map(summary_key)
            .collect::<HashSet<_>>();

        let current_only = current_keys.difference(&baseline_keys).count();
        let baseline_only = baseline_keys.difference(&current_keys).count();
        (current_only != 0 || baseline_only != 0).then_some((current_only, baseline_only))
    } else {
        None
    };

    let mut combined_rows = Vec::with_capacity(current_rows.len() + baseline_rows.len());
    combined_rows.extend(baseline_rows);
    combined_rows.extend(current_rows.clone());

    let raw_index = if let Some(path) = &inputs.raw_path {
        load_raw_rows(path, inputs.subject_filter.as_ref())?
    } else {
        HashMap::new()
    };

    Ok(BenchData {
        suite: inputs.suite,
        current_rows,
        combined_rows,
        baseline_paths_mismatch,
        raw_index,
        summary_path: inputs.summary_path.clone(),
        baseline_path: inputs.baseline_path.clone(),
        raw_path: inputs.raw_path.clone(),
        plot_dir: inputs.plot_dir.clone(),
    })
}

fn summary_key(row: &RenderRow) -> (String, String, u64, String) {
    (
        row.operation.clone(),
        row.subject.clone(),
        row.size_bytes,
        row.compression_backend.clone(),
    )
}

fn load_summary_rows(
    path: &Path,
    source: SummarySource,
    subject_filter: Option<&BTreeSet<String>>,
) -> Result<Vec<RenderRow>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open summary TSV {}", path.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("failed to read header from {}", path.display()))?
        .clone();

    let idx_operation = header_index(&headers, "operation")?;
    let idx_subject = header_index(&headers, "subject")?;
    let idx_series = header_index(&headers, "series")?;
    let idx_size_bytes = header_index(&headers, "size_bytes")?;
    let idx_compression_backend = header_index(&headers, "compression_backend")?;
    let idx_real_seconds_median = header_index(&headers, "real_seconds_median")?;
    let idx_rss_kib_median = header_index(&headers, "rss_kib_median")?;
    let idx_entropy_bpb_median = header_index(&headers, "entropy_bpb_median")?;
    let idx_archive_ratio_median = header_index(&headers, "archive_ratio_median")?;

    let mut rows = Vec::new();
    for (row_idx, row) in reader.records().enumerate() {
        let row = row.with_context(|| {
            format!(
                "failed to parse summary row {} in {}",
                row_idx + 2,
                path.display()
            )
        })?;

        let subject = canonicalize_subject(get_field(&row, idx_subject).trim()).to_string();
        if let Some(selected) = subject_filter
            && !selected.contains(&subject)
        {
            continue;
        }

        let operation = get_field(&row, idx_operation).trim().to_string();
        let series = get_field(&row, idx_series)
            .trim()
            .replace(":rwkv", ":rwkv7");
        let size_bytes = parse_size_bytes(get_field(&row, idx_size_bytes), row_idx + 2, path)?;

        let row = RenderRow {
            source,
            operation,
            subject,
            series,
            size_bytes,
            compression_backend: get_field(&row, idx_compression_backend).trim().to_string(),
            real_seconds_median: parse_opt_f64(
                get_field(&row, idx_real_seconds_median),
                "real_seconds_median",
                row_idx + 2,
                path,
            )?,
            rss_kib_median: parse_opt_f64(
                get_field(&row, idx_rss_kib_median),
                "rss_kib_median",
                row_idx + 2,
                path,
            )?,
            entropy_bpb_median: parse_opt_f64(
                get_field(&row, idx_entropy_bpb_median),
                "entropy_bpb_median",
                row_idx + 2,
                path,
            )?,
            archive_ratio_median: parse_opt_f64(
                get_field(&row, idx_archive_ratio_median),
                "archive_ratio_median",
                row_idx + 2,
                path,
            )?,
        };

        rows.push(row);
    }

    Ok(rows)
}

fn load_raw_rows(
    path: &Path,
    subject_filter: Option<&BTreeSet<String>>,
) -> Result<HashMap<RawKey, Vec<RawSample>>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open raw TSV {}", path.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("failed to read header from {}", path.display()))?
        .clone();

    let idx_operation = header_index(&headers, "operation")?;
    let idx_subject = header_index(&headers, "subject")?;
    let idx_size_bytes = header_index(&headers, "size_bytes")?;
    let idx_repetition = header_index(&headers, "repetition")?;
    let idx_compression_backend = header_index(&headers, "compression_backend")?;
    let idx_real_seconds = header_index(&headers, "real_seconds")?;
    let idx_user_seconds = header_index(&headers, "user_seconds")?;
    let idx_sys_seconds = header_index(&headers, "sys_seconds")?;
    let idx_rss_kib = header_index(&headers, "rss_kib")?;
    let idx_archive_bytes = header_index(&headers, "archive_bytes")?;
    let idx_entropy_bpb = header_index(&headers, "entropy_bpb")?;

    let mut index: HashMap<RawKey, Vec<RawSample>> = HashMap::new();

    for (row_idx, row) in reader.records().enumerate() {
        let row = row.with_context(|| {
            format!(
                "failed to parse raw row {} in {}",
                row_idx + 2,
                path.display()
            )
        })?;

        let subject = canonicalize_subject(get_field(&row, idx_subject).trim()).to_string();
        if let Some(selected) = subject_filter
            && !selected.contains(&subject)
        {
            continue;
        }

        let key = RawKey {
            operation: get_field(&row, idx_operation).trim().to_string(),
            subject,
            size_bytes: parse_size_bytes(get_field(&row, idx_size_bytes), row_idx + 2, path)?,
            compression_backend: get_field(&row, idx_compression_backend).trim().to_string(),
        };

        let sample = RawSample {
            repetition: parse_u32(
                get_field(&row, idx_repetition),
                "repetition",
                row_idx + 2,
                path,
            )?,
            real_seconds: parse_required_f64(
                get_field(&row, idx_real_seconds),
                "real_seconds",
                row_idx + 2,
                path,
            )?,
            user_seconds: parse_required_f64(
                get_field(&row, idx_user_seconds),
                "user_seconds",
                row_idx + 2,
                path,
            )?,
            sys_seconds: parse_required_f64(
                get_field(&row, idx_sys_seconds),
                "sys_seconds",
                row_idx + 2,
                path,
            )?,
            rss_kib: parse_required_f64(
                get_field(&row, idx_rss_kib),
                "rss_kib",
                row_idx + 2,
                path,
            )?,
            archive_bytes: parse_opt_u64(
                get_field(&row, idx_archive_bytes),
                "archive_bytes",
                row_idx + 2,
                path,
            )?,
            entropy_bpb: parse_opt_f64(
                get_field(&row, idx_entropy_bpb),
                "entropy_bpb",
                row_idx + 2,
                path,
            )?,
        };

        index.entry(key).or_default().push(sample);
    }

    for samples in index.values_mut() {
        samples.sort_by_key(|s| s.repetition);
    }

    Ok(index)
}

fn header_index(headers: &csv::StringRecord, column: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h == column)
        .with_context(|| format!("required TSV column not found: {column}"))
}

fn get_field(row: &csv::StringRecord, idx: usize) -> &str {
    row.get(idx).unwrap_or("")
}

fn parse_u64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<u64> {
    let value = raw.trim().parse::<u64>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })?;
    Ok(value)
}

fn parse_positive_u64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<u64> {
    let value = parse_u64(raw, field, row_no, path)?;
    if value == 0 {
        bail!(
            "{} row {}: {} must be > 0, got {:?}",
            path.display(),
            row_no,
            field,
            raw
        );
    }
    Ok(value)
}

fn parse_size_bytes(raw: &str, row_no: usize, path: &Path) -> Result<u64> {
    parse_positive_u64(raw, "size_bytes", row_no, path)
}

fn parse_u32(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<u32> {
    let value = raw.trim().parse::<u32>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })?;
    Ok(value)
}

fn parse_required_f64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<f64> {
    let value = raw.trim().parse::<f64>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })?;
    if !value.is_finite() {
        bail!(
            "{} row {}: non-finite {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        );
    }
    Ok(value)
}

fn parse_opt_f64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<Option<f64>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value = parse_required_f64(trimmed, field, row_no, path)?;
    Ok(Some(value))
}

fn parse_opt_u64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<Option<u64>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(parse_u64(trimmed, field, row_no, path)?))
}

fn build_graph_specs(
    suite: BenchSuite,
    baseline_available: bool,
    subjects_present: &HashSet<String>,
) -> Vec<GraphSpec> {
    let suite_label = suite.spec_label();
    let mut specs = vec![
        spec_current(
            "h-rss",
            &format!("{suite_label} h RSS vs size"),
            Some("h"),
            Metric::RssKibMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "compress-rss",
            &format!("{suite_label} compress RSS vs size"),
            Some("compress"),
            Metric::RssKibMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "decompress-rss",
            &format!("{suite_label} decompress RSS vs size"),
            Some("decompress"),
            Metric::RssKibMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "h-time",
            &format!("{suite_label} h wall time vs size"),
            Some("h"),
            Metric::RealSecondsMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "compress-time",
            &format!("{suite_label} compress wall time vs size"),
            Some("compress"),
            Metric::RealSecondsMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "decompress-time",
            &format!("{suite_label} decompress wall time vs size"),
            Some("decompress"),
            Metric::RealSecondsMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "h-entropy",
            &format!("{suite_label} h bits per byte vs size"),
            Some("h"),
            Metric::EntropyBpbMedian,
            SeriesField::Subject,
        ),
        spec_current(
            "all-time",
            &format!("{suite_label} all operations wall time vs size"),
            None,
            Metric::RealSecondsMedian,
            SeriesField::Series,
        ),
        spec_current(
            "all-rss",
            &format!("{suite_label} all operations RSS vs size"),
            None,
            Metric::RssKibMedian,
            SeriesField::Series,
        ),
    ];

    if baseline_available {
        specs.extend([
            spec_combined(
                "h-rss-baseline",
                &format!("{suite_label} h RSS vs size (current vs baseline)"),
                Some("h"),
                Metric::RssKibMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "compress-rss-baseline",
                &format!("{suite_label} compress RSS vs size (current vs baseline)"),
                Some("compress"),
                Metric::RssKibMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "decompress-rss-baseline",
                &format!("{suite_label} decompress RSS vs size (current vs baseline)"),
                Some("decompress"),
                Metric::RssKibMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "h-time-baseline",
                &format!("{suite_label} h wall time vs size (current vs baseline)"),
                Some("h"),
                Metric::RealSecondsMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "compress-time-baseline",
                &format!("{suite_label} compress wall time vs size (current vs baseline)"),
                Some("compress"),
                Metric::RealSecondsMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "decompress-time-baseline",
                &format!("{suite_label} decompress wall time vs size (current vs baseline)"),
                Some("decompress"),
                Metric::RealSecondsMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "h-entropy-baseline",
                &format!("{suite_label} h bits per byte vs size (current vs baseline)"),
                Some("h"),
                Metric::EntropyBpbMedian,
                SeriesField::SubjectOverlay,
            ),
            spec_combined(
                "all-time-baseline",
                &format!("{suite_label} all operations wall time vs size (current vs baseline)"),
                None,
                Metric::RealSecondsMedian,
                SeriesField::SeriesOverlay,
            ),
            spec_combined(
                "all-rss-baseline",
                &format!("{suite_label} all operations RSS vs size (current vs baseline)"),
                None,
                Metric::RssKibMedian,
                SeriesField::SeriesOverlay,
            ),
        ]);

        for subject in suite.focus_subjects() {
            if !subjects_present.contains(*subject) {
                continue;
            }
            specs.extend([
                GraphSpec {
                    id: format!("{}-h-time-baseline", subject),
                    title: format!(
                        "{suite_label} {} h wall time vs size (current vs baseline)",
                        subject
                    ),
                    operation_filter: Some("h".to_string()),
                    subject_filter: Some((*subject).to_string()),
                    metric: Metric::RealSecondsMedian,
                    series_field: SeriesField::SummarySource,
                    data_view: DataView::Combined,
                },
                GraphSpec {
                    id: format!("{}-h-entropy-baseline", subject),
                    title: format!(
                        "{suite_label} {} h bits per byte vs size (current vs baseline)",
                        subject
                    ),
                    operation_filter: Some("h".to_string()),
                    subject_filter: Some((*subject).to_string()),
                    metric: Metric::EntropyBpbMedian,
                    series_field: SeriesField::SummarySource,
                    data_view: DataView::Combined,
                },
                GraphSpec {
                    id: format!("{}-compress-time-baseline", subject),
                    title: format!(
                        "{suite_label} {} compress wall time vs size (current vs baseline)",
                        subject
                    ),
                    operation_filter: Some("compress".to_string()),
                    subject_filter: Some((*subject).to_string()),
                    metric: Metric::RealSecondsMedian,
                    series_field: SeriesField::SummarySource,
                    data_view: DataView::Combined,
                },
                GraphSpec {
                    id: format!("{}-compress-archive-ratio-baseline", subject),
                    title: format!(
                        "{suite_label} {} compress archive ratio vs size (current vs baseline)",
                        subject
                    ),
                    operation_filter: Some("compress".to_string()),
                    subject_filter: Some((*subject).to_string()),
                    metric: Metric::ArchiveRatioMedian,
                    series_field: SeriesField::SummarySource,
                    data_view: DataView::Combined,
                },
                GraphSpec {
                    id: format!("{}-decompress-time-baseline", subject),
                    title: format!(
                        "{suite_label} {} decompress wall time vs size (current vs baseline)",
                        subject
                    ),
                    operation_filter: Some("decompress".to_string()),
                    subject_filter: Some((*subject).to_string()),
                    metric: Metric::RealSecondsMedian,
                    series_field: SeriesField::SummarySource,
                    data_view: DataView::Combined,
                },
            ]);
        }
    }

    specs
}

fn spec_current(
    id: &str,
    title: &str,
    operation_filter: Option<&str>,
    metric: Metric,
    series_field: SeriesField,
) -> GraphSpec {
    GraphSpec {
        id: id.to_string(),
        title: title.to_string(),
        operation_filter: operation_filter.map(str::to_string),
        subject_filter: None,
        metric,
        series_field,
        data_view: DataView::Current,
    }
}

fn spec_combined(
    id: &str,
    title: &str,
    operation_filter: Option<&str>,
    metric: Metric,
    series_field: SeriesField,
) -> GraphSpec {
    GraphSpec {
        id: id.to_string(),
        title: title.to_string(),
        operation_filter: operation_filter.map(str::to_string),
        subject_filter: None,
        metric,
        series_field,
        data_view: DataView::Combined,
    }
}

fn build_graph_model(spec: GraphSpec, rows: &[RenderRow]) -> GraphModel {
    let mut grouped: BTreeMap<String, Vec<PointMeta>> = BTreeMap::new();

    for row in rows {
        if let Some(operation) = &spec.operation_filter
            && row.operation != *operation
        {
            continue;
        }
        if let Some(subject) = &spec.subject_filter
            && row.subject != *subject
        {
            continue;
        }

        let Some(y) = row.metric_value(spec.metric) else {
            continue;
        };
        if !y.is_finite() {
            continue;
        }

        let series_label = match spec.series_field {
            SeriesField::Subject => row.subject.clone(),
            SeriesField::Series => row.series.clone(),
            SeriesField::SubjectOverlay => row.subject_overlay(),
            SeriesField::SeriesOverlay => row.series_overlay(),
            SeriesField::SummarySource => row.source.as_str().to_string(),
        };

        let point = PointMeta {
            x_bytes: row.size_bytes,
            x_log10: (row.size_bytes as f64).log10(),
            y,
            operation: row.operation.clone(),
            subject: row.subject.clone(),
            compression_backend: row.compression_backend.clone(),
            source: row.source,
        };

        grouped.entry(series_label).or_default().push(point);
    }

    let mut series = Vec::with_capacity(grouped.len());
    for (idx, (name, mut points)) in grouped.into_iter().enumerate() {
        points.sort_by_key(|p| p.x_bytes);
        series.push(SeriesData {
            name,
            color: COLOR_PALETTE[idx % COLOR_PALETTE.len()],
            points,
        });
    }

    GraphModel { spec, series }
}

fn run_tui(app: &mut TuiApp) -> Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let loop_result = event_loop(&mut terminal, app);

    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen")?;
    terminal.show_cursor().context("failed to restore cursor")?;

    loop_result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
) -> Result<()> {
    while !app.should_quit() {
        terminal
            .draw(|frame| app.render(frame))
            .context("failed to draw TUI frame")?;

        if event::poll(Duration::from_millis(200)).context("failed to poll terminal events")?
            && let Event::Key(key) = event::read().context("failed to read terminal event")?
        {
            app.handle_key(key);
        }
    }

    Ok(())
}

fn render_bench(frame: &mut Frame<'_>, app: &App) {
    let root = frame.area();
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(5)])
        .split(root);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(55),
            Constraint::Percentage(20),
        ])
        .split(vertical[0]);

    render_graph_list(frame, top[0], app);
    render_chart(frame, top[1], app);
    render_series_panel(frame, top[2], app);
    render_status(frame, vertical[1], app);

    if let Some(popup) = &app.filter_popup {
        render_filter_popup(frame, popup);
    }
    if app.help_popup {
        render_help_popup(frame);
    } else if let Some(inspection) = &app.inspection {
        render_inspection_popup(frame, inspection);
    }
}

fn render_graph_list(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let items = app
        .models
        .iter()
        .enumerate()
        .map(|(idx, model)| {
            let prefix = if idx == app.current_graph { ">" } else { " " };
            ListItem::new(format!("{} {}", prefix, model.spec.title))
        })
        .collect::<Vec<_>>();

    let mut state = ListState::default();
    state.select(Some(app.current_graph));

    let list = List::new(items)
        .block(Block::default().title("Graphs").borders(Borders::ALL))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_stateful_widget(list, area, &mut state);
}

fn render_chart(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let model = app.current_model();
    let visible_indices = app.visible_series_indices();

    let visible_series = visible_indices
        .iter()
        .map(|idx| &model.series[*idx])
        .collect::<Vec<_>>();

    let data_store: Vec<Vec<(f64, f64)>> = visible_series
        .iter()
        .map(|series| {
            series
                .points
                .iter()
                .map(|point| (point.x_log10, point.y))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut datasets = Vec::new();

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let x_values = app.visible_x_values();

    for series in &visible_series {
        for point in &series.points {
            min_x = min_x.min(point.x_log10);
            max_x = max_x.max(point.x_log10);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }
    }

    for (series, points) in visible_series.iter().zip(data_store.iter()) {
        datasets.push(
            Dataset::default()
                .name(series.name.clone())
                .graph_type(GraphType::Line)
                .marker(Marker::Braille)
                .style(Style::default().fg(series.color))
                .data(points.as_slice()),
        );
    }

    let mut cursor_store: Option<Vec<(f64, f64)>> = None;
    if let Some((_, point)) = app.selected_cursor_point() {
        cursor_store = Some(vec![(point.x_log10, point.y)]);
    }
    if let Some(points) = cursor_store.as_ref() {
        datasets.push(
            Dataset::default()
                .name("cursor")
                .graph_type(GraphType::Scatter)
                .marker(Marker::Dot)
                .style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
                .data(points.as_slice()),
        );
    }

    let (x_bounds, x_labels) = if x_values.is_empty() || !min_x.is_finite() || !max_x.is_finite() {
        (
            [0.0, 1.0],
            vec![
                Span::raw("0"),
                Span::raw("size (bytes, log10)"),
                Span::raw("1"),
            ],
        )
    } else {
        let (mut lo, mut hi) = (min_x, max_x);
        if (hi - lo).abs() < f64::EPSILON {
            lo -= 0.5;
            hi += 0.5;
        }

        let min_label = x_values
            .first()
            .map(|v| format_size_bytes(*v))
            .unwrap_or_else(|| "-".to_string());
        let mid_label = x_values
            .get(x_values.len() / 2)
            .map(|v| format_size_bytes(*v))
            .unwrap_or_else(|| "-".to_string());
        let max_label = x_values
            .last()
            .map(|v| format_size_bytes(*v))
            .unwrap_or_else(|| "-".to_string());

        (
            [lo, hi],
            vec![
                Span::raw(min_label),
                Span::raw(mid_label),
                Span::raw(max_label),
            ],
        )
    };

    let (y_bounds, y_labels) = if !min_y.is_finite() || !max_y.is_finite() {
        (
            [0.0, 1.0],
            vec![Span::raw("0"), Span::raw("0.5"), Span::raw("1")],
        )
    } else {
        let mut lo = min_y;
        let mut hi = max_y;
        if (hi - lo).abs() < f64::EPSILON {
            let pad = if hi.abs() < 1.0 { 1.0 } else { hi.abs() * 0.1 };
            lo -= pad;
            hi += pad;
        } else {
            let pad = (hi - lo) * 0.08;
            lo -= pad;
            hi += pad;
        }
        let mid = (lo + hi) / 2.0;
        (
            [lo, hi],
            vec![
                Span::raw(format_float(lo)),
                Span::raw(format_float(mid)),
                Span::raw(format_float(hi)),
            ],
        )
    };

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(model.spec.title.clone())
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("size (bytes, log10)")
                .bounds(x_bounds)
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title(model.spec.metric.y_label())
                .bounds(y_bounds)
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

fn render_series_panel(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let model = app.current_model();
    let state = app.current_state();
    let x_value = app.current_x_value();
    let selected_series = app.selected_series_index();
    let ordered_indices = app.ordered_series_indices();

    let mut items = Vec::new();
    for idx in ordered_indices {
        let series = &model.series[idx];
        let visible = state.visible_series.contains(&series.name);
        let mark = if visible { "[x]" } else { "[ ]" };
        let y_text = match x_value {
            Some(x) => series
                .points
                .iter()
                .find(|p| p.x_bytes == x)
                .map(|p| format_float(p.y))
                .unwrap_or_else(|| "n/a".to_string()),
            None => "n/a".to_string(),
        };

        let mut style = Style::default().fg(series.color);
        if Some(idx) == selected_series {
            style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
        }

        items.push(ListItem::new(format!("{} {}  y={}", mark, series.name, y_text)).style(style));
    }

    if items.is_empty() {
        items.push(ListItem::new("No series"));
    }

    let title = format!(
        "Series ({}/{})",
        state.visible_series.len(),
        model.series.len()
    );

    let list = List::new(items).block(Block::default().title(title).borders(Borders::ALL));
    frame.render_widget(list, area);
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let graph = app.current_model();
    let x = app
        .current_x_value()
        .map(format_size_bytes)
        .unwrap_or_else(|| "n/a".to_string());

    let selected = app
        .selected_cursor_point()
        .map(|(name, point)| {
            format!(
                "{} => {}={}",
                name,
                graph.spec.metric.key_name(),
                format_float(point.y)
            )
        })
        .unwrap_or_else(|| "no point selected".to_string());

    let mut lines = vec![
        Line::from(format!(
            "Graph {}/{} [{}]    x={}    {}",
            app.current_graph + 1,
            app.models.len(),
            graph.spec.id,
            x,
            selected
        )),
        Line::from(format!(
            "summary={}  baseline={}  raw={}  plot_dir={}",
            app.summary_path.display(),
            app.baseline_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            app.raw_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            app.plot_dir.display()
        )),
        Line::from(
            "Keys: [ ] graph | g/G first/last | h/j/k/l + arrows cursor | Enter inspect | c clear | f focus subjects | q quit",
        ),
    ];

    if let Some(msg) = &app.warning_message {
        lines.push(Line::from(Span::styled(
            msg.clone(),
            Style::default().fg(Color::Yellow),
        )));
    }

    let block = Paragraph::new(lines)
        .block(Block::default().title("Status").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    frame.render_widget(block, area);
}

fn render_filter_popup(frame: &mut Frame<'_>, popup: &FilterPopup) {
    let area = centered_rect(70, 70, frame.area());
    frame.render_widget(Clear, area);

    let items = popup
        .series_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let selected = popup.visible_series.contains(name);
            let marker = if selected { "[x]" } else { "[ ]" };
            let mut text = format!("{} {}", marker, name);
            if idx == popup.selected_idx {
                text = format!("> {}", text);
            } else {
                text = format!("  {}", text);
            }
            ListItem::new(text)
        })
        .collect::<Vec<_>>();

    let mut state = ListState::default();
    state.select(Some(popup.selected_idx));

    let list = List::new(items)
        .block(
            Block::default()
                .title("Focus Series (space toggle, a all, n none, Enter apply, Esc cancel)")
                .borders(Borders::ALL),
        )
        .highlight_style(Style::default().fg(Color::Cyan));

    frame.render_stateful_widget(list, area, &mut state);
}

fn render_help_popup(frame: &mut Frame<'_>) {
    let area = centered_rect(78, 40, frame.area());
    frame.render_widget(Clear, area);

    let lines = vec![
        Line::from(
            "Keys: [ ] graph | g/G first/last | h/j/k/l + arrows cursor | Enter inspect | c clear | f focus subjects | q quit",
        ),
        Line::from("press c or Esc to clear this panel."),
    ];

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("Help").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_inspection_popup(frame: &mut Frame<'_>, inspection: &Inspection) {
    let area = centered_rect(78, 75, frame.area());
    frame.render_widget(Clear, area);

    let mut lines = vec![Line::from(format!(
        "x = {} bytes   metric = {}",
        format_size_bytes(inspection.x_bytes),
        inspection.metric.key_name()
    ))];

    for entry in &inspection.entries {
        let overlap = entry
            .overlap_group
            .map(|id| format!("[overlap #{}] ", id))
            .unwrap_or_default();
        lines.push(Line::from(format!(
            "{}{} ({}) -> y={} at x={}",
            overlap,
            entry.series,
            entry.source.as_str(),
            format_float(entry.y),
            format_size_bytes(inspection.x_bytes)
        )));

        if !entry.raw_samples.is_empty() {
            let reps = entry
                .raw_samples
                .iter()
                .map(|sample| {
                    let entropy = sample
                        .entropy_bpb
                        .map(format_float)
                        .unwrap_or_else(|| "-".to_string());
                    let archive = sample
                        .archive_bytes
                        .map(format_size_bytes)
                        .unwrap_or_else(|| "-".to_string());
                    format!(
                        "r{}: real={} user={} sys={} rss={}KiB archive={} entropy={}",
                        sample.repetition,
                        format_float(sample.real_seconds),
                        format_float(sample.user_seconds),
                        format_float(sample.sys_seconds),
                        format_float(sample.rss_kib),
                        archive,
                        entropy
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            lines.push(Line::from(format!("  raw: {}", reps)));
        }
    }

    if inspection.entries.is_empty() {
        lines.push(Line::from("No values found at current cursor x"));
    }

    lines.push(Line::from("Press c or Esc to clear this panel."));

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .title("Point Inspector")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

pub(crate) fn format_float(value: f64) -> String {
    let abs = value.abs();
    if abs != 0.0 && !(1e-4..1e6).contains(&abs) {
        return format!("{value:.6e}");
    }

    let mut out = format!("{value:.12}");
    while out.contains('.') && out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    if out == "-0" {
        out = "0".to_string();
    }
    out
}

pub(crate) fn format_size_bytes(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (idx, ch) in s.chars().rev().enumerate() {
        if idx != 0 && idx % 3 == 0 {
            out.push('_');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_DIR_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(label: &str) -> Self {
            let seq = NEXT_TEST_DIR_ID.fetch_add(1, Ordering::Relaxed);
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock before unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "benchman-{label}-{}-{stamp}-{seq}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("failed to create test directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn parse_size_bytes_rejects_zero() {
        let err = parse_size_bytes("0", 7, Path::new("summary.tsv"))
            .expect_err("size_bytes=0 should be rejected");
        let message = format!("{err:#}");
        assert!(
            message.contains("size_bytes must be > 0"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn parse_size_bytes_accepts_positive_values() {
        assert_eq!(
            parse_size_bytes("4096", 11, Path::new("summary.tsv")).expect("valid positive value"),
            4096
        );
    }

    #[test]
    fn plot_dir_deletion_policy_prefers_marker() {
        let dir = TestDir::new("marker");
        fs::write(plot_dir_marker_path(dir.path()), "managed by benchman\n")
            .expect("failed to write marker");
        fs::write(dir.path().join("keep.txt"), "non-legacy file")
            .expect("failed to write fixture file");

        assert_eq!(
            plot_dir_deletion_policy(dir.path()).expect("policy should evaluate"),
            PlotDirDeletionPolicy::ManagedMarker
        );
    }

    #[test]
    fn plot_dir_deletion_policy_accepts_legacy_artifacts() {
        let dir = TestDir::new("legacy");
        fs::write(
            dir.path().join("infotheory-two-json-h-rss-run.svg"),
            "<svg/>",
        )
        .expect("failed to write legacy artifact");

        assert_eq!(
            plot_dir_deletion_policy(dir.path()).expect("policy should evaluate"),
            PlotDirDeletionPolicy::LegacyArtifacts
        );
    }

    #[test]
    fn plot_dir_deletion_policy_accepts_extra_suite_artifacts() {
        let dir = TestDir::new("legacy-extra");
        fs::write(dir.path().join("infotheory-extra-h-rss-run.svg"), "<svg/>")
            .expect("failed to write legacy artifact");

        assert_eq!(
            plot_dir_deletion_policy(dir.path()).expect("policy should evaluate"),
            PlotDirDeletionPolicy::LegacyArtifacts
        );
    }

    #[test]
    fn plot_dir_deletion_policy_accepts_one_sse_suite_artifacts() {
        let dir = TestDir::new("legacy-one-sse");
        fs::write(
            dir.path().join("infotheory-one-sse-h-rss-run.svg"),
            "<svg/>",
        )
        .expect("failed to write legacy artifact");

        assert_eq!(
            plot_dir_deletion_policy(dir.path()).expect("policy should evaluate"),
            PlotDirDeletionPolicy::LegacyArtifacts
        );
    }

    #[test]
    fn plot_dir_deletion_policy_requires_confirmation_for_unknown_contents() {
        let dir = TestDir::new("unknown");
        fs::write(dir.path().join("notes.txt"), "not a plot artifact")
            .expect("failed to write fixture file");

        assert_eq!(
            plot_dir_deletion_policy(dir.path()).expect("policy should evaluate"),
            PlotDirDeletionPolicy::RequiresExplicitConfirmation
        );
    }
}
