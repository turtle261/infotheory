use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(test)]
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use csv::ReaderBuilder;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, Frame, Line, Modifier, Span, Style};
use ratatui::symbols::Marker;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, GraphType, List, ListItem, ListState, Paragraph,
    Wrap,
};

use crate::{COLOR_PALETTE, centered_rect, format_float, format_size_bytes};

const DEFAULT_CHART_BINS: usize = 240;
const DEFAULT_COARSE_ROW_TARGET: usize = 1024;
const EXACT_NODE_SCAN_ROW_LIMIT: usize = 131_072;
const HIGHLIGHT_COUNT: usize = 4;
const TOP_EXACT_POINTS: usize = 8;

#[derive(Args, Clone, Debug)]
pub(crate) struct LogLossCli {
    #[arg(
        help = "Log-loss diagnostic prefix (for <prefix>.trace.tsv / .nodes.tsv / .summary.tsv)"
    )]
    pub(crate) prefix: PathBuf,
}

#[derive(Clone, Debug)]
struct LogLossPaths {
    prefix: PathBuf,
    trace_path: PathBuf,
    nodes_path: PathBuf,
    summary_path: PathBuf,
}

#[derive(Clone, Debug)]
struct LogLossNodeMeta {
    node_id: usize,
    display_name: String,
    backend_label: String,
    short_label: String,
}

#[derive(Clone, Debug, Default)]
struct LogLossNodeSummary {
    total_bits: f64,
    regret_bits: f64,
    oracle_win_count: u64,
    avg_local_weight: f64,
    avg_effective_weight: f64,
}

#[derive(Clone, Debug)]
struct LogLossSummary {
    positions: usize,
    input_bytes: u64,
    mix_total_bits: f64,
    oracle_total_bits: f64,
    oracle_regret_bits: f64,
    root_top1_switch_count: u64,
    oracle_switch_count: u64,
    root_weight_entropy_bits_avg: f64,
    root_top12_margin_avg: f64,
    ac_payload_bits_raw: u64,
    coder_overhead_bits: f64,
    node_summaries: Vec<LogLossNodeSummary>,
}

#[derive(Clone, Debug)]
struct TraceNodeColumns {
    node_pos: usize,
    bits_idx: usize,
    local_weight_idx: usize,
    effective_weight_idx: usize,
}

#[derive(Clone, Debug)]
struct LogLossTraceSchema {
    byte_u8_idx: usize,
    mix_bits_idx: usize,
    root_weight_entropy_bits_idx: usize,
    root_top12_margin_idx: usize,
    oracle_best_id_idx: usize,
    oracle_best_bits_idx: usize,
    oracle_regret_bits_idx: usize,
    node_columns: Vec<TraceNodeColumns>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactTraceRow {
    byte: u8,
    mix_bits: f32,
    oracle_bits: f32,
    regret_bits: f32,
    root_weight_entropy_bits: f32,
    root_top12_margin: f32,
    best_gap_bits: f32,
    oracle_best_node_pos: u32,
}

#[derive(Clone, Debug)]
struct TileAggregate {
    start_row: usize,
    end_row: usize,
    count: usize,
    mix_bits_sum: f64,
    mix_bits_max: f64,
    oracle_bits_sum: f64,
    oracle_bits_max: f64,
    regret_bits_sum: f64,
    regret_bits_max: f64,
    root_weight_entropy_sum: f64,
    root_top12_margin_sum: f64,
    best_gap_bits_sum: f64,
    ensemble_advantage_count: usize,
    node_bits_sum: Vec<f64>,
    node_local_weight_sum: Vec<f64>,
    node_effective_weight_sum: Vec<f64>,
    node_oracle_win_count: Vec<u32>,
}

impl TileAggregate {
    fn new(start_row: usize, node_count: usize) -> Self {
        Self {
            start_row,
            end_row: start_row,
            count: 0,
            mix_bits_sum: 0.0,
            mix_bits_max: f64::NEG_INFINITY,
            oracle_bits_sum: 0.0,
            oracle_bits_max: f64::NEG_INFINITY,
            regret_bits_sum: 0.0,
            regret_bits_max: f64::NEG_INFINITY,
            root_weight_entropy_sum: 0.0,
            root_top12_margin_sum: 0.0,
            best_gap_bits_sum: 0.0,
            ensemble_advantage_count: 0,
            node_bits_sum: vec![0.0; node_count],
            node_local_weight_sum: vec![0.0; node_count],
            node_effective_weight_sum: vec![0.0; node_count],
            node_oracle_win_count: vec![0; node_count],
        }
    }

    fn push(
        &mut self,
        row_idx: usize,
        compact: CompactTraceRow,
        node_bits: &[f64],
        node_local_weights: &[f64],
        node_effective_weights: &[f64],
    ) {
        self.end_row = row_idx + 1;
        self.count += 1;
        self.mix_bits_sum += compact.mix_bits as f64;
        self.mix_bits_max = self.mix_bits_max.max(compact.mix_bits as f64);
        self.oracle_bits_sum += compact.oracle_bits as f64;
        self.oracle_bits_max = self.oracle_bits_max.max(compact.oracle_bits as f64);
        self.regret_bits_sum += compact.regret_bits as f64;
        self.regret_bits_max = self.regret_bits_max.max(compact.regret_bits as f64);
        self.root_weight_entropy_sum += compact.root_weight_entropy_bits as f64;
        self.root_top12_margin_sum += compact.root_top12_margin as f64;
        self.best_gap_bits_sum += compact.best_gap_bits as f64;
        if compact.mix_bits + 1e-6 < compact.oracle_bits {
            self.ensemble_advantage_count += 1;
        }
        for (dst, value) in self.node_bits_sum.iter_mut().zip(node_bits.iter().copied()) {
            *dst += value;
        }
        for (dst, value) in self
            .node_local_weight_sum
            .iter_mut()
            .zip(node_local_weights.iter().copied())
        {
            *dst += value;
        }
        for (dst, value) in self
            .node_effective_weight_sum
            .iter_mut()
            .zip(node_effective_weights.iter().copied())
        {
            *dst += value;
        }
        let best_idx = compact.oracle_best_node_pos as usize;
        if let Some(slot) = self.node_oracle_win_count.get_mut(best_idx) {
            *slot = slot.saturating_add(1);
        }
    }
}

#[derive(Clone, Debug)]
struct LogLossData {
    paths: LogLossPaths,
    non_root_nodes: Vec<LogLossNodeMeta>,
    node_id_to_pos: HashMap<usize, usize>,
    summary: LogLossSummary,
    schema: LogLossTraceSchema,
    row_offsets: Vec<u64>,
    rows: Vec<CompactTraceRow>,
    bytes: Vec<u8>,
    tiles: Vec<TileAggregate>,
    coarse_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LogLossGraphKind {
    MixBitsAvg,
    OracleBitsAvg,
    RegretBitsAvg,
    RegretBitsMax,
    RootWeightEntropyAvg,
    RootWeightMarginAvg,
    BestGapAvg,
    DomainFraction,
    UncoveredFraction,
    BlindspotFraction,
    EnsembleAdvantageFraction,
    ExpertBitsAvg,
    ExpertLocalWeightAvg,
    ExpertEffectiveWeightAvg,
    OracleWinFraction,
}

#[derive(Clone, Debug)]
struct LogLossGraphSpec {
    id: &'static str,
    title: &'static str,
    description: &'static str,
    kind: LogLossGraphKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Viewport {
    start_row: usize,
    end_row: usize,
}

impl Viewport {
    fn len(self) -> usize {
        self.end_row.saturating_sub(self.start_row)
    }
}

#[derive(Clone, Copy, Debug)]
struct Thresholds {
    uncovered_bits: f64,
    contested_margin_bits: f64,
    blindspot_regret_bits: f64,
    blindspot_good_expert_bits: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            uncovered_bits: 6.0,
            contested_margin_bits: 0.25,
            blindspot_regret_bits: 0.5,
            blindspot_good_expert_bits: 4.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RenderCacheKey {
    graph_idx: usize,
    viewport_start: usize,
    viewport_end: usize,
    uncovered_bits_q: i32,
    contested_margin_bits_q: i32,
    blindspot_regret_bits_q: i32,
    blindspot_good_bits_q: i32,
}

#[derive(Clone, Debug)]
struct LogLossPointMeta {
    start_row: usize,
    end_row: usize,
    x_mid: f64,
    y: f64,
}

#[derive(Clone, Debug)]
struct LogLossSeriesData {
    name: String,
    color: Color,
    points: Vec<LogLossPointMeta>,
}

#[derive(Clone, Debug)]
struct LogLossGraphModel {
    spec: LogLossGraphSpec,
    series: Vec<LogLossSeriesData>,
}

#[derive(Clone, Debug)]
struct SeriesVisibilityState {
    visible_series: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct FocusPopup {
    series_names: Vec<String>,
    selected_idx: usize,
    visible_series: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct HighlightRegion {
    label: String,
    start_row: usize,
    end_row: usize,
    score: f64,
}

#[derive(Clone, Debug)]
struct RegionWorstPoint {
    row: usize,
    byte: u8,
    mix_bits: f64,
    oracle_bits: f64,
    regret_bits: f64,
    best_node_pos: usize,
    best_gap_bits: f64,
}

#[derive(Clone, Debug)]
struct RegionNodeStats {
    node_pos: usize,
    avg_bits: f64,
    avg_local_weight: f64,
    avg_effective_weight: f64,
    oracle_wins: usize,
    assigned_count: usize,
}

#[derive(Clone, Debug)]
struct RegionInspection {
    start_row: usize,
    end_row: usize,
    bytes_ascii_preview: String,
    bytes_hex_preview: String,
    avg_mix_bits: f64,
    avg_oracle_bits: f64,
    avg_regret_bits: f64,
    max_regret_bits: f64,
    avg_root_entropy_bits: f64,
    avg_root_margin: f64,
    avg_best_gap_bits: f64,
    uncovered_count: usize,
    contested_count: usize,
    blindspot_count: usize,
    ensemble_advantage_count: usize,
    assigned_counts: Vec<usize>,
    node_stats: Vec<RegionNodeStats>,
    worst_regret_points: Vec<RegionWorstPoint>,
    worst_oracle_points: Vec<RegionWorstPoint>,
}

pub(crate) struct LogLossApp {
    data: LogLossData,
    specs: Vec<LogLossGraphSpec>,
    visibility: Vec<SeriesVisibilityState>,
    current_graph: usize,
    current_model: LogLossGraphModel,
    current_cache_key: RenderCacheKey,
    cursor_x_idx: usize,
    cursor_series_idx: usize,
    viewport: Viewport,
    thresholds: Thresholds,
    focus_popup: Option<FocusPopup>,
    help_popup: bool,
    inspection: Option<RegionInspection>,
    should_quit: bool,
}

impl LogLossApp {
    pub(crate) fn from_cli(cli: &LogLossCli) -> Result<Self> {
        let paths = resolve_paths(&cli.prefix)?;
        let data = load_log_loss_data(paths)?;
        let specs = log_loss_graph_specs();
        let initial_viewport = Viewport {
            start_row: 0,
            end_row: data.rows.len(),
        };
        let thresholds = Thresholds::default();
        let initial_cache_key = RenderCacheKey {
            graph_idx: 0,
            viewport_start: initial_viewport.start_row,
            viewport_end: initial_viewport.end_row,
            uncovered_bits_q: quantize_threshold(thresholds.uncovered_bits),
            contested_margin_bits_q: quantize_threshold(thresholds.contested_margin_bits),
            blindspot_regret_bits_q: quantize_threshold(thresholds.blindspot_regret_bits),
            blindspot_good_bits_q: quantize_threshold(thresholds.blindspot_good_expert_bits),
        };
        let current_model =
            build_log_loss_graph_model(&data, specs[0].clone(), initial_viewport, thresholds)?;
        let visibility = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let model = if index == 0 {
                    current_model.clone()
                } else {
                    build_log_loss_graph_model(&data, spec.clone(), initial_viewport, thresholds)
                        .unwrap_or_else(|_| LogLossGraphModel {
                            spec: spec.clone(),
                            series: Vec::new(),
                        })
                };
                SeriesVisibilityState {
                    visible_series: model
                        .series
                        .iter()
                        .map(|series| series.name.clone())
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        let mut app = Self {
            data,
            specs,
            visibility,
            current_graph: 0,
            current_model,
            current_cache_key: initial_cache_key,
            cursor_x_idx: 0,
            cursor_series_idx: 0,
            viewport: initial_viewport,
            thresholds,
            focus_popup: None,
            help_popup: false,
            inspection: None,
            should_quit: false,
        };
        app.clamp_cursor();
        Ok(app)
    }

    pub(crate) fn should_quit(&self) -> bool {
        self.should_quit
    }

    pub(crate) fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::{KeyCode, KeyEventKind};

        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return;
        }

        if self.focus_popup.is_some() {
            self.handle_focus_key(key.code);
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
                self.refresh_current_model();
            }
            KeyCode::Char('G') => {
                self.current_graph = self.specs.len().saturating_sub(1);
                self.refresh_current_model();
            }
            KeyCode::Left | KeyCode::Char('h') => self.move_x_cursor(-1),
            KeyCode::Right | KeyCode::Char('l') => self.move_x_cursor(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_series_cursor(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_series_cursor(1),
            KeyCode::Char('f') => self.open_focus_popup(),
            KeyCode::Char('?') => self.help_popup = true,
            KeyCode::Enter => {
                let _ = self.open_inspection();
            }
            KeyCode::Char('c') | KeyCode::Esc => self.inspection = None,
            KeyCode::Char('z') => self.zoom_in(),
            KeyCode::Char('Z') => self.zoom_out(),
            KeyCode::Char('a') => self.reset_viewport(),
            KeyCode::Char('b') => self.adjust_uncovered_threshold(-0.25),
            KeyCode::Char('B') => self.adjust_uncovered_threshold(0.25),
            KeyCode::Char('m') => self.adjust_contested_margin(-0.05),
            KeyCode::Char('M') => self.adjust_contested_margin(0.05),
            KeyCode::Char('r') => self.adjust_blindspot_regret(-0.1),
            KeyCode::Char('R') => self.adjust_blindspot_regret(0.1),
            KeyCode::Char('w') => self.adjust_blindspot_good(-0.25),
            KeyCode::Char('W') => self.adjust_blindspot_good(0.25),
            _ => {}
        }
    }

    pub(crate) fn render(&self, frame: &mut Frame<'_>) {
        let root = frame.area();
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(8), Constraint::Length(7)])
            .split(root);

        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(23),
                Constraint::Percentage(52),
                Constraint::Percentage(25),
            ])
            .split(vertical[0]);

        render_log_loss_graph_list(frame, top[0], self);
        render_log_loss_chart(frame, top[1], self);
        render_log_loss_side_panel(frame, top[2], self);
        render_log_loss_status(frame, vertical[1], self);

        if let Some(popup) = &self.focus_popup {
            render_focus_popup(frame, popup);
        }
        if self.help_popup {
            render_help_popup(frame);
        } else if let Some(inspection) = &self.inspection {
            render_inspection_popup(frame, inspection, &self.data.non_root_nodes);
        }
    }

    fn refresh_current_model(&mut self) {
        let key = self.current_cache_key_for_current_graph();
        if key == self.current_cache_key {
            return;
        }
        if let Ok(model) = build_log_loss_graph_model(
            &self.data,
            self.specs[self.current_graph].clone(),
            self.viewport,
            self.thresholds,
        ) {
            let visible_series = model
                .series
                .iter()
                .map(|series| series.name.clone())
                .collect::<BTreeSet<_>>();
            if let Some(state) = self.visibility.get_mut(self.current_graph) {
                if state.visible_series.is_empty() {
                    state.visible_series = visible_series;
                } else {
                    state.visible_series = state
                        .visible_series
                        .iter()
                        .filter(|name| visible_series.contains(*name))
                        .cloned()
                        .collect();
                    if state.visible_series.is_empty() {
                        state.visible_series = visible_series;
                    }
                }
            }
            self.current_model = model;
            self.current_cache_key = key;
            self.inspection = None;
            self.focus_popup = None;
            self.clamp_cursor();
        }
    }

    fn current_cache_key_for_current_graph(&self) -> RenderCacheKey {
        RenderCacheKey {
            graph_idx: self.current_graph,
            viewport_start: self.viewport.start_row,
            viewport_end: self.viewport.end_row,
            uncovered_bits_q: quantize_threshold(self.thresholds.uncovered_bits),
            contested_margin_bits_q: quantize_threshold(self.thresholds.contested_margin_bits),
            blindspot_regret_bits_q: quantize_threshold(self.thresholds.blindspot_regret_bits),
            blindspot_good_bits_q: quantize_threshold(self.thresholds.blindspot_good_expert_bits),
        }
    }

    fn current_visibility(&self) -> &SeriesVisibilityState {
        &self.visibility[self.current_graph]
    }

    fn current_visibility_mut(&mut self) -> &mut SeriesVisibilityState {
        &mut self.visibility[self.current_graph]
    }

    fn visible_series_indices(&self) -> Vec<usize> {
        let state = self.current_visibility();
        let mut indices = (0..self.current_model.series.len()).collect::<Vec<_>>();
        indices.retain(|idx| {
            let series = &self.current_model.series[*idx];
            state.visible_series.contains(&series.name)
        });
        indices
    }

    fn clamp_cursor(&mut self) {
        let visible_points = self.current_x_values_len();
        if visible_points == 0 {
            self.cursor_x_idx = 0;
        } else {
            self.cursor_x_idx = self.cursor_x_idx.min(visible_points.saturating_sub(1));
        }

        let visible_series = self.visible_series_indices();
        if visible_series.is_empty() {
            self.cursor_series_idx = 0;
        } else {
            self.cursor_series_idx = self
                .cursor_series_idx
                .min(visible_series.len().saturating_sub(1));
        }
    }

    fn current_x_values_len(&self) -> usize {
        self.current_model
            .series
            .iter()
            .map(|series| series.points.len())
            .max()
            .unwrap_or(0)
    }

    fn current_point_range(&self) -> Option<(usize, usize)> {
        let visible_series = self.visible_series_indices();
        let series_idx = *visible_series.get(self.cursor_series_idx)?;
        let point = self.current_model.series[series_idx]
            .points
            .get(self.cursor_x_idx)?;
        Some((point.start_row, point.end_row))
    }

    fn selected_series_index(&self) -> Option<usize> {
        let visible = self.visible_series_indices();
        visible.get(self.cursor_series_idx).copied()
    }

    fn selected_cursor_point(&self) -> Option<(&str, &LogLossPointMeta)> {
        let series_idx = self.selected_series_index()?;
        let series = &self.current_model.series[series_idx];
        let point = series.points.get(self.cursor_x_idx)?;
        Some((series.name.as_str(), point))
    }

    fn move_graph(&mut self, delta: i32) {
        let len = self.specs.len() as i32;
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
        self.refresh_current_model();
    }

    fn move_x_cursor(&mut self, delta: i32) {
        let len = self.current_x_values_len();
        if len == 0 {
            return;
        }
        let mut next = self.cursor_x_idx as i32 + delta;
        if next < 0 {
            next = 0;
        }
        let max_idx = len.saturating_sub(1) as i32;
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

    fn open_focus_popup(&mut self) {
        let series_names = self
            .current_model
            .series
            .iter()
            .map(|series| series.name.clone())
            .collect::<Vec<_>>();
        if series_names.is_empty() {
            return;
        }
        self.focus_popup = Some(FocusPopup {
            series_names,
            selected_idx: 0,
            visible_series: self.current_visibility().visible_series.clone(),
        });
    }

    fn handle_focus_key(&mut self, code: crossterm::event::KeyCode) {
        use crossterm::event::KeyCode;

        let Some(popup) = self.focus_popup.as_mut() else {
            return;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('q') => self.focus_popup = None,
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
            KeyCode::Char('n') => popup.visible_series.clear(),
            KeyCode::Enter => {
                if let Some(popup) = self.focus_popup.take() {
                    self.current_visibility_mut().visible_series = popup.visible_series;
                    self.clamp_cursor();
                    self.inspection = None;
                }
            }
            _ => {}
        }
    }

    fn zoom_in(&mut self) {
        let Some((start, end)) = self.current_point_range() else {
            return;
        };
        if end.saturating_sub(start) < 16 {
            return;
        }
        self.viewport = Viewport {
            start_row: start,
            end_row: end,
        };
        self.cursor_x_idx = 0;
        self.cursor_series_idx = 0;
        self.refresh_current_model();
    }

    fn zoom_out(&mut self) {
        let total = self.data.rows.len();
        let len = self.viewport.len();
        if len >= total {
            return;
        }
        let center = self
            .current_point_range()
            .map(|(start, end)| (start + end) / 2)
            .unwrap_or((self.viewport.start_row + self.viewport.end_row) / 2);
        let new_len = (len.saturating_mul(2))
            .min(total)
            .max(DEFAULT_COARSE_ROW_TARGET);
        let mut start = center.saturating_sub(new_len / 2);
        let end = (start + new_len).min(total);
        start = end.saturating_sub(new_len);
        self.viewport = Viewport {
            start_row: start,
            end_row: end,
        };
        self.cursor_x_idx = 0;
        self.cursor_series_idx = 0;
        self.refresh_current_model();
    }

    fn reset_viewport(&mut self) {
        self.viewport = Viewport {
            start_row: 0,
            end_row: self.data.rows.len(),
        };
        self.cursor_x_idx = 0;
        self.cursor_series_idx = 0;
        self.refresh_current_model();
    }

    fn adjust_uncovered_threshold(&mut self, delta: f64) {
        self.thresholds.uncovered_bits = (self.thresholds.uncovered_bits + delta).max(0.0);
        self.refresh_current_model();
    }

    fn adjust_contested_margin(&mut self, delta: f64) {
        self.thresholds.contested_margin_bits =
            (self.thresholds.contested_margin_bits + delta).max(0.0);
        self.refresh_current_model();
    }

    fn adjust_blindspot_regret(&mut self, delta: f64) {
        self.thresholds.blindspot_regret_bits =
            (self.thresholds.blindspot_regret_bits + delta).max(0.0);
        self.refresh_current_model();
    }

    fn adjust_blindspot_good(&mut self, delta: f64) {
        self.thresholds.blindspot_good_expert_bits =
            (self.thresholds.blindspot_good_expert_bits + delta).max(0.0);
        self.refresh_current_model();
    }

    fn interpretation_lines(&self) -> Vec<Line<'static>> {
        let spec = &self.current_model.spec;
        let range = self.current_point_range();
        let highlights = self.current_highlights();
        let mut global_oracle = self
            .data
            .non_root_nodes
            .iter()
            .zip(self.data.summary.node_summaries.iter())
            .map(|(node, summary)| {
                (
                    format!("{} [{}]", node.display_name, node.backend_label),
                    summary.oracle_win_count,
                )
            })
            .collect::<Vec<_>>();
        global_oracle.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut global_effective = self
            .data
            .non_root_nodes
            .iter()
            .zip(self.data.summary.node_summaries.iter())
            .map(|(node, summary)| {
                (
                    format!("{} [{}]", node.display_name, node.backend_label),
                    summary.avg_effective_weight,
                )
            })
            .collect::<Vec<_>>();
        global_effective.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut global_totals = self
            .data
            .non_root_nodes
            .iter()
            .zip(self.data.summary.node_summaries.iter())
            .map(|(node, summary)| {
                (
                    format!("{} [{}]", node.display_name, node.backend_label),
                    summary.total_bits,
                )
            })
            .collect::<Vec<_>>();
        global_totals.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut global_regrets = self
            .data
            .non_root_nodes
            .iter()
            .zip(self.data.summary.node_summaries.iter())
            .map(|(node, summary)| {
                (
                    format!("{} [{}]", node.display_name, node.backend_label),
                    summary.regret_bits,
                )
            })
            .collect::<Vec<_>>();
        global_regrets.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut global_local = self
            .data
            .non_root_nodes
            .iter()
            .zip(self.data.summary.node_summaries.iter())
            .map(|(node, summary)| {
                (
                    format!("{} [{}]", node.display_name, node.backend_label),
                    summary.avg_local_weight,
                )
            })
            .collect::<Vec<_>>();
        global_local.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let selected = self
            .selected_cursor_point()
            .map(|(name, point)| format!("{name} = {}", format_float(point.y)))
            .unwrap_or_else(|| "no series point selected".to_string());
        let range_text = range
            .map(|(start, end)| {
                format!(
                    "t=[{}..{})",
                    format_size_bytes(start as u64),
                    format_size_bytes(end as u64)
                )
            })
            .unwrap_or_else(|| "t=n/a".to_string());

        let mut lines = vec![
            Line::from(Span::styled(
                spec.description.to_string(),
                Style::default().fg(Color::Cyan),
            )),
            Line::from(format!("{range_text}    {selected}")),
            Line::from(format!(
                "thresholds: uncovered>={:.2}b  contested-gap<{:.2}b  blindspot regret>={:.2}b with oracle<={:.2}b",
                self.thresholds.uncovered_bits,
                self.thresholds.contested_margin_bits,
                self.thresholds.blindspot_regret_bits,
                self.thresholds.blindspot_good_expert_bits
            )),
            Line::from(format!(
                "totals: mix={}b  oracle={}b  regret={}b  ac_payload={}b",
                format_float(self.data.summary.mix_total_bits),
                format_float(self.data.summary.oracle_total_bits),
                format_float(self.data.summary.oracle_regret_bits),
                format_size_bytes(self.data.summary.ac_payload_bits_raw)
            )),
            Line::from(format!(
                "input={} rows={} coder_overhead={}b",
                format_size_bytes(self.data.summary.input_bytes),
                format_size_bytes(self.data.summary.positions as u64),
                format_float(self.data.summary.coder_overhead_bits)
            )),
        ];

        if !global_oracle.is_empty() {
            let summary = global_oracle
                .iter()
                .take(2)
                .map(|(name, wins)| format!("{name}={}", format_size_bytes(*wins)))
                .collect::<Vec<_>>()
                .join("  ");
            lines.push(Line::from(format!("global oracle winners: {summary}")));
        }
        if !global_totals.is_empty() {
            let summary = global_totals
                .iter()
                .take(2)
                .map(|(name, bits)| format!("{name}={}", format_float(*bits)))
                .collect::<Vec<_>>()
                .join("  ");
            lines.push(Line::from(format!("best standalone totals: {summary}")));
        }
        if !global_regrets.is_empty() {
            let summary = global_regrets
                .iter()
                .take(2)
                .map(|(name, regret)| format!("{name}={}", format_float(*regret)))
                .collect::<Vec<_>>()
                .join("  ");
            lines.push(Line::from(format!("lowest regret to mixture: {summary}")));
        }
        if !global_local.is_empty() {
            let summary = global_local
                .iter()
                .take(2)
                .map(|(name, weight)| format!("{name}={}", format_float(*weight)))
                .collect::<Vec<_>>()
                .join("  ");
            lines.push(Line::from(format!("highest avg local weight: {summary}")));
        }
        if !global_effective.is_empty() {
            let summary = global_effective
                .iter()
                .take(2)
                .map(|(name, weight)| format!("{name}={}", format_float(*weight)))
                .collect::<Vec<_>>()
                .join("  ");
            lines.push(Line::from(format!(
                "global avg effective weight: {summary}"
            )));
        }

        if !highlights.is_empty() {
            lines.push(Line::from(Span::styled(
                "top coarse regions:",
                Style::default().add_modifier(Modifier::BOLD),
            )));
            for highlight in highlights {
                lines.push(Line::from(format!(
                    "  {} [{}..{}) score={}",
                    highlight.label,
                    format_size_bytes(highlight.start_row as u64),
                    format_size_bytes(highlight.end_row as u64),
                    format_float(highlight.score)
                )));
            }
        }

        lines
    }

    fn current_highlights(&self) -> Vec<HighlightRegion> {
        let Some(series_idx) = self.selected_series_index() else {
            return Vec::new();
        };
        let series = &self.current_model.series[series_idx];
        let mut points = series.points.clone();
        points.sort_by(|a, b| b.y.total_cmp(&a.y));
        points
            .into_iter()
            .take(HIGHLIGHT_COUNT)
            .map(|point| HighlightRegion {
                label: series.name.clone(),
                start_row: point.start_row,
                end_row: point.end_row,
                score: point.y,
            })
            .collect()
    }

    fn open_inspection(&mut self) -> Result<()> {
        let Some((start_row, end_row)) = self.current_point_range() else {
            self.inspection = None;
            return Ok(());
        };
        let inspection = inspect_region(&self.data, start_row, end_row, self.thresholds)?;
        self.inspection = Some(inspection);
        Ok(())
    }
}

fn quantize_threshold(value: f64) -> i32 {
    (value * 1000.0).round() as i32
}

fn resolve_paths(raw: &Path) -> Result<LogLossPaths> {
    let prefix_str = raw.to_string_lossy();
    let prefix = if let Some(base) = prefix_str.strip_suffix(".trace.tsv") {
        PathBuf::from(base)
    } else if let Some(base) = prefix_str.strip_suffix(".nodes.tsv") {
        PathBuf::from(base)
    } else if let Some(base) = prefix_str.strip_suffix(".summary.tsv") {
        PathBuf::from(base)
    } else {
        raw.to_path_buf()
    };

    let trace_path = prefix.with_extension("trace.tsv");
    let nodes_path = prefix.with_extension("nodes.tsv");
    let summary_path = prefix.with_extension("summary.tsv");
    if !trace_path.is_file() {
        bail!("trace TSV not found: {}", trace_path.display());
    }
    if !nodes_path.is_file() {
        bail!("nodes TSV not found: {}", nodes_path.display());
    }
    if !summary_path.is_file() {
        bail!("summary TSV not found: {}", summary_path.display());
    }

    Ok(LogLossPaths {
        prefix,
        trace_path,
        nodes_path,
        summary_path,
    })
}

fn load_log_loss_data(paths: LogLossPaths) -> Result<LogLossData> {
    let non_root_nodes = load_nodes(&paths.nodes_path)?
        .iter()
        .filter(|node| node.node_id != 0)
        .cloned()
        .collect::<Vec<_>>();
    if non_root_nodes.is_empty() {
        bail!("nodes TSV contained no non-root mixture constituents");
    }
    let node_id_to_pos = non_root_nodes
        .iter()
        .enumerate()
        .map(|(pos, node)| (node.node_id, pos))
        .collect::<HashMap<_, _>>();
    let summary = load_summary(&paths.summary_path, &non_root_nodes)?;
    let (schema, row_offsets, rows, bytes, tiles) = load_trace(
        &paths.trace_path,
        &non_root_nodes,
        &node_id_to_pos,
        summary.positions,
    )?;

    if row_offsets.len() != summary.positions {
        bail!(
            "trace rows ({}) did not match summary positions ({})",
            row_offsets.len(),
            summary.positions
        );
    }

    let mix_sum = rows.iter().map(|row| row.mix_bits as f64).sum::<f64>();
    if (mix_sum - summary.mix_total_bits).abs() > 1e-3 {
        bail!(
            "trace/summary mismatch: sum(trace.mix_bits)={} summary.mix_total_bits={}",
            mix_sum,
            summary.mix_total_bits
        );
    }

    let coarse_rows = if tiles.is_empty() {
        1
    } else {
        tiles[0].count.max(1)
    };

    Ok(LogLossData {
        paths,
        non_root_nodes,
        node_id_to_pos,
        summary,
        schema,
        row_offsets,
        rows,
        bytes,
        tiles,
        coarse_rows,
    })
}

fn load_nodes(path: &Path) -> Result<Vec<LogLossNodeMeta>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open nodes TSV {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read header from {}", path.display()))?
        .clone();

    let idx_node_id = header_index(&headers, "node_id")?;
    let idx_parent_id = header_index(&headers, "parent_id")?;
    let idx_depth = header_index(&headers, "depth")?;
    let idx_path = header_index(&headers, "path")?;
    let idx_display_name = header_index(&headers, "display_name")?;
    let idx_backend_label = header_index(&headers, "backend_label")?;
    let idx_is_mixture = header_index(&headers, "is_mixture")?;
    let idx_is_leaf = header_index(&headers, "is_leaf")?;
    let idx_is_root_child = header_index(&headers, "is_root_child")?;

    let mut nodes = Vec::new();
    for (row_idx, row) in reader.records().enumerate() {
        let row = row.with_context(|| {
            format!(
                "failed to parse nodes row {} in {}",
                row_idx + 2,
                path.display()
            )
        })?;
        let node_id =
            parse_u64(get_field(&row, idx_node_id), "node_id", row_idx + 2, path)? as usize;
        let display_name = get_field(&row, idx_display_name).trim().to_string();
        let _parent_id = parse_optional_usize(
            get_field(&row, idx_parent_id),
            "parent_id",
            row_idx + 2,
            path,
        )?;
        let _depth = parse_u64(get_field(&row, idx_depth), "depth", row_idx + 2, path)? as usize;
        let _path = get_field(&row, idx_path).trim();
        let _is_mixture = parse_bool_flag(
            get_field(&row, idx_is_mixture),
            "is_mixture",
            row_idx + 2,
            path,
        )?;
        let _is_leaf = parse_bool_flag(get_field(&row, idx_is_leaf), "is_leaf", row_idx + 2, path)?;
        let _is_root_child = parse_bool_flag(
            get_field(&row, idx_is_root_child),
            "is_root_child",
            row_idx + 2,
            path,
        )?;
        nodes.push(LogLossNodeMeta {
            node_id,
            display_name: display_name.clone(),
            backend_label: get_field(&row, idx_backend_label).trim().to_string(),
            short_label: if node_id == 0 {
                "root".to_string()
            } else {
                format!("n{} {}", node_id, display_name)
            },
        });
    }
    nodes.sort_by_key(|node| node.node_id);
    Ok(nodes)
}

fn load_summary(path: &Path, non_root_nodes: &[LogLossNodeMeta]) -> Result<LogLossSummary> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open summary TSV {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read header from {}", path.display()))?
        .clone();
    let row = reader
        .records()
        .next()
        .transpose()
        .with_context(|| format!("failed to parse summary row in {}", path.display()))?
        .context("summary TSV was empty")?;

    let positions = parse_u64(
        get_field(&row, header_index(&headers, "positions")?),
        "positions",
        2,
        path,
    )? as usize;
    let input_bytes = parse_u64(
        get_field(&row, header_index(&headers, "input_bytes")?),
        "input_bytes",
        2,
        path,
    )?;
    let mix_total_bits = parse_required_f64(
        get_field(&row, header_index(&headers, "mix_total_bits")?),
        "mix_total_bits",
        2,
        path,
    )?;
    let oracle_total_bits = parse_required_f64(
        get_field(&row, header_index(&headers, "oracle_total_bits")?),
        "oracle_total_bits",
        2,
        path,
    )?;
    let oracle_regret_bits = parse_required_f64(
        get_field(&row, header_index(&headers, "oracle_regret_bits")?),
        "oracle_regret_bits",
        2,
        path,
    )?;
    let root_top1_switch_count = parse_u64(
        get_field(&row, header_index(&headers, "root_top1_switch_count")?),
        "root_top1_switch_count",
        2,
        path,
    )?;
    let oracle_switch_count = parse_u64(
        get_field(&row, header_index(&headers, "oracle_switch_count")?),
        "oracle_switch_count",
        2,
        path,
    )?;
    let root_weight_entropy_bits_avg = parse_required_f64(
        get_field(
            &row,
            header_index(&headers, "root_weight_entropy_bits_avg")?,
        ),
        "root_weight_entropy_bits_avg",
        2,
        path,
    )?;
    let root_top12_margin_avg = parse_required_f64(
        get_field(&row, header_index(&headers, "root_top12_margin_avg")?),
        "root_top12_margin_avg",
        2,
        path,
    )?;
    let ac_payload_bits_raw = parse_u64(
        get_field(&row, header_index(&headers, "ac_payload_bits_raw")?),
        "ac_payload_bits_raw",
        2,
        path,
    )?;
    let coder_overhead_bits = parse_required_f64(
        get_field(&row, header_index(&headers, "coder_overhead_bits")?),
        "coder_overhead_bits",
        2,
        path,
    )?;

    let mut node_summaries = Vec::with_capacity(non_root_nodes.len());
    for node in non_root_nodes {
        node_summaries.push(LogLossNodeSummary {
            total_bits: parse_required_f64(
                get_field(
                    &row,
                    header_index(&headers, &format!("n{}__total_bits", node.node_id))?,
                ),
                "node_total_bits",
                2,
                path,
            )?,
            regret_bits: parse_required_f64(
                get_field(
                    &row,
                    header_index(&headers, &format!("n{}__regret_bits", node.node_id))?,
                ),
                "node_regret_bits",
                2,
                path,
            )?,
            oracle_win_count: parse_u64(
                get_field(
                    &row,
                    header_index(&headers, &format!("n{}__oracle_win_count", node.node_id))?,
                ),
                "node_oracle_win_count",
                2,
                path,
            )?,
            avg_local_weight: parse_required_f64(
                get_field(
                    &row,
                    header_index(&headers, &format!("n{}__avg_local_weight", node.node_id))?,
                ),
                "node_avg_local_weight",
                2,
                path,
            )?,
            avg_effective_weight: parse_required_f64(
                get_field(
                    &row,
                    header_index(
                        &headers,
                        &format!("n{}__avg_effective_weight", node.node_id),
                    )?,
                ),
                "node_avg_effective_weight",
                2,
                path,
            )?,
        });
    }

    Ok(LogLossSummary {
        positions,
        input_bytes,
        mix_total_bits,
        oracle_total_bits,
        oracle_regret_bits,
        root_top1_switch_count,
        oracle_switch_count,
        root_weight_entropy_bits_avg,
        root_top12_margin_avg,
        ac_payload_bits_raw,
        coder_overhead_bits,
        node_summaries,
    })
}

fn load_trace(
    path: &Path,
    non_root_nodes: &[LogLossNodeMeta],
    node_id_to_pos: &HashMap<usize, usize>,
    expected_rows: usize,
) -> Result<(
    LogLossTraceSchema,
    Vec<u64>,
    Vec<CompactTraceRow>,
    Vec<u8>,
    Vec<TileAggregate>,
)> {
    let file =
        File::open(path).with_context(|| format!("failed to open trace TSV {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut header_line = String::new();
    let header_len = reader
        .read_line(&mut header_line)
        .with_context(|| format!("failed to read trace header from {}", path.display()))?;
    if header_len == 0 {
        bail!("trace TSV was empty: {}", path.display());
    }

    let mut header_fields = Vec::new();
    split_tsv_line(&header_line, &mut header_fields);
    let schema = build_trace_schema(&header_fields, non_root_nodes)?;

    let coarse_rows = coarse_rows_for_len(expected_rows);
    let mut row_offsets = Vec::with_capacity(expected_rows);
    let mut rows = Vec::with_capacity(expected_rows);
    let mut bytes = Vec::with_capacity(expected_rows);
    let mut tiles = Vec::new();
    let mut line = String::new();
    let mut offset = header_len as u64;
    let mut current_tile = TileAggregate::new(0, non_root_nodes.len());
    let mut node_bits = vec![0.0; non_root_nodes.len()];
    let mut node_local_weights = vec![0.0; non_root_nodes.len()];
    let mut node_effective_weights = vec![0.0; non_root_nodes.len()];

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read trace row from {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }

        let row_idx = rows.len();
        row_offsets.push(offset);
        offset += bytes_read as u64;

        let mut fields = Vec::with_capacity(header_fields.len());
        split_tsv_line(&line, &mut fields);
        let byte = parse_inline_u8(
            fields.get(schema.byte_u8_idx).copied().unwrap_or(""),
            "byte_u8",
            row_idx + 2,
            path,
        )?;
        let mix_bits = parse_inline_f64(
            fields.get(schema.mix_bits_idx).copied().unwrap_or(""),
            "mix_bits",
            row_idx + 2,
            path,
        )?;
        let entropy_bits = parse_inline_f64(
            fields
                .get(schema.root_weight_entropy_bits_idx)
                .copied()
                .unwrap_or(""),
            "root_weight_entropy_bits",
            row_idx + 2,
            path,
        )?;
        let root_top12_margin = parse_inline_f64(
            fields
                .get(schema.root_top12_margin_idx)
                .copied()
                .unwrap_or(""),
            "root_top12_margin",
            row_idx + 2,
            path,
        )?;
        let oracle_best_id = parse_inline_usize(
            fields.get(schema.oracle_best_id_idx).copied().unwrap_or(""),
            "oracle_best_id",
            row_idx + 2,
            path,
        )?;
        let oracle_best_bits = parse_inline_f64(
            fields
                .get(schema.oracle_best_bits_idx)
                .copied()
                .unwrap_or(""),
            "oracle_best_bits",
            row_idx + 2,
            path,
        )?;
        let oracle_regret_bits = parse_inline_f64(
            fields
                .get(schema.oracle_regret_bits_idx)
                .copied()
                .unwrap_or(""),
            "oracle_regret_bits",
            row_idx + 2,
            path,
        )?;
        let oracle_best_node_pos = *node_id_to_pos.get(&oracle_best_id).with_context(|| {
            format!(
                "{} row {}: oracle_best_id {} not found in nodes TSV",
                path.display(),
                row_idx + 2,
                oracle_best_id
            )
        })?;

        let mut best_bits = f64::INFINITY;
        let mut second_best_bits = f64::INFINITY;
        for columns in &schema.node_columns {
            let node_pos = columns.node_pos;
            let bits = parse_inline_f64(
                fields.get(columns.bits_idx).copied().unwrap_or(""),
                "node_bits",
                row_idx + 2,
                path,
            )?;
            let local_weight = parse_inline_f64(
                fields.get(columns.local_weight_idx).copied().unwrap_or(""),
                "node_local_weight",
                row_idx + 2,
                path,
            )?;
            let effective_weight = parse_inline_f64(
                fields
                    .get(columns.effective_weight_idx)
                    .copied()
                    .unwrap_or(""),
                "node_effective_weight",
                row_idx + 2,
                path,
            )?;
            node_bits[node_pos] = bits;
            node_local_weights[node_pos] = local_weight;
            node_effective_weights[node_pos] = effective_weight;
            if bits < best_bits {
                second_best_bits = best_bits;
                best_bits = bits;
            } else if bits < second_best_bits {
                second_best_bits = bits;
            }
        }
        let best_gap_bits = if second_best_bits.is_finite() {
            (second_best_bits - best_bits).max(0.0)
        } else {
            f64::INFINITY
        };

        let compact = CompactTraceRow {
            byte,
            mix_bits: mix_bits as f32,
            oracle_bits: oracle_best_bits as f32,
            regret_bits: oracle_regret_bits as f32,
            root_weight_entropy_bits: entropy_bits as f32,
            root_top12_margin: root_top12_margin as f32,
            best_gap_bits: best_gap_bits as f32,
            oracle_best_node_pos: oracle_best_node_pos as u32,
        };
        rows.push(compact);
        bytes.push(byte);

        if current_tile.count == 0 {
            current_tile = TileAggregate::new(row_idx, non_root_nodes.len());
        }
        current_tile.push(
            row_idx,
            compact,
            &node_bits,
            &node_local_weights,
            &node_effective_weights,
        );
        if current_tile.count >= coarse_rows {
            tiles.push(current_tile);
            current_tile = TileAggregate::new(row_idx + 1, non_root_nodes.len());
        }
    }

    if current_tile.count > 0 {
        tiles.push(current_tile);
    }

    Ok((schema, row_offsets, rows, bytes, tiles))
}

fn coarse_rows_for_len(row_count: usize) -> usize {
    let target = row_count.div_ceil(DEFAULT_CHART_BINS * 4).max(1);
    target.max(DEFAULT_COARSE_ROW_TARGET)
}

fn build_trace_schema(
    header_fields: &[&str],
    non_root_nodes: &[LogLossNodeMeta],
) -> Result<LogLossTraceSchema> {
    let mut header_map = HashMap::new();
    for (idx, field) in header_fields.iter().enumerate() {
        header_map.insert((*field).to_string(), idx);
    }

    let mut node_columns = Vec::with_capacity(non_root_nodes.len());
    for (node_pos, node) in non_root_nodes.iter().enumerate() {
        node_columns.push(TraceNodeColumns {
            node_pos,
            bits_idx: *header_map
                .get(&format!("n{}__bits", node.node_id))
                .with_context(|| format!("trace header missing n{}__bits", node.node_id))?,
            local_weight_idx: *header_map
                .get(&format!("n{}__local_weight", node.node_id))
                .with_context(|| format!("trace header missing n{}__local_weight", node.node_id))?,
            effective_weight_idx: *header_map
                .get(&format!("n{}__effective_weight", node.node_id))
                .with_context(|| {
                    format!("trace header missing n{}__effective_weight", node.node_id)
                })?,
        });
    }

    Ok(LogLossTraceSchema {
        byte_u8_idx: require_header(&header_map, "byte_u8")?,
        mix_bits_idx: require_header(&header_map, "mix_bits")?,
        root_weight_entropy_bits_idx: require_header(&header_map, "root_weight_entropy_bits")?,
        root_top12_margin_idx: require_header(&header_map, "root_top12_margin")?,
        oracle_best_id_idx: require_header(&header_map, "oracle_best_id")?,
        oracle_best_bits_idx: require_header(&header_map, "oracle_best_bits")?,
        oracle_regret_bits_idx: require_header(&header_map, "oracle_regret_bits")?,
        node_columns,
    })
}

fn require_header(header_map: &HashMap<String, usize>, key: &str) -> Result<usize> {
    header_map
        .get(key)
        .copied()
        .with_context(|| format!("trace header missing {key}"))
}

fn split_tsv_line<'a>(line: &'a str, fields: &mut Vec<&'a str>) {
    fields.clear();
    let bytes = line.as_bytes();
    let mut start = 0usize;
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\t' => {
                fields.push(&line[start..idx]);
                idx += 1;
                start = idx;
            }
            b'\n' => {
                fields.push(&line[start..idx]);
                return;
            }
            b'\r' => {
                fields.push(&line[start..idx]);
                return;
            }
            _ => idx += 1,
        }
    }
    fields.push(&line[start..]);
}

fn log_loss_graph_specs() -> Vec<LogLossGraphSpec> {
    vec![
        LogLossGraphSpec {
            id: "mix-bits",
            title: "Mixture Avg Bits vs Position",
            description: "Average online arithmetic-coding loss of the full mixture in each region.",
            kind: LogLossGraphKind::MixBitsAvg,
        },
        LogLossGraphSpec {
            id: "oracle-bits",
            title: "Oracle Avg Bits vs Position",
            description: "Average best-constituent counterfactual loss in each region.",
            kind: LogLossGraphKind::OracleBitsAvg,
        },
        LogLossGraphSpec {
            id: "regret-avg",
            title: "Mixture Regret Avg vs Position",
            description: "Average excess bits paid by the mixture relative to the oracle best constituent.",
            kind: LogLossGraphKind::RegretBitsAvg,
        },
        LogLossGraphSpec {
            id: "regret-max",
            title: "Mixture Regret Max vs Position",
            description: "Worst single-position excess mixture loss in each region.",
            kind: LogLossGraphKind::RegretBitsMax,
        },
        LogLossGraphSpec {
            id: "weight-entropy",
            title: "Root Weight Entropy Avg vs Position",
            description: "Average entropy of the root mixture weights; high values indicate uncertainty over experts.",
            kind: LogLossGraphKind::RootWeightEntropyAvg,
        },
        LogLossGraphSpec {
            id: "weight-margin",
            title: "Root Top-2 Weight Margin Avg vs Position",
            description: "Average gap between the top two root-child mixture weights; low values indicate ambiguous routing.",
            kind: LogLossGraphKind::RootWeightMarginAvg,
        },
        LogLossGraphSpec {
            id: "best-gap",
            title: "Best-vs-Runner-up Expert Gap Avg",
            description: "Average gap in bits between the best and second-best constituents. Useful for contested-vs-clean domains.",
            kind: LogLossGraphKind::BestGapAvg,
        },
        LogLossGraphSpec {
            id: "domain-fraction",
            title: "Thresholded Domain Partition",
            description: "Fraction of positions claimed by each constituent, plus contested and uncovered categories, under the current thresholds.",
            kind: LogLossGraphKind::DomainFraction,
        },
        LogLossGraphSpec {
            id: "uncovered-fraction",
            title: "Uncovered Fraction",
            description: "Fraction of positions where even the oracle best constituent exceeds the current uncovered-bits threshold.",
            kind: LogLossGraphKind::UncoveredFraction,
        },
        LogLossGraphSpec {
            id: "blindspot-fraction",
            title: "Mixture Blindspot Fraction",
            description: "Fraction of positions where a good constituent exists but the mixture still pays the configured regret penalty.",
            kind: LogLossGraphKind::BlindspotFraction,
        },
        LogLossGraphSpec {
            id: "ensemble-advantage",
            title: "Ensemble Advantage Fraction",
            description: "Fraction of positions where the mixture beats every constituent model alone.",
            kind: LogLossGraphKind::EnsembleAdvantageFraction,
        },
        LogLossGraphSpec {
            id: "expert-bits",
            title: "Per-Node Avg Bits",
            description: "Average counterfactual loss for every flattened non-root node in the mixture tree.",
            kind: LogLossGraphKind::ExpertBitsAvg,
        },
        LogLossGraphSpec {
            id: "expert-local-weight",
            title: "Per-Node Avg Local Weight",
            description: "Average local mixture responsibility inside each node's immediate parent mixture.",
            kind: LogLossGraphKind::ExpertLocalWeightAvg,
        },
        LogLossGraphSpec {
            id: "expert-effective-weight",
            title: "Per-Node Avg Effective Weight",
            description: "Average effective root-to-node responsibility; this is the most direct routing mass view.",
            kind: LogLossGraphKind::ExpertEffectiveWeightAvg,
        },
        LogLossGraphSpec {
            id: "oracle-win-fraction",
            title: "Per-Node Oracle Win Fraction",
            description: "Fraction of positions for which each flattened node is the oracle best constituent.",
            kind: LogLossGraphKind::OracleWinFraction,
        },
    ]
}

fn build_log_loss_graph_model(
    data: &LogLossData,
    spec: LogLossGraphSpec,
    viewport: Viewport,
    thresholds: Thresholds,
) -> Result<LogLossGraphModel> {
    let series = match spec.kind {
        LogLossGraphKind::MixBitsAvg
        | LogLossGraphKind::OracleBitsAvg
        | LogLossGraphKind::RegretBitsAvg
        | LogLossGraphKind::RegretBitsMax
        | LogLossGraphKind::RootWeightEntropyAvg
        | LogLossGraphKind::RootWeightMarginAvg
        | LogLossGraphKind::BestGapAvg
        | LogLossGraphKind::DomainFraction
        | LogLossGraphKind::UncoveredFraction
        | LogLossGraphKind::BlindspotFraction
        | LogLossGraphKind::EnsembleAdvantageFraction => {
            build_compact_row_series(data, &spec, viewport, thresholds)
        }
        LogLossGraphKind::ExpertBitsAvg
        | LogLossGraphKind::ExpertLocalWeightAvg
        | LogLossGraphKind::ExpertEffectiveWeightAvg
        | LogLossGraphKind::OracleWinFraction => build_node_metric_series(data, &spec, viewport)?,
    };

    Ok(LogLossGraphModel { spec, series })
}

fn build_compact_row_series(
    data: &LogLossData,
    spec: &LogLossGraphSpec,
    viewport: Viewport,
    thresholds: Thresholds,
) -> Vec<LogLossSeriesData> {
    let len = viewport.len();
    if len == 0 {
        return Vec::new();
    }
    let bin_count = len.min(DEFAULT_CHART_BINS).max(1);
    let mut series_builders: BTreeMap<String, Vec<LogLossPointMeta>> = BTreeMap::new();

    for bin_idx in 0..bin_count {
        let start = viewport.start_row + (bin_idx * len) / bin_count;
        let mut end = viewport.start_row + ((bin_idx + 1) * len) / bin_count;
        if end <= start {
            end = (start + 1).min(viewport.end_row);
        }
        if end <= start {
            continue;
        }

        let slice = &data.rows[start..end];
        let x_mid = ((start + end) as f64) / 2.0;
        match spec.kind {
            LogLossGraphKind::MixBitsAvg => {
                let value =
                    slice.iter().map(|row| row.mix_bits as f64).sum::<f64>() / (slice.len() as f64);
                series_builders
                    .entry("mixture".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::OracleBitsAvg => {
                let value = slice.iter().map(|row| row.oracle_bits as f64).sum::<f64>()
                    / (slice.len() as f64);
                series_builders
                    .entry("oracle best".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::RegretBitsAvg => {
                let value = slice.iter().map(|row| row.regret_bits as f64).sum::<f64>()
                    / (slice.len() as f64);
                series_builders
                    .entry("mixture regret".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::RegretBitsMax => {
                let value = slice
                    .iter()
                    .map(|row| row.regret_bits as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                series_builders
                    .entry("max regret".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::RootWeightEntropyAvg => {
                let value = slice
                    .iter()
                    .map(|row| row.root_weight_entropy_bits as f64)
                    .sum::<f64>()
                    / (slice.len() as f64);
                series_builders
                    .entry("root weight entropy".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::RootWeightMarginAvg => {
                let value = slice
                    .iter()
                    .map(|row| row.root_top12_margin as f64)
                    .sum::<f64>()
                    / (slice.len() as f64);
                series_builders
                    .entry("root top-2 margin".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::BestGapAvg => {
                let value = slice
                    .iter()
                    .map(|row| row.best_gap_bits as f64)
                    .sum::<f64>()
                    / (slice.len() as f64);
                series_builders
                    .entry("best-vs-runner-up gap".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::UncoveredFraction => {
                let count = slice
                    .iter()
                    .filter(|row| row.oracle_bits as f64 >= thresholds.uncovered_bits)
                    .count();
                let value = count as f64 / (slice.len() as f64);
                series_builders
                    .entry("uncovered".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::BlindspotFraction => {
                let count = slice
                    .iter()
                    .filter(|row| {
                        row.regret_bits as f64 >= thresholds.blindspot_regret_bits
                            && row.oracle_bits as f64 <= thresholds.blindspot_good_expert_bits
                    })
                    .count();
                let value = count as f64 / (slice.len() as f64);
                series_builders
                    .entry("mixture blindspot".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::EnsembleAdvantageFraction => {
                let count = slice
                    .iter()
                    .filter(|row| row.mix_bits + 1e-6 < row.oracle_bits)
                    .count();
                let value = count as f64 / (slice.len() as f64);
                series_builders
                    .entry("mixture beats all".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: value,
                    });
            }
            LogLossGraphKind::DomainFraction => {
                let mut assigned_counts = vec![0usize; data.non_root_nodes.len()];
                let mut contested_count = 0usize;
                let mut uncovered_count = 0usize;
                for row in slice {
                    if row.oracle_bits as f64 >= thresholds.uncovered_bits {
                        uncovered_count += 1;
                    } else if (row.best_gap_bits as f64) < thresholds.contested_margin_bits {
                        contested_count += 1;
                    } else {
                        let pos = row.oracle_best_node_pos as usize;
                        if let Some(slot) = assigned_counts.get_mut(pos) {
                            *slot += 1;
                        }
                    }
                }
                for (node, &count) in data.non_root_nodes.iter().zip(assigned_counts.iter()) {
                    let value = count as f64 / (slice.len() as f64);
                    series_builders
                        .entry(node.short_label.clone())
                        .or_default()
                        .push(LogLossPointMeta {
                            start_row: start,
                            end_row: end,
                            x_mid,
                            y: value,
                        });
                }
                series_builders
                    .entry("contested".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: contested_count as f64 / (slice.len() as f64),
                    });
                series_builders
                    .entry("uncovered".to_string())
                    .or_default()
                    .push(LogLossPointMeta {
                        start_row: start,
                        end_row: end,
                        x_mid,
                        y: uncovered_count as f64 / (slice.len() as f64),
                    });
            }
            _ => {}
        }
    }

    finalize_series_builders(series_builders)
}

fn build_node_metric_series(
    data: &LogLossData,
    spec: &LogLossGraphSpec,
    viewport: Viewport,
) -> Result<Vec<LogLossSeriesData>> {
    if viewport.len() == 0 {
        return Ok(Vec::new());
    }
    if viewport.len() <= EXACT_NODE_SCAN_ROW_LIMIT {
        build_exact_node_metric_series(data, spec, viewport)
    } else {
        Ok(build_tile_node_metric_series(data, spec, viewport))
    }
}

fn build_tile_node_metric_series(
    data: &LogLossData,
    spec: &LogLossGraphSpec,
    viewport: Viewport,
) -> Vec<LogLossSeriesData> {
    let tile_indices = data
        .tiles
        .iter()
        .enumerate()
        .filter_map(|(idx, tile)| {
            if tile.end_row <= viewport.start_row || tile.start_row >= viewport.end_row {
                None
            } else {
                Some(idx)
            }
        })
        .collect::<Vec<_>>();
    if tile_indices.is_empty() {
        return Vec::new();
    }

    let bin_count = tile_indices.len().min(DEFAULT_CHART_BINS).max(1);
    let mut series_builders: BTreeMap<String, Vec<LogLossPointMeta>> = BTreeMap::new();

    for bin_idx in 0..bin_count {
        let start_tile_idx = (bin_idx * tile_indices.len()) / bin_count;
        let mut end_tile_idx = ((bin_idx + 1) * tile_indices.len()) / bin_count;
        if end_tile_idx <= start_tile_idx {
            end_tile_idx = (start_tile_idx + 1).min(tile_indices.len());
        }
        if end_tile_idx <= start_tile_idx {
            continue;
        }
        let tile_slice = &tile_indices[start_tile_idx..end_tile_idx];
        let start_row = data.tiles[*tile_slice.first().unwrap()]
            .start_row
            .max(viewport.start_row);
        let end_row = data.tiles[*tile_slice.last().unwrap()]
            .end_row
            .min(viewport.end_row);
        let total_count = tile_slice
            .iter()
            .map(|tile_idx| data.tiles[*tile_idx].count)
            .sum::<usize>()
            .max(1);
        let x_mid = ((start_row + end_row) as f64) / 2.0;

        for (node_pos, node) in data.non_root_nodes.iter().enumerate() {
            let value = match spec.kind {
                LogLossGraphKind::ExpertBitsAvg => {
                    tile_slice
                        .iter()
                        .map(|tile_idx| data.tiles[*tile_idx].node_bits_sum[node_pos])
                        .sum::<f64>()
                        / (total_count as f64)
                }
                LogLossGraphKind::ExpertLocalWeightAvg => {
                    tile_slice
                        .iter()
                        .map(|tile_idx| data.tiles[*tile_idx].node_local_weight_sum[node_pos])
                        .sum::<f64>()
                        / (total_count as f64)
                }
                LogLossGraphKind::ExpertEffectiveWeightAvg => {
                    tile_slice
                        .iter()
                        .map(|tile_idx| data.tiles[*tile_idx].node_effective_weight_sum[node_pos])
                        .sum::<f64>()
                        / (total_count as f64)
                }
                LogLossGraphKind::OracleWinFraction => {
                    tile_slice
                        .iter()
                        .map(|tile_idx| {
                            data.tiles[*tile_idx].node_oracle_win_count[node_pos] as f64
                        })
                        .sum::<f64>()
                        / (total_count as f64)
                }
                _ => 0.0,
            };
            series_builders
                .entry(node.short_label.clone())
                .or_default()
                .push(LogLossPointMeta {
                    start_row,
                    end_row,
                    x_mid,
                    y: value,
                });
        }
    }

    finalize_series_builders(series_builders)
}

fn build_exact_node_metric_series(
    data: &LogLossData,
    spec: &LogLossGraphSpec,
    viewport: Viewport,
) -> Result<Vec<LogLossSeriesData>> {
    let len = viewport.len();
    let bin_count = len.min(DEFAULT_CHART_BINS).max(1);
    let mut accum = (0..bin_count)
        .map(|bin_idx| {
            ExactNodeBin::new(
                bin_range(viewport, len, bin_count, bin_idx),
                data.non_root_nodes.len(),
            )
        })
        .collect::<Vec<_>>();

    scan_trace_range(
        data,
        viewport.start_row,
        viewport.end_row,
        |row_idx, fields| {
            let bin_idx = row_to_bin(viewport, len, bin_count, row_idx);
            let bin = &mut accum[bin_idx];
            bin.count += 1;
            for columns in &data.schema.node_columns {
                let node_pos = columns.node_pos;
                let bits = parse_inline_f64(
                    fields.get(columns.bits_idx).copied().unwrap_or(""),
                    "node_bits",
                    row_idx + 2,
                    &data.paths.trace_path,
                )?;
                let local_weight = parse_inline_f64(
                    fields.get(columns.local_weight_idx).copied().unwrap_or(""),
                    "node_local_weight",
                    row_idx + 2,
                    &data.paths.trace_path,
                )?;
                let effective_weight = parse_inline_f64(
                    fields
                        .get(columns.effective_weight_idx)
                        .copied()
                        .unwrap_or(""),
                    "node_effective_weight",
                    row_idx + 2,
                    &data.paths.trace_path,
                )?;
                bin.node_bits_sum[node_pos] += bits;
                bin.node_local_weight_sum[node_pos] += local_weight;
                bin.node_effective_weight_sum[node_pos] += effective_weight;
            }
            let oracle_best_id = parse_inline_usize(
                fields
                    .get(data.schema.oracle_best_id_idx)
                    .copied()
                    .unwrap_or(""),
                "oracle_best_id",
                row_idx + 2,
                &data.paths.trace_path,
            )?;
            if let Some(&pos) = data.node_id_to_pos.get(&oracle_best_id) {
                bin.node_oracle_win_count[pos] += 1;
            }
            Ok(())
        },
    )?;

    let mut series_builders: BTreeMap<String, Vec<LogLossPointMeta>> = BTreeMap::new();
    for bin in accum {
        if bin.count == 0 {
            continue;
        }
        let x_mid = ((bin.start_row + bin.end_row) as f64) / 2.0;
        for (node_pos, node) in data.non_root_nodes.iter().enumerate() {
            let value = match spec.kind {
                LogLossGraphKind::ExpertBitsAvg => bin.node_bits_sum[node_pos] / (bin.count as f64),
                LogLossGraphKind::ExpertLocalWeightAvg => {
                    bin.node_local_weight_sum[node_pos] / (bin.count as f64)
                }
                LogLossGraphKind::ExpertEffectiveWeightAvg => {
                    bin.node_effective_weight_sum[node_pos] / (bin.count as f64)
                }
                LogLossGraphKind::OracleWinFraction => {
                    bin.node_oracle_win_count[node_pos] as f64 / (bin.count as f64)
                }
                _ => 0.0,
            };
            series_builders
                .entry(node.short_label.clone())
                .or_default()
                .push(LogLossPointMeta {
                    start_row: bin.start_row,
                    end_row: bin.end_row,
                    x_mid,
                    y: value,
                });
        }
    }

    Ok(finalize_series_builders(series_builders))
}

#[derive(Clone, Debug)]
struct ExactNodeBin {
    start_row: usize,
    end_row: usize,
    count: usize,
    node_bits_sum: Vec<f64>,
    node_local_weight_sum: Vec<f64>,
    node_effective_weight_sum: Vec<f64>,
    node_oracle_win_count: Vec<usize>,
}

impl ExactNodeBin {
    fn new((start_row, end_row): (usize, usize), node_count: usize) -> Self {
        Self {
            start_row,
            end_row,
            count: 0,
            node_bits_sum: vec![0.0; node_count],
            node_local_weight_sum: vec![0.0; node_count],
            node_effective_weight_sum: vec![0.0; node_count],
            node_oracle_win_count: vec![0; node_count],
        }
    }
}

fn finalize_series_builders(
    series_builders: BTreeMap<String, Vec<LogLossPointMeta>>,
) -> Vec<LogLossSeriesData> {
    series_builders
        .into_iter()
        .enumerate()
        .map(|(idx, (name, points))| LogLossSeriesData {
            name,
            color: COLOR_PALETTE[idx % COLOR_PALETTE.len()],
            points,
        })
        .collect()
}

fn bin_range(viewport: Viewport, len: usize, bin_count: usize, bin_idx: usize) -> (usize, usize) {
    let start = viewport.start_row + (bin_idx * len) / bin_count;
    let mut end = viewport.start_row + ((bin_idx + 1) * len) / bin_count;
    if end <= start {
        end = (start + 1).min(viewport.end_row);
    }
    (start, end)
}

fn row_to_bin(viewport: Viewport, len: usize, bin_count: usize, row_idx: usize) -> usize {
    let relative = row_idx.saturating_sub(viewport.start_row);
    let mut bin = (relative * bin_count) / len.max(1);
    if bin >= bin_count {
        bin = bin_count - 1;
    }
    bin
}

fn inspect_region(
    data: &LogLossData,
    start_row: usize,
    end_row: usize,
    thresholds: Thresholds,
) -> Result<RegionInspection> {
    if start_row >= end_row || end_row > data.rows.len() {
        bail!("invalid inspection range [{start_row}..{end_row})");
    }

    let slice = &data.rows[start_row..end_row];
    let count = slice.len().max(1);
    let avg_mix_bits = slice.iter().map(|row| row.mix_bits as f64).sum::<f64>() / (count as f64);
    let avg_oracle_bits =
        slice.iter().map(|row| row.oracle_bits as f64).sum::<f64>() / (count as f64);
    let avg_regret_bits =
        slice.iter().map(|row| row.regret_bits as f64).sum::<f64>() / (count as f64);
    let max_regret_bits = slice
        .iter()
        .map(|row| row.regret_bits as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_root_entropy_bits = slice
        .iter()
        .map(|row| row.root_weight_entropy_bits as f64)
        .sum::<f64>()
        / (count as f64);
    let avg_root_margin = slice
        .iter()
        .map(|row| row.root_top12_margin as f64)
        .sum::<f64>()
        / (count as f64);
    let avg_best_gap_bits = slice
        .iter()
        .map(|row| row.best_gap_bits as f64)
        .sum::<f64>()
        / (count as f64);
    let mut assigned_counts = vec![0usize; data.non_root_nodes.len()];
    let mut uncovered_count = 0usize;
    let mut contested_count = 0usize;
    let mut blindspot_count = 0usize;
    let mut ensemble_advantage_count = 0usize;
    let mut worst_regret_points = Vec::new();
    let mut worst_oracle_points = Vec::new();

    for (offset, row) in slice.iter().enumerate() {
        let absolute_row = start_row + offset;
        let best_pos = row.oracle_best_node_pos as usize;
        let point = RegionWorstPoint {
            row: absolute_row,
            byte: row.byte,
            mix_bits: row.mix_bits as f64,
            oracle_bits: row.oracle_bits as f64,
            regret_bits: row.regret_bits as f64,
            best_node_pos: best_pos,
            best_gap_bits: row.best_gap_bits as f64,
        };
        worst_regret_points.push(point.clone());
        worst_oracle_points.push(point);
        if row.oracle_bits as f64 >= thresholds.uncovered_bits {
            uncovered_count += 1;
        } else if (row.best_gap_bits as f64) < thresholds.contested_margin_bits {
            contested_count += 1;
        } else if let Some(slot) = assigned_counts.get_mut(best_pos) {
            *slot += 1;
        }
        if row.regret_bits as f64 >= thresholds.blindspot_regret_bits
            && row.oracle_bits as f64 <= thresholds.blindspot_good_expert_bits
        {
            blindspot_count += 1;
        }
        if row.mix_bits + 1e-6 < row.oracle_bits {
            ensemble_advantage_count += 1;
        }
    }

    worst_regret_points.sort_by(|a, b| {
        b.regret_bits
            .total_cmp(&a.regret_bits)
            .then_with(|| a.row.cmp(&b.row))
    });
    worst_regret_points.truncate(TOP_EXACT_POINTS);
    worst_oracle_points.sort_by(|a, b| {
        b.oracle_bits
            .total_cmp(&a.oracle_bits)
            .then_with(|| a.row.cmp(&b.row))
    });
    worst_oracle_points.truncate(TOP_EXACT_POINTS);

    let mut node_stats = data
        .non_root_nodes
        .iter()
        .enumerate()
        .map(|(node_pos, _)| RegionNodeStats {
            node_pos,
            avg_bits: 0.0,
            avg_local_weight: 0.0,
            avg_effective_weight: 0.0,
            oracle_wins: 0,
            assigned_count: assigned_counts[node_pos],
        })
        .collect::<Vec<_>>();

    scan_trace_range(data, start_row, end_row, |row_idx, fields| {
        for columns in &data.schema.node_columns {
            let node_pos = columns.node_pos;
            node_stats[node_pos].avg_bits += parse_inline_f64(
                fields.get(columns.bits_idx).copied().unwrap_or(""),
                "node_bits",
                row_idx + 2,
                &data.paths.trace_path,
            )?;
            node_stats[node_pos].avg_local_weight += parse_inline_f64(
                fields.get(columns.local_weight_idx).copied().unwrap_or(""),
                "node_local_weight",
                row_idx + 2,
                &data.paths.trace_path,
            )?;
            node_stats[node_pos].avg_effective_weight += parse_inline_f64(
                fields
                    .get(columns.effective_weight_idx)
                    .copied()
                    .unwrap_or(""),
                "node_effective_weight",
                row_idx + 2,
                &data.paths.trace_path,
            )?;
        }
        let oracle_best_id = parse_inline_usize(
            fields
                .get(data.schema.oracle_best_id_idx)
                .copied()
                .unwrap_or(""),
            "oracle_best_id",
            row_idx + 2,
            &data.paths.trace_path,
        )?;
        if let Some(&pos) = data.node_id_to_pos.get(&oracle_best_id) {
            node_stats[pos].oracle_wins += 1;
        }
        Ok(())
    })?;

    for stats in &mut node_stats {
        stats.avg_bits /= count as f64;
        stats.avg_local_weight /= count as f64;
        stats.avg_effective_weight /= count as f64;
    }
    node_stats.sort_by(|a, b| {
        a.avg_bits
            .total_cmp(&b.avg_bits)
            .then_with(|| b.oracle_wins.cmp(&a.oracle_wins))
            .then_with(|| a.node_pos.cmp(&b.node_pos))
    });

    let preview_len = (end_row - start_row).min(64);
    let preview_bytes = &data.bytes[start_row..start_row + preview_len];
    let bytes_ascii_preview = preview_bytes
        .iter()
        .map(|&byte| {
            if byte.is_ascii_graphic() || byte == b' ' {
                byte as char
            } else {
                '.'
            }
        })
        .collect::<String>();
    let bytes_hex_preview = preview_bytes
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect::<Vec<_>>()
        .join(" ");

    Ok(RegionInspection {
        start_row,
        end_row,
        bytes_ascii_preview,
        bytes_hex_preview,
        avg_mix_bits,
        avg_oracle_bits,
        avg_regret_bits,
        max_regret_bits,
        avg_root_entropy_bits,
        avg_root_margin,
        avg_best_gap_bits,
        uncovered_count,
        contested_count,
        blindspot_count,
        ensemble_advantage_count,
        assigned_counts,
        node_stats,
        worst_regret_points,
        worst_oracle_points,
    })
}

fn scan_trace_range<F>(data: &LogLossData, start_row: usize, end_row: usize, mut f: F) -> Result<()>
where
    F: FnMut(usize, &[&str]) -> Result<()>,
{
    if start_row >= end_row {
        return Ok(());
    }
    let offset = *data
        .row_offsets
        .get(start_row)
        .with_context(|| format!("row offset out of range: {start_row}"))?;
    let mut file = File::open(&data.paths.trace_path)
        .with_context(|| format!("failed to open {}", data.paths.trace_path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("failed to seek {}", data.paths.trace_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    for row_idx in start_row..end_row {
        line.clear();
        let bytes_read = reader.read_line(&mut line).with_context(|| {
            format!(
                "failed to read row {} from {}",
                row_idx,
                data.paths.trace_path.display()
            )
        })?;
        if bytes_read == 0 {
            bail!(
                "unexpected EOF reading rows [{}..{}) from {}",
                start_row,
                end_row,
                data.paths.trace_path.display()
            );
        }
        let mut fields = Vec::new();
        split_tsv_line(&line, &mut fields);
        f(row_idx, &fields)?;
    }
    Ok(())
}

fn render_log_loss_graph_list(frame: &mut Frame<'_>, area: Rect, app: &LogLossApp) {
    let items = app
        .specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| {
            let prefix = if idx == app.current_graph { ">" } else { " " };
            ListItem::new(format!("{prefix} {}", spec.title))
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    state.select(Some(app.current_graph));
    let list = List::new(items)
        .block(
            Block::default()
                .title("Log-Loss Views")
                .borders(Borders::ALL),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_log_loss_chart(frame: &mut Frame<'_>, area: Rect, app: &LogLossApp) {
    let visible_indices = app.visible_series_indices();
    let visible_series = visible_indices
        .iter()
        .map(|idx| &app.current_model.series[*idx])
        .collect::<Vec<_>>();

    let mut datasets = Vec::new();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let data_store = visible_series
        .iter()
        .map(|series| {
            series
                .points
                .iter()
                .map(|point| {
                    min_x = min_x.min(point.x_mid);
                    max_x = max_x.max(point.x_mid);
                    min_y = min_y.min(point.y);
                    max_y = max_y.max(point.y);
                    (point.x_mid, point.y)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

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
        cursor_store = Some(vec![(point.x_mid, point.y)]);
    }
    if let Some(cursor) = cursor_store.as_ref() {
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
                .data(cursor.as_slice()),
        );
    }

    let (x_bounds, x_labels) = if !min_x.is_finite() || !max_x.is_finite() {
        (
            [0.0, 1.0],
            vec![Span::raw("0"), Span::raw("position"), Span::raw("1")],
        )
    } else {
        let lo = min_x;
        let mut hi = max_x;
        if (hi - lo).abs() < f64::EPSILON {
            hi += 1.0;
        }
        let mid = (lo + hi) / 2.0;
        (
            [lo, hi],
            vec![
                Span::raw(format_size_bytes(lo.max(0.0).round() as u64)),
                Span::raw(format_size_bytes(mid.max(0.0).round() as u64)),
                Span::raw(format_size_bytes(hi.max(0.0).round() as u64)),
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

    let title = format!(
        "{}   [{}..{})",
        app.current_model.spec.title,
        format_size_bytes(app.viewport.start_row as u64),
        format_size_bytes(app.viewport.end_row as u64)
    );
    let chart = Chart::new(datasets)
        .block(Block::default().title(title).borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .title("file position t")
                .bounds(x_bounds)
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("value")
                .bounds(y_bounds)
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

fn render_log_loss_side_panel(frame: &mut Frame<'_>, area: Rect, app: &LogLossApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(area);
    render_log_loss_series_panel(frame, chunks[0], app);
    let paragraph = Paragraph::new(app.interpretation_lines())
        .block(
            Block::default()
                .title("Interpretation")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, chunks[1]);
}

fn render_log_loss_series_panel(frame: &mut Frame<'_>, area: Rect, app: &LogLossApp) {
    let visible = app.current_visibility();
    let selected_series = app.selected_series_index();
    let mut items = Vec::new();
    for idx in 0..app.current_model.series.len() {
        let series = &app.current_model.series[idx];
        let marked = if visible.visible_series.contains(&series.name) {
            "[x]"
        } else {
            "[ ]"
        };
        let value = series
            .points
            .get(app.cursor_x_idx)
            .map(|point| format_float(point.y))
            .unwrap_or_else(|| "n/a".to_string());
        let mut style = Style::default().fg(series.color);
        if Some(idx) == selected_series {
            style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
        }
        items.push(ListItem::new(format!("{marked} {}  y={value}", series.name)).style(style));
    }
    if items.is_empty() {
        items.push(ListItem::new("No visible series"));
    }
    let list = List::new(items).block(
        Block::default()
            .title(format!(
                "Series ({}/{})",
                visible.visible_series.len(),
                app.current_model.series.len()
            ))
            .borders(Borders::ALL),
    );
    frame.render_widget(list, area);
}

fn render_log_loss_status(frame: &mut Frame<'_>, area: Rect, app: &LogLossApp) {
    let x_range = app
        .current_point_range()
        .map(|(start, end)| {
            format!(
                "[{}..{})",
                format_size_bytes(start as u64),
                format_size_bytes(end as u64)
            )
        })
        .unwrap_or_else(|| "n/a".to_string());
    let selected = app
        .selected_cursor_point()
        .map(|(name, point)| format!("{name}={}", format_float(point.y)))
        .unwrap_or_else(|| "no point selected".to_string());

    let lines = vec![
        Line::from(format!(
            "Graph {}/{} [{}]  viewport=[{}..{})  cursor={}  {}",
            app.current_graph + 1,
            app.specs.len(),
            app.current_model.spec.id,
            format_size_bytes(app.viewport.start_row as u64),
            format_size_bytes(app.viewport.end_row as u64),
            x_range,
            selected
        )),
        Line::from(format!(
            "prefix={}  trace={}  rows={}  coarse_rows={}  nodes={}",
            app.data.paths.prefix.display(),
            app.data.paths.trace_path.display(),
            format_size_bytes(app.data.rows.len() as u64),
            format_size_bytes(app.data.coarse_rows as u64),
            app.data.non_root_nodes.len()
        )),
        Line::from(format!(
            "switches: root_top1={}  oracle={}  avg root entropy={}  avg root top2 margin={}  coder overhead={}b",
            format_size_bytes(app.data.summary.root_top1_switch_count),
            format_size_bytes(app.data.summary.oracle_switch_count),
            format_float(app.data.summary.root_weight_entropy_bits_avg),
            format_float(app.data.summary.root_top12_margin_avg),
            format_float(app.data.summary.coder_overhead_bits)
        )),
        Line::from(
            "Keys: [ ] view | h/j/k/l cursor | z/Z zoom | a full range | b/B uncovered | m/M contested gap | r/R blindspot regret | w/W blindspot good | f focus | Enter inspect | c clear | q quit",
        ),
    ];

    let block = Paragraph::new(lines)
        .block(Block::default().title("Status").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(block, area);
}

fn render_focus_popup(frame: &mut Frame<'_>, popup: &FocusPopup) {
    let area = centered_rect(70, 70, frame.area());
    frame.render_widget(Clear, area);
    let items = popup
        .series_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let selected = popup.visible_series.contains(name);
            let marker = if selected { "[x]" } else { "[ ]" };
            let prefix = if idx == popup.selected_idx { ">" } else { " " };
            ListItem::new(format!("{prefix} {marker} {name}"))
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
    let area = centered_rect(82, 48, frame.area());
    frame.render_widget(Clear, area);
    let lines = vec![
        Line::from(
            "Keys: [ ] view | h/j/k/l cursor | z/Z zoom | a full range | Enter inspect | c clear | f focus | q quit",
        ),
        Line::from(
            "Threshold controls: b/B uncovered bits, m/M contested margin, r/R blindspot regret, w/W blindspot good-expert bits.",
        ),
        Line::from(
            "Interpretation rule: uncovered if oracle>=threshold; otherwise contested if best-gap<threshold; otherwise assigned to the oracle-best node.",
        ),
        Line::from("Press c or Esc to clear this panel."),
    ];
    let paragraph = Paragraph::new(lines)
        .block(Block::default().title("Help").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_inspection_popup(
    frame: &mut Frame<'_>,
    inspection: &RegionInspection,
    non_root_nodes: &[LogLossNodeMeta],
) {
    let area = centered_rect(88, 82, frame.area());
    frame.render_widget(Clear, area);
    let mut lines = vec![
        Line::from(format!(
            "region=[{}..{}) rows={}",
            format_size_bytes(inspection.start_row as u64),
            format_size_bytes(inspection.end_row as u64),
            format_size_bytes((inspection.end_row - inspection.start_row) as u64)
        )),
        Line::from(format!(
            "avg mix={}  avg oracle={}  avg regret={}  max regret={}  avg entropy={}  avg root margin={}  avg best gap={}",
            format_float(inspection.avg_mix_bits),
            format_float(inspection.avg_oracle_bits),
            format_float(inspection.avg_regret_bits),
            format_float(inspection.max_regret_bits),
            format_float(inspection.avg_root_entropy_bits),
            format_float(inspection.avg_root_margin),
            format_float(inspection.avg_best_gap_bits)
        )),
        Line::from(format!(
            "counts: uncovered={} contested={} blindspot={} mixture-beats-all={}",
            format_size_bytes(inspection.uncovered_count as u64),
            format_size_bytes(inspection.contested_count as u64),
            format_size_bytes(inspection.blindspot_count as u64),
            format_size_bytes(inspection.ensemble_advantage_count as u64)
        )),
        Line::from(format!("ascii preview: {}", inspection.bytes_ascii_preview)),
        Line::from(format!("hex preview: {}", inspection.bytes_hex_preview)),
        Line::from(Span::styled(
            "node stats (sorted by avg bits):",
            Style::default().add_modifier(Modifier::BOLD),
        )),
    ];

    let mut assigned_summary = inspection
        .assigned_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(node_pos, count)| (node_pos, *count))
        .collect::<Vec<_>>();
    assigned_summary.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if !assigned_summary.is_empty() {
        let summary = assigned_summary
            .iter()
            .take(6)
            .map(|(node_pos, count)| {
                format!(
                    "{}={}",
                    non_root_nodes[*node_pos].short_label,
                    format_size_bytes(*count as u64)
                )
            })
            .collect::<Vec<_>>()
            .join("  ");
        lines.push(Line::from(format!("assigned domains: {summary}")));
    }

    for stats in inspection.node_stats.iter().take(12) {
        let node = &non_root_nodes[stats.node_pos];
        lines.push(Line::from(format!(
            "  {} [{}] avg_bits={} avg_local={} avg_effective={} oracle_wins={} assigned={}",
            node.short_label,
            node.backend_label,
            format_float(stats.avg_bits),
            format_float(stats.avg_local_weight),
            format_float(stats.avg_effective_weight),
            format_size_bytes(stats.oracle_wins as u64),
            format_size_bytes(stats.assigned_count as u64)
        )));
    }

    lines.push(Line::from(Span::styled(
        "worst positions by regret:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    for point in &inspection.worst_regret_points {
        let node = &non_root_nodes[point.best_node_pos];
        lines.push(Line::from(format!(
            "  t={} byte=0x{:02X} mix={} oracle={} regret={} best={} gap={}",
            format_size_bytes(point.row as u64),
            point.byte,
            format_float(point.mix_bits),
            format_float(point.oracle_bits),
            format_float(point.regret_bits),
            node.short_label,
            format_float(point.best_gap_bits)
        )));
    }

    lines.push(Line::from(Span::styled(
        "worst positions by oracle bits:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    for point in &inspection.worst_oracle_points {
        let node = &non_root_nodes[point.best_node_pos];
        lines.push(Line::from(format!(
            "  t={} byte=0x{:02X} oracle={} mix={} regret={} best={}",
            format_size_bytes(point.row as u64),
            point.byte,
            format_float(point.oracle_bits),
            format_float(point.mix_bits),
            format_float(point.regret_bits),
            node.short_label
        )));
    }

    lines.push(Line::from("Press c or Esc to clear this panel."));

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .title("Region Inspector")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn header_index(headers: &csv::StringRecord, column: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h == column)
        .with_context(|| format!("required TSV column not found: {column}"))
}

fn get_field<'a>(row: &'a csv::StringRecord, idx: usize) -> &'a str {
    row.get(idx).unwrap_or("")
}

fn parse_u64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<u64> {
    raw.trim().parse::<u64>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })
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

fn parse_optional_usize(
    raw: &str,
    field: &str,
    row_no: usize,
    path: &Path,
) -> Result<Option<usize>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(parse_u64(trimmed, field, row_no, path)? as usize))
}

fn parse_bool_flag(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<bool> {
    match raw.trim() {
        "1" => Ok(true),
        "0" => Ok(false),
        other => bail!(
            "{} row {}: invalid {} flag {:?}",
            path.display(),
            row_no,
            field,
            other
        ),
    }
}

fn parse_inline_u8(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<u8> {
    raw.trim().parse::<u8>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })
}

fn parse_inline_usize(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<usize> {
    raw.trim().parse::<usize>().with_context(|| {
        format!(
            "{} row {}: invalid {} value {:?}",
            path.display(),
            row_no,
            field,
            raw
        )
    })
}

fn parse_inline_f64(raw: &str, field: &str, row_no: usize, path: &Path) -> Result<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_prefix(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "benchman-log-loss-{label}-{}-{stamp}",
            std::process::id()
        ))
    }

    fn write_fixture(prefix: &Path) {
        let nodes = "\
node_id\tparent_id\tdepth\tpath\tdisplay_name\tbackend_label\tis_mixture\tis_leaf\tis_root_child\n\
0\t\t0\t0:root\troot\tmixture:bayes\t1\t0\t0\n\
1\t0\t1\t0:root/1:ctw\tctw\tctw(depth=4)\t0\t1\t1\n\
2\t0\t1\t0:root/2:ppmd\tppmd\tppmd(order=4,memory_mb=8)\t0\t1\t1\n";
        let summary = "\
positions\tinput_bytes\tmix_total_bits\toracle_total_bits\toracle_regret_bits\troot_top1_switch_count\toracle_switch_count\troot_weight_entropy_bits_avg\troot_top12_margin_avg\tac_payload_bits_raw\tcoder_overhead_bits\tn1__total_bits\tn1__regret_bits\tn1__oracle_win_count\tn1__avg_local_weight\tn1__avg_effective_weight\tn2__total_bits\tn2__regret_bits\tn2__oracle_win_count\tn2__avg_local_weight\tn2__avg_effective_weight\n\
4\t4\t1.20000000000000000e+01\t1.00000000000000000e+01\t2.00000000000000000e+00\t1\t1\t5.00000000000000000e-01\t3.00000000000000000e-01\t1\t-1.10000000000000000e+01\t1.02000000000000000e+01\t1.80000000000000000e+00\t2\t6.00000000000000000e-01\t6.00000000000000000e-01\t1.10000000000000000e+01\t1.00000000000000000e+00\t2\t4.00000000000000000e-01\t4.00000000000000000e-01\n";
        let trace = "\
t\tbyte_u8\tbyte_hex\tmix_prob\tmix_bits\troot_weight_entropy_bits\troot_top1_id\troot_top1_weight\troot_top2_id\troot_top2_weight\troot_top12_margin\troot_top1_switched\toracle_best_id\toracle_best_bits\toracle_regret_bits\toracle_best_switched\tn1__prob\tn1__bits\tn1__local_weight\tn1__effective_weight\tn2__prob\tn2__bits\tn2__local_weight\tn2__effective_weight\n\
0\t65\t41\t5.00000000000000000e-01\t2.00000000000000000e+00\t6.00000000000000000e-01\t1\t7.00000000000000000e-01\t2\t3.00000000000000000e-01\t4.00000000000000000e-01\t0\t1\t1.50000000000000000e+00\t5.00000000000000000e-01\t0\t3.53553390593273786e-01\t1.50000000000000000e+00\t7.00000000000000000e-01\t7.00000000000000000e-01\t2.97301778750680263e-01\t1.75000000000000000e+00\t3.00000000000000000e-01\t3.00000000000000000e-01\n\
1\t66\t42\t5.00000000000000000e-01\t3.00000000000000000e+00\t5.00000000000000000e-01\t2\t6.00000000000000000e-01\t1\t4.00000000000000000e-01\t2.00000000000000000e-01\t1\t2\t2.00000000000000000e+00\t1.00000000000000000e+00\t1\t2.50000000000000000e-01\t2.00000000000000000e+00\t4.00000000000000000e-01\t4.00000000000000000e-01\t2.50000000000000000e-01\t2.00000000000000000e+00\t6.00000000000000000e-01\t6.00000000000000000e-01\n\
2\t67\t43\t5.00000000000000000e-01\t4.00000000000000000e+00\t4.00000000000000000e-01\t1\t5.50000000000000044e-01\t2\t4.49999999999999956e-01\t1.00000000000000089e-01\t0\t1\t3.00000000000000000e+00\t1.00000000000000000e+00\t1\t1.25000000000000000e-01\t3.00000000000000000e+00\t5.50000000000000044e-01\t5.50000000000000044e-01\t6.25000000000000000e-02\t4.00000000000000000e+00\t4.49999999999999956e-01\t4.49999999999999956e-01\n\
3\t68\t44\t5.00000000000000000e-01\t3.00000000000000000e+00\t5.00000000000000000e-01\t2\t5.50000000000000044e-01\t1\t4.49999999999999956e-01\t1.00000000000000089e-01\t1\t2\t3.50000000000000000e+00\t-5.00000000000000000e-01\t0\t8.83883476483184440e-02\t3.50000000000000000e+00\t4.49999999999999956e-01\t4.49999999999999956e-01\t1.76776695296636888e-01\t2.50000000000000000e+00\t5.50000000000000044e-01\t5.50000000000000044e-01\n";

        fs::write(prefix.with_extension("nodes.tsv"), nodes).expect("write nodes");
        fs::write(prefix.with_extension("summary.tsv"), summary).expect("write summary");
        fs::write(prefix.with_extension("trace.tsv"), trace).expect("write trace");
    }

    #[test]
    fn resolve_paths_strips_known_suffixes() {
        let prefix = PathBuf::from("/tmp/example-prefix");
        let resolved = resolve_paths(&prefix.with_extension("trace.tsv")).unwrap_err();
        let message = format!("{resolved:#}");
        assert!(message.contains("/tmp/example-prefix.trace.tsv"));
    }

    #[test]
    fn load_and_interpret_fixture() {
        let prefix = temp_prefix("fixture");
        write_fixture(&prefix);
        let data = load_log_loss_data(resolve_paths(&prefix).expect("paths")).expect("data loads");
        assert_eq!(data.rows.len(), 4);
        assert_eq!(data.non_root_nodes.len(), 2);
        assert_eq!(data.bytes, b"ABCD");
        assert_eq!(data.summary.positions, 4);

        let app = LogLossApp::from_cli(&LogLossCli {
            prefix: prefix.clone(),
        })
        .expect("app builds");
        assert!(!app.current_model.series.is_empty());

        let inspection =
            inspect_region(&data, 0, 4, Thresholds::default()).expect("inspection should work");
        assert_eq!(inspection.node_stats.len(), 2);
        assert_eq!(
            inspection.worst_regret_points.len(),
            4.min(TOP_EXACT_POINTS)
        );

        let _ = fs::remove_file(prefix.with_extension("nodes.tsv"));
        let _ = fs::remove_file(prefix.with_extension("summary.tsv"));
        let _ = fs::remove_file(prefix.with_extension("trace.tsv"));
    }

    #[test]
    fn domain_partition_uses_thresholds() {
        let rows = vec![
            CompactTraceRow {
                oracle_bits: 7.0,
                best_gap_bits: 1.0,
                oracle_best_node_pos: 0,
                ..CompactTraceRow::default()
            },
            CompactTraceRow {
                oracle_bits: 3.0,
                best_gap_bits: 0.1,
                oracle_best_node_pos: 1,
                ..CompactTraceRow::default()
            },
            CompactTraceRow {
                oracle_bits: 2.0,
                best_gap_bits: 1.0,
                oracle_best_node_pos: 1,
                ..CompactTraceRow::default()
            },
        ];
        let thresholds = Thresholds::default();
        let mut uncovered = 0;
        let mut contested = 0;
        let mut assigned = vec![0usize; 2];
        for row in &rows {
            if row.oracle_bits as f64 >= thresholds.uncovered_bits {
                uncovered += 1;
            } else if (row.best_gap_bits as f64) < thresholds.contested_margin_bits {
                contested += 1;
            } else {
                assigned[row.oracle_best_node_pos as usize] += 1;
            }
        }
        assert_eq!(uncovered, 1);
        assert_eq!(contested, 1);
        assert_eq!(assigned, vec![0, 1]);
    }
}
