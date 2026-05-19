# Neuraxon Game of Life v.4.79 ui dashboard (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 171
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   "Multi-Neuraxon: Emergent Specialization, Modular, Frequency-Gated Neural Dynamics" by David Vivancos & Jose Sanchez
"""
ui/dashboard.py  (NEW in v146 / v4.79)
=======================================
Overlay dashboard rendered on top of the game world when the user presses L.

Shows live readings + history-as-you-play for the 10 paper-fidelity metrics
(M1-M10) introduced in v145. Each panel:
  - title with the section reference of the paper claim being tested
  - large current value, colour-coded by in-band / out-of-band status
  - thin sparkline of the metric over the entire game so far
  - faint horizontal band marking the paper-derived healthy range

A "Save metrics" button at the bottom-right exports a tab-separated text file
named  <GameID>__KeyMetrics.txt  with the time series of the 10 headline keys
only (one column per metric + tick + timestamp). The file is small enough to
open in any editor or load into pandas with read_csv(... sep='\\t').

Design notes
------------
* The dashboard does **NOT** drive any simulation. It just reads the existing
  logger.time_series. Metrics already populate every full-analytics tick
  (every ~10 game ticks under the standard throttle).
* Rendering is throttled — the overlay refreshes on every frame but the
  underlying data only changes once every 10 game ticks, so we cache the
  computed sparklines and only rebuild them when len(ticks) advances.
* Layout: 2 columns × 5 rows, each panel ~480x180 with 16px padding.
  Adapts to window size — minimum 1280x900 to render comfortably.
"""
import os
import math
import time
from typing import Optional, List, Tuple, Dict

import pygame

# ============================================================================
# CONFIG
# ============================================================================

PANEL_PAD = 12
ROW_COUNT = 5
COL_COUNT = 2
MIN_WIDTH = 1280
MIN_HEIGHT = 880

# Colours (consistent with existing renderer style — dark theme)
BG_OUTER         = (8, 8, 12, 235)       # near-black with alpha
BG_PANEL         = (22, 22, 30)
BG_PANEL_BORDER  = (90, 90, 110)
TEXT_HEADER      = (235, 235, 240)
TEXT_BODY        = (200, 200, 210)
TEXT_DIM         = (140, 140, 150)
TEXT_HINT        = (160, 160, 200)
COLOR_IN_BAND    = (90, 215, 130)         # green
COLOR_OUT_BAND   = (235, 95, 95)          # red
COLOR_NO_DATA    = (180, 180, 95)         # amber
COLOR_BAND_FILL  = (90, 215, 130, 35)     # translucent green band on sparkline
COLOR_SPARK      = (160, 200, 255)        # default sparkline colour
COLOR_BUTTON     = (45, 60, 90)
COLOR_BUTTON_HI  = (70, 100, 150)
COLOR_BUTTON_TXT = (240, 240, 240)
COLOR_BUTTON_OK  = (60, 150, 90)

# ============================================================================
# Per-metric panel descriptors
# ============================================================================
# Each tuple is the metric headline shown on the dashboard. The metric_key
# resolves into self.time_series; healthy_band_key resolves into the
# research_probes.HEALTHY_BANDS dict (None = display only, no in-band check).
# units is just the format suffix shown after the value.

PANEL_DESCRIPTORS: List[dict] = [
    {
        'panel_id': 'M1',
        'title': 'M1 — Trinary E/I/Neutral',
        'paper_ref': 'Neuraxon v2.0 §I/§VII',
        'subtitle': 'E should sit in [0.18, 0.28]. Neutral state is the buffer.',
        'metric_key': 'M1_excitatory_fraction',
        'metric_secondary_keys': ['M1_inhibitory_fraction', 'M1_neutral_fraction'],
        'metric_secondary_labels': ['I', 'N'],
        'metric_secondary_colours': [(255, 130, 130), (180, 180, 200)],
        'healthy_band_key': 'M1_excitatory_fraction',
        'fmt': '.3f',
        'value_label': 'E',
        'spark_colour': (130, 220, 255),
    },
    {
        'panel_id': 'M2',
        'title': 'M2 — CTC inter-sphere gate',
        'paper_ref': 'Multi-Neuraxon §3.2 (Eq. 1)',
        'subtitle': 'Healthy gate ~0.5-0.7; saturation at 1.0 = CTC dead.',
        'metric_key': 'M2_mean_gate',
        'metric_secondary_keys': ['M2_gate_modulation_std'],
        'metric_secondary_labels': ['mod σ'],
        'metric_secondary_colours': [(220, 200, 130)],
        'healthy_band_key': 'M2_mean_gate',
        'fmt': '.3f',
        'value_label': 'gate',
        'spark_colour': (255, 180, 100),
    },
    {
        'panel_id': 'M3',
        'title': 'M3 — Phase-amplitude coupling',
        'paper_ref': 'Neuraxon v2.0 §VI',
        'subtitle': 'Tort 2010 modulation index. ≈0 means oscillators independent.',
        'metric_key': 'M3_pac_modulation_idx',
        'metric_secondary_keys': ['M3_pac_delta_theta_idx'],
        'metric_secondary_labels': ['δ-θ'],
        'metric_secondary_colours': [(190, 140, 220)],
        'healthy_band_key': 'M3_pac_modulation_idx',
        'fmt': '.4f',
        'value_label': 'θ-γ MI',
        'spark_colour': (180, 140, 230),
    },
    {
        'panel_id': 'M4',
        'title': 'M4 — Multi-timescale weights',
        'paper_ref': 'Neuraxon v2.0 §III',
        'subtitle': 'corr(Δw_fast, Δw_meta). Lower = more independent timescales.',
        'metric_key': 'M4_temporal_divergence',
        'metric_secondary_keys': ['M4_w_meta_active_fraction'],
        'metric_secondary_labels': ['w_meta'],
        'metric_secondary_colours': [(150, 230, 180)],
        'healthy_band_key': 'M4_temporal_divergence',
        'fmt': '.3f',
        'value_label': 'corr',
        'spark_colour': (130, 200, 200),
    },
    {
        'panel_id': 'M5',
        'title': 'M5 — Branching ratio σ',
        'paper_ref': 'Neuraxon v2.0 §V',
        'subtitle': 'Self-organised criticality target σ ≈ 1.0.',
        'metric_key': 'M5_branching_ratio',
        'metric_secondary_keys': ['M5_distance_from_critical'],
        'metric_secondary_labels': ['|σ-1|'],
        'metric_secondary_colours': [(220, 130, 130)],
        'healthy_band_key': 'M5_branching_ratio',
        'fmt': '.3f',
        'value_label': 'σ',
        'spark_colour': (130, 240, 130),
    },
    {
        'panel_id': 'M6',
        'title': 'M6 — Spontaneous + ACW',
        'paper_ref': 'Neuraxon v2.0 §V',
        'subtitle': 'Spontaneous fraction & timescale heterogeneity.',
        'metric_key': 'M6_spontaneous_fraction',
        'metric_secondary_keys': ['M6_acw_heterogeneity'],
        'metric_secondary_labels': ['ACW σ'],
        'metric_secondary_colours': [(220, 200, 130)],
        'healthy_band_key': 'M6_spontaneous_fraction',
        'fmt': '.3f',
        'value_label': 'spont',
        'spark_colour': (220, 220, 130),
    },
    {
        'panel_id': 'M7',
        'title': 'M7 — Self-sustained activity',
        'paper_ref': 'Multi-Neuraxon §3.4 / §3.8',
        'subtitle': 'The Nengo-distinguishing signal. Zero-input motor / driven motor.',
        'metric_key': 'M7_zero_input_mi_ratio',
        'metric_secondary_keys': ['M7_n_paired_nxers'],
        'metric_secondary_labels': ['n'],
        'metric_secondary_colours': [TEXT_DIM],
        'healthy_band_key': 'M7_zero_input_mi_ratio',
        'fmt': '.3f',
        'value_label': 'ratio',
        'spark_colour': (255, 130, 200),
    },
    {
        'panel_id': 'M8',
        'title': 'M8 — Sphere specialisation',
        'paper_ref': 'Multi-Neuraxon §3.5',
        'subtitle': 'Sensory selectivity should rise above association.',
        'metric_key': 'M8_sensory_vs_association_dissociation',
        'metric_secondary_keys': [],
        'metric_secondary_labels': [],
        'metric_secondary_colours': [],
        'healthy_band_key': None,  # not in HEALTHY_BANDS — display only
        'fmt': '.3f',
        'value_label': 'spec',
        'spark_colour': (180, 230, 180),
    },
    {
        'panel_id': 'M9',
        'title': 'M9 — Compositional transfer',
        'paper_ref': 'Multi-Neuraxon §3.7',
        'subtitle': 'Novel V×A combinations / trained pairs. Target ≥ 1.0.',
        'metric_key': 'M9_transfer_ratio',
        'metric_secondary_keys': ['M9_compositional_similarity'],
        'metric_secondary_labels': ['sim'],
        'metric_secondary_colours': [(180, 200, 230)],
        'healthy_band_key': 'M9_transfer_ratio',
        'fmt': '.3f',
        'value_label': 'R/T',
        'spark_colour': (255, 200, 130),
    },
    {
        'panel_id': 'M10',
        'title': 'M10 — Heritability + lesion',
        'paper_ref': 'Aigarth §VIII; Multi-Neuraxon §3.9',
        'subtitle': 'Parent-child fitness correlation. >0.3 means evolution works.',
        'metric_key': 'M10_heritability_r',
        'metric_secondary_keys': ['M10_lesion_retention_50', 'M10_lesion_retention_75'],
        'metric_secondary_labels': ['lesion-50', 'lesion-75'],
        'metric_secondary_colours': [(180, 230, 220), (140, 200, 200)],
        'healthy_band_key': 'M10_heritability_r',
        'fmt': '.3f',
        'value_label': 'r',
        'spark_colour': (255, 200, 200),
    },
]


# ============================================================================
# Helper drawing primitives
# ============================================================================

def _draw_translucent_rect(surface: pygame.Surface, rect: pygame.Rect,
                            colour: Tuple[int, int, int, int],
                            border_radius: int = 0):
    """Solid + alpha — draw via a temporary per-pixel surface."""
    s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(s, colour, s.get_rect(), border_radius=border_radius)
    surface.blit(s, rect.topleft)


def _draw_sparkline(surface: pygame.Surface, rect: pygame.Rect,
                    series: List[float], colour: Tuple[int, int, int],
                    band: Optional[Tuple[float, float]] = None,
                    bg_alpha: int = 0):
    """Draw a sparkline for `series` inside `rect`. Band, if provided, is
    drawn as a translucent green horizontal stripe behind the line."""
    if bg_alpha > 0:
        _draw_translucent_rect(surface, rect, (16, 16, 22, bg_alpha), border_radius=4)
    # axis frame
    pygame.draw.rect(surface, (50, 55, 68), rect, 1, border_radius=4)
    if not series:
        return
    n = len(series)
    if n < 2:
        return
    # Compute y-axis range. We use min/max of the series, but if a healthy
    # band is given, expand the range to include it so it's always visible.
    y_min = min(series)
    y_max = max(series)
    if band:
        b_lo, b_hi = band
        y_min = min(y_min, b_lo)
        y_max = max(y_max, b_hi)
    # pad slightly
    if y_max - y_min < 1e-6:
        y_min -= 0.5; y_max += 0.5
    pad = (y_max - y_min) * 0.10
    y_min -= pad; y_max += pad

    def y2px(v):
        # invert y: top of rect = high value
        return rect.bottom - int((v - y_min) / (y_max - y_min) * (rect.height - 4)) - 2

    # Draw band (translucent green) behind the line
    if band:
        b_lo, b_hi = band
        band_top = y2px(b_hi)
        band_bot = y2px(b_lo)
        if band_bot > band_top:
            band_rect = pygame.Rect(rect.x + 1, band_top,
                                     rect.width - 2, band_bot - band_top)
            _draw_translucent_rect(surface, band_rect, COLOR_BAND_FILL,
                                    border_radius=2)

    # Draw line. Compress to at most rect.width-2 segments.
    inner_w = rect.width - 4
    if n > inner_w:
        # decimate
        step = n / inner_w
        sampled = [series[int(i * step)] for i in range(inner_w)]
        n2 = len(sampled)
    else:
        sampled = series
        n2 = n
    pts = []
    for i, v in enumerate(sampled):
        x = rect.x + 2 + int(i * (rect.width - 4) / max(1, n2 - 1))
        y = y2px(v)
        pts.append((x, y))
    if len(pts) >= 2:
        pygame.draw.lines(surface, colour, False, pts, 2)
    # draw last point as a small dot
    if pts:
        pygame.draw.circle(surface, colour, pts[-1], 3)


# ============================================================================
# Main dashboard class
# ============================================================================

class MetricsDashboard:
    """Real-time overlay of the 10 v145 paper-fidelity metrics.
    
    Lifecycle:
        d = MetricsDashboard(screen)            # once at game start
        if d.visible: d.handle_event(ev)        # in event loop
        if d.visible: d.draw(logger, game_id)   # once per frame
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.visible = False
        # caches keyed by len(time_series['ticks']) so we only rebuild when
        # the underlying data has actually advanced.
        self._cached_at_len: int = -1
        self._cached_series: Dict[str, List[float]] = {}
        self._cached_ticks: List[int] = []
        # Healthy-bands dict — fetched lazily from research_probes
        self._healthy_bands: Optional[dict] = None
        # Buttons defined per-frame; resolved on click
        self._buttons: Dict[str, pygame.Rect] = {}
        # Save feedback
        self._last_save_path: Optional[str] = None
        self._last_save_at_ts: float = 0.0
        # Fonts — shared with renderer style (consolas, cached pygame fonts)
        self._font_h1   = pygame.font.SysFont("consolas", 26, bold=True)
        self._font_h2   = pygame.font.SysFont("consolas", 18, bold=True)
        self._font_body = pygame.font.SysFont("consolas", 14)
        self._font_tiny = pygame.font.SysFont("consolas", 11)
        self._font_big_value = pygame.font.SysFont("consolas", 32, bold=True)
        # v158 (v4.79) — VIEW SELECTOR (ComboBox)
        # =======================================
        # The dashboard now supports two view modes:
        #   "aggregate" — population-level M1-M10 + survivability strip
        #                 (the v144-v157 default behaviour)
        #   <nxer_id>  — per-NxEr view: that specific agent's stats &
        #                network metrics in 10 panels, drawn from the
        #                logger's per_nxer_time_series (populated in
        #                logger._log_nxer_individual). When the selected
        #                NxEr dies or no longer exists, falls back to
        #                "aggregate" gracefully.
        # The combo box is rebuilt per-frame from current champions so
        # users see the live #1 in each category as they evolve.
        from .widgets import ComboBox
        self._view_combo = ComboBox(
            rect=pygame.Rect(0, 0, 360, 26),   # position set at draw time
            options=[("Overall aggregate", "aggregate")],
            default_value="aggregate",
        )
        # Reference to the current NxEr dict — set by game_loop just before
        # calling draw() so the dashboard can populate the combo box and
        # render per-NxEr views. None means "no live nxers data available
        # — show aggregate only".
        self._nxers_ref: Optional[Dict] = None

    # ---- lifecycle ----
    def toggle(self):
        self.visible = not self.visible

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    # ---- event handling ----
    def handle_event(self, ev: pygame.event.Event, logger, game_id: str) -> bool:
        """Returns True if the event was consumed by the dashboard (so
        the caller should not pass it on to other handlers)."""
        if not self.visible:
            return False
        # v158 — let the ComboBox handle the event first. If it consumes
        # (opens, selects an option, etc.), we're done. The combo box's
        # own click-outside-to-close logic returns False so other dashboard
        # handlers still see the event.
        if self._view_combo.handle_event(ev):
            return True
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_l or ev.key == pygame.K_ESCAPE:
                self.hide()
                return True
            if ev.key == pygame.K_k:
                # Save key-metrics file
                self._save_metrics_txt(logger, game_id)
                return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for key, rect in self._buttons.items():
                if rect.collidepoint(ev.pos):
                    if key == 'close':
                        self.hide()
                    elif key == 'save_metrics':
                        self._save_metrics_txt(logger, game_id)
                    return True
            # absorb clicks anywhere on the overlay so they don't fall through
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN:
            return True
        if ev.type == pygame.MOUSEMOTION:
            # don't consume motion — let it pass for other tooltips, but the
            # overlay is opaque so this isn't really necessary
            return False
        return False

    # ---- v158 — set NxErs reference for per-NxEr view ----
    def set_nxers(self, nxers: Dict):
        """Called by game_loop before draw() so the dashboard can build
        the per-champion combo box options and render per-NxEr panels.
        Pass None or {} during headless / NAS runs — the combo box will
        just have the "Overall aggregate" option."""
        self._nxers_ref = nxers
    
    def _refresh_combo_options(self):
        """Rebuild combo box options from current rankings. Called each
        frame at draw time. Each champion category contributes one option:
        the current top-stat live NxEr."""
        options = [("Overall aggregate", "aggregate")]
        if self._nxers_ref:
            alive = [a for a in self._nxers_ref.values() if getattr(a, 'alive', False)]
            if alive:
                category_specs = [
                    ('Fitness',        'fitness_score',   '{:.2f}'),
                    ('Food Found',     'food_found',      '{:.1f}'),
                    ('Food Stolen',    'food_taken',      '{:.1f}'),
                    ('Mates',          'mates_performed', '{}'),
                    ('Explorer',       'explored',        '{}'),
                    ('Time Lived',     'time_lived_s',    '{:.0f}s'),
                ]
                for cat_label, attr, fmt in category_specs:
                    try:
                        champ = max(alive,
                                    key=lambda a: getattr(a.stats, attr, 0) or 0)
                        val = getattr(champ.stats, attr, 0) or 0
                        try:
                            val_str = fmt.format(val)
                        except (ValueError, TypeError):
                            val_str = str(val)
                        label = f"Best {cat_label}: {champ.name} ({val_str})"
                        options.append((label, champ.id))
                    except Exception:
                        # Defensive — never let combo-box construction break the dashboard
                        continue
        self._view_combo.set_options(options, preserve_value=True)

    # ---- save-to-file ----
    def _save_metrics_txt(self, logger, game_id: str) -> Optional[str]:
        """v155 (v4.79) — thin wrapper around DataLogger.save_key_metrics().
        Kept for API compatibility (K-key and Save-button still call this).
        The actual file-writing lives on the logger so the auto-save in
        game_loop.finally can call it directly without a UI surface."""
        if logger is None:
            return None
        try:
            saved = logger.save_key_metrics(game_id)
        except Exception as exc:
            print(f"[DASHBOARD] Save failed: {exc}")
            return None
        if saved:
            self._last_save_path = saved
            self._last_save_at_ts = time.time()
            try:
                n = len(logger.time_series.get('ticks', []))
            except Exception:
                n = 0
            print(f"[DASHBOARD] Saved {n} samples to {saved}")
            # v154 — companion MembraneDiag save (K-press writes both)
            try:
                diag_path = logger.save_membrane_diagnostics(game_id)
                if diag_path:
                    print(f"[DASHBOARD] Membrane diag → {diag_path}")
            except Exception as exc:
                print(f"[DASHBOARD] Membrane diag save failed: {exc}")
        return saved

    # ---- main draw ----
    def draw(self, logger, game_id: str):
        if not self.visible:
            return
        if self._healthy_bands is None:
            try:
                from neuraxon.research_probes import HEALTHY_BANDS
                self._healthy_bands = HEALTHY_BANDS
            except ImportError:
                self._healthy_bands = {}
        self._buttons.clear()
        W, H = self.screen.get_size()
        # full-screen translucent backdrop
        backdrop = pygame.Surface((W, H), pygame.SRCALPHA)
        backdrop.fill(BG_OUTER)
        self.screen.blit(backdrop, (0, 0))

        # outer frame
        margin = max(20, (W - MIN_WIDTH) // 8)
        if margin > 80: margin = 80
        outer = pygame.Rect(margin, margin // 2, W - 2 * margin, H - margin)
        pygame.draw.rect(self.screen, BG_PANEL, outer, border_radius=14)
        pygame.draw.rect(self.screen, BG_PANEL_BORDER, outer, 2, border_radius=14)

        # Header
        self._draw_header(outer, logger, game_id)
        
        # v158 (v4.79) — VIEW COMBO BOX
        # Refresh options from current champions, then position & draw
        # the closed box. The OPEN dropdown is drawn at end-of-frame so
        # it sits on top of all other content (z-order).
        self._refresh_combo_options()
        combo_x = outer.x + PANEL_PAD
        combo_y = outer.y + 60  # below the title line in header
        self._view_combo.rect = pygame.Rect(combo_x, combo_y, 360, 26)
        self._view_combo.draw_closed(self.screen, self._font_body)
        # "View:" label to the left of the box
        lbl = self._font_tiny.render("View:", True, (170, 175, 180))
        self.screen.blit(lbl, (combo_x, combo_y - 13))

        # Refresh cache if data has advanced
        ts = logger.time_series if logger is not None and hasattr(logger, 'time_series') else {}
        cur_len = len(ts.get('ticks', []))
        if cur_len != self._cached_at_len:
            self._refresh_cache(ts)
            self._cached_at_len = cur_len
        
        # v158 — DISPATCH on current view selection
        view_value = self._view_combo.get_value()
        if view_value == "aggregate" or view_value is None:
            self._draw_aggregate_view(outer, logger)
        else:
            # Per-NxEr view — find the agent, fall back to aggregate if dead/missing
            nxer = None
            if self._nxers_ref and isinstance(view_value, int):
                nxer = self._nxers_ref.get(view_value)
                if nxer is not None and not getattr(nxer, 'alive', False):
                    nxer = None
            if nxer is None:
                # Selected NxEr is gone — fall back, and reset to aggregate
                self._view_combo.selected_index = 0
                self._draw_aggregate_view(outer, logger)
            else:
                self._draw_per_nxer_view(outer, logger, nxer)

        # Footer
        self._draw_footer(outer)
        
        # v158 — draw the OPEN dropdown LAST so it sits on top of the
        # whole dashboard (z-order). If the combo isn't open this is a
        # no-op.
        self._view_combo.draw_open(self.screen, self._font_body)
    
    def _draw_aggregate_view(self, outer: pygame.Rect, logger):
        """v158 — extracted as a method so the per-NxEr view can be a
        sibling. Renders the v144-v157 default dashboard: survivability
        strip + worker stats line + 10-panel M1-M10 grid."""
        # v147 — Survivability strip sits between the header and the metric grid
        SURV_STRIP_HEIGHT = 76
        surv_rect = pygame.Rect(outer.x + PANEL_PAD,
                                outer.y + 88 + 32,  # +32 for combo box
                                outer.width - 2 * PANEL_PAD,
                                SURV_STRIP_HEIGHT)
        self._draw_survivability_strip(surv_rect, logger)

        # v147 — worker-stats one-line indicator under the survivability strip
        WORKER_LINE_H = 22
        wstats_rect = pygame.Rect(outer.x + PANEL_PAD,
                                   surv_rect.bottom + 4,
                                   outer.width - 2 * PANEL_PAD,
                                   WORKER_LINE_H)
        self._draw_worker_stats_line(wstats_rect, logger)

        # Compute panel grid layout (now starts BELOW the survivability strip + worker line)
        grid_top    = wstats_rect.bottom + 8
        grid_bottom = outer.bottom - 90
        grid_left   = outer.x + PANEL_PAD
        grid_right  = outer.right - PANEL_PAD
        avail_w = grid_right - grid_left - PANEL_PAD * (COL_COUNT - 1)
        avail_h = grid_bottom - grid_top - PANEL_PAD * (ROW_COUNT - 1)
        panel_w = avail_w // COL_COUNT
        panel_h = avail_h // ROW_COUNT

        for idx, desc in enumerate(PANEL_DESCRIPTORS):
            r = idx // COL_COUNT
            c = idx % COL_COUNT
            x = grid_left + c * (panel_w + PANEL_PAD)
            y = grid_top  + r * (panel_h + PANEL_PAD)
            self._draw_panel(pygame.Rect(x, y, panel_w, panel_h), desc)
    
    def _draw_per_nxer_view(self, outer: pygame.Rect, logger, nxer):
        """v158 (v4.79) — Per-NxEr dashboard view. Replaces the
        population-aggregate M1-M10 grid with 10 panels showing this one
        agent's metrics, computed from logger.per_nxer_time_series and
        current network state.
        
        Layout:
          Row 1: Vitals    | Stats         | Network basics
          Row 2: Food/take | Mates/explore | Time/fitness
          Row 3: E/I/N     | Energy/branch | Phase coherence
          Row 4: Recent moves | Receptors  | Voice/sing
        """
        # === HEADER STRIP: this NxEr's identity ===
        STRIP_H = 76
        info_rect = pygame.Rect(outer.x + PANEL_PAD,
                                 outer.y + 88 + 32,  # +32 for combo box
                                 outer.width - 2 * PANEL_PAD,
                                 STRIP_H)
        pygame.draw.rect(self.screen, (24, 30, 40), info_rect, border_radius=6)
        pygame.draw.rect(self.screen, (90, 110, 130), info_rect, 1, border_radius=6)
        # Name + colour swatch + alive status
        col = getattr(nxer, 'color', (200, 200, 200))
        pygame.draw.circle(self.screen, col,
                            (info_rect.x + 22, info_rect.y + 22), 12)
        pygame.draw.circle(self.screen, (255, 255, 255),
                            (info_rect.x + 22, info_rect.y + 22), 12, 1)
        # Name + id
        name_text = self._font_h2.render(
            f"NxEr {nxer.name} (id={nxer.id})", True, (240, 240, 240))
        self.screen.blit(name_text, (info_rect.x + 42, info_rect.y + 10))
        # Status: alive, rounds_survived, age
        alive_color = (120, 220, 130) if getattr(nxer, 'alive', False) else (220, 100, 100)
        status_text = self._font_tiny.render(
            f"alive={getattr(nxer, 'alive', False)}  "
            f"round={getattr(nxer, 'rounds_survived', 0)}  "
            f"resting={getattr(nxer, 'is_resting', False)}  "
            f"temp={getattr(nxer, 'body_temperature', 37.0):.1f}°C",
            True, alive_color)
        self.screen.blit(status_text, (info_rect.x + 42, info_rect.y + 36))
        # Hint on right
        hint_text = self._font_tiny.render(
            "View: per-NxEr — Use combo above to switch back to Overall aggregate",
            True, (160, 165, 175))
        self.screen.blit(hint_text,
                          (info_rect.right - hint_text.get_width() - 12,
                           info_rect.y + 50))
        
        # === 12 stats panels (3x4 grid) ===
        # Pull per-NxEr time series if available
        pnts = (logger.per_nxer_time_series.get(nxer.id, {})
                if logger and hasattr(logger, 'per_nxer_time_series') else {})
        
        # Panel descriptors: (title, value_str, sparkline_key, sparkline_values, band, units)
        stats = getattr(nxer, 'stats', None)
        food_now = getattr(nxer, 'food', 0.0)
        # Helper to read last value of a per-NxEr time series, defaulting safely
        def _last(key, default=0.0):
            seq = pnts.get(key, [])
            return seq[-1] if seq else default
        
        # Compute current E/I/N from per_nxer last samples if available
        ei_e = _last('excitatory_fraction', 0.0)
        ei_i = _last('inhibitory_fraction', 0.0)
        ei_n = _last('neutral_fraction', 0.0)
        branch_now = _last('branching_ratio', 1.0)
        energy_now = _last('average_energy', 0.0)
        phase_coh = _last('phase_coherence', 0.0)
        harm = _last('voice_harmonicity', 0.0)
        sing_lvl = _last('sing_level', 0.0)
        mp_mean = _last('membrane_potential_mean', 0.0)
        mp_std = _last('membrane_potential_std', 0.0)
        
        panel_specs = [
            # Row 1
            {'title': 'Vitals',
             'value': f"{food_now:.1f}",
             'unit': 'food',
             'sub': f"E={ei_e:.2f}  I={ei_i:.2f}  N={ei_n:.2f}",
             'series': pnts.get('food', []),
             'series_label': 'food',
             'color_value': (210, 200, 90)},
            {'title': 'Fitness score',
             'value': f"{getattr(stats, 'fitness_score', 0.0):.3f}",
             'unit': 'score',
             'sub': '',
             'series': pnts.get('fitness_score', []),
             'series_label': 'fitness',
             'color_value': (220, 220, 90)},
            {'title': 'Time lived',
             'value': f"{getattr(stats, 'time_lived_s', 0.0):.0f}",
             'unit': 's',
             'sub': f"round {getattr(nxer, 'rounds_survived', 0)}",
             'series': [],  # time monotonic — sparkline not useful
             'series_label': '',
             'color_value': (180, 220, 220)},
            # Row 2
            {'title': 'Food found',
             'value': f"{getattr(stats, 'food_found', 0):.1f}",
             'unit': 'units',
             'sub': '',
             'series': pnts.get('food_found', []),
             'series_label': 'cumul.',
             'color_value': (110, 200, 130)},
            {'title': 'Food stolen',
             'value': f"{getattr(stats, 'food_taken', 0):.1f}",
             'unit': 'units',
             'sub': 'from clan',
             'series': [],
             'series_label': '',
             'color_value': (220, 130, 90)},
            {'title': 'Mates',
             'value': f"{getattr(stats, 'mates_performed', 0)}",
             'unit': '',
             'sub': '',
             'series': pnts.get('mates_performed', []),
             'series_label': 'cumul.',
             'color_value': (220, 140, 200)},
            # Row 3
            {'title': 'World explored',
             'value': f"{getattr(stats, 'explored', 0)}",
             'unit': 'tiles',
             'sub': '',
             'series': pnts.get('explored', []),
             'series_label': 'cumul.',
             'color_value': (130, 200, 220)},
            {'title': 'Energy',
             'value': f"{energy_now:.1f}",
             'unit': '',
             'sub': f"branching σ={branch_now:.2f}",
             'series': pnts.get('average_energy', []),
             'series_label': 'avg energy',
             'color_value': (240, 200, 90)},
            {'title': 'Phase coherence',
             'value': f"{phase_coh:.3f}",
             'unit': '',
             'sub': f"mp mean={mp_mean:+.2f}, σ={mp_std:.2f}",
             'series': pnts.get('phase_coherence', []),
             'series_label': 'coh',
             'color_value': (180, 180, 220)},
            # Row 4
            {'title': 'Dopamine',
             'value': f"{_last('dopamine', 0.0):.3f}",
             'unit': '',
             'sub': f"5HT={_last('serotonin', 0.0):.2f}  ACh={_last('acetylcholine', 0.0):.2f}  NE={_last('norepinephrine', 0.0):.2f}",
             'series': pnts.get('dopamine', []),
             'series_label': 'DA',
             'color_value': (220, 150, 200)},
            {'title': 'Voice harmonicity',
             'value': f"{harm:.2f}",
             'unit': '',
             'sub': f"sing level={sing_lvl:.0f}  freq={_last('voice_base_freq', 0.0):.0f}Hz",
             'series': pnts.get('voice_harmonicity', []),
             'series_label': 'harm',
             'color_value': (200, 220, 120)},
            {'title': 'Network activity',
             'value': f"{_last('network_activity', 0.0):.3f}",
             'unit': '',
             'sub': f"forced turns={getattr(getattr(nxer, 'proprioceptron', None), 'forced_turn_count', 0)}  rocks={getattr(getattr(nxer, 'proprioceptron', None), 'total_rock_hits', 0)}",
             'series': pnts.get('network_activity', []),
             'series_label': 'act',
             'color_value': (160, 200, 240)},
        ]
        
        # Grid layout — 4 rows × 3 cols
        PER_NXER_ROWS = 4
        PER_NXER_COLS = 3
        grid_top    = info_rect.bottom + 8
        grid_bottom = outer.bottom - 90
        grid_left   = outer.x + PANEL_PAD
        grid_right  = outer.right - PANEL_PAD
        avail_w = grid_right - grid_left - PANEL_PAD * (PER_NXER_COLS - 1)
        avail_h = grid_bottom - grid_top - PANEL_PAD * (PER_NXER_ROWS - 1)
        panel_w = avail_w // PER_NXER_COLS
        panel_h = avail_h // PER_NXER_ROWS
        for idx, spec in enumerate(panel_specs[:PER_NXER_ROWS * PER_NXER_COLS]):
            r = idx // PER_NXER_COLS
            c = idx % PER_NXER_COLS
            x = grid_left + c * (panel_w + PANEL_PAD)
            y = grid_top  + r * (panel_h + PANEL_PAD)
            self._draw_per_nxer_panel(pygame.Rect(x, y, panel_w, panel_h), spec)
    
    def _draw_per_nxer_panel(self, rect: pygame.Rect, spec: dict):
        """Draw one panel of the per-NxEr view. Layout matches the
        aggregate panel: big value top-left, optional sub-line, sparkline."""
        # Background
        pygame.draw.rect(self.screen, (22, 28, 38), rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 80, 100), rect, 1, border_radius=6)
        # Title (top left, small)
        title_surf = self._font_tiny.render(spec['title'], True, (170, 175, 185))
        self.screen.blit(title_surf, (rect.x + 8, rect.y + 6))
        # Big value
        col = spec.get('color_value', (220, 220, 220))
        value_surf = self._font_big_value.render(spec['value'], True, col)
        self.screen.blit(value_surf, (rect.x + 8, rect.y + 18))
        # Unit (right of value)
        unit = spec.get('unit', '')
        if unit:
            unit_surf = self._font_tiny.render(unit, True, (140, 145, 155))
            self.screen.blit(unit_surf, (rect.x + 8 + value_surf.get_width() + 4,
                                          rect.y + 18 + value_surf.get_height() - 14))
        # Sub line
        sub = spec.get('sub', '')
        if sub:
            sub_surf = self._font_tiny.render(sub, True, (180, 185, 195))
            self.screen.blit(sub_surf, (rect.x + 8, rect.y + 56))
        # Sparkline (right half)
        series = spec.get('series', [])
        if series and len(series) >= 2:
            sx = rect.x + rect.width // 2
            sy = rect.y + 22
            sw = rect.width // 2 - 12
            sh = rect.height - 32
            # Subsample to fit width
            n = len(series)
            stride = max(1, n // sw)
            samples = series[::stride][-sw:]
            if samples:
                lo = min(samples)
                hi = max(samples)
                rng = hi - lo if hi > lo else 1.0
                points = []
                for i, v in enumerate(samples):
                    px = sx + int(i * sw / max(1, len(samples) - 1))
                    py = sy + sh - int((v - lo) / rng * sh)
                    points.append((px, py))
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, col, False, points, 2)
            # n samples label
            n_lbl = self._font_tiny.render(
                f"n={len(series)}  [{spec.get('series_label', '')}]",
                True, (130, 135, 145))
            self.screen.blit(n_lbl, (sx, rect.bottom - 16))
    
    # ---- v147 survivability strip ----
    def _draw_survivability_strip(self, rect: pygame.Rect, logger):
        """Population-health top strip — addresses the 'NxErs stop moving and
        die' concern with five concrete numbers + a sparkline of the
        composite score over time + colour-coded health label."""
        # Background
        pygame.draw.rect(self.screen, (24, 32, 28), rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 100, 90), rect, 1, border_radius=6)
        try:
            sd = logger.get_survivability_dashboard() if logger else None
        except Exception:
            sd = None
        if sd is None:
            no_data = self._font_body.render("survivability: no data yet",
                                                True, TEXT_DIM)
            self.screen.blit(no_data, (rect.x + 12, rect.y + 18))
            return

        # Left side: composite score gauge + label
        gauge_w = 260
        gauge_x = rect.x + 14
        gauge_y = rect.y + 12
        # title
        title = self._font_h2.render("Survivability", True, TEXT_HEADER)
        self.screen.blit(title, (gauge_x, gauge_y))
        # big value
        score_pct = sd['survivability_score'] * 100.0
        score_str = f"{score_pct:5.1f}%"
        score_surf = self._font_big_value.render(score_str, True, sd['health_colour'])
        self.screen.blit(score_surf, (gauge_x, gauge_y + 24))
        # label
        lab_surf = self._font_body.render(sd['health_label'], True, sd['health_colour'])
        self.screen.blit(lab_surf, (gauge_x + score_surf.get_width() + 14,
                                      gauge_y + 28 + (score_surf.get_height() - lab_surf.get_height())))
        # gauge bar
        bar_x = gauge_x
        bar_y = gauge_y + 24 + score_surf.get_height() + 2
        bar_w = gauge_w - 14
        bar_h = 6
        # outer track
        pygame.draw.rect(self.screen, (60, 60, 70),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=3)
        # filled portion in health colour
        fill_w = int(bar_w * sd['survivability_score'])
        if fill_w > 0:
            pygame.draw.rect(self.screen, sd['health_colour'],
                             pygame.Rect(bar_x, bar_y, fill_w, bar_h),
                             border_radius=3)

        # Middle: stat blocks
        stats_x = gauge_x + gauge_w + 12
        # v149 — added "N stuck" block exposing neuron-level lock-in directly.
        # This is the "smoking gun" indicator for the M1 stuck-at-+1 pathology
        # that v149 step 1 (stronger adaptation) targets. If the v149 fix
        # works, this number stays low (< 30%); if it doesn't, you'll see
        # it climb past 50% and lock there.
        stuck_pos1 = sd.get('stuck_fraction_at_pos1', 0.0)
        # v150 — sensory→motor coupling. The "S→M corr" is the smoking-gun
        # indicator for the v150 sensory-input-overwhelmed-by-recurrence
        # pathology. v149 sample run had this near 0; v150 fix targets
        # > 0.20 (positive coupling, motor follows sensory).
        sm_corr = sd.get('sensory_motor_corr', 0.0)
        # v151 — input saturation indicator. The v150 sample run reached 1.0
        # at tick 600 and stayed there — that's WHY S→M corr collapsed.
        # Healthy < 0.30; saturating > 0.50 is the alarm.
        sat = sd.get('input_saturation_fraction', 0.0)
        # v152 — pop_mean_idle_seconds is the SMOKING-GUN indicator for the
        # "stop moving and die" pathology. v151 sample showed this reaching
        # 14 seconds while the population stood still. Healthy < 1.5s; > 5s
        # is critical (atrophy compounding even with the cap).
        idle_pop = sd.get('pop_mean_idle_seconds', 0.0)
        # v152 — exploration_trigger_rate: tells us if the v152 safety net
        # is firing. In v151 this was 0.0 ALWAYS (bug — never fired).
        # In v152 with the bug fixed, expect 0.05-0.30 in healthy state
        # (occasional fallback), > 0.50 means networks are mostly broken.
        explr = sd.get('exploration_trigger_rate', 0.0)
        # v153 — input_locked_fraction is the TRUE saturation pathology
        # (input neurons stuck in ONE fixed state for 30 ticks, no
        # transitions). Replaces InSat in the strip since InSat conflated
        # constant signals (fine) with locked states (bad). Healthy < 0.20.
        locked = sd.get('input_locked_fraction', 0.0)
        stats = [
            ("Alive",        f"{sd['alive_count']}",          TEXT_HEADER),
            ("Mean food",    f"{sd['mean_food']:.1f}",        TEXT_BODY),
            ("Moving",       f"{sd['mean_motion_rate']*100:.1f}%",
             COLOR_IN_BAND if sd['mean_motion_rate'] >= 0.30 else
             COLOR_OUT_BAND if sd['mean_motion_rate'] < 0.10 else COLOR_NO_DATA),
            ("Pop idle",     f"{idle_pop:.1f}s",
             COLOR_IN_BAND if idle_pop < 1.5 else
             COLOR_OUT_BAND if idle_pop >= 5.0 else COLOR_NO_DATA),
            ("N@+1 stuck",   f"{stuck_pos1*100:.1f}%",
             COLOR_IN_BAND if stuck_pos1 < 0.30 else
             COLOR_OUT_BAND if stuck_pos1 >= 0.50 else COLOR_NO_DATA),
            ("S→M corr",     f"{sm_corr:+.2f}",
             COLOR_IN_BAND if sm_corr >= 0.20 else
             COLOR_OUT_BAND if sm_corr < 0.05 else COLOR_NO_DATA),
            ("Locked",       f"{locked*100:.0f}%",
             COLOR_IN_BAND if locked < 0.20 else
             COLOR_OUT_BAND if locked >= 0.50 else COLOR_NO_DATA),
            ("Explore",      f"{explr*100:.0f}%",
             COLOR_IN_BAND if 0.01 <= explr <= 0.40 else
             COLOR_OUT_BAND if explr > 0.60 else COLOR_NO_DATA),
            ("B/D 200t",
             f"{sd['births_window']}/{sd['deaths_window']}",
             COLOR_IN_BAND if sd['births_window'] > sd['deaths_window'] else
             COLOR_OUT_BAND if sd['births_window'] * 2 < sd['deaths_window'] else COLOR_NO_DATA),
            ("Lifespan",     f"{sd['mean_lifespan_s']:.0f}s",  TEXT_BODY),
        ]
        block_w = 84
        for i, (label, value, colour) in enumerate(stats):
            bx = stats_x + i * block_w
            if bx + block_w > rect.right - 240:   # leave room for sparkline
                break
            # label
            lab = self._font_tiny.render(label, True, TEXT_DIM)
            self.screen.blit(lab, (bx, rect.y + 12))
            # value
            val = self._font_h2.render(value, True, colour)
            self.screen.blit(val, (bx, rect.y + 28))

        # Right: sparkline of survivability score over time
        spark_w = 220
        spark_x = rect.right - spark_w - 14
        spark_rect = pygame.Rect(spark_x, rect.y + 14,
                                 spark_w, rect.height - 28)
        ts = logger.time_series if logger else {}
        score_series = list(ts.get('surv_score', []))
        if score_series:
            _draw_sparkline(self.screen, spark_rect, score_series,
                            sd['health_colour'], band=(0.65, 1.00),
                            bg_alpha=180)
            spark_lab = self._font_tiny.render(
                f"score history (target ≥ 65%, n={len(score_series)})",
                True, TEXT_DIM)
            self.screen.blit(spark_lab,
                              (spark_rect.x, spark_rect.bottom + 1))
        else:
            pygame.draw.rect(self.screen, (50, 55, 68), spark_rect, 1, border_radius=4)
            no = self._font_tiny.render("no samples yet", True, TEXT_DIM)
            self.screen.blit(no, (spark_rect.x + 4, spark_rect.y + 8))
    
    # ---- v147 worker stats line ----
    def _draw_worker_stats_line(self, rect: pygame.Rect, logger):
        """Single-line strip showing the metrics-worker thread health.
        Displays: thread enabled? | mean compute ms | queue depth |
        jobs processed | jobs dropped | last error if any."""
        try:
            ws = logger.get_metrics_worker_stats() if logger else {'enabled': False}
        except Exception:
            ws = {'enabled': False}
        # Background
        pygame.draw.rect(self.screen, (20, 24, 30), rect, border_radius=4)
        pygame.draw.rect(self.screen, (60, 65, 80), rect, 1, border_radius=4)
        x = rect.x + 10
        y = rect.y + (rect.height - self._font_tiny.get_height()) // 2
        if not ws.get('enabled', False):
            txt = self._font_tiny.render(
                "metrics worker: DISABLED — running inline (slower)",
                True, COLOR_NO_DATA)
            self.screen.blit(txt, (x, y))
            return
        parts = [
            ("worker:", "ON",                                  COLOR_IN_BAND),
            (f"  tid={ws.get('thread_id', 0)}", "",            TEXT_DIM),
            (f"  mean_compute={ws.get('mean_compute_ms', 0):.2f}ms", "",
             COLOR_IN_BAND if ws.get('mean_compute_ms', 0) < 30
             else COLOR_NO_DATA if ws.get('mean_compute_ms', 0) < 80
             else COLOR_OUT_BAND),
            (f"  last={ws.get('last_compute_ms', 0):.2f}ms", "",          TEXT_DIM),
            (f"  queue={ws.get('queue_depth', 0)}/4", "",
             COLOR_IN_BAND if ws.get('queue_depth', 0) <= 1 else COLOR_NO_DATA),
            (f"  processed={ws.get('jobs_processed', 0)}", "",            TEXT_DIM),
            (f"  dropped={ws.get('jobs_dropped', 0)}", "",
             COLOR_IN_BAND if ws.get('jobs_dropped', 0) == 0 else COLOR_OUT_BAND),
        ]
        for prefix, value, colour in parts:
            psurf = self._font_tiny.render(prefix, True, colour)
            self.screen.blit(psurf, (x, y)); x += psurf.get_width()
            if value:
                vsurf = self._font_tiny.render(value, True, colour)
                self.screen.blit(vsurf, (x, y)); x += vsurf.get_width() + 4
        if ws.get('last_error'):
            err_x = rect.right - 360
            err_txt = self._font_tiny.render(
                f"last err: {ws['last_error'][:50]}", True, COLOR_OUT_BAND)
            self.screen.blit(err_txt, (err_x, y))

    # ---- header ----
    def _draw_header(self, outer: pygame.Rect, logger, game_id: str):
        title_surf = self._font_h1.render(
            "Neuraxon Game of Life — Realtime Metrics Dashboard",
            True, TEXT_HEADER)
        self.screen.blit(title_surf, (outer.x + 20, outer.y + 12))
        # right-aligned: game id + log info
        meta = []
        if game_id:
            meta.append(f"GameID: {game_id}")
        if logger is not None:
            ticks = (logger.time_series or {}).get('ticks', [])
            meta.append(f"samples: {len(ticks)}")
            meta.append(f"latest tick: {ticks[-1] if ticks else 0}")
            try:
                dash = logger.get_research_dashboard() if hasattr(logger, 'get_research_dashboard') else None
                if dash and dash.get('summary'):
                    s = dash['summary']
                    meta.append(f"in-band: {s['in_band_count']}/{s['total']}  ({s['pct_in_band']:.0f}%)")
            except Exception:
                pass
        meta_y = outer.y + 14
        for line in meta:
            ts = self._font_body.render(line, True, TEXT_HINT)
            self.screen.blit(ts, (outer.right - 20 - ts.get_width(), meta_y))
            meta_y += 18

        # subtitle row with the legend
        sub = self._font_body.render(
            "Press L or ESC to close · Press K (or click button) to export key-metrics file ·  Green = paper-band, Red = out-of-band",
            True, TEXT_DIM)
        self.screen.blit(sub, (outer.x + 20, outer.y + 60))

    # ---- panel ----
    def _draw_panel(self, rect: pygame.Rect, desc: dict):
        # Background
        pygame.draw.rect(self.screen, (28, 28, 38), rect, border_radius=8)
        pygame.draw.rect(self.screen, (70, 70, 90), rect, 1, border_radius=8)

        # Title
        title = self._font_h2.render(desc['title'], True, TEXT_HEADER)
        self.screen.blit(title, (rect.x + 12, rect.y + 8))
        # Paper ref under title
        ref = self._font_tiny.render(desc['paper_ref'], True, TEXT_DIM)
        self.screen.blit(ref, (rect.x + 12, rect.y + 32))

        # Pull current and historical values
        primary_series = self._cached_series.get(desc['metric_key'], [])
        cur_val = primary_series[-1] if primary_series else None

        # Healthy band
        band: Optional[Tuple[float, float]] = None
        in_band: Optional[bool] = None
        if desc['healthy_band_key'] and self._healthy_bands:
            band = self._healthy_bands.get(desc['healthy_band_key'])
        if band is not None and cur_val is not None:
            lo, hi = band
            in_band = (lo <= cur_val <= hi)

        # Layout: left third for value, right two-thirds for sparkline
        left_w = int(rect.width * 0.32)
        spark_x = rect.x + left_w + 4
        spark_w = rect.right - spark_x - 12
        spark_y = rect.y + 60
        spark_h = rect.height - spark_y + rect.y - 12
        spark_rect = pygame.Rect(spark_x, spark_y, spark_w, spark_h)

        # Current value (large)
        if cur_val is None:
            value_str = "—"
            value_col = COLOR_NO_DATA
        else:
            value_str = f"{cur_val:{desc['fmt']}}"
            if in_band is True:
                value_col = COLOR_IN_BAND
            elif in_band is False:
                value_col = COLOR_OUT_BAND
            else:
                value_col = TEXT_HEADER
        big = self._font_big_value.render(value_str, True, value_col)
        self.screen.blit(big, (rect.x + 14, rect.y + 56))
        lab = self._font_body.render(desc['value_label'], True, TEXT_DIM)
        self.screen.blit(lab, (rect.x + 14 + big.get_width() + 6,
                                 rect.y + 64 + (big.get_height() - lab.get_height())))

        # In-band tag
        if in_band is True:
            tag = self._font_tiny.render("✓ in band", True, COLOR_IN_BAND)
            self.screen.blit(tag, (rect.x + 14, rect.y + 56 + big.get_height() + 4))
        elif in_band is False:
            tag = self._font_tiny.render("✗ out of band", True, COLOR_OUT_BAND)
            self.screen.blit(tag, (rect.x + 14, rect.y + 56 + big.get_height() + 4))
        elif desc['healthy_band_key'] is None:
            tag = self._font_tiny.render("(no band defined)", True, TEXT_DIM)
            self.screen.blit(tag, (rect.x + 14, rect.y + 56 + big.get_height() + 4))

        # Healthy-band numeric range below tag
        if band is not None:
            lo, hi = band
            rstr = f"target [{lo:g}, {hi:g}]"
            rsurf = self._font_tiny.render(rstr, True, TEXT_DIM)
            self.screen.blit(rsurf, (rect.x + 14, rect.y + 56 + big.get_height() + 18))

        # Subtitle (paper claim) at bottom of left column
        sub_y = rect.y + 56 + big.get_height() + 38
        if sub_y < rect.bottom - 10:
            self._wrap_text(desc['subtitle'], rect.x + 14, sub_y,
                            left_w - 16, TEXT_BODY, self._font_tiny)

        # Sparkline (primary)
        _draw_sparkline(self.screen, spark_rect, primary_series,
                        desc['spark_colour'], band=band, bg_alpha=160)

        # Secondary tiny series — drawn as additional thin lines, dimmed
        for sk, sl, sc in zip(desc.get('metric_secondary_keys', []),
                                 desc.get('metric_secondary_labels', []),
                                 desc.get('metric_secondary_colours', [])):
            sub_series = self._cached_series.get(sk, [])
            if sub_series:
                self._draw_secondary_line(spark_rect, sub_series, sc)

        # Tick range under the sparkline
        if self._cached_ticks:
            t0, t1 = self._cached_ticks[0], self._cached_ticks[-1]
            t_str = f"ticks  {t0} → {t1}    n={len(primary_series)}"
            ts_surf = self._font_tiny.render(t_str, True, TEXT_DIM)
            self.screen.blit(ts_surf, (spark_rect.x + 4,
                                         spark_rect.bottom - ts_surf.get_height() - 2))

        # Secondary series legend (top-right of sparkline)
        leg_x = spark_rect.right - 6
        leg_y = spark_rect.y + 4
        for sk, sl, sc in zip(reversed(desc.get('metric_secondary_keys', [])),
                                 reversed(desc.get('metric_secondary_labels', [])),
                                 reversed(desc.get('metric_secondary_colours', []))):
            sub_series = self._cached_series.get(sk, [])
            if sub_series:
                v = sub_series[-1]
                txt = self._font_tiny.render(f"{sl}: {v:.3f}", True, sc)
                leg_x -= txt.get_width() + 12
                self.screen.blit(txt, (leg_x, leg_y))

    def _draw_secondary_line(self, rect: pygame.Rect, series: List[float],
                              colour: Tuple[int, int, int]):
        """Draw a faded secondary line in the same axis range as primary
        (we just normalise to the secondary's own min/max)."""
        if len(series) < 2:
            return
        y_min = min(series); y_max = max(series)
        if y_max - y_min < 1e-9:
            return
        n = len(series)
        inner_w = rect.width - 4
        if n > inner_w:
            step = n / inner_w
            sampled = [series[int(i * step)] for i in range(inner_w)]
            n2 = len(sampled)
        else:
            sampled = series; n2 = n
        pts = []
        for i, v in enumerate(sampled):
            x = rect.x + 2 + int(i * (rect.width - 4) / max(1, n2 - 1))
            y = rect.bottom - int((v - y_min) / (y_max - y_min) * (rect.height - 4)) - 2
            pts.append((x, y))
        # Faded version of the colour
        faded = (max(80, colour[0] // 2), max(80, colour[1] // 2), max(80, colour[2] // 2))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, faded, False, pts, 1)

    def _wrap_text(self, text: str, x: int, y: int, width: int,
                   colour: Tuple[int, int, int], font: pygame.font.Font):
        """Naive word-wrap; renders lines stacked top-down from (x, y)."""
        words = text.split(' ')
        line = ''
        cur_y = y
        for w in words:
            trial = (line + ' ' + w) if line else w
            if font.size(trial)[0] > width and line:
                self.screen.blit(font.render(line, True, colour), (x, cur_y))
                cur_y += font.get_linesize()
                line = w
            else:
                line = trial
        if line:
            self.screen.blit(font.render(line, True, colour), (x, cur_y))

    # ---- footer ----
    def _draw_footer(self, outer: pygame.Rect):
        # save button (right) + save-feedback line (left)
        btn_w, btn_h = 220, 36
        btn_x = outer.right - btn_w - 20
        btn_y = outer.bottom - btn_h - 16
        rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        # hover effect
        mx, my = pygame.mouse.get_pos()
        col = COLOR_BUTTON_HI if rect.collidepoint(mx, my) else COLOR_BUTTON
        pygame.draw.rect(self.screen, col, rect, border_radius=6)
        pygame.draw.rect(self.screen, (110, 130, 170), rect, 1, border_radius=6)
        label = self._font_h2.render("Save key metrics (K)", True, COLOR_BUTTON_TXT)
        self.screen.blit(label, (rect.x + (btn_w - label.get_width()) // 2,
                                  rect.y + (btn_h - label.get_height()) // 2))
        self._buttons['save_metrics'] = rect

        # close button
        cw, ch = 80, 36
        cx = btn_x - cw - 12
        crect = pygame.Rect(cx, btn_y, cw, ch)
        col2 = COLOR_BUTTON_HI if crect.collidepoint(mx, my) else COLOR_BUTTON
        pygame.draw.rect(self.screen, col2, crect, border_radius=6)
        pygame.draw.rect(self.screen, (110, 130, 170), crect, 1, border_radius=6)
        clab = self._font_h2.render("Close (L)", True, COLOR_BUTTON_TXT)
        self.screen.blit(clab, (crect.x + (cw - clab.get_width()) // 2,
                                  crect.y + (ch - clab.get_height()) // 2))
        self._buttons['close'] = crect

        # Last save feedback
        if self._last_save_path:
            age = time.time() - self._last_save_at_ts
            if age < 8.0:
                # show save-feedback in green for 8 seconds
                fade = max(0.0, 1.0 - age / 8.0)
                col = (int(60 + 100 * fade), int(150 + 80 * fade), int(90 + 80 * fade))
                msg = f"Saved: {os.path.basename(self._last_save_path)}"
                fs = self._font_body.render(msg, True, col)
                self.screen.blit(fs, (outer.x + 20, btn_y + 8))
            else:
                msg = f"Last save: {os.path.basename(self._last_save_path)}"
                fs = self._font_body.render(msg, True, TEXT_DIM)
                self.screen.blit(fs, (outer.x + 20, btn_y + 8))

    # ---- caching ----
    def _refresh_cache(self, ts: dict):
        """Snapshot the 10 headline keys plus their secondaries plus the
        v147 survivability series into the cache."""
        self._cached_ticks = list(ts.get('ticks', []))
        keys = set()
        for d in PANEL_DESCRIPTORS:
            keys.add(d['metric_key'])
            for sk in d.get('metric_secondary_keys', []):
                keys.add(sk)
        # v147 — survivability score sparkline lives on the top strip
        keys.add('surv_score')
        self._cached_series = {}
        for k in keys:
            self._cached_series[k] = list(ts.get(k, []))
