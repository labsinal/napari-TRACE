"""
Per-track research plots that mirror the figures in the analysis deck.

This module complements ``stats.py`` (which focuses on per-track *summary*
boxplots and population curves) with the "one line per cell over time" family:

  * track_series     - any per-frame quantity vs absolute frame, one coloured
                       line per cell, outcome colouring + treatment shading.
                       (Area, circularity, perimeter, speed, cumulative
                       distance, absolute displacement, directionality, ...)
  * spider_plot      - X/Y trajectories recentred on a common origin, one line
                       per cell, coloured by outcome.
  * windowed_metric  - a sliding-window descriptor per cell over time
                       (persistence, area CV / "pulsation", slope, mean speed,
                       windowed path length / net displacement).

It also provides ``annotate_morphology`` which adds per-frame ``perimeter`` and
``circularity`` columns from the mask, because the raw table only carries area.

Design mirrors stats.py: matplotlib only, never imports Qt, reuses config and
analysis. Every function returns a matplotlib Figure (use ``stats.show``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import analysis
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_AREA, COL_PARENT,
    OUTCOME_MITOSIS, OUTCOME_EXIT, OUTCOME_DEATH, OUTCOME_AMBIGUOUS,
    TREAT_TREATED, TREAT_WASHOUT,
)

# Outcome -> line colour. Green = Mitosis, blue = Exit, to match the rest of
# reference figures, with distinct colours for the other final outcomes.
OUTCOME_COLORS = {
    OUTCOME_MITOSIS: "#55C57A",   # green  (Mitosis)
    OUTCOME_EXIT:    "#4C9BE8",   # blue   (Exit / left the field)
    OUTCOME_DEATH:   "#C44E52",   # red
    OUTCOME_AMBIGUOUS: "#E0A030",  # amber  (Ambiguo / incerto)
    "none":          "#B0B0B0",   # grey   (uncurated)
}
DEFAULT_LINE_COLOR = "#9E9E9E"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _final_outcome_per_track(df):
    """Return {track_id: outcome_label} using the first non-empty outcome.

    A track is also treated as Mitosis if it is some cell's parent, matching
    ``analysis.compute_track_summary``.
    """
    out = {}
    parents = set()
    if COL_PARENT in df.columns:
        parents = set(df.loc[df[COL_PARENT] > 0, COL_PARENT]
                      .dropna().astype(int).unique())
    has_outcome = COL_OUTCOME in df.columns
    for tid, g in df.groupby(COL_TRACK):
        label = "none"
        if has_outcome:
            vals = [v for v in g[COL_OUTCOME].astype(str).unique()
                    if v not in ("", "nan", "None")]
            if vals:
                label = vals[0]
        if label == "none" and int(tid) in parents:
            label = OUTCOME_MITOSIS
        out[int(tid)] = label
    return out


def _shade_treatment(ax, treatment_config, max_frame):
    """Pink TMZ-style shading for the treated (and washout) window.

    Accepts the same ``treatment_config`` dict the rest of the app uses
    ({'mode', 'start', 'end'}); a None config is a safe no-op.
    """
    if not treatment_config or treatment_config.get("mode") != TREAT_TREATED:
        return
    start = treatment_config.get("start", 0)
    end = treatment_config.get("end", max_frame)
    ax.axvspan(start, end, color="#F4A6A6", alpha=0.18, zorder=0, label="TMZ")
    if 0 <= end < max_frame:
        ax.axvspan(end, max_frame, color="#F4A6A6", alpha=0.08, zorder=0)


def _outcome_legend(ax):
    handles = [
        Line2D([0], [0], color=OUTCOME_COLORS[OUTCOME_MITOSIS], lw=2, label="Mitosis"),
        Line2D([0], [0], color=OUTCOME_COLORS[OUTCOME_EXIT], lw=2, label="Exit"),
        Line2D([0], [0], color=OUTCOME_COLORS[OUTCOME_DEATH], lw=2, label="Death"),
        Line2D([0], [0], color=OUTCOME_COLORS[OUTCOME_AMBIGUOUS], lw=2, label="Ambiguo"),
    ]
    ax.legend(handles=handles, title="Identificacao", fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# Morphology columns from the mask (perimeter + circularity)
# ---------------------------------------------------------------------------
def annotate_morphology(df, mask):
    """Add per-frame ``perimeter`` and ``circularity`` columns from the mask.

    Circularity is 4*pi*area / perimeter^2 (1.0 = perfect circle). Returns a
    copy; rows whose label is absent in that frame get NaN. Safe no-op (NaN
    columns) when there is no mask.

    Note: the column names returned ("perimeter", "circularity") are then
    selectable as the Y axis of :func:`track_series`.
    """
    from skimage.measure import regionprops_table
    out = df.copy()
    out["perimeter"] = np.nan
    out["circularity"] = np.nan
    if mask is None or out.empty:
        return out

    frame_vals = out[COL_FRAME].to_numpy()
    track_vals = out[COL_TRACK].to_numpy()
    for f in range(mask.shape[0]):
        plane = mask[f]
        if not plane.any():
            continue
        t = regionprops_table(plane, properties=("label", "area", "perimeter"))
        labels = np.asarray(t["label"])
        areas = np.asarray(t["area"], dtype=float)
        perims = np.asarray(t["perimeter"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            circ = np.where(perims > 0, 4.0 * np.pi * areas / (perims ** 2), np.nan)
        peri_lut = {int(l): float(p) for l, p in zip(labels, perims)}
        circ_lut = {int(l): float(c) for l, c in zip(labels, circ)}
        sel = np.where(frame_vals == f)[0]
        for i in sel:
            tid = track_vals[i]
            if not np.isnan(tid):
                out.iat[i, out.columns.get_loc("perimeter")] = peri_lut.get(int(tid), np.nan)
                out.iat[i, out.columns.get_loc("circularity")] = circ_lut.get(int(tid), np.nan)
    return out


# ---------------------------------------------------------------------------
# Per-track time series (Area / Circularity / Perimeter / Speed / ...)
# ---------------------------------------------------------------------------
# Derived per-frame quantities that are computed on the fly from x/y so the user
# can plot them without precomputing columns. Key -> (label, function).
def _speed_series(g, pixel_size, frame_interval):
    xs = g[COL_X].to_numpy(float) * pixel_size
    ys = g[COL_Y].to_numpy(float) * pixel_size
    d = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / frame_interval
    return np.concatenate([[np.nan], d])


def _cumdist_series(g, pixel_size, frame_interval):
    xs = g[COL_X].to_numpy(float) * pixel_size
    ys = g[COL_Y].to_numpy(float) * pixel_size
    d = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return np.concatenate([[0.0], np.cumsum(d)])


def _absdisp_series(g, pixel_size, frame_interval):
    xs = g[COL_X].to_numpy(float) * pixel_size
    ys = g[COL_Y].to_numpy(float) * pixel_size
    return np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)


def _directionality_series(g, pixel_size, frame_interval):
    xs = g[COL_X].to_numpy(float) * pixel_size
    ys = g[COL_Y].to_numpy(float) * pixel_size
    net = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)
    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total = np.concatenate([[0.0], np.cumsum(step)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, net / total, np.nan)


DERIVED_SERIES = {
    "speed":          ("Speed", _speed_series),
    "cum_distance":   ("Cumulative Distance", _cumdist_series),
    "abs_displacement": ("Linear Displacement", _absdisp_series),
    "directionality": ("Direcionalidade (0 a 1)", _directionality_series),
}


def track_series(df, y, mask=None, pixel_size=1.0, frame_interval=1.0,
                 treatment_config=None, max_frame=None, smooth=1,
                 min_frames=2, title=None, ylabel=None):
    """One coloured line per cell of quantity ``y`` over absolute frame.

    Parameters
    ----------
    y : str
        Either a numeric column already in ``df`` (e.g. the raw area column,
        or "perimeter"/"circularity" after :func:`annotate_morphology`), or one
        of the derived keys in ``DERIVED_SERIES``: "speed", "cum_distance",
        "abs_displacement", "directionality".
    mask, pixel_size, frame_interval : passed through for derived series.
    treatment_config, max_frame : enable the pink TMZ shading.
    smooth : int
        Rolling-mean window (in frames) applied per track; 1 = no smoothing.
    min_frames : int
        Skip tracks shorter than this.

    Returns the matplotlib Figure.
    """
    valid = df.dropna(subset=[COL_TRACK, COL_FRAME]).copy()
    valid[COL_TRACK] = valid[COL_TRACK].astype(int)
    valid = valid[valid[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])
    if valid.empty:
        raise ValueError("No valid tracks to plot.")
    if max_frame is None:
        max_frame = float(valid[COL_FRAME].max())

    is_derived = y in DERIVED_SERIES
    if not is_derived and y not in valid.columns:
        raise ValueError(
            f"'{y}' is neither a column nor a derived series "
            f"({', '.join(DERIVED_SERIES)}).")

    outcome = _final_outcome_per_track(valid)
    fig, ax = plt.subplots(figsize=(13, 6))
    _shade_treatment(ax, treatment_config, max_frame)

    drew = 0
    for tid, g in valid.groupby(COL_TRACK):
        if len(g) < min_frames:
            continue
        frames = g[COL_FRAME].to_numpy(float)
        if is_derived:
            label, fn = DERIVED_SERIES[y]
            yv = fn(g, pixel_size, frame_interval)
        else:
            yv = g[y].to_numpy(float)
            label = ylabel or y
        if smooth and smooth > 1:
            yv = pd.Series(yv).rolling(smooth, min_periods=1, center=True).mean().to_numpy()
        color = OUTCOME_COLORS.get(outcome.get(int(tid), "none"), DEFAULT_LINE_COLOR)
        ax.plot(frames, yv, color=color, alpha=0.65, linewidth=1.2)
        # Label the last point with the track id, like the reference figures.
        finite = np.isfinite(yv)
        if finite.any():
            j = np.flatnonzero(finite)[-1]
            ax.annotate(str(int(tid)), (frames[j], yv[j]), fontsize=7,
                        color=color, fontweight="bold")
        drew += 1
    if drew == 0:
        raise ValueError("No tracks long enough to plot (raise data or lower min_frames).")

    ax.set_xlabel("Time (frame)")
    ax.set_ylabel(ylabel or (DERIVED_SERIES[y][0] if is_derived else y))
    ax.set_title(title or (DERIVED_SERIES[y][0] if is_derived else y),
                 fontweight="bold")
    ax.grid(alpha=0.3)
    _outcome_legend(ax)
    return fig


# ---------------------------------------------------------------------------
# Spider plot (trajectories recentred on the origin)
# ---------------------------------------------------------------------------
def spider_plot(df, pixel_size=1.0, min_frames=2, title="Spider Plot Global"):
    """Recentre every track to start at (0, 0) and overlay the trajectories.

    Each line is coloured by final outcome (green = Mitosis, blue = Exit, ...).
    Returns the matplotlib Figure.
    """
    valid = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    valid[COL_TRACK] = valid[COL_TRACK].astype(int)
    valid = valid[valid[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])
    if valid.empty:
        raise ValueError("No valid tracks to plot.")

    outcome = _final_outcome_per_track(valid)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.axvline(0, color="k", ls="--", lw=1)

    for tid, g in valid.groupby(COL_TRACK):
        if len(g) < min_frames:
            continue
        xs = (g[COL_X].to_numpy(float) - g[COL_X].iloc[0]) * pixel_size
        ys = (g[COL_Y].to_numpy(float) - g[COL_Y].iloc[0]) * pixel_size
        color = OUTCOME_COLORS.get(outcome.get(int(tid), "none"), DEFAULT_LINE_COLOR)
        ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.2)
        ax.annotate(str(int(tid)), (xs[-1], ys[-1]), fontsize=7,
                    color=color, fontweight="bold")

    ax.set_xlabel("X displacement")
    ax.set_ylabel("Y displacement")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    _outcome_legend(ax)
    return fig


# ---------------------------------------------------------------------------
# Sliding-window metrics (persistence, CV, slope, windowed path / net / speed)
# ---------------------------------------------------------------------------
def _win_persistence(frames, xs, ys, vals, frame_interval):
    # net / total path within the window (directional persistence)
    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total = step.sum()
    net = np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
    return (net / total) if total > 0 else np.nan


def _win_path(frames, xs, ys, vals, frame_interval):
    return np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum()


def _win_net(frames, xs, ys, vals, frame_interval):
    return np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)


def _win_mean_speed(frames, xs, ys, vals, frame_interval):
    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    span = (frames[-1] - frames[0]) * frame_interval
    return (step.sum() / span) if span > 0 else np.nan


def _win_cv(frames, xs, ys, vals, frame_interval):
    # coefficient of variation of the value column (e.g. area "pulsation")
    v = vals[np.isfinite(vals)]
    m = v.mean()
    return (v.std() / m) if (v.size and m != 0) else np.nan


def _win_slope(frames, xs, ys, vals, frame_interval):
    # linear trend (slope) of the value column over the window
    ok = np.isfinite(vals)
    if ok.sum() < 2 or np.ptp(frames[ok]) == 0:
        return np.nan
    return float(np.polyfit(frames[ok], vals[ok], 1)[0])


WINDOW_METRICS = {
    "persistence":  ("Persistence", "Local Migratory Persistence", _win_persistence, False),
    "path":         ("Path Length", "Motor Effort in Window", _win_path, False),
    "net":          ("Net Displacement", "Migration Efficacy in Window", _win_net, False),
    "mean_speed":   ("Mean Speed", "Mean Speed per Window", _win_mean_speed, False),
    "area_cv":      ("Area Variation (CV)", "Nuclear Envelope Instability (CV)", _win_cv, True),
    "slope":        ("Area Slope (trend)", "Expansion / Retraction Rate (slope)", _win_slope, True),
}


def windowed_metric(df, metric, window=11, mask=None, value_col=None,
                    pixel_size=1.0, frame_interval=1.0,
                    treatment_config=None, max_frame=None, min_frames=2,
                    title=None, ylabel=None):
    """Sliding-window descriptor per cell, one coloured line over time.

    Parameters
    ----------
    metric : key of ``WINDOW_METRICS``
        "persistence", "path", "net", "mean_speed" use x/y;
        "area_cv", "slope" need a value column (``value_col``), e.g. the area
        column or "perimeter"/"circularity" after :func:`annotate_morphology`.
    window : int
        Window length in frames (odd is natural; the point is plotted at the
        window centre, mirroring "Centro da Janela" in the reference figures).
    value_col : str or None
        Required for value-based metrics (area_cv, slope).

    Returns the matplotlib Figure.
    """
    if metric not in WINDOW_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. "
                         f"Choose from: {', '.join(WINDOW_METRICS)}.")
    short_lbl, default_title, fn, needs_value = WINDOW_METRICS[metric]
    if needs_value and not value_col:
        raise ValueError(f"Metric '{metric}' needs value_col "
                         f"(e.g. the area column, 'perimeter' or 'circularity').")

    valid = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    valid[COL_TRACK] = valid[COL_TRACK].astype(int)
    valid = valid[valid[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])
    if valid.empty:
        raise ValueError("No valid tracks to plot.")
    if needs_value and value_col not in valid.columns:
        raise ValueError(f"Value column '{value_col}' not in dataframe.")
    if max_frame is None:
        max_frame = float(valid[COL_FRAME].max())

    outcome = _final_outcome_per_track(valid)
    half = max(1, window // 2)
    fig, ax = plt.subplots(figsize=(13, 6))
    _shade_treatment(ax, treatment_config, max_frame)

    drew = 0
    for tid, g in valid.groupby(COL_TRACK):
        if len(g) < max(min_frames, window):
            continue
        frames = g[COL_FRAME].to_numpy(float)
        xs = g[COL_X].to_numpy(float) * pixel_size
        ys = g[COL_Y].to_numpy(float) * pixel_size
        vals = g[value_col].to_numpy(float) if needs_value else np.full(len(g), np.nan)
        centres, out_vals = [], []
        for c in range(half, len(g) - half):
            sl = slice(c - half, c + half + 1)
            r = fn(frames[sl], xs[sl], ys[sl], vals[sl], frame_interval)
            centres.append(frames[c])
            out_vals.append(r)
        if not centres:
            continue
        color = OUTCOME_COLORS.get(outcome.get(int(tid), "none"), DEFAULT_LINE_COLOR)
        ax.plot(centres, out_vals, color=color, alpha=0.6, linewidth=1.1)
        ax.annotate(str(int(tid)), (centres[-1], out_vals[-1]), fontsize=7,
                    color=color, fontweight="bold")
        drew += 1
    if drew == 0:
        raise ValueError("No tracks long enough for this window size.")

    ax.set_xlabel("Centro da Janela (Frame Absoluto)")
    ax.set_ylabel(ylabel or short_lbl)
    ax.set_title(title or default_title, fontweight="bold")
    ax.grid(alpha=0.3)
    _outcome_legend(ax)
    return fig
