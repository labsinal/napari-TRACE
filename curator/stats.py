"""
Statistics and plotting.

Contains the existing research plots plus two requested additions:

  * Temporal-gradient scatter: a population mean line with one point per cell
    per frame, colored on a red->blue gradient by frame, to reveal whether a
    quantity drifts in a consistent direction over time.

  * Customizable plotting: pick any numeric column for X and Y and a grouping
    key (frame, track_id, treatment or outcome) and get a scatter / mean-line.

All figures are produced with matplotlib; the module never imports Qt directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_TREATMENT,
    GROUP_COLORS, TREAT_TREATED, TREAT_WASHOUT, TREAT_CONTROL,
)
from . import analysis


def show(fig):
    fig.tight_layout()
    try:
        plt.show(block=False)
    except Exception:
        plt.show()


def box_with_points(ax, groups, title, ylabel):
    """Box plot with jittered points for a dict of label -> values."""
    labels, data = [], []
    for lab, arr in groups.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            labels.append(lab)
            data.append(arr)
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return
    positions = list(range(1, len(labels) + 1))
    bp = ax.boxplot(data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(GROUP_COLORS.get(lab, "#9E9E9E"))
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    for pos, arr in zip(positions, data):
        jitter = np.random.normal(0, 0.06, size=arr.size)
        ax.scatter(np.full(arr.size, pos) + jitter, arr, s=14,
                   color="black", alpha=0.35, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={d.size})" for l, d in zip(labels, data)])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def shade_treatment(ax, treatment_config, max_frame):
    """Shade treated and washout windows on a time-axis plot."""
    if treatment_config["mode"] != TREAT_TREATED:
        return
    start = treatment_config["start"]
    end = treatment_config["end"]
    ax.axvspan(start, end, color=GROUP_COLORS[TREAT_TREATED], alpha=0.12, label="treated")
    if 0 <= end < max_frame:
        ax.axvspan(end, max_frame, color=GROUP_COLORS[TREAT_WASHOUT], alpha=0.12, label="washout")
    ax.legend(loc="best", fontsize=8)


# ---------------------------------------------------------------------------
# Free-form request: temporal-gradient boxplot
# ---------------------------------------------------------------------------
# Human-readable names for the categorical grouping keys, used on the X axis.
_GRADIENT_GROUP_LABELS = {
    "final_outcome": "Desfecho",
    "outcome": "Desfecho",
    "treatment": "Tratamento",
    "is_mitotic": "Mitótica?",
    "time": "Janela de tempo",
}


def _group_for_cellframe(df, group_by):
    """Map (frame, track) -> group label for a categorical grouping key.

    Supports 'final_outcome'/'outcome' (each track's final fate), 'treatment'
    (per-row treatment phase) and 'is_mitotic' (track is a mother / flagged
    Mitosis). Returns a dict keyed by (int frame, int track). Used to colour a
    cell/frame point's box by an attribute of its track rather than by time.
    """
    lookup = {}
    d = df.dropna(subset=[COL_FRAME, COL_TRACK]).copy()
    if d.empty:
        return lookup
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d[COL_FRAME] = d[COL_FRAME].astype(int)

    if group_by in ("final_outcome", "outcome"):
        per_track = _final_outcome_per_track(d)
        # Use the Portuguese outcome labels (Mitose / Fim / ...) to match the
        # rest of the figures; unknown/none stays "Sem desfecho".
        for f, t in zip(d[COL_FRAME], d[COL_TRACK]):
            oc = per_track.get(int(t), "none")
            lookup[(int(f), int(t))] = TRAJ_OUTCOME_LABELS.get(oc, "Sem desfecho")
    elif group_by == "is_mitotic":
        mothers = set()
        if COL_PARENT in d.columns:
            mothers = set(int(p) for p in d.loc[d[COL_PARENT] > 0, COL_PARENT]
                          .dropna().unique())
        per_track = _final_outcome_per_track(d)
        for f, t in zip(d[COL_FRAME], d[COL_TRACK]):
            ti = int(t)
            is_mito = ti in mothers or per_track.get(ti) == "Mitosis"
            lookup[(int(f), ti)] = "Mitótica" if is_mito else "Não-mitótica"
    elif group_by == "treatment":
        col = d[COL_TREATMENT] if COL_TREATMENT in d.columns else None
        for i, (f, t) in enumerate(zip(d[COL_FRAME], d[COL_TRACK])):
            lookup[(int(f), int(t))] = (str(col.iloc[i]) if col is not None
                                        else TREAT_CONTROL)
    return lookup


def gradient_over_time(df, mask, value="area", pixel_size=1.0, frame_interval=1.0,
                       cmap_name="coolwarm", group_by="final_outcome", n_time_bins=8):
    """Boxplot per group with a blue->red temporal-gradient point overlay.

    Underneath: a boxplot of ``value`` summarising EVERY cell/frame pair in each
    group (median, quartiles, whiskers over all frames). On top: one point per
    cell/frame pair, jittered within its box and coloured on a blue->red
    gradient by frame (blue = frame 0 / early, red = last frame / late), so any
    temporal drift in the quantity shows up as a colour trend inside each box.

    ``group_by`` selects what the X axis (the boxes) represents:
      - "final_outcome" : one box per final fate (Mitose / Fim / ...). This is
        the example case: X = outcome, Y = area over all frames of those cells.
      - "treatment"     : one box per treatment phase (control/treated/washout).
      - "is_mitotic"    : mitotic vs non-mitotic tracks.
      - "time"          : bin frames into ``n_time_bins`` equal time windows
        (the legacy "by time" view).
    ``value`` is 'area' (read from the mask) or any numeric column in ``df``.
    Returns the matplotlib Figure.
    """
    is_time = group_by == "time"

    # Build the (frame, track) -> group lookup for categorical groupings.
    group_lookup = {} if is_time else _group_for_cellframe(df, group_by)

    # Collect (frame, value, group) per cell/frame pair.
    frames_all, vals_all, groups_all = [], [], []

    if value == "area":
        if mask is None:
            raise ValueError("O gradiente de área precisa de uma máscara.")
        px_area = pixel_size ** 2
        for f in range(mask.shape[0]):
            for idv, ar in analysis.areas_for_frame(mask[f], px_area).items():
                frames_all.append(f)
                vals_all.append(ar)
                if not is_time:
                    groups_all.append(group_lookup.get((int(f), int(idv)),
                                                        "Sem desfecho"))
        ylabel = f"Área ({'µm²' if pixel_size != 1.0 else 'px²'})"
    else:
        if value not in df.columns:
            raise ValueError(f"Coluna '{value}' não está na tabela.")
        d = df.dropna(subset=[COL_FRAME, value, COL_TRACK])
        frames_all = d[COL_FRAME].to_numpy(dtype=float).tolist()
        vals_all = d[value].to_numpy(dtype=float).tolist()
        if not is_time:
            for f, t in zip(d[COL_FRAME], d[COL_TRACK]):
                groups_all.append(group_lookup.get((int(f), int(t)), "Sem desfecho"))
        ylabel = value

    if not frames_all:
        raise ValueError("Nenhum dado para plotar.")

    frames_all = np.asarray(frames_all, dtype=float)
    vals_all = np.asarray(vals_all, dtype=float)

    # Define the groups (boxes) and assign each point to one.
    if is_time:
        fmin, fmax = frames_all.min(), frames_all.max()
        nb = max(1, int(n_time_bins))
        edges = np.linspace(fmin, fmax + 1e-9, nb + 1)
        assign = np.clip(np.digitize(frames_all, edges) - 1, 0, nb - 1)
        box_labels = []
        for i in range(nb):
            lo, hi = edges[i] * frame_interval, edges[i + 1] * frame_interval
            box_labels.append(f"{lo:.0f}-{hi:.0f}")
    else:
        groups_all = np.asarray(groups_all, dtype=object)
        present = set(groups_all)
        # Order the boxes sensibly per grouping; unknown labels go last.
        if group_by in ("final_outcome", "outcome"):
            preferred = ["Mitose", "Fim", "Morte/Senescência", "Ambíguo", "Sem desfecho"]
        elif group_by == "treatment":
            preferred = [TREAT_CONTROL, TREAT_TREATED, TREAT_WASHOUT]
        elif group_by == "is_mitotic":
            preferred = ["Mitótica", "Não-mitótica"]
        else:
            preferred = []
        box_labels = [g for g in preferred if g in present]
        box_labels += sorted(g for g in present if g not in box_labels)
        point_box = {lab: i for i, lab in enumerate(box_labels)}
        assign = np.array([point_box[g] for g in groups_all])

    xlabel = _GRADIENT_GROUP_LABELS.get(group_by, str(group_by))

    # Build the per-box value arrays (all cell/frame pairs of that group).
    box_data = [vals_all[assign == i] for i in range(len(box_labels))]
    positions = list(range(1, len(box_labels) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    # Boxplot underneath: median/quartiles/whiskers over all frames in the group.
    bp = ax.boxplot([d if d.size else [np.nan] for d in box_data],
                    positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, zorder=1)
    for patch in bp["boxes"]:
        patch.set_facecolor("#E8E8E8")
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")

    # Points on top, coloured blue (early) -> red (late) by frame, jittered.
    norm = Normalize(vmin=frames_all.min(), vmax=frames_all.max())
    cmap = cm.get_cmap(cmap_name)
    for i in range(len(box_labels)):
        sel = assign == i
        if not np.any(sel):
            continue
        jitter = np.random.normal(0, 0.08, size=int(sel.sum()))
        ax.scatter(np.full(int(sel.sum()), positions[i]) + jitter, vals_all[sel],
                   c=frames_all[sel], cmap=cmap, norm=norm, s=16, alpha=0.6,
                   zorder=3, edgecolors="none")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Frame (azul = início, vermelho = fim)")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={d.size})" for l, d in zip(box_labels, box_data)],
                       fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: boxplot por {xlabel.lower()} com gradiente temporal",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Free-form request: customizable plotting
# ---------------------------------------------------------------------------
GROUPING_KEYS = (COL_FRAME, COL_TRACK, COL_TREATMENT, COL_OUTCOME)


def custom_plot(df, x_col, y_col, group_by=None, agg="mean", kind="scatter",
                show_legend=True):
    """Flexible plot of ``y_col`` against ``x_col``.

    Parameters
    ----------
    group_by : str or None
        A column to group by (e.g. frame, track_id, treatment, outcome). When
        set with ``kind='line'`` the aggregated value per x is drawn per group.
    agg : {'mean','median','sum','count'}
        Aggregation used when grouping for a line plot.
    kind : {'scatter','line'}
    show_legend : bool
        When False, no legend is drawn even if the plot is grouped.

    ``df`` may be the raw per-frame table or a per-track summary; any numeric
    column present is valid for either axis. Returns the matplotlib Figure.
    """
    for c in (x_col, y_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in dataframe.")
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        raise ValueError("No data after dropping NaNs for the chosen columns.")

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and group_by in d.columns and kind == "line":
        for i, (key, g) in enumerate(d.groupby(group_by)):
            grp = g.groupby(x_col)[y_col].agg(agg).sort_index()
            color = GROUP_COLORS.get(str(key),
                                     cm.get_cmap("tab10")(i % 10))
            ax.plot(grp.index, grp.values, marker="o", ms=3, linewidth=1.6,
                    label=str(key), color=color)
        if show_legend:
            ax.legend(title=group_by, fontsize=8, loc="best")
    elif group_by and group_by in d.columns and kind == "scatter":
        for i, (key, g) in enumerate(d.groupby(group_by)):
            color = GROUP_COLORS.get(str(key), cm.get_cmap("tab10")(i % 10))
            ax.scatter(g[x_col], g[y_col], s=16, alpha=0.5,
                       label=str(key), color=color, edgecolors="none")
        if show_legend:
            ax.legend(title=group_by, fontsize=8, loc="best")
    else:
        if kind == "line":
            grp = d.groupby(x_col)[y_col].agg(agg).sort_index()
            ax.plot(grp.index, grp.values, marker="o", ms=3, color="#4C72B0")
        else:
            ax.scatter(d[x_col], d[y_col], s=16, alpha=0.5,
                       color="#4C72B0", edgecolors="none")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col if not (group_by and kind == "line") else f"{agg}({y_col})")
    title = f"{y_col} vs {x_col}"
    if group_by:
        title += f" by {group_by}"
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Per-track trajectory plots (one line per cell, colored by final outcome)
# ---------------------------------------------------------------------------
# Color and legend label per final outcome, matching the uploaded figures
# (Mitosis = green, Exit/"Fim" = blue). Anything else falls back to grey.
TRAJ_OUTCOME_COLORS = {
    "Mitosis": "#5BC85B",
    "Exit": "#5B9BD5",
    "Death/Senescence": "#C44E52",
    "Ambiguous": "#E0A030",
}
TRAJ_OUTCOME_LABELS = {
    "Mitosis": "Mitose",
    "Exit": "Fim",
    "Death/Senescence": "Morte/Senescência",
    "Ambiguous": "Ambíguo",
}
_TRAJ_DEFAULT_COLOR = "#9E9E9E"


def _final_outcome_per_track(df):
    """Map track_id -> final outcome string (first non-empty outcome seen).

    Mirrors compute_track_summary's rule so trajectory colors match the rest of
    the tool. Tracks with no outcome map to "none".
    """
    out = {}
    if COL_OUTCOME not in df.columns:
        return out
    for tid, g in df.groupby(COL_TRACK):
        vals = [v for v in g[COL_OUTCOME].astype(str)
                if v not in ("", "nan", "None")]
        out[int(tid)] = vals[0] if vals else "none"
    return out


def _shade_tmz(ax, treatment_config):
    """Shade the treated (TMZ) window as a soft pink band, like the figures.

    Returns the matplotlib handle of the band (or None) so the caller can build
    a combined legend. Unlike shade_treatment(), this draws no legend itself and
    uses the figures' pink rather than the by-treatment palette.
    """
    if not treatment_config or treatment_config.get("mode") != TREAT_TREATED:
        return None
    start = treatment_config.get("start", 0)
    end = treatment_config.get("end", -1)
    if end is None or end < 0:
        end = ax.get_xlim()[1]
    return ax.axvspan(start, end, color="#F4C7C3", alpha=0.5, zorder=0, label="TMZ")


def _label_line_end(ax, x, y, text, color):
    """Put a small bold track label just past the last point of a line."""
    if len(x) == 0:
        return
    ax.annotate(str(text), xy=(x[-1], y[-1]), xytext=(4, 0),
                textcoords="offset points", color=color, fontsize=7,
                fontweight="bold", va="center", clip_on=True)


def _rolling(values, win, stat):
    """Centered rolling statistic over a 1-D array; returns (centers_idx, vals).

    stat is one of: 'persistence', 'cv', 'slope', 'path', 'net', 'mean'.
    For directional stats ('persistence','path','net') the input is an (n,2)
    array of x/y positions; otherwise a 1-D scalar series. Windows shorter than
    3 points are skipped.
    """
    n = len(values)
    if n < 3 or win < 3:
        return np.array([]), np.array([])
    half = win // 2
    centers, out = [], []
    for c in range(n):
        lo = max(0, c - half)
        hi = min(n, c + half + 1)
        seg = values[lo:hi]
        if len(seg) < 3:
            continue
        if stat == "cv":
            m = np.mean(seg)
            out.append(float(np.std(seg) / m) if m else np.nan)
        elif stat == "slope":
            t = np.arange(len(seg), dtype=float)
            out.append(float(np.polyfit(t, seg, 1)[0]))
        elif stat == "mean":
            out.append(float(np.mean(seg)))
        else:
            # Directional: seg is (m,2) positions.
            d = np.diff(seg, axis=0)
            steplen = np.sqrt((d ** 2).sum(axis=1))
            path = float(steplen.sum())
            net = float(np.sqrt(((seg[-1] - seg[0]) ** 2).sum()))
            if stat == "path":
                out.append(path)
            elif stat == "net":
                out.append(net)
            else:  # persistence = net / path (directionality in the window)
                out.append((net / path) if path > 0 else np.nan)
        centers.append(c)
    return np.asarray(centers, dtype=int), np.asarray(out, dtype=float)


# Window-mode catalogue: label -> (stat, needs_xy, ylabel, title).
WINDOW_MODES = {
    "persistence": ("persistence", True, "Persistência", "Persistência Migratória Local"),
    "area_cv": ("cv", False, "Variação (CV)", "Instabilidade do Envelope Nuclear (CV)"),
    "area_slope": ("slope", False, "Slope (Tendência)", "Taxa de Expansão/Retração (Slope)"),
    "path": ("path", True, "Distância Caminhada", "Esforço Motor na Janela"),
    "net": ("net", True, "Deslocamento Reto", "Eficácia de Migração na Janela"),
    "speed_mean": ("mean", False, "Velocidade Média", "Velocidade Média por Blocos Temporais"),
}


def trajectories(df, mask=None, mode="timeseries", y_col=None,
                 treatment_config=None, pixel_size=1.0, frame_interval=1.0,
                 window=11, smooth=1, min_frames=3, max_tracks=None,
                 show_labels=True, title=None, ylabel=None):
    """One line per cell over time, colored by final outcome, with a TMZ band.

    This reproduces the family of figures the user provided. ``mode`` selects
    the flavour:

      * ``"timeseries"`` : plot ``y_col`` (e.g. ``area_px``, a speed column,
        ``circularity``, ``perimeter``) against frame, one line per track.
      * ``"cumulative"`` : cumulative sum of the per-step value of ``y_col``
        along each track (the odometer / distance-travelled plot). If
        ``y_col`` is None, uses per-step euclidean displacement from x/y.
      * ``"spider"``     : x/y trajectory translated so each track starts at the
        origin (deslocamento X vs Y).
      * ``"window"``     : a centered rolling-window statistic (see
        ``WINDOW_MODES``) selected through ``y_col`` being one of its keys.

    Coloring uses each track's final outcome (Mitose=green, Fim=blue), matching
    the rest of the tool. ``smooth`` applies a centered moving average of that
    many frames to scalar series before plotting (1 = no smoothing). Returns the
    matplotlib Figure.
    """
    needed = [COL_TRACK, COL_FRAME, COL_X, COL_Y]
    d = df.dropna(subset=needed).copy()
    if d.empty:
        raise ValueError("No data with track/frame/position to plot.")
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])

    # For perimeter/circularity (not in the per-frame table) merge them from the
    # mask on demand. Area uses COL_AREA if present, else from the mask too.
    if mode in ("timeseries",) and y_col in ("perimeter", "circularity"):
        morph = analysis.morphology_timeseries(mask)
        if morph.empty:
            raise ValueError("Perimeter/circularity need a mask.")
        d = d.merge(morph[["frame", "track_id", y_col]],
                    on=["frame", "track_id"], how="left")
    if mode in ("timeseries",) and y_col == "area_px" and "area_px" not in d.columns:
        d = analysis.annotate_area(d, mask)

    outcomes = _final_outcome_per_track(d)

    # Optionally cap the number of tracks (longest-lived first) to keep the plot
    # readable on huge datasets.
    track_ids = list(d[COL_TRACK].unique())
    if max_tracks and len(track_ids) > max_tracks:
        order = d.groupby(COL_TRACK).size().sort_values(ascending=False).index
        track_ids = list(order[:max_tracks])
        d = d[d[COL_TRACK].isin(track_ids)]

    fig, ax = plt.subplots(figsize=(13, 6) if mode != "spider" else (9, 8))
    is_window = mode == "window"
    win_stat, needs_xy, win_ylabel, win_title = (None, False, None, None)
    if is_window:
        if y_col not in WINDOW_MODES:
            raise ValueError(f"window y_col must be one of {list(WINDOW_MODES)}")
        win_stat, needs_xy, win_ylabel, win_title = WINDOW_MODES[y_col]

    seen_outcomes = set()
    max_x = 0
    for tid, g in d.groupby(COL_TRACK):
        if len(g) < min_frames:
            continue
        oc = outcomes.get(int(tid), "none")
        color = TRAJ_OUTCOME_COLORS.get(oc, _TRAJ_DEFAULT_COLOR)
        seen_outcomes.add(oc)
        frames = g[COL_FRAME].to_numpy(dtype=float)
        xs = g[COL_X].to_numpy(dtype=float) * pixel_size
        ys = g[COL_Y].to_numpy(dtype=float) * pixel_size

        if mode == "spider":
            X = xs - xs[0]
            Y = ys - ys[0]
            ax.plot(X, Y, color=color, alpha=0.6, linewidth=1.0)
            if show_labels:
                _label_line_end(ax, X, Y, int(tid), color)
            continue

        if mode == "cumulative":
            if y_col and y_col in g.columns:
                step = np.abs(np.diff(g[y_col].to_numpy(dtype=float), prepend=g[y_col].iloc[0]))
            else:
                step = np.sqrt(np.diff(xs, prepend=xs[0]) ** 2 +
                               np.diff(ys, prepend=ys[0]) ** 2)
            yvals = np.cumsum(step)
            xvals = frames
        elif is_window:
            if needs_xy:
                centers, yvals = _rolling(np.stack([xs, ys], axis=1), window, win_stat)
            else:
                if y_col == "speed_mean":
                    series = np.sqrt(np.diff(xs, prepend=xs[0]) ** 2 +
                                     np.diff(ys, prepend=ys[0]) ** 2) / frame_interval
                else:  # area-based windows
                    if "area_px" in g.columns:
                        series = g["area_px"].to_numpy(dtype=float)
                    else:
                        raise ValueError("Area window needs an area column / mask.")
                centers, yvals = _rolling(series, window, win_stat)
            if len(centers) == 0:
                continue
            xvals = frames[centers]
        else:  # timeseries
            if y_col not in g.columns:
                raise ValueError(f"Column '{y_col}' not available for plotting.")
            yvals = g[y_col].to_numpy(dtype=float)
            if smooth and smooth > 1 and len(yvals) >= smooth:
                kern = np.ones(smooth) / smooth
                yvals = np.convolve(yvals, kern, mode="same")
            xvals = frames

        ax.plot(xvals, yvals, color=color, alpha=0.6, linewidth=1.0)
        if len(xvals):
            max_x = max(max_x, float(np.nanmax(xvals)))
        if show_labels:
            _label_line_end(ax, xvals, yvals, int(tid), color)

    # TMZ band (time-axis modes only).
    band = None
    if mode != "spider":
        band = _shade_tmz(ax, treatment_config)
    else:
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Legend: one entry per outcome present, plus the TMZ band if drawn.
    from matplotlib.lines import Line2D
    handles = []
    for oc in ("Mitosis", "Exit", "Death/Senescence", "Ambiguous"):
        if oc in seen_outcomes:
            handles.append(Line2D([0], [0], color=TRAJ_OUTCOME_COLORS[oc],
                                  lw=2, label=TRAJ_OUTCOME_LABELS[oc]))
    if band is not None:
        handles.append(band)
    if handles:
        ax.legend(handles=handles, title="Identificação", loc="upper right",
                  fontsize=9, framealpha=0.9)

    # Axis labels and title per mode.
    if mode == "spider":
        ax.set_xlabel("Deslocamento X")
        ax.set_ylabel("Deslocamento Y")
        ax.set_title(title or "Spider Plot Global", fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
    else:
        tu = "Frame" if frame_interval == 1.0 else "min"
        if is_window:
            ax.set_xlabel(f"Centro da Janela ({tu} Absoluto)")
            ax.set_ylabel(ylabel or win_ylabel)
            ax.set_title(title or win_title, fontweight="bold")
        elif mode == "cumulative":
            ax.set_xlabel(f"Tempo Absoluto ({tu} do Microscópio)")
            ax.set_ylabel(ylabel or "Distância Acumulada")
            ax.set_title(title or "Odômetro: Distância Total Percorrida", fontweight="bold")
        else:
            ax.set_xlabel(f"Tempo Absoluto ({tu} do Microscópio)")
            ax.set_ylabel(ylabel or (y_col or "Valor"))
            ax.set_title(title or (y_col or "Trajetórias"), fontweight="bold")
    ax.grid(alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Single-cell vs population / groups: does one cell stand out as an outlier?
# ---------------------------------------------------------------------------
# Metrics from the per-track summary worth comparing, with display labels. Only
# those actually present (and numeric) in a given summary are used.
COMPARE_METRICS = {
    "lifetime": "Tempo de vida",
    "mean_speed": "Velocidade média",
    "net_displacement": "Deslocamento líquido",
    "total_distance": "Distância total",
    "directionality": "Direcionalidade",
    "confinement_ratio": "Razão de confinamento",
    "diffusion_coeff": "Coef. de difusão",
    "persistence_time": "Tempo de persistência",
    "mean_turning_angle": "Ângulo de giro médio",
    "mean_area": "Área média",
    "max_area": "Área máxima",
}

# Robust z above which a cell is called an outlier for a given metric. z is
# |value - median| / (1.4826*MAD): ~2 ≈ 95th percentile of a normal, 3 ≈ 99.7th.
OUTLIER_Z = 3.0

# Short X-tick labels so long group names (e.g. "desfecho: Morte/Senescência")
# don't overflow into the subplot below. Keys match _grouping_sets() labels;
# anything not listed is shortened generically (prefix dropped, truncated).
_GROUP_SHORT = {
    "mitóticas": "mitót.",
    "não-mitóticas": "n-mitót.",
    TREAT_CONTROL: "control",
    TREAT_TREATED: "treated",
    TREAT_WASHOUT: "washout",
}


def _short_group_label(name):
    """Compact a (possibly long) group label for an X tick."""
    if name in _GROUP_SHORT:
        return _GROUP_SHORT[name]
    s = str(name)
    if s.startswith("desfecho: "):
        s = s[len("desfecho: "):]
    # Common outcome names that are still long.
    s = s.replace("Morte/Senescência", "Morte/Sen.")
    return s if len(s) <= 12 else s[:11] + "…"


def _robust_z(value, arr):
    """Robust z-score of ``value`` against the distribution ``arr``.

    z = |value - median| / (1.4826 * MAD). Returns (z, median, scale) or
    (nan, median, nan) when the spread is degenerate / the sample too small.
    The MAD-based scale is resistant to the very outliers we are hunting for,
    unlike mean+std which an extreme cell would inflate and so hide itself.
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3 or value is None or not np.isfinite(value):
        return np.nan, (float(np.median(a)) if a.size else np.nan), np.nan
    med = float(np.median(a))
    scale = 1.4826 * float(np.median(np.abs(a - med)))
    if scale <= 1e-12:
        scale = float(np.std(a))
    if scale <= 1e-12:
        return np.nan, med, np.nan
    return abs(float(value) - med) / scale, med, scale


def _grouping_sets(summary):
    """Return {group_label: boolean mask into summary} for the natural groups.

    Always includes mitotic vs non-mitotic. Adds one mask per treatment phase
    and one per final outcome that is actually present, so the user gets exactly
    the comparison they asked for ('cell 40 vs mitotic vs non-mitotic vs control').
    """
    groups = {}
    if "is_mitotic" in summary.columns:
        groups["mitóticas"] = summary["is_mitotic"].astype(bool).to_numpy()
        groups["não-mitóticas"] = ~summary["is_mitotic"].astype(bool).to_numpy()
    if "treatment" in summary.columns:
        for t in sorted(str(x) for x in summary["treatment"].dropna().unique()):
            groups[t] = (summary["treatment"].astype(str) == t).to_numpy()
    if "final_outcome" in summary.columns:
        for oc in sorted(str(x) for x in summary["final_outcome"].dropna().unique()):
            if oc in ("", "none", "nan", "None"):
                continue
            label = TRAJ_OUTCOME_LABELS.get(oc, oc)
            groups[f"desfecho: {label}"] = (summary["final_outcome"].astype(str) == oc).to_numpy()
    return groups


def compare_cell(summary, track_id, metrics=None, outlier_z=OUTLIER_Z):
    """Compare one cell to the whole dataset and to each natural group.

    Returns a dict:
      {
        "track_id": int,
        "found": bool,
        "metrics": {metric: {
            "label": str, "value": float,
            "all":   {"z","median","scale","n","is_outlier"},
            "groups": {group_label: {"z","median","scale","n","is_outlier","in_group"}},
        }},
        "outlier_metrics": [metric, ...],   # metrics where it's an outlier vs ALL
      }

    "vs ALL" uses every other track (the cell itself is excluded from the
    reference distribution so it can't mask its own deviation). "vs group" uses
    the other members of that group. A cell is flagged an outlier on a metric
    when its robust z exceeds ``outlier_z`` against the dataset.
    """
    out = {"track_id": int(track_id), "found": False, "metrics": {},
           "outlier_metrics": []}
    if summary is None or summary.empty or "track_id" not in summary.columns:
        return out
    row = summary[summary["track_id"].astype(int) == int(track_id)]
    if row.empty:
        return out
    out["found"] = True
    row = row.iloc[0]

    avail = metrics or [m for m in COMPARE_METRICS
                        if m in summary.columns
                        and pd.api.types.is_numeric_dtype(summary[m])]
    groups = _grouping_sets(summary)
    self_mask = (summary["track_id"].astype(int) == int(track_id)).to_numpy()

    for m in avail:
        value = row.get(m)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        col = summary[m].to_numpy(dtype=float)

        # vs ALL other cells (exclude self from the reference distribution).
        others = col[~self_mask]
        z_all, med_all, sc_all = _robust_z(value, others)
        is_out_all = bool(np.isfinite(z_all) and z_all > outlier_z)
        entry = {
            "label": COMPARE_METRICS.get(m, m),
            "value": value,
            "all": {"z": z_all, "median": med_all, "scale": sc_all,
                    "n": int(np.isfinite(others).sum()), "is_outlier": is_out_all},
            "groups": {},
        }
        for gname, gmask in groups.items():
            in_group = bool(gmask[self_mask][0]) if self_mask.any() else False
            ref = col[gmask & ~self_mask]
            z_g, med_g, sc_g = _robust_z(value, ref)
            entry["groups"][gname] = {
                "z": z_g, "median": med_g, "scale": sc_g,
                "n": int(np.isfinite(ref).sum()),
                "is_outlier": bool(np.isfinite(z_g) and z_g > outlier_z),
                "in_group": in_group,
            }
        out["metrics"][m] = entry
        if is_out_all:
            out["outlier_metrics"].append(m)
    return out


def compare_cell_text(comparison):
    """Human-readable summary of compare_cell() for the notifications / console."""
    tid = comparison.get("track_id")
    if not comparison.get("found"):
        return f"Célula {tid} não está no resumo por-track (sem trajetória válida?)."
    n_out = len(comparison["outlier_metrics"])
    if n_out == 0:
        head = f"Célula {tid}: dentro do normal em todas as métricas (nenhum outlier vs dataset)."
    else:
        labs = ", ".join(comparison["metrics"][m]["label"]
                         for m in comparison["outlier_metrics"])
        head = (f"Célula {tid}: destaca-se como OUTLIER em {n_out} "
                f"métrica(s) vs dataset — {labs}.")
    lines = [head]
    for m, e in comparison["metrics"].items():
        z = e["all"]["z"]
        zt = "n/d" if not np.isfinite(z) else f"{z:.1f}σ"
        mark = "  <-- OUTLIER" if e["all"]["is_outlier"] else ""
        med = e["all"]["median"]
        lines.append(
            f"  {e['label']}: {e['value']:.2f} "
            f"(mediana dataset {med:.2f}, z={zt}){mark}")
    return "\n".join(lines)


def _draw_compare_axis(ax, summary, comp, m, track_id, groups, self_mask):
    """Draw ONE metric's comparison on ``ax``: violin + slim box per group + star.

    With thousands of cells a point cloud collapses into a solid black smear
    that hides the target, so the distribution is shown purely as a violin
    (density shape) plus a slim boxplot for the quartiles/median. The target
    cell is a red star on the dataset and on each group it belongs to.
    """
    e = comp["metrics"][m]
    col = summary[m].to_numpy(dtype=float)

    labels, data, colors = ["dataset"], [col[np.isfinite(col)]], ["#B0BEC5"]
    for gname, gmask in groups.items():
        ref = col[gmask]
        ref = ref[np.isfinite(ref)]
        if ref.size:
            labels.append(gname)
            data.append(ref)
            colors.append(GROUP_COLORS.get(gname, "#9E9E9E"))

    positions = list(range(1, len(labels) + 1))

    # Violin (density). A single-member or zero-spread group can't form a
    # violin, so those simply show their boxplot/median line instead.
    vparts_idx = [i for i, d in enumerate(data) if d.size >= 2 and np.ptp(d) > 0]
    if vparts_idx:
        vp = ax.violinplot([data[i] for i in vparts_idx],
                           positions=[positions[i] for i in vparts_idx],
                           widths=0.8, showmeans=False, showmedians=False,
                           showextrema=False)
        for body, i in zip(vp["bodies"], vparts_idx):
            body.set_facecolor(colors[i])
            body.set_alpha(0.45)
            body.set_edgecolor("#555555")
            body.set_linewidth(0.6)

    # Slim boxplot on top for quartiles/median.
    bp = ax.boxplot(data, positions=positions, widths=0.18,
                    patch_artist=True, showfliers=False, zorder=3)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    for med in bp["medians"]:
        med.set_color("black")

    # Target cell: red star on the dataset violin and on every group it belongs
    # to (so you see it inside its own group AND vs the rest).
    val = e["value"]
    if np.isfinite(val):
        ax.scatter([1], [val], marker="*", s=420, color="#D32F2F",
                   edgecolors="black", linewidths=0.9, zorder=6)
        for gi, gname in enumerate(labels[1:], start=2):
            if e["groups"].get(gname, {}).get("in_group"):
                ax.scatter([gi], [val], marker="*", s=420, color="#D32F2F",
                           edgecolors="black", linewidths=0.9, zorder=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{_short_group_label(l)}\n(n={d.size})" for l, d in zip(labels, data)],
        fontsize=10)
    ax.tick_params(axis="x", labelrotation=0)
    z = e["all"]["z"]
    zt = "n/d" if not np.isfinite(z) else f"{z:.1f}σ"
    flag = "   ⚠ OUTLIER" if e["all"]["is_outlier"] else ""
    ax.set_title(f"{e['label']}  —  z={zt}{flag}", fontsize=14, fontweight="bold",
                 color=("#D32F2F" if e["all"]["is_outlier"] else "#222222"),
                 pad=12)
    ax.grid(axis="y", alpha=0.3)


def compare_cell_pages(summary, track_id, metrics=None, outlier_z=OUTLIER_Z,
                       per_page=1, ncol=1):
    """Build a LIST of figures (pages), each with up to ``per_page`` metrics.

    Default is ONE metric per page: a single large, legible plot you page
    through with the navigable dialog (CompareCellDialog), so titles and X
    labels never overlap. ``per_page``/``ncol`` can be raised to pack more
    metrics per page if desired.
    Returns (pages, comp) where pages is list[Figure] and comp is the data dict.
    """
    comp = compare_cell(summary, track_id, metrics=metrics, outlier_z=outlier_z)
    if not comp["found"]:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, f"Célula {track_id} não encontrada no resumo.",
                ha="center", va="center")
        ax.axis("off")
        return [fig], comp

    mets = list(comp["metrics"].keys())
    if not mets:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "Nenhuma métrica numérica disponível para comparar.",
                ha="center", va="center")
        ax.axis("off")
        return [fig], comp

    groups = _grouping_sets(summary)
    self_mask = (summary["track_id"].astype(int) == int(track_id)).to_numpy()

    n_out = len(comp["outlier_metrics"])
    head = (f"Célula {track_id} vs dataset e grupos  —  "
            + (f"outlier em {n_out} métrica(s)" if n_out
               else "dentro do normal em todas as métricas"))

    per_page = max(1, int(per_page))
    ncol = max(1, min(int(ncol), per_page))
    chunks = [mets[i:i + per_page] for i in range(0, len(mets), per_page)]

    pages = []
    for pi, chunk in enumerate(chunks):
        nrow = int(np.ceil(len(chunk) / ncol))
        if per_page == 1:
            # One big plot per page: wide enough for ~8 group violins + labels.
            figsize = (11.5, 6.4)
        else:
            figsize = (7.2 * ncol, 4.4 * nrow)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax in axes[len(chunk):]:
            ax.axis("off")
        for ax, m in zip(axes, chunk):
            _draw_compare_axis(ax, summary, comp, m, track_id, groups, self_mask)
        page_tag = (f"   (página {pi + 1}/{len(chunks)})" if len(chunks) > 1 else "")
        fig.suptitle(head + page_tag, fontsize=13, fontweight="bold")
        # Generous spacing so titles/X-labels never collide.
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0, w_pad=2.5)
        pages.append(fig)
    return pages, comp


def compare_cell_figure(summary, track_id, metrics=None, outlier_z=OUTLIER_Z):
    """Backward-compatible single-figure entry point.

    Returns one figure containing ALL metrics (2-per-row, well spaced). Prefer
    compare_cell_pages() + CompareCellDialog for the paginated, navigable view.
    """
    pages, _ = compare_cell_pages(summary, track_id, metrics=metrics,
                                  outlier_z=outlier_z,
                                  per_page=max(1, _count_metrics(summary, metrics)),
                                  ncol=2)
    return pages[0]


def _count_metrics(summary, metrics):
    if metrics:
        return len(metrics)
    if summary is None or summary.empty:
        return 1
    return max(1, sum(1 for m in COMPARE_METRICS
                      if m in summary.columns
                      and pd.api.types.is_numeric_dtype(summary[m])))


# ---------------------------------------------------------------------------
# Standard research plots (operate on the per-track summary)
# ---------------------------------------------------------------------------
def time_unit(frame_interval):
    return "min" if frame_interval != 1.0 else "frames"


def length_unit(pixel_size):
    return "um" if pixel_size != 1.0 else "px"


def lifetime_figure(summary, frame_interval):
    treat_groups = {t: summary.loc[summary.treatment == t, "lifetime"].values
                    for t in summary.treatment.unique()}
    mito_groups = {
        "mitotic": summary.loc[summary.is_mitotic, "lifetime"].values,
        "non-mitotic": summary.loc[~summary.is_mitotic, "lifetime"].values,
    }
    unit = time_unit(frame_interval)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    box_with_points(axes[0], treat_groups, "Lifetime by treatment", f"Lifetime ({unit})")
    box_with_points(axes[1], mito_groups, "Lifetime: mitotic vs non-mitotic", f"Lifetime ({unit})")
    return fig


def migration_figure(summary, pixel_size, frame_interval):
    lu, tu = length_unit(pixel_size), time_unit(frame_interval)
    groups = sorted(summary.treatment.unique())
    speed = {t: summary.loc[summary.treatment == t, "mean_speed"].values for t in groups}
    disp = {t: summary.loc[summary.treatment == t, "net_displacement"].values for t in groups}
    direction = {t: summary.loc[summary.treatment == t, "directionality"].values for t in groups}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    box_with_points(axes[0], speed, "Mean speed", f"Speed ({lu}/{tu})")
    box_with_points(axes[1], disp, "Net displacement", f"Displacement ({lu})")
    box_with_points(axes[2], direction, "Directionality ratio", "Net / total path")
    return fig


def motility_figure(summary, pixel_size, frame_interval):
    """Boxplots of the richer motility metrics by treatment."""
    lu, tu = length_unit(pixel_size), time_unit(frame_interval)
    groups = sorted(summary.treatment.unique())

    def by_group(col):
        return {t: summary.loc[summary.treatment == t, col].dropna().values for t in groups}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    box_with_points(axes[0], by_group("diffusion_coeff"),
                    "Diffusion coefficient", f"D ({lu}^2/{tu})")
    box_with_points(axes[1], by_group("persistence_time"),
                    "Persistence time", f"Time ({tu})")
    box_with_points(axes[2], by_group("mean_turning_angle"),
                    "Mean turning angle", "Angle (rad)")
    box_with_points(axes[3], by_group("confinement_ratio"),
                    "Confinement ratio", "Net / total path")
    return fig


def outcomes_figure(summary):
    counts = summary["final_outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ax.bar(counts.index, counts.values, color=colors[:len(counts)])
    ax.set_title("Track outcome distribution", fontweight="bold")
    ax.set_ylabel("Number of tracks")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    return fig, counts


def growth_figure(df, treatment_config, max_frame):
    valid = df.dropna(subset=[COL_TRACK, COL_FRAME]).copy()
    valid = valid[valid[COL_TRACK] > 0]
    counts = valid.groupby(COL_FRAME)[COL_TRACK].nunique()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(counts.index, counts.values, color="#4C72B0", linewidth=2)
    ax.set_title("Population growth (unique cells per frame)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cell count")
    ax.grid(alpha=0.3)
    shade_treatment(ax, treatment_config, max_frame)
    return fig


def division_figure(df, treatment_config, max_frame):
    links = df[df[COL_PARENT] > 0]
    if links.empty:
        return None
    birth = links.groupby(COL_TRACK)[COL_FRAME].min()
    events = birth.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(events.index, events.values, color="#C44E52", width=0.9)
    ax.set_title("Division events over time", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of divisions")
    ax.grid(axis="y", alpha=0.3)
    shade_treatment(ax, treatment_config, max_frame)
    return fig, int(events.sum())


def msd_figure(df, pixel_size, frame_interval, max_lag=20):
    valid = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    valid[COL_TRACK] = valid[COL_TRACK].astype(int)
    valid = valid[valid[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])
    if valid.empty:
        return None
    ps = pixel_size
    lag_values = {lag: [] for lag in range(1, max_lag + 1)}
    for _, g in valid.groupby(COL_TRACK):
        xs = g[COL_X].to_numpy(dtype=float) * ps
        ys = g[COL_Y].to_numpy(dtype=float) * ps
        n = len(xs)
        for lag in range(1, min(max_lag, n - 1) + 1):
            dx = xs[lag:] - xs[:-lag]
            dy = ys[lag:] - ys[:-lag]
            lag_values[lag].append(np.mean(dx ** 2 + dy ** 2))
    lags = [lag for lag in lag_values if lag_values[lag]]
    if not lags:
        return None
    msd = [np.mean(lag_values[lag]) for lag in lags]
    tu, lu = time_unit(frame_interval), length_unit(pixel_size)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(lags) * frame_interval, msd, "o-", color="#4C72B0", linewidth=2)
    ax.set_title("Ensemble mean squared displacement", fontweight="bold")
    ax.set_xlabel(f"Time lag ({tu})")
    ax.set_ylabel(f"MSD ({lu}^2)")
    ax.grid(alpha=0.3)
    return fig


def area_figure(summary, df, mask, pixel_size, treatment_config, max_frame):
    lu = length_unit(pixel_size)
    area_groups = {t: summary.loc[summary.treatment == t, "mean_area"].values
                   for t in sorted(summary.treatment.unique())}
    px_area = pixel_size ** 2
    frames, mean_areas = [], []
    for f in range(mask.shape[0]):
        sizes = list(analysis.areas_for_frame(mask[f], px_area).values())
        if sizes:
            frames.append(f)
            mean_areas.append(np.mean(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    box_with_points(axes[0], area_groups, "Mean area by treatment", f"Area ({lu}^2)")
    axes[1].plot(frames, mean_areas, color="#4C72B0", linewidth=2)
    axes[1].set_title("Mean cell area over time", fontweight="bold")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel(f"Mean area ({lu}^2)")
    axes[1].grid(alpha=0.3)
    shade_treatment(axes[1], treatment_config, max_frame)
    return fig
