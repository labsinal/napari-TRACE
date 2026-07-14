"""
Derived export tables written alongside the main CSV on SAVE ALL.

Pure pandas/numpy (no Qt) so it is unit-testable. Produces, for the full dataset
and for the validated subset:

  * validated cells       -- the rows of the main table for validated tracks
                             (every cell of a reviewed lineage + every
                             individually validated cell);
  * features table        -- one row per (track, frame) with instantaneous
                             nuclear morphometry (area_px, perimeter,
                             circularity, aspect, area_box, radius_ratio,
                             roundness, NII) and per-step migration (step
                             displacement, cumulative path, net-from-start,
                             instantaneous speed);
  * windows table         -- one row per (track, non-overlapping window of
                             ``window`` frames) with the variation accumulated
                             over the window: displacement, delta path,
                             persistence, area_cv, area_slope, path, net,
                             speed_mean, and the per-window mean and delta of
                             every feature (including NII);
  * per-track summary     -- the statistics summary, full and validated.

"validated" is defined exactly as the user describes: a whole reviewed lineage
contributes all of its cells; an individually validated cell contributes itself.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import analysis, lineage, fluorescence
from .config import (COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT,
                     COL_TREATMENT, COL_BORDER, COL_AREA, is_quarantined_id)


# Feature columns whose per-window mean and delta are tabulated.
_FEATURE_COLS = ["area_px", "perimeter", "circularity", "aspect", "area_box",
                 "radius_ratio", "roundness", "nii",
                 "step_disp", "cum_path", "net_disp", "speed"]


# ---------------------------------------------------------------------------
# Validated-cell selection
# ---------------------------------------------------------------------------
def validated_track_ids(df, reviewed_roots, validated_cells) -> set:
    """Track IDs counted as validated.

    A track is validated if its lineage root is in ``reviewed_roots`` (the whole
    family was reviewed) or the track itself is in ``validated_cells`` (an
    individually confirmed cell).
    """
    if df is None or df.empty:
        return set()
    reviewed = {int(r) for r in (reviewed_roots or set())}
    individual = {int(t) for t in (validated_cells or set())}
    ids = df[COL_TRACK].dropna().astype(int).unique()
    out = set(int(t) for t in individual)
    if reviewed:
        p_of = lineage.parent_of(df)
        cache = {}
        for tid in ids:
            root = lineage._root_in_map(p_of, int(tid), cache)
            if int(root) in reviewed:
                out.add(int(tid))
    return {t for t in out if t in set(int(x) for x in ids)}


def subset_tracks(df, track_ids):
    """Rows of ``df`` whose track is in ``track_ids`` (empty frame-safe)."""
    if df is None or df.empty or not track_ids:
        return df.iloc[0:0].copy() if df is not None else None
    ids = {int(t) for t in track_ids}
    return df[df[COL_TRACK].astype(int).isin(ids)].copy()


# ---------------------------------------------------------------------------
# Per-frame feature table
# ---------------------------------------------------------------------------
def features_table(df, mask, pixel_size=1.0, frame_interval=1.0,
                   channels=None, ring_opts=None, texture=True,
                   extra_tables=None):
    """One row per (track, frame): morphometry + per-step migration."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[(d[COL_TRACK] > 0) & (~d[COL_TRACK].map(is_quarantined_id))]
    if d.empty:
        return pd.DataFrame()
    d[COL_FRAME] = d[COL_FRAME].astype(int)
    d = d.sort_values([COL_TRACK, COL_FRAME])

    keep = [c for c in (COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME,
                        COL_PARENT, COL_TREATMENT, COL_BORDER) if c in d.columns]
    out = d[keep].copy()

    # Per-step migration (scaled to physical units when pixel_size != 1).
    gx = d.groupby(COL_TRACK)[COL_X]
    gy = d.groupby(COL_TRACK)[COL_Y]
    dx = gx.diff().to_numpy(float) * pixel_size
    dy = gy.diff().to_numpy(float) * pixel_size
    step = np.sqrt(dx ** 2 + dy ** 2)
    step[~np.isfinite(step)] = 0.0
    out["step_disp"] = step
    out["cum_path"] = (pd.Series(step, index=d.index)
                       .groupby(d[COL_TRACK]).cumsum().to_numpy())
    x0 = d.groupby(COL_TRACK)[COL_X].transform("first").to_numpy(float)
    y0 = d.groupby(COL_TRACK)[COL_Y].transform("first").to_numpy(float)
    out["net_disp"] = np.sqrt(((d[COL_X].to_numpy(float) - x0) * pixel_size) ** 2 +
                              ((d[COL_Y].to_numpy(float) - y0) * pixel_size) ** 2)
    out["speed"] = step / frame_interval if frame_interval else step

    # Nuclear morphometry merged from the mask.
    if mask is not None:
        nm = analysis.nuclear_morphometry(mask)
        if not nm.empty:
            nm = nm.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
            out = out.merge(nm, on=[COL_TRACK, COL_FRAME], how="left")

    # Per-channel fluorescence features (marked channels only), merged by cell/frame.
    ropts = ring_opts or {}
    dil = int(ropts.get("dilation", fluorescence.RING_DILATION_DEFAULT))
    gap = int(ropts.get("gap", fluorescence.RING_GAP_DEFAULT))
    for name, stack in (channels or {}).items():
        inten = fluorescence.measure_intensity(stack, mask, channel_name=name,
                                               dilation=dil, gap=gap)
        if not inten.empty:
            inten = inten.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
            out = out.merge(inten, on=[COL_TRACK, COL_FRAME], how="left")
        if texture:
            hara = fluorescence.haralick_features(stack, mask, channel_name=name)
            if not hara.empty:
                hara = hara.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
                out = out.merge(hara, on=[COL_TRACK, COL_FRAME], how="left")

    # Interactive-readout tables (ERK-KTR, 53BP1) accumulated in the session.
    for extra in (extra_tables or []):
        if extra is None or extra.empty:
            continue
        e = extra.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
        merge_cols = [c for c in (COL_TRACK, COL_FRAME) if c in e.columns]
        if len(merge_cols) == 2:
            dup = [c for c in e.columns if c in out.columns and c not in merge_cols]
            e = e.drop(columns=dup)
            out = out.merge(e, on=merge_cols, how="left")

    out = out.reset_index(drop=True)
    _add_lifetime_midframe(out, frame_interval)
    return out


def _add_lifetime_midframe(out, frame_interval=1.0):
    """Add a per-track ``lifetime`` carried on the middle-of-life row only.

    ``lifetime = (last_frame - first_frame + 1) * frame_interval`` is written on
    the single (track, frame) row at the centre of the track's frames; every
    other row is NaN. This gives exactly one value per cell for averaging, and
    puts it away from the ends so trimming the first/last frames in a later
    cleanup never drops it. Operates in place on ``out``.
    """
    out["lifetime"] = np.nan
    if out.empty:
        return out
    for _tid, g in out.groupby(COL_TRACK):
        g = g.sort_values(COL_FRAME)
        frames = g[COL_FRAME].to_numpy(dtype=float)
        lifetime = (frames.max() - frames.min() + 1) * frame_interval
        mid_label = g.index[len(g) // 2]        # middle row in frame order
        out.loc[mid_label, "lifetime"] = lifetime
    return out


# ---------------------------------------------------------------------------
# Windowed (accumulated) table
# ---------------------------------------------------------------------------
def _first_last_finite(v):
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan, np.nan
    idx = np.nonzero(ok)[0]
    return float(v[idx[0]]), float(v[idx[-1]])


def windows_table(df, mask, window=7, pixel_size=1.0, frame_interval=1.0,
                  feats=None):
    """Sliding-window (per-frame) accumulated table, one row per (track, frame).

    For every frame the metrics are accumulated over a window of ``window``
    frames centred on that frame (clamped at the track edges, so short tracks and
    edge frames still get a row, with a smaller ``n_frames``). Each row reports
    the variation over its window: net displacement, path length, persistence
    (net/path), area_cv, area_slope, mean speed, and the per-window mean and
    delta (last - first) of every feature, including NII. The window is centred
    on ``frame_center``, matching the trajectory "window" plot convention.
    """
    window = max(2, int(window))
    half = max(1, window // 2)
    feats = feats or _FEATURE_COLS
    base = features_table(df, mask, pixel_size, frame_interval)
    if base is None or base.empty:
        return pd.DataFrame()

    rows = []
    for tid, g in base.groupby(COL_TRACK):
        g = g.sort_values(COL_FRAME)
        frames = g[COL_FRAME].to_numpy(float)
        xs = g[COL_X].to_numpy(float) * pixel_size
        ys = g[COL_Y].to_numpy(float) * pixel_size
        n = len(g)
        for c in range(n):
            lo, hi = max(0, c - half), min(n, c + half + 1)
            if hi - lo < 2:
                continue
            sl = slice(lo, hi)
            fr = frames[sl]
            wx, wy = xs[sl], ys[sl]
            step = np.sqrt(np.diff(wx) ** 2 + np.diff(wy) ** 2)
            path = float(step.sum())
            net = float(np.sqrt((wx[-1] - wx[0]) ** 2 + (wy[-1] - wy[0]) ** 2))
            span = (fr[-1] - fr[0]) * frame_interval
            rec = {
                COL_TRACK: int(tid),
                "frame_center": int(frames[c]),
                "frame_start": int(fr[0]), "frame_end": int(fr[-1]),
                "n_frames": int(hi - lo),
                "displacement": net, "delta_path": path,
                "persistence": (net / path) if path > 0 else np.nan,
                "path": path, "net": net,
                "speed_mean": (path / span) if span > 0 else np.nan,
            }
            area = g["area_px"].to_numpy(float)[sl] if "area_px" in g else np.array([])
            af = area[np.isfinite(area)] if area.size else area
            m = af.mean() if af.size else np.nan
            rec["area_cv"] = (af.std() / m) if (af.size and m not in (0, np.nan)) else np.nan
            if area.size and np.isfinite(area).sum() >= 2 and np.ptp(fr) > 0:
                ok = np.isfinite(area)
                rec["area_slope"] = float(np.polyfit(fr[ok], area[ok], 1)[0])
            else:
                rec["area_slope"] = np.nan
            for col in feats:
                if col in g.columns:
                    v = g[col].to_numpy(float)[sl]
                    vf = v[np.isfinite(v)]
                    rec[f"mean_{col}"] = float(vf.mean()) if vf.size else np.nan
                    a, b = _first_last_finite(v)
                    rec[f"delta_{col}"] = (b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Atomic writer
# ---------------------------------------------------------------------------
def _atomic_csv(frame, path):
    if frame is None:
        return None
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def write_all(work_dir, df, mask, reviewed_roots, validated_cells, base_name,
              window=7, pixel_size=1.0, frame_interval=1.0,
              channels=None, ring_opts=None, extra_tables=None):
    """Write every derived table (full + validated) into ``work_dir/exports``.

    Returns a dict of {key: path} for the files written and the count of
    validated tracks.
    """
    out_dir = os.path.join(work_dir, "exports")
    os.makedirs(out_dir, exist_ok=True)

    valid_ids = validated_track_ids(df, reviewed_roots, validated_cells)

    features = features_table(df, mask, pixel_size, frame_interval,
                              channels=channels, ring_opts=ring_opts,
                              extra_tables=extra_tables)
    windows = windows_table(df, mask, window, pixel_size, frame_interval)
    summary = analysis.compute_track_summary(df, mask, pixel_size, frame_interval)
    validated_main = subset_tracks(df, valid_ids)

    def p(name):
        return os.path.join(out_dir, f"{base_name}_{name}.csv")

    written = {}
    written["validated_cells"] = _atomic_csv(validated_main, p("validated_cells"))
    written["features"] = _atomic_csv(features, p("features"))
    written["features_validated"] = _atomic_csv(
        subset_tracks(features, valid_ids), p("features_validated"))
    written["windows"] = _atomic_csv(windows, p("windows"))
    written["windows_validated"] = _atomic_csv(
        subset_tracks(windows, valid_ids), p("windows_validated"))
    written["summary"] = _atomic_csv(summary, p("summary"))
    written["summary_validated"] = _atomic_csv(
        subset_tracks(summary, valid_ids), p("summary_validated"))

    if channels and features is not None and not features.empty:
        chan_cols = [c for c in features.columns
                     if any(c.startswith(f"{n}_") for n in channels)]
        if chan_cols:
            fl_only = features[[COL_TRACK, COL_FRAME] + chan_cols]
            written["fluorescence"] = _atomic_csv(fl_only, p("fluorescence"))
    return {"files": {k: v for k, v in written.items() if v},
            "n_validated_tracks": len(valid_ids),
            "dir": out_dir}
