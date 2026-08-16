"""
Analysis primitives: centroids, anomaly detection, morphology and per-track
summaries.

Provides a single anomaly detector shared by the diagnostics panel and the
pre-save filter, a vectorized centroid helper, and morphology-based
segmentation error detection (area/solidity/eccentricity) using
outlier-robust statistics (median + MAD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_TREATMENT,
    OUTCOME_MITOSIS, OUTCOME_EXIT, OUTCOME_DEATH, TREAT_CONTROL, Thresholds,
)

# skimage is a hard dependency of the curator; import at top.
from skimage.measure import label as cc_label, regionprops_table


# ---------------------------------------------------------------------------
# Vectorized centroids
# ---------------------------------------------------------------------------
def centroids_for_frame(mask_plane: np.ndarray) -> dict[int, tuple[float, float]]:
    """Return {label_id: (x, y)} centroids for every nonzero label in a plane.

    Computed in one vectorized pass with ``np.bincount`` instead of looping
    ``np.where`` per cell. ``x`` is the column coordinate, ``y`` the row.
    """
    flat = mask_plane.ravel()
    nz = flat > 0
    if not nz.any():
        return {}
    ids = flat[nz]
    ys, xs = np.divmod(np.flatnonzero(nz), mask_plane.shape[1])
    maxid = int(ids.max())
    counts = np.bincount(ids, minlength=maxid + 1).astype(float)
    sum_x = np.bincount(ids, weights=xs.astype(float), minlength=maxid + 1)
    sum_y = np.bincount(ids, weights=ys.astype(float), minlength=maxid + 1)
    out = {}
    present = np.flatnonzero(counts > 0)
    for i in present:
        if i == 0:
            continue
        out[int(i)] = (float(sum_x[i] / counts[i]), float(sum_y[i] / counts[i]))
    return out


def centroid_of_label(mask_plane: np.ndarray, label_id: int):
    """Centroid (x, y) of a single label in a plane, or None if absent."""
    coords = np.where(mask_plane == label_id)
    if coords[0].size == 0:
        return None
    return float(np.mean(coords[1])), float(np.mean(coords[0]))


def areas_for_frame(mask_plane: np.ndarray, pixel_area: float = 1.0) -> dict[int, float]:
    """Return {label_id: area} in physical units for one plane (vectorized)."""
    ids, counts = np.unique(mask_plane, return_counts=True)
    return {int(i): int(c) * pixel_area for i, c in zip(ids, counts) if i != 0}


# ---------------------------------------------------------------------------
# Mask <-> track_id reconciliation
# ---------------------------------------------------------------------------
# The whole tool assumes the mask's pixel value equals the row's track_id: a
# click reads the pixel value as the ID, focus highlights selected_label ==
# track_id, and every mask-derived feature (area/morphometry/fluorescence) is
# joined to the table by that equality. When a dataset's mask is labelled by
# per-frame segmentation labels instead (the tracker stored the identity only in
# the CSV), that equality is broken and every mask-derived feature comes out NaN
# -- silently. These helpers detect the mismatch and repair it by relabelling the
# mask so pixel value == track_id, keyed on each row's centroid.
def mask_track_correspondence(df, mask) -> float:
    """Fraction of table rows whose mask pixel at (round y, round x) == track_id.

    1.0 means the mask is already labelled by track_id (the tool's assumption);
    a low value means the mask uses different labels and must be reconciled.
    Returns 1.0 for empty input (nothing to reconcile).
    """
    if df is None or df.empty or mask is None:
        return 1.0
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y])
    if d.empty:
        return 1.0
    h, w = mask.shape[-2:]
    hits = total = 0
    for f, g in d.groupby(d[COL_FRAME].astype(int)):
        if f < 0 or f >= mask.shape[0]:
            continue
        plane = mask[f]
        xs = g[COL_X].round().astype(int).to_numpy()
        ys = g[COL_Y].round().astype(int).to_numpy()
        tids = g[COL_TRACK].astype(int).to_numpy()
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        for xi, yi, tid in zip(xs[ok], ys[ok], tids[ok]):
            total += 1
            if int(plane[yi, xi]) == int(tid):
                hits += 1
    return hits / total if total else 1.0


def relabel_mask_to_track_ids(df, mask):
    """Return (new_mask, n_relabeled): mask relabelled so pixel value == track_id.

    For each frame the mask label found under each row's centroid is remapped to
    that row's track_id. Remapping goes through a transient offset so a relabelled
    value can never collide with an as-yet-unprocessed original label. Blobs with
    no table row in a frame keep their original label (they are untracked objects;
    Rescue orphans / the duplicate check handle them). Operates on a copy.
    """
    if df is None or df.empty or mask is None:
        return mask, 0
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y])
    if d.empty:
        return mask, 0
    new = mask.copy()
    h, w = mask.shape[-2:]
    off = config.RESEQ_OFFSET
    n = 0
    for f, g in d.groupby(d[COL_FRAME].astype(int)):
        if f < 0 or f >= mask.shape[0]:
            continue
        plane = mask[f]
        xs = g[COL_X].round().astype(int).to_numpy()
        ys = g[COL_Y].round().astype(int).to_numpy()
        tids = g[COL_TRACK].astype(int).to_numpy()
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mapping = {}
        for xi, yi, tid in zip(xs[ok], ys[ok], tids[ok]):
            lbl = int(plane[yi, xi])
            if lbl > 0:
                mapping[lbl] = int(tid)
        if not mapping:
            continue
        out = new[f]
        for old, newid in mapping.items():
            out[plane == old] = newid + off
        shifted = out >= off
        out[shifted] -= off
        n += len(mapping)
    return new, n


# ---------------------------------------------------------------------------
# Unified anomaly detection
# ---------------------------------------------------------------------------
@dataclass
class AnomalyReport:
    short: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    jumps: list = field(default_factory=list)
    no_outcome: list = field(default_factory=list)
    mitosis_anomalies: list = field(default_factory=list)

    @property
    def flagged_ids(self) -> set:
        """IDs considered defective for the pre-save quality filter."""
        return set(self.short) | set(self.gaps) | set(self.jumps)

    def is_empty(self) -> bool:
        return not (self.short or self.gaps or self.jumps
                    or self.no_outcome or self.mitosis_anomalies)


def detect_anomalies(df: pd.DataFrame, thresholds: Thresholds,
                     include_outcome_checks: bool = True) -> AnomalyReport:
    """Single source of truth for short tracks, gaps, jumps and outcome issues.

    Used by both the diagnostic panel and the pre-save filter, so the
    two can never drift apart again. Quarantined IDs are skipped.
    """
    rep = AnomalyReport()
    if df is None or df.empty:
        return rep

    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return rep
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0]
    d = d.sort_values([COL_TRACK, COL_FRAME])

    cycles = []
    for tid, g in d.groupby(COL_TRACK):
        if config.is_quarantined_id(tid):
            continue
        frames = g[COL_FRAME].to_numpy(dtype=float)
        x = g[COL_X].to_numpy(dtype=float)
        y = g[COL_Y].to_numpy(dtype=float)

        # Does this track already have a valid final outcome? A curated cell
        # that ends in Mitosis / Exit / Death / Ambiguous has a deliberate,
        # reviewed ending, so being "short" or having a small frame gap is
        # expected (a daughter that divides again is naturally short; a cell
        # vanishing for a few frames around a division/exit is normal) and must
        # NOT be reported as a likely error. This is the single biggest source
        # of false positives on a fully-curated dataset.
        outcomes = (set(g[COL_OUTCOME].astype(str).unique())
                    if COL_OUTCOME in g else set())
        has_final = bool(outcomes & set(config.FINAL_OUTCOMES))

        if len(frames) < thresholds.min_valid_frames and not has_final:
            rep.short.append(int(tid))
        if (len(frames) > 1 and not has_final
                and np.any(np.diff(frames) > thresholds.gap_tolerance)):
            rep.gaps.append(int(tid))
        if len(frames) > 1:
            dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            if np.any(dist > thresholds.max_jump_px):
                rep.jumps.append(int(tid))

        if include_outcome_checks:
            if not has_final:
                rep.no_outcome.append(
                    f"Cell {int(tid)} has no final outcome [first frame {int(frames[0])}]")
            last_outcome = str(g[COL_OUTCOME].iloc[-1]) if COL_OUTCOME in g else ""
            if last_outcome == OUTCOME_MITOSIS:
                cycles.append({"id": int(tid), "life": float(frames[-1] - frames[0])})

    if include_outcome_checks and len(cycles) >= 4:
        # Robust spread: flag a mitosis lifetime only if it is a strong outlier
        # by median + MAD (not merely outside median*0.5..1.5, which on a
        # naturally wide lifetime distribution flags ~half the mothers).
        lives = np.array([c["life"] for c in cycles], dtype=float)
        med = float(np.median(lives))
        mad = float(np.median(np.abs(lives - med)))
        if mad > 0:
            for c in cycles:
                z = abs(c["life"] - med) / (1.4826 * mad)
                if z > thresholds.mitosis_life_mad_k:
                    rep.mitosis_anomalies.append(
                        f"ID {c['id']} (lived {c['life']:.0f} frames; median {med:.0f})")
    return rep


# ---------------------------------------------------------------------------
# Morphology-based segmentation errors (median + MAD)
# ---------------------------------------------------------------------------
def _robust_outliers(values: np.ndarray, k: float):
    """Return a boolean 'too large' mask using median + MAD (robust to outliers)."""
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad <= 0:
        # Degenerate spread: fall back to mean+std so we still catch gross outliers.
        std = values.std()
        if std <= 0:
            return np.zeros_like(values, dtype=bool)
        return values > med + k * std
    # 1.4826 scales MAD to be comparable to the standard deviation for normals.
    return (values - med) / (1.4826 * mad) > k


def detect_morphology_errors(mask: np.ndarray, thresholds: Thresholds) -> pd.DataFrame:
    """Flag cells whose shape is anomalous (possible fusion or leak).

    Returns a dataframe with columns: frame, track_id, area, solidity,
    eccentricity, reason. Area outliers use a robust (median/MAD) test computed
    per-frame; solidity/eccentricity use the fixed thresholds in ``thresholds``.
    """
    rows = []
    if mask is None:
        return pd.DataFrame(columns=["frame", "track_id", "area", "solidity",
                                     "eccentricity", "reason"])
    props = ("label", "area", "solidity", "eccentricity")
    for f in range(mask.shape[0]):
        plane = mask[f]
        if not plane.any():
            continue
        table = regionprops_table(plane, properties=props)
        if len(table["label"]) == 0:
            continue
        areas = np.asarray(table["area"], dtype=float)
        big = _robust_outliers(areas, thresholds.area_mad_k)
        for i, lab in enumerate(table["label"]):
            reasons = []
            if big[i]:
                reasons.append("area>>median (possible fusion)")
            sol = float(table["solidity"][i])
            ecc = float(table["eccentricity"][i])
            if sol < thresholds.min_solidity:
                reasons.append(f"low solidity {sol:.2f}")
            if ecc > thresholds.max_eccentricity:
                reasons.append(f"high eccentricity {ecc:.2f}")
            if reasons:
                rows.append({"frame": int(f), "track_id": int(lab),
                             "area": float(areas[i]), "solidity": sol,
                             "eccentricity": ecc, "reason": "; ".join(reasons)})
    return pd.DataFrame(rows, columns=["frame", "track_id", "area", "solidity",
                                       "eccentricity", "reason"])


# ---------------------------------------------------------------------------
# Identity-swap detection (label exchange between two co-existing tracks)
# ---------------------------------------------------------------------------
def _area_lookup_from_df(df):
    """Return {(frame, track_id): area} from COL_AREA if present, else {}."""
    if df is None or config.COL_AREA not in df.columns:
        return {}
    sub = df.dropna(subset=[COL_FRAME, COL_TRACK, config.COL_AREA])
    if sub.empty:
        return {}
    keys = zip(sub[COL_FRAME].astype(int).to_numpy().tolist(),
               sub[COL_TRACK].astype(int).to_numpy().tolist())
    return dict(zip(keys, sub[config.COL_AREA].astype(float).to_numpy().tolist()))


def detect_identity_swaps(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    """Flag frame transitions where two co-existing tracks likely swapped labels.

    A clean label swap leaves no jump, no gap and conserves the cell count, so
    the per-track checks miss it. Its signature is two nearby tracks A and B
    whose frame f -> f+1 assignment is cheaper when exchanged: A lands where B
    was and B lands where A was. For each frame pair the nearest neighbour of
    every track is tested as a swap partner; a pair is flagged when the swapped
    assignment beats the kept assignment by ``swap_cost_margin`` (and at least
    ``swap_min_improvement_px``) and the two were within ``swap_search_radius_px``.

    When per-frame areas are available (COL_AREA) the result also reports whether
    the swap is corroborated by area continuity (A inherits B's area and vice
    versa), which separates real label swaps from cells that merely cross paths.

    Returns columns: frame, track_a, track_b, separation, cost_improvement,
    area_corroborates. Sorted with corroborated, larger-improvement swaps first.
    """
    cols = ["frame", "track_a", "track_b", "separation",
            "cost_improvement", "area_corroborates"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[(d[COL_TRACK] > 0) & (~d[COL_TRACK].map(config.is_quarantined_id))]
    if d.empty:
        return pd.DataFrame(columns=cols)

    pos = {}
    for f, g in d.groupby(d[COL_FRAME].astype(int)):
        ids = g[COL_TRACK].to_numpy(dtype=int)
        xy = np.column_stack([g[COL_X].to_numpy(float), g[COL_Y].to_numpy(float)])
        pos[int(f)] = (ids, xy, {int(t): k for k, t in enumerate(ids)})

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    area_at = _area_lookup_from_df(d)
    R = float(thresholds.swap_search_radius_px)
    margin = float(thresholds.swap_cost_margin)
    min_impr = float(thresholds.swap_min_improvement_px)

    rows = []
    frames = sorted(pos)
    for f in frames:
        nxt = f + 1
        if nxt not in pos:
            continue
        ids0, xy0, _ = pos[f]
        _, _, idx1 = pos[nxt]
        common_mask = np.array([t in idx1 for t in ids0])
        if common_mask.sum() < 2:
            continue
        ids = ids0[common_mask]
        P0 = xy0[common_mask]
        P1 = pos[nxt][1][np.array([idx1[int(t)] for t in ids])]

        if cKDTree is not None:
            tree = cKDTree(P0)
            dists, nbrs = tree.query(P0, k=2)
            partner = nbrs[:, 1]
            sep = dists[:, 1]
        else:
            diff = P0[:, None, :] - P0[None, :, :]
            dmat = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dmat, np.inf)
            partner = dmat.argmin(axis=1)
            sep = dmat[np.arange(len(P0)), partner]

        seen = set()
        for i in range(len(ids)):
            j = int(partner[i])
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            if sep[i] > R:
                continue
            d_ii = float(np.hypot(*(P1[i] - P0[i])))
            d_jj = float(np.hypot(*(P1[j] - P0[j])))
            d_ij = float(np.hypot(*(P1[i] - P0[j])))
            d_ji = float(np.hypot(*(P1[j] - P0[i])))
            kept = d_ii + d_jj
            swapped = d_ij + d_ji
            improvement = kept - swapped
            if not (d_ij < d_ii and d_ji < d_jj):
                continue
            if improvement < max(margin * kept, min_impr):
                continue

            a, b = int(ids[i]), int(ids[j])
            corroborates = False
            if area_at:
                aa0, ab0 = area_at.get((f, a)), area_at.get((f, b))
                aa1, ab1 = area_at.get((nxt, a)), area_at.get((nxt, b))
                if None not in (aa0, ab0, aa1, ab1):
                    keep_da = abs(aa1 - aa0) + abs(ab1 - ab0)
                    swap_da = abs(aa1 - ab0) + abs(ab1 - aa0)
                    corroborates = swap_da < keep_da
            rows.append({
                "frame": int(f), "track_a": a, "track_b": b,
                "separation": round(float(sep[i]), 2),
                "cost_improvement": round(float(improvement), 2),
                "area_corroborates": bool(corroborates),
            })

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(
            ["area_corroborates", "cost_improvement"],
            ascending=[False, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Per-track summary (used by the statistics panel)
# ---------------------------------------------------------------------------
def compute_track_summary(df, mask, pixel_size=1.0, frame_interval=1.0):
    """One row per track: lifetime, motility, area and outcome metrics."""
    valid = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid[COL_TRACK] = valid[COL_TRACK].astype(int)
    valid = valid[valid[COL_TRACK] > 0]
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values([COL_TRACK, COL_FRAME])

    area_lookup = {}
    if mask is not None:
        px_area = pixel_size ** 2
        for f in range(mask.shape[0]):
            for idv, ar in areas_for_frame(mask[f], px_area).items():
                area_lookup[(f, idv)] = ar

    parents = set()
    if COL_PARENT in df:
        parents = set(df.loc[df[COL_PARENT] > 0, COL_PARENT].dropna().astype(int).unique())

    records = []
    for tid, g in valid.groupby(COL_TRACK):
        frames = g[COL_FRAME].to_numpy(dtype=float)
        xs = g[COL_X].to_numpy(dtype=float) * pixel_size
        ys = g[COL_Y].to_numpy(dtype=float) * pixel_size
        first, last, n = frames.min(), frames.max(), len(frames)
        lifetime = (last - first + 1) * frame_interval

        if n > 1:
            steps = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            total = float(steps.sum())
            net = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
            span = (last - first) * frame_interval
            mean_speed = total / span if span > 0 else 0.0
            directionality = net / total if total > 0 else 0.0
        else:
            total = net = mean_speed = directionality = 0.0

        # Richer motility metrics (diffusion, persistence, turning angles).
        motil = motility_metrics(frames, xs, ys, frame_interval)

        treat_mode = g[COL_TREATMENT].mode() if COL_TREATMENT in g else pd.Series([], dtype=object)
        treatment = treat_mode.iloc[0] if not treat_mode.empty else TREAT_CONTROL

        outs = [o for o in g[COL_OUTCOME].astype(str).unique()
                if o not in ("", "nan", "None")] if COL_OUTCOME in g else []
        final_outcome = outs[0] if outs else "none"
        is_mitotic = (final_outcome == OUTCOME_MITOSIS) or (int(tid) in parents)

        if area_lookup:
            areas = np.array([area_lookup.get((int(fr), int(tid)), np.nan)
                              for fr in g[COL_FRAME].to_numpy()], dtype=float)
            areas = areas[~np.isnan(areas)]
            mean_area = float(areas.mean()) if areas.size else np.nan
            max_area = float(areas.max()) if areas.size else np.nan
        else:
            mean_area = max_area = np.nan

        rec = {
            "track_id": int(tid), "first_frame": int(first), "last_frame": int(last),
            "n_frames": int(n), "lifetime": lifetime, COL_TREATMENT: treatment,
            "is_mitotic": bool(is_mitotic), "final_outcome": final_outcome,
            "total_distance": total, "net_displacement": net,
            "mean_speed": mean_speed, "directionality": directionality,
            "mean_area": mean_area, "max_area": max_area,
        }
        rec.update(motil)
        records.append(rec)
    summary = pd.DataFrame(records)

    # Mean nuclear morphometry (NII and its components) per track, from the mask.
    if mask is not None and not summary.empty:
        nm = nuclear_morphometry(mask)
        if not nm.empty:
            means = (nm.groupby("track_id")[MORPHOMETRY_COLS].mean()
                     .rename(columns={c: f"mean_{c}" for c in MORPHOMETRY_COLS}))
            summary = summary.merge(means, on="track_id", how="left")
    return summary


def motility_metrics(frames, xs, ys, frame_interval=1.0):
    """Per-track motility descriptors beyond speed/displacement.

    Inputs are 1-D arrays for a single track: frame indices and already
    physically-scaled x/y coordinates (e.g. micrometres). Returns a dict with:

      diffusion_coeff    : D from a linear fit MSD = 4*D*t over short lags
                           (2-D Brownian convention MSD=4Dt); NaN if too short.
      persistence_time   : characteristic time over which direction is retained,
                           from the exponential decay of the velocity
                           autocorrelation; NaN if not estimable.
      mean_turning_angle : mean absolute angle (radians) between consecutive
                           step vectors; small = straight, ~pi/2 = random.
      confinement_ratio  : net / total displacement (same as directionality;
                           repeated here so the motility block is self-contained).
      anomalous_exponent : slope alpha of a log-log fit of MSD vs lag (1 = normal
                           diffusion, <1 subdiffusive, >1 superdiffusive); NaN if
                           not estimable.

    All metrics are defined to degrade gracefully (NaN) for short tracks rather
    than raise, so they are safe to compute for every track.
    """
    out = {"diffusion_coeff": np.nan, "persistence_time": np.nan,
           "mean_turning_angle": np.nan, "confinement_ratio": np.nan,
           "anomalous_exponent": np.nan}
    n = len(xs)
    if n < 2:
        return out
    dx = np.diff(xs)
    dy = np.diff(ys)
    step = np.sqrt(dx ** 2 + dy ** 2)
    total = float(step.sum())
    net = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
    out["confinement_ratio"] = (net / total) if total > 0 else 0.0

    # Turning angles between consecutive step vectors.
    if n >= 3:
        v1 = np.stack([dx[:-1], dy[:-1]], axis=1)
        v2 = np.stack([dx[1:], dy[1:]], axis=1)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        ok = (n1 > 0) & (n2 > 0)
        if ok.any():
            cos = np.einsum("ij,ij->i", v1[ok], v2[ok]) / (n1[ok] * n2[ok])
            cos = np.clip(cos, -1.0, 1.0)
            out["mean_turning_angle"] = float(np.mean(np.abs(np.arccos(cos))))

    # Diffusion coefficient from a short-lag MSD fit (MSD = 4 D t in 2-D).
    max_lag = min(10, n - 1)
    if max_lag >= 2:
        lags, msd = [], []
        for lag in range(1, max_lag + 1):
            ddx = xs[lag:] - xs[:-lag]
            ddy = ys[lag:] - ys[:-lag]
            msd.append(float(np.mean(ddx ** 2 + ddy ** 2)))
            lags.append(lag * frame_interval)
        lags = np.asarray(lags)
        msd = np.asarray(msd)
        if np.ptp(lags) > 0:
            slope = np.polyfit(lags, msd, 1)[0]
            out["diffusion_coeff"] = float(slope / 4.0)
            pos = (lags > 0) & (msd > 0)
            if pos.sum() >= 2:
                out["anomalous_exponent"] = float(
                    np.polyfit(np.log(lags[pos]), np.log(msd[pos]), 1)[0])

    # Persistence time from velocity autocorrelation decay.
    if n >= 4 and frame_interval > 0:
        vx = dx / frame_interval
        vy = dy / frame_interval
        speeds = np.sqrt(vx ** 2 + vy ** 2)
        ok = speeds > 0
        if ok.sum() >= 3:
            ux = vx[ok] / speeds[ok]
            uy = vy[ok] / speeds[ok]
            m = len(ux)
            max_dt = min(10, m - 1)
            ac = []
            for dt in range(0, max_dt + 1):
                if dt == 0:
                    ac.append(1.0)
                else:
                    c = np.mean(ux[:-dt] * ux[dt:] + uy[:-dt] * uy[dt:])
                    ac.append(float(c))
            ac = np.asarray(ac)
            # Time of first crossing below 1/e gives a robust persistence time.
            thresh = 1.0 / np.e
            below = np.where(ac < thresh)[0]
            if below.size:
                out["persistence_time"] = float(below[0] * frame_interval)
    return out


# ---------------------------------------------------------------------------
# Border-contact detection (cells touching the frame edge)
# ---------------------------------------------------------------------------
def labels_touching_border(mask_plane, margin: int = 0):
    """Return the set of label IDs whose mask touches the frame border.

    ``margin`` widens the border band: margin=0 means only the outermost row/
    column count; margin=2 means a label within 2 px of any edge is flagged.
    Vectorized: reads only the border bands, not the whole plane.
    """
    if mask_plane is None or mask_plane.size == 0:
        return set()
    m = max(0, int(margin))
    h, w = mask_plane.shape[-2:]
    top = mask_plane[: m + 1, :]
    bottom = mask_plane[h - m - 1 :, :]
    left = mask_plane[:, : m + 1]
    right = mask_plane[:, w - m - 1 :]
    ids = set()
    for band in (top, bottom, left, right):
        u = np.unique(band)
        ids.update(int(v) for v in u if v != 0)
    return ids


def annotate_border_contact(df, mask, margin: int = 0):
    """Set df[COL_BORDER]=True for each (frame, track) whose mask touches the edge.

    Operates in place on a copy and returns it. Rows with no matching mask label
    are left False. This is the detection step; how the flag is *used* in
    statistics is a separate, explicit choice.
    """
    from .config import COL_BORDER, COL_TRACK, COL_FRAME
    out = df.copy()
    out[COL_BORDER] = False
    if mask is None or out.empty:
        return out
    # Build a lookup of border IDs per frame, then mark matching rows.
    for f in range(mask.shape[0]):
        border_ids = labels_touching_border(mask[f], margin=margin)
        if not border_ids:
            continue
        sel = (out[COL_FRAME] == f) & (out[COL_TRACK].isin(border_ids))
        out.loc[sel, COL_BORDER] = True
    return out


def annotate_area(df, mask):
    """Set df[COL_AREA] to the mask pixel-count for each (frame, track) row.

    Areas are stored in raw pixels (not micrometres) so the table stays
    independent of the current pixel-size calibration; statistics scale it when
    needed. Rows whose label is absent from the mask in that frame get NaN.
    Operates on a copy and returns it.
    """
    from .config import COL_AREA, COL_TRACK, COL_FRAME
    out = df.copy()
    if mask is None or out.empty:
        out[COL_AREA] = np.nan
        return out
    # Per-frame {label: pixel_area}, then map onto rows.
    areas = np.full(len(out), np.nan, dtype=float)
    frame_vals = out[COL_FRAME].to_numpy()
    track_vals = out[COL_TRACK].to_numpy()
    for f in range(mask.shape[0]):
        lut = areas_for_frame(mask[f], pixel_area=1.0)  # raw pixels
        if not lut:
            continue
        sel = np.where(frame_vals == f)[0]
        for i in sel:
            tid = track_vals[i]
            if not np.isnan(tid):
                areas[i] = lut.get(int(tid), np.nan)
    out[COL_AREA] = areas
    return out


def auto_flag_border_exits(df, mask=None, tail=5, min_tail=3,
                           shrink_ratio=0.6, margin=0):
    """Auto-flag tracks that leave the field as EXIT (in place on a copy).

    A track is flagged ``OUTCOME_EXIT`` when ALL of the following hold and it has
    no outcome yet (so Mitosis and any CSV/manual outcome are never overwritten):

      1. Border contact at the end: the track touches the frame border in its
         final frame (or within its last ``min_tail`` frames).
      2. Shrinking to vanishing: the mask area declines over the track's tail --
         the median area of the final ``min_tail`` frames is at most
         ``shrink_ratio`` times the peak area of the preceding tail window, i.e.
         the cell got substantially smaller before disappearing.
      3. It actually disappears mid-movie: the track's last frame is before the
         last movie frame (it stops being tracked rather than running to the
         end of the acquisition).

    This matches the visual pattern "cell at the edge gets smaller and smaller
    until it's gone" -- a cell migrating out of the field of view.

    Parameters
    ----------
    df : per-frame table; must already have COL_AREA and COL_BORDER populated
        (load_data calls annotate_area / annotate_border_contact first).
    mask : optional; used to recover the movie length (last frame =
        ``mask.shape[0] - 1``). Without a mask the last frame seen in ``df`` is
        used as the movie end.
    tail : how many trailing frames to inspect for the area trend.
    min_tail : minimum number of trailing frames required to judge (a track
        shorter than this is left alone -- too little evidence).
    shrink_ratio : the tail-end median area must be <= this fraction of the
        earlier tail peak to count as "shrinking to vanishing" (0.6 = shrank to
        60% or less).
    margin : border band width passed through to the border test semantics.

    Returns a new DataFrame; also returns nothing extra -- count is logged by
    the caller via the number of changed tracks if desired.
    """
    from .config import (COL_AREA, COL_BORDER, COL_TRACK, COL_FRAME, COL_OUTCOME,
                         OUTCOME_EXIT)
    out = df.copy()
    if out.empty or COL_AREA not in out.columns or COL_BORDER not in out.columns:
        return out

    last_movie_frame = (int(mask.shape[0]) - 1) if mask is not None \
        else int(out[COL_FRAME].dropna().max())

    # Resolve current per-track outcome (a track's first non-empty outcome).
    def _has_outcome(series):
        for v in series:
            s = str(v)
            if s and s not in ("nan", "None"):
                return True
        return False

    flagged = []
    for tid, g in out.dropna(subset=[COL_FRAME]).groupby(COL_TRACK):
        if int(tid) <= 0:
            continue
        g = g.sort_values(COL_FRAME)
        # 0. Skip tracks that already carry any outcome (incl. auto-Mitosis).
        if _has_outcome(g[COL_OUTCOME]):
            continue

        last_frame = int(g[COL_FRAME].iloc[-1])
        # 3. Must vanish before the movie ends.
        if last_frame >= last_movie_frame:
            continue

        areas = g[COL_AREA].to_numpy(dtype=float)
        border = g[COL_BORDER].to_numpy()
        if len(areas) < min_tail:
            continue

        # 1. Border contact at the very end (any of the last min_tail frames).
        if not np.any(border[-min_tail:].astype(bool)):
            continue

        # 2. Shrinking-to-vanishing over the tail: compare the end vs the peak
        #    of the trailing window. Use nan-safe stats.
        tail_window = areas[-int(tail):] if len(areas) >= tail else areas
        end_area = np.nanmedian(areas[-int(min_tail):])
        peak_area = np.nanmax(tail_window)
        if not np.isfinite(end_area) or not np.isfinite(peak_area) or peak_area <= 0:
            continue
        if end_area <= shrink_ratio * peak_area:
            flagged.append(int(tid))

    if flagged:
        sel = out[COL_TRACK].astype("Int64").isin(flagged)
        empty = out[COL_OUTCOME].astype(str).isin(["", "nan", "None"])
        out.loc[sel & empty, COL_OUTCOME] = OUTCOME_EXIT
    return out


# Nuclear morphometry (NII and its components, after Filippi-Chiela et al. 2012)
# ---------------------------------------------------------------------------
NMA_COLS = ["aspect", "area_box", "radius_ratio", "roundness", "nii"]
_SHAPE_COLS = ["eccentricity", "solidity", "extent", "orientation",
               "axis_major_length", "axis_minor_length"]
MORPHOMETRY_COLS = ["area_px", "perimeter", "circularity"] + NMA_COLS + _SHAPE_COLS


def _radius_ratio(region):
    """Max/min distance from the region centroid to its boundary pixels.

    1.0 for a perfect disc, growing as the outline departs from a circle. Uses
    the local crop so it is independent of absolute position.
    """
    from scipy import ndimage as ndi
    img = region.image
    if img.sum() < 4:
        return np.nan
    boundary = img & ~ndi.binary_erosion(img)
    ys, xs = np.nonzero(boundary)
    if ys.size < 2:
        return np.nan
    try:
        cy, cx = region.centroid_local
    except AttributeError:
        cy, cx = region.local_centroid
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    d = d[d > 0]
    if d.size < 2:
        return np.nan
    rmin = float(d.min())
    return float(d.max() / rmin) if rmin > 0 else np.nan


def nuclear_morphometry(mask):
    """Per (frame, track_id) nuclear morphometry including the NII.

    Computes, from the segmentation mask, the four shape descriptors the NMA
    method combines (Image-Pro Plus conventions):

      aspect       = major axis / minor axis            (>= 1, 1 = circle)
      area_box     = object area / bounding-box area     (<= 1)
      radius_ratio = max/min centroid-to-boundary radius (>= 1, 1 = circle)
      roundness    = perimeter^2 / (4*pi*area)           (>= 1, 1 = circle)

    and the Nuclear Irregularity Index

      NII = aspect - area_box + radius_ratio + roundness

    Also returns area_px, perimeter and circularity (= 1/roundness) so the table
    is a superset of morphology_timeseries(), plus the standard regionprops shape
    descriptors (eccentricity, solidity, extent, orientation, axis_major_length,
    axis_minor_length). Returns columns: frame, track_id, area_px, perimeter,
    circularity, aspect, area_box, radius_ratio, roundness, nii, eccentricity,
    solidity, extent, orientation, axis_major_length, axis_minor_length.
    """
    from skimage.measure import regionprops
    cols = ["frame", "track_id"] + MORPHOMETRY_COLS
    if mask is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for f in range(mask.shape[0]):
        plane = mask[f]
        if not plane.any():
            continue
        for r in regionprops(plane):
            area = float(r.area)
            perim = float(r.perimeter)
            minr, minc, maxr, maxc = r.bbox
            box = float((maxr - minr) * (maxc - minc))
            major = float(r.axis_major_length)
            minor = float(r.axis_minor_length)
            with np.errstate(divide="ignore", invalid="ignore"):
                circ = (4.0 * np.pi * area / perim ** 2) if perim > 0 else np.nan
                aspect = (major / minor) if minor > 0 else np.nan
                area_box = (area / box) if box > 0 else np.nan
                roundness = (perim ** 2 / (4.0 * np.pi * area)) if area > 0 else np.nan
            rr = _radius_ratio(r)
            nii = (aspect - area_box + rr + roundness
                   if np.all(np.isfinite([aspect, area_box, rr, roundness]))
                   else np.nan)
            rows.append({
                "frame": int(f), "track_id": int(r.label),
                "area_px": area, "perimeter": perim, "circularity": float(circ),
                "aspect": float(aspect), "area_box": float(area_box),
                "radius_ratio": float(rr), "roundness": float(roundness),
                "nii": float(nii),
                "eccentricity": float(r.eccentricity),
                "solidity": float(r.solidity),
                "extent": float(r.extent),
                "orientation": float(r.orientation),
                "axis_major_length": float(r.axis_major_length),
                "axis_minor_length": float(r.axis_minor_length)})
    return pd.DataFrame(rows, columns=cols)


def morphology_timeseries(mask):
    """Per (frame, label) shape descriptors read straight from the mask.

    Returns a DataFrame with columns: frame, track_id, area_px, perimeter,
    circularity. Circularity is the isoperimetric ratio 4*pi*area / perimeter^2
    (1 = perfect circle, ->0 = elongated/line). Computed via regionprops, so it
    matches the morphology used elsewhere. Empty frames are skipped; a label
    with zero perimeter (single pixel) gets circularity NaN.

    This is the source for trajectory plots of perimeter/circularity, which are
    not stored in the per-frame table (only area is). Intended to be merged onto
    the per-frame table on demand rather than persisted.
    """
    cols = ["frame", "track_id", "area_px", "perimeter", "circularity"]
    if mask is None:
        return pd.DataFrame(columns=cols)
    rows = []
    props = ("label", "area", "perimeter")
    for f in range(mask.shape[0]):
        plane = mask[f]
        if not plane.any():
            continue
        table = regionprops_table(plane, properties=props)
        labels = table["label"]
        areas = np.asarray(table["area"], dtype=float)
        perims = np.asarray(table["perimeter"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            circ = np.where(perims > 0, 4.0 * np.pi * areas / (perims ** 2), np.nan)
        for i, lab in enumerate(labels):
            rows.append({"frame": int(f), "track_id": int(lab),
                         "area_px": float(areas[i]), "perimeter": float(perims[i]),
                         "circularity": float(circ[i])})
    return pd.DataFrame(rows, columns=cols)


def exclude_border_rows(df):
    """Return df without rows flagged as touching the frame border.

    Safe to call when the border column is absent (returns df unchanged). This
    is the single place statistics use to honor the 'exclude border points'
    option, so every plot stays consistent.
    """
    from .config import COL_BORDER
    if df is None or COL_BORDER not in df.columns:
        return df
    return df[~df[COL_BORDER].fillna(False).astype(bool)]


# ---------------------------------------------------------------------------
# Curation progress
# ---------------------------------------------------------------------------
def curation_progress(df):
    """Return a dict summarizing how much of the dataset is curated.

    A track counts as 'curated' if it has a final outcome flag OR participates
    in a lineage (is a mother or a daughter). Returns total/curated/pending
    counts and a percentage, plus a short human-readable string.
    """
    from .config import (COL_TRACK, COL_OUTCOME, COL_PARENT, NON_FLAG_VALUES)
    out = {"total": 0, "curated": 0, "pending": 0, "pct": 0.0, "text": "0 / 0 tracks curated"}
    if df is None or df.empty or COL_TRACK not in df.columns:
        return out

    tracks = set(int(t) for t in df[COL_TRACK].dropna().unique() if t > 0)
    total = len(tracks)
    if total == 0:
        return out

    flagged = set()
    if COL_OUTCOME in df.columns:
        mask = ~df[COL_OUTCOME].astype(str).isin(NON_FLAG_VALUES)
        flagged = set(int(t) for t in df.loc[mask, COL_TRACK].dropna().unique() if t > 0)

    in_lineage = set()
    if COL_PARENT in df.columns:
        mothers = set(int(p) for p in df.loc[df[COL_PARENT] > 0, COL_PARENT].dropna().unique())
        daughters = set(int(t) for t in df.loc[df[COL_PARENT] > 0, COL_TRACK].dropna().unique())
        in_lineage = mothers | daughters

    curated = (flagged | in_lineage) & tracks
    n_cur = len(curated)
    pct = 100.0 * n_cur / total
    out.update(total=total, curated=n_cur, pending=total - n_cur, pct=pct,
               text=f"{n_cur} / {total} tracks curated ({pct:.0f}%)")
    return out


# ---------------------------------------------------------------------------
# Mass-conservation check between consecutive frames
# ---------------------------------------------------------------------------
@dataclass
class MassConservationReport:
    """Frame-to-frame cell-count balance: observed vs. biologically expected.

    ``rows`` is a list of dicts (one per flagged frame transition f -> f+1),
    each with: frame (= f, the LATER frame f+1 is frame+1), n_prev, n_curr,
    observed_delta (= n_curr - n_prev), expected_delta, residual
    (= observed - expected), and the IDs that appeared / disappeared at that
    transition, split into explained and unexplained sets. ``threshold`` records
    the residual magnitude above which a transition was flagged.

    A nonzero residual means the change in cell number is NOT accounted for by
    divisions (+1 each), field exits / deaths (-1 each) or legitimate entries
    from the border (+1 each) — the classic signature of a mask fusion
    (unexplained -1) or a spurious split / appearing object (unexplained +1).
    """
    rows: list = field(default_factory=list)
    threshold: int = 1

    def is_empty(self) -> bool:
        return not self.rows

    def messages(self) -> list:
        msgs = []
        for r in self.rows:
            kind = "split/appearance" if r["residual"] > 0 else "fusion/disappearance"
            extra = []
            if r["unexplained_appeared"]:
                extra.append(f"appeared {sorted(r['unexplained_appeared'])}")
            if r["unexplained_disappeared"]:
                extra.append(f"vanished {sorted(r['unexplained_disappeared'])}")
            detail = ("; " + ", ".join(extra)) if extra else ""
            msgs.append(
                f"frames {r['frame']}->{r['frame']+1}: count {r['n_prev']}->"
                f"{r['n_curr']} (obs {r['observed_delta']:+d}, exp "
                f"{r['expected_delta']:+d}, residual {r['residual']:+d}, "
                f"likely {kind}{detail})")
        return msgs


def check_mass_conservation(df, mask=None, thresholds=None, border_margin=0,
                            residual_threshold=1):
    """Validate that the cell count between consecutive frames is explainable.

    For every transition f -> f+1 the *observed* change in the number of
    distinct cells is compared against the change *expected* from known causes:

      +1  for each division whose daughter first appears at f+1
      -1  for each track whose last frame is f and is flagged Exit or
          Death/Senescence (a legitimate field exit / death)
      +1  for each track that first appears at f+1 while touching the border
          (a legitimate new cell entering the field of view)

    The residual ``observed - expected`` should be 0 when the segmentation and
    tracking are consistent. ``|residual| >= residual_threshold`` is flagged:
    a positive residual suggests a spurious split or an object appearing in the
    middle of the field (often an over-segmentation / mask split); a negative
    residual suggests a fusion or a cell vanishing mid-field (often an
    under-segmentation / mask merge or a dropped detection).

    Border entries are only credited when a ``mask`` is supplied (so we can tell
    whether the new cell touches the edge). Without a mask, every mid-field
    appearance counts toward the residual, which is the conservative choice.

    Parameters
    ----------
    df : per-frame table (must have track_id, frame; outcome/parent optional).
    mask : optional TxHxW label volume, used to credit legitimate border
        entries (and to validate counts against the mask if df is sparse).
    thresholds : optional Thresholds; only ``gap_tolerance`` is consulted, to
        avoid double-counting a track that merely has a 1-frame gap.
    border_margin : int, widens the border band for entry detection.
    residual_threshold : int, minimum |residual| to flag (default 1).

    Returns a :class:`MassConservationReport`. Quarantined IDs are ignored.
    """
    rep = MassConservationReport(threshold=int(residual_threshold))
    if df is None or df.empty:
        return rep
    d = df.dropna(subset=[COL_TRACK, COL_FRAME]).copy()
    if d.empty:
        return rep
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d[COL_FRAME] = d[COL_FRAME].astype(int)
    d = d[d[COL_TRACK] > 0]
    d = d[~d[COL_TRACK].map(config.is_quarantined_id)]
    if d.empty:
        return rep

    # Cells present per frame (from the dataframe).
    present = {f: set(g[COL_TRACK].astype(int))
               for f, g in d.groupby(COL_FRAME)}
    frames = sorted(present.keys())
    if len(frames) < 2:
        return rep

    first_frame = d.groupby(COL_TRACK)[COL_FRAME].min().astype(int).to_dict()
    last_frame = d.groupby(COL_TRACK)[COL_FRAME].max().astype(int).to_dict()

    # Division frames: daughter's first frame (parent_id > 0).
    daughters_born = {}   # frame -> set of daughter ids appearing by division
    if COL_PARENT in d.columns:
        links = d[d[COL_PARENT] > 0][[COL_TRACK]].drop_duplicates()
        for tid in links[COL_TRACK].astype(int):
            f0 = first_frame.get(int(tid))
            if f0 is not None:
                daughters_born.setdefault(f0, set()).add(int(tid))

    # Tracks ending in a field-exit / death, keyed by their last frame.
    exit_like = {OUTCOME_EXIT, OUTCOME_DEATH}
    exits_at = {}   # frame f -> set of ids whose last frame is f and exit/death
    if COL_OUTCOME in d.columns:
        final_oc = {}
        for tid, g in d.groupby(COL_TRACK):
            outs = [str(o) for o in g[COL_OUTCOME] if str(o) not in ("", "nan", "None")]
            final_oc[int(tid)] = outs[0] if outs else ""
        for tid, oc in final_oc.items():
            if oc in exit_like:
                lf = last_frame.get(int(tid))
                if lf is not None:
                    exits_at.setdefault(lf, set()).add(int(tid))

    # Border IDs per frame (legitimate entries), only if a mask is available.
    border_ids_at = {}
    if mask is not None:
        for f in range(mask.shape[0]):
            border_ids_at[f] = labels_touching_border(mask[f], margin=border_margin)

    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        # Only compare strictly consecutive movie frames; a temporal gap is a
        # different anomaly handled by detect_anomalies, so skip it here.
        if f1 != f0 + 1:
            continue
        prev_ids = present.get(f0, set())
        curr_ids = present.get(f1, set())
        n_prev, n_curr = len(prev_ids), len(curr_ids)
        observed = n_curr - n_prev

        appeared = curr_ids - prev_ids        # in f1 but not f0
        disappeared = prev_ids - curr_ids     # in f0 but not f1

        # Expected contributions.
        born = daughters_born.get(f1, set()) & appeared
        # Legitimate entries from the border (must actually be new + on border).
        border_entries = set()
        if mask is not None:
            border_entries = {tid for tid in appeared
                              if tid in border_ids_at.get(f1, set())
                              and first_frame.get(tid) == f1}
        exits = exits_at.get(f0, set()) & disappeared

        expected = len(born) + len(border_entries) - len(exits)
        residual = observed - expected

        # Unexplained movers: appearances not from division or border entry,
        # and disappearances not from a flagged exit/death.
        unexplained_appeared = appeared - born - border_entries
        unexplained_disappeared = disappeared - exits

        if abs(residual) >= int(residual_threshold) or \
           unexplained_appeared or unexplained_disappeared:
            rep.rows.append({
                "frame": int(f0),
                "n_prev": int(n_prev), "n_curr": int(n_curr),
                "observed_delta": int(observed),
                "expected_delta": int(expected),
                "residual": int(residual),
                "appeared": set(int(x) for x in appeared),
                "disappeared": set(int(x) for x in disappeared),
                "unexplained_appeared": set(int(x) for x in unexplained_appeared),
                "unexplained_disappeared": set(int(x) for x in unexplained_disappeared),
            })
    return rep
