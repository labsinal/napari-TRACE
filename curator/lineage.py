"""
Lineage / genealogy logic.

Provides:
  * a helper that classifies tracks into mothers, daughters and "flagged
    singles", used by the tree plot, re-sequencing and the
    jump-to-next-uncurated navigation;
  * a topological validator (>2 daughters, daughter born at/before the mother,
    mother surviving its own division), used by diagnostics and to block
    biologically impossible links;
  * assisted gap relinking that estimates where a vanished cell should be and
    proposes the best candidate to reconnect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT,
    NON_FLAG_VALUES,
)


# ---------------------------------------------------------------------------
# Lineage classification
# ---------------------------------------------------------------------------
@dataclass
class LineageGroups:
    mothers: set
    daughters: set
    flagged_singles: set
    children_of: dict          # mother_id -> sorted list of daughter ids

    @property
    def all_nodes(self) -> set:
        return self.mothers | self.daughters | self.flagged_singles

    @property
    def roots(self) -> set:
        """Family roots: mothers that are not themselves daughters, plus singles."""
        return (self.mothers - self.daughters) | self.flagged_singles


def classify_lineage(df: pd.DataFrame) -> LineageGroups:
    """Split tracks into mothers, daughters and flagged-but-unlinked singles."""
    if df is None or df.empty or COL_PARENT not in df:
        return LineageGroups(set(), set(), set(), {})

    links = df[df[COL_PARENT] > 0][[COL_PARENT, COL_TRACK]].drop_duplicates()
    children_of: dict[int, list[int]] = {}
    for _, row in links.iterrows():
        children_of.setdefault(int(row[COL_PARENT]), []).append(int(row[COL_TRACK]))
    for p in children_of:
        children_of[p] = sorted(set(children_of[p]))

    mothers = set(children_of.keys())
    daughters = {c for subs in children_of.values() for c in subs}

    if COL_OUTCOME in df:
        flagged = ~df[COL_OUTCOME].astype(str).isin(NON_FLAG_VALUES)
        flagged_ids = set(df[flagged][COL_TRACK].dropna().astype(int).unique())
    else:
        flagged_ids = set()
    flagged_singles = flagged_ids - (mothers | daughters)

    return LineageGroups(mothers, daughters, flagged_singles, children_of)


def hierarchical_labels(df: pd.DataFrame) -> dict:
    """Return {track_id: "1.2.1"}-style genealogy labels derived from parent_id.

    Purely a *display* translation of the lineage already stored in the table:
    family roots (ordered by first frame) are numbered 1, 2, 3 ...; each cell's
    daughters (ordered by birth frame) get the parent's label with ".1", ".2"
    appended. Nothing about the actual track_id or the mask changes -- this is
    what the tree plot shows instead of raw IDs, so there is no need to rewrite
    IDs (which used to risk colliding with the reserved ranges). Iterative, so a
    very deep lineage cannot hit the recursion limit; cycle-safe.
    """
    if df is None or df.empty or COL_PARENT not in df:
        return {}
    groups = classify_lineage(df)
    children_of = groups.children_of
    firsts = _first_frames(df)

    def _order(ids):
        return sorted((int(i) for i in ids),
                      key=lambda i: (firsts.get(i, 10**9), i))

    labels: dict[int, str] = {}
    for fam, root in enumerate(_order(groups.roots), start=1):
        stack = [(root, str(fam))]
        while stack:
            node, label = stack.pop()
            if node in labels:
                continue
            labels[node] = label
            for idx, child in enumerate(_order(children_of.get(node, [])), start=1):
                if child not in labels:
                    stack.append((child, f"{label}.{idx}"))
    return labels


# ---------------------------------------------------------------------------
# Topological validator
# ---------------------------------------------------------------------------
@dataclass
class LineageViolations:
    multi_daughter: list = field(default_factory=list)   # (mother, [daughters])
    born_before_mother: list = field(default_factory=list)  # (mother, daughter)
    mother_survives: list = field(default_factory=list)  # (mother, division_frame, last_frame)

    def is_empty(self) -> bool:
        return not (self.multi_daughter or self.born_before_mother
                    or self.mother_survives)

    def messages(self) -> list[str]:
        msgs = []
        for m, ds in self.multi_daughter:
            msgs.append(f"Mother {m} has {len(ds)} daughters {ds} (expected <= 2).")
        for m, d in self.born_before_mother:
            msgs.append(f"Daughter {d} appears at/before mother {m}.")
        for m, dfr, last in self.mother_survives:
            msgs.append(f"Mother {m} still present at frame {last} after dividing at {dfr}.")
        return msgs


def _first_frames(df: pd.DataFrame) -> dict[int, int]:
    g = df.dropna(subset=[COL_TRACK, COL_FRAME])
    return g.groupby(g[COL_TRACK].astype(int))[COL_FRAME].min().astype(int).to_dict()


def _last_frames(df: pd.DataFrame) -> dict[int, int]:
    g = df.dropna(subset=[COL_TRACK, COL_FRAME])
    return g.groupby(g[COL_TRACK].astype(int))[COL_FRAME].max().astype(int).to_dict()


def validate_lineage(df: pd.DataFrame) -> LineageViolations:
    """Detect biologically impossible lineage configurations across the table."""
    v = LineageViolations()
    if df is None or df.empty or COL_PARENT not in df:
        return v
    groups = classify_lineage(df)
    firsts = _first_frames(df)
    lasts = _last_frames(df)

    for mother, daughters in groups.children_of.items():
        if len(daughters) > 2:
            v.multi_daughter.append((int(mother), [int(d) for d in daughters]))

        division_frame = min((firsts.get(int(d), 10**9) for d in daughters), default=None)
        for d in daughters:
            df_d = firsts.get(int(d))
            mf = firsts.get(int(mother))
            if df_d is not None and mf is not None and df_d <= mf:
                v.born_before_mother.append((int(mother), int(d)))

        if division_frame is not None:
            m_last = lasts.get(int(mother))
            if m_last is not None and m_last >= division_frame:
                v.mother_survives.append((int(mother), int(division_frame), int(m_last)))
    return v


def can_link(df: pd.DataFrame, mother: int, daughter: int,
             sovereign: bool = False) -> tuple[bool, str]:
    """Pre-flight check used by the link-mother-daughter operation.

    Returns (ok, reason). Blocks self-links, missing cells, a daughter that
    already has a different mother, and a daughter born at/before the mother.

    When ``sovereign`` is True the two-daughter cap and the "daughter already
    has a different mother" guard are NOT enforced here: the caller (link_parent)
    detects those conflicts, asks the user to confirm, and then evicts/reassigns
    so a manual link always wins. The self-link, missing-cell and
    born-at/before-mother guards still apply (the last is biologically
    impossible, not an overridable conflict).
    """
    if mother == daughter:
        return False, "A cell cannot be its own mother."
    tracks = set(df[COL_TRACK].dropna().astype(int).unique())
    if mother not in tracks:
        return False, f"Mother cell {mother} does not exist."
    if daughter not in tracks:
        return False, f"Daughter cell {daughter} does not exist."

    groups = classify_lineage(df)
    existing = groups.children_of.get(int(mother), [])
    if not sovereign and daughter not in existing and len(existing) >= 2:
        return False, f"Mother {mother} already has 2 daughters {existing}."

    if not sovereign:
        cur_parent = df.loc[df[COL_TRACK] == daughter, COL_PARENT]
        cur_parent = cur_parent[cur_parent > 0]
        if not cur_parent.empty and int(cur_parent.iloc[0]) not in (mother, -1):
            return False, (f"Daughter {daughter} already has mother "
                           f"{int(cur_parent.iloc[0])}.")

    firsts = _first_frames(df)
    if daughter in firsts and mother in firsts and firsts[daughter] <= firsts[mother]:
        return False, (f"Daughter {daughter} appears at/before mother {mother} "
                       f"(frames {firsts[daughter]} <= {firsts[mother]}).")
    return True, "ok"


# ---------------------------------------------------------------------------
# Assisted gap relinking
# ---------------------------------------------------------------------------
@dataclass
class RelinkSuggestion:
    track_id: int
    gap_start: int            # last frame before the gap
    gap_end: int              # first frame after the gap
    missing_frame: int        # the frame we try to fill (gap_start + 1)
    expected_xy: tuple        # predicted (x, y) at missing_frame
    candidate_id: int         # best candidate track to merge from
    candidate_xy: tuple
    distance: float


# ---------------------------------------------------------------------------
# Trajectory prediction across a gap (linear or smoothing-spline)
# ---------------------------------------------------------------------------
# A long gap is exactly where a naive cubic spline misbehaves: forced through a
# handful of noisy flanking centroids, an *interpolating* cubic overshoots and
# can place the predicted centroid well outside the cell's real range (Runge
# oscillation), giving a WORSE motility estimate than the linear baseline it is
# meant to improve on. So the spline path here is deliberately conservative:
#   * it fits a SMOOTHING spline (s > 0), not an interpolating one, so the curve
#     follows the trend without chasing every jitter;
#   * it uses points on BOTH sides of the gap when available, so the fill is
#     anchored, not extrapolated into the void;
#   * it falls back to straight linear interpolation when there are too few
#     flanking points, when the gap is long relative to the surrounding data, or
#     when scipy is unavailable; and
#   * every predicted point is clamped to the bounding box of the flanking
#     control points (plus a small margin), so a fit can never fling a centroid
#     off-screen.
# The window length (frames on each side) is the ``flank`` argument; the spec
# asked for the last/next ~5 frames, which is the default.

SPLINE_FLANK = 5            # frames of history used on each side of a gap
SPLINE_SMOOTH_PER_PT = 2.0  # smoothing factor s = this * n_control_points
SPLINE_MAX_GAP_FOR_SPLINE = 12   # gaps longer than this fall back to linear
SPLINE_CLAMP_MARGIN_PX = 25.0    # predicted points may exceed the control bbox
                                 # by at most this many px


def _fit_axis_spline(frames, values, query_frames, smooth):
    """Smoothing-spline fit of one coordinate vs frame, evaluated at queries.

    Returns the predicted values, or ``None`` to signal the caller should fall
    back to linear (too few points, or scipy missing). Never raises.
    """
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:
        return None
    n = len(frames)
    if n < 4:
        # A cubic smoothing spline needs k+1 = 4 points; below that, linear.
        return None
    # Degree is capped at 3 but reduced when there are few points.
    k = min(3, n - 1)
    try:
        spl = UnivariateSpline(frames, values, k=k, s=float(smooth))
        return spl(query_frames)
    except Exception:
        return None


def predict_path(frames, xs, ys, query_frames, method="linear",
                 flank=SPLINE_FLANK):
    """Predict (x, y) at ``query_frames`` from a track's known points.

    ``method`` is "linear" (piecewise-linear interpolation/extrapolation, the
    original behaviour) or "spline" (clamped smoothing spline with a guaranteed
    linear fallback). ``frames``/``xs``/``ys`` are the track's known samples
    (need not be contiguous). Only the ``flank`` points nearest each side of the
    queried span are used for the spline, so far-away history does not distort a
    local fill. Returns ``(pred_x, pred_y)`` as float arrays the same length as
    ``query_frames``.
    """
    frames = np.asarray(frames, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    q = np.asarray(query_frames, dtype=float)
    order = np.argsort(frames)
    frames, xs, ys = frames[order], xs[order], ys[order]

    # Linear baseline (also the fallback). np.interp holds the endpoints flat
    # beyond the data, which for a bounded gap means honest interpolation.
    lin_x = np.interp(q, frames, xs)
    lin_y = np.interp(q, frames, ys)
    if method != "spline":
        return lin_x, lin_y

    gap_len = float(q.max() - q.min()) + 1.0 if len(q) else 0.0
    if gap_len > SPLINE_MAX_GAP_FOR_SPLINE:
        return lin_x, lin_y

    # Restrict to the flanking window on each side of the queried span.
    lo, hi = q.min(), q.max()
    before = np.flatnonzero(frames < lo)
    after = np.flatnonzero(frames > hi)
    keep = np.concatenate([before[-flank:], after[:flank]]).astype(int)
    if keep.size < 4:
        return lin_x, lin_y
    fk, xk, yk = frames[keep], xs[keep], ys[keep]
    smooth = SPLINE_SMOOTH_PER_PT * len(fk)

    px = _fit_axis_spline(fk, xk, q, smooth)
    py = _fit_axis_spline(fk, yk, q, smooth)
    if px is None or py is None:
        return lin_x, lin_y

    # Clamp to the flanking bounding box (+ margin) so the fit can never throw a
    # centroid off into nowhere; outside the box we trust the linear value more.
    m = SPLINE_CLAMP_MARGIN_PX
    xlo, xhi = xk.min() - m, xk.max() + m
    ylo, yhi = yk.min() - m, yk.max() + m
    bad = (px < xlo) | (px > xhi) | (py < ylo) | (py > yhi)
    px = np.where(bad, lin_x, px)
    py = np.where(bad, lin_y, py)
    return px, py


def suggest_relinks(df: pd.DataFrame, thresholds: config.Thresholds,
                    method: str = "linear") -> list[RelinkSuggestion]:
    """For each track with a temporal gap, propose a reconnection candidate.

    The expected position in the first missing frame is predicted from the
    track's known points and the best candidate is the nearest *other* track
    present in that frame within ``relink_search_radius_px``. ``method`` selects
    the predictor: "linear" (last two points, the original behaviour) or
    "spline" (a clamped smoothing spline over the flanking window, which follows
    a curved trajectory better; see :func:`predict_path`). Only the predicted
    position changes; candidate selection and approval are unchanged. The caller
    decides whether to approve each suggestion.
    """
    suggestions: list[RelinkSuggestion] = []
    if df is None or df.empty:
        return suggestions
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return suggestions
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])

    # Index of (frame) -> list of (track_id, x, y) for candidate lookup.
    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for _, r in d.iterrows():
        by_frame.setdefault(int(r[COL_FRAME]), []).append(
            (int(r[COL_TRACK]), float(r[COL_X]), float(r[COL_Y])))

    R = thresholds.relink_search_radius_px
    for tid, g in d.groupby(COL_TRACK):
        if config.is_quarantined_id(tid):
            continue
        frames = g[COL_FRAME].to_numpy(dtype=int)
        xs = g[COL_X].to_numpy(dtype=float)
        ys = g[COL_Y].to_numpy(dtype=float)
        diffs = np.diff(frames)
        gap_idx = np.flatnonzero(diffs > thresholds.gap_tolerance)
        for gi in gap_idx:
            gap_start = int(frames[gi])
            gap_end = int(frames[gi + 1])
            missing = gap_start + 1
            if method == "spline":
                ex_arr, ey_arr = predict_path(frames, xs, ys, [missing],
                                              method="spline")
                ex, ey = float(ex_arr[0]), float(ey_arr[0])
            else:
                # Linear extrapolation from the last two known points before the gap.
                if gi >= 1:
                    vx = xs[gi] - xs[gi - 1]
                    vy = ys[gi] - ys[gi - 1]
                else:
                    vx = vy = 0.0
                ex, ey = xs[gi] + vx, ys[gi] + vy

            best = None
            for cand_id, cx, cy in by_frame.get(missing, []):
                if cand_id == tid:
                    continue
                dist = float(np.hypot(cx - ex, cy - ey))
                if dist <= R and (best is None or dist < best[3]):
                    best = (cand_id, cx, cy, dist)
            if best is not None:
                suggestions.append(RelinkSuggestion(
                    track_id=int(tid), gap_start=gap_start, gap_end=gap_end,
                    missing_frame=missing, expected_xy=(ex, ey),
                    candidate_id=int(best[0]), candidate_xy=(best[1], best[2]),
                    distance=best[3]))
    return suggestions


# ---------------------------------------------------------------------------
# Gap FILLING: synthesize interpolated centroid rows in the missing frames
# ---------------------------------------------------------------------------
@dataclass
class GapFill:
    """A proposed set of synthesized rows that bridge one within-track gap.

    Unlike a relink (which reconnects an *existing* candidate track across the
    gap), a fill invents new per-frame centroid rows for the SAME track in the
    frames where it vanished, so downstream motility/MSD sees a continuous path.
    The rows carry no mask (there is no segmentation for an invented frame); they
    are flagged via ``interpolated=True`` rows so they can be told apart.
    """
    track_id: int
    gap_start: int                 # last real frame before the gap
    gap_end: int                   # first real frame after the gap
    frames: list                   # the missing frame indices to fill
    xy: list                       # predicted (x, y) per missing frame
    method: str                    # "linear" or "spline"


def suggest_gap_fills(df: pd.DataFrame, thresholds: config.Thresholds,
                      method: str = "spline",
                      max_gap: int | None = None) -> list[GapFill]:
    """Propose interpolated centroid rows for every within-track temporal gap.

    For each track that disappears for one or more frames and reappears, predict
    its centroid in the missing frames with :func:`predict_path` (spline by
    default, falling back to linear as that function guarantees). Gaps longer
    than ``max_gap`` are skipped (a very long disappearance is more likely a
    genuine exit/re-entry than a tracking dropout worth inventing positions for).
    Returns one :class:`GapFill` per bridged gap; the caller approves and applies.
    """
    fills: list[GapFill] = []
    if df is None or df.empty:
        return fills
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return fills
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0].sort_values([COL_TRACK, COL_FRAME])
    if max_gap is None:
        max_gap = SPLINE_MAX_GAP_FOR_SPLINE

    for tid, g in d.groupby(COL_TRACK):
        if config.is_quarantined_id(tid):
            continue
        frames = g[COL_FRAME].to_numpy(dtype=int)
        xs = g[COL_X].to_numpy(dtype=float)
        ys = g[COL_Y].to_numpy(dtype=float)
        diffs = np.diff(frames)
        gap_idx = np.flatnonzero(diffs > thresholds.gap_tolerance)
        for gi in gap_idx:
            gap_start = int(frames[gi])
            gap_end = int(frames[gi + 1])
            missing = list(range(gap_start + 1, gap_end))
            if not missing or len(missing) > int(max_gap):
                continue
            px, py = predict_path(frames, xs, ys, missing, method=method)
            xy = [(float(px[i]), float(py[i])) for i in range(len(missing))]
            fills.append(GapFill(track_id=int(tid), gap_start=gap_start,
                                 gap_end=gap_end, frames=missing, xy=xy,
                                 method=method))
    return fills


# ---------------------------------------------------------------------------
# Family membership: full lineage (ancestral root + every descendant)
# ---------------------------------------------------------------------------
def parent_of(df: pd.DataFrame) -> dict:
    """Return {track_id: parent_id} for every track that has a parent (>0).

    Vectorized (no per-row Python loop): on a large table this is the difference
    between milliseconds and minutes, and it is called by the lineage helpers.
    """
    if df is None or df.empty or COL_PARENT not in df:
        return {}
    links = df[df[COL_PARENT] > 0][[COL_TRACK, COL_PARENT]].drop_duplicates()
    return dict(zip(links[COL_TRACK].astype(int).to_numpy().tolist(),
                    links[COL_PARENT].astype(int).to_numpy().tolist()))


def _root_in_map(p_of: dict, track_id: int, cache: dict | None = None) -> int:
    """Walk up a prebuilt {child: parent} map to the root, memoizing results.

    ``cache`` (if given) maps already-resolved track_id -> root so repeated calls
    across a whole dataset are near-linear instead of quadratic. Cycles are
    broken defensively.
    """
    tid = int(track_id)
    if cache is not None and tid in cache:
        return cache[tid]
    path = []
    cur = tid
    while cur in p_of:
        if cache is not None and cur in cache:
            cur = cache[cur]
            break
        if cur in path:  # defensive cycle break
            break
        path.append(cur)
        cur = p_of[cur]
    if cache is not None:
        for node in path:
            cache[node] = cur
    return cur


def root_of(df: pd.DataFrame, track_id: int, p_of: dict | None = None) -> int:
    """Walk parent links up to the ancestral root of ``track_id``.

    The root is the oldest ancestor reachable by following parent_id. A track
    with no parent is its own root. Pass a prebuilt ``p_of`` map (from
    ``parent_of``) to avoid rebuilding it on repeated calls. Cycles are broken
    defensively so this always terminates.
    """
    if p_of is None:
        p_of = parent_of(df)
    return _root_in_map(p_of, int(track_id))


def _children_map(df: pd.DataFrame) -> dict:
    """Return {parent_id: [child ids]} built once, vectorized."""
    if df is None or df.empty or COL_PARENT not in df:
        return {}
    links = df[df[COL_PARENT] > 0][[COL_TRACK, COL_PARENT]].drop_duplicates()
    ch: dict[int, list[int]] = {}
    for tid, par in zip(links[COL_TRACK].astype(int).to_numpy().tolist(),
                        links[COL_PARENT].astype(int).to_numpy().tolist()):
        ch.setdefault(par, []).append(tid)
    return ch


def family_of(df: pd.DataFrame, track_id: int) -> set:
    """Return the set of every track_id in the same lineage as ``track_id``.

    This is the whole family tree the cell belongs to: its ancestral root, that
    root's descendants of every generation (children, grandchildren, ...), and
    therefore also the cell's own siblings, cousins, parents and so on. Used by
    the lineage-focus view, which highlights an entire genealogy at once.

    Builds the parent/children maps once (fast even on very large tables) and is
    robust to a cell with no lineage (returns just that cell) and to accidental
    cycles.
    """
    if df is None or df.empty:
        return {int(track_id)}
    p_of = parent_of(df)
    children_of = _children_map(df)
    root = _root_in_map(p_of, int(track_id))

    family = set()
    stack = [int(root)]
    while stack:
        node = stack.pop()
        if node in family:
            continue
        family.add(node)
        for child in children_of.get(node, []):
            if child not in family:
                stack.append(int(child))
    # Always include the queried cell, even if it is unlinked / flagged-single.
    family.add(int(track_id))
    return family


def lineage_roots(df: pd.DataFrame) -> set:
    """Return the set of family-root IDs (one per distinct lineage).

    Each connected genealogy is identified by its ancestral root. Flagged
    singles (curated cells with no links) count as their own one-member lineage,
    so the set covers every track the user might want to review.

    Builds the parent map once and memoizes roots, so it is fast (sub-second)
    even with thousands of lineages over hundreds of thousands of rows.
    """
    if df is None or df.empty:
        return set()
    groups = classify_lineage(df)
    roots = set(int(r) for r in groups.roots)
    if groups.children_of:
        p_of = parent_of(df)
        cache: dict[int, int] = {}
        for m in groups.children_of:
            roots.add(_root_in_map(p_of, int(m), cache))
    return roots
