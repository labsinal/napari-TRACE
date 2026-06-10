"""
Triage for very large datasets: score each cell's confidence, build a review
queue, accept the confident remainder in bulk, and validate that bulk with a
random sample.

Why this exists
---------------
With thousands of cells, curating one-by-one is infeasible and most tracks are
probably fine. The workflow this module enables is "curate by exception":

  1. score_cells()      -> a 0..1 confidence per track (low = needs review),
                           PLUS a 0..1 sub-score per cell characteristic
                           (area, mobility, lifetime, outcome).
  2. triage_queue()     -> tracks split into REVIEW (low score) and ACCEPT
                           (high score), ordered worst-first.
  3. (user reviews the REVIEW bucket, then bulk-accepts the ACCEPT bucket.)
  4. validation_sample()-> a random sample of the accepted bucket to eyeball,
                           giving a defensible empirical error rate for a paper.

Per-characteristic scoring
--------------------------
Each cell characteristic ("dimension") gets a 0..1 sub-score with TWO parts,
and the design deliberately makes WITHIN-track inconsistency the primary driver
and mere atypicality a weak secondary nudge. On heterogeneous / high-density
datasets a cell being unusual is poor evidence that it is wrong, so deviation
alone should rarely send a cell to review.

  (A) Local anomaly penalties -- the PRIMARY driver. WITHIN-track red flags the
      population cannot see: an impossible single-frame jump (likely ID swap), a
      temporal gap, a sudden area step (fusion/leak, e.g. a merge-split that
      fakes a mitosis), a missing or incoherent outcome. These reduce the
      relevant dimension's sub-score.

  (B) Population deviation -- a SECONDARY nudge, attenuated by DEVIATION_GAIN.
      For the continuous traits (area, mobility, lifetime) the track's value is
      compared to the dataset distribution (robust z = |value - median| /
      (1.4826*MAD)); being far off lowers the sub-score only slightly. With
      DEVIATION_GAIN = 1.0 this reverts to the old deviation-dominated behaviour.

      area      <- area_jump      x  small_deviation(mean area)
      mobility  <- jump           x  small_deviation(mean step)
      lifetime  <- gap            x  small_deviation(n_frames)
      outcome   <- no_outcome (coverage-adaptive), outcome_incoherent

The ``no_outcome`` penalty is scaled by annotation COVERAGE: on a mostly
uncurated dataset "has no outcome yet" is true of nearly everything and carries
no information, so it is down-weighted toward zero; as more cells are curated an
uncurated straggler becomes genuinely notable and the penalty returns to full.

A single-frame track (no trajectory at all) has its temporal characteristics
(mobility, lifetime) hard-zeroed regardless. The FINAL aggregate score is
derived FROM the sub-scores as ``1 - sum_d (1 - subscore_d)``, clamped to
[0, 1]; the triage cutoff then operates on this aggregate as before.

No Qt / napari imports here -- pure pandas/numpy so it is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analysis
from .config import (COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT,
                     COL_AREA, COL_BORDER, FINAL_OUTCOMES, OUTCOME_EXIT,
                     OUTCOME_DEATH, NON_FLAG_VALUES, is_quarantined_id)


# Local-anomaly penalty weights (within-track red flags the population can't
# see). These are *additional* multiplicative reductions on top of the
# population-deviation sub-score (the primary driver).
DEFAULT_WEIGHTS = {
    "no_outcome": 0.45,          # no final outcome at all -> must review
    "jump": 0.30,                # an impossible step -> likely an ID swap
    "gap": 0.20,                 # a temporal hole -> possible missed detection
    "outcome_incoherent": 0.25,  # outcome contradicts what the trajectory shows
    "area_jump": 0.18,           # sudden area change (fusion/leak; merge-split)
}

# --- population-deviation parameters (the relationship to the dataset means) ---
# Robust z = |value - centre| / scale. Within DEVIATION_FREE_Z robust-sigma the
# sub-score is 1.0; beyond it, a Gaussian fall-off with width DEVIATION_TOLERANCE
# (in robust-sigma) brings the sub-score down. Smaller tolerance / free-zone =>
# stricter (a track must hug the population centre to score high).
DEVIATION_FREE_Z = 1.5       # robust-sigma of "this is normal", scored 1.0
DEVIATION_TOLERANCE = 1.25   # Gaussian width (robust-sigma) of the fall-off
# How much population deviation is allowed to lower a sub-score. The score
# prioritizes within-track inconsistency over atypicality, so deviation only
# nudges. Kept at/below (1 - cutoff) so that a single deviating axis alone never
# pushes a cell below the default 0.85 review cutoff: only a real within-track
# anomaly, or atypicality on several axes at once, sends a cell to review. 1.0 =
# old deviation-dominated behaviour; 0.0 = ignore population deviation entirely.
DEVIATION_GAIN = 0.15
DEVIATION_REASON_SCORE = 0.75   # add an explanatory reason below this sub-score
MIN_POP_FOR_STATS = 5        # need at least this many tracks to trust a distribution
DEVIATION_CENTER = "median"  # "median" (robust, default) or "mean"

# Continuous traits that are scored relative to the dataset distribution.
DEVIATION_DIMS = ("area", "mobility", "lifetime")

# Which characteristic ("dimension") each LOCAL penalty signal belongs to. The
# validation reliability plot uses exactly these dimensions, so keep the keys
# of CHARACTERISTICS and the values of SIGNAL_DIMENSION in sync.
CHARACTERISTICS = ("area", "mobility", "lifetime", "outcome")
# Characteristics that are *only* defined over time: a single-frame track has no
# measurable trajectory, so these are hard-zeroed (you cannot trust what cannot
# be observed). Area stays scored -- it is a single-frame morphological quantity.
TEMPORAL_CHARACTERISTICS = ("mobility", "lifetime")
SIGNAL_DIMENSION = {
    "area_jump": "area",
    "jump": "mobility",
    "gap": "lifetime",
    "no_outcome": "outcome",
    "outcome_incoherent": "outcome",
}


@dataclass
class CellScore:
    track_id: int
    score: float                                   # aggregate (derived) 0..1
    subscores: dict = field(default_factory=dict)  # {dimension: 0..1}
    penalties: dict = field(default_factory=dict)  # {signal: value} (raw)
    reasons: list = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    outcome: str = ""

    def weakest_dimension(self):
        """Return the characteristic with the lowest sub-score (or None)."""
        if not self.subscores:
            return None
        return min(self.subscores, key=self.subscores.get)


@dataclass
class TriageResult:
    scores: list = field(default_factory=list)   # list[CellScore], worst first
    review: list = field(default_factory=list)   # track_ids to review (low score)
    accept: list = field(default_factory=list)   # track_ids confident enough
    cutoff: float = 0.0
    population: dict = field(default_factory=dict)  # {dim: {center, scale, mean, n}}
    coverage: float = 0.0                           # annotation coverage in [0,1]

    def summary(self) -> str:
        n = len(self.scores)
        if n == 0:
            return "No cells to triage."
        base = (f"{n} cells: {len(self.review)} to review "
                f"({100*len(self.review)/n:.0f}%), {len(self.accept)} accept "
                f"({100*len(self.accept)/n:.0f}%) at cutoff {self.cutoff:.2f} "
                f"[annotation coverage {100*self.coverage:.0f}%].")
        if self.population:
            bits = []
            for dim, st in self.population.items():
                if st:
                    bits.append(f"{dim}~{st['center']:.0f}")
            if bits:
                base += "  Dataset centres: " + ", ".join(bits) + "."
        return base

    def score_by_id(self) -> dict:
        return {c.track_id: c for c in self.scores}


def _per_track_metrics(d):
    """Per-track continuous traits, vectorized. ``d`` must be cleaned & sorted.

    Returns {dim: pandas.Series indexed by track_id}:
      lifetime -> number of distinct frames the track is observed in
      mobility -> mean per-frame step length sqrt(dx^2 + dy^2)
      area     -> mean per-frame mask area (only if COL_AREA is present)
    """
    g = d.groupby(COL_TRACK)
    metrics = {}
    metrics["lifetime"] = g[COL_FRAME].nunique().astype(float)
    dx = g[COL_X].diff()
    dy = g[COL_Y].diff()
    step = np.sqrt(dx.to_numpy(float) ** 2 + dy.to_numpy(float) ** 2)
    metrics["mobility"] = (pd.Series(step, index=d.index)
                           .groupby(d[COL_TRACK]).mean())
    if COL_AREA in d.columns:
        metrics["area"] = g[COL_AREA].mean()
    return metrics


def _center_scale(values, center=DEVIATION_CENTER):
    """Robust (median+MAD) or classical (mean+std) centre & scale of a 1-D array.

    Returns {center, scale, mean, median, n} or None if the sample is too small
    or degenerate (no spread), in which case deviation scoring is skipped.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < MIN_POP_FOR_STATS:
        return None
    med = float(np.median(v))
    mean = float(np.mean(v))
    if center == "mean":
        c = mean
        scale = float(np.std(v))
    else:
        c = med
        scale = 1.4826 * float(np.median(np.abs(v - med)))
        if scale <= 1e-9:                       # degenerate MAD -> fall back to std
            scale = float(np.std(v))
    if scale <= 1e-9:
        return None
    return {"center": c, "scale": scale, "mean": mean, "median": med, "n": int(v.size)}


def annotation_coverage(df):
    """Fraction of non-quarantined tracks that already have a final outcome.

    Used to make the ``no_outcome`` penalty adaptive: at coverage ~0 (a fresh,
    uncurated dataset) "no outcome" is non-discriminative and is down-weighted
    to near zero; at coverage ~1 an uncurated straggler is notable and keeps the
    full penalty. Returns a float in [0, 1].
    """
    if df is None or df.empty or COL_OUTCOME not in df.columns:
        return 0.0
    d = df.dropna(subset=[COL_TRACK]).copy()
    if d.empty:
        return 0.0
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[(d[COL_TRACK] > 0) & (~d[COL_TRACK].map(is_quarantined_id))]
    if d.empty:
        return 0.0
    finals = set(FINAL_OUTCOMES)
    has_final = d.groupby(COL_TRACK)[COL_OUTCOME].agg(
        lambda s: bool(set(s.astype(str)) & finals))
    return float(has_final.mean()) if has_final.size else 0.0


def population_stats(df, center=DEVIATION_CENTER):
    """Distribution centre & scale of each continuous trait across the dataset.

    Computed on the CURRENT dataframe, so after curation the reference
    distribution reflects the curated state. Quarantined / non-positive IDs are
    excluded. Returns {dim: stats-or-None}.
    """
    if df is None or df.empty:
        return {dim: None for dim in DEVIATION_DIMS}
    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return {dim: None for dim in DEVIATION_DIMS}
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0]
    d = d[~d[COL_TRACK].map(is_quarantined_id)]
    d = d.sort_values([COL_TRACK, COL_FRAME])
    metrics = _per_track_metrics(d)
    return {dim: (_center_scale(metrics[dim].to_numpy(), center)
                  if dim in metrics else None)
            for dim in DEVIATION_DIMS}


def _deviation_score(value, st):
    """1.0 near the population centre, falling off with robust distance.

    Two-sided: a track that is far ABOVE the centre (e.g. an over-long track =
    likely ID merge, or an over-large area = likely fusion) is just as suspect
    as one far below. A 1-sigma free zone keeps ordinary spread from scoring < 1.
    """
    if st is None or value is None or not np.isfinite(value):
        return 1.0
    z = abs(float(value) - st["center"]) / st["scale"]
    if z <= DEVIATION_FREE_Z:
        return 1.0
    zz = z - DEVIATION_FREE_Z
    return float(np.exp(-0.5 * (zz / DEVIATION_TOLERANCE) ** 2))


def _outcome_coherence_penalty(g, last_movie_frame):
    """Penalty if the recorded outcome contradicts the visible trajectory.

    Trajectory-focused checks (the user's priority):
      - Exit but the cell neither touches the border at its end nor disappears
        before the movie ends -> suspicious (an "Exit" that didn't leave).
      - Death but the cell keeps a large, stable area to the end -> suspicious
        (a "Death" that doesn't shrink/stop).
    Returns (penalty, reason or None). Conservative: only penalizes clear
    contradictions, and only when the needed columns exist.
    """
    oc = ""
    if COL_OUTCOME in g:
        vals = [str(v) for v in g[COL_OUTCOME] if str(v) not in ("", "nan", "None")]
        oc = vals[0] if vals else ""
    if oc not in (OUTCOME_EXIT, OUTCOME_DEATH):
        return 0.0, None

    last_frame = int(g[COL_FRAME].iloc[-1])
    vanished = last_frame < last_movie_frame

    if oc == OUTCOME_EXIT:
        touches_border = False
        if COL_BORDER in g:
            tailb = g[COL_BORDER].to_numpy()[-3:]
            touches_border = bool(np.any(tailb.astype(bool)))
        if not touches_border and not vanished:
            return DEFAULT_WEIGHTS["outcome_incoherent"], \
                "Exit but stays in-frame to the end (no border, no vanish)"
    elif oc == OUTCOME_DEATH:
        if COL_AREA in g and len(g) >= 4:
            areas = g[COL_AREA].to_numpy(dtype=float)
            end = np.nanmedian(areas[-3:])
            peak = np.nanmax(areas)
            if np.isfinite(end) and np.isfinite(peak) and peak > 0 \
                    and end > 0.8 * peak and not vanished:
                return DEFAULT_WEIGHTS["outcome_incoherent"], \
                    "Death but area stays high to the end"
    return 0.0, None


def _aggregate_from_subscores(subscores: dict) -> float:
    """Final score derived FROM the per-characteristic sub-scores.

    score = 1 - sum_d (1 - subscore_d), clamped to [0, 1]. Equivalent to
    1 - (total penalty) whenever no dimension's sub-score was clamped at 0.
    """
    deficit = sum(1.0 - s for s in subscores.values())
    return round(max(0.0, 1.0 - deficit), 3)


def score_cells(df, thresholds, mask=None, weights=None, center=DEVIATION_CENTER,
                population=None):
    """Return a list[CellScore], worst (lowest aggregate) first.

    Sub-scores combine population deviation (primary, dataset-relative) with
    local within-track anomaly penalties (secondary). ``population`` may be a
    precomputed dict from population_stats() to avoid recomputation; otherwise it
    is computed from ``df``.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    if df is None or df.empty:
        return []

    d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
    if d.empty:
        return []
    d[COL_TRACK] = d[COL_TRACK].astype(int)
    d = d[d[COL_TRACK] > 0]
    d = d.sort_values([COL_TRACK, COL_FRAME])
    last_movie_frame = (int(mask.shape[0]) - 1) if mask is not None \
        else int(d[COL_FRAME].max())

    # Population distribution per continuous trait (the dataset "means") and the
    # matching per-track values, both off the same cleaned frame.
    metrics = _per_track_metrics(d[~d[COL_TRACK].map(is_quarantined_id)])
    pop = population if population is not None else {
        dim: (_center_scale(metrics[dim].to_numpy(), center) if dim in metrics else None)
        for dim in DEVIATION_DIMS}

    # Reuse the (calibrated) anomaly detector as the backbone of the signals.
    # Annotation coverage drives the adaptive no_outcome weight (see module
    # docstring): on an uncurated dataset "no outcome" floods the queue without
    # discriminating, so its penalty is scaled down toward zero.
    coverage = annotation_coverage(d)
    w["no_outcome"] = w["no_outcome"] * coverage

    rep = analysis.detect_anomalies(d, thresholds, include_outcome_checks=True)
    gap_ids = set(rep.gaps)
    jump_ids = set(rep.jumps)
    # no_outcome messages are "Cell <id> has ..."; recover the ids robustly.
    no_outcome_ids = set()
    for msg in rep.no_outcome:
        try:
            no_outcome_ids.add(int(str(msg).split()[1]))
        except (IndexError, ValueError):
            pass

    # Human-readable reason per signal (for the review list / tooltips).
    reason_text = {
        "no_outcome": "no final outcome",
        "jump": "impossible jump (possible ID swap)",
        "gap": "temporal gap",
        "area_jump": "sudden area change",
    }

    out = []
    for tid, g in d.groupby(COL_TRACK):
        if is_quarantined_id(tid):
            continue

        penalties = {}
        if int(tid) in no_outcome_ids:
            penalties["no_outcome"] = w["no_outcome"]
        if int(tid) in jump_ids:
            penalties["jump"] = w["jump"]
        if int(tid) in gap_ids:
            penalties["gap"] = w["gap"]

        n_frames = int(g[COL_FRAME].nunique())

        # Area jump (sudden fusion/leak) -- a morphology red flag.
        if COL_AREA in g and len(g) >= 3:
            a = g[COL_AREA].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(np.diff(a)) / np.maximum(a[:-1], 1.0)
            if np.nanmax(rel) > 1.0:   # area more than doubled/halved in one step
                penalties["area_jump"] = w["area_jump"]

        # Outcome coherence with the visible trajectory.
        pen_oc, reason_oc = _outcome_coherence_penalty(g, last_movie_frame)
        if pen_oc > 0:
            penalties["outcome_incoherent"] = pen_oc

        # ---- collapse LOCAL signals into per-dimension penalties ----
        dim_penalty = {dim: 0.0 for dim in CHARACTERISTICS}
        reasons = []
        for sig, val in penalties.items():
            dim_penalty[SIGNAL_DIMENSION[sig]] += val
            if sig == "outcome_incoherent":
                reasons.append(reason_oc or "outcome incoherent")
            else:
                reasons.append(reason_text.get(sig, sig))

        # ---- per-dimension sub-score = population deviation x local penalty ----
        subscores = {}
        for dim in CHARACTERISTICS:
            local = max(0.0, 1.0 - dim_penalty[dim])
            if dim in DEVIATION_DIMS:
                st = pop.get(dim)
                value = metrics[dim].get(int(tid)) if dim in metrics else None
                dev_raw = _deviation_score(value, st)
                # Attenuate: deviation only nudges the sub-score (see B0 design).
                dev = 1.0 - DEVIATION_GAIN * (1.0 - dev_raw)
                sub = dev * local
                if st is not None and dev < DEVIATION_REASON_SCORE \
                        and value is not None and np.isfinite(value):
                    reasons.append(
                        f"{dim} {float(value):.0f} far from dataset (~{st['center']:.0f})")
            else:                                  # outcome: penalty-only
                sub = local
            subscores[dim] = round(sub, 3)

        # A single-frame track carries no trajectory: zero every temporal
        # characteristic outright (overrides the deviation/penalty product).
        if n_frames <= 1:
            for dim in TEMPORAL_CHARACTERISTICS:
                subscores[dim] = 0.0
            reasons.append("single-frame track: temporal scores zeroed")
        score = _aggregate_from_subscores(subscores)

        oc = ""
        if COL_OUTCOME in g:
            vals = [str(v) for v in g[COL_OUTCOME] if str(v) not in ("", "nan", "None")]
            oc = vals[0] if vals else ""

        out.append(CellScore(
            track_id=int(tid), score=score, subscores=subscores,
            penalties=penalties, reasons=reasons,
            first_frame=int(g[COL_FRAME].iloc[0]),
            last_frame=int(g[COL_FRAME].iloc[-1]),
            outcome=oc))

    out.sort(key=lambda c: (c.score, c.first_frame))   # worst first
    return out


def triage_queue(df, thresholds, mask=None, cutoff=0.85, weights=None,
                 center=DEVIATION_CENTER):
    """Split cells into REVIEW (score < cutoff) and ACCEPT (>= cutoff).

    ``cutoff`` defaults to 0.85 (generous -> review more), matching a
    maximum-precision goal. REVIEW is ordered worst-first so the user attacks
    the most suspicious cells first. The dataset distribution used for the
    population-relative scoring is attached to the result for transparency.
    """
    pop = population_stats(df, center=center)
    scores = score_cells(df, thresholds, mask=mask, weights=weights,
                          center=center, population=pop)
    review = [c.track_id for c in scores if c.score < cutoff]
    accept = [c.track_id for c in scores if c.score >= cutoff]
    return TriageResult(scores=scores, review=review, accept=accept,
                        cutoff=cutoff, population=pop,
                        coverage=annotation_coverage(df))


def validation_sample(accept_ids, n=50, seed=None):
    """Return a random sample of accepted track_ids for manual spot-checking.

    Reviewing this sample after a bulk-accept gives an empirical error rate:
    if k of n sampled cells are wrong, the estimated error is k/n with a simple
    binomial confidence interval (see error_estimate). For a paper, this is the
    defensible "we validated the auto-accepted set by random sampling" step.
    """
    ids = list(accept_ids)
    if not ids:
        return []
    rng = np.random.default_rng(seed)
    n = min(int(n), len(ids))
    idx = rng.choice(len(ids), size=n, replace=False)
    return sorted(int(ids[i]) for i in idx)


def error_estimate(n_sampled, n_wrong):
    """Wilson 95% interval (point, upper) on the error rate from a sample.

    The Wilson interval is well-behaved even when n_wrong is 0 (the usual,
    hoped-for case), unlike the naive 0% +/- 0%.
    """
    if n_sampled <= 0:
        return 0.0, 1.0
    p = n_wrong / n_sampled
    z = 1.96
    denom = 1 + z * z / n_sampled
    centre = (p + z * z / (2 * n_sampled)) / denom
    half = (z * np.sqrt(p * (1 - p) / n_sampled
                        + z * z / (4 * n_sampled * n_sampled))) / denom
    return round(p, 4), round(min(1.0, centre + half), 4)
