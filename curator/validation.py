"""
Validation session + reliability plots for the random spot-check workflow.

Pure pandas/numpy/matplotlib (no Qt/napari) so it stays unit-testable, mirroring
triage.py and track_plots.py. The Qt dialog that drives it lives in
validation_dialog.py.

Flow
----
1. triage draws a random sample of the auto-ACCEPTED cells (high score).
2. The user steps through them one cell at a time. For each cell they either:
     - tick "OK" without editing  -> the automatic call was CORRECT, or
     - edit it (any curation op)   -> the automatic call was an ERROR; every
                                       edit is captured in a transient per-cell
                                       log (this object only; the persistent
                                       audit.log.csv is written independently).
3. When every sampled cell has been reviewed (ticked OK), reliability_figure()
   shows whether the confidence score -- globally AND per characteristic
   (area / mobility / lifetime / outcome) -- actually predicts the error.

A cell counts as an ERROR iff it was modified during the session, regardless of
whether the OK tick was given before or after the edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt

import numpy as np
import matplotlib.pyplot as plt

from .triage import CHARACTERISTICS, error_estimate


# A characteristic counts as "flagged by the auto" (in the reliability panel)
# when its sub-score drops below this — i.e. a real deviation, not just being a
# hair off the population centre.
FLAG_SUBSCORE = 0.85


# Edits that don't change a cell's annotation/identity and so should NOT be
# counted as "the automatic was wrong" if they fire on a sampled id.
NON_ERROR_ACTIONS = {"validation_sample", "bulk_accept", "set_treatment",
                     "shuffle", "harmonize"}


@dataclass
class CellReview:
    track_id: int
    score: float
    subscores: dict
    reasons: list = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    outcome: str = ""
    checked: bool = False                       # user confirmed it's correct
    modified: bool = False                      # an edit touched this cell
    edits: list = field(default_factory=list)   # [(ts, action, detail)]

    @property
    def status(self) -> str:
        if self.modified and self.checked:
            return "fixed"          # was wrong, edited, confirmed
        if self.modified:
            return "modified"       # edited, not yet confirmed
        if self.checked:
            return "correct"        # auto was right, confirmed
        return "pending"


class ValidationSession:
    """Transient state for one random-sample validation pass."""

    def __init__(self, scored_cells, n_total=None):
        # scored_cells: list[triage.CellScore] for the sampled ids.
        self.reviews = {}
        for c in scored_cells:
            self.reviews[int(c.track_id)] = CellReview(
                track_id=int(c.track_id), score=float(c.score),
                subscores=dict(c.subscores), reasons=list(c.reasons),
                first_frame=int(c.first_frame), last_frame=int(c.last_frame),
                outcome=str(c.outcome or ""))
        self.created_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.n_total_accepted = n_total

    # -- queries --------------------------------------------------------
    @property
    def ids(self):
        return list(self.reviews.keys())

    @property
    def n(self):
        return len(self.reviews)

    @property
    def n_checked(self):
        return sum(1 for r in self.reviews.values() if r.checked)

    @property
    def n_modified(self):
        return sum(1 for r in self.reviews.values() if r.modified)

    @property
    def complete(self):
        return self.n > 0 and self.n_checked == self.n

    def first_unchecked(self):
        for tid in self.ids:
            if not self.reviews[tid].checked:
                return self.reviews[tid]
        return None

    # -- mutations ------------------------------------------------------
    def note_action(self, action, ids, detail=""):
        """Record a curation action against any sampled cell it touched.

        Returns the list of sampled ids actually touched (empty if none / if the
        action is in NON_ERROR_ACTIONS).
        """
        if not ids or action in NON_ERROR_ACTIONS:
            return []
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        touched = []
        for raw in ids:
            try:
                tid = int(raw)
            except (TypeError, ValueError):
                continue
            r = self.reviews.get(tid)
            if r is None:
                continue
            r.modified = True
            r.edits.append((ts, str(action), str(detail)))
            touched.append(tid)
        return touched

    def set_checked(self, track_id, value=True):
        r = self.reviews.get(int(track_id))
        if r is not None:
            r.checked = bool(value)

    # -- summaries ------------------------------------------------------
    def summary_text(self):
        p, upper = error_estimate(self.n_checked, self.n_modified)
        base = (f"Reviewed {self.n_checked}/{self.n}  ·  "
                f"errors (auto wrong) {self.n_modified}  ·  "
                f"confirmed correct {self.n_checked - self.n_modified}")
        if self.n_checked:
            base += (f"  ·  error rate {p*100:.0f}% "
                     f"(95% upper bound {upper*100:.0f}%)")
        return base

    def edit_log_lines(self):
        """Flat, newest-first transient log of every edit in this session."""
        rows = []
        for r in self.reviews.values():
            for ts, action, detail in r.edits:
                rows.append((ts, r.track_id, action, detail))
        rows.sort(reverse=True)
        return rows


# ---------------------------------------------------------------------------
# Reliability / calibration figure
# ---------------------------------------------------------------------------
def _binned_error(scores, errors, edges):
    """Return (centers, rate, counts) of error rate per score bin."""
    centers, rates, counts = [], [], []
    idx = np.digitize(scores, edges[1:-1], right=False)
    for b in range(len(edges) - 1):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        rates.append(float(np.mean(errors[m])))
        counts.append(k)
    return np.array(centers), np.array(rates), np.array(counts)


def reliability_figure(session: ValidationSession, n_bins=4):
    """score-vs-error-rate reliability of the automatic, global + per-trait.

    Left  : aggregate confidence score (binned) vs observed error rate, with the
            "ideal" line error = 1 - score. Bars below the line => auto better
            than its own score implies; on the line => well calibrated.
    Right : per-characteristic error rate -- for area / mobility / lifetime /
            outcome, the error rate among cells the auto FLAGGED on that trait
            (sub-score < 1) vs the cells it considered CLEAN (sub-score == 1).
            A trustworthy signal: the flagged bar is much taller than clean.
    """
    reviewed = [r for r in session.reviews.values() if r.checked]
    if not reviewed:
        reviewed = list(session.reviews.values())  # show partial if nothing ticked

    scores = np.array([r.score for r in reviewed], dtype=float)
    errors = np.array([1.0 if r.modified else 0.0 for r in reviewed], dtype=float)
    n = len(reviewed)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- LEFT: calibration of the aggregate score ----
    lo = float(np.floor(min(scores.min(), 0.85) * 20) / 20) if n else 0.8
    edges = np.linspace(lo, 1.0, n_bins + 1)
    centers, rates, counts = _binned_error(scores, errors, edges)

    axL.plot([lo, 1.0], [1.0 - lo, 0.0], "--", color="#888888",
             label="ideal (error = 1 - score)")
    if centers.size:
        width = (edges[1] - edges[0]) * 0.7
        bars = axL.bar(centers, rates, width=width, color="#4C72B0",
                       edgecolor="#2A2A2A", label="observed error rate")
        for x, y, c in zip(centers, rates, counts):
            axL.text(x, y + 0.02, f"n={c}", ha="center", va="bottom", fontsize=8)
    axL.scatter(scores, errors + np.random.uniform(-0.015, 0.015, n),
                s=18, color="#C44E52", alpha=0.5, zorder=5,
                label="cells (1=auto wrong)")
    axL.set_xlim(lo, 1.0)
    axL.set_ylim(-0.05, 1.05)
    axL.set_xlabel("automatic confidence score (aggregate)")
    axL.set_ylabel("error rate (fraction I had to edit)")
    axL.set_title("Global calibration", fontsize=11, fontweight="bold")
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(alpha=0.25)

    # ---- RIGHT: per-characteristic discrimination ----
    dims = list(CHARACTERISTICS)
    flagged_rate, clean_rate, flagged_n, clean_n = [], [], [], []
    for d in dims:
        sub = np.array([r.subscores.get(d, 1.0) for r in reviewed], dtype=float)
        fmask = sub < FLAG_SUBSCORE      # auto considered this trait suspect
        cmask = ~fmask
        flagged_rate.append(float(np.mean(errors[fmask])) if fmask.any() else np.nan)
        clean_rate.append(float(np.mean(errors[cmask])) if cmask.any() else np.nan)
        flagged_n.append(int(fmask.sum()))
        clean_n.append(int(cmask.sum()))

    xpos = np.arange(len(dims))
    bw = 0.38
    b1 = axR.bar(xpos - bw / 2, np.nan_to_num(flagged_rate), bw,
                 color="#DD8452", edgecolor="#2A2A2A",
                 label="auto flagged this trait")
    b2 = axR.bar(xpos + bw / 2, np.nan_to_num(clean_rate), bw,
                 color="#55A868", edgecolor="#2A2A2A",
                 label="auto clean on this trait")
    for x, r, cnt in zip(xpos - bw / 2, flagged_rate, flagged_n):
        axR.text(x, (0 if np.isnan(r) else r) + 0.02, f"n={cnt}",
                 ha="center", va="bottom", fontsize=8)
    for x, r, cnt in zip(xpos + bw / 2, clean_rate, clean_n):
        axR.text(x, (0 if np.isnan(r) else r) + 0.02, f"n={cnt}",
                 ha="center", va="bottom", fontsize=8)
    axR.set_xticks(xpos)
    axR.set_xticklabels([d.capitalize() for d in dims])
    axR.set_ylim(-0.05, 1.05)
    axR.set_ylabel("error rate within group")
    axR.set_title("Per-characteristic reliability", fontsize=11, fontweight="bold")
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(alpha=0.25, axis="y")

    p, upper = error_estimate(session.n_checked, session.n_modified)
    state = "complete" if session.complete else f"PARTIAL {session.n_checked}/{session.n}"
    fig.suptitle(
        f"Automatic reliability on {n} validated cells ({state})  —  "
        f"overall error {p*100:.0f}%  (95% upper bound {upper*100:.0f}%)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
