"""
Auxiliary Qt panels used by the main UI.

  * DiagnosticsDialog - navigable visual diagnostics (item 14): distributions
    of track lifetime, per-frame displacement and divisions, plus clickable
    lists of flagged cells that jump the viewer to the offending frame/cell.
  * RelinkDialog - review and approve assisted gap-relink suggestions (item 13).

These import Qt and so are only loaded inside the napari environment.
"""

from __future__ import annotations

import numpy as np

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QTabWidget, QWidget,
)
from qtpy.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from . import analysis
from .config import COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_PARENT


class DiagnosticsDialog(QDialog):
    """Visual, navigable replacement for the console diagnostic report."""

    def __init__(self, viewer, state, report, summary, thresholds, jump_cb,
                 parent=None, initial_tab=0):
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.resize(720, 640)
        self.viewer = viewer
        self.state = state
        self.report = report
        self.summary = summary
        self.thresholds = thresholds
        self.jump_cb = jump_cb

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._distributions_tab(), "Distributions")
        tabs.addTab(self._anomaly_list_tab(), "Flagged cells")
        tabs.addTab(self._morphology_tab(), "Morphology")
        tabs.addTab(self._mass_balance_tab(), "Mass balance")
        tabs.setCurrentIndex(int(initial_tab))

        hint = QLabel("Double-click any listed cell to jump to its frame and select it.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # -- distributions -------------------------------------------------
    def _distributions_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        fig = Figure(figsize=(7, 6))
        canvas = FigureCanvas(fig)
        v.addWidget(canvas)

        df = self.state.df
        ax1 = fig.add_subplot(311)
        if not self.summary.empty:
            ax1.hist(self.summary["lifetime"].dropna().values, bins=30, color="#4C72B0")
        ax1.set_title("Track lifetime (tails = suspicious)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("count")

        ax2 = fig.add_subplot(312)
        steps = []
        d = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).sort_values([COL_TRACK, COL_FRAME])
        for _, g in d.groupby(COL_TRACK):
            x = g[COL_X].to_numpy(dtype=float)
            y = g[COL_Y].to_numpy(dtype=float)
            if len(x) > 1:
                steps.append(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        if steps:
            allsteps = np.concatenate(steps)
            ax2.hist(allsteps, bins=40, color="#DD8452")
            ax2.axvline(self.thresholds.max_jump_px, color="red", linestyle="--",
                        label=f"jump threshold {self.thresholds.max_jump_px:.0f}px")
            ax2.legend(fontsize=8)
        ax2.set_title("Per-frame displacement", fontsize=10, fontweight="bold")
        ax2.set_ylabel("count")

        ax3 = fig.add_subplot(313)
        links = df[df[COL_PARENT] > 0]
        if not links.empty:
            birth = links.groupby(COL_TRACK)[COL_FRAME].min()
            events = birth.value_counts().sort_index()
            ax3.bar(events.index, events.values, color="#C44E52")
        ax3.set_title("Divisions per frame", fontsize=10, fontweight="bold")
        ax3.set_xlabel("frame")
        ax3.set_ylabel("count")

        fig.tight_layout()
        return w

    # -- anomaly lists -------------------------------------------------
    def _anomaly_list_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        lst = QListWidget()
        v.addWidget(lst)

        def add_section(title, ids):
            header = QListWidgetItem(f"— {title} ({len(ids)}) —")
            header.setFlags(Qt.NoItemFlags)
            lst.addItem(header)
            for tid in ids:
                first = self._first_frame_of(int(tid))
                it = QListWidgetItem(f"  cell {int(tid)}  (frame {first})")
                it.setData(Qt.UserRole, (first, int(tid)))
                lst.addItem(it)

        add_section("Short tracks", self.report.short)
        add_section("Gaps (vanished then returned)", self.report.gaps)
        add_section("Impossible jumps", self.report.jumps)

        # Lineage violations as informational rows.
        from . import lineage as lin
        viols = lin.validate_lineage(self.state.df)
        if not viols.is_empty():
            header = QListWidgetItem(f"— Lineage violations ({len(viols.messages())}) —")
            header.setFlags(Qt.NoItemFlags)
            lst.addItem(header)
            for m in viols.messages():
                lst.addItem(QListWidgetItem(f"  {m}"))

        lst.itemDoubleClicked.connect(self._on_item)
        return w

    def _mass_balance_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel(
            "Frame transitions whose cell-count change is not explained by "
            "divisions, exits/deaths or border entries.\n"
            "Positive residual = likely split / spurious object; "
            "negative = likely fusion / dropped cell.\n"
            "Double-click to jump to the earlier frame of the transition."))
        lst = QListWidget()
        v.addWidget(lst)
        rep = analysis.check_mass_conservation(
            self.state.df, self.state.mask, self.thresholds)
        if rep.is_empty():
            lst.addItem(QListWidgetItem("No unexplained count changes found."))
        else:
            for r in rep.rows:
                kind = ("split/appearance" if r["residual"] > 0
                        else "fusion/disappearance")
                bits = []
                if r["unexplained_appeared"]:
                    bits.append(f"+{sorted(r['unexplained_appeared'])}")
                if r["unexplained_disappeared"]:
                    bits.append(f"-{sorted(r['unexplained_disappeared'])}")
                detail = ("  " + " ".join(bits)) if bits else ""
                it = QListWidgetItem(
                    f"frame {r['frame']}->{r['frame']+1}  "
                    f"count {r['n_prev']}->{r['n_curr']}  "
                    f"(obs {r['observed_delta']:+d}, exp {r['expected_delta']:+d}, "
                    f"res {r['residual']:+d}, {kind}){detail}")
                # Jump to the earlier frame; preselect an unexplained cell if any.
                pick = (sorted(r["unexplained_disappeared"])
                        or sorted(r["unexplained_appeared"]))
                tid = int(pick[0]) if pick else None
                it.setData(Qt.UserRole, (int(r["frame"]), tid))
                lst.addItem(it)
        lst.itemDoubleClicked.connect(self._on_item)
        return w

    def _morphology_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        lst = QListWidget()
        v.addWidget(lst)
        me = analysis.detect_morphology_errors(self.state.mask, self.thresholds)
        if me.empty:
            lst.addItem(QListWidgetItem("No morphology anomalies found."))
        else:
            for _, row in me.iterrows():
                it = QListWidgetItem(
                    f"frame {int(row['frame'])}  cell {int(row['track_id'])}  "
                    f"area={row['area']:.0f}  sol={row['solidity']:.2f}  "
                    f"ecc={row['eccentricity']:.2f}  [{row['reason']}]")
                it.setData(Qt.UserRole, (int(row["frame"]), int(row["track_id"])))
                lst.addItem(it)
        lst.itemDoubleClicked.connect(self._on_item)
        return w

    # -- helpers -------------------------------------------------------
    def _first_frame_of(self, tid):
        g = self.state.df[self.state.df[COL_TRACK] == tid]
        if g.empty:
            return 0
        return int(g[COL_FRAME].min())

    def _on_item(self, item):
        data = item.data(Qt.UserRole)
        if data:
            frame, tid = data
            self.jump_cb(frame, tid)


class RelinkDialog(QDialog):
    """Review assisted gap-relink suggestions and approve them one by one."""

    def __init__(self, viewer, state, suggestions, jump_cb, approve_cb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assisted gap relinking")
        self.resize(560, 420)
        self.suggestions = suggestions
        self.jump_cb = jump_cb
        self.approve_cb = approve_cb

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Suggested reconnections (double-click to preview the missing frame):"))
        self.lst = QListWidget()
        for s in suggestions:
            it = QListWidgetItem(
                f"track {s.track_id}: gap {s.gap_start}->{s.gap_end}; "
                f"candidate {s.candidate_id} at frame {s.missing_frame} "
                f"(dist {s.distance:.1f}px)")
            it.setData(Qt.UserRole, s)
            self.lst.addItem(it)
        self.lst.itemDoubleClicked.connect(self._preview)
        layout.addWidget(self.lst)

        row = QHBoxLayout()
        btn_approve = QPushButton("Approve selected")
        btn_close = QPushButton("Close")
        btn_approve.clicked.connect(self._approve)
        btn_close.clicked.connect(self.accept)
        row.addWidget(btn_approve)
        row.addWidget(btn_close)
        layout.addLayout(row)

    def _preview(self, item):
        s = item.data(Qt.UserRole)
        if s:
            self.jump_cb(s.missing_frame, s.track_id)

    def _approve(self):
        item = self.lst.currentItem()
        if item is None:
            return
        s = item.data(Qt.UserRole)
        if s is not None:
            self.approve_cb(s)
            # Remove the approved row so it can't be applied twice.
            self.lst.takeItem(self.lst.row(item))


# ---------------------------------------------------------------------------
# Triage dialog: confidence-ranked review queue + bulk accept + validation
# ---------------------------------------------------------------------------
class TriageDialog(QDialog):
    """Curate-by-exception workflow for very large datasets.

    Shows cells ranked worst-first by a confidence score, lets the user jump to
    each suspicious cell, bulk-accept the confident remainder, and draw a random
    validation sample of the accepted set to estimate an empirical error rate.
    """

    def __init__(self, parent, state, triage_result, jump_cb, accept_cb,
                 sample_cb):
        super().__init__(parent)
        self.state = state
        self.res = triage_result
        self.jump_cb = jump_cb
        self.accept_cb = accept_cb          # called with list of accepted ids
        self.sample_cb = sample_cb          # called with sampled ids -> opens review
        self.setWindowTitle("Triage queue (curate by exception)")
        self.resize(640, 720)
        v = QVBoxLayout(self)

        v.addWidget(QLabel(self.res.summary()))
        v.addWidget(QLabel(
            "Cells are ranked by a confidence score (low = needs review). Review "
            "the list below (double-click to jump), then bulk-accept the rest and "
            "validate it with a random sample."))

        v.addWidget(QLabel("— Cells to REVIEW (worst first) —"))
        self.review_list = QListWidget()
        score_by_id = {c.track_id: c for c in self.res.scores}
        for tid in self.res.review:
            c = score_by_id.get(tid)
            if c is None:
                continue
            why = "; ".join(c.reasons) if c.reasons else "low score"
            it = QListWidgetItem(
                f"  cell {tid}  score {c.score:.2f}  [{c.outcome or 'no outcome'}]  "
                f"(frame {c.first_frame})  — {why}")
            it.setData(Qt.UserRole, (c.first_frame, tid))
            self.review_list.addItem(it)
        if not self.res.review:
            self.review_list.addItem(QListWidgetItem("  (nothing to review at this cutoff)"))
        self.review_list.itemDoubleClicked.connect(self._on_item)
        v.addWidget(self.review_list)

        if self.res.outliers:
            v.addWidget(QLabel(
                "— Biological OUTLIERS (far from the population, NOT flagged as "
                "errors — double-click to inspect) —"))
            self.outlier_list = QListWidget()
            for tid in self.res.outliers:
                c = score_by_id.get(tid)
                if c is None:
                    continue
                why = "; ".join(c.deviation_reasons) if c.deviation_reasons \
                    else "far from dataset"
                it = QListWidgetItem(
                    f"  cell {tid}  deviation {c.deviation_score:.2f}  "
                    f"[{c.outcome or 'no outcome'}]  (frame {c.first_frame})  — {why}")
                it.setData(Qt.UserRole, (c.first_frame, tid))
                self.outlier_list.addItem(it)
            self.outlier_list.itemDoubleClicked.connect(self._on_item)
            v.addWidget(self.outlier_list)

        row = QHBoxLayout()
        self.btn_accept = QPushButton(f"Bulk-accept {len(self.res.accept)} confident cells")
        self.btn_accept.clicked.connect(self._accept)
        row.addWidget(self.btn_accept)
        self.btn_sample = QPushButton("Draw validation sample (50)")
        self.btn_sample.clicked.connect(self._sample)
        row.addWidget(self.btn_sample)
        v.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

    def _on_item(self, item):
        data = item.data(Qt.UserRole)
        if data:
            frame, tid = data
            self.jump_cb(int(frame), int(tid))

    def _accept(self):
        n = self.accept_cb(list(self.res.accept))
        self.status.setText(
            f"Accepted {n} cells (logged). Now draw a validation sample to "
            f"confirm the accepted set has a low error rate.")
        self.btn_accept.setEnabled(False)

    def _sample(self):
        self.sample_cb(list(self.res.accept))


# ---------------------------------------------------------------------------
# Identity-swap review dialog
# ---------------------------------------------------------------------------
class SwapDialog(QDialog):
    """List suspected label swaps and jump to each, preselecting the pair.

    The detector cannot tell a real label swap from two cells that genuinely
    cross paths, so it does not edit anything automatically: double-clicking a
    row jumps to the transition and loads the two tracks into [A] and [B] so the
    user can confirm and fix it with a local or future swap.
    """

    def __init__(self, parent, swaps_df, jump_pair_cb):
        super().__init__(parent)
        self.setWindowTitle("Suspected identity swaps")
        self.resize(640, 520)
        self.jump_pair_cb = jump_pair_cb

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Pairs of nearby tracks whose next-frame assignment is cheaper when "
            "exchanged (likely a label swap with no jump or gap).\n"
            "★ = area continuity corroborates the swap. Double-click to jump to "
            "the frame with [A] and [B] preselected, then use Swap/cut or Local "
            "swap to fix it."))
        self.lst = QListWidget()
        for _, r in swaps_df.iterrows():
            star = "★ " if bool(r["area_corroborates"]) else "  "
            it = QListWidgetItem(
                f"{star}frame {int(r['frame'])}->{int(r['frame'])+1}  "
                f"{int(r['track_a'])} <-> {int(r['track_b'])}  "
                f"(sep {r['separation']:.0f}px, gain {r['cost_improvement']:.0f}px)")
            it.setData(Qt.UserRole,
                       (int(r["frame"]), int(r["track_a"]), int(r["track_b"])))
            self.lst.addItem(it)
        self.lst.itemDoubleClicked.connect(self._on_item)
        v.addWidget(self.lst)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        v.addWidget(btn_close)

    def _on_item(self, item):
        data = item.data(Qt.UserRole)
        if data:
            frame, a, b = data
            self.jump_pair_cb(int(frame), int(a), int(b))
