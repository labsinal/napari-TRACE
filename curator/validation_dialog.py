"""
ValidationDialog -- the Qt front-end for a random-sample validation pass.

Mirrors the navigable style of DiagnosticsDialog (double-click a row to jump to
that cell), but adds:

  * one sampled cell per row, with its aggregate score AND its per-characteristic
    sub-scores (area / mobility / lifetime / outcome),
  * a per-row "OK" checkbox so the user confirms each cell is correct (before or
    after editing it),
  * a transient, session-only log that lists every edit made to any sampled cell
    as it happens (the persistent audit.log.csv is written separately),
  * a "Show reliability plot" button (score vs error rate, global + per-trait),
    enabled once every sampled cell has been reviewed.

Pure Qt/presentation; all logic lives in validation.ValidationSession and the
matplotlib figure in validation.reliability_figure (passed in as plot_cb).
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QListWidget, QListWidgetItem, QHeaderView, QAbstractItemView,
)
from qtpy.QtGui import QColor, QBrush
from qtpy.QtCore import Qt


# Column layout of the cell table.
_COLS = ["Cell", "Frame", "Score", "Area", "Mobility", "Lifetime", "Outcome",
         "Why", "OK"]
_C_CELL, _C_FRAME, _C_SCORE, _C_AREA, _C_MOB, _C_LIFE, _C_OUT, _C_WHY, _C_OK = range(9)

# Columns that reveal the automatic's confidence. During a BLIND validation pass
# these stay hidden so the reviewer judges each cell on its own merits and can't
# be biased by the score it was given. They are revealed only once every sampled
# cell has been reviewed -- at which point the score/error relationship is shown
# honestly in the reliability plot. See the "Why" column too: its per-cell
# reasons are also score-derived, so it is masked during the blind pass.
_SCORE_COLS = (_C_SCORE, _C_AREA, _C_MOB, _C_LIFE, _C_OUT)

# Row background by status.
_STATUS_BG = {
    "pending":  None,
    "correct":  QColor(214, 245, 214),   # light green  (auto right, confirmed)
    "modified": QColor(250, 226, 196),    # amber        (edited, not confirmed)
    "fixed":    QColor(204, 229, 255),    # light blue   (edited + confirmed)
}


class ValidationDialog(QDialog):

    def __init__(self, parent, state, session, jump_cb, plot_cb):
        super().__init__(parent)
        self.state = state
        self.session = session
        self.jump_cb = jump_cb            # jump_cb(frame, track_id)
        self.plot_cb = plot_cb            # plot_cb(session) -> shows the figure
        self._updating = False            # guards programmatic checkbox changes
        self._row_of = {}                 # track_id -> table row

        self.setWindowTitle("Validation sample (spot-check the automatic)")
        self.resize(900, 720)
        v = QVBoxLayout(self)

        intro = QLabel(
            "Blind validation: the automatic score is HIDDEN while you review so "
            "your judgement isn't biased by it. Step through each sampled cell "
            "(double-click a row to jump to it). Edit any cell that is wrong — "
            "every edit is logged below — then tick OK. Tick OK directly if the "
            "cell was already correct. When all cells are reviewed, the scores "
            "are revealed and you can plot the score-vs-error reliability.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        # ---- cell table ----
        self.table = QTableWidget(self.session.n, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(_C_WHY, QHeaderView.Stretch)
        for c in (_C_CELL, _C_FRAME, _C_SCORE, _C_AREA, _C_MOB, _C_LIFE, _C_OUT, _C_OK):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._build_rows()
        self._set_score_columns_hidden(True)   # blind during the pass
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.table, stretch=3)

        # ---- transient edit log ----
        v.addWidget(QLabel("— Transient edit log (this session only) —"))
        self.log = QListWidget()
        self.log.setMaximumHeight(160)
        v.addWidget(self.log, stretch=1)

        # ---- footer ----
        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_next = QPushButton("Jump to next unchecked")
        self.btn_next.clicked.connect(self._jump_next_unchecked)
        row.addWidget(self.btn_next)
        self.btn_plot = QPushButton("Show reliability plot")
        self.btn_plot.clicked.connect(self._plot)
        row.addWidget(self.btn_plot)
        v.addLayout(row)

        self.refresh()
        self._jump_next_unchecked()   # start on the first cell

    # ------------------------------------------------------------------
    # Build / refresh
    # ------------------------------------------------------------------
    def _set(self, row, col, text, tid=None):
        it = QTableWidgetItem(text)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        if tid is not None:
            it.setData(Qt.UserRole, int(tid))
        self.table.setItem(row, col, it)
        return it

    def _build_rows(self):
        self._updating = True
        for row, tid in enumerate(self.session.ids):
            r = self.session.reviews[tid]
            self._row_of[tid] = row
            self._set(row, _C_CELL, str(tid), tid)
            self._set(row, _C_FRAME, str(r.first_frame), tid)
            self._set(row, _C_SCORE, f"{r.score:.2f}", tid)
            self._set(row, _C_AREA, f"{r.subscores.get('area', 1.0):.2f}", tid)
            self._set(row, _C_MOB, f"{r.subscores.get('mobility', 1.0):.2f}", tid)
            self._set(row, _C_LIFE, f"{r.subscores.get('lifetime', 1.0):.2f}", tid)
            self._set(row, _C_OUT, f"{r.subscores.get('outcome', 1.0):.2f}", tid)
            self._set(row, _C_WHY, r.outcome or "—", tid)

            ok = QTableWidgetItem()
            ok.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            ok.setCheckState(Qt.Checked if r.checked else Qt.Unchecked)
            ok.setData(Qt.UserRole, int(tid))
            ok.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, _C_OK, ok)
        self._updating = False

    def refresh(self):
        """Re-sync table colors, the 'Why' column, the log and the footer."""
        done = self.session.complete
        self._updating = True
        for tid, row in self._row_of.items():
            r = self.session.reviews[tid]
            bg = _STATUS_BG.get(r.status)
            brush = QBrush(bg) if bg is not None else QBrush()
            for c in range(len(_COLS)):
                it = self.table.item(row, c)
                if it is not None:
                    it.setBackground(brush)
            why = self.table.item(row, _C_WHY)
            if why is not None:
                # During the blind pass, do NOT reveal the score-derived reasons
                # (they would betray why the automatic was unsure). Only show the
                # neutral edit status. Once complete, reveal the full reasoning.
                if done:
                    tag = "; ".join(r.reasons) if r.reasons else (r.outcome or "—")
                else:
                    tag = "—"
                if r.modified:
                    tag = f"[EDITED {len(r.edits)}x] " + tag
                why.setText(tag)
            ok = self.table.item(row, _C_OK)
            if ok is not None:
                ok.setCheckState(Qt.Checked if r.checked else Qt.Unchecked)
        self._updating = False

        # Reveal the hidden score columns only once the blind pass is complete.
        self._set_score_columns_hidden(not done)

        # transient log (newest first)
        self.log.clear()
        for ts, tid, action, detail in self.session.edit_log_lines():
            self.log.addItem(QListWidgetItem(
                f"{ts.split('T')[-1]}  ·  cell {tid}  ·  {action}  ·  {detail}"))

        self.status.setText(self.session.summary_text())
        self.btn_plot.setEnabled(True)
        self.btn_plot.setText(
            "Show reliability plot" if done
            else f"Show reliability plot (partial {self.session.n_checked}/{self.session.n})")
        self.btn_next.setEnabled(not done)

    def _set_score_columns_hidden(self, hidden: bool):
        """Hide/show the score-revealing columns (the blind-test guard)."""
        for c in _SCORE_COLS:
            self.table.setColumnHidden(c, bool(hidden))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_double_click(self, row, col):
        if col == _C_OK:
            return
        it = self.table.item(row, _C_CELL)
        if it is None:
            return
        tid = int(it.data(Qt.UserRole))
        r = self.session.reviews.get(tid)
        if r is not None:
            self.jump_cb(int(r.first_frame), int(tid))

    def _on_item_changed(self, item):
        if self._updating or item.column() != _C_OK:
            return
        tid = item.data(Qt.UserRole)
        if tid is None:
            return
        self.session.set_checked(int(tid), item.checkState() == Qt.Checked)
        self.refresh()

    def _jump_next_unchecked(self):
        r = self.session.first_unchecked()
        if r is None:
            self.status.setText("All cells reviewed. " + self.session.summary_text())
            return
        row = self._row_of.get(r.track_id)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, _C_CELL))
        self.jump_cb(int(r.first_frame), int(r.track_id))

    def _plot(self):
        self.plot_cb(self.session)
