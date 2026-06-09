"""
Qt startup dialogs: column mapping and treatment setup.

Kept separate from the rest of the UI so they can be imported only after the
QApplication exists. No business logic here beyond gathering user choices.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog, QComboBox, QFormLayout, QDialogButtonBox, QRadioButton,
    QSpinBox, QVBoxLayout, QGroupBox, QButtonGroup, QLabel,
)

from . import config
from .config import COL_TRACK, COL_FRAME, COL_X, COL_Y, COLUMN_CANDIDATES, TREAT_TREATED, TREAT_CONTROL
from . import io_adapters


class ColumnMappingDialog(QDialog):
    """Map the CSV columns onto the internal schema."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map CSV columns")
        self.setMinimumWidth(420)
        self.columns = list(columns)

        layout = QFormLayout(self)
        labels = {
            COL_TRACK: "Track ID column:",
            COL_FRAME: "Frame / time column:",
            COL_X: "X (column / horizontal) coordinate:",
            COL_Y: "Y (row / vertical) coordinate:",
        }
        self.combos = {}
        guesses = io_adapters.default_mapping(self.columns)
        for target in COLUMN_CANDIDATES:
            combo = QComboBox()
            combo.addItems(self.columns)
            if guesses.get(target):
                combo.setCurrentText(guesses[target])
            layout.addRow(labels[target], combo)
            self.combos[target] = combo

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_mapping(self):
        return {target: combo.currentText() for target, combo in self.combos.items()}


class TreatmentDialog(QDialog):
    """Define the experiment-wide treatment phase."""

    def __init__(self, max_frame, config_dict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Treatment setup")
        self.setMinimumWidth(380)
        self.max_frame = int(max_frame)

        layout = QVBoxLayout(self)
        group = QGroupBox("Experiment type")
        gl = QVBoxLayout(group)
        self.radio_control = QRadioButton("Control (whole movie)")
        self.radio_treated = QRadioButton("Treated (define the treatment window)")
        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.radio_control)
        self.btn_group.addButton(self.radio_treated)
        gl.addWidget(self.radio_control)
        gl.addWidget(self.radio_treated)
        layout.addWidget(group)

        form = QFormLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, self.max_frame)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, self.max_frame)
        self.spin_end.setValue(self.max_frame)
        form.addRow("Treatment start frame:", self.spin_start)
        form.addRow("Treatment end frame (last = no washout):", self.spin_end)
        layout.addLayout(form)

        hint = QLabel("Frames < start = control | start..end = treated | > end = washout")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.radio_treated.toggled.connect(self._sync_enabled)
        if config_dict and config_dict.get("mode") == TREAT_TREATED:
            self.radio_treated.setChecked(True)
            self.spin_start.setValue(int(config_dict.get("start", 0)))
            self.spin_end.setValue(int(config_dict.get("end", self.max_frame)))
        else:
            self.radio_control.setChecked(True)
        self._sync_enabled()

    def _sync_enabled(self):
        treated = self.radio_treated.isChecked()
        self.spin_start.setEnabled(treated)
        self.spin_end.setEnabled(treated)

    def get_config(self):
        if self.radio_treated.isChecked():
            return {"mode": TREAT_TREATED, "start": int(self.spin_start.value()),
                    "end": int(self.spin_end.value())}
        return {"mode": TREAT_CONTROL, "start": 0, "end": -1}
