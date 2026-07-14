"""
Qt startup dialogs: column mapping and treatment setup.

Kept separate from the rest of the UI so they can be imported only after the
QApplication exists. No business logic here beyond gathering user choices.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog, QComboBox, QFormLayout, QDialogButtonBox, QRadioButton,
    QSpinBox, QVBoxLayout, QGroupBox, QButtonGroup, QLabel,
    QGridLayout, QLineEdit, QCheckBox,
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


CHANNEL_COLORS = ["green", "red", "blue", "magenta", "cyan", "yellow", "gray"]


def apply_channel_config(layers, config):
    """Apply a list of {name,color,measure} dicts onto ChannelLayer objects."""
    for layer, cfg in zip(layers, config or []):
        if cfg.get("name"):
            layer.name = str(cfg["name"])
        if cfg.get("color"):
            layer.colormap = str(cfg["color"])
            layer.color = str(cfg["color"])
        layer.measure = bool(cfg.get("measure", False))
    return layers


class ChannelConfigDialog(QDialog):
    """One row per extra channel: name, color, and a 'measure' checkbox."""

    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fluorescence channels")
        self._layers = layers
        self._rows = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Set each extra channel's display color and whether to measure "
            "features from it (intensity + texture per cell)."))
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Channel</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Color</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Measure</b>"), 0, 2)
        for i, L in enumerate(layers, start=1):
            name = QLineEdit(L.name)
            color = QComboBox(); color.addItems(CHANNEL_COLORS)
            if L.colormap in CHANNEL_COLORS:
                color.setCurrentText(L.colormap)
            measure = QCheckBox()
            grid.addWidget(name, i, 0)
            grid.addWidget(color, i, 1)
            grid.addWidget(measure, i, 2)
            self._rows.append((name, color, measure))
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        return [{"name": n.text(), "color": c.currentText(),
                 "measure": m.isChecked()} for n, c, m in self._rows]
