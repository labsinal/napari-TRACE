"""
Application entry point: Qt bootstrap, argument parsing and the load flow.

QApplication MUST exist before any QWidget is constructed -- including those
created implicitly by magicgui/napari during import. We create it at module
import time, before anything else touches Qt.
"""

from __future__ import annotations

import os
import sys
import argparse

import pandas as pd

# Bare Qt bootstrap FIRST.
from qtpy.QtWidgets import QApplication


def _ensure_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_qt_app = _ensure_qt_app()

# Safe to import Qt-dependent code only after the app exists.
from qtpy.QtWidgets import QFileDialog, QMessageBox, QDialog  # noqa: E402

from . import config  # noqa: E402
from .config import COL_FRAME, Thresholds  # noqa: E402
from . import data_io, io_adapters, treatment as treat  # noqa: E402
from .state import CuratorState  # noqa: E402
from .dialogs import ColumnMappingDialog, TreatmentDialog  # noqa: E402


def ask_file(title, folder, file_filter):
    start = folder if folder and os.path.isdir(folder) else ""
    path, _ = QFileDialog.getOpenFileName(None, title, start, file_filter)
    return path or None


def ask_directory(title, folder):
    start = folder if folder and os.path.isdir(folder) else ""
    return QFileDialog.getExistingDirectory(None, title, start) or None


def main():
    _ensure_qt_app()

    parser = argparse.ArgumentParser(description="Track curation tool (napari)")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Folder holding the CSV, mask and image")
    parser.add_argument("--csv", default=None, help="Explicit CSV/TXT path")
    parser.add_argument("--mask", default=None, help="Explicit mask .tif path")
    parser.add_argument("--image", default=None, help="Explicit image .tif path")
    parser.add_argument("--auto-thresholds", action="store_true",
                        help="(default on) Derive jump/length thresholds from the dataset")
    parser.add_argument("--no-auto-thresholds", action="store_true",
                        help="Use fixed default thresholds instead of data-derived ones")
    args = parser.parse_args()

    # -- working directory --
    source_folder = args.folder or ask_directory("Select experiment folder", "")
    if not source_folder:
        print("Loading cancelled: no folder selected.")
        return

    meta = data_io.read_meta(source_folder)
    if meta.get("source_folder"):
        work_dir, is_first = source_folder, False
        print(f"Reopening curated folder: {work_dir}")
    else:
        work_dir, is_first = data_io.setup_working_directory(source_folder)
        print(f"{'First run -- working copy created' if is_first else 'Reopening work folder'}: {work_dir}")

    # -- resolve inputs --
    csv_path, mask_path, image_path, mask_folder, image_folder = io_adapters.resolve_paths(work_dir)
    csv_path = args.csv or csv_path
    mask_path = args.mask or mask_path
    image_path = args.image or image_path

    if not csv_path or not os.path.exists(csv_path):
        csv_path = ask_file("Select tracking CSV/TXT", work_dir, "Tables (*.csv *.txt)")

    if not mask_path and not mask_folder:
        choice = QMessageBox.question(
            None, "Mask source",
            "Select a folder of individual TIF frames for the mask?\n"
            "(No = select a single TIF stack file)",
            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            mask_folder = ask_directory("Select mask frames folder", work_dir)
        else:
            mask_path = ask_file("Select label mask stack", work_dir, "TIFF (*.tif *.tiff)")

    if not image_path and not image_folder:
        choice = QMessageBox.question(
            None, "Image source",
            "Select a folder of individual TIF frames for the raw image?\n"
            "(No = select a single TIF stack file)",
            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            image_folder = ask_directory("Select image frames folder", work_dir)
        else:
            image_path = ask_file("Select raw image stack", work_dir, "TIFF (*.tif *.tiff)")

    if not csv_path:
        return print("Loading cancelled: no CSV selected.")
    if not (mask_path or mask_folder):
        return print("Loading cancelled: no mask selected.")
    if not (image_path or image_folder):
        return print("Loading cancelled: no image selected.")

    # -- column mapping (skip the dialog for headerless CTC .txt) --
    column_map = None
    if not csv_path.lower().endswith(".txt"):
        header = pd.read_csv(csv_path, nrows=0)
        existing_outcome, existing_parent = io_adapters.detect_existing_lineage(list(header.columns))
        if existing_outcome or existing_parent:
            print(f"Existing lineage detected -- outcome: '{existing_outcome}', "
                  f"parent: '{existing_parent}'. Preserving without reset.")
        dlg = ColumnMappingDialog(list(header.columns))
        if dlg.exec_() != QDialog.Accepted:
            return print("Loading cancelled at column mapping.")
        column_map = dlg.get_mapping()

    # -- load --
    try:
        df, masks, images, mask_path, image_path, warnings = data_io.load_data(
            csv_path, mask_path, image_path, column_map,
            mask_folder=mask_folder, image_folder=image_folder, work_dir=work_dir)
    except Exception as exc:
        return print(f"Failed to load files: {exc}")

    # -- image/mask coherence: warn (don't block) before going further --
    if warnings:
        box = QMessageBox(None)
        box.setWindowTitle("Image / mask mismatch")
        box.setIcon(QMessageBox.Warning)
        box.setText("The image and mask may not correspond:")
        box.setInformativeText("\n".join(warnings) +
                               "\n\nContinue loading anyway?")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec_() != QMessageBox.Yes:
            return print("Loading cancelled: image/mask mismatch.")

    # -- thresholds: data-derived by DEFAULT (the fixed 40px/20-frame values
    # mis-fire on higher-motility / shorter-track datasets). Pass
    # --no-auto-thresholds to force the fixed defaults.
    thresholds = Thresholds()
    if not args.no_auto_thresholds:
        thresholds = config.thresholds_from_dataset(df, base=thresholds)
        print(f"Auto thresholds (default): jump={thresholds.max_jump_px:.1f}px, "
              f"min_frames={thresholds.min_valid_frames}")
    else:
        print("Using fixed default thresholds (--no-auto-thresholds).")

    # -- treatment setup --
    max_frame = int(df[COL_FRAME].dropna().max()) if not df.empty else 0
    treatment_config = treat.default_config()
    tdlg = TreatmentDialog(max_frame)
    if tdlg.exec_() == QDialog.Accepted:
        treatment_config = tdlg.get_config()
    df = treat.apply_treatment(df, treatment_config)

    # -- build state and launch UI --
    state = CuratorState(df, masks)

    # Import the UI only now (after Qt + data are ready).
    from .ui import build_viewer
    build_viewer(state, images, csv_path, mask_path, work_dir,
                 treatment_config, thresholds, column_map)


if __name__ == "__main__":
    _ensure_qt_app()
    main()
