"""
napari UI assembly.

Builds the viewer, layers, the curation panel, the statistics panel (with the
customizable plots and the temporal-gradient scatter) and wires the navigable
diagnostics panel and the relink dialog.

All heavy logic is delegated to the tested core modules; this file is wiring
and presentation only. It is imported only inside the napari environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The curator runs inside napari's Qt event loop, so matplotlib figures should
# open in interactive Qt windows. Pin the Qt backend before any pyplot import
# below pulls in a (possibly non-interactive) default, so every stats/preview
# plot can actually be shown.
import matplotlib
try:
    matplotlib.use("QtAgg", force=True)
except Exception:
    pass

import napari
from magicgui.widgets import Container, PushButton, SpinBox, FloatSpinBox, Label, ComboBox, CheckBox
from napari.utils.notifications import show_info, show_error
from qtpy.QtWidgets import (QFileDialog, QMessageBox, QScrollArea, QDialog,
                            QWidget, QVBoxLayout, QGroupBox)

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_TREATMENT,
    OUTCOME_MITOSIS, OUTCOME_EXIT, OUTCOME_DEATH, OUTCOME_AMBIGUOUS,
    FILTER_ALL, FILTER_MITOSIS, FILTER_EXIT, FILTER_DEATH, FILTER_AMBIGUOUS,
    LAYER_RAW, LAYER_MASK, LAYER_TRACKS, LAYER_IDS, LAYER_XRAY,
    MAX_ID_LABELS, MAX_TRACK_VERTICES, SHORT_TAIL_LENGTH,
    Thresholds,
)
from . import analysis, lineage, stats, curation_ops as ops, treatment as treat
from . import fluorescence
from . import exports
from .audit import AuditLog
from .review import LineageReview
from .validated import CellValidation
from .triage_review import TriageReview
from .dialogs import TreatmentDialog
from .tree_plot import lineage_tree_figure
from .ui_panels import (DiagnosticsDialog, RelinkDialog, GapFillDialog,
                        TriageDialog, SwapDialog,
                        LineageEditorDialog, CompareCellDialog)
from .validation_dialog import ValidationDialog
from . import triage as triage_mod
from . import validation as validation_mod


def build_viewer(state, images, csv_path, mask_path, work_dir,
                 treatment_config, thresholds: Thresholds, column_map,
                 channel_layers=None):
    """Construct and run the napari viewer around a CuratorState."""
    audit = AuditLog(work_dir)
    review = LineageReview(work_dir)
    cellval = CellValidation(work_dir)
    triage_review = TriageReview(work_dir)
    max_frame = int(state.df[COL_FRAME].dropna().max()) if not state.df.empty else 0

    viewer = napari.Viewer()
    viewer.add_image(images, name=LAYER_RAW, colormap="gray", blending="additive")
    labels_layer = viewer.add_labels(state.mask, name=LAYER_MASK, opacity=0.5)
    state.attach_mask_layer(labels_layer)

    # Extra fluorescence channels: display-only image layers. They are added
    # ABOVE the raw movie with additive blending so a reporter (Caspase-3, FUCCI,
    # a viability dye) can be toggled on at the exact frame an anomaly is flagged.
    # They are intentionally NOT registered in state, the ID pool, update_visuals
    # or any save path -- they are read-only context, not segmentation. They
    # start hidden so they never obscure the curation view until asked for.
    # Normalize to a mutable list so the in-app "Add channel" button can append
    # to it and on_save / the measure export see the newly added channels.
    channel_layers = list(channel_layers or [])
    channel_layer_names = []
    for ch in channel_layers:
        try:
            lyr = viewer.add_image(ch.data, name=ch.name, colormap=ch.colormap,
                                   blending="additive", visible=False)
            channel_layer_names.append(lyr.name)
        except Exception as exc:
            show_error(f"Could not add channel '{ch.name}': {exc}")
    focus_active = {"on": False}
    # Lineage-focus view: like focus mode, but highlights the WHOLE family
    # (root ancestor + every descendant) of the selected cell. ``ids`` caches
    # the family set so update_visuals doesn't recompute it on every refresh.
    lineage_active = {"on": False, "ids": set(), "root": 0}
    # The colored tracks layer is the single most expensive thing to RENDER on
    # the real GPU for a dense movie (thousands of trajectories drawn every
    # frame). ``mode`` is "auto" (draw only when subset is small / focus),
    # "on" (always draw, heavy), or "off" (never draw).
    tracks_show = {"mode": "auto"}

    # Subscribers notified after every successful curation op, as
    # hook(action, ids, detail). Used by the validation pass to flag which
    # sampled cells were edited (i.e. where the automatic call was wrong).
    action_hooks = []

    # Accumulated interactive fluorescence tables (ERK-KTR / 53BP1). Each is a
    # per-(track,frame) DataFrame; on the next save they merge into features.
    extra_feature_tables = []

    def _image_layer_choices(widget=None):
        return [ly.name for ly in viewer.layers
                if ly.__class__.__name__ == "Image"]

    def _channel_stack_by_name(name):
        for ly in viewer.layers:
            if ly.name == name and ly.__class__.__name__ == "Image":
                return np.asarray(ly.data)
        return None

    # Background-ROI layer choices: "(auto)" plus any Labels/Shapes layer the user
    # draws to mark a genuinely cell-free region (Labels or Shapes).
    _ROI_AUTO = "(auto: non-cell median)"

    def _roi_layer_choices(widget=None):
        names = [ly.name for ly in viewer.layers
                 if ly.__class__.__name__ in ("Labels", "Shapes")
                 and ly.name != LAYER_MASK]
        return [_ROI_AUTO] + names

    def _roi_from_layer(name):
        """Boolean cell-free ROI (H x W or T x H x W) from a Labels/Shapes layer.

        Returns None for the "(auto)" choice so measure_intensity falls back to
        the non-cell-median background.
        """
        if not name or name == _ROI_AUTO:
            return None
        h, w = state.mask.shape[-2:]
        for ly in viewer.layers:
            if ly.name != name:
                continue
            cls = ly.__class__.__name__
            if cls == "Labels":
                return np.asarray(ly.data) > 0
            if cls == "Shapes":
                try:
                    masks = ly.to_masks(mask_shape=(h, w))
                    return (np.any(masks, axis=0) if len(masks)
                            else np.zeros((h, w), bool))
                except Exception as exc:
                    show_error(f"Could not read ROI from '{name}': {exc}")
                    return None
        return None

    def _exports_dir_and_base():
        """(exports_dir, csv_base_name), creating the exports dir if needed."""
        import os
        out_dir = os.path.join(work_dir, "exports")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, os.path.splitext(os.path.basename(csv_path))[0]

    # ------------------------------------------------------------------
    # Shared inputs
    # ------------------------------------------------------------------
    id_a_input = SpinBox(name="ID_A", label="ID [A] (target / mother):", value=0, max=2_000_000_000)
    id_b_input = SpinBox(name="ID_B", label="ID [B] (destination / daughter):", value=0, max=2_000_000_000)
    flag_filter = ComboBox(name="Filter", label="Flag x-ray:",
                           choices=[FILTER_ALL, FILTER_MITOSIS, FILTER_EXIT, FILTER_DEATH,
                                    FILTER_AMBIGUOUS])

    def current_frame():
        return int(viewer.dims.current_step[0])

    def clear_inputs():
        if not focus_active["on"] and not lineage_active["on"]:
            id_a_input.value = 0
            id_b_input.value = 0

    def finish(result, action="", ids=None):
        """Handle an OpResult: notify, refresh, audit, fan-out, clear."""
        if not result.ok:
            return show_error(result.message)
        update_visuals()
        show_info(result.message)
        if action:
            audit.record(action, ids=ids, detail=result.message)
            for hook in list(action_hooks):
                try:
                    hook(action, ids, result.message)
                except Exception:
                    pass
        clear_inputs()

    def jump_to(frame, track_id=None):
        viewer.dims.current_step = (int(frame), 0, 0)
        if track_id is not None:
            id_a_input.value = int(track_id)
            labels_layer.selected_label = int(track_id)

    def jump_and_focus(frame, track_id=None):
        """Jump to a cell AND enter single-cell focus mode on it.

        Used by the triage / validation dialogs so double-clicking a listed cell
        isolates it immediately (drops any lineage focus first).
        """
        jump_to(frame, track_id)
        if track_id is None or int(track_id) <= 0:
            return
        if lineage_active["on"]:
            lineage_active.update(on=False, ids=set(), root=0)
            btn_lineage_focus.text = "Lineage focus: OFF  [Shift+F]"
        focus_active["on"] = True
        btn_focus.text = f"Focus mode: ON (ID {int(track_id)})  [f]"
        labels_layer.selected_label = int(track_id)
        _update_reviewed_button()
        update_visuals()

    # ------------------------------------------------------------------
    # Visual engine
    # ------------------------------------------------------------------
    def update_visuals():
        df = state.df
        # Refresh the curation progress indicator.
        try:
            progress_label.value = "Progress: " + analysis.curation_progress(df)["text"]
        except Exception:
            pass
        if not df.empty:
            df[COL_OUTCOME] = df[COL_OUTCOME].fillna("").astype(str)
            df.loc[df[COL_OUTCOME] == "Start", COL_OUTCOME] = ""
            real = ~df[COL_OUTCOME].isin(["", "nan", "None"])
            if real.any():
                flag_map = df[real].groupby(COL_TRACK)[COL_OUTCOME].last()
                df[COL_OUTCOME] = df[COL_TRACK].map(flag_map).fillna("")
            else:
                df[COL_OUTCOME] = ""

        for name in (LAYER_IDS,):
            if name in viewer.layers:
                viewer.layers.remove(name)

        dv = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
        dv[COL_TRACK] = dv[COL_TRACK].astype(int)
        dv = dv[dv[COL_TRACK] > 0]

        if lineage_active["on"]:
            # Recompute the family each refresh so that links/cuts made while in
            # lineage mode are reflected immediately (cheap: O(tracks)).
            root = lineage_active.get("root", 0)
            if root:
                fam = lineage.family_of(df, root)
                lineage_active["ids"] = fam
            else:
                fam = lineage_active["ids"]
            dv = dv[dv[COL_TRACK].isin(fam)]
            # Hide every OTHER cell's mask too (like single-cell focus). A lineage
            # is many IDs, so show_selected_label (one label only) can't isolate
            # it; instead render an x-ray of just the family's labels and hide the
            # full mask layer.
            fam_ids = [int(t) for t in fam]
            labels_layer.show_selected_label = False
            labels_layer.visible = False
            xray = np.where(np.isin(labels_layer.data, fam_ids), labels_layer.data, 0)
            if LAYER_XRAY in viewer.layers:
                viewer.layers[LAYER_XRAY].data = xray
                viewer.layers[LAYER_XRAY].visible = True
            else:
                viewer.add_labels(xray, name=LAYER_XRAY, opacity=0.7)
        elif focus_active["on"]:
            dv = dv[dv[COL_TRACK] == id_a_input.value]
            labels_layer.show_selected_label = True
            labels_layer.visible = True
            if LAYER_XRAY in viewer.layers:
                viewer.layers[LAYER_XRAY].visible = False
        else:
            labels_layer.show_selected_label = False
            target = {FILTER_MITOSIS: OUTCOME_MITOSIS, FILTER_EXIT: OUTCOME_EXIT,
                      FILTER_DEATH: OUTCOME_DEATH,
                      FILTER_AMBIGUOUS: OUTCOME_AMBIGUOUS}.get(flag_filter.value)
            if target is not None:
                ids = list(df[df[COL_OUTCOME].str.lower() == target.lower()]
                           [COL_TRACK].dropna().astype(int).unique())
                dv = dv[dv[COL_TRACK].isin(ids)]
                labels_layer.visible = False
                xray = np.where(np.isin(labels_layer.data, ids), labels_layer.data, 0)
                if LAYER_XRAY in viewer.layers:
                    viewer.layers[LAYER_XRAY].data = xray
                    viewer.layers[LAYER_XRAY].visible = True
                else:
                    viewer.add_labels(xray, name=LAYER_XRAY, opacity=0.7)
            else:
                labels_layer.visible = True
                if LAYER_XRAY in viewer.layers:
                    viewer.layers[LAYER_XRAY].visible = False

        if dv.empty:
            if LAYER_TRACKS in viewer.layers:
                viewer.layers.remove(LAYER_TRACKS)
            return
        dv = dv.sort_values([COL_TRACK, COL_FRAME])
        track_data = dv[[COL_TRACK, COL_FRAME, COL_Y, COL_X]].values

        # The tracks layer is the heaviest thing to render for a dense movie.
        # mode "off" never draws it; "on" always draws it (heavy); "auto" draws
        # it for focus/lineage views or when the dataset is small enough.
        small_enough = len(track_data) <= MAX_TRACK_VERTICES
        mode = tracks_show["mode"]
        if mode == "off":
            want_tracks = False
        elif mode == "on":
            want_tracks = True
        else:  # auto
            want_tracks = focus_active["on"] or lineage_active["on"] or small_enough
        if want_tracks:
            # Build the lineage graph for the tracks layer. Vectorized: restrict
            # mother links to mothers that are actually present in the view.
            graph = {}
            if COL_PARENT in dv.columns:
                present_tracks = set(dv[COL_TRACK].to_numpy().tolist())
                daughters = dv[dv[COL_PARENT] > 0].groupby(COL_TRACK)[COL_PARENT].first()
                for d_id, m_id in daughters.items():
                    if int(m_id) in present_tracks:
                        graph[int(d_id)] = [int(m_id)]
            tail = SHORT_TAIL_LENGTH if not small_enough else 40
            if LAYER_TRACKS in viewer.layers:
                # Update in place so the layer never disappears or loses its slot
                # in the layer list (e.g. after exterminating a track).
                tl = viewer.layers[LAYER_TRACKS]
                tl.data = track_data
                tl.graph = graph
                tl.tail_length = tail
                tl.visible = True
            else:
                viewer.add_tracks(track_data, name=LAYER_TRACKS, tail_width=2,
                                  tail_length=tail,
                                  color_by="track_id", colormap="turbo", graph=graph)
        else:
            if LAYER_TRACKS in viewer.layers:
                viewer.layers.remove(LAYER_TRACKS)

        # ID labels. Drawing one text label per cell per frame is both illegible
        # and very slow once there are tens of thousands of them, which shows up
        # as napari hanging on a white screen. So when the full set would exceed
        # MAX_ID_LABELS we label only the cells in the current frame (the ones
        # the user is actually looking at). Focus / lineage / x-ray views are
        # already small, so they keep full labels. The tracks and masks are
        # always shown in full regardless.
        dlabel = dv
        thinned = False
        if not (focus_active["on"] or lineage_active["on"]) and len(dv) > MAX_ID_LABELS:
            cf = current_frame()
            dlabel = dv[dv[COL_FRAME] == cf]
            thinned = True

        if len(dlabel):
            pts = dlabel[[COL_FRAME, COL_Y, COL_X]].to_numpy()
            tid_arr = dlabel[COL_TRACK].to_numpy().astype(int)
            oc = dlabel[COL_OUTCOME].astype(str).to_numpy() if COL_OUTCOME in dlabel \
                else np.array([""] * len(dlabel))
            has_flag = ~np.isin(oc, ["", "nan", "None"])
            texts = np.where(
                has_flag,
                np.char.add(np.char.add(tid_arr.astype(str), " - "), oc),
                tid_arr.astype(str)).tolist()
            viewer.add_points(pts, name=LAYER_IDS,
                              properties={"label": texts},
                              text={"string": "{label}", "color": "yellow", "size": 12},
                              size=0.1, face_color="transparent", border_color="transparent")
            if thinned:
                try:
                    show_info(f"{len(dv):,} cell-frames: showing ID labels for the "
                              f"current frame only (paths/masks shown in full). "
                              f"Use Focus or Lineage focus for full labels on a subset.")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Undo / mouse / keys
    # ------------------------------------------------------------------
    def _refresh_frame_labels(event=None):
        """Redraw ID labels for the current frame only, in thinned (large) mode.

        Connected to the frame slider so that, on big datasets where labels are
        capped to the current frame, scrubbing keeps the visible IDs in sync
        without paying for a full update_visuals(). No-op for small datasets
        (where all labels are already shown) and for focus/lineage views.
        """
        if focus_active["on"] or lineage_active["on"]:
            return
        df = state.df
        dv = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y])
        if len(dv) <= MAX_ID_LABELS or LAYER_IDS not in viewer.layers:
            return
        cf = current_frame()
        sub = dv[dv[COL_FRAME] == cf]
        if sub.empty:
            return
        tid_arr = sub[COL_TRACK].astype(int).to_numpy()
        oc = sub[COL_OUTCOME].astype(str).to_numpy() if COL_OUTCOME in sub \
            else np.array([""] * len(sub))
        has_flag = ~np.isin(oc, ["", "nan", "None"])
        texts = np.where(
            has_flag,
            np.char.add(np.char.add(tid_arr.astype(str), " - "), oc),
            tid_arr.astype(str)).tolist()
        layer = viewer.layers[LAYER_IDS]
        layer.data = sub[[COL_FRAME, COL_Y, COL_X]].to_numpy()
        layer.properties = {"label": texts}

    try:
        viewer.dims.events.current_step.connect(_refresh_frame_labels)
    except Exception:
        pass

    def do_undo():
        if not state.can_undo():
            return show_info("Nothing to undo.")
        label = state.undo()
        update_visuals()
        show_info(f"Undone: {label} (Ctrl+Z).")

    @viewer.bind_key("Control-Z")
    def _undo_key(viewer):
        do_undo()

    @viewer.mouse_drag_callbacks.append
    def on_click(viewer, event):
        if "Shift" in event.modifiers or "Control" in event.modifiers:
            coords = tuple(int(np.round(p)) for p in event.position)
            d = labels_layer.data
            if (0 <= coords[0] < d.shape[0] and 0 <= coords[1] < d.shape[1]
                    and 0 <= coords[2] < d.shape[2]):
                cid = int(d[coords])
                if "Shift" in event.modifiers:
                    id_a_input.value = cid
                    show_info(f"ID [A]: {cid}")
                else:
                    id_b_input.value = cid
                    show_info(f"ID [B]: {cid}")

    # ==================================================================
    # CURATION WIDGETS
    # ==================================================================
    instruction = Label(value="Shift+click = [A] | Ctrl+click = [B]")
    progress_label = Label(value="Progress: -")
    lbl_setup = Label(value="--- SETUP ---")
    btn_treatment = PushButton(text="Set treatment / phases")
    lbl_view = Label(value="--- NAVIGATION & VIEW ---")
    btn_focus = PushButton(text="Focus mode: OFF  [f]")
    btn_tracks = PushButton(text="Show tracks layer: AUTO")
    btn_lineage_focus = PushButton(text="Lineage focus: OFF  [Shift+F]")
    lineage_progress = Label(value="Lineages reviewed: -")
    btn_lineage_reviewed = PushButton(text="Mark this lineage as reviewed")
    btn_next_unreviewed = PushButton(text="Jump to next unreviewed lineage  [.]")
    btn_skip_single = PushButton(text="Jump to next cell without lineage  [,]")
    btn_shuffle = PushButton(text="Shuffle colors")
    btn_undo = PushButton(text="Undo  [Ctrl+Z]")
    lbl_curation = Label(value="--- BASIC CURATION ---")
    btn_merge = PushButton(text="Merge (A becomes B across the movie)  [g]")
    btn_swap_future = PushButton(text="Swap / cut (from here onward)  [s]")
    btn_swap_local = PushButton(text="Local swap (this frame only)  [Shift+S]")
    btn_new_track = PushButton(text="Relabel mask (new ID)  [n]")
    btn_sync_frame = PushButton(text="Sync masks (this frame)  [y]")
    btn_sync_all = PushButton(text="Sync masks (whole movie)")
    btn_harmonize = PushButton(text="Harmonize colors")
    btn_delete = PushButton(text="Delete cell [A] (this frame)  [x]")
    btn_delete_track = PushButton(text="Exterminate track [A] (whole movie)")
    btn_rescue = PushButton(text="Rescue orphan masks (assign new ID)")
    btn_autotrack = PushButton(text="Auto-tracking (ID [A] only)")
    lbl_flags = Label(value="--- OUTCOME FLAGS (on ID [A]) ---")
    btn_flag_mitosis = PushButton(text="Mark: MITOSIS  [d]")
    btn_flag_exit = PushButton(text="Mark: EXIT (left the field)  [w]")
    btn_flag_death = PushButton(text="Mark: DEATH / SENESCENCE  [k]")
    btn_flag_ambiguous = PushButton(text="Mark: AMBIGUOUS (unsure)  [b]")
    btn_flag_clear = PushButton(text="Clear flags of ID [A]  [c]")
    lbl_lineage = Label(value="--- LINEAGE / GENEALOGY ---")
    btn_link_parent = PushButton(text="Link mother [A] -> daughter [B]  [l]")
    btn_lineage_editor = PushButton(text="Edit lineage of [A] (parent + daughters)")
    btn_cut_ghosts = PushButton(text="Cut post-mitosis ghosts")
    btn_tree_plot = PushButton(text="Show lineage tree plot (1.1, 1.2 labels)")
    tree_max_families = SpinBox(label="Tree: max families", value=60, min=1, max=100000)
    tree_include_singles = CheckBox(text="Tree: include single-cell families", value=False)
    btn_validate = PushButton(text="Validate lineage topology")
    lbl_diag = Label(value="--- DIAGNOSTICS & EXPORT ---")
    btn_diagnose = PushButton(text="Open diagnostics panel")
    btn_triage = PushButton(text="Open triage queue (large datasets)")
    btn_relink = PushButton(text="Assisted gap relinking")
    gap_method = ComboBox(name="GapMethod", label="Gap predictor:",
                          choices=["linear", "spline"], value="spline")
    btn_fill_gaps = PushButton(text="Fill gaps (interpolate trajectory)")
    btn_swaps = PushButton(text="Detect identity swaps (no jump)")
    btn_morph = PushButton(text="Detect segmentation errors (morphology)")
    btn_export_cell = PushButton(text="Export cell [A] video")
    btn_export_video = PushButton(text="Export presentation (screen)")
    btn_save = PushButton(text="SAVE ALL (backup + atomic)  [Ctrl+S]")

    # ---- setup ----
    @btn_treatment.clicked.connect
    def on_set_treatment():
        dlg = TreatmentDialog(max_frame, config_dict=treatment_config)
        if dlg.exec_() == QDialog.Accepted:
            state.save_state(frames=[], label="set treatment")
            treatment_config.clear()
            treatment_config.update(dlg.get_config())
            treat.apply_treatment(state.df, treatment_config)
            update_visuals()
            audit.record("set_treatment", detail=str(dict(treatment_config)))
            show_info("Treatment updated.")

    @flag_filter.changed.connect
    def on_filter(value):
        update_visuals()
        show_info("X-ray off." if value == FILTER_ALL else f"X-ray: {value}.")

    # ---- flags ----
    @btn_flag_mitosis.clicked.connect
    def _fm(): finish(ops.apply_flag(state, id_a_input.value, OUTCOME_MITOSIS), "flag_mitosis", [id_a_input.value])

    @btn_flag_exit.clicked.connect
    def _fe(): finish(ops.apply_flag(state, id_a_input.value, OUTCOME_EXIT), "flag_exit", [id_a_input.value])

    @btn_flag_death.clicked.connect
    def _fd(): finish(ops.apply_flag(state, id_a_input.value, OUTCOME_DEATH), "flag_death", [id_a_input.value])

    @btn_flag_ambiguous.clicked.connect
    def _fa(): finish(ops.apply_flag(state, id_a_input.value, OUTCOME_AMBIGUOUS), "flag_ambiguous", [id_a_input.value])

    @btn_flag_clear.clicked.connect
    def _fc(): finish(ops.clear_flag(state, id_a_input.value), "flag_clear", [id_a_input.value])

    # ---- lineage ----
    @btn_link_parent.clicked.connect
    def _link():
        a, b = id_a_input.value, id_b_input.value
        res = ops.link_parent(state, a, b)
        if getattr(res, "needs_confirm", False):
            confirm = QMessageBox.question(
                None, "Confirm lineage link", res.message,
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
            if confirm != QMessageBox.Ok:
                return show_info("Link cancelled.")
            res = ops.link_parent(state, a, b, force=True)
        finish(res, "link", [a, b])

    def _ed_link(mother, daughter):
        """Add/change a lineage link from the editor, honoring confirmation.

        Mirrors the main Link button: runs link_parent, shows the same OK/Cancel
        dialog when an override needs confirmation, then refreshes visuals and
        audits without clearing the [A]/[B] inputs. Returns (ok, message) for the
        editor's status line.
        """
        res = ops.link_parent(state, int(mother), int(daughter))
        if getattr(res, "needs_confirm", False):
            confirm = QMessageBox.question(
                None, "Confirm lineage link", res.message,
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
            if confirm != QMessageBox.Ok:
                return False, "Link cancelled."
            res = ops.link_parent(state, int(mother), int(daughter), force=True)
        if res.ok:
            update_visuals()
            audit.record("link", ids=[int(mother), int(daughter)], detail=res.message)
            _refresh_lineage_progress()
            _update_reviewed_button()
        else:
            show_error(res.message)
        return res.ok, res.message

    def _ed_unlink(child):
        """Detach a parent/daughter link from the editor; refresh + audit."""
        res = ops.unlink_parent(state, int(child))
        if res.ok:
            update_visuals()
            audit.record("unlink", ids=[int(child)], detail=res.message)
            _refresh_lineage_progress()
            _update_reviewed_button()
        else:
            show_error(res.message)
        return res.ok, res.message

    @btn_lineage_editor.clicked.connect
    def _edit_lineage():
        a = id_a_input.value
        if a in (0, None) or int(a) <= 0:
            return show_error("Put a cell ID in [A] first, then open the editor.")
        if state.df[state.df[COL_TRACK] == int(a)].empty:
            return show_error(f"Cell {int(a)} does not exist in the table.")
        dlg = LineageEditorDialog(None, state, int(a), _ed_link, _ed_unlink, jump_to)
        open_dialogs["lineage_editor"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    @btn_validate.clicked.connect
    def _validate():
        v = lineage.validate_lineage(state.df)
        if v.is_empty():
            return show_info("Lineage topology OK: no violations.")
        for m in v.messages():
            print("LINEAGE VIOLATION:", m)
        show_error(f"{len(v.messages())} lineage violation(s); see console / diagnostics panel.")

    @btn_cut_ghosts.clicked.connect
    def _ghosts(): finish(ops.cut_ghosts(state), "cut_ghosts")

    @btn_tree_plot.clicked.connect
    def _tree():
        fig = lineage_tree_figure(
            state.df,
            max_families=int(tree_max_families.value),
            include_singles=bool(tree_include_singles.value))
        if fig is None:
            return show_error("No lineage or flagged cells to plot.")
        stats.show(fig)

    @btn_skip_single.clicked.connect
    def _skip():
        df = state.df
        if df.empty:
            return show_info("Dataset empty.")
        groups = lineage.classify_lineage(df)
        pending = df[~df[COL_TRACK].isin(groups.all_nodes)]
        if pending.empty:
            return show_info("All cells curated or flagged.")
        frames = np.sort(pending[COL_FRAME].unique())
        ahead = frames[frames > current_frame()]
        if len(ahead) == 0:
            return show_info("No pending cells ahead.")
        nf = int(ahead[0])
        tid = int(pending[pending[COL_FRAME] == nf][COL_TRACK].iloc[0])
        jump_to(nf, tid)
        show_info(f"Frame {nf}; cell {tid} uncurated.")

    @btn_tracks.clicked.connect
    def _toggle_tracks():
        # Cycle AUTO -> ON -> OFF -> AUTO. AUTO draws tracks only when the view
        # is small (focus/lineage or small dataset); ON forces them (heavy on
        # big movies); OFF never draws them.
        order = {"auto": "on", "on": "off", "off": "auto"}
        tracks_show["mode"] = order[tracks_show["mode"]]
        btn_tracks.text = f"Show tracks layer: {tracks_show['mode'].upper()}"
        if tracks_show["mode"] == "off" and LAYER_TRACKS in viewer.layers:
            viewer.layers.remove(LAYER_TRACKS)
        update_visuals()

    @btn_focus.clicked.connect
    def _focus():
        a = id_a_input.value
        if not focus_active["on"]:
            if a == 0:
                return show_error("Select ID [A] to focus.")
            focus_active["on"] = True
            btn_focus.text = f"Focus mode: ON (ID {a})  [f]"
            labels_layer.selected_label = a
        else:
            focus_active["on"] = False
            btn_focus.text = "Focus mode: OFF  [f]"
        update_visuals()

    @id_a_input.changed.connect
    def _id_a_changed(value):
        if focus_active["on"] and value > 0:
            btn_focus.text = f"Focus mode: ON (ID {value})  [f]"
            labels_layer.selected_label = value
            update_visuals()
        # Keep the "mark reviewed" button label in sync with the selected cell's
        # lineage, even when neither focus mode is active.
        if not lineage_active["on"]:
            _update_reviewed_button()

    # ---- lineage focus + review tracking ----
    def _refresh_lineage_progress():
        try:
            total = len(lineage.lineage_roots(state.df))
            lineage_progress.value = "Lineages reviewed: " + review.progress_text(total)
        except Exception:
            lineage_progress.value = "Lineages reviewed: -"

    def _current_root():
        """Root ID of the lineage the selected cell [A] belongs to (0 if none)."""
        a = id_a_input.value
        if a <= 0 or state.df.empty:
            return 0
        try:
            return int(lineage.root_of(state.df, int(a)))
        except Exception:
            return 0

    def _update_reviewed_button():
        root = lineage_active["root"] if lineage_active["on"] else _current_root()
        if root and review.is_reviewed(root):
            btn_lineage_reviewed.text = f"Unmark lineage (root {root}) reviewed ✓"
        elif root:
            btn_lineage_reviewed.text = f"Mark lineage (root {root}) as reviewed"
        else:
            btn_lineage_reviewed.text = "Mark this lineage as reviewed"

    @btn_lineage_focus.clicked.connect
    def _lineage_focus():
        if not lineage_active["on"]:
            a = id_a_input.value
            if a == 0:
                return show_error("Select ID [A] to focus its lineage.")
            fam = lineage.family_of(state.df, int(a))
            root = int(lineage.root_of(state.df, int(a)))
            lineage_active.update(on=True, ids=fam, root=root)
            # Turn off single-cell focus so the two modes don't fight.
            if focus_active["on"]:
                focus_active["on"] = False
                btn_focus.text = "Focus mode: OFF  [f]"
            n = len(fam)
            btn_lineage_focus.text = f"Lineage focus: ON (root {root}, {n} cells)  [Shift+F]"
            show_info(f"Lineage of {a}: root {root}, {n} cell(s) "
                      f"{'✓ reviewed' if review.is_reviewed(root) else 'not yet reviewed'}.")
        else:
            lineage_active.update(on=False, ids=set(), root=0)
            btn_lineage_focus.text = "Lineage focus: OFF  [Shift+F]"
        _update_reviewed_button()
        update_visuals()

    @btn_lineage_reviewed.clicked.connect
    def _toggle_reviewed():
        root = lineage_active["root"] if lineage_active["on"] else _current_root()
        if not root:
            return show_error("Select a cell (ID [A]) whose lineage to mark.")
        now_reviewed = review.toggle(root)
        audit.record("lineage_reviewed" if now_reviewed else "lineage_unreviewed",
                     ids=[root])
        _update_reviewed_button()
        _refresh_lineage_progress()
        show_info(f"Lineage root {root} marked "
                  f"{'reviewed ✓' if now_reviewed else 'NOT reviewed'} (saved).")

    @btn_next_unreviewed.clicked.connect
    def _next_unreviewed():
        df = state.df
        if df.empty:
            return show_info("Dataset empty.")
        roots = lineage.lineage_roots(df)
        pending = {r for r in roots if not review.is_reviewed(r)}
        if not pending:
            return show_info("All lineages reviewed. 🎉")
        # Map every track to its lineage root once (cheap, memoized), then find
        # the earliest-appearing cell whose root is still pending. This avoids
        # rebuilding each family in a loop (which is slow with many lineages).
        dvalid = df.dropna(subset=[COL_TRACK, COL_FRAME])
        if dvalid.empty:
            return show_info("No cells to navigate.")
        p_of = lineage.parent_of(df)
        cache = {}
        # First frame per track, sorted so the earliest comes first.
        firsts = (dvalid.groupby(dvalid[COL_TRACK].astype(int))[COL_FRAME]
                  .min().astype(int).sort_values())
        best = None
        for tid, f0 in firsts.items():
            root = lineage._root_in_map(p_of, int(tid), cache)
            if root in pending:
                best = (int(f0), root, int(tid))
                break  # firsts is sorted ascending, so this is the earliest
        if best is None:
            return show_info("No pending lineage with visible cells.")
        f0, root, tid0 = best
        id_a_input.value = tid0
        if lineage_active["on"]:
            lineage_active.update(ids=lineage.family_of(df, root), root=root)
            btn_lineage_focus.text = (f"Lineage focus: ON (root {root}, "
                                      f"{len(lineage_active['ids'])} cells)  [Shift+F]")
        jump_to(f0, tid0)
        _update_reviewed_button()
        show_info(f"Next unreviewed lineage: root {root} (cell {tid0}, frame {f0}). "
                  f"{len(pending)} pending.")

    # ---- basic curation ----
    @btn_undo.clicked.connect
    def _u():
        do_undo(); clear_inputs()

    @btn_shuffle.clicked.connect
    def _shuffle():
        import random
        try:
            labels_layer.new_colormap()
        except Exception:
            try:
                labels_layer.seed = random.random()
            except Exception:
                pass
        show_info("Colors shuffled.")

    @btn_merge.clicked.connect
    def _merge():
        finish(ops.merge(state, id_a_input.value, id_b_input.value),
               "merge", [id_a_input.value, id_b_input.value])

    @btn_swap_future.clicked.connect
    def _sf():
        finish(ops.swap_future(state, id_a_input.value, id_b_input.value, current_frame()),
               "swap_future", [id_a_input.value, id_b_input.value])

    @btn_swap_local.clicked.connect
    def _sl():
        finish(ops.swap_local(state, id_a_input.value, id_b_input.value, current_frame()),
               "swap_local", [id_a_input.value, id_b_input.value])

    @btn_new_track.clicked.connect
    def _nt(): finish(ops.relabel_new(state, id_a_input.value), "relabel", [id_a_input.value])

    @btn_sync_frame.clicked.connect
    def _syncf(): finish(ops.sync_masks(state, [current_frame()], treatment_config), "sync_frame")

    @btn_sync_all.clicked.connect
    def _synca():
        show_info("Recomputing centroids across all frames...")
        finish(ops.sync_masks(state, range(state.mask.shape[0]), treatment_config), "sync_all")

    @btn_harmonize.clicked.connect
    def _harm(): finish(ops.harmonize(state, treatment_config), "harmonize")

    @btn_delete.clicked.connect
    def _del():
        finish(ops.delete_here(state, id_a_input.value, current_frame()),
               "delete_frame", [id_a_input.value])

    @viewer.bind_key("x", overwrite=True)
    @viewer.bind_key("X", overwrite=True)
    def _del_key(viewer):
        finish(ops.delete_here(state, id_a_input.value, current_frame()),
               "delete_frame", [id_a_input.value])

    @btn_delete_track.clicked.connect
    def _delt(): finish(ops.delete_track(state, id_a_input.value), "delete_track", [id_a_input.value])

    @btn_rescue.clicked.connect
    def _rescue(): finish(ops.rescue_orphans(state, treatment_config), "rescue")

    @btn_autotrack.clicked.connect
    def _autotrack():
        if id_a_input.value == 0:
            return show_error("Select ID [A].")
        show_info(f"Auto-tracking with conflict resolution for ID {id_a_input.value}...")
        finish(ops.autotrack(state, id_a_input.value, treatment_config),
               "autotrack", [id_a_input.value])

    # ---- diagnostics ----
    # Keep references to open dialogs so they are not garbage-collected
    # (a non-parented QDialog with no Python reference can vanish immediately).
    open_dialogs = {}

    def _open_diagnostics(initial_tab=0):
        rep = analysis.detect_anomalies(state.df, thresholds)
        summary = analysis.compute_track_summary(
            state.df, state.mask, pixel_size_input.value, frame_interval_input.value)
        dlg = DiagnosticsDialog(None, state, rep, summary, thresholds, jump_to,
                                initial_tab=initial_tab)
        open_dialogs["diag"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    @btn_diagnose.clicked.connect
    def _diag():
        _open_diagnostics(initial_tab=0)

    def _open_triage():
        res = triage_mod.triage_queue(state.df, thresholds, mask=state.mask,
                                      cutoff=0.85)
        if not res.scores:
            return show_info("No cells to triage.")

        def _accept(ids):
            # Non-destructive: log the bulk-accept so it is traceable, without
            # fabricating outcomes. The value of accept is that these cells leave
            # the review queue; their existing (auto/CSV) state is kept.
            audit.record("bulk_accept", ids=list(ids),
                         detail=f"Accepted {len(ids)} confident cells via triage.")
            show_info(f"Bulk-accepted {len(ids)} cells (logged in the audit). "
                      f"Draw a validation sample to confirm the error rate.")
            return len(ids)

        def _sample(accept_ids):
            sample_ids = triage_mod.validation_sample(accept_ids, n=50)
            if not sample_ids:
                return show_info("Nothing accepted to sample.")

            by_id = res.score_by_id()
            scored = [by_id[i] for i in sample_ids if i in by_id]
            audit.record("validation_sample", ids=sample_ids,
                         detail=f"Random validation sample of {len(sample_ids)} cells.")

            session = validation_mod.ValidationSession(scored, n_total=len(res.accept))

            def _plot(sess):
                stats.show(validation_mod.reliability_figure(sess))

            dlg = ValidationDialog(None, state, session, jump_and_focus, _plot,
                                   validate_cb=lambda tid, ok: cellval.set(tid, ok))
            open_dialogs["validation"] = dlg

            def _hook(action, ids, detail):
                if session.note_action(action, ids, detail):
                    dlg.refresh()
            action_hooks.append(_hook)
            # Detach the hook when the dialog closes so it can't leak / double-fire.
            dlg.finished.connect(
                lambda *_: action_hooks.remove(_hook) if _hook in action_hooks else None)

            dlg.show(); dlg.raise_(); dlg.activateWindow()
            show_info(f"Validation sample of {len(sample_ids)} cells drawn. "
                      f"Step through them one by one, fix what's wrong, tick OK, "
                      f"then plot the reliability.")

        dlg = TriageDialog(None, state, res, jump_and_focus, _accept, _sample,
                           triage_review=triage_review)
        open_dialogs["triage"] = dlg
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    @btn_triage.clicked.connect
    def _triage():
        _open_triage()

    @btn_morph.clicked.connect
    def _morph():
        me = analysis.detect_morphology_errors(state.mask, thresholds)
        if me.empty:
            return show_info("No morphology anomalies found.")
        show_info(f"{len(me)} morphology anomalies found; opening the diagnostics panel.")
        _open_diagnostics(initial_tab=2)  # Morphology tab

    @btn_relink.clicked.connect
    def _relink():
        method = gap_method.value
        sugs = lineage.suggest_relinks(state.df, thresholds, method=method)
        if not sugs:
            return show_info("No gap-relink suggestions found.")

        def approve(s):
            finish(ops.apply_relink(state, s), "relink", [s.track_id, s.candidate_id])

        dlg = RelinkDialog(None, state, sugs, jump_to, approve)
        open_dialogs["relink"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    @btn_fill_gaps.clicked.connect
    def _fill_gaps():
        method = gap_method.value
        fills = lineage.suggest_gap_fills(state.df, thresholds, method=method)
        if not fills:
            return show_info("No fillable gaps found (within the length limit).")

        def approve(fl):
            finish(ops.fill_gap(state, fl, treatment_config),
                   "fill_gap", [fl.track_id])

        dlg = GapFillDialog(None, state, fills, jump_to, approve)
        open_dialogs["gapfill"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def jump_pair(frame, a, b):
        """Jump to a frame and load a pair of tracks into [A] and [B]."""
        id_a_input.value = int(a)
        id_b_input.value = int(b)
        viewer.dims.current_step = (int(frame), 0, 0)
        labels_layer.selected_label = int(a)
        show_info(f"Frame {frame}: [A]={a}, [B]={b}. Use Swap/cut (s) or "
                  f"Local swap (Shift+S) to fix.")

    @btn_swaps.clicked.connect
    def _swaps():
        sw = analysis.detect_identity_swaps(state.df, thresholds)
        if sw.empty:
            return show_info("No suspected identity swaps found.")
        n_corr = int(sw["area_corroborates"].sum())
        show_info(f"{len(sw)} suspected swap(s); {n_corr} corroborated by area.")
        dlg = SwapDialog(None, sw, jump_pair)
        open_dialogs["swaps"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ---- export ----
    @btn_export_cell.clicked.connect
    def _exp_cell():
        try:
            import imageio
        except ImportError:
            return show_error("Install imageio: pip install imageio[ffmpeg]")
        a = id_a_input.value
        if a == 0:
            return show_error("Select the ID in box [A].")
        cell = state.df[state.df[COL_TRACK] == a].sort_values(COL_FRAME)
        if cell.empty:
            return show_error(f"ID {a} has no frames.")
        path, _ = QFileDialog.getSaveFileName(
            None, "Save cell video", f"cell_{a}.mp4", "MP4 Files (*.mp4)")
        if not path:
            return
        show_info(f"Recording cell {a}...")
        window = 50
        frames = []
        for _, row in cell.iterrows():
            f = int(row[COL_FRAME]); cx, cy = int(row[COL_X]), int(row[COL_Y])
            img = images[f]; mask = state.mask[f]
            y1, y2 = max(0, cy - window), min(img.shape[0], cy + window)
            x1, x2 = max(0, cx - window), min(img.shape[1], cx + window)
            ci = img[y1:y2, x1:x2]; cm_ = mask[y1:y2, x1:x2]
            lo, hi = ci.min(), ci.max()
            norm = ((ci - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(ci, np.uint8)
            rgb = np.stack([norm, norm, norm], axis=-1)
            rgb[cm_ == a] = [0, 255, 255]
            frames.append(rgb)
        imageio.mimsave(path, frames, fps=10)
        show_info(f"Video saved: {path}")

    @btn_export_video.clicked.connect
    def _exp_video():
        try:
            import imageio
        except ImportError:
            return show_error("Install imageio: pip install imageio[ffmpeg]")
        path, _ = QFileDialog.getSaveFileName(
            None, "Export presentation", "presentation.mp4", "MP4 Files (*.mp4)")
        if not path:
            return
        show_info("Recording canvas; do not touch the mouse...")
        frames = []
        original = viewer.dims.current_step
        for f in range(images.shape[0]):
            viewer.dims.current_step = (f, 0, 0)
            frames.append(viewer.screenshot(canvas_only=True, flash=False))
        viewer.dims.current_step = original
        imageio.mimsave(path, frames, fps=10)
        show_info("Presentation exported.")

    # ---- save (backup + unified anomaly filter) ----
    @btn_save.clicked.connect
    def on_save():
        from . import data_io
        df_clean = state.df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
        df_clean[COL_TRACK] = df_clean[COL_TRACK].astype(int)
        df_clean = df_clean[df_clean[COL_TRACK] > 0]

        rep = analysis.detect_anomalies(df_clean, thresholds, include_outcome_checks=False)
        flagged = rep.flagged_ids
        if flagged:
            msg = QMessageBox(None)
            msg.setWindowTitle("Pre-save quality filter")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Detected {len(flagged)} cells with anomalous tracking "
                        f"(short < {thresholds.min_valid_frames} frames, gaps, or "
                        f"jumps > {thresholds.max_jump_px:.0f}px).")
            msg.setInformativeText("What do you want to do with these tracks?")
            b_clean = msg.addButton("Remove and save clean", QMessageBox.AcceptRole)
            msg.addButton("Ignore and save with errors", QMessageBox.RejectRole)
            b_cancel = msg.addButton("Cancel save", QMessageBox.DestructiveRole)
            msg.exec_()
            choice = msg.clickedButton()
            if choice == b_cancel:
                return show_info("Save cancelled.")
            if choice == b_clean:
                df_clean = df_clean[~df_clean[COL_TRACK].isin(flagged)]
                for tid in flagged:
                    state.mask[state.mask == tid] = 0
                labels_layer.refresh()
                show_info(f"{len(flagged)} defective cells removed.")

        dup = df_clean[df_clean.duplicated(subset=[COL_FRAME, COL_TRACK], keep=False)]
        if not dup.empty:
            print("\nERROR REPORT: DUPLICATE CELLS")
            for fr, g in dup.groupby(COL_FRAME):
                ids = ", ".join(str(int(i)) for i in g[COL_TRACK].unique())
                print(f"  Frame {int(fr)}: IDs [{ids}]")
            return show_error("Could not save: duplicate IDs. Check the console.")

        show_info("Backing up and saving atomically...")
        df_clean[COL_FRAME] = df_clean[COL_FRAME].astype(int)
        if COL_PARENT in df_clean.columns:
            df_clean[COL_PARENT] = df_clean[COL_PARENT].fillna(-1).astype(int)
        # Refresh the border-contact flag against the final mask so the saved
        # CSV reflects the curated state, not the state at load time.
        df_clean = analysis.annotate_border_contact(df_clean, state.mask)
        # Refresh the per (frame, track) mask area (pixels) against the final mask.
        df_clean = analysis.annotate_area(df_clean, state.mask)
        state.df = df_clean

        try:
            bdir = data_io.save_session(state.df, state.mask, csv_path, mask_path, work_dir)
        except Exception as exc:
            return show_error(f"Save failed (originals untouched): {exc}")

        audit.record("save", detail=f"backup={bdir}")

        # Reload from disk to guarantee integrity and refresh the view.
        # IMPORTANT: the CSV we just wrote is ALREADY in the internal schema
        # (track_id, pos_x, ...), so it must be read back as-is. Re-applying the
        # original column_map here would look for the source file's column names
        # (which no longer exist) and drop track_id, breaking update_visuals.
        current = viewer.dims.current_step
        reloaded = pd.read_csv(csv_path)
        # Coerce the core numeric columns and restore defaults defensively.
        for col in (COL_TRACK, COL_FRAME, COL_X, COL_Y):
            if col in reloaded.columns:
                reloaded[col] = pd.to_numeric(reloaded[col], errors="coerce")
        if COL_OUTCOME not in reloaded.columns:
            reloaded[COL_OUTCOME] = ""
        reloaded[COL_OUTCOME] = reloaded[COL_OUTCOME].fillna("").astype(str)
        if COL_PARENT not in reloaded.columns:
            reloaded[COL_PARENT] = -1
        reloaded[COL_PARENT] = pd.to_numeric(
            reloaded[COL_PARENT], errors="coerce").fillna(-1).astype(int)
        state.df = reloaded
        import tifffile
        state.mask = tifffile.imread(mask_path).astype(np.uint32)
        labels_layer.data = state.mask
        state.clear_history()
        update_visuals()
        viewer.dims.current_step = current
        show_info(f"Saved. Backup at: {bdir}")

        # Derived export tables (full + validated copies). Best-effort: the core
        # save already succeeded, so a failure here never loses curation work.
        try:
            import os
            existing = set(state.df[COL_TRACK].dropna().astype(int).unique())
            cellval.prune(existing)
            triage_review.prune(existing)
            review.prune(lineage.lineage_roots(state.df))
            base = os.path.splitext(os.path.basename(csv_path))[0]
            marked = {L.name: np.asarray(L.data) for L in (channel_layers or [])
                      if getattr(L, "measure", False)}
            res = exports.write_all(
                work_dir, state.df, state.mask,
                reviewed_roots=review.reviewed_roots(),
                validated_cells=cellval.ids(),
                base_name=base, window=int(export_window.value),
                pixel_size=pixel_size_input.value,
                frame_interval=frame_interval_input.value,
                channels=marked or None,
                ring_opts={"dilation": int(ring_dilation.value),
                           "gap": int(ring_gap.value)},
                extra_tables=list(extra_feature_tables) or None)
            audit.record("export_tables",
                         detail=f"{len(res['files'])} tables, "
                                f"{res['n_validated_tracks']} validated tracks -> {res['dir']}")
            show_info(f"Exported {len(res['files'])} tables "
                      f"({res['n_validated_tracks']} validated tracks) to {res['dir']}.")
        except Exception as exc:
            show_error(f"Main save OK, but export tables failed: {exc}")

        clear_inputs()

    # ==================================================================
    # STATISTICS WIDGETS
    # ==================================================================
    lbl_stats = Label(value="--- QUICK STATISTICS ---")
    pixel_size_input = FloatSpinBox(label="Pixel size (um/px):", value=1.0, min=0.0001, step=0.05)
    frame_interval_input = FloatSpinBox(label="Frame interval (min/frame):", value=1.0, min=0.0001, step=0.5)
    exclude_border_cb = CheckBox(label="Exclude border-touching points", value=False)
    exclude_interp_cb = CheckBox(label="Exclude interpolated (gap-filled) frames", value=False)
    export_window = SpinBox(label="Window for accumulated export:", value=7, min=2, max=1000)

    # --- Fluorescence: ring params, preview, ERK-KTR, 53BP1 ---
    ring_dilation = SpinBox(label="Cytoplasm ring width (px):", value=2, min=1, max=20)
    ring_gap = SpinBox(label="Ring gap from nucleus (px):", value=1, min=0, max=10)
    # Two explicit channel roles so both can be set at once: the cytoplasm/reporter
    # channel drives ERK-KTR C/N and the ring preview; the nucleus channel drives
    # the 53BP1 nuclear-texture measure. Each button uses the right one.
    cyto_channel = ComboBox(label="Cytoplasm channel (ERK-KTR):",
                            choices=_image_layer_choices)
    nuc_channel = ComboBox(label="Nucleus channel (53BP1):",
                           choices=_image_layer_choices)
    bg_roi_layer = ComboBox(label="Background ROI (cell-free):", choices=_roi_layer_choices)
    btn_add_channel = PushButton(text="Add fluorescence channel...")
    btn_ring_preview = PushButton(text="Preview ring on a random nucleus")

    # Keep the channel/ROI dropdowns in sync as the user adds/removes layers
    # (e.g. draws a Shapes layer for the background ROI, or adds a channel).
    def _refresh_layer_combos(event=None):
        for cb in (cyto_channel, nuc_channel, bg_roi_layer):
            try:
                cb.reset_choices()
            except Exception:
                pass
    viewer.layers.events.inserted.connect(_refresh_layer_combos)
    viewer.layers.events.removed.connect(_refresh_layer_combos)

    def _add_channel():
        """Load an extra channel (stack file OR per-frame folder), persist it into
        the working copy, tag it, add it as a layer, and record it for reuse."""
        import os
        import shutil
        import tifffile
        from .channels import ChannelLayer
        from .dialogs import ChannelConfigDialog, apply_channel_config
        from . import io_adapters, data_io as _dio
        choice = QMessageBox.question(
            None, "Channel source",
            "Select a folder of individual TIF frames?\n"
            "(No = select a single TIF stack file)",
            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            path = QFileDialog.getExistingDirectory(
                None, "Select channel frames folder", work_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Select channel TIF stack", work_dir, "TIFF (*.tif *.tiff)")
        if not path:
            return
        base = os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
        try:
            if os.path.isdir(path):
                saved_file = base + "_stack.tif"
                dest = os.path.join(work_dir, saved_file)
                arr = io_adapters.build_stack_from_folder(path, dest)
            else:
                saved_file = os.path.basename(path)
                dest = os.path.join(work_dir, saved_file)
                if os.path.abspath(path) != os.path.abspath(dest):
                    shutil.copy2(path, dest)
                arr = tifffile.imread(dest)
        except Exception as exc:
            return show_error(f"Could not load channel: {exc}")

        n_frames = state.mask.shape[0]
        ch_frames = arr.shape[0] if getattr(arr, "ndim", 0) >= 3 else 1
        if ch_frames != n_frames:
            show_error(f"Note: channel has {ch_frames} frames, movie has "
                       f"{n_frames}; overlay/measures may be misaligned.")

        layer_obj = ChannelLayer(name=base, data=arr, colormap="green")
        dlg = ChannelConfigDialog([layer_obj])
        if dlg.exec_() == QDialog.Accepted:
            apply_channel_config([layer_obj], dlg.get_config())
        try:
            viewer.add_image(layer_obj.data, name=layer_obj.name,
                             colormap=layer_obj.colormap, blending="additive",
                             visible=True)
        except Exception as exc:
            return show_error(f"Could not add channel layer: {exc}")
        channel_layers.append(layer_obj)
        _dio.add_channel_to_meta(work_dir, layer_obj.name, saved_file,
                                 color=layer_obj.colormap, measure=layer_obj.measure)
        _refresh_layer_combos()
        show_info(f"Channel '{layer_obj.name}' added and saved to the working "
                  f"folder (reloads automatically next session).")
    btn_add_channel.clicked.connect(_add_channel)

    def _ring_preview():
        # Preview the ring geometry over the cytoplasm channel (fallback: nucleus).
        name = cyto_channel.value or nuc_channel.value
        stack = _channel_stack_by_name(name)
        try:
            fig = fluorescence.ring_preview_figure(
                state.mask, stack, track_id=None,
                dilations=tuple(range(1, int(ring_dilation.value) + 1)) or (1,),
                gap=int(ring_gap.value))
            stats.show(fig)
        except Exception as exc:
            show_error(f"Ring preview failed: {exc}")
    btn_ring_preview.clicked.connect(_ring_preview)

    btn_erk = PushButton(text="Compute ERK-KTR C/N (cytoplasm channel -> features)")
    btn_53bp1 = PushButton(text="Measure 53BP1 nuclear texture (nucleus channel -> features)")

    def _erk_ktr():
        name = cyto_channel.value
        stack = _channel_stack_by_name(name)
        if stack is None:
            return show_error("Pick a valid Cytoplasm channel.")
        df = fluorescence.measure_intensity(
            stack, state.mask, channel_name=name,
            dilation=int(ring_dilation.value), gap=int(ring_gap.value),
            background_roi=_roi_from_layer(bg_roi_layer.value))
        if df.empty:
            return show_error("No cells measured.")
        col = f"{name}_cn_ratio"
        keep = df[["track_id", "frame", col]]
        extra_feature_tables.append(keep)
        import os
        out_dir, base = _exports_dir_and_base()
        keep.to_csv(os.path.join(out_dir, f"{base}_erk_ktr.csv"), index=False)
        try:
            fig = stats.custom_plot(keep, "frame", col, group_by="track_id",
                                    kind="line", show_legend=False)
            stats.show(fig)
        except Exception:
            pass
        show_info(f"ERK-KTR C/N computed for {keep['track_id'].nunique()} cells; "
                  f"added to next export.")
    btn_erk.clicked.connect(_erk_ktr)

    def _dsb_53bp1():
        name = nuc_channel.value
        stack = _channel_stack_by_name(name)
        if stack is None:
            return show_error("Pick a valid Nucleus channel.")
        inten = fluorescence.measure_intensity(
            stack, state.mask, channel_name=name,
            dilation=int(ring_dilation.value), gap=int(ring_gap.value),
            background_roi=_roi_from_layer(bg_roi_layer.value))
        hara = fluorescence.haralick_features(stack, state.mask, channel_name=name)
        cols = ["track_id", "frame", f"{name}_nuc_median", f"{name}_nuc_std"]
        merged = inten[cols].merge(hara, on=["track_id", "frame"], how="left")
        merged[f"{name}_nuc_cv"] = (
            merged[f"{name}_nuc_std"] /
            merged[f"{name}_nuc_median"].replace(0, np.nan))
        extra_feature_tables.append(merged)
        import os
        out_dir, base = _exports_dir_and_base()
        merged.to_csv(os.path.join(out_dir, f"{base}_53bp1.csv"), index=False)
        show_info(f"53BP1 texture computed for {merged['track_id'].nunique()} cells; "
                  f"added to next export.")
    btn_53bp1.clicked.connect(_dsb_53bp1)

    btn_nma = PushButton(text="Plot NMA (Area vs NII, per cell-frame)")
    btn_compare = PushButton(text="Compare cell [A] vs dataset (by group)")

    btn_stats_lifetime = PushButton(text="Lifetime (treated vs control, mitotic vs not)")
    btn_stats_migration = PushButton(text="Migration (speed, displacement, directionality)")
    btn_stats_motility = PushButton(text="Motility (diffusion, persistence, turning, confinement)")
    btn_stats_area = PushButton(text="Area (by group and over time)")
    btn_stats_growth = PushButton(text="Population growth curve")
    btn_stats_outcomes = PushButton(text="Outcome distribution")
    btn_stats_division = PushButton(text="Division events over time")
    btn_stats_msd = PushButton(text="Mean squared displacement (MSD)")
    btn_stats_export = PushButton(text="Export per-track summary (CSV)")

    # ---- new: temporal-gradient boxplot ----
    lbl_gradient = Label(value="--- TEMPORAL GRADIENT (boxplot) ---")
    # Quantity for the Y axis: area (from the mask) plus any numeric per-frame
    # column (e.g. area_px, perimeter once merged, x/y, speed columns).
    _grad_cols = ["area"] + [c for c in state.df.columns
                             if pd.api.types.is_numeric_dtype(state.df[c])
                             and c not in (COL_TRACK, COL_FRAME)]
    gradient_value = ComboBox(label="Quantity (Y):", choices=_grad_cols, value="area")
    gradient_group = ComboBox(
        label="Group boxes by (X):",
        choices=["final_outcome", "treatment", "is_mitotic", "time"],
        value="final_outcome")
    btn_gradient = PushButton(text="Plot boxplot + temporal gradient")

    # ---- new: per-track trajectory plots (one line per cell) ----
    lbl_traj = Label(value="--- TRAJECTORY PLOTS (one line per cell) ---")
    traj_mode = ComboBox(
        label="Plot type:",
        choices=["timeseries", "cumulative", "spider", "window"],
        value="timeseries")
    # Y choices for the timeseries mode: any numeric per-frame column, plus the
    # mask-derived perimeter/circularity (computed on demand).
    _traj_ts_cols = [c for c in state.df.columns
                     if pd.api.types.is_numeric_dtype(state.df[c])
                     and c not in (COL_TRACK, COL_FRAME)]
    _traj_ts_cols = (["area_px"] if "area_px" not in _traj_ts_cols else []) + _traj_ts_cols
    # Mask-derived columns computable on demand: perimeter/circularity, the NII
    # morphometry and the shape descriptors (eccentricity, solidity, extent,
    # orientation, axis lengths). Deduped against columns already in the table.
    _extra_traj = ["perimeter", "circularity"] + list(analysis.MORPHOMETRY_COLS)
    _traj_ts_cols += [c for c in _extra_traj if c not in _traj_ts_cols]
    traj_y = ComboBox(label="Quantity (Y):", choices=_traj_ts_cols,
                      value=_traj_ts_cols[0])
    traj_window_kind = ComboBox(label="Window metric:",
                                choices=list(stats.WINDOW_MODES.keys()),
                                value="persistence")
    traj_window = SpinBox(label="Window size (frames):", value=11, min=3, max=999)
    traj_smooth = SpinBox(label="Smoothing (frames):", value=1, min=1, max=99)
    traj_maxtracks = SpinBox(label="Max tracks (0 = all):", value=0, min=0, max=100000)
    traj_labels = CheckBox(label="Label track IDs at line ends", value=True)
    btn_traj = PushButton(text="Plot trajectories")

    def _on_traj_mode_changed(value):
        # Show the relevant Y selector depending on mode.
        is_ts = value == "timeseries"
        is_win = value == "window"
        traj_y.visible = is_ts or value == "cumulative"
        traj_window_kind.visible = is_win
        traj_window.visible = is_win
        traj_smooth.visible = is_ts
    traj_mode.changed.connect(_on_traj_mode_changed)

    # ---- new: customizable plot ----
    lbl_custom = Label(value="--- CUSTOM PLOT ---")
    # Per-track summary columns (area, displacement, speed, etc.) the user can
    # plot in addition to the raw per-frame columns.
    SUMMARY_COLS = ["lifetime", "n_frames", "total_distance", "net_displacement",
                    "mean_speed", "directionality", "mean_area", "max_area",
                    "diffusion_coeff", "persistence_time", "mean_turning_angle",
                    "confinement_ratio", "first_frame", "last_frame"]
    raw_numeric = [c for c in state.df.columns
                   if pd.api.types.is_numeric_dtype(state.df[c])] or [COL_FRAME, COL_X, COL_Y]
    custom_source = ComboBox(label="Data source:",
                             choices=["per-frame table", "per-track summary"],
                             value="per-frame table")
    custom_x = ComboBox(label="X axis:", choices=raw_numeric,
                        value=COL_FRAME if COL_FRAME in raw_numeric else raw_numeric[0])
    custom_y = ComboBox(label="Y axis:", choices=raw_numeric,
                        value=COL_X if COL_X in raw_numeric else raw_numeric[0])
    custom_group = ComboBox(label="Group by:", choices=["(none)"] + list(stats.GROUPING_KEYS),
                            value="(none)")
    custom_kind = ComboBox(label="Kind:", choices=["scatter", "line"], value="scatter")
    custom_agg = ComboBox(label="Line aggregation:", choices=["mean", "median", "sum", "count"],
                          value="mean")
    custom_legend = CheckBox(label="Show legend", value=True)
    btn_custom = PushButton(text="Plot custom")

    @custom_source.changed.connect
    def _on_source_changed(value):
        # Swap the axis choices to match the selected data source.
        if value == "per-track summary":
            cols = SUMMARY_COLS + [COL_TRACK]
            grp = ["(none)", "treatment", "final_outcome", "is_mitotic"]
        else:
            cols = raw_numeric
            grp = ["(none)"] + list(stats.GROUPING_KEYS)
        custom_x.choices = cols
        custom_y.choices = cols
        custom_group.choices = grp
        if cols:
            custom_x.value = cols[0]
            custom_y.value = cols[1] if len(cols) > 1 else cols[0]
        custom_group.value = "(none)"

    def stats_df():
        """The dataframe statistics should use, honoring the filter checkboxes.

        Two independent, composable filters: border-touching rows (partial
        measurements) and interpolated rows (gap-filled centroids with no real
        segmentation). Toggling the interpolated box lets you generate the same
        centroid-based figure (migration, MSD, directionality) with and without
        the synthesized frames, straight from the panel.
        """
        from .config import COL_INTERPOLATED
        df = state.df
        if exclude_border_cb.value:
            df = analysis.exclude_border_rows(df)
        if exclude_interp_cb.value and COL_INTERPOLATED in df.columns:
            df = df[~df[COL_INTERPOLATED].fillna(False).astype(bool)]
        return df

    def get_summary():
        return analysis.compute_track_summary(
            stats_df(), state.mask,
            pixel_size=pixel_size_input.value,
            frame_interval=frame_interval_input.value)

    @btn_stats_lifetime.clicked.connect
    def _s_life():
        s = get_summary()
        if s.empty:
            return show_error("No valid tracks to analyze.")
        stats.show(stats.lifetime_figure(s, frame_interval_input.value))

    @btn_stats_migration.clicked.connect
    def _s_mig():
        s = get_summary()
        if s.empty:
            return show_error("No valid tracks to analyze.")
        stats.show(stats.migration_figure(s, pixel_size_input.value, frame_interval_input.value))

    @btn_stats_motility.clicked.connect
    def _s_motil():
        s = get_summary()
        if s.empty:
            return show_error("No valid tracks to analyze.")
        stats.show(stats.motility_figure(s, pixel_size_input.value, frame_interval_input.value))

    @btn_stats_area.clicked.connect
    def _s_area():
        s = get_summary()
        if s.empty or s["mean_area"].isna().all():
            return show_error("No area data available from the mask.")
        stats.show(stats.area_figure(s, state.df, state.mask, pixel_size_input.value,
                                     treatment_config, max_frame))

    @btn_stats_growth.clicked.connect
    def _s_growth():
        try:
            stats.show(stats.growth_figure(stats_df(), treatment_config, max_frame))
        except Exception as exc:
            show_error(str(exc))

    @btn_stats_outcomes.clicked.connect
    def _s_out():
        s = get_summary()
        if s.empty:
            return show_error("No valid tracks to analyze.")
        fig, counts = stats.outcomes_figure(s)
        stats.show(fig)
        print("\nOutcome distribution:\n", counts)

    @btn_stats_division.clicked.connect
    def _s_div():
        res = stats.division_figure(stats_df(), treatment_config, max_frame)
        if res is None:
            return show_error("No lineage links to analyze.")
        fig, total = res
        stats.show(fig)
        print(f"\nTotal divisions: {total}")

    @btn_stats_msd.clicked.connect
    def _s_msd():
        fig = stats.msd_figure(stats_df(), pixel_size_input.value, frame_interval_input.value)
        if fig is None:
            return show_error("Tracks too short / no data for MSD.")
        stats.show(fig)

    @btn_stats_export.clicked.connect
    def _s_export():
        s = get_summary()
        if s.empty:
            return show_error("No valid tracks to export.")
        path, _ = QFileDialog.getSaveFileName(
            None, "Export per-track summary",
            "track_summary.csv", "CSV Files (*.csv)")
        if not path:
            return
        s.to_csv(path, index=False)
        show_info(f"Summary exported: {path}")

    @btn_gradient.clicked.connect
    def _gradient():
        try:
            fig = stats.gradient_over_time(
                stats_df(), state.mask, value=gradient_value.value,
                pixel_size=pixel_size_input.value, frame_interval=frame_interval_input.value,
                group_by=gradient_group.value)
            stats.show(fig)
        except Exception as exc:
            show_error(str(exc))

    @btn_nma.clicked.connect
    def _nma():
        try:
            fig = stats.nma_scatter(
                stats_df(), state.mask,
                pixel_size=pixel_size_input.value,
                treatment_config=treatment_config)
            stats.show(fig)
        except Exception as exc:
            show_error(str(exc))

    @btn_compare.clicked.connect
    def _compare():
        a = id_a_input.value
        if a in (0, None) or int(a) <= 0:
            return show_error("Put a cell ID in [A] first.")
        try:
            pages = stats.compare_cell_pages(
                stats_df(), state.mask, int(a),
                pixel_size=pixel_size_input.value,
                frame_interval=frame_interval_input.value)
        except Exception as exc:
            return show_error(str(exc))
        dlg = CompareCellDialog(None, int(a), pages)
        open_dialogs["compare"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    @btn_traj.clicked.connect
    def _traj():
        mode = traj_mode.value
        # In cumulative mode an empty/area Y means "use displacement"; allow a
        # blank meaning by passing None when the chosen column is the default.
        y = traj_y.value
        if mode == "window":
            y = traj_window_kind.value
        elif mode == "spider":
            y = None
        try:
            fig = stats.trajectories(
                stats_df(), mask=state.mask, mode=mode, y_col=y,
                treatment_config=treatment_config,
                pixel_size=pixel_size_input.value,
                frame_interval=frame_interval_input.value,
                window=int(traj_window.value), smooth=int(traj_smooth.value),
                max_tracks=(int(traj_maxtracks.value) or None),
                show_labels=traj_labels.value)
            stats.show(fig)
        except Exception as exc:
            show_error(str(exc))

    @btn_custom.clicked.connect
    def _custom():
        group = None if custom_group.value == "(none)" else custom_group.value
        # Choose the data source: raw per-frame table or the per-track summary.
        if custom_source.value == "per-track summary":
            source = get_summary()
            if source.empty:
                return show_error("No valid tracks for the summary.")
        else:
            source = stats_df()
        try:
            fig = stats.custom_plot(source, custom_x.value, custom_y.value,
                                    group_by=group, agg=custom_agg.value,
                                    kind=custom_kind.value,
                                    show_legend=custom_legend.value)
            stats.show(fig)
        except Exception as exc:
            show_error(str(exc))

    # ==================================================================
    # HOTKEYS
    # ==================================================================
    # Single-key shortcuts for the most-used tools. overwrite=True makes them
    # win even when napari binds the same key, so they always fire; each bind is
    # guarded so an unrecognized key on an older napari never breaks startup.
    # The napari Labels painting keys are left untouched: the number row (layer
    # modes), "m" (new label) and the brush keys keep their napari meaning, so
    # manual mask painting still works.
    def _bind(key, fn):
        def _cb(viewer, _fn=fn):
            _fn()
        try:
            viewer.bind_key(key, _cb, overwrite=True)
        except Exception:
            pass

    def _hk_flag(outcome, action):
        finish(ops.apply_flag(state, id_a_input.value, outcome), action,
               [id_a_input.value])

    _bind("d", lambda: _hk_flag(OUTCOME_MITOSIS, "flag_mitosis"))   # division
    _bind("w", lambda: _hk_flag(OUTCOME_EXIT, "flag_exit"))         # went off-field
    _bind("k", lambda: _hk_flag(OUTCOME_DEATH, "flag_death"))       # killed
    _bind("b", lambda: _hk_flag(OUTCOME_AMBIGUOUS, "flag_ambiguous"))
    _bind("c", lambda: finish(ops.clear_flag(state, id_a_input.value),
                              "flag_clear", [id_a_input.value]))     # clear

    _bind("g", lambda: finish(
        ops.merge(state, id_a_input.value, id_b_input.value),
        "merge", [id_a_input.value, id_b_input.value]))             # glue A->B
    _bind("s", lambda: finish(
        ops.swap_future(state, id_a_input.value, id_b_input.value, current_frame()),
        "swap_future", [id_a_input.value, id_b_input.value]))
    _bind("Shift-S", lambda: finish(
        ops.swap_local(state, id_a_input.value, id_b_input.value, current_frame()),
        "swap_local", [id_a_input.value, id_b_input.value]))
    _bind("n", lambda: finish(
        ops.relabel_new(state, id_a_input.value), "relabel", [id_a_input.value]))
    _bind("y", lambda: finish(
        ops.sync_masks(state, [current_frame()], treatment_config), "sync_frame"))
    _bind("l", _link)

    _bind("f", _focus)
    _bind("Shift-F", _lineage_focus)

    # Fluorescence channels: `v` toggles the visibility of the extra channel
    # layers as a group (off->on->off) at the current frame, so a reporter can be
    # flashed on to confirm a flagged event without leaving the canvas. With more
    # than one channel, Shift+V cycles which single channel is shown solo.
    _chan_state = {"idx": -1}  # -1 = all hidden; >=0 = that channel solo

    def _toggle_channels():
        if not channel_layer_names:
            return show_info("No extra fluorescence channels loaded.")
        any_visible = any(viewer.layers[n].visible for n in channel_layer_names
                          if n in viewer.layers)
        for n in channel_layer_names:
            if n in viewer.layers:
                viewer.layers[n].visible = not any_visible
        _chan_state["idx"] = -1
        show_info("Fluorescence channels " + ("hidden." if any_visible else "shown."))

    def _cycle_channels():
        if not channel_layer_names:
            return show_info("No extra fluorescence channels loaded.")
        _chan_state["idx"] = (_chan_state["idx"] + 1) % len(channel_layer_names)
        sel = _chan_state["idx"]
        for i, n in enumerate(channel_layer_names):
            if n in viewer.layers:
                viewer.layers[n].visible = (i == sel)
        show_info(f"Channel solo: {channel_layer_names[sel]}.")

    _bind("v", _toggle_channels)
    _bind("Shift-V", _cycle_channels)

    _bind(".", _next_unreviewed)
    _bind(",", _skip)
    _bind("Control-S", on_save)

    # ==================================================================
    # PANELS
    # ==================================================================
    # The panel is long (40+ tools). Instead of a flat column separated by faint
    # text labels, the tools are grouped into titled, collapsible sections so the
    # ones you are not using can be folded away. Each section is still its own
    # magicgui Container (so input labels like "ID [A]:" are preserved and no
    # button is recreated or re-wired); the Container is only wrapped in a
    # collapsible box. superqt's QCollapsible ships with napari; if it is somehow
    # missing we fall back to a plain titled QGroupBox (no folding, still grouped).
    _section_refs = []  # keep Container wrappers alive past GC

    try:
        from superqt import QCollapsible
        _HAVE_COLLAPSIBLE = True
    except Exception:
        _HAVE_COLLAPSIBLE = False

    def _section(title, widgets, collapsed=False):
        inner = Container(widgets=widgets, labels=True)
        _section_refs.append(inner)
        try:
            inner.native.layout().setContentsMargins(8, 6, 8, 6)
        except Exception:
            pass
        if _HAVE_COLLAPSIBLE:
            box = QCollapsible(title)
            box.addWidget(inner.native)
            # A slightly heavier, left-aligned header reads as a real section
            # divider without shouting; keep it sober for a scientific tool.
            try:
                box.toggleButton().setStyleSheet(
                    "text-align:left; border:none; outline:none;"
                    "font-weight:600; padding:4px 2px;")
                box.setDuration(120)
            except Exception:
                pass
            (box.collapse if collapsed else box.expand)(animate=False)
            return box
        gb = QGroupBox(title)
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.addWidget(inner.native)
        return gb

    def _build_panel(sections):
        """Stack section boxes (each a QCollapsible/QGroupBox) into a scrollable column."""
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(6)
        for box in sections:
            col.addWidget(box)
        col.addStretch(1)
        return host

    # ---- Curation tab ----
    # A fixed header (selection + progress) stays visible above the sections so
    # the [A]/[B] inputs are always reachable; only the tool groups fold.
    curation_header = Container(widgets=[
        instruction, progress_label, id_a_input, id_b_input, flag_filter], labels=True)
    _section_refs.append(curation_header)
    try:
        curation_header.native.layout().setContentsMargins(8, 6, 8, 4)
    except Exception:
        pass

    curation_sections = [
        _section("SETUP", [btn_treatment], collapsed=True),
        _section("NAVIGATION & VIEW",
                 [btn_focus, btn_tracks, btn_lineage_focus, lineage_progress,
                  btn_lineage_reviewed, btn_next_unreviewed, btn_skip_single,
                  btn_shuffle, btn_undo]),
        _section("BASIC CURATION",
                 [btn_merge, btn_swap_future, btn_swap_local, btn_new_track,
                  btn_sync_frame, btn_sync_all, btn_harmonize, btn_delete,
                  btn_delete_track, btn_rescue, btn_autotrack]),
        _section("OUTCOME FLAGS (on ID [A])",
                 [btn_flag_mitosis, btn_flag_exit, btn_flag_death,
                  btn_flag_ambiguous, btn_flag_clear]),
        _section("LINEAGE / GENEALOGY",
                 [btn_link_parent, btn_lineage_editor,
                  btn_cut_ghosts, btn_tree_plot, tree_max_families,
                  tree_include_singles, btn_validate]),
        _section("DIAGNOSTICS & EXPORT",
                 [btn_diagnose, btn_triage, btn_relink, gap_method, btn_fill_gaps,
                  btn_swaps, btn_morph, btn_export_cell, btn_export_video, btn_save],
                 collapsed=True),
    ]
    curation_host = QWidget()
    _clay = QVBoxLayout(curation_host)
    _clay.setContentsMargins(4, 4, 4, 4)
    _clay.setSpacing(6)
    _clay.addWidget(curation_header.native)
    for box in curation_sections:
        _clay.addWidget(box)
    _clay.addStretch(1)

    # ---- Statistics tab ----
    # Calibration + filters stay visible (they affect every plot); the plot
    # groups fold. The advanced plot builders start collapsed.
    stats_header = Container(widgets=[
        pixel_size_input, frame_interval_input, exclude_border_cb,
        exclude_interp_cb, export_window], labels=True)
    _section_refs.append(stats_header)
    try:
        stats_header.native.layout().setContentsMargins(8, 6, 8, 4)
    except Exception:
        pass

    stats_sections = [
        _section("QUICK STATISTICS",
                 [btn_nma, btn_compare, btn_stats_lifetime, btn_stats_migration,
                  btn_stats_motility, btn_stats_area, btn_stats_growth,
                  btn_stats_outcomes, btn_stats_division, btn_stats_msd,
                  btn_stats_export]),
        _section("TEMPORAL GRADIENT",
                 [gradient_value, gradient_group, btn_gradient], collapsed=True),
        _section("TRAJECTORY PLOTS",
                 [traj_mode, traj_y, traj_window_kind, traj_window, traj_smooth,
                  traj_maxtracks, traj_labels, btn_traj], collapsed=True),
        _section("CUSTOM PLOT",
                 [custom_source, custom_x, custom_y, custom_group, custom_kind,
                  custom_agg, custom_legend, btn_custom], collapsed=True),
        _section("FLUORESCENCE",
                 [btn_add_channel, cyto_channel, nuc_channel, bg_roi_layer,
                  ring_dilation, ring_gap, btn_ring_preview,
                  btn_erk, btn_53bp1], collapsed=True),
    ]
    stats_host = QWidget()
    _slay = QVBoxLayout(stats_host)
    _slay.setContentsMargins(4, 4, 4, 4)
    _slay.setSpacing(6)
    _slay.addWidget(stats_header.native)
    for box in stats_sections:
        _slay.addWidget(box)
    _slay.addStretch(1)

    curation_scroll = QScrollArea()
    curation_scroll.setWidgetResizable(True)
    curation_scroll.setWidget(curation_host)
    curation_scroll.setMinimumWidth(380)

    stats_scroll = QScrollArea()
    stats_scroll.setWidgetResizable(True)
    stats_scroll.setWidget(stats_host)
    stats_scroll.setMinimumWidth(380)

    viewer.window.add_dock_widget(curation_scroll, name="Curation tools", area="right")
    try:
        viewer.window.add_dock_widget(stats_scroll, name="Statistics", area="right", tabify=True)
    except TypeError:
        viewer.window.add_dock_widget(stats_scroll, name="Statistics", area="right")

    update_visuals()
    _refresh_lineage_progress()
    _update_reviewed_button()
    napari.run()
    return viewer
