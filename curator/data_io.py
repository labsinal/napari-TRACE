"""
Data loading and safe, versioned, atomic saving.

Performs timestamped backups before every save plus an atomic write
(temp file + rename) so a crash mid-write never corrupts the originals.

Also handles working-directory setup and the load path that assembles the
dataframe + mask + image, including the new rule that any detected mother
(via parent_id) is auto-flagged "Mitosis" when no outcome column exists.
"""

from __future__ import annotations

import os
import json
import shutil
import datetime as _dt

import numpy as np
import pandas as pd
import tifffile

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_CLABEL,
    COL_TREATMENT, COL_TREATMENT_LEGACY, OUTCOME_MITOSIS, TREAT_CONTROL, COL_BORDER,
)
from . import io_adapters

METADATA_FILE = "curator_meta.json"
BACKUP_DIRNAME = "backups"


# ---------------------------------------------------------------------------
# Metadata / working directory
# ---------------------------------------------------------------------------
def read_meta(work_dir):
    path = os.path.join(work_dir, METADATA_FILE)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def write_meta(work_dir, meta):
    path = os.path.join(work_dir, METADATA_FILE)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)


def meta_channels(work_dir):
    """Return the list of persisted extra-channel specs recorded in the meta.

    Each spec is a dict {name, file, color, measure}; ``file`` is relative to the
    working directory. Channels added through the in-app "Add channel" button are
    recorded here so they reload automatically in later sessions.
    """
    chans = read_meta(work_dir).get("channels", [])
    return chans if isinstance(chans, list) else []


def add_channel_to_meta(work_dir, name, filename, color="green", measure=False):
    """Record (or update) one extra channel in the meta, keyed by file/name.

    ``filename`` is the channel stack's name inside ``work_dir``. Returns the new
    channel list. Preserves the rest of the metadata (source_folder, version).
    """
    meta = read_meta(work_dir)
    chans = [c for c in meta.get("channels", [])
             if c.get("file") != filename and c.get("name") != name]
    chans.append({"name": str(name), "file": str(filename),
                  "color": str(color), "measure": bool(measure)})
    meta["channels"] = chans
    write_meta(work_dir, meta)
    return chans


def set_channel_measure(work_dir, name, measure, color=None):
    """Update the measure flag (and optionally color) of a recorded channel.

    Matches the meta entry by name; a no-op (returns False) when the channel is
    not recorded (e.g. an auto-discovered channel, whose tags are re-asked at
    load anyway). So toggling "measure" off in the UI sticks across sessions for
    channels added through the "Add channel" button.
    """
    meta = read_meta(work_dir)
    chans = meta.get("channels", [])
    changed = False
    for c in chans:
        if c.get("name") == str(name):
            c["measure"] = bool(measure)
            if color:
                c["color"] = str(color)
            changed = True
    if changed:
        meta["channels"] = chans
        write_meta(work_dir, meta)
    return changed


def setup_working_directory(source_folder):
    """Create a _curated sibling folder on first run; reuse it afterwards."""
    curated_name = os.path.basename(source_folder.rstrip("/\\")) + "_curated"
    work_dir = os.path.join(os.path.dirname(source_folder), curated_name)

    meta = read_meta(work_dir)
    if meta.get("source_folder") == source_folder:
        return work_dir, False

    os.makedirs(work_dir, exist_ok=True)
    # Copy the whole source into the working copy -- BOTH top-level files (CSV,
    # stacks) AND subfolders (per-frame TIF folders for the mask/image/extra
    # channels). Copying only files left frame-folder inputs behind, so the
    # working copy was missing channels and stacks lived next to the source.
    for fname in os.listdir(source_folder):
        src = os.path.join(source_folder, fname)
        dst = os.path.join(work_dir, fname)
        if os.path.exists(dst):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
    write_meta(work_dir, {"source_folder": source_folder, "version": 6})
    return work_dir, True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _translate_outcomes(df):
    """Map foreign/legacy outcome labels onto the internal vocabulary in place.

    - Known translations (Portuguese etc.) become the canonical Exit/Mitosis/
      Death values.
    - Legacy start-of-life markers ("Início", "Start", ...) are blanked so the
      track's real final outcome (propagated later in the UI) takes over.
    - Unknown non-empty values are left untouched (the user may have custom
      flags); they simply won't be treated as final outcomes.
    """
    from .config import OUTCOME_TRANSLATIONS, START_MARKERS
    if COL_OUTCOME not in df.columns:
        return df

    def _map(v):
        key = str(v).strip().lower()
        if key in ("", "nan", "none"):
            return ""
        if key in START_MARKERS:
            return ""
        if key in OUTCOME_TRANSLATIONS:
            return OUTCOME_TRANSLATIONS[key]
        return v  # leave unrecognized labels as-is

    df[COL_OUTCOME] = df[COL_OUTCOME].map(_map)
    return df


def _auto_flag_mothers(df):
    """Any track that is some cell's parent is flagged Mitosis (in place)."""
    if COL_PARENT not in df.columns:
        return df
    mothers = set(df.loc[df[COL_PARENT] > 0, COL_PARENT].dropna().astype(int).unique())
    if mothers:
        is_mother = df[COL_TRACK].astype("Int64").isin(mothers)
        # Only fill where currently empty, so we never clobber a manual flag.
        empty = df[COL_OUTCOME].astype(str).isin(["", "nan", "None"])
        df.loc[is_mother & empty, COL_OUTCOME] = OUTCOME_MITOSIS
    return df


def load_data(csv_path, mask_path, image_path, column_map,
              mask_folder=None, image_folder=None, work_dir=None):
    """Load and standardize the table plus the mask and image stacks.

    Outcome/parent columns are preserved when present. When the parent column
    exists (or is detected) every mother is auto-flagged "Mitosis" if it has no
    outcome yet -- so a freshly created outcome column already reflects lineage.
    """
    existing_outcome_col = existing_parent_col = None
    if not csv_path.lower().endswith(".txt"):
        header = pd.read_csv(csv_path, nrows=0)
        existing_outcome_col, existing_parent_col = \
            io_adapters.detect_existing_lineage(header.columns)

    df = io_adapters.load_table_any(csv_path, column_map)

    # Outcome column.
    if COL_OUTCOME not in df.columns:
        if existing_outcome_col and existing_outcome_col in df.columns:
            df = df.rename(columns={existing_outcome_col: COL_OUTCOME})
        else:
            df[COL_OUTCOME] = ""
    df[COL_OUTCOME] = df[COL_OUTCOME].fillna("").astype(str)
    df.loc[df[COL_OUTCOME].isin(["nan", "None"]), COL_OUTCOME] = ""

    # Translate foreign/legacy outcome labels (e.g. Portuguese) onto the
    # internal vocabulary, and drop legacy start-of-life markers so the track's
    # real final outcome takes over. Done before mother auto-flagging so the
    # propagation in update_visuals works on canonical values.
    df = _translate_outcomes(df)

    # Parent column.
    if COL_PARENT not in df.columns:
        if existing_parent_col and existing_parent_col in df.columns:
            df = df.rename(columns={existing_parent_col: COL_PARENT})
        else:
            df[COL_PARENT] = -1
    df[COL_PARENT] = pd.to_numeric(df[COL_PARENT], errors="coerce").fillna(-1).astype(int)

    # Auto-flag mothers as Mitosis (new requirement).
    df = _auto_flag_mothers(df)

    if COL_CLABEL not in df.columns:
        df[COL_CLABEL] = df[COL_TRACK]
    # Tables written before the phase column was renamed carry "treatment".
    # Migrate them so every CSV this tool has ever produced still opens.
    if COL_TREATMENT not in df.columns and COL_TREATMENT_LEGACY in df.columns:
        df = df.rename(columns={COL_TREATMENT_LEGACY: COL_TREATMENT})
    if COL_TREATMENT not in df.columns:
        df[COL_TREATMENT] = TREAT_CONTROL

    # Mask stack.
    if mask_path is None and mask_folder is not None:
        stack_name = os.path.basename(mask_folder.rstrip("/\\")) + "_stack.tif"
        base = work_dir or os.path.dirname(csv_path)
        stack_path = os.path.join(base, stack_name)
        if os.path.exists(stack_path):
            masks = tifffile.imread(stack_path).astype(np.uint32)
        else:
            masks = io_adapters.build_stack_from_folder(mask_folder, stack_path).astype(np.uint32)
        mask_path = stack_path
    else:
        masks = tifffile.imread(mask_path).astype(np.uint32)

    # Image stack.
    if image_path is None and image_folder is not None:
        stack_name = os.path.basename(image_folder.rstrip("/\\")) + "_stack.tif"
        base = work_dir or os.path.dirname(csv_path)
        stack_path = os.path.join(base, stack_name)
        if os.path.exists(stack_path):
            images = tifffile.imread(stack_path)
        else:
            images = io_adapters.build_stack_from_folder(image_folder, stack_path)
        image_path = stack_path
    else:
        images = tifffile.imread(image_path)

    warnings = check_image_mask_coherence(images, masks)
    for w in warnings:
        print("WARNING:", w)

    # Border-contact flag (separate from biological outcome). Preserve an
    # existing column if the CSV already had one; otherwise compute from masks.
    from . import analysis
    if COL_BORDER in df.columns:
        df[COL_BORDER] = df[COL_BORDER].fillna(False).astype(bool)
    else:
        df = analysis.annotate_border_contact(df, masks)

    # Per (frame, track) mask area in pixels. Always recomputed from the mask
    # so it reflects the actual segmentation, even if the CSV carried an old
    # value. Available afterwards as a column in the raw per-frame table.
    df = analysis.annotate_area(df, masks)

    # Auto-classify EXIT: a cell that touches the border AND shrinks to nothing
    # before the movie ends (migrated out of the field), as long as it has no
    # outcome yet (Mitosis / manual / CSV outcomes are never overwritten).
    before = (df[COL_OUTCOME].astype(str)
              .isin([config.OUTCOME_EXIT])).sum() if COL_OUTCOME in df else 0
    df = analysis.auto_flag_border_exits(df, masks)
    after = (df[COL_OUTCOME].astype(str).isin([config.OUTCOME_EXIT])).sum()
    if after > before:
        print(f"Auto-flagged {after - before} track(s) as EXIT "
              f"(border contact + shrinking to vanishing).")

    return df, masks, images, mask_path, image_path, warnings


def check_image_mask_coherence(images, masks):
    """Return a list of human-readable warnings if image and mask don't match.

    Catches the common mistake of selecting an image and a mask from different
    experiments / folders: a different number of frames, or different per-frame
    height/width. Returns an empty list when everything lines up. This only
    warns; it never raises, so the user can still proceed if they know better.
    """
    warns = []
    if images is None or masks is None:
        return warns

    img_frames = images.shape[0] if images.ndim >= 3 else 1
    msk_frames = masks.shape[0] if masks.ndim >= 3 else 1
    if img_frames != msk_frames:
        warns.append(
            f"Frame count mismatch: image has {img_frames} frames, "
            f"mask has {msk_frames}. They may be from different experiments.")

    # Compare the per-frame spatial dimensions (last two axes).
    img_hw = tuple(images.shape[-2:]) if images.ndim >= 2 else None
    msk_hw = tuple(masks.shape[-2:]) if masks.ndim >= 2 else None
    if img_hw and msk_hw and img_hw != msk_hw:
        warns.append(
            f"Frame size mismatch: image frames are {img_hw[1]}x{img_hw[0]} "
            f"(WxH), mask frames are {msk_hw[1]}x{msk_hw[0]}. Overlay will be "
            f"misaligned.")
    return warns


# ---------------------------------------------------------------------------
# Safe, versioned, atomic saving
# ---------------------------------------------------------------------------
def _timestamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_files(paths, work_dir):
    """Copy each existing path into work_dir/backups/<timestamp>/. Returns dir."""
    stamp = _timestamp()
    bdir = os.path.join(work_dir, BACKUP_DIRNAME, stamp)
    os.makedirs(bdir, exist_ok=True)
    for p in paths:
        if p and os.path.exists(p):
            shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
    return bdir


def atomic_write_csv(df, path):
    """Write a CSV atomically (temp file in same dir, then os.replace)."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp_{_timestamp()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_tif(array, path):
    """Write a TIFF atomically (temp file in same dir, then os.replace)."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp_{_timestamp()}.tif")
    tifffile.imwrite(tmp, array)
    os.replace(tmp, path)


def save_session(df, mask, csv_path, mask_path, work_dir):
    """Backup the current files, then atomically write the new CSV and TIF.

    Returns the backup directory used.
    """
    bdir = backup_files([csv_path, mask_path], work_dir) if work_dir else None
    atomic_write_csv(df, csv_path)
    atomic_write_tif(mask.astype(np.uint32), mask_path)
    return bdir
