"""
Input adapters and file/stack resolution.

Provides translator functions that convert common external formats
(Cell Tracking Challenge, TrackMate) onto the internal schema, so the rest
of the program never sees foreign conventions.

Also hosts the folder/stack discovery helpers (formerly resolve_paths and the
TIF-folder stacking utilities).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tifffile

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT,
    COLUMN_CANDIDATES, OUTCOME_CANDIDATES, PARENT_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------------
def best_guess(columns, candidates):
    """Return the first candidate present in *columns* (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def default_mapping(columns):
    """Best-guess source column for each internal target column."""
    return {target: best_guess(columns, cands)
            for target, cands in COLUMN_CANDIDATES.items()}


def detect_existing_lineage(df_columns):
    """Return (outcome_col, parent_col) found in df_columns, or None each."""
    cols_lower = {c.lower(): c for c in df_columns}

    def find(cands):
        for c in cands:
            if c in df_columns:
                return c
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    return find(OUTCOME_CANDIDATES), find(PARENT_CANDIDATES)


def standardize_columns(df, column_map):
    """Rename mapped source columns onto the internal schema and coerce numerics."""
    for canonical, source in column_map.items():
        if source and source != canonical and canonical in df.columns:
            df = df.drop(columns=[canonical])
    rename = {src: tgt for tgt, src in column_map.items()
              if src and src in df.columns and src != tgt}
    df = df.rename(columns=rename)
    for col in (COL_TRACK, COL_FRAME, COL_X, COL_Y):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Format adapters
# ---------------------------------------------------------------------------
def detect_format(columns) -> str:
    """Heuristically name the input format: 'trackmate', 'ctc' or 'generic'."""
    cset = {c.lower() for c in columns}
    if {"track_id", "spot_id"} <= cset or {"position_x", "position_t"} <= cset:
        return "trackmate"
    # CTC res_track.txt has no header; the man_track / acTIWS exports use these.
    if {"l", "b", "e", "p"} <= cset or {"label", "begin", "end", "parent"} <= cset:
        return "ctc"
    return "generic"


def adapt_trackmate(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a TrackMate spots export onto the internal schema.

    TrackMate CSVs frequently carry two header rows; callers should pass a frame
    already read with the data rows only. Maps TRACK_ID/POSITION_X/Y/FRAME and
    preserves a parent if present.
    """
    colmap = {
        COL_TRACK: best_guess(df.columns, ["TRACK_ID", "track_id", "TrackID"]),
        COL_FRAME: best_guess(df.columns, ["FRAME", "POSITION_T", "frame", "t"]),
        COL_X: best_guess(df.columns, ["POSITION_X", "X", "x"]),
        COL_Y: best_guess(df.columns, ["POSITION_Y", "Y", "y"]),
    }
    out = standardize_columns(df.copy(), colmap)
    # Drop TrackMate's non-data header remnants (rows where frame is NaN).
    out = out.dropna(subset=[COL_FRAME])
    return out


def adapt_ctc(track_txt_path: str) -> pd.DataFrame:
    """Read a Cell Tracking Challenge ``res_track.txt`` into internal lineage.

    The CTC table has one row per track: ``L B E P`` = label, begin frame,
    end frame, parent label. We expand it to one row per (track, frame) with
    placeholder coordinates (-1) to be filled later by a mask sync, and set
    parent_id from P. Returns a dataframe on the internal schema.
    """
    arr = np.loadtxt(track_txt_path, dtype=int).reshape(-1, 4)
    rows = []
    for label, begin, end, parent in arr:
        for f in range(int(begin), int(end) + 1):
            rows.append({
                COL_TRACK: int(label), COL_FRAME: int(f),
                COL_X: -1.0, COL_Y: -1.0,
                COL_PARENT: int(parent) if parent > 0 else -1,
                COL_OUTCOME: "",
            })
    return pd.DataFrame(rows)


def load_table_any(csv_path: str, column_map: dict | None = None) -> pd.DataFrame:
    """Load a tracking table from any supported format onto the internal schema.

    For ``.txt`` CTC files this reads the lineage directly. For CSVs it detects
    TrackMate vs generic; a supplied ``column_map`` (from the UI dialog) always
    wins for generic CSVs.
    """
    if csv_path.lower().endswith(".txt"):
        return adapt_ctc(csv_path)

    df = pd.read_csv(csv_path)
    fmt = detect_format(df.columns)
    if fmt == "trackmate" and not column_map:
        return adapt_trackmate(df)
    if column_map:
        return standardize_columns(df, column_map)
    return standardize_columns(df, default_mapping(df.columns))


# ---------------------------------------------------------------------------
# TIF-folder -> stack helpers
# ---------------------------------------------------------------------------
def tif_files_in_folder(folder):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".tif", ".tiff")))
    return [os.path.join(folder, f) for f in files]


def looks_like_frame_folder(folder):
    return os.path.isdir(folder) and len(tif_files_in_folder(folder)) > 1


def build_stack_from_folder(frame_folder, stack_path):
    paths = tif_files_in_folder(frame_folder)
    frames = [tifffile.imread(p) for p in paths]
    stack = np.stack(frames, axis=0)
    tifffile.imwrite(stack_path, stack)
    print(f"Built stack from {len(frames)} frames -> {stack_path}")
    return stack


def resolve_paths(folder):
    """Auto-discover the CSV, mask and image inside a folder (files or frame dirs)."""
    csv_path = mask_path = image_path = None
    mask_folder = image_folder = None

    if folder and os.path.isdir(folder):
        entries = os.listdir(folder)
        files = [e for e in entries if os.path.isfile(os.path.join(folder, e))]
        subdirs = [e for e in entries if os.path.isdir(os.path.join(folder, e))]

        csvs = [f for f in files if f.lower().endswith((".csv", ".txt"))]
        tifs = [f for f in files if f.lower().endswith((".tif", ".tiff"))]

        if csvs:
            csv_path = os.path.join(folder, csvs[0])
            for c in csvs:
                if "track" in c.lower():
                    csv_path = os.path.join(folder, c)
                    break

        mask_files = [f for f in tifs if any(k in f.lower() for k in ("mask", "label", "seg"))]
        if mask_files:
            mask_path = os.path.join(folder, mask_files[0])
        image_files = [f for f in tifs if f not in mask_files]
        if image_files:
            image_path = os.path.join(folder, image_files[0])
            for f in image_files:
                if any(k in f.lower() for k in ("image", "raw", "img")):
                    image_path = os.path.join(folder, f)
                    break

        for sd in subdirs:
            sd_path = os.path.join(folder, sd)
            if not looks_like_frame_folder(sd_path):
                continue
            if any(k in sd.lower() for k in ("mask", "label", "seg")):
                if mask_path is None:
                    mask_folder = sd_path
            else:
                if image_path is None:
                    image_folder = sd_path

    return csv_path, mask_path, image_path, mask_folder, image_folder
