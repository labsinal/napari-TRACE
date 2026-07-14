"""
Central configuration: internal column schema, label vocabulary, reserved
ID ranges and curation thresholds.

This module is intentionally free of any Qt / napari / pandas-heavy logic so it
can be imported anywhere (including from test code) with no side effects.

Provides a single source of truth for every reserved ID range (namespacing)
and the curation thresholds, which can optionally be derived automatically
from the dataset (see thresholds_from_dataset()).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Internal column schema
# ---------------------------------------------------------------------------
# The whole tool works on this fixed internal schema. Any input CSV is remapped
# onto these names at load time so the rest of the code never has to care about
# the original tracker conventions.
COL_TRACK = "track_id"
COL_FRAME = "frame"
COL_X = "pos_x"
COL_Y = "pos_y"
COL_OUTCOME = "outcome"
COL_PARENT = "parent_id"
COL_CLABEL = "continuous_label"
COL_TREATMENT = "treatment"
# Separate, "invisible" flag (kept apart from the biological COL_OUTCOME):
# True when the cell's mask touches the frame border in that frame. Used to
# truncate / down-weight / impute border-affected measurements in statistics.
COL_BORDER = "at_border"
# Per (frame, track) mask area in PIXELS (px^2). Computed from the mask at load
# time and recomputed at save time so it always reflects the curated mask.
# Conversion to physical units (um^2) is applied in statistics via pixel size.
COL_AREA = "area_px"
# True for per-frame rows whose centroid was SYNTHESIZED to bridge a tracking
# gap (see lineage.suggest_gap_fills / curation_ops.fill_gap). These rows have a
# position but no mask, so area/morphometry are absent for them; statistics that
# need a real segmentation can exclude them via this flag.
COL_INTERPOLATED = "interpolated"

INTERNAL_NUMERIC_COLS = (COL_TRACK, COL_FRAME, COL_X, COL_Y)

# Candidate source names per target column, ordered by likelihood. Used to
# pre-fill the column-mapping dialog with sensible defaults for common trackers.
COLUMN_CANDIDATES = {
    COL_TRACK: ["track_id", "trackId", "TrackID", "TRACK_ID", "track", "particle",
                "label", "Label", "lineage", "spot_id", "cell_id", "Cell_ID",
                "CellID", "CELL_ID", "id", "ID"],
    COL_FRAME: ["frame", "Frame", "FRAME", "FRAME_INDEX", "t", "T", "time", "Time",
                "TIME", "POSITION_T", "timepoint", "time_point", "slice", "Slice"],
    COL_X: ["pos_x", "x", "X", "POSITION_X", "centroid_x", "Centroid_X", "CENTROID_X",
            "centroid-1", "Center_of_the_object_0", "center_x", "Center_X",
            "x_pixel", "x_px", "XM"],
    COL_Y: ["pos_y", "y", "Y", "POSITION_Y", "centroid_y", "Centroid_Y", "CENTROID_Y",
            "centroid-0", "Center_of_the_object_1", "center_y", "Center_Y",
            "y_pixel", "y_px", "YM"],
}

# Names used by common trackers for the outcome / parent columns. Used to
# auto-detect already-curated CSVs.
OUTCOME_CANDIDATES = ["Desfecho", "outcome", "Outcome", "OUTCOME", "fate", "Fate", "flag"]
PARENT_CANDIDATES = ["Parent_ID", "parent_id", "ParentID", "PARENT_ID", "mother_id",
                     "MotherID", "parent", "Parent"]


# ---------------------------------------------------------------------------
# Outcome flags (cell-cycle fate)
# ---------------------------------------------------------------------------
OUTCOME_MITOSIS = "Mitosis"
OUTCOME_EXIT = "Exit"               # left the field of view
OUTCOME_DEATH = "Death/Senescence"
OUTCOME_AMBIGUOUS = "Ambiguous"     # reviewed but none of the others fit
OUTCOME_UNKNOWN = "Unknown"
# "Empty-ish" outcome values: treated as "not curated yet". NOTE: "Ambiguous" is
# intentionally NOT here -- it is a real, deliberate flag (counts as curated),
# whereas "Unknown" remains the internal "needs review / not a real flag" marker.
NON_FLAG_VALUES = ["", "nan", "None", OUTCOME_UNKNOWN]
FINAL_OUTCOMES = (OUTCOME_MITOSIS, OUTCOME_EXIT, OUTCOME_DEATH, OUTCOME_AMBIGUOUS)

# Translation of foreign / legacy outcome labels onto the internal vocabulary.
# Keys are compared case-insensitively. Used at load time so an old Portuguese
# CSV is understood automatically. Anything mapping to "" is treated as a
# non-final marker to be dropped (e.g. a former start-of-life flag).
OUTCOME_TRANSLATIONS = {
    "fim": OUTCOME_EXIT,
    "saida": OUTCOME_EXIT,
    "saída": OUTCOME_EXIT,
    "exit": OUTCOME_EXIT,
    "mitose": OUTCOME_MITOSIS,
    "mitosis": OUTCOME_MITOSIS,
    "divisao": OUTCOME_MITOSIS,
    "divisão": OUTCOME_MITOSIS,
    "morte": OUTCOME_DEATH,
    "senescencia": OUTCOME_DEATH,
    "senescência": OUTCOME_DEATH,
    "death": OUTCOME_DEATH,
    "death/senescence": OUTCOME_DEATH,
    "ambiguous": OUTCOME_AMBIGUOUS,
    "ambiguo": OUTCOME_AMBIGUOUS,
    "ambíguo": OUTCOME_AMBIGUOUS,
    "ambigua": OUTCOME_AMBIGUOUS,
    "ambígua": OUTCOME_AMBIGUOUS,
    "indefinido": OUTCOME_AMBIGUOUS,
    "incerto": OUTCOME_AMBIGUOUS,
}
# Legacy start-of-life markers that must be discarded (replaced by the track's
# real final outcome). Compared case-insensitively.
START_MARKERS = {"inicio", "início", "start", "begin", "nascimento"}


# ---------------------------------------------------------------------------
# Treatment phases
# ---------------------------------------------------------------------------
TREAT_CONTROL = "control"
TREAT_TREATED = "treated"
TREAT_WASHOUT = "washout"


# ---------------------------------------------------------------------------
# X-ray filter choices and layer names
# ---------------------------------------------------------------------------
FILTER_ALL = "Show all"
FILTER_MITOSIS = "Mitosis only"
FILTER_EXIT = "Exit only"
FILTER_DEATH = "Death only"
FILTER_AMBIGUOUS = "Ambiguous only"

LAYER_RAW = "raw_video"
LAYER_MASK = "cell_mask"
LAYER_TRACKS = "tracks"
LAYER_IDS = "ids"
LAYER_XRAY = "xray_view"


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
GROUP_COLORS = {
    TREAT_CONTROL: "#4C72B0",
    TREAT_TREATED: "#DD8452",
    TREAT_WASHOUT: "#55A868",
    "mitotic": "#C44E52",
    "non-mitotic": "#8172B3",
}

# Above this many on-screen ID labels, the viewer switches to drawing labels for
# the CURRENT frame only (instead of every cell in every frame). Tens of
# thousands of overlapping text labels are both illegible and very slow to
# render in napari, so this keeps large movies responsive. The track paths and
# masks are still shown in full; only the text labels are thinned.
MAX_ID_LABELS = 3000

# Above this many track vertices (rows fed to the tracks layer), the colored
# trajectory tails are shortened so the GPU does not have to draw hundreds of
# thousands of connected line segments at once (a common cause of the viewer
# hanging on a white screen). The tracks are still all present; only the
# trailing length drawn per cell is reduced. Set very high to disable.
MAX_TRACK_VERTICES = 60000
SHORT_TAIL_LENGTH = 12


# ---------------------------------------------------------------------------
# Reserved ID ranges (namespacing)
# ---------------------------------------------------------------------------
# A single, coherent layout for the positive-integer ID space. Every operation
# that mints special IDs and every filter that must ignore them refer to THESE
# constants, so the diagnostic/save filters always cover exactly the ranges the
# operations create.
#
#   [1                 .. NORMAL_ID_MAX)    normal, curatable cell IDs
#   [QUARANTINE_START  .. QUARANTINE_END)   evicted / orphaned cells (ignored by
#                                           diagnostics, dropped on save)
#   [RESEQ_OFFSET      .. ...)              transient offset used only *inside*
#                                           the mask<->track_id relabel; never
#                                           persisted, never on disk.
#
# The pool's free-ID cursor only ever hands out IDs in [1, NORMAL_ID_MAX).
#
# The normal range is deliberately large (up to 9-digit IDs) for backward
# compatibility: the old "Re-sequence tree" op (now replaced by display-only
# genealogy labels -- see lineage.hierarchical_labels) numbered cells by
# concatenating each generation's branch index (1 -> 11, 12 -> 111, ...), so
# datasets curated with it may have 7-9 digit IDs baked into the table and mask.
# Those must stay BELOW the quarantine range or a legitimate cell gets misread as
# evicted and vanishes from the feature export. The whole layout stays under the
# uint32 mask limit (2**32 ~ 4.29e9): the largest value ever put in the mask is a
# normal id (< NORMAL_ID_MAX) plus RESEQ_OFFSET, i.e. < 3e9 < 4.29e9.
NORMAL_ID_MAX = 1_000_000_000      # normal IDs live strictly below this (<=9 digits)
QUARANTINE_START = 1_000_000_000   # evicted/orphan "graveyard" range start
QUARANTINE_END = 1_500_000_000     # ... and end (exclusive)
RESEQ_OFFSET = 2_000_000_000       # transient collision-avoidance offset (uint32-safe)


def is_normal_id(value: int) -> bool:
    """True for IDs in the normal, curatable range."""
    return 0 < int(value) < NORMAL_ID_MAX


def is_quarantined_id(value: int) -> bool:
    """True for IDs that have been evicted into the quarantine range."""
    return QUARANTINE_START <= int(value) < QUARANTINE_END


# ---------------------------------------------------------------------------
# Curation thresholds (no more magic numbers in the code)
# ---------------------------------------------------------------------------
@dataclass
class Thresholds:
    """Tunable thresholds for anomaly detection and quality control.

    Defaults reproduce the original hard-coded behaviour (40 px jump,
    20-frame minimum). They can also be derived automatically from a dataset
    via thresholds_from_dataset() so the curator adapts to different
    magnifications and frame rates.
    """
    min_valid_frames: int = 20          # tracks shorter than this are "short"
    max_jump_px: float = 40.0           # per-frame displacement above this is a "jump"
    gap_tolerance: int = 1              # frame gaps larger than this count as a gap
    # Morphology, in robust-statistics (MAD) multiples:
    area_mad_k: float = 3.5             # |area - median| > k*MAD  -> flagged
    min_solidity: float = 0.85          # solidity below this -> concave/leaky mask
    max_eccentricity: float = 0.98      # eccentricity above this -> sliver/artefact
    # Mitosis lifetime sanity (used by diagnostics):
    mitosis_life_mad_k: float = 4.0     # flag if |life-median| > k*MAD (robust)
    # Assisted gap relinking:
    relink_search_radius_px: float = 60.0
    # Identity-swap detection: two co-existing tracks within this distance at a
    # frame whose next-frame assignment is cheaper when exchanged.
    swap_search_radius_px: float = 40.0
    swap_cost_margin: float = 0.25      # swapped cost must beat kept cost by this fraction
    swap_min_improvement_px: float = 1.0  # and by at least this many pixels

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls().to_dict()}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def thresholds_from_dataset(df, base: "Thresholds | None" = None) -> "Thresholds":
    """Derive data-adaptive thresholds from per-frame displacement statistics.

    The jump threshold is set to a high percentile of the observed per-step
    displacement distribution, and the minimum-track-length to a small fraction
    of the movie duration. Falls back to the supplied/base defaults when there
    is not enough data to estimate robustly.
    """
    import numpy as np

    t = base or Thresholds()
    try:
        valid = df.dropna(subset=[COL_TRACK, COL_FRAME, COL_X, COL_Y]).copy()
        valid = valid[valid[COL_TRACK].astype(float) > 0]
        if valid.empty:
            return t
        valid = valid.sort_values([COL_TRACK, COL_FRAME])

        steps = []
        for _, g in valid.groupby(COL_TRACK):
            if len(g) < 2:
                continue
            x = g[COL_X].to_numpy(dtype=float)
            y = g[COL_Y].to_numpy(dtype=float)
            steps.append(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        if steps:
            allsteps = np.concatenate(steps)
            allsteps = allsteps[np.isfinite(allsteps)]
            if allsteps.size >= 20:
                # 99.5th percentile of normal motion, with a safety factor.
                t.max_jump_px = float(np.percentile(allsteps, 99.5) * 1.5)

        n_frames = int(valid[COL_FRAME].max() - valid[COL_FRAME].min() + 1)
        if n_frames > 0:
            # At least 5% of the movie, but never below 3 frames.
            t.min_valid_frames = max(3, int(round(n_frames * 0.05)))
    except Exception:
        # Auto-derivation is best-effort; never let it break loading.
        return base or Thresholds()
    return t
