# Napari-TRACE
## Tracking Review and Correction Engine
[![DOI](https://zenodo.org/badge/1264173831.svg)](https://doi.org/10.5281/zenodo.20653779)

A [napari](https://napari.org)-based desktop tool for reviewing and correcting
automated cell-tracking results from time-lapse microscopy. It overlays the raw
movie, the segmentation mask and the tracking table, gives you a full set of
tools to fix identity swaps, merges, orphan masks and misassigned divisions,
lets you record each cell's outcome and build its lineage, and ships a
statistics panel that turns the curated data into publication-ready figures and
a per-track summary.

The design goal is **curation by exception**: on large datasets you do not
review every cell. The tool scores each track's confidence, sends you to the
least trustworthy ones first, lets you bulk-accept the rest, and lets you
validate that accepted batch with a random sample and an empirical error rate.

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project layout](#project-layout)
- [Running](#running)
- [Input data](#input-data)
- [The loading flow](#the-loading-flow)
- [Core concepts](#core-concepts)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Curation tools](#curation-tools)
- [Statistics tools](#statistics-tools)
- [Generated files](#generated-files)
- [Recommended first-time workflow](#recommended-first-time-workflow)
- [Known limitations](#known-limitations)

---

## Features

- Overlay of raw video, segmentation mask, trajectories and ID labels in napari.
- Optional extra **fluorescence channels** (apoptosis reporters such as cleaved
  Caspase-3, cell-cycle indicators such as FUCCI, viability dyes, ERK-KTR,
  53BP1) loaded as image layers, tagged at import with a display color and an
  optional **measure** flag. Unmarked channels are display-only context you
  flash on at the frame an anomaly is flagged, to turn an "Ambiguous"/"Death"
  call into a biological one; marked channels are additionally passed through
  the fluorescence engine to compute per-cell intensity, ratio and texture
  features. The tracking mask is always the only segmentation — channels
  never contribute to it, measured or not.
- Identity fixes: merge, swap, cut, relabel, delete (single frame or whole track).
- Mask reconciliation: sync the table to the mask, rescue orphan masks, split
  disconnected blobs, auto-track a single cell across the movie.
- Outcome flags: mitosis, exit, death/senescence, ambiguous.
- Lineage editing with a confirmation step so a manual link always wins, a visual
  per-cell lineage editor (parent + daughters), a topology validator and a
  re-sequencing pass that numbers families by generation.
- Diagnostics: distributions, flagged-cell lists, morphology outliers, an
  identity-swap detector for label exchanges that leave no jump or gap, and a
  mass-balance check for unexplained cell-count changes.
- Triage queue driven by within-track tracking-error signals, with biological
  outliers (cells far from the population) surfaced separately instead of being
  treated as errors; bulk-accept, and a random validation sample that yields an
  empirical error rate with a 95% confidence interval.
- Assisted gap relinking with a choice of predictor: linear extrapolation, or a
  clamped cubic **smoothing spline** fitted on the frames flanking the gap (for
  curved trajectories). A separate **gap-fill** tool synthesizes interpolated
  centroid rows in the missing frames (marked `interpolated`) so motility/MSD see
  a continuous path.
- Nuclear morphometry: per-frame Nuclear Irregularity Index (NII) and its
  components (aspect, area/box, radius ratio, roundness) computed from the mask,
  plottable and exported.
- Statistics: lifetime, migration, motility, area, growth, outcomes, divisions,
  MSD, a temporal-gradient boxplot, per-cell trajectory plots and a custom plot.
- Video export (single cell or full screen) and a per-track CSV summary.
- Atomic, backed-up saving that never touches your original files, and writes
  derived export tables on save (per-frame features, accumulated time-windows,
  per-track summary, plus a validated-cells copy of each).

---

## Requirements

Python dependencies (see [`requirements.txt`](requirements.txt)):

- `napari` with a Qt backend (`napari[all]`)
- `magicgui`, `qtpy`
- `numpy`, `pandas`
- `scikit-image`, `tifffile`
- `scipy` — used for the smoothing-spline gap predictor; if absent, the spline
  option falls back to linear automatically
- `matplotlib`
- `imageio[ffmpeg]` — optional, only for the video-export buttons

A working display is required: napari is an interactive GUI and will not run on
a headless server without a virtual display.

---

## Installation

A clean conda environment is recommended. Conda creates the environment and the
Python interpreter; the packages are installed with `pip` from
`requirements.txt`.

```bash
# 1. Create the environment (Python 3.10 or 3.11 work well with napari)
conda create -n curator python=3.11

# 2. Activate it
conda activate curator

# 3. Install the dependencies
pip install -r requirements.txt
```

Notes:

- `requirements.txt` installs napari and its Qt backend through `napari[all]`.
  Do **not** also run `conda install napari` in the same environment; installing
  napari through two package managers commonly produces a broken Qt backend.
- If you prefer napari from conda-forge, use
  `conda create -n curator -c conda-forge napari pyqt` and then
  `pip install -r requirements.txt` for the rest, after removing the
  `napari[all]` line from `requirements.txt`.

Plain pip (no conda) also works:

```bash
pip install -r requirements.txt
```

---

## Project layout

`run_curator.py` is a thin entry point that imports the `curator` package, so it
sits one level above the package folder. Run the command from the directory that
contains `curator/`.

```text
project/                       # run from here
├── run_curator.py             # entry point: from curator.app import main
├── requirements.txt
└── curator/                   # the package
    ├── __init__.py
    ├── app.py                 # Qt bootstrap, argument parsing, load flow
    ├── config.py              # column schema, vocabulary, thresholds, ID ranges
    ├── state.py               # session state, undo history, ID pool
    ├── data_io.py             # load/save (backup + atomic write)
    ├── io_adapters.py         # format adapters (TrackMate, CTC) and file discovery
    ├── channels.py            # extra fluorescence channels (display-only loading)
    ├── dialogs.py             # startup dialogs (column mapping, treatment)
    ├── treatment.py           # treatment-phase logic (control/treated/washout)
    ├── curation_ops.py        # curation operations (merge, swap, flags, lineage)
    ├── analysis.py            # centroids, anomalies, morphology, per-track summary
    ├── lineage.py             # genealogy (classification, validation, relinking)
    ├── triage.py              # confidence scoring and the triage queue
    ├── validation.py          # validation-sample session and reliability plot
    ├── validation_dialog.py   # Qt window for sample validation
    ├── review.py              # persistence of reviewed lineages
    ├── validated.py           # persistence of individually validated cells
    ├── triage_review.py       # persistence of triage checkboxes
    ├── exports.py             # derived export tables (features, windows, validated)
    ├── audit.py               # action audit log
    ├── stats.py               # statistics and all plots
    ├── tree_plot.py           # lineage tree plot
    ├── track_plots.py         # track-plot utilities
    ├── ui.py                  # napari viewer and panel assembly
    └── ui_panels.py           # auxiliary panels (diagnostics, triage, relink)
```

Experiment data (CSV, mask, image) lives **outside** this tree, in the folder you
pass as an argument; the tool creates a working copy named `<experiment>_curated`
next to it.

---

## Running

```bash
python run_curator.py /path/to/experiment_folder
```

The folder argument is optional. Without it, a folder-picker dialog opens:

```bash
python run_curator.py
```

### Command-line arguments

| Argument | Description |
|---|---|
| `folder` (positional) | Folder containing the CSV, mask and image. |
| `--csv PATH` | Point at the CSV/TXT explicitly (skips auto-discovery). |
| `--mask PATH` | Point at the mask `.tif` explicitly. |
| `--image PATH` | Point at the image `.tif` explicitly. |
| `--channel PATH` | Add an extra fluorescence channel (TIF stack, frame folder, or multi-channel TIF). Repeatable; display-only, never modified or saved. |
| `--auto-thresholds` | Default. Derive jump/length thresholds from the dataset. |
| `--no-auto-thresholds` | Use fixed thresholds (40 px jump, 20-frame minimum). |

Pointing at each file directly:

```bash
python run_curator.py --csv tracks.csv --mask labels.tif --image raw.tif
```

---

## Input data

Three inputs from the same experiment (same frame count, same image size):

1. **Tracking table** — a `.csv` or `.txt`. Supported formats:
   - Generic CSV from any tracker (you map the columns at load time).
   - TrackMate CSV (auto-detected).
   - Cell Tracking Challenge `res_track.txt` (no header, columns `L B E P`:
     label, begin frame, end frame, parent).
2. **Segmentation mask** — a `.tif` stack (T×H×W, one integer label per cell) or
   a folder with one `.tif` per frame.
3. **Raw image** — the microscopy movie, also a `.tif` stack or a folder of frames.

**Optional — extra fluorescence channels.** Any number of additional channels
(e.g. a cleaved-Caspase-3 apoptosis reporter, a FUCCI cell-cycle indicator, a
viability dye, ERK-KTR, 53BP1) can be loaded alongside the movie. Two storage
conventions are accepted, and the tool handles either:

- **Separate files / folders, one per channel** — each its own TIF stack
  (T×H×W) or its own folder of per-frame TIFs. Auto-discovered inside the
  experiment folder when the file/folder name contains a channel token
  (`caspase`, `fucci`, `gfp`, `mcherry`, `dapi`, `_ch1`, …), or passed
  explicitly with `--channel`.
- **A single multi-channel TIF** — one stack carrying a channel axis (T×C×H×W
  or T×H×W×C); the channel axis is detected and split into one layer each.

Each channel is tagged at import (the "Fluorescence channels" dialog) with a
display **color** and an optional **measure** flag. These channels are
**never** treated as segmentation — the single nuclear mask (the tracking
mask) is always the segmentation, and channels stay out of the mask, the ID
pool and the tracking table regardless of the measure flag. An unmarked
channel is pure read-only context; a marked one is additionally passed
through the fluorescence engine to produce per-cell intensity/ratio/texture
columns in the export tables (see "Fluorescence channels & features" under
Statistics tools). They load hidden so they don't obscure the curation view;
toggle them with `v` (see shortcuts).

If your CSV already has an outcome column (`outcome`, `fate`, ...) or a parent
column (`parent_id`, `mother_id`, ...), it is detected and preserved.

---

## The loading flow

1. **Working copy.** On first open of a folder, a sibling folder
   `<name>_curated` is created and the whole source is copied into it — both
   top-level files and subfolders (e.g. per-frame TIF folders for the mask,
   image or extra channels). Everything, including saving and any derived
   stacks, happens on that copy; the original folder is never modified.
   Reopening the `_curated` folder resumes the session (recognized via
   `curator_meta.json`).
2. **File discovery.** The tool tries to find the CSV (names containing
   "track"), the mask (names containing "mask"/"label"/"seg") and the image (the
   remaining `.tif`). Anything it cannot resolve is requested through a dialog;
   for mask and image it asks whether the source is a folder of frames or a
   single stack.
3. **Column mapping** (CSV only). The "Map CSV columns" dialog asks which of your
   columns are Track ID, Frame, X and Y, pre-filled with guesses from common
   names. CTC `.txt` files skip this step.
4. **Mismatch warning.** If image and mask differ in frame count or size you get
   a non-blocking warning. You can proceed, but area and centroid measurements
   will be wrong if the inputs are genuinely mismatched.
5. **Mask ↔ track_id check.** The tool needs each cell's mask painted with its
   `track_id` (clicks, focus and every mask-derived feature — area, morphometry,
   fluorescence — rely on it). If the mask instead uses per-frame segmentation
   labels, a dialog reports the low match and offers to relabel the mask (by each
   row's centroid) so pixel value == `track_id`; the corrected mask is written
   into the working copy. Declining leaves the mask as-is (mask-derived features
   would be NaN).
6. **Automatic thresholds** (default). The anomaly thresholds are derived from
   the data: the maximum per-frame jump is the 99.5th percentile of observed
   displacement times 1.5, and the minimum valid track length is 5% of the movie
   (never below 3 frames). Use `--no-auto-thresholds` for fixed values.
7. **Treatment setup.** The "Treatment setup" dialog defines whether the movie is
   all control or has a treatment window.
8. **Automatic flags on load.** Before you touch anything: any track that is a
   parent of another is flagged Mitosis; a cell that touches the border and
   shrinks to nothing before the end is flagged Exit; legacy/foreign outcome
   labels are translated to the internal vocabulary. Automatic flags never
   overwrite an outcome you already set, but they are heuristics — review them.

---

## Core concepts

**Layers.** `raw_video` (grayscale image), `cell_mask` (colored labels),
`tracks` (trajectories), `ids` (numeric labels), plus one image layer per extra
fluorescence channel when supplied (hidden by default, toggled with `v`). Two
side tabs: **Curation tools** and **Statistics**.

Within each tab the tools are grouped into **titled, collapsible sections**
(SETUP, NAVIGATION & VIEW, BASIC CURATION, OUTCOME FLAGS, LINEAGE, DIAGNOSTICS &
EXPORT on the curation tab; QUICK STATISTICS, TEMPORAL GRADIENT, TRAJECTORY
PLOTS, CUSTOM PLOT, FLUORESCENCE on the statistics tab). Click a section title
to fold or unfold it, so the groups you are not using stay out of the way. The
selection inputs ([A]/[B], progress) and the statistics calibration/filters sit
in a fixed header above the sections and are always visible. The rarely-used
groups (SETUP, DIAGNOSTICS & EXPORT, and the advanced plot builders including
FLUORESCENCE) start collapsed.

**The two working IDs, [A] and [B].** Most operations act on one or two IDs in
the input boxes. [A] is the primary target / mother; [B] is the destination /
daughter. Fill them by clicking in the viewer:

- **Shift + click** → puts that cell's ID in **[A]**.
- **Ctrl + click** → puts that cell's ID in **[B]**.

**Outcomes.** Each cell can have one final outcome: Mitosis, Exit,
Death/Senescence, or Ambiguous (a deliberate "reviewed but none of the above").
An empty outcome means the cell is not curated yet.

**Lineage.** The mother→daughter relationship is stored in `parent_id`. A family
is the whole connected genealogy; its root is the oldest ancestor. Linking a
mother to a daughter flags the mother as Mitosis.

**Reserved ID ranges.** Normal editable IDs: 1–999,999. Quarantine (evicted /
orphan cells, ignored in diagnostics and dropped on save): 1,000,000–2,000,000. A
transient offset is used only during re-sequencing and never reaches disk.

**Nuclear morphometry (NII).** From the mask, the tool computes per cell and
frame the Nuclear Irregularity Index after Filippi-Chiela et al. (2012):
`NII = aspect − area/box + radius_ratio + roundness`, where aspect = major/minor
axis, area/box = object area ÷ bounding-box area, radius_ratio = max÷min
centroid-to-boundary radius, and roundness = perimeter² ÷ (4·π·area). A perfect
disc sits near 2.2; irregular, elongated or multilobed nuclei score higher. The
NII and its components are available as statistics (trajectory timeseries, custom
plot via the per-track summary) and in the export tables.

**Undo.** Every operation is undoable with **Ctrl+Z** or the Undo button; the
history holds the last 10 actions. Operations that rewrite the whole movie
snapshot the entire stack; the rest snapshot only the frames they touch.

---

## Keyboard shortcuts

Most-used tools have single-key shortcuts so you can curate without leaving the
canvas. They act on the current **[A]**/**[B]** selection and the current frame,
exactly like the matching buttons (selection still comes from Shift/Ctrl+click).

Each tool's button also shows its key in brackets (e.g. `Merge (...)  [g]`), so
the shortcut is discoverable from the panel.

The shortcuts are registered with `overwrite=True`, so they always fire even when
napari binds the same key. The **manual-painting keys are deliberately left to
napari**: the number row (Labels layer modes), `m` (new label) and the brush keys
keep their napari meaning, so painting masks by hand is unaffected.

| Key | Action |
|---|---|
| `Shift` + click | Set **[A]** (target / mother) |
| `Ctrl` + click | Set **[B]** (destination / daughter) |
| `d` | Flag **[A]** as Mitosis (division) |
| `w` | Flag **[A]** as Exit (left the field) |
| `k` | Flag **[A]** as Death / Senescence |
| `b` | Flag **[A]** as Ambiguous |
| `c` | Clear flags of **[A]** |
| `g` | Merge **[A]** into **[B]** (whole movie) |
| `s` | Swap / cut from this frame onward |
| `Shift` + `s` | Local swap (this frame only) |
| `n` | Relabel **[A]** mask to a new ID |
| `y` | Sync masks (this frame) |
| `x` | Delete **[A]** (this frame) |
| `l` | Link mother **[A]** → daughter **[B]** |
| `f` | Toggle Focus mode |
| `Shift` + `f` | Toggle Lineage focus |
| `v` | Toggle extra fluorescence channels on/off (group) |
| `Shift` + `v` | Cycle which single channel is shown solo |
| `.` | Jump to next unreviewed lineage |
| `,` | Jump to next cell without lineage |
| `Ctrl` + `Z` | Undo |
| `Ctrl` + `S` | Save all (backup + atomic) |

Because the shortcuts are single keys on the canvas, a stray keypress while a cell
is selected can apply an unintended edit; `Ctrl+Z` reverts it.

---

## Curation tools

### Setup

**Set treatment / phases.** Opens the treatment dialog. *Control* makes the whole
movie control. *Treated* takes a start and end frame: frames before the start are
`control`, between start and end are `treated`, after the end are `washout` (no
washout if the end is the last frame). Treatment bands are shaded in the
time-based statistics plots.

> Example: a 0–200 movie treated from frame 50 to 120 → frames 0–49 control,
> 50–120 treated, 121–200 washout.

### Navigation and view

**Focus mode.** Isolates **[A]** only: hides every other mask, shows just its
trajectory and label. Click again to turn it off.

**Lineage focus.** Like focus mode, but highlights the **whole family** of [A]:
the ancestral root and every descendant. Useful for auditing a division tree at
once.

**Show tracks layer.** The trajectory layer is the most expensive thing to draw.
Cycles AUTO (draw only in focus/lineage views or when the dataset fits, up to
~60,000 vertices) → ON (always) → OFF (never). Switch to OFF if the screen
freezes on a large dataset.

**Mark this lineage as reviewed.** Marks the family of [A] (by root) as reviewed;
this persists across sessions in `lineage_review.json`. The "Lineages reviewed:
X / Y" indicator tracks progress.

**Jump to next unreviewed lineage.** Moves the viewer to the oldest cell of the
first family not yet marked reviewed. The engine of the "review family by family"
workflow.

**Jump to next cell without lineage.** Moves to the next cell ahead of the
current frame that has no outcome and no lineage — i.e. one not yet touched.

**Shuffle colors.** Re-randomizes label colors. Purely visual.

**Undo.** Reverts the last operation (also Ctrl+Z).

### Basic curation

**Merge (A becomes B across the movie).** Merges track [A] into [B] across the
whole movie. Daughters that pointed at [A] are repointed to [B] so the lineage
follows the rename. Use it when the tracker split one cell into two IDs.

> Example: cell 8 becomes cell 19 mid-movie by mistake. [A]=8, [B]=19, Merge.

> Dissolving a false mitosis: if [A] and [B] are in a mother–daughter
> relationship (e.g. a "division" of cell 1 into 2 and 3 was really a
> segmentation artefact and 1 and 2 were the same cell), merging them proves the
> division never happened. The merge therefore **dissolves the whole division**:
> every other daughter (here cell 3) is detached (`parent_id = -1`) and the
> mother's Mitosis flag is cleared. Self-loops and divisions that an edit leaves
> impossible (mother gone, daughter born at/before the mother) are repaired the
> same way after every structural edit (merge, swap, relabel, exterminate).

**Swap / cut (from here onward).** From the current frame on, swaps labels [A]↔[B]
in every later frame. If [B] is empty, a new ID is reserved, turning the action
into a cut ("from here on this is a new identity"). Outcomes and parent of the
swapped segment are cleared from the cut point on, since the identity changed.
Daughters born at/after the cut frame follow the swap; daughters born earlier keep
their original mother.

> Swap example: at frame 60 the tracker swapped cells 4 and 7. Go to frame 60,
> [A]=4, [B]=7, Swap. Cut example: from frame 90 cell 4 is actually a new cell.
> Go to frame 90, [A]=4, [B]=0, Swap.

**Local swap (this frame only).** The same, but only on the current frame.

**Relabel mask (new ID).** Renames the whole [A] track to a fresh ID, clearing
outcome and parent. Daughters of [A] are repointed to the new ID.

**Sync masks (this frame).** Reconciles the table with the mask on the current
frame: recomputes centroids, updates X/Y, adds rows for masks present but not in
the table, and removes rows for table entries with no mask. Use it after editing
masks by hand in napari.

**Sync masks (whole movie).** The same across all frames. Can be slow on large
movies.

**Harmonize colors.** Two actions: (1) splits disconnected blobs — if a label
appears as two separate components in one frame, the largest keeps the ID and the
smaller pieces get new IDs (flagged for review); (2) forces the mask value to
match the track ID. Use it to clean up fragments after other operations.

**Delete cell [A] (this frame).** Deletes [A] from the current frame only.
Keyboard shortcut: **X**.

**Exterminate track [A] (whole movie).** Deletes [A] from every frame. Daughters
that pointed at it lose their mother (`parent_id = -1`). The napari tracks layer is
updated in place after every edit (including this one), so it no longer disappears
from the layer list when a track is exterminated.

**Rescue orphan masks (assign new ID).** Scans all frames for masks present in the
image but absent from the table and creates a new row with a new ID for each one.

**Auto-tracking (ID [A] only).** Consolidates [A] across the movie in three
phases: (1) where the mask of A splits into several components, identify which
island is A and which is another track B by recorded centroids and auto-swap, or
keep only the largest component; (2) absorb any track whose centroid falls on an A
pixel, merging it into A — daughters of an absorbed track are repointed to A so
they are not orphaned; (3) fill gaps and recompute the centroid frame by frame.
Use with care on large datasets: it iterates over every frame.

### Outcome flags (act on [A])

- **Mark: MITOSIS**
- **Mark: EXIT (left the field)**
- **Mark: DEATH / SENESCENCE**
- **Mark: AMBIGUOUS (unsure)** — a deliberate "reviewed, none of the above".
- **Clear flags of ID [A]** — removes any outcome from [A].

> Example: you followed cell 22 and saw it die at frame 140. [A]=22, Mark: DEATH.

### Lineage / genealogy

**Link mother [A] → daughter [B].** Records [B] as a daughter of [A] and flags [A]
as Mitosis. Before linking it checks for impossible configurations (a cell cannot
be its own mother; both must exist; a daughter cannot be born at or before its
mother).

The manual link is the final word, but it confirms before overwriting. If the
daughter already has a different mother, or the mother already has two daughters,
a dialog explains exactly what will be overwritten and waits for OK. On OK the
daughter is reassigned (and, when over capacity, the latest-born existing daughter
is detached to make room); on Cancel nothing changes.

> Example: cell 5 divides into 12 and 13. Link [A]=5, [B]=12, then [A]=5, [B]=13.
> If you later link [A]=5, [B]=20 while 5 already has two daughters and 20 already
> has another mother, you get a confirmation listing both before it proceeds.

**Edit lineage of [A] (parent + daughters).** Opens a dialog that shows the
lineage of the cell in [A] and lets you edit it visually in one place, instead of
juggling the [A]/[B] boxes and separate buttons. It displays the cell's parent
(`Parent = <id>` or none) and the list of its daughters, and offers:

- **Add as daughter** — type an ID in "Target ID" and add it as a daughter of the
  current cell.
- **Remove selected daughter** — detach the daughter selected in the list.
- **Set as parent** — type an ID in "Target ID" and make it the parent of the
  current cell.
- **Remove parent (detach)** — detach the current cell from its mother.

Double-clicking the parent or a daughter jumps the viewer there and re-centers the
editor on that cell, so the dialog doubles as a small lineage browser. Every change
goes through the same operations as the buttons above, so the confirmation dialog
still appears when an edit would override an existing mother or a third daughter,
and undo/audit behave identically. Removing a parent does not clear the mother's
Mitosis flag (it may still have another daughter); clear it explicitly with "Clear
flags of ID [A]" if needed.

> Example: put 5 in [A], open the editor. You see `Parent = none` and daughters
> `12, 13`. Type 20 in Target ID and click "Add as daughter"; if that would exceed
> two daughters you get the confirmation. Double-click daughter 12 to jump to it and
> continue editing 12's own lineage.

**Cut post-mitosis ghosts.** After a mother divides it should not continue to
exist. This finds mothers still present from the frame a daughter was born and
cuts that ghost segment to a new ID (clearing outcome and parent).

**Show lineage tree plot.** Draws the genealogies, largest families first. Nodes
are labelled with a readable genealogy path derived from `parent_id` — the first
family is `1`, its daughters `1.1` and `1.2`, granddaughters `1.1.1`, `1.1.2`, and
so on (ordered by birth frame), with the real `track_id` shown small underneath.
This is a display-only translation: no IDs or masks are rewritten (so nothing can
collide with the reserved ID ranges), and the labels always reflect the current
lineage. Side controls set the maximum number of families (default 60) and whether
to include single-cell families (flagged cells with no relatives, off by default).
Very large trees are truncated to stay legible.

**Validate lineage topology.** Reports biologically impossible configurations: a
mother with more than two daughters, a daughter appearing at/before its mother, or
a mother still alive after dividing. It only reports; it does not fix.

### Diagnostics and export

**Open diagnostics panel.** A window with four tabs; double-clicking a listed cell
jumps to its frame and selects it.

- **Distributions** — histograms of track lifetime, per-frame displacement (with
  the jump-threshold line) and divisions per frame.
- **Flagged cells** — clickable lists of short tracks, tracks with temporal gaps,
  tracks with impossible jumps, and lineage violations. Tracks that already have a
  final outcome are not reported as short or gapped, which avoids a flood of false
  positives on a curated dataset.
- **Morphology** — cells with anomalous shape (see the morphology button).
- **Mass balance** — frame transitions where the change in cell count is not
  explained by divisions, exits/deaths or border entries. A positive residual
  suggests a spurious division or an appearing object; a negative one suggests a
  merge or a disappearing cell.

**Open triage queue (large datasets).** Scores each track from 0 to 1 (1 =
trustworthy) and lists the worst cells first with score and reason. The queue is
driven by a **tracking-error score** built only from within-track inconsistencies
— an impossible single-frame jump (likely ID swap), a temporal gap, an abrupt area
step (fusion/leak, e.g. a merge-split that fakes a mitosis), or an outcome that
contradicts the trajectory. **Population deviation** (how far a track sits from the
dataset's own distribution in area, motility and lifetime, via robust median + MAD)
is computed as a **separate** score and is **not** mixed into the queue: a cell
that is merely unusual is not evidence that it is wrong, so atypicality alone never
sends a cell to review. Cells far from the population are instead listed in a
**Biological outliers** section of the dialog for inspection — this is where real
rare phenotypes (a giant cell, an unusually long-lived one, a hyper-motile one)
show up without being mislabelled as errors. Each review row has a **checkbox**:
tick it once you have triaged that cell. The ticks are **saved between sessions**
(in `triage_review.json`, created on first use, so old curations get it from the
first reopen) and the dialog shows a "triaged X / N" counter. You review the
worst, **bulk-accept**
the confident remainder (above the cutoff, default 0.85, non-destructive — it only
logs that those cells left the queue), then **draw a validation sample** to validate
that batch. The penalty weights, the cutoff and the deviation parameters are
defaults at the top of `triage.py`; pass `blend_deviation=True` to `triage_queue`
to fold deviation back into the score (the legacy combined behaviour). The
"no outcome yet" penalty is **scaled by annotation coverage**: on a mostly
uncurated dataset it is near zero (almost everything is uncurated, so it carries
no information) and it grows back to full as you curate, so a lone uncurated
straggler in a finished dataset becomes notable again. The dialog summary reports
the current coverage.

**Detect identity swaps (no jump).** Finds frame transitions where two nearby
tracks most likely exchanged labels. A clean swap leaves no jump, no gap and
conserves the cell count, so the per-track checks miss it; this looks instead for
two co-existing tracks whose next-frame assignment is cheaper when **swapped** (A
lands where B was and vice versa). When per-frame area is available, each row is
marked (★) when **area continuity corroborates** the swap (A inherits B's area and
vice versa), which separates a real label swap from two cells that merely cross
paths. It does **not** edit anything: double-click a row to jump to the transition
with the pair loaded into **[A]** and **[B]**, then fix it with Swap/cut (`s`) or
Local swap (`Shift+s`). Search radius and cost margin are in the thresholds
(`swap_search_radius_px`, `swap_cost_margin`).

**Validation sample.** Step through 50 randomly sampled cells from the accepted
batch. Mark each OK; if one was wrong, edit it (any edit is captured) and mark OK.
A cell counts as an error if it was edited. When all are reviewed, **Show
reliability plot** shows global calibration (confidence score vs observed error
rate against the ideal line) and per-feature reliability (error rate among cells
the tool flagged vs cells it considered clean, for area, motility, lifetime and
outcome). The result is an empirical error rate with a 95% Wilson confidence
interval — the defensible "we validated the auto-accepted set by random sampling"
argument for a paper.

**Assisted gap relinking.** For each track with a temporal gap, the tool
predicts where the cell should be in the missing frame and proposes the nearest
candidate within the search radius (60 px by default). Double-click previews;
Approve merges the candidate into the track across the gap. The **Gap predictor**
dropdown selects how the position is predicted:

- **linear** — extrapolation from the last two known points (the original
  behaviour).
- **spline** — a cubic **smoothing** spline fitted on up to five frames on each
  side of the gap. Because an unconstrained interpolating cubic overshoots on
  sparse, noisy points (it can fling the predicted centroid off-screen), this
  path is deliberately conservative: it smooths rather than interpolates, uses
  points on both sides of the gap, clamps every prediction to the flanking
  bounding box, and **falls back to linear** for long gaps (> 12 frames), when
  fewer than four flanking points exist, or when `scipy` is unavailable.

**Fill gaps (interpolate trajectory).** A separate tool that, instead of
reconnecting an existing track, **synthesizes new centroid rows** for the same
track in the frames it was missing, using the same predictor (spline by
default). The synthesized rows carry a position but no mask and are marked with
an `interpolated` column, so area/morphometry stay blank for them and any
statistic that needs a real segmentation can exclude them; migration, MSD and
directionality, which only need centroids, become continuous across the gap.
Review per track (double-click to preview the first filled frame) and Approve
one, or Approve ALL. Gaps longer than the limit are skipped, on the assumption
that a long disappearance is more likely a genuine exit/re-entry than a dropout
worth inventing positions for.

**Detect segmentation errors (morphology).** Flags cells with anomalous shape:
area far above the frame median (robust median + MAD test, factor 3.5) suggesting a
merge; solidity below 0.85 suggesting a concave/leaked mask; eccentricity above
0.98 suggesting a thin fragment. Opens on the Morphology tab.

**Export cell [A] video.** Writes a cropped `.mp4` (a 50 px window around the
centroid) that follows [A] with its mask highlighted. Requires `imageio`.

**Export presentation (screen).** Records an `.mp4` of the whole napari screen,
frame by frame, exactly as displayed. Requires `imageio`.

**SAVE ALL (backup + atomic).** The final step. It runs a pre-save quality filter
(detects anomalous tracks and asks whether to remove and save clean, ignore and
save with errors, or cancel), checks for duplicate IDs (same cell twice in one
frame) and refuses to save if any exist, backs up the current files to
`backups/<timestamp>/`, writes atomically (temp file + rename), recomputes border
contact and area against the final mask, and reloads from disk. It overwrites the
CSV and mask inside the `_curated` working folder — never your originals.

After saving, it also writes a set of **derived export tables** into an
`exports/` subfolder (best-effort: a failure here never affects the saved CSV).
For both the full dataset and the **validated subset**, it writes:

- `<base>_validated_cells.csv` — the main table restricted to validated tracks.
- `<base>_features.csv` (+ `_validated`) — one row per (track, frame) with
  per-frame nuclear morphometry (area_px, perimeter, circularity, aspect,
  area_box, radius_ratio, roundness, NII, plus shape descriptors eccentricity,
  solidity, extent, orientation, axis_major_length, axis_minor_length),
  per-step migration (step displacement, cumulative path, net-from-start,
  instantaneous speed) and, for every **measured** fluorescence channel, its
  per-compartment intensity/texture columns (see "Fluorescence channels &
  features" under Statistics tools). Also carries a per-track **`lifetime`**
  column (`(last_frame − first_frame + 1) × frame interval`) written on a single
  row — the cell's **middle-of-life** frame — with NaN on every other row, so a
  mean of the column gives one value per cell and survives trimming of the
  first/last frames in a later cleanup.
- `<base>_fluorescence.csv` (full dataset only, written when at least one
  channel is marked "measure") — the same (track, frame) key restricted to
  just the fluorescence columns, for quick loading without the rest of the
  feature table.
- `<base>_windows.csv` (+ `_validated`) — one row per (track, frame), with a
  **sliding window of N frames centred on that frame** (**N set by "Window for
  accumulated export", default 7**; clamped at track edges, so `n_frames` is
  smaller near the ends), accumulating the variation over the window:
  displacement, delta path, persistence, area_cv, area_slope, path, net,
  speed_mean, and the per-window mean and delta of every feature (including NII).
- `<base>_summary.csv` (+ `_validated`) — the per-track statistics summary.

A cell is **validated** when its whole lineage was marked reviewed (all of the
family's cells are then included) or when it was individually confirmed in a
validation sample. This applies on the **next save** of any session, including
curations already in progress.

---

## Statistics tools

The **Statistics** tab. All figures open in separate matplotlib windows.

### Calibration and filters

- **Pixel size (µm/px)** — leave at 1.0 to work in pixels; set it to report
  distances and areas in micrometers. Affects every distance/area metric.
- **Frame interval (min/frame)** — leave at 1.0 to work in frames; set it to
  report time in minutes. Affects every temporal metric.
- **Exclude border-touching points** — when on, removes rows where the cell
  touches the frame border (partial measurements) from every plot.
- **Exclude interpolated (gap-filled) frames** — when on, removes the
  synthesized centroid rows created by the gap-fill tool from every plot and from
  the per-track summary. Off by default (including them is the point of filling a
  gap). Toggle it to generate the same centroid-based figure — migration, MSD,
  directionality — with and without the imputed frames, e.g. to show that a fill
  did not distort a reported number, without leaving the tool. The two exclude
  boxes are independent and compose.
- **Window for accumulated export (default 7)** — the window length (in frames)
  used for the `<base>_windows.csv` accumulated table written on SAVE ALL.
- **Plot NMA (Area vs NII, per cell-frame)** — a scatter of area (Y) against the
  Nuclear Irregularity Index (X) with **one point per (track, frame)**, so each
  nucleus contributes a point for every frame it lived; points are coloured by the
  track's final outcome. This reproduces the area-versus-NII view of the NMA
  method and is handy for spotting senescent (large, regular), apoptotic (small,
  regular) and irregular (high-NII) populations.
- **Compare cell [A] vs dataset (by group)** — opens a **paginated** window with
  **one metric per page** (Previous/Next), so nothing is squished. Each page shows
  the distribution of that metric as **violin/density plots** for several groups
  (All cells, Mitotic, Non-mitotic, and Control/Treated/Washout when present),
  with the cell in [A] drawn as a red line and its **robust Z-score per group**
  under each violin. The page title flags the cell as an **OUTLIER** when its
  robust Z against all cells is ≥ 3, so you can tell at a glance, characteristic
  by characteristic, where the cell stands out. Covers migration, area and the
  nuclear-morphometry metrics (including NII).

### Quick statistics

Each button works on the per-track summary (one row per cell), except where noted.

- **Lifetime** — boxplots of lifetime by treatment phase and by mitotic vs
  non-mitotic.
- **Migration** — boxplots of mean speed, net displacement and directionality
  (net displacement / total path length; near 1 = straight) by treatment.
- **Motility** — boxplots of diffusion coefficient (from an MSD fit at short lags,
  2D convention MSD = 4·D·t), persistence time (from velocity autocorrelation
  decay), mean turning angle, and confinement ratio, by treatment.
- **Area** — boxplot of mean area by treatment plus a mean-area-over-time curve
  with treatment bands. Area is measured from the mask.
- **Population growth curve** — unique cells per frame over time, with treatment
  bands. Works on the raw table.
- **Outcome distribution** — bar chart of how many tracks ended in each outcome.
- **Division events over time** — divisions per frame (by daughter birth frame),
  with treatment bands.
- **Mean squared displacement (MSD)** — ensemble MSD vs time lag; the slope
  indicates diffusive, sub- or super-diffusive motion.
- **Export per-track summary (CSV)** — one row per cell with every computed metric
  (lifetime, total distance, net displacement, speed, directionality, mean/max
  area, diffusion, persistence, turning angle, confinement, `anomalous_exponent`
  (MSD power-law exponent: ~1 = normal diffusion, <1 = sub-, >1 = super-diffusive),
  outcome, mitotic flag, first/last frame, plus mean nuclear morphometry:
  `mean_nii` and the mean of each component). This is the file you take to
  external analysis.

### Temporal-gradient boxplot

**Plot boxplot + temporal gradient.** One box per group (median/quartiles over all
cell-frame pairs) with one point per cell per frame overlaid, jittered inside the
box and colored on a blue→red gradient by frame (blue = early, red = late), so any
temporal drift shows as a color trend inside each box. Choose the quantity (Y)
— area or any numeric column — and how to group the boxes (X): `final_outcome`,
`treatment`, `is_mitotic` or `time` (time windows).

> Example: Y = area, X = final_outcome shows the area distribution of cells that
> ended in Mitosis vs Exit vs Death, and whether area drifts over time (by color)
> within each group.

### Trajectory plots (one line per cell)

**Plot trajectories.** One line per cell, colored by final outcome (Mitosis green,
Exit blue, Death red, Ambiguous orange), with the treatment window shaded. The
type is set by **Plot type**:

- **timeseries** — one quantity (`area_px`, a velocity column, `perimeter`,
  `circularity`, the nuclear-morphometry columns `aspect`, `area_box`,
  `radius_ratio`, `roundness`, `nii`, ...) vs frame; `perimeter`, `circularity`
  and the morphometry/NII columns are computed from the mask on demand.
- **cumulative** — running accumulation along each track of the chosen quantity:
  with a quantity selected it shows the **cumulative change of that quantity**
  (including `nii` and the other morphometry columns, computed from the mask on
  demand); with no quantity it falls back to cumulative euclidean displacement
  (the distance odometer). Earlier builds ignored the quantity here and always
  drew the odometer — that is fixed.
- **spider** — each cell's X/Y trajectory translated to start at the origin.
- **window** — a centered rolling-window statistic.

Auxiliary controls: window metric (`persistence`, `area_cv`, `area_slope`, `path`,
`net`, `speed_mean`, and the NII-variation metrics `nii_mean`, `nii_cv`,
`nii_slope` — the within-window mean, coefficient of variation and trend of the
Nuclear Irregularity Index), window size (default 11), smoothing (timeseries
only), max tracks (longest first, 0 = all), and label track IDs at line ends.
All plot titles, axes and legends (including outcome names) are in English, and
quantity titles are shown with friendly names (e.g. `nii` → "Nuclear Irregularity
Index").

### Custom plot

**Plot custom.** Free choice of axes and grouping. Pick the data source (per-frame
table or per-track summary), the X and Y columns, how to group (`none`, frame,
track_id, treatment, outcome, is_mitotic depending on source), the kind (scatter or
line), the line aggregation (mean, median, sum, count) and whether to show the
legend.

> Example: source = per-track summary, X = lifetime, Y = mean_speed, group by
> treatment, kind = scatter — shows whether longer-lived cells move slower, split
> by treatment phase.

### Fluorescence channels & features

The **FLUORESCENCE** section (collapsed by default) drives the fluorescence
engine: for each marked channel and (track, frame), intensity/texture are
measured in three mask-derived compartments — **nuc** (the nuclear label
itself), **ring** (a cytoplasm annulus around it) and **cell** (nuc + ring).
The ring is grown outward from the nucleus and never overlaps the nucleus
itself or a neighboring cell's nucleus, so a tightly packed field does not
bleed one cell's signal into another's ring.

- **Add fluorescence channel…** — load an extra channel without restarting: pick
  a single TIF **stack** or a **folder of per-frame TIFs** (same as the mask and
  image), tag its color and "measure" flag, and it is added as a layer and
  **copied into the `_curated` working folder**, so it reloads automatically in
  later sessions (recorded in `curator_meta.json`).
- **Cytoplasm channel (ERK-KTR)** and **Nucleus channel (53BP1)** — two separate
  pickers so both roles are selected at once. The ERK-KTR button and the ring
  preview use the cytoplasm channel; the 53BP1 button uses the nucleus channel.
- **Cytoplasm ring width (px)** (`dilation`) and **ring gap from nucleus (px)**
  (`gap`) — the ring starts `gap` px outside the nuclear boundary and is
  `dilation` px wide. Defaults are 2 px dilation, 1 px gap.
- **Background ROI (cell-free)** — a picker listing any Labels or Shapes layer.
  Leave it on `(auto: non-cell median)` to subtract the default per-frame
  background, or draw a Labels/Shapes layer over a genuinely empty region and
  select it to measure the background there instead (more accurate in confluent
  fields, where "outside every nucleus" still contains cytoplasm). The
  channel/ROI pickers refresh automatically as you add or remove layers.
- **Preview ring on a random nucleus** — opens a small-multiples figure showing
  the ring growing (dilation = 1, 2, 3, ...) on one real nucleus from the
  movie, overlaid on the channel and with neighboring nuclei shaded, so you can
  sanity-check the parameters against the actual cell density before running a
  full measurement.
- **Compute ERK-KTR C/N (cytoplasm channel → features)** — for the selected
  **Cytoplasm channel**, computes the ring/nucleus (cytoplasm/nucleus) median
  intensity ratio per (track, frame), plots it per track, adds it to the export
  as `<channel>_cn_ratio`, and writes `<base>_erk_ktr.csv`. A high C/N indicates
  active ERK signalling driving the KTR reporter out of the nucleus, per the
  ERK-KTR ring/annulus convention (Regot et al. 2014; Kudo et al. 2018).
- **Measure 53BP1 nuclear texture (nucleus channel → features)** — for the
  selected **Nucleus channel**, computes nuclear median intensity, its standard
  deviation and
  coefficient of variation, plus the Haralick texture set, as a proxy for
  DNA-damage (53BP1) foci graininess: a more textured (grainier) nucleus
  indicates more/brighter foci than a uniformly diffuse one. Writes
  `<base>_53bp1.csv`.

Both buttons accumulate their result in memory and merge it into
`<base>_features.csv` on the next **SAVE ALL**; they do not write to the main
tracking table.

Every per-compartment intensity statistic and the C/N ratio is
**background-subtracted** first: a per-frame background is subtracted and
clipped at zero. By default the background is the median intensity of pixels
outside every cell in that frame; if a **Background ROI** layer is selected, the
median inside that cell-free region is used instead. Haralick texture is
computed on the raw (not background-subtracted) nuclear intensity, scaled by its
own 1st/99th-percentile range.

The columns produced for every marked channel, in `<base>_features.csv` /
`<base>_fluorescence.csv`, are:

- `<channel>_<compartment>_<stat>` for `compartment` in `nuc`/`ring`/`cell` and
  `stat` in `mean`/`median`/`sum`/`min`/`max`/`std`/`p90`.
- `<channel>_cn_ratio` — ring-median ÷ nucleus-median (the cytoplasm/nucleus
  ratio used by KTR reporters).
- `<channel>_hara_contrast` / `_correlation` / `_energy` / `_homogeneity` /
  `_entropy` — Haralick texture (gray-level co-occurrence matrix, 16 gray
  levels, averaged over 4 angles) computed on the nuclear ROI.

---

## Generated files

Everything lives inside the `<experiment>_curated` working folder:

| File | Purpose |
|---|---|
| working copies of CSV, mask, image | The working data, overwritten on save. |
| `curator_meta.json` | Marks the folder as a curation session. |
| `backups/<timestamp>/` | Copy of the files before each save. |
| `curation_audit.log.csv` | One row per action (timestamp, action, IDs, detail). |
| `lineage_review.json` | Which lineages (by root) you marked reviewed. |
| `cell_validation.json` | Which individual cells you confirmed in a validation sample. |
| `triage_review.json` | Which triage-queue cells you ticked off as triaged. |
| `exports/<base>_*.csv` | Derived tables. On SAVE ALL: `validated_cells`, `features`, `windows`, `summary` (each also as a `_validated` copy), plus `fluorescence` when a channel is marked "measure". Written immediately by their buttons: `erk_ktr`, `53bp1` (see Fluorescence channels & features). |

The saved CSV uses the internal schema (`track_id`, `frame`, `pos_x`, `pos_y`,
`outcome`, `parent_id`, `continuous_label`, `treatment`, `at_border`, `area_px`)
and is read back as such when you reopen the `_curated` folder. If you used the
gap-fill tool, an extra `interpolated` boolean column marks the synthesized rows
and round-trips with the file.

---

## Recommended first-time workflow

1. Run on the experiment folder and confirm the column mapping.
2. Set the treatment (or leave it as control).
3. Keep automatic thresholds.
4. Open **Diagnostics** to see distributions and flagged-cell lists. On a large
   dataset, go straight to the **Triage queue**.
5. Use Shift/Ctrl+click to select IDs and the basic/lineage tools to fix swaps,
   merges, orphans and lineages. Mark outcomes.
6. Use **Lineage focus** + **Mark reviewed** + **Jump to next unreviewed** to sweep
   family by family.
7. Run **Validate lineage topology** and the **Mass balance** tab to catch what
   slipped through.
8. (Large datasets) bulk-accept the confident batch and validate with the random
   sample.
9. **SAVE ALL.**
10. Open the **Statistics** tab, calibrate pixel/frame, then generate figures and
    export the summary.

---

## Known limitations

- **Performance on large movies.** Harmonize, auto-tracking, rescue orphans and
  whole-movie sync iterate over every frame and, in places, row by row; they can
  take minutes on long movies with many cells. The trajectory layer is the
  heaviest to draw — set it to OFF if the screen freezes.
- **Memory on full-movie operations.** Merge, exterminate and harmonize snapshot
  the whole stack for undo; the history keeps the last 10 steps.
- **Swap / cut clears outcomes** of the swapped segment from the cut point on. If
  you had already set an outcome there, set it again.
- **The pre-save quality filter can delete tracks** if you choose "Remove and save
  clean". Tracks with a final outcome are exempt from the short/gap checks, but the
  jump check still applies to any track.
- **Automatic flags on load are heuristics.** Every mother becomes Mitosis and
  every cell that shrinks at the border becomes Exit. Review these; they do not
  overwrite outcomes you already set.
- **Sync masks cannot carry lineage.** It reconciles the table with the mask and
  has no record of identity changes. If you repaint a mask from ID 21 to 45 by
  hand, sync sees 21 disappear and 45 appear as unrelated, orphaning 21's
  daughters. To rename a cell and keep its lineage, use merge or relabel, which
  repoint the daughters.
- **Genealogy labels are display-only.** The lineage tree shows readable paths
  (`1.1.2`) derived from `parent_id` at plot time; the stored `track_id` is
  unchanged. Datasets curated with the removed "Re-sequence tree" op may still
  carry large baked-in IDs (up to 9 digits), which the reserved-range layout
  accommodates; IDs beyond that are not expected.
- **The mismatch warning does not block.** If you proceed with image and mask of
  different sizes, area and centroid measurements will be wrong.
- **Extra channels are not checked for alignment.** A fluorescence channel whose
  frame count or size differs from the movie is still loaded (with a printed
  warning), so a mismatched channel will overlay misaligned — and, if it is
  marked "measure", its per-cell features will be measured against the wrong
  frame/geometry too. Unmarked channels stay display-only and never enter any
  measurement, so they cannot corrupt the data. Trust the overlay, and any
  measured feature, only when the channel matches the movie.
- **Channel-axis detection is heuristic.** For a single multi-channel TIF the
  channel axis is inferred (the small non-time, non-spatial axis). An unusual
  layout — e.g. a genuine channel count above 8, or a movie with very few frames
  — can be guessed wrong; in that case split the channels into separate files and
  pass them with `--channel`.
- **Interpolated rows are positions without masks.** The gap-fill tool
  synthesizes centroids, not segmentation: those rows have `area_px`/morphometry
  blank (NaN) and are marked `interpolated`. They make centroid-only statistics
  (migration, MSD, directionality) continuous across a gap, but a spline fill is
  still a model of where the cell *probably* was, not a measurement — treat long
  fills with the same caution as any imputation. Gaps beyond the length limit are
  left unfilled on purpose.
- **The spline predictor degrades to linear silently.** On long gaps, sparse
  flanking data, or a missing `scipy`, the "spline" option returns the linear
  prediction. This is intended (it prevents overshoot), but it means choosing
  "spline" does not guarantee a curved fill — the status message names the method
  actually used per fill.
