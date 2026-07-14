# Fluorescence channels + expanded features — Design

Date: 2026-07-13
Status: Approved (design), pending implementation plan.

## Goal

Let the curator (a) import extra fluorescence channels as TIFF stacks tagged with
a color/role and optionally marked for measurement, and (b) export a richer
`_features.csv` (one row per `(track, frame)`) covering morphology, motility,
per-channel/per-compartment fluorescence intensity and Haralick texture, plus
two interactive readouts: ERK-KTR cytoplasm/nucleus ratio and a 53BP1 nuclear
texture proxy.

Hard constraints (from the user): scientific validity of every measure; the
existing curation/save path must not break; a single nuclear mask (`state.mask`,
the one used for tracking) is always the segmentation source.

## Scope decisions (locked)

- **Auto features per save:** only channels the user explicitly marks "measure"
  at import produce fluorescence features (keeps save fast on large movies).
- **Compartments:** nucleus + cytoplasm ring + whole-cell + C/N ratio.
- **Background:** auto per-frame = median of channel pixels where `mask == 0`,
  subtracted (clipped at 0) before any intensity/ratio.
- **ERK-KTR / 53BP1:** interactive buttons in the statistics panel; results are
  plotted, written to a dedicated CSV immediately, and merged into `_features.csv`
  on the next save.

## Deliverables (built in order, each its own commit + test)

### A. Channel import with color/role + threading to export

- `channels.ChannelLayer` gains `color: str` (napari colormap) and
  `measure: bool`.
- A load-time Qt dialog (in `dialogs.py`) lists every channel source (discovered
  or `--channel`) with: color combo (green/red/blue/magenta/cyan/yellow/gray),
  editable name, and a "measure (compute features)" checkbox. Defaults: color
  from the existing palette, `measure=False`.
- `app.main` passes the enriched `ChannelLayer`s to `build_viewer` (already does);
  `build_viewer` keeps them in closure and exposes the measurable arrays to
  `on_save`.
- The primary raw layer (53BP1-Apple, `LAYER_RAW`) is always available as a
  selectable measurement source for the interactive buttons, even though it is
  not a `ChannelLayer`.

### B. Feature engine + wiring

New module `curator/fluorescence.py` — pure numpy/scipy/skimage, no Qt, headless
unit-testable (mirrors `analysis.py`).

#### B1. Compartment segmentation (per frame, per label)

- nucleus = `mask == id`
- cytoplasm ring = `dilate(nucleus, gap+dilation) & ~dilate(nucleus, gap) & ~(mask>0)`
  — an annulus starting `gap` px outside the nucleus, `dilation` px wide,
  excluding **all** nuclei (avoids nuclear-edge bleed and neighbor-nucleus
  contamination). Defaults `dilation=2`, `gap=1` (Regot 2014 / Kudo 2018);
  appropriate for the user's ~30 px-diameter nuclei.
  `# ponytail: neighbor rings can overlap in confluence; Voronoi split if it matters`
- whole cell = nucleus ∪ ring
- Implemented with bbox crops per label for speed.
- **`dilation` and `gap` are user-modifiable** (spinboxes in the statistics
  panel) with a **preview**: `fluorescence.ring_preview_figure(mask, channel,
  track_id=None, dilations=(1,2,3,4), gap=1)` returns a matplotlib figure of one
  real nucleus (random if `track_id` is None) as small multiples showing the ring
  growing at increasing `dilation` over the channel intensity, with other nuclei
  shaded to make the exclusion visible. A "Preview ring" button renders it.

#### B2. Background

`background_per_frame(channel, mask)` = `np.median(channel[mask == 0])` per frame;
subtracted and clipped at 0 before measurement. Returns the per-frame value too
(logged, not a feature).

#### B3. Intensity measures — per `(frame, track, compartment)`

mean, **median (primary)**, integrated (sum of bg-subtracted pixels ≈ total
protein), min, max, std, p90. Ratio **C/N = median_cyto / median_nucleus**
(convention recorded: high C/N = active ERK). Columns are prefixed by channel
name and compartment, e.g. `green_nuc_median`, `green_ring_median`, `green_cn_ratio`.

#### B4. Haralick texture (GLCM)

`skimage.feature.graycomatrix` / `graycoprops` (already a dependency — no new
package). Nuclear ROI cropped to bbox, quantized to **16 levels** (default;
configurable), distances=(1,), 4 angles averaged (rotation-invariant): contrast,
correlation, energy (ASM), homogeneity, plus entropy. 16 levels (not 32) because
a ~30 px-diameter nucleus is ~700 px: a 32×32 GLCM would be too sparse to give a
stable texture estimate; levels² should stay well below the pixel count. Only for
marked channels / via button (the expensive path).

#### B5. Morphology additions (free, always-on)

In `analysis.nuclear_morphometry`, add to the existing `regionprops` pass:
`eccentricity, solidity, extent, orientation, axis_major_length,
axis_minor_length`. Already merged into `features_table`, so no perf cost.

#### B6. Motility addition

In `analysis.motility_metrics`, add anomalous exponent `alpha` = slope of a
log–log MSD fit (per-track, in the summary). Existing MSD/persistence/
directionality/diffusion stay.

#### B7. Interactive readouts (statistics panel)

- **"Calcular razão C/N (ERK-KTR)"**: combos to pick the channel layer and the
  nuclear labels layer → per-`(track,frame)` C/N (median-based, bg-subtracted);
  plots the time series; writes `<base>_erk_ktr.csv` immediately; injects the
  `*_cn_ratio` column into the next `_features.csv`.
- **"Medir textura nuclear (53BP1)"**: on the chosen channel, per-`(track,frame)`
  nuclear intensity stats + variance + std + CV + Haralick contrast/entropy (the
  foci "graininess" proxy). Same plot + persist behavior.
- Persistence: a `build_viewer` closure dict `extra_feature_tables`
  (`{colname_or_frame: per-(track,frame) DataFrame}`) that `on_save` passes to
  `exports.write_all` for a left-merge on `(track_id, frame)`. No new persistence
  layer.

#### B8. Export wiring

`exports.write_all` / `features_table` gain `channels=None`
(`{name: stack}` for marked channels), `measure_opts`, and `extra_tables=None`.
`features_table` computes B3/B4 for each marked channel and left-merges the
columns, then left-merges `extra_tables`. `on_save` builds `channels` from the
marked `ChannelLayer`s (reading live `.data`) and passes both through.

## Testing

`test_fluorescence.py` (Qt-free, runnable via `python` or `pytest`):
- synthetic 1-frame mask with two nuclei + known intensities;
- nucleus median equals the injected value;
- the ring of one nucleus excludes the neighbor nucleus pixels;
- C/N > 1 for a constructed cytoplasm-bright case, < 1 for nucleus-bright;
- background subtraction removes a constant offset;
- Haralick contrast is higher on a textured patch than a flat one (16 levels);
- `ring_preview_figure` returns a matplotlib Figure on synthetic data without
  raising (and picks a valid nucleus when `track_id` is None).

Existing `test_curation_ops.py` must still pass (no regressions in the save path).

## Out of scope now (YAGNI, upgrade path noted)

- Foci counting via LoG/DoG blob detection — the texture proxy is more stable for
  a temporal pipeline (user's own reasoning); leave a `# ponytail:` marker.
- Radial intensity distribution / granularity (profiling, not core).
- Manual background ROI (auto median is the default; add if the auto proxy fails).
- N/C ratio: **excluded** (redundant with C/N — same information inverted).
```
