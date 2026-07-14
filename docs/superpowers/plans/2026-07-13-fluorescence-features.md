# Fluorescence Channels + Expanded Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the curator import extra fluorescence channels (color/role tagged, optionally marked "measure") and export a richer per-`(track, frame)` feature set — morphology, motility, per-channel/per-compartment intensity + Haralick texture, plus interactive ERK-KTR (C/N) and 53BP1 nuclear-texture readouts.

**Architecture:** A new pure-numpy `curator/fluorescence.py` measurement engine (compartment masks, background, intensity, texture, ring preview) with no Qt, mirroring `analysis.py`. `channels.ChannelLayer` gains `color`/`measure`; a load-time dialog sets them. `analysis` gains a few free morphology/motility columns. `exports.features_table`/`write_all` merge per-channel features and interactive-readout tables. UI buttons in the statistics panel drive the interactive parts.

**Tech Stack:** Python, numpy, pandas, scipy.ndimage, scikit-image (regionprops, graycomatrix/graycoprops — already deps), matplotlib, magicgui/qtpy (UI only).

## Global Constraints

- Single nuclear mask always = `state.mask` (the tracking mask). Never segment from the KTR/green channel.
- Median is the primary intensity statistic (robust to hot pixels / foci).
- Background = per-frame median of channel pixels where `mask == 0`, subtracted (clipped at 0) before any intensity/ratio.
- Cytoplasm ring excludes ALL nuclei (`mask > 0`): `dilate(nuc, gap+dilation) & ~dilate(nuc, gap) & ~(mask>0)`. Defaults `dilation=2`, `gap=1`.
- Haralick GLCM uses **16 levels** (nuclei ~30 px diameter ≈ 700 px; 32² GLCM too sparse), distances=(1,), 4 angles averaged.
- No new pip dependency. No N/C ratio (redundant with C/N). Foci LoG counting is out of scope (texture proxy instead).
- Fluorescence features are computed only for channels the user marked "measure" (keeps save fast). Morphology/motility additions are free/always-on.
- New engine code is pure (no Qt/napari import) and headless-testable. `test_curation_ops.py` must keep passing (no save-path regressions).
- Branch: `feat/fluorescence-features`. Commit after every task.

---

## File Structure

- Create `curator/fluorescence.py` — compartment masks, background, intensity, Haralick, ring preview. Pure.
- Create `test_fluorescence.py` — headless engine tests (runnable via `python` or `pytest`).
- Modify `curator/channels.py` — `ChannelLayer` gains `color`, `measure`.
- Modify `curator/dialogs.py` — add `ChannelConfigDialog` (color/name/measure per channel).
- Modify `curator/app.py` — call the dialog, pass enriched layers.
- Modify `curator/analysis.py` — extend `nuclear_morphometry` (morphology cols) and `motility_metrics` (alpha).
- Modify `curator/exports.py` — `features_table`/`write_all` accept `channels`, `extra_tables`, `ring_opts` and merge fluorescence + interactive columns.
- Modify `curator/ui.py` — statistics-panel widgets: ring params + preview, ERK-KTR button, 53BP1 button; wire `on_save` to pass marked channels + accumulated `extra_feature_tables`.

---

## Task 1: `ChannelLayer` gains `color` and `measure`

**Files:**
- Modify: `curator/channels.py:54-63` (the `ChannelLayer` dataclass)
- Test: `test_fluorescence.py` (new file, first test)

**Interfaces:**
- Produces: `ChannelLayer(name:str, data:np.ndarray, colormap:str, color:str="green", measure:bool=False)`. `colormap` stays (napari display); `color` is the user-chosen tag (defaults to `colormap`); `measure` gates feature computation.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py
import numpy as np
from curator.channels import ChannelLayer

def test_channel_layer_has_color_and_measure_defaults():
    L = ChannelLayer(name="green", data=np.zeros((3, 4, 4)), colormap="green")
    assert L.color == "green"      # defaults to the colormap
    assert L.measure is False
    L2 = ChannelLayer(name="erk", data=np.zeros((3, 4, 4)), colormap="green",
                      color="green", measure=True)
    assert L2.measure is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py::test_channel_layer_has_color_and_measure_defaults -v`
Expected: FAIL (`ChannelLayer` has no `color`/`measure`).

- [ ] **Step 3: Implement**

In `curator/channels.py`, replace the dataclass body:

```python
@dataclass
class ChannelLayer:
    """One display-only channel ready to be added as a napari image layer."""
    name: str
    data: np.ndarray          # T x H x W (single channel)
    colormap: str
    color: str = ""           # user-chosen tag (green/red/...); defaults to colormap
    measure: bool = False     # compute per-cell fluorescence features for this channel

    def __post_init__(self):
        if not self.color:
            self.color = self.colormap

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim >= 3 else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py::test_channel_layer_has_color_and_measure_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/channels.py test_fluorescence.py
git commit -m "feat: ChannelLayer gains color/measure tags"
```

---

## Task 2: Compartment masks (nucleus / cytoplasm ring / whole cell)

**Files:**
- Create: `curator/fluorescence.py`
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces:
  - `compartment_masks(label_plane, track_id, dilation=2, gap=1) -> dict[str, np.ndarray]` returning boolean masks `{"nuc", "ring", "cell"}` the same shape as `label_plane`; `ring` excludes every nucleus (`label_plane > 0`); empty masks when the id is absent.
  - Module constants `RING_DILATION_DEFAULT = 2`, `RING_GAP_DEFAULT = 1`.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
from curator import fluorescence as fl

def _two_nuclei_plane():
    # 20x20 with nucleus 1 (rows 4-8, cols 4-8) and nucleus 2 (rows 4-8, cols 12-16)
    p = np.zeros((20, 20), dtype=np.int32)
    p[4:9, 4:9] = 1
    p[4:9, 12:17] = 2
    return p

def test_ring_excludes_neighbor_nucleus():
    p = _two_nuclei_plane()
    m = fl.compartment_masks(p, track_id=1, dilation=2, gap=1)
    assert m["nuc"].sum() == 25                 # 5x5 nucleus
    assert not (m["ring"] & (p == 2)).any()     # ring never covers nucleus 2
    assert not (m["ring"] & (p == 1)).any()     # ring never covers its own nucleus
    assert m["ring"].sum() > 0                   # a ring exists
    assert (m["cell"] == (m["nuc"] | m["ring"])).all()

def test_compartments_absent_id_is_empty():
    p = _two_nuclei_plane()
    m = fl.compartment_masks(p, track_id=99)
    assert m["nuc"].sum() == 0 and m["ring"].sum() == 0 and m["cell"].sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k compartment -v` and `-k ring_excludes`
Expected: FAIL (`fluorescence` module / `compartment_masks` missing).

- [ ] **Step 3: Implement**

Create `curator/fluorescence.py`:

```python
"""
Fluorescence measurement engine (pure numpy/scipy/skimage, no Qt/napari).

Given the single nuclear label mask (the tracking mask) and one or more
intensity channels (the raw 53BP1 movie and/or extra channels), it measures
per (frame, track_id) intensity and texture in three compartments -- nucleus,
a cytoplasm ring, and the whole cell -- and the cytoplasm/nucleus ratio used by
KTR reporters. Median is the primary statistic; a per-frame background (median
of non-cell pixels) is subtracted first. Headless-testable; the UI turns the
returned tables into plots/exports.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

RING_DILATION_DEFAULT = 2
RING_GAP_DEFAULT = 1


def compartment_masks(label_plane, track_id, dilation=RING_DILATION_DEFAULT,
                      gap=RING_GAP_DEFAULT):
    """Boolean nucleus/ring/cell masks for one label in a 2-D label plane.

    ring = dilate(nuc, gap+dilation) minus dilate(nuc, gap) minus ALL nuclei,
    i.e. an annulus starting ``gap`` px outside the nucleus, ``dilation`` px
    wide, that never overlaps any nucleus (avoids nuclear-edge bleed and
    neighbor-nucleus contamination). Returns empty masks if the id is absent.
    """
    nuc = (label_plane == int(track_id))
    out = {"nuc": nuc, "ring": np.zeros_like(nuc), "cell": nuc.copy()}
    if not nuc.any():
        return out
    any_nuc = (label_plane > 0)
    inner = ndi.binary_dilation(nuc, iterations=int(gap)) if gap > 0 else nuc
    outer = ndi.binary_dilation(nuc, iterations=int(gap) + int(dilation))
    ring = outer & ~inner & ~any_nuc
    out["ring"] = ring
    out["cell"] = nuc | ring
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fluorescence.py -k "compartment or ring_excludes" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/fluorescence.py test_fluorescence.py
git commit -m "feat: fluorescence compartment masks (nucleus/ring/cell)"
```

---

## Task 3: Per-frame background

**Files:**
- Modify: `curator/fluorescence.py`
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces: `background_per_frame(channel_plane, label_plane) -> float` = median of `channel_plane[label_plane == 0]` (0.0 if no background pixels).

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
def test_background_is_median_of_non_cell():
    p = _two_nuclei_plane()
    ch = np.full((20, 20), 10.0)
    ch[p == 1] = 100.0
    ch[p == 2] = 100.0
    assert fl.background_per_frame(ch, p) == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k background -v`
Expected: FAIL (`background_per_frame` missing).

- [ ] **Step 3: Implement** (append to `curator/fluorescence.py`)

```python
def background_per_frame(channel_plane, label_plane) -> float:
    """Per-frame background = median intensity where no cell is present."""
    bg_pixels = channel_plane[label_plane == 0]
    if bg_pixels.size == 0:
        return 0.0
    return float(np.median(bg_pixels))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py -k background -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/fluorescence.py test_fluorescence.py
git commit -m "feat: per-frame fluorescence background"
```

---

## Task 4: Intensity measures per (frame, track, compartment) + C/N

**Files:**
- Modify: `curator/fluorescence.py`
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces: `measure_intensity(channel_stack, mask, channel_name="ch", dilation=2, gap=1, subtract_background=True) -> pandas.DataFrame` with columns `frame, track_id`, and for each compartment `nuc`/`ring`/`cell`: `<name>_<comp>_mean/median/sum/min/max/std/p90`, plus `<name>_cn_ratio` = `ring_median / nuc_median`. `channel_stack` is T×H×W; `mask` is T×H×W labels. Background subtracted (clipped ≥ 0) per frame before stats.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
import pandas as pd

def test_intensity_median_and_cn_ratio():
    mask = _two_nuclei_plane()[None, ...]           # 1 x 20 x 20
    ch = np.full((1, 20, 20), 5.0)                  # background 5
    ch[0][mask[0] == 1] = 25.0                      # nucleus 1 bright
    # make the cytoplasm ring of nucleus 1 brighter than its nucleus
    m = fl.compartment_masks(mask[0], 1)
    ch[0][m["ring"]] = 45.0
    df = fl.measure_intensity(ch, mask, channel_name="green")
    row = df[df["track_id"] == 1].iloc[0]
    assert abs(row["green_nuc_median"] - 20.0) < 1e-6      # 25 - bg 5
    assert row["green_cn_ratio"] > 1.0                     # cyto brighter -> active
    assert row["green_nuc_sum"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k intensity -v`
Expected: FAIL (`measure_intensity` missing).

- [ ] **Step 3: Implement** (append to `curator/fluorescence.py`)

```python
import pandas as pd

_COMPARTMENTS = ("nuc", "ring", "cell")
_STATS = ("mean", "median", "sum", "min", "max", "std", "p90")


def _compartment_stats(values):
    """Return the stat dict for one compartment's (bg-subtracted) pixel values."""
    if values.size == 0:
        return {s: np.nan for s in _STATS}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sum": float(np.sum(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
    }


def measure_intensity(channel_stack, mask, channel_name="ch",
                      dilation=RING_DILATION_DEFAULT, gap=RING_GAP_DEFAULT,
                      subtract_background=True):
    """Per (frame, track_id) intensity in nucleus/ring/cell + C/N ratio.

    Columns: frame, track_id, <name>_<comp>_<stat> for comp in nuc/ring/cell and
    stat in mean/median/sum/min/max/std/p90, plus <name>_cn_ratio. Background
    (per-frame median of non-cell pixels) is subtracted and clipped at 0 first.
    """
    cols = ["frame", "track_id"]
    for c in _COMPARTMENTS:
        for s in _STATS:
            cols.append(f"{channel_name}_{c}_{s}")
    cols.append(f"{channel_name}_cn_ratio")
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=cols)

    rows = []
    n = mask.shape[0]
    for f in range(n):
        lab = mask[f]
        ch = channel_stack[f].astype(float)
        if subtract_background:
            bg = background_per_frame(ch, lab)
            ch = np.clip(ch - bg, 0, None)
        ids = np.unique(lab)
        for tid in ids[ids > 0]:
            m = compartment_masks(lab, int(tid), dilation=dilation, gap=gap)
            rec = {"frame": int(f), "track_id": int(tid)}
            per_comp = {}
            for c in _COMPARTMENTS:
                st = _compartment_stats(ch[m[c]])
                per_comp[c] = st
                for s in _STATS:
                    rec[f"{channel_name}_{c}_{s}"] = st[s]
            nuc_med = per_comp["nuc"]["median"]
            ring_med = per_comp["ring"]["median"]
            rec[f"{channel_name}_cn_ratio"] = (
                ring_med / nuc_med if nuc_med and nuc_med > 0 else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows, columns=cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py -k intensity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/fluorescence.py test_fluorescence.py
git commit -m "feat: per-compartment intensity measures + C/N ratio"
```

---

## Task 5: Haralick texture (16 levels)

**Files:**
- Modify: `curator/fluorescence.py`
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces: `haralick_features(channel_stack, mask, channel_name="ch", levels=16, vmin=None, vmax=None) -> pandas.DataFrame` with `frame, track_id, <name>_hara_contrast/correlation/energy/homogeneity/entropy`. GLCM on the nuclear ROI (bbox crop), quantized to `levels` using the channel's global [vmin, vmax] (1st/99th percentile if None), distances=(1,), 4 angles averaged. Non-nucleus pixels in the crop are excluded from the GLCM by setting them to a sentinel and masking.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
def test_haralick_contrast_higher_on_textured_than_flat():
    mask = np.zeros((1, 12, 24), dtype=np.int32)
    mask[0, 2:10, 2:10] = 1          # flat nucleus
    mask[0, 2:10, 14:22] = 2         # textured nucleus
    ch = np.zeros((1, 12, 24), dtype=float)
    ch[0][mask[0] == 1] = 100.0                       # uniform
    checker = (np.indices((8, 8)).sum(axis=0) % 2) * 200.0
    ch[0, 2:10, 14:22] = checker                      # high-frequency texture
    df = fl.haralick_features(ch, mask, channel_name="green", levels=16)
    c_flat = df[df["track_id"] == 1]["green_hara_contrast"].iloc[0]
    c_tex = df[df["track_id"] == 2]["green_hara_contrast"].iloc[0]
    assert c_tex > c_flat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k haralick -v`
Expected: FAIL (`haralick_features` missing).

- [ ] **Step 3: Implement** (append to `curator/fluorescence.py`)

```python
from skimage.feature import graycomatrix, graycoprops

_HARA_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
_HARA_COLS = ("contrast", "correlation", "energy", "homogeneity", "entropy")


def _quantize(values, vmin, vmax, levels):
    """Scale float values into 1..levels-1 integers over [vmin, vmax].

    Bin 0 is reserved as the background/outside-nucleus sentinel, so a real but
    dark in-nucleus pixel is never confused with background (a low-intensity
    nucleus pixel must survive the background-bin removal in _glcm_row).
    """
    if vmax <= vmin:
        return np.ones_like(values, dtype=np.uint8)
    q = (values - vmin) / (vmax - vmin)
    q = np.clip(q, 0.0, 1.0) * (levels - 2)     # 0..levels-2
    return (np.rint(q).astype(np.uint8) + 1)    # 1..levels-1


def _glcm_row(crop_q, nuc_crop, levels):
    """One Haralick dict from a quantized nuclear crop, or NaNs if too small.

    ``crop_q`` holds in-nucleus values in 1..levels-1 (from _quantize); pixels
    outside the nucleus are set to the background sentinel bin 0 here, then bin 0
    is dropped so texture reflects only nucleus-to-nucleus co-occurrences.
    """
    nan = {c: np.nan for c in _HARA_COLS}
    if nuc_crop.sum() < 8:
        return nan
    img = crop_q.copy()
    img[~nuc_crop] = 0
    glcm = graycomatrix(img, distances=[1], angles=list(_HARA_ANGLES),
                        levels=levels, symmetric=True, normed=True)
    # Drop the background bin (index 0) contributions along both axes, then
    # renormalize per angle so texture reflects only in-nucleus pairs.
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1.0
    glcm = glcm / sums
    out = {}
    for prop in ("contrast", "correlation", "energy", "homogeneity"):
        out[prop] = float(np.mean(graycoprops(glcm, prop)))
    # Entropy averaged over the angle planes (axis 3); axis 2 is distances (=1).
    n_angles = glcm.shape[3]
    p = glcm[glcm > 0]
    out["entropy"] = float(-np.sum(p * np.log2(p)) / n_angles) if p.size else np.nan
    return out


def haralick_features(channel_stack, mask, channel_name="ch", levels=16,
                      vmin=None, vmax=None):
    """Per (frame, track_id) Haralick texture on the nuclear ROI (16 levels)."""
    cols = ["frame", "track_id"] + [f"{channel_name}_hara_{c}" for c in _HARA_COLS]
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=cols)
    ch_all = channel_stack.astype(float)
    if vmin is None or vmax is None:
        inside = ch_all[mask > 0]
        if inside.size:
            vmin = float(np.percentile(inside, 1)) if vmin is None else vmin
            vmax = float(np.percentile(inside, 99)) if vmax is None else vmax
        else:
            vmin, vmax = 0.0, 1.0
    rows = []
    for f in range(mask.shape[0]):
        lab = mask[f]
        # _quantize returns 1..levels-1 (bin 0 is the background sentinel).
        q = _quantize(ch_all[f], vmin, vmax, levels)
        ids = np.unique(lab)
        for tid in ids[ids > 0]:
            nuc = (lab == int(tid))
            ys, xs = np.nonzero(nuc)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            hd = _glcm_row(q[y0:y1, x0:x1], nuc[y0:y1, x0:x1], levels)
            rec = {"frame": int(f), "track_id": int(tid)}
            for c in _HARA_COLS:
                rec[f"{channel_name}_hara_{c}"] = hd[c]
            rows.append(rec)
    return pd.DataFrame(rows, columns=cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py -k haralick -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/fluorescence.py test_fluorescence.py
git commit -m "feat: Haralick nuclear texture (16 levels)"
```

---

## Task 6: Ring preview figure

**Files:**
- Modify: `curator/fluorescence.py`
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces: `ring_preview_figure(mask, channel_stack, track_id=None, dilations=(1,2,3,4), gap=1, frame=None, rng=None) -> matplotlib.figure.Figure` — picks a real nucleus (random if `track_id` None), shows small multiples of the ring growing over the channel crop, with neighbor nuclei shaded.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
def test_ring_preview_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    mask = _two_nuclei_plane()[None, ...]
    ch = np.random.RandomState(0).rand(1, 20, 20) * 50
    fig = fl.ring_preview_figure(mask, ch, track_id=None,
                                 dilations=(1, 2, 3), gap=1, rng=np.random.RandomState(1))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k preview -v`
Expected: FAIL (`ring_preview_figure` missing).

- [ ] **Step 3: Implement** (append to `curator/fluorescence.py`)

```python
def _pick_nucleus(mask, track_id, frame, rng):
    """Return (frame, track_id) of a real nucleus; random if track_id is None."""
    rng = rng or np.random.RandomState()
    if track_id is not None:
        if frame is None:
            for f in range(mask.shape[0]):
                if (mask[f] == int(track_id)).any():
                    return f, int(track_id)
        return (frame or 0), int(track_id)
    frames = list(range(mask.shape[0]))
    rng.shuffle(frames)
    for f in frames:
        ids = np.unique(mask[f])
        ids = ids[ids > 0]
        if ids.size:
            return f, int(rng.choice(ids))
    return 0, 0


def ring_preview_figure(mask, channel_stack, track_id=None,
                        dilations=(1, 2, 3, 4), gap=RING_GAP_DEFAULT,
                        frame=None, rng=None):
    """Small-multiples preview of the cytoplasm ring growing on one nucleus."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    f, tid = _pick_nucleus(mask, track_id, frame, rng)
    lab = mask[f]
    ch = channel_stack[f].astype(float) if channel_stack is not None else (lab > 0).astype(float)
    ys, xs = np.nonzero(lab == tid)
    pad = int(max(dilations)) + int(gap) + 3
    if ys.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no nucleus", ha="center")
        return fig
    y0, y1 = max(0, ys.min() - pad), min(lab.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(lab.shape[1], xs.max() + pad)
    lab_c, ch_c = lab[y0:y1, x0:x1], ch[y0:y1, x0:x1]

    n = len(dilations)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    overlay = ListedColormap(["#00000000", "#ff3b3b"])         # ring in red
    others = ListedColormap(["#00000000", "#3b6bff55"])        # neighbor nuclei shaded
    for ax, dil in zip(axes, dilations):
        m = compartment_masks(lab_c, tid, dilation=int(dil), gap=int(gap))
        ax.imshow(ch_c, cmap="gray")
        ax.imshow((lab_c > 0) & (lab_c != tid), cmap=others, vmin=0, vmax=1)
        ax.contour((lab_c == tid).astype(float), levels=[0.5], colors="#33ff88", linewidths=1.2)
        ax.imshow(m["ring"], cmap=overlay, vmin=0, vmax=1, alpha=0.7)
        ax.set_title(f"dilation={dil}px, gap={gap}px", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Ring preview — nucleus {tid} @ frame {f}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py -k preview -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/fluorescence.py test_fluorescence.py
git commit -m "feat: cytoplasm ring preview figure"
```

---

## Task 7: Morphology + motility additions (free, always-on)

**Files:**
- Modify: `curator/analysis.py` — `nuclear_morphometry` (add regionprops columns) and `motility_metrics` (add `alpha`).
- Test: `test_fluorescence.py`

**Interfaces:**
- Produces:
  - `analysis.nuclear_morphometry(mask)` gains columns `eccentricity, solidity, extent, orientation, axis_major_length, axis_minor_length`; `MORPHOMETRY_COLS` extended so they flow into `features_table` and the summary means.
  - `analysis.motility_metrics(...)` output dict gains `anomalous_exponent` = slope of `log(MSD)` vs `log(lag)` (NaN if not estimable).

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
from curator import analysis

def test_nuclear_morphometry_has_new_shape_columns():
    mask = np.zeros((1, 20, 20), dtype=np.int32)
    mask[0, 4:16, 6:12] = 1                      # a rectangle (elongated)
    nm = analysis.nuclear_morphometry(mask)
    for col in ["eccentricity", "solidity", "extent", "orientation",
                "axis_major_length", "axis_minor_length"]:
        assert col in nm.columns
    assert nm["eccentricity"].iloc[0] > 0.5      # clearly elongated
    assert 0 < nm["extent"].iloc[0] <= 1.0

def test_motility_has_anomalous_exponent():
    frames = np.arange(20, dtype=float)
    xs = frames * 1.0            # ballistic-ish straight motion
    ys = np.zeros_like(frames)
    out = analysis.motility_metrics(frames, xs, ys, frame_interval=1.0)
    assert "anomalous_exponent" in out
    assert np.isfinite(out["anomalous_exponent"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k "morphometry or anomalous" -v`
Expected: FAIL (new columns/keys missing).

- [ ] **Step 3: Implement**

In `curator/analysis.py`, extend the morphometry column list (near line 698):

```python
NMA_COLS = ["aspect", "area_box", "radius_ratio", "roundness", "nii"]
_SHAPE_COLS = ["eccentricity", "solidity", "extent", "orientation",
               "axis_major_length", "axis_minor_length"]
MORPHOMETRY_COLS = ["area_px", "perimeter", "circularity"] + NMA_COLS + _SHAPE_COLS
```

In `nuclear_morphometry`, inside the `for r in regionprops(plane):` loop, after computing `nii`, add the extra props and include them in the appended dict:

```python
            rows.append({
                "frame": int(f), "track_id": int(r.label),
                "area_px": area, "perimeter": perim, "circularity": float(circ),
                "aspect": float(aspect), "area_box": float(area_box),
                "radius_ratio": float(rr), "roundness": float(roundness),
                "nii": float(nii),
                "eccentricity": float(r.eccentricity),
                "solidity": float(r.solidity),
                "extent": float(r.extent),
                "orientation": float(r.orientation),
                "axis_major_length": float(r.axis_major_length),
                "axis_minor_length": float(r.axis_minor_length)})
```

In `motility_metrics`, initialize `anomalous_exponent` in the `out` dict:

```python
    out = {"diffusion_coeff": np.nan, "persistence_time": np.nan,
           "mean_turning_angle": np.nan, "confinement_ratio": np.nan,
           "anomalous_exponent": np.nan}
```

and in the MSD block (where `lags`/`msd` are built, after the linear `slope` fit) add the log–log fit:

```python
        if np.ptp(lags) > 0:
            slope = np.polyfit(lags, msd, 1)[0]
            out["diffusion_coeff"] = float(slope / 4.0)
            pos = (lags > 0) & (msd > 0)
            if pos.sum() >= 2:
                out["anomalous_exponent"] = float(
                    np.polyfit(np.log(lags[pos]), np.log(msd[pos]), 1)[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fluorescence.py -k "morphometry or anomalous" -v`
Expected: PASS

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `python -m pytest test_fluorescence.py test_curation_ops.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add curator/analysis.py test_fluorescence.py
git commit -m "feat: add shape morphometry + anomalous exponent columns"
```

---

## Task 8: Export wiring — merge channel features + interactive tables

**Files:**
- Modify: `curator/exports.py` — `features_table` and `write_all`.
- Test: `test_fluorescence.py`

**Interfaces:**
- Consumes: `fluorescence.measure_intensity`, `fluorescence.haralick_features`.
- Produces:
  - `exports.features_table(df, mask, pixel_size=1.0, frame_interval=1.0, channels=None, ring_opts=None, texture=True, extra_tables=None)` — `channels` is `{name: TxHxW array}` for marked channels; merges `measure_intensity` (+`haralick_features` when `texture`) per channel and left-merges each DataFrame in `extra_tables` on `[track_id, frame]`.
  - `exports.write_all(..., channels=None, ring_opts=None, extra_tables=None)` — threads the same through and also writes `<base>_fluorescence.csv` (the channel feature block alone) when `channels` given.

- [ ] **Step 1: Write the failing test**

```python
# test_fluorescence.py  (append)
from curator import exports
from curator.config import (COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME,
                            COL_PARENT, COL_CLABEL, COL_TREATMENT, TREAT_CONTROL)

def _mini_df_and_mask():
    mask = np.zeros((2, 20, 20), dtype=np.int32)
    mask[0, 4:9, 4:9] = 1
    mask[1, 5:10, 5:10] = 1
    rows = []
    for f in (0, 1):
        ys, xs = np.nonzero(mask[f] == 1)
        rows.append({COL_TRACK: 1, COL_FRAME: f, COL_X: float(xs.mean()),
                     COL_Y: float(ys.mean()), COL_OUTCOME: "", COL_PARENT: -1,
                     COL_CLABEL: 1, COL_TREATMENT: TREAT_CONTROL})
    return pd.DataFrame(rows), mask

def test_features_table_merges_channel_columns():
    df, mask = _mini_df_and_mask()
    ch = np.full((2, 20, 20), 5.0)
    ch[mask == 1] = 30.0
    out = exports.features_table(df, mask, channels={"green": ch}, texture=False)
    assert "green_nuc_median" in out.columns
    assert "green_cn_ratio" in out.columns
    # per-frame row for (track 1, frame 0) carries the nucleus median (30-bg5=25)
    r0 = out[(out[COL_TRACK] == 1) & (out[COL_FRAME] == 0)].iloc[0]
    assert abs(r0["green_nuc_median"] - 25.0) < 1e-6

def test_features_table_merges_extra_tables():
    df, mask = _mini_df_and_mask()
    extra = pd.DataFrame({COL_TRACK: [1, 1], COL_FRAME: [0, 1],
                          "erk_cn": [1.5, 2.0]})
    out = exports.features_table(df, mask, extra_tables=[extra])
    assert "erk_cn" in out.columns
    assert out.loc[out[COL_FRAME] == 1, "erk_cn"].iloc[0] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fluorescence.py -k "features_table_merges" -v`
Expected: FAIL (`features_table` has no `channels`/`extra_tables` kwargs).

- [ ] **Step 3: Implement**

In `curator/exports.py`, import the engine at top:

```python
from . import analysis, lineage, fluorescence
```

Change `features_table` signature and append the merges before `return`:

```python
def features_table(df, mask, pixel_size=1.0, frame_interval=1.0,
                   channels=None, ring_opts=None, texture=True,
                   extra_tables=None):
```

At the end of `features_table`, replace `return out.reset_index(drop=True)` with:

```python
    # Per-channel fluorescence features (marked channels only), merged by cell/frame.
    ropts = ring_opts or {}
    dil = int(ropts.get("dilation", fluorescence.RING_DILATION_DEFAULT))
    gap = int(ropts.get("gap", fluorescence.RING_GAP_DEFAULT))
    for name, stack in (channels or {}).items():
        inten = fluorescence.measure_intensity(stack, mask, channel_name=name,
                                               dilation=dil, gap=gap)
        if not inten.empty:
            inten = inten.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
            out = out.merge(inten, on=[COL_TRACK, COL_FRAME], how="left")
        if texture:
            hara = fluorescence.haralick_features(stack, mask, channel_name=name)
            if not hara.empty:
                hara = hara.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
                out = out.merge(hara, on=[COL_TRACK, COL_FRAME], how="left")

    # Interactive-readout tables (ERK-KTR, 53BP1) accumulated in the session.
    for extra in (extra_tables or []):
        if extra is None or extra.empty:
            continue
        e = extra.rename(columns={"track_id": COL_TRACK, "frame": COL_FRAME})
        merge_cols = [c for c in (COL_TRACK, COL_FRAME) if c in e.columns]
        if len(merge_cols) == 2:
            dup = [c for c in e.columns if c in out.columns and c not in merge_cols]
            e = e.drop(columns=dup)
            out = out.merge(e, on=merge_cols, how="left")
    return out.reset_index(drop=True)
```

Update `windows_table`'s internal `features_table` call to stay morphology-only (it passes no channels, so no change needed — verify the call at line ~151 is `features_table(df, mask, pixel_size, frame_interval)`).

Change `write_all` signature and body to thread the new args and write the fluorescence block:

```python
def write_all(work_dir, df, mask, reviewed_roots, validated_cells, base_name,
              window=7, pixel_size=1.0, frame_interval=1.0,
              channels=None, ring_opts=None, extra_tables=None):
```

Replace the `features = features_table(...)` line with:

```python
    features = features_table(df, mask, pixel_size, frame_interval,
                              channels=channels, ring_opts=ring_opts,
                              extra_tables=extra_tables)
```

And after the existing `written[...]` assignments, add the standalone fluorescence
block — **reuse the already-computed `features`** (do not recompute; Haralick is
expensive), selecting just the channel columns:

```python
    if channels and features is not None and not features.empty:
        chan_cols = [c for c in features.columns
                     if any(c.startswith(f"{n}_") for n in channels)]
        if chan_cols:
            fl_only = features[[COL_TRACK, COL_FRAME] + chan_cols]
            written["fluorescence"] = _atomic_csv(fl_only, p("fluorescence"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fluorescence.py -k "features_table_merges" -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest test_fluorescence.py test_curation_ops.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add curator/exports.py test_fluorescence.py
git commit -m "feat: merge fluorescence + interactive features into export tables"
```

---

## Task 9: Channel import dialog (color / name / measure)

**Files:**
- Modify: `curator/dialogs.py` — add `ChannelConfigDialog`.
- Modify: `curator/app.py` — invoke it after channels are loaded, apply results.
- Test: manual (Qt dialog); a headless helper is unit-tested.

**Interfaces:**
- Consumes: `channels.ChannelLayer` (list from `load_channel_sources`).
- Produces: `dialogs.apply_channel_config(layers, config)` — pure function that, given a list of `ChannelLayer` and a list of dicts `{"name", "color", "measure"}`, returns the updated layers (colormap set to the chosen color). `ChannelConfigDialog(layers).get_config()` returns that list of dicts.

- [ ] **Step 1: Write the failing test** (append to `test_fluorescence.py`)

```python
from curator import dialogs

def test_apply_channel_config_sets_color_and_measure():
    from curator.channels import ChannelLayer
    layers = [ChannelLayer("green", np.zeros((2, 4, 4)), "green"),
              ChannelLayer("red", np.zeros((2, 4, 4)), "magenta")]
    cfg = [{"name": "ERK-KTR", "color": "green", "measure": True},
           {"name": "red", "color": "red", "measure": False}]
    out = dialogs.apply_channel_config(layers, cfg)
    assert out[0].name == "ERK-KTR" and out[0].colormap == "green" and out[0].measure
    assert out[1].colormap == "red" and out[1].measure is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_fluorescence.py -k channel_config -v`
Expected: FAIL (`apply_channel_config` missing).

- [ ] **Step 3: Implement**

In `curator/dialogs.py`, add the pure helper and the dialog (follow the existing `ColumnMappingDialog` pattern in this file for the Qt layout):

```python
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
```

Ensure the imports at the top of `dialogs.py` include the widgets used
(`QGridLayout`, `QLineEdit`, `QComboBox`, `QCheckBox`, `QDialogButtonBox`,
`QLabel`, `QVBoxLayout`, `QDialog`) — add any missing to the existing qtpy import.

In `curator/app.py`, after `channel_layers = channels_mod.load_channel_sources(...)`
and before `build_viewer(...)`, add:

```python
    if channel_layers:
        from .dialogs import ChannelConfigDialog, apply_channel_config
        cdlg = ChannelConfigDialog(channel_layers)
        if cdlg.exec_() == QDialog.Accepted:
            channel_layers = apply_channel_config(channel_layers, cdlg.get_config())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_fluorescence.py -k channel_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add curator/dialogs.py curator/app.py test_fluorescence.py
git commit -m "feat: channel config dialog (color/name/measure)"
```

---

## Task 10: Statistics-panel wiring — ring params/preview, ERK-KTR, 53BP1, save

**Files:**
- Modify: `curator/ui.py` — add widgets + handlers in `build_viewer`; extend `on_save`.
- Test: manual (Qt/napari). Engine already covered by `test_fluorescence.py`.

**Interfaces:**
- Consumes: `fluorescence.measure_intensity`, `fluorescence.haralick_features`, `fluorescence.ring_preview_figure`, `channels.ChannelLayer` (from closure `channel_layers`).
- Produces: closure dict `extra_feature_tables` (list of per-`(track,frame)` DataFrames) passed to `exports.write_all`; three new buttons.

- [ ] **Step 1: Add the engine import and state**

At the top of `curator/ui.py` (with the other `from . import ...`), add `fluorescence`:

```python
from . import (analysis, stats, exports, lineage, ...)  # add: fluorescence
```

Inside `build_viewer`, near the other closure state (e.g. after `action_hooks = []`), add:

```python
    extra_feature_tables = []       # accumulated ERK-KTR / 53BP1 per-(track,frame) tables

    def _image_layer_choices():
        return [ly.name for ly in viewer.layers
                if ly.__class__.__name__ == "Image"]

    def _channel_stack_by_name(name):
        for ly in viewer.layers:
            if ly.name == name and ly.__class__.__name__ == "Image":
                return np.asarray(ly.data)
        return None
```

- [ ] **Step 2: Add ring param + preview widgets**

In the statistics-widgets section (after `exclude_interp_cb`, ~line 1101), add:

```python
    ring_dilation = SpinBox(label="Cytoplasm ring width (px):", value=2, min=1, max=20)
    ring_gap = SpinBox(label="Ring gap from nucleus (px):", value=1, min=0, max=10)
    ring_channel = ComboBox(label="Preview/measure channel:", choices=_image_layer_choices())
    btn_ring_preview = PushButton(text="Preview ring on a random nucleus")

    def _ring_preview():
        stack = _channel_stack_by_name(ring_channel.value)
        try:
            fig = fluorescence.ring_preview_figure(
                state.mask, stack, track_id=None,
                dilations=tuple(range(1, int(ring_dilation.value) + 1)) or (1,),
                gap=int(ring_gap.value))
            fig.show()
        except Exception as exc:
            show_error(f"Ring preview failed: {exc}")
    btn_ring_preview.clicked.connect(_ring_preview)
```

- [ ] **Step 3: Add ERK-KTR and 53BP1 buttons**

```python
    btn_erk = PushButton(text="Compute ERK-KTR C/N (channel -> features)")
    btn_53bp1 = PushButton(text="Measure 53BP1 nuclear texture (channel -> features)")

    def _erk_ktr():
        stack = _channel_stack_by_name(ring_channel.value)
        if stack is None:
            return show_error("Pick a valid channel layer.")
        df = fluorescence.measure_intensity(
            stack, state.mask, channel_name=ring_channel.value,
            dilation=int(ring_dilation.value), gap=int(ring_gap.value))
        if df.empty:
            return show_error("No cells measured.")
        col = f"{ring_channel.value}_cn_ratio"
        keep = df[["track_id", "frame", col]]
        extra_feature_tables.append(keep)
        import os
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(work_dir, "exports"); os.makedirs(out_dir, exist_ok=True)
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
        stack = _channel_stack_by_name(ring_channel.value)
        if stack is None:
            return show_error("Pick a valid channel layer.")
        inten = fluorescence.measure_intensity(
            stack, state.mask, channel_name=ring_channel.value,
            dilation=int(ring_dilation.value), gap=int(ring_gap.value))
        hara = fluorescence.haralick_features(
            stack, state.mask, channel_name=ring_channel.value)
        cols = ["track_id", "frame",
                f"{ring_channel.value}_nuc_median", f"{ring_channel.value}_nuc_std"]
        merged = inten[cols].merge(hara, on=["track_id", "frame"], how="left")
        merged[f"{ring_channel.value}_nuc_cv"] = (
            inten[f"{ring_channel.value}_nuc_std"] /
            inten[f"{ring_channel.value}_nuc_median"].replace(0, np.nan))
        extra_feature_tables.append(merged)
        import os
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(work_dir, "exports"); os.makedirs(out_dir, exist_ok=True)
        merged.to_csv(os.path.join(out_dir, f"{base}_53bp1.csv"), index=False)
        show_info(f"53BP1 texture computed for {merged['track_id'].nunique()} cells; "
                  f"added to next export.")
    btn_53bp1.clicked.connect(_dsb_53bp1)
```

Reuse `stats.custom_plot(df, x_col, y_col, group_by=..., kind="line")` (returns a
Figure) + `stats.show(fig)` — the same pair `_traj` uses; do not add a new plot
function.

- [ ] **Step 4: Register the widgets in the stats container**

`build_viewer` assembles the stats panel from a `stats_sections` list of
`_section(title, [widgets], collapsed=)` entries (`curator/ui.py:1575-1589`). Add
a new section to that list:

```python
        _section("FLUORESCENCE",
                 [ring_channel, ring_dilation, ring_gap, btn_ring_preview,
                  btn_erk, btn_53bp1], collapsed=True),
```

- [ ] **Step 5: Pass marked channels + extra tables into `write_all`**

In `on_save`, replace the `res = exports.write_all(...)` call with:

```python
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
```

- [ ] **Step 6: Smoke-test the app launches and saves**

Run: `python run_curator.py --help`
Expected: argparse help prints with no import error (verifies `ui.py`/`exports.py`/`fluorescence.py` import cleanly).

Then a manual load + "Save all" on a real dataset with one marked channel:
Expected: `exports/<base>_features.csv` contains `<channel>_nuc_median` etc.; `exports/<base>_fluorescence.csv` written; ERK-KTR button writes `<base>_erk_ktr.csv` and the C/N column appears in features after a save.

- [ ] **Step 7: Commit**

```bash
git add curator/ui.py
git commit -m "feat: stats-panel ring preview + ERK-KTR/53BP1 buttons + export wiring"
```

---

## Task 11: Full regression pass + README note

**Files:**
- Modify: `README.md` (features/export section — document the new channel import + feature columns).
- Test: full suite.

- [ ] **Step 1: Run every test**

Run: `python -m pytest test_fluorescence.py test_curation_ops.py -v`
Expected: PASS (all).

Run: `python test_fluorescence.py` and `python test_curation_ops.py`
Expected: both print "all checks passed".

- [ ] **Step 2: Document**

Add a short subsection to `README.md` (near the export/features documentation)
listing: extra fluorescence channels (TIFF stacks, color/measure tagged at
import), the new per-`(track,frame)` columns (shape morphology, `<channel>_<compartment>_<stat>`,
`<channel>_hara_*`, `<channel>_cn_ratio`), the ERK-KTR C/N and 53BP1 texture
buttons, and the ring `dilation`/`gap` parameters with preview. Keep it factual,
match the README's existing tone.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document fluorescence channels + expanded features"
```

---

## Self-Review Notes

- **Spec coverage:** channel color/measure import (T1, T9); compartment ring excluding neighbors (T2); background (T3); intensity+median+C/N (T4); Haralick 16 levels (T5); ring preview modifiable (T6); morphology ecc/solidity/extent/orientation/axes + motility alpha (T7); export merge + fluorescence CSV (T8); ERK-KTR + 53BP1 persisted (T10); marked-channels-only auto features (T10 `on_save`). All spec sections map to a task.
- **Deferred (spec out-of-scope):** foci LoG, radial/granularity, manual bg ROI, N/C — not planned, matching the spec.
- **Type consistency:** engine tables use `track_id`/`frame`; export renames to `COL_TRACK`/`COL_FRAME` before merging (T8). `ring_opts` keys `dilation`/`gap` consistent across T8/T10. `extra_feature_tables` (UI) → `extra_tables` (exports arg) consistently mapped in T10 Step 5.
- **UI integration verified:** plots reuse `stats.custom_plot` + `stats.show` (the pair `_traj` uses); widgets register via a new `_section(...)` in the `stats_sections` list (`ui.py:1575`). No invented functions.
```
