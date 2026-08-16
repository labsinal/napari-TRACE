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
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops

RING_DILATION_DEFAULT = 2
RING_GAP_DEFAULT = 1
NEIGHBOR_GAP_DEFAULT = 0
BACKGROUND_DILATE_DEFAULT = 30


def compartment_masks(label_plane, track_id, dilation=RING_DILATION_DEFAULT,
                      gap=RING_GAP_DEFAULT, neighbor_gap=NEIGHBOR_GAP_DEFAULT):
    """Boolean nucleus/ring/cell masks for one label in a 2-D label plane.

    ring = dilate(nuc, gap+dilation) minus dilate(nuc, gap) minus ALL nuclei,
    i.e. an annulus starting ``gap`` px outside the nucleus, ``dilation`` px
    wide, that never overlaps any nucleus (avoids nuclear-edge bleed and
    neighbor-nucleus contamination). Returns empty masks if the id is absent.

    ``neighbor_gap`` > 0 additionally keeps the ring ``neighbor_gap`` px clear of
    every OTHER nucleus. Excluding the neighbour nuclei themselves (the default)
    stops nuclear signal leaking in, but in a colony the ring can still sit on a
    neighbour's cytoplasm; widening the exclusion trades ring area for purity.
    """
    nuc = (label_plane == int(track_id))
    out = {"nuc": nuc, "ring": np.zeros_like(nuc), "cell": nuc.copy()}
    if not nuc.any():
        return out
    any_nuc = (label_plane > 0)
    if neighbor_gap > 0:
        others = any_nuc & ~nuc
        if others.any():
            others = ndi.binary_dilation(others, iterations=int(neighbor_gap))
        any_nuc = others | nuc
    inner = ndi.binary_dilation(nuc, iterations=int(gap)) if gap > 0 else nuc
    outer = ndi.binary_dilation(nuc, iterations=int(gap) + int(dilation))
    ring = outer & ~inner & ~any_nuc
    out["ring"] = ring
    out["cell"] = nuc | ring
    return out


def background_per_frame(channel_plane, label_plane) -> float:
    """Per-frame background = median intensity where no cell is present."""
    bg_pixels = channel_plane[label_plane == 0]
    if bg_pixels.size == 0:
        return 0.0
    return float(np.median(bg_pixels))


def background_far_from_cells(channel_plane, label_plane,
                              dilate_px=BACKGROUND_DILATE_DEFAULT):
    """Per-frame background from pixels far away from EVERY nucleus.

    :func:`background_per_frame` calls every non-nucleus pixel "background",
    which in a colony includes the cytoplasm surrounding each nucleus. As the
    field fills up, that estimate drifts upward and the subtraction becomes
    progressively excessive -- an artefact that grows with confluence and can
    imitate a real trend over a long timelapse.

    Here every nucleus is dilated by ``dilate_px`` first, and the background is
    the median of what is left. Returns ``(value, free_fraction)`` so the caller
    can warn when the field is too crowded for the estimate to mean anything;
    falls back to :func:`background_per_frame` when nothing is left.
    """
    cells = (label_plane > 0)
    if cells.any() and dilate_px > 0:
        cells = ndi.binary_dilation(cells, iterations=int(dilate_px))
    free = ~cells
    frac = float(free.mean())
    vals = channel_plane[free]
    if vals.size == 0:
        return background_per_frame(channel_plane, label_plane), frac
    return float(np.median(vals)), frac


def background_image_min(channel_plane) -> float:
    """Per-frame background = the single darkest pixel in the whole frame.

    An alternative to :func:`background_per_frame`: instead of the median of
    non-cell pixels (robust to one bad pixel, but biased upward in a confluent
    field where "non-cell" still contains cytoplasm), this uses the frame's
    global minimum intensity as the background reference -- unaffected by
    confluence, but sensitive to a single hot/dead/noise pixel. Neither is
    strictly better; pick whichever the dataset's failure mode calls for.
    """
    return float(np.min(channel_plane))


def compartment_masks_ellipse(label_plane, track_id, dilation=RING_DILATION_DEFAULT,
                              gap=RING_GAP_DEFAULT,
                              neighbor_gap=NEIGHBOR_GAP_DEFAULT):
    """Boolean nucleus/ring/cell masks using an ellipse fit for the RING geometry.

    An alternative to :func:`compartment_masks`: the nucleus compartment is
    still the real segmentation mask (measured as-is), but the ring is built
    from an ellipse fitted to the nucleus (centroid, orientation and axis
    lengths via ``regionprops``) instead of dilating the irregular contour.
    More stable for concave / irregular nuclei, at the cost of ignoring the
    nucleus's real shape when sampling the surrounding cytoplasm. The ellipse's
    semi-axes are grown by a constant pixel amount for the inner/outer ring
    boundary (an approximation, not a true morphological offset of the ellipse
    curve -- adequate when dilation/gap are small relative to the nucleus).
    Falls back to :func:`compartment_masks` for a degenerate (near-zero-area)
    region. Returns empty masks if the id is absent.
    """
    nuc = (label_plane == int(track_id))
    out = {"nuc": nuc, "ring": np.zeros_like(nuc), "cell": nuc.copy()}
    if not nuc.any():
        return out

    from skimage.measure import regionprops
    ys, xs = np.nonzero(nuc)
    _fallback = dict(dilation=dilation, gap=gap, neighbor_gap=neighbor_gap)
    y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = nuc[y0b:y1b, x0b:x1b]
    props = regionprops(crop.astype(np.uint8))
    if not props:
        return compartment_masks(label_plane, track_id, **_fallback)
    r = props[0]
    a0, b0 = r.axis_major_length / 2.0, r.axis_minor_length / 2.0
    if a0 <= 0 or b0 <= 0:
        return compartment_masks(label_plane, track_id, **_fallback)

    y0_local, x0_local = r.centroid
    y0, x0 = y0_local + y0b, x0_local + x0b
    theta = r.orientation
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Y, X = np.indices(label_plane.shape)
    dx, dy = X - x0, Y - y0
    # Major/minor axis projections (skimage's orientation convention: the major
    # axis direction is (-sin(theta), -cos(theta)) in (x, y); sign is irrelevant
    # once squared below).
    p_major = dx * sin_t + dy * cos_t
    p_minor = dx * cos_t - dy * sin_t

    def _ellipse(extra):
        aa, bb = a0 + extra, b0 + extra
        return (p_major / aa) ** 2 + (p_minor / bb) ** 2 <= 1.0

    any_nuc = (label_plane > 0)
    if neighbor_gap > 0:
        others = any_nuc & ~nuc
        if others.any():
            others = ndi.binary_dilation(others, iterations=int(neighbor_gap))
        any_nuc = others | nuc
    inner = _ellipse(gap) if gap > 0 else nuc
    outer = _ellipse(gap + dilation)
    ring = outer & ~inner & ~any_nuc
    out["ring"] = ring
    out["cell"] = nuc | ring
    return out


_COMPARTMENTS = ("nuc", "ring", "cell")
_STATS = ("mean", "median", "sum", "min", "max", "std", "p90", "px")


def _compartment_stats(values):
    """Return the stat dict for one compartment's (bg-subtracted) pixel values.

    ``px`` is the compartment's pixel count, carried alongside the statistics so
    a failed measurement (an empty or vanishing ring) is auditable in the export
    instead of being indistinguishable from a genuine low reading.
    """
    if values.size == 0:
        return dict({s: np.nan for s in _STATS}, px=0.0)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sum": float(np.sum(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "px": float(values.size),
    }


def _ratio(num, den):
    """num/den, or NaN unless BOTH sides are finite and strictly positive.

    Guarding only the denominator (the previous behaviour) let a zero numerator
    through as a hard 0.0 -- a value that reads as "no signal in this
    compartment" when it actually means the compartment measured nothing, e.g. a
    ring whose every pixel fell to or below the background estimate and was
    clipped. Those rows must be NaN so they can be excluded rather than silently
    dragging the distribution down.
    """
    ok = (num is not None and den is not None
          and np.isfinite(num) and np.isfinite(den)
          and num > 0 and den > 0)
    return float(num / den) if ok else np.nan


def measure_intensity(channel_stack, mask, channel_name="ch",
                      dilation=RING_DILATION_DEFAULT, gap=RING_GAP_DEFAULT,
                      subtract_background=True, background_roi=None,
                      background_mode="non_cell_median", use_ellipse=False,
                      neighbor_gap=NEIGHBOR_GAP_DEFAULT,
                      background_dilate=BACKGROUND_DILATE_DEFAULT,
                      on_warning=None):
    """Per (frame, track_id) intensity in nucleus/ring/cell + C/N and N/C ratios.

    Columns: frame, track_id, <name>_<comp>_<stat> for comp in nuc/ring/cell and
    stat in mean/median/sum/min/max/std/p90/px, plus <name>_cn_ratio,
    <name>_cn_ratio_mean, <name>_nc_ratio, <name>_nc_ratio_mean. Every ratio is
    NaN unless BOTH compartments measured a finite, strictly positive value (see
    :func:`_ratio`), so a ring that vanished or fell entirely to the background
    is reported as missing rather than as a zero ratio. The ``_px`` columns say
    how many pixels each compartment actually had, which is what distinguishes
    "no signal" from "no ring". Background is subtracted and clipped at 0 first.

    ``background_roi`` (a boolean array, H x W or T x H x W marking a genuinely
    cell-free region) always wins when given. Otherwise ``background_mode``
    selects the per-frame automatic estimate:
      "far_from_cells" (default) -- median of pixels at least
        ``background_dilate`` px away from every nucleus (see
        :func:`background_far_from_cells`); the only automatic estimate that does
        not count perinuclear cytoplasm as background.
      "non_cell_median" -- median of pixels outside every nucleus (mask == 0);
        robust to a single bad pixel, but biased upward in a confluent field, and
        increasingly so as the field fills up.
      "image_min" -- the frame's single darkest pixel (see
        :func:`background_image_min`); unaffected by confluence, but fragile to
        one hot/dead/noise pixel.

    ``use_ellipse`` selects the ring geometry: the default irregular-contour
    dilation (:func:`compartment_masks`) or an ellipse fitted to the nucleus
    (:func:`compartment_masks_ellipse`), which is more stable for concave /
    irregular nuclei. ``neighbor_gap`` keeps the ring clear of neighbouring
    nuclei by that many extra pixels.

    ``on_warning`` is called with a message string when the background estimate
    is measured on a nearly-full field; ``None`` prints nothing.
    """
    cols = ["frame", "track_id"]
    for c in _COMPARTMENTS:
        for s in _STATS:
            cols.append(f"{channel_name}_{c}_{s}")
    cols += [f"{channel_name}_cn_ratio", f"{channel_name}_cn_ratio_mean",
            f"{channel_name}_nc_ratio", f"{channel_name}_nc_ratio_mean"]
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=cols)

    mask_fn = compartment_masks_ellipse if use_ellipse else compartment_masks
    rows = []
    n = mask.shape[0]
    crowded = []
    for f in range(n):
        lab = mask[f]
        ch = channel_stack[f].astype(float)
        if subtract_background:
            if background_roi is not None:
                roi = (background_roi[f] if background_roi.ndim == 3
                       else background_roi).astype(bool)
                vals = ch[roi]
                bg = float(np.median(vals)) if vals.size else 0.0
            elif background_mode == "image_min":
                bg = background_image_min(ch)
            elif background_mode == "non_cell_median":
                bg = background_per_frame(ch, lab)
            else:
                bg, free_frac = background_far_from_cells(
                    ch, lab, dilate_px=background_dilate)
                if free_frac < 0.02:
                    crowded.append(f)
            ch = np.clip(ch - bg, 0, None)
        ids = np.unique(lab)
        for tid in ids[ids > 0]:
            m = mask_fn(lab, int(tid), dilation=dilation, gap=gap,
                        neighbor_gap=neighbor_gap)
            rec = {"frame": int(f), "track_id": int(tid)}
            per_comp = {}
            for c in _COMPARTMENTS:
                st = _compartment_stats(ch[m[c]])
                per_comp[c] = st
                for s in _STATS:
                    rec[f"{channel_name}_{c}_{s}"] = st[s]
            nuc_med, ring_med = per_comp["nuc"]["median"], per_comp["ring"]["median"]
            nuc_mean, ring_mean = per_comp["nuc"]["mean"], per_comp["ring"]["mean"]
            rec[f"{channel_name}_cn_ratio"] = _ratio(ring_med, nuc_med)
            rec[f"{channel_name}_cn_ratio_mean"] = _ratio(ring_mean, nuc_mean)
            rec[f"{channel_name}_nc_ratio"] = _ratio(nuc_med, ring_med)
            rec[f"{channel_name}_nc_ratio_mean"] = _ratio(nuc_mean, ring_mean)
            rows.append(rec)
    if crowded and on_warning:
        on_warning(f"Background: less than 2% of the field was free of cells in "
                   f"{len(crowded)} of {n} frames (first: {crowded[0]}). The "
                   f"background estimate there is unreliable; draw a cell-free "
                   f"ROI instead.")
    return pd.DataFrame(rows, columns=cols)


def ring_sweep(channel_stack, mask, dilations=(2, 3, 4, 6, 8, 10, 12),
               gap=RING_GAP_DEFAULT, frames=None, **kwargs):
    """C/N ratio and ring size as a function of ring width, for picking a width.

    The default ring is thin enough to sit inside the nucleus's own out-of-focus
    halo, which makes C/N report the nuclear signal twice instead of comparing
    two compartments. Widening the ring pushes it into real cytoplasm, and the
    measured C/N stops changing once it gets there -- so the usable width is the
    start of that plateau, not a fixed default.

    Returns one row per dilation: median C/N, median ring/nucleus pixel counts,
    and the fraction of cells whose ratio came back NaN. ``frames`` selects the
    frames to sample (default: 8 spread across the movie), keeping the sweep
    cheap on a long timelapse.
    """
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=["dilation", "cn_median", "ring_px_median",
                                     "nuc_px_median", "frac_nan", "n"])
    n = mask.shape[0]
    if frames is None:
        frames = np.unique(np.linspace(0, n - 1, min(8, n)).astype(int))
    frames = [int(f) for f in frames]
    sub_mask = mask[frames]
    sub_stack = channel_stack[frames]
    rows = []
    for dil in dilations:
        df = measure_intensity(sub_stack, sub_mask, channel_name="s",
                               dilation=int(dil), gap=int(gap), **kwargs)
        if df.empty:
            continue
        cn = df["s_cn_ratio"]
        rows.append({
            "dilation": int(dil),
            "cn_median": float(cn.median(skipna=True)),
            "ring_px_median": float(df["s_ring_px"].median()),
            "nuc_px_median": float(df["s_nuc_px"].median()),
            "frac_nan": float(cn.isna().mean()),
            "n": int(len(df)),
        })
    return pd.DataFrame(rows)


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


def signal_drift(channel_stack, mask):
    """Per-frame in-mask median relative to the first frame's.

    Exposes how the whole in-cell intensity distribution moves over a long
    timelapse, so an absolute-intensity or texture trend can be checked against
    it instead of being read straight as biology.

    DO NOT DIVIDE BY THIS. The quantity mixes two causes it cannot separate:
    photobleaching (optical, pushes it down) and any real change in the marker
    (biological, pushes it either way). For a CONSTITUTIVE marker it is
    essentially the bleaching curve. For an INDUCIBLE one -- 53BP1, gammaH2AX,
    a stress reporter -- it rises with the very response being measured, so
    dividing by it removes the signal. Measured on a real dataset: the in-mask
    median rose from 0.60 to 0.83 across a treatment window and correlated with
    the 53BP1 level at rho = +0.58, and "correcting" a highly significant damage
    response (p = 0.004) by it drove the same test to p = 1.0.

    Separating the two needs a reference that does not respond to the treatment
    (a constitutive channel, or an untreated well imaged alongside). Absent
    that, prefer texture features, which describe the SHAPE of the in-nucleus
    distribution and are far less sensitive to a global scaling than the median.

    Returns a DataFrame with columns frame, signal_drift (1.0 at the first frame
    that has any cell, NaN for empty frames).
    """
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=["frame", "signal_drift"])
    meds = []
    for f in range(mask.shape[0]):
        inside = channel_stack[f][mask[f] > 0]
        meds.append(float(np.median(inside)) if inside.size else np.nan)
    meds = np.asarray(meds, dtype=float)
    ok = np.nonzero(np.isfinite(meds) & (meds > 0))[0]
    ref = meds[ok[0]] if ok.size else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = meds / ref if (ref and np.isfinite(ref)) else np.full_like(meds, np.nan)
    return pd.DataFrame({"frame": np.arange(mask.shape[0]),
                         "signal_drift": factor})


def haralick_features(channel_stack, mask, channel_name="ch", levels=16,
                      vmin=None, vmax=None, subtract_background=True,
                      background_mode="far_from_cells",
                      background_dilate=BACKGROUND_DILATE_DEFAULT):
    """Per (frame, track_id) Haralick texture on the nuclear ROI (16 levels).

    Quantisation is global (one vmin/vmax for the whole stack) so texture is
    comparable across frames. That only holds if the frames are on a common
    intensity baseline, which is why the background is subtracted per frame
    first: an uncorrected baseline shift walks pixels between quantisation bins
    and shows up as a texture trend that is really an illumination trend. See
    :func:`signal_drift` for the part this cannot fix.
    """
    cols = ["frame", "track_id"] + [f"{channel_name}_hara_{c}" for c in _HARA_COLS]
    if channel_stack is None or mask is None:
        return pd.DataFrame(columns=cols)
    ch_all = channel_stack.astype(float)
    if subtract_background:
        corrected = np.empty_like(ch_all)
        for f in range(ch_all.shape[0]):
            plane, lab = ch_all[f], mask[f]
            if background_mode == "image_min":
                bg = background_image_min(plane)
            elif background_mode == "non_cell_median":
                bg = background_per_frame(plane, lab)
            else:
                bg, _ = background_far_from_cells(plane, lab,
                                                  dilate_px=background_dilate)
            corrected[f] = np.clip(plane - bg, 0, None)
        ch_all = corrected
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
                        frame=None, rng=None, use_ellipse=False,
                        neighbor_gap=NEIGHBOR_GAP_DEFAULT):
    """Small-multiples preview of the cytoplasm ring growing on one nucleus.

    ``use_ellipse`` previews the same geometry :func:`measure_intensity` would
    use (see :func:`compartment_masks_ellipse`), so the parameters can be
    sanity-checked before running a full measurement.

    Uses whatever matplotlib backend is already active (the app selects an
    interactive Qt backend; headless tests select Agg themselves) -- it must not
    force a backend here, or clicking Preview would switch the whole session to a
    non-interactive backend and no plot would ever show again.
    """
    mask_fn = compartment_masks_ellipse if use_ellipse else compartment_masks
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
        m = mask_fn(lab_c, tid, dilation=int(dil), gap=int(gap),
                    neighbor_gap=int(neighbor_gap))
        ax.imshow(ch_c, cmap="gray")
        ax.imshow((lab_c > 0) & (lab_c != tid), cmap=others, vmin=0, vmax=1)
        ax.contour((lab_c == tid).astype(float), levels=[0.5], colors="#33ff88", linewidths=1.2)
        ax.imshow(m["ring"], cmap=overlay, vmin=0, vmax=1, alpha=0.7)
        ax.set_title(f"dilation={dil}px, gap={gap}px", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    geom = "ellipse-fit" if use_ellipse else "contour dilation"
    fig.suptitle(f"Ring preview ({geom}) — nucleus {tid} @ frame {f}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
