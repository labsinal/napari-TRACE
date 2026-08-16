"""
Extra fluorescence-channel loading (display-only).

Beyond the raw movie, the mask and the tracking table, an experiment often has
additional fluorescence channels -- an apoptosis reporter (cleaved Caspase-3), a
cell-cycle indicator (FUCCI), a viability dye -- whose value to curation is
purely visual: at the frame where the algorithm flags an "Ambiguous" or "Death"
event, toggling the reporter on turns a guess into a biological call.

These channels are therefore loaded as **display-only image layers**. They are
deliberately kept out of:
  * ``state.mask`` and the ID pool (they are not segmentation),
  * ``update_visuals`` label/track logic (they carry no track identity),
  * every save / export path (they are read-only inputs, never rewritten).

Two storage conventions are supported, because real datasets use both:

1. **Separate files / folders, one per channel.** Each channel is its own TIF
   stack (T x H x W) or its own folder of per-frame TIFs. Discovered by name
   (files/dirs containing a known marker token, or matching ``*_chN``), or
   passed explicitly.

2. **One multi-channel stack.** A single TIF carrying a channel axis, e.g.
   T x C x H x W or T x H x W x C. The channel axis is detected as the small
   non-time, non-spatial axis and split into one display layer per channel.

Nothing here imports Qt or napari, so it can be unit-tested headless; the UI
layer turns the returned arrays into ``viewer.add_image`` layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import tifffile

from . import io_adapters


# Tokens that commonly name a fluorescence channel in a file/folder name. Used
# only to AUTO-DISCOVER separate channel files; explicit paths bypass this.
CHANNEL_NAME_TOKENS = (
    "caspase", "casp3", "annexin", "propidium", "pi_", "fucci", "gfp", "rfp",
    "yfp", "cfp", "mcherry", "dapi", "hoechst", "fluo", "fluor", "channel",
    "_ch", "ch0", "ch1", "ch2", "ch3", "c0", "c1", "c2", "c3",
)

# A reasonable palette so several channels are visually distinct by default.
DEFAULT_CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")


@dataclass
class ChannelLayer:
    """One channel: what is shown on screen, and what is measured from.

    ``data`` is the DISPLAY array. It is routinely a contrast-enhanced or
    denoised version of the acquisition, because that is what a human can
    actually see -- and enhancing it is the whole point of preprocessing.

    ``measure_data`` is the array quantification reads. It must be the raw
    acquisition: CLAHE, per-frame min-max normalisation, gamma and denoising are
    all local and/or non-monotonic, so any intensity RATIO computed on them
    (a KTR's cytoplasm/nucleus ratio above all) no longer measures what it
    claims to. Keeping the two apart is the only way the display can stay
    readable while the numbers stay true.

    When ``measure_data`` is None the measurement falls back to ``data`` -- the
    historical behaviour, preserved so existing sessions keep working -- and
    ``measurement_is_display`` is True so the UI can say so out loud instead of
    letting it pass unnoticed.
    """
    name: str
    data: np.ndarray          # T x H x W (single channel), for display
    colormap: str
    color: str = ""           # user-chosen tag (green/red/...); defaults to colormap
    measure: bool = False     # compute per-cell fluorescence features for this channel
    role: str = ""            # "" | "nucleus" | "cytoplasm"
    measure_data: np.ndarray | None = None   # raw source for quantification
    measure_source: str = ""  # provenance: where measure_data came from

    def __post_init__(self):
        if not self.color:
            self.color = self.colormap

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim >= 3 else 1

    @property
    def measurement_is_display(self) -> bool:
        """True when quantification is falling back to the display array."""
        return self.measure_data is None

    @property
    def quant(self) -> np.ndarray:
        """The array to quantify: the raw source if set, else the display."""
        return self.data if self.measure_data is None else self.measure_data


# ---------------------------------------------------------------------------
# Measurement-source validation
# ---------------------------------------------------------------------------
# Folder/file name fragments that mark an image as already processed. Matching
# one is not proof of anything (a user may legitimately name a raw folder
# "adjusted"), so these only ever raise a warning, never block.
PREPROCESSED_NAME_TOKENS = (
    "clahe", "preprocessed", "preprocess", "denoised", "denoise", "normalized",
    "normalised", "equalized", "equalised", "adjusted", "enhanced", "contrast",
    "gamma", "rescaled", "8bit", "8-bit",
)


def looks_preprocessed(array, source_path="", display=None) -> list:
    """Reasons to suspect ``array`` is not a raw acquisition. Empty == no doubt.

    Three independent hints, none conclusive on its own:
      * the path names a known preprocessing step;
      * the array is 8-bit while the display it accompanies is deeper (the
        display should never carry MORE information than the measurement);
      * the histogram is pinned at both ends, which is what a full-range
        normalisation or histogram equalisation leaves behind.
    """
    reasons = []
    low = str(source_path).lower().replace("\\", "/")
    hit = [t for t in PREPROCESSED_NAME_TOKENS if t in low]
    if hit:
        reasons.append(f"path contains {', '.join(repr(h) for h in hit)}")

    arr = np.asarray(array)
    if display is not None:
        disp = np.asarray(display)
        if (arr.dtype == np.uint8 and disp.dtype != np.uint8):
            reasons.append(f"measurement is {arr.dtype} but display is "
                           f"{disp.dtype} (display richer than measurement)")

    if arr.dtype == np.uint8 and arr.size:
        sample = arr if arr.size <= 4_000_000 else arr.ravel()[::max(1, arr.size // 4_000_000)]
        at_floor = float(np.mean(sample == 0))
        at_ceiling = float(np.mean(sample == 255))
        if at_floor > 0.01 and at_ceiling > 0.01:
            reasons.append(f"histogram pinned at both ends "
                           f"({at_floor:.1%} at 0, {at_ceiling:.1%} at 255): "
                           f"looks range-normalised or equalised")
    return reasons


def match_frames_by_name(candidate_names, reference_names):
    """Pick, in reference order, the candidate files that the reference names.

    A raw channel folder routinely holds MORE frames than the segmentation was
    run on: frames dropped for focus, or an acquisition that was interrupted and
    resumed. Taking the first N, or trusting position, silently pairs each mask
    with the wrong photo from the first missing frame onward. Names are the only
    thing that survives both the drop and the stacking, so they are what we match
    on.

    Returns ``(picked, missing)``: ``picked`` are the candidate entries in
    reference order, ``missing`` are reference names with no candidate. An empty
    ``missing`` means the reference is a subset of the candidates and the subset
    is exact.
    """
    by_base = {}
    for c in candidate_names:
        by_base.setdefault(os.path.basename(str(c)), str(c))
    picked, missing = [], []
    for r in reference_names:
        base = os.path.basename(str(r))
        if base in by_base:
            picked.append(by_base[base])
        else:
            missing.append(base)
    return picked, missing


def check_measurement_source(array, mask_frames, reference=None, display=None,
                             source_path="", max_offset_px=2.0, n_probe=5):
    """Validate a candidate measurement stack against the mask it will be used with.

    Returns ``(errors, warnings)``. Errors are conditions that make the numbers
    meaningless (blocking); warnings are conditions that are usually mistakes but
    can be legitimate.

    The alignment probe is the one that catches an UNREGISTERED channel: drift
    correction is applied per channel, and a channel that skipped it lands tens
    or hundreds of pixels away from the mask, so every "nucleus" is sampled off
    the actual nucleus. Nothing else in the pipeline notices this.
    """
    errors, warnings = [], []
    arr = np.asarray(array)
    if arr.ndim < 3:
        errors.append(f"expected a T x H x W stack, got shape {arr.shape}")
        return errors, warnings
    if arr.shape[0] != int(mask_frames):
        errors.append(f"stack has {arr.shape[0]} frames, mask has {mask_frames}")

    warnings.extend(looks_preprocessed(arr, source_path=source_path, display=display))

    if reference is not None:
        ref = np.asarray(reference)
        if ref.shape[0] == arr.shape[0] and ref.shape[1:] == arr.shape[1:]:
            try:
                import cv2
            except ImportError:
                return errors, warnings
            n = arr.shape[0]
            probes = np.unique(np.linspace(0, n - 1, min(int(n_probe), n)).astype(int))
            worst, worst_f = 0.0, 0
            for f in probes:
                (dx, dy), _ = cv2.phaseCorrelate(np.float32(ref[f]), np.float32(arr[f]))
                off = float(np.hypot(dx, dy))
                if off > worst:
                    worst, worst_f = off, int(f)
            if worst > max_offset_px:
                errors.append(
                    f"misaligned with the mask by {worst:.1f} px at frame {worst_f} "
                    f"(tolerance {max_offset_px} px). This channel looks "
                    f"un-registered: run the drift correction on it first.")
        elif ref.shape[1:] != arr.shape[1:]:
            errors.append(f"frame size {arr.shape[1:]} does not match the "
                          f"movie's {ref.shape[1:]}")
    return errors, warnings


# ---------------------------------------------------------------------------
# Multi-channel stack splitting
# ---------------------------------------------------------------------------
def _guess_channel_axis(arr: np.ndarray, n_frames_hint: int | None) -> int | None:
    """Return the axis index that is the CHANNEL axis of a >3-D stack, or None.

    Works for both T x C x H x W (channel leading) and T x H x W x C (channel
    trailing). Strategy: the two LARGEST axes are spatial (H, W); of the
    remaining axes, the time axis is the one matching the known frame count (or,
    absent a hint, the largest remaining); the channel axis is then the other
    remaining axis, accepted only if it is small (<= 8), which is what a real
    channel count looks like.
    """
    if arr.ndim < 4:
        return None
    shape = list(arr.shape)
    order_by_size = sorted(range(arr.ndim), key=lambda a: shape[a], reverse=True)
    spatial = set(order_by_size[:2])            # two largest axes = H, W
    rest = [a for a in range(arr.ndim) if a not in spatial]
    if not rest:
        return None
    # Identify time among the rest.
    if n_frames_hint is not None and any(shape[a] == n_frames_hint for a in rest):
        time_ax = next(a for a in rest if shape[a] == n_frames_hint)
    else:
        time_ax = max(rest, key=lambda a: shape[a])
    cand = [a for a in rest if a != time_ax]
    if not cand:
        return None
    ax = min(cand, key=lambda a: shape[a])
    if shape[ax] <= 8:
        return ax
    return None


def split_multichannel(arr: np.ndarray, n_frames_hint: int | None = None,
                       base_name: str = "channel") -> list[ChannelLayer]:
    """Split a multi-channel TIF into one ChannelLayer per channel.

    Accepts T x C x H x W or T x H x W x C (or any layout where the channel axis
    is a small non-spatial axis). Returns [] when no channel axis is found (the
    caller then treats the array as an ordinary single-channel stack).
    """
    ax = _guess_channel_axis(arr, n_frames_hint)
    if ax is None:
        return []
    moved = np.moveaxis(arr, ax, 0)    # C first
    layers = []
    for i in range(moved.shape[0]):
        cmap = DEFAULT_CHANNEL_COLORMAPS[i % len(DEFAULT_CHANNEL_COLORMAPS)]
        layers.append(ChannelLayer(name=f"{base_name}_{i}", data=moved[i],
                                    colormap=cmap))
    return layers


# ---------------------------------------------------------------------------
# Loading a single channel source (file or folder)
# ---------------------------------------------------------------------------
def _load_stack(path: str) -> np.ndarray:
    """Read a TIF file or a folder of per-frame TIFs into a T x H x W array."""
    if os.path.isdir(path):
        stack_path = os.path.join(
            os.path.dirname(path.rstrip("/\\")),
            os.path.basename(path.rstrip("/\\")) + "_stack.tif")
        if os.path.exists(stack_path):
            return tifffile.imread(stack_path)
        return io_adapters.build_stack_from_folder(path, stack_path)
    return tifffile.imread(path)


def load_channel_sources(paths, n_frames_hint: int | None = None
                         ) -> list[ChannelLayer]:
    """Load an explicit list of channel sources (files and/or folders).

    Each source may itself be a multi-channel stack, which is split. Channels
    whose frame count disagrees with ``n_frames_hint`` are still returned (with a
    printed warning) so the user can decide; nothing is dropped silently.
    """
    layers: list[ChannelLayer] = []
    for p in (paths or []):
        if not p or not os.path.exists(p):
            print(f"Channel source not found, skipping: {p}")
            continue
        try:
            arr = _load_stack(p)
        except Exception as exc:
            print(f"Failed to read channel '{p}': {exc}")
            continue
        base = os.path.splitext(os.path.basename(p.rstrip("/\\")))[0]
        sub = split_multichannel(arr, n_frames_hint, base_name=base)
        if sub:
            layers.extend(sub)
        else:
            cmap = DEFAULT_CHANNEL_COLORMAPS[len(layers) % len(DEFAULT_CHANNEL_COLORMAPS)]
            layers.append(ChannelLayer(name=base, data=arr, colormap=cmap))

    if n_frames_hint is not None:
        for L in layers:
            if L.n_frames != n_frames_hint:
                print(f"WARNING: channel '{L.name}' has {L.n_frames} frames, "
                      f"movie has {n_frames_hint}; overlay may be misaligned.")
    # Re-assign colormaps in final order so they stay distinct after splitting.
    for i, L in enumerate(layers):
        L.colormap = DEFAULT_CHANNEL_COLORMAPS[i % len(DEFAULT_CHANNEL_COLORMAPS)]
    return layers


# ---------------------------------------------------------------------------
# Auto-discovery inside the working folder
# ---------------------------------------------------------------------------
def discover_channel_sources(folder, used_paths=()) -> list[str]:
    """Find likely fluorescence-channel files/folders in ``folder``.

    Returns paths NOT already consumed as the raw image, mask or table
    (``used_paths``). A file or subfolder qualifies if its name contains a known
    channel token. This is best-effort: anything missed can still be added
    explicitly via the loader dialog / CLI.
    """
    if not folder or not os.path.isdir(folder):
        return []
    used = {os.path.abspath(p) for p in used_paths if p}
    found = []
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        ap = os.path.abspath(full)
        if ap in used:
            continue
        low = entry.lower()
        is_tif = low.endswith((".tif", ".tiff"))
        is_framedir = os.path.isdir(full) and io_adapters.looks_like_frame_folder(full)
        if not (is_tif or is_framedir):
            continue
        # Exclude the things already used and the curator's own derived stacks.
        if low.endswith("_stack.tif") and ap in used:
            continue
        if any(tok in low for tok in CHANNEL_NAME_TOKENS):
            found.append(full)
    return found
