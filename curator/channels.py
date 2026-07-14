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
