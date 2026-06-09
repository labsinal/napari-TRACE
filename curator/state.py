"""
Mutable session state, undo history and the ID pool.

All mutable state lives on a single :class:`CuratorState` object that curation
operations receive explicitly.

Key designs
-----------
Delta undo.
    ``save_state`` no longer copies the entire mask stack on every action.
    Callers declare which frames they are about to dirty and only those planes
    are snapshotted, together with a copy of the dataframe. Operations that
    genuinely rewrite every frame (merge, exterminate, harmonize, re-sequence)
    pass ``full=True`` to snapshot the whole stack.

Incremental ID pool.
    The set of occupied IDs is computed once at construction. New IDs are served
    from a monotonically advancing cursor and the occupied set is updated in
    place, so no ``np.unique`` scan of the whole stack happens per call.

State encapsulation.
    ``CuratorState`` owns ``df`` and ``mask`` and exposes the few mutation
    primitives operations need (``set_mask_plane``, ``reserve_id`` ...). The UI
    binds a napari labels layer onto the same array via :meth:`attach_mask_layer`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config


class IDPool:
    """Serves fresh positive integer IDs without rescanning the mask stack.

    The occupied set is seeded once; thereafter a cursor only ever advances,
    and freshly minted IDs are recorded so they are never handed out twice in a
    session even before they appear in the dataframe or mask.
    """

    def __init__(self, occupied: Iterable[int] | None = None):
        self._occupied: set[int] = set()
        if occupied:
            self.bulk_add(occupied)
        self._cursor = 1  # smallest candidate not yet ruled out

    def bulk_add(self, ids: Iterable[int]) -> None:
        for v in ids:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                self._occupied.add(iv)

    def add(self, value: int) -> None:
        if value and int(value) > 0:
            self._occupied.add(int(value))

    def is_free(self, value: int) -> bool:
        return int(value) not in self._occupied

    def new_id(self) -> int:
        """Return the smallest free *normal* ID and mark it occupied."""
        c = self._cursor
        while c in self._occupied:
            c += 1
        if c >= config.NORMAL_ID_MAX:
            # Extremely unlikely; warn but keep going.
            print(f"Warning: normal-ID space exhausted near {c}.")
        self._occupied.add(c)
        self._cursor = c + 1
        return c

    def new_quarantine_id(self) -> int:
        """Return a fresh ID inside the quarantine range (for evictions)."""
        c = max(config.QUARANTINE_START, getattr(self, "_q_cursor", config.QUARANTINE_START))
        while c in self._occupied:
            c += 1
        self._occupied.add(c)
        self._q_cursor = c + 1
        return c

    @classmethod
    def from_state(cls, df: pd.DataFrame, mask: np.ndarray | None) -> "IDPool":
        occ: set[int] = set()
        if df is not None and not df.empty:
            if config.COL_TRACK in df:
                occ |= {int(v) for v in pd.unique(df[config.COL_TRACK].dropna()) if v > 0}
            if config.COL_PARENT in df:
                occ |= {int(v) for v in pd.unique(df[config.COL_PARENT].dropna()) if v > 0}
        if mask is not None:
            occ |= {int(v) for v in np.unique(mask) if v > 0}
        return cls(occ)


class _UndoEntry:
    """One reversible step: a dataframe snapshot plus selected mask planes.

    If ``full`` is set, ``planes`` holds the entire stack; otherwise it maps
    frame index -> copied 2-D plane for only the frames the action dirtied.
    """

    __slots__ = ("df", "planes", "full", "label")

    def __init__(self, df, planes, full, label):
        self.df = df
        self.planes = planes
        self.full = full
        self.label = label


class CuratorState:
    """Owns the mutable curation state (dataframe + mask) and its undo history."""

    def __init__(self, df: pd.DataFrame, mask: np.ndarray,
                 max_history: int = 10):
        self.df = df
        self.mask = mask
        self.max_history = max_history
        self._history: list[_UndoEntry] = []
        self.pool = IDPool.from_state(df, mask)
        # The UI sets this so the state can refresh the on-screen layer after
        # an undo. Kept as a plain callable to avoid importing Qt here.
        self._mask_layer = None

    # -- UI binding ------------------------------------------------------
    def attach_mask_layer(self, layer) -> None:
        """Bind a napari labels layer whose ``.data`` IS ``self.mask``."""
        self._mask_layer = layer
        self.mask = layer.data

    def _refresh_layer(self, full: bool) -> None:
        if self._mask_layer is None:
            return
        if full:
            # Reassigning .data is needed when the underlying array object changed.
            self._mask_layer.data = self.mask
        else:
            self._mask_layer.refresh()

    # -- ID pool ---------------------------------------------------------
    def reserve_id(self) -> int:
        return self.pool.new_id()

    def reserve_quarantine_id(self) -> int:
        return self.pool.new_quarantine_id()

    # -- Undo ---------------------------------------------------
    def save_state(self, frames: Optional[Iterable[int]] = None,
                   full: bool = False, label: str = "") -> None:
        """Snapshot before a mutation.

        Parameters
        ----------
        frames : iterable of int, optional
            The frame indices the upcoming action will modify. Only these mask
            planes are copied. Ignored when ``full`` is True.
        full : bool
            Snapshot the entire mask stack (for operations that touch every
            frame). Use sparingly.
        label : str
            Human-readable action name, surfaced on undo.
        """
        if full or frames is None:
            planes = self.mask.copy()
            full = True
        else:
            planes = {}
            n = self.mask.shape[0]
            for f in set(int(f) for f in frames):
                if 0 <= f < n:
                    planes[f] = self.mask[f].copy()
        entry = _UndoEntry(self.df.copy(deep=True), planes, full, label)
        self._history.append(entry)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> str:
        """Revert the last snapshot. Returns a status message."""
        if not self._history:
            return ""
        entry = self._history.pop()
        self.df = entry.df
        if entry.full:
            self.mask = entry.planes
            if self._mask_layer is not None:
                self._mask_layer.data = self.mask
        else:
            for f, plane in entry.planes.items():
                self.mask[f] = plane
            if self._mask_layer is not None:
                self._mask_layer.refresh()
        return entry.label or "action"

    def clear_history(self) -> None:
        self._history.clear()

    # -- Mask helpers ----------------------------------------------------
    def set_mask_plane(self, frame: int, plane: np.ndarray) -> None:
        self.mask[frame] = plane

    def relabel_everywhere(self, old_id: int, new_id: int) -> None:
        """Replace one label by another across the whole stack (full op)."""
        self.mask[self.mask == old_id] = new_id
