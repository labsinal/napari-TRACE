"""
Curation operations.

Pure(ish) functions that mutate a :class:`~curator.state.CuratorState`. They
contain no Qt code: the UI layer wires buttons to these and handles
notifications. Each operation declares which frames it dirties so the delta
undo only snapshots what it must.

A single ``sync_masks`` works over a frame range ("this frame" is just a
one-frame range) and every new-ID request is routed through the incremental
pool. Disconnected-blob splitting and harmonization use the vectorized
centroid helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label as cc_label

from . import config
from .config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_CLABEL,
    COL_TREATMENT, OUTCOME_MITOSIS, OUTCOME_UNKNOWN, TREAT_CONTROL,
)
from . import analysis
from . import lineage


class OpResult:
    """Small carrier for an operation's outcome message and success flag."""

    def __init__(self, ok: bool, message: str, needs_confirm: bool = False):
        self.ok = ok
        self.message = message
        self.needs_confirm = needs_confirm

    def __bool__(self):
        return self.ok


def _treatment_label(frame, treatment_config):
    from .treatment import treatment_for_frame
    return treatment_for_frame(int(frame), treatment_config)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
def apply_flag(state, track_id, outcome):
    if track_id == 0:
        return OpResult(False, "Select an ID in box [A].")
    state.save_state(frames=[], label=f"flag {outcome}")  # df-only change
    state.df.loc[state.df[COL_TRACK] == track_id, COL_OUTCOME] = outcome
    return OpResult(True, f"Flag '{outcome}' applied to track {track_id}.")


def clear_flag(state, track_id):
    if track_id == 0:
        return OpResult(False, "Select an ID [A].")
    state.save_state(frames=[], label="clear flag")
    state.df.loc[state.df[COL_TRACK] == track_id, COL_OUTCOME] = ""
    return OpResult(True, f"Flags cleared from track {track_id}.")


# ---------------------------------------------------------------------------
# Lineage link
# ---------------------------------------------------------------------------
def link_parent(state, mother, daughter, force=False):
    if mother == 0 or daughter == 0:
        return OpResult(False, "Select the mother in [A] and the daughter in [B].")
    ok, why = lineage.can_link(state.df, mother, daughter, sovereign=True)
    if not ok:
        return OpResult(False, why)

    df = state.df
    groups = lineage.classify_lineage(df)

    # What would this link override?
    cur = df.loc[df[COL_TRACK] == daughter, COL_PARENT]
    cur = cur[cur > 0]
    current_mother = int(cur.iloc[0]) if not cur.empty else None
    if current_mother in (int(mother), None):
        current_mother = None  # already this mother, or none -> no conflict
    existing = [int(d) for d in groups.children_of.get(int(mother), [])
                if int(d) != int(daughter)]
    over_capacity = len(existing) >= 2

    # Manual link is the final word, but a destructive override must be confirmed
    # by the caller (the UI shows OK/Cancel). Until forced, report what it would
    # do and ask for confirmation instead of silently overwriting.
    if not force and (current_mother is not None or over_capacity):
        warn = []
        if current_mother is not None:
            warn.append(f"Cell {daughter} already has mother {current_mother}.")
        if over_capacity:
            warn.append(f"Mother {mother} already has two daughters "
                        f"({existing[0]}, {existing[1]}).")
        warn.append("Click OK to replace; the manual link wins.")
        return OpResult(False, " ".join(warn), needs_confirm=True)

    state.save_state(frames=[], label="link lineage")
    evicted = None
    # Sovereign link: if the mother already has two daughters and this is a new
    # one, drop one existing daughter (parent_id -> -1) to make room, so the
    # manual link always wins. Evict the daughter that appears LATEST (most
    # likely the spurious assignment); ties broken by larger track_id.
    if over_capacity:
        firsts = df.dropna(subset=[COL_FRAME]).groupby(COL_TRACK)[COL_FRAME].min()
        evicted = sorted(existing,
                         key=lambda d: (int(firsts.get(int(d), 0)), int(d)),
                         reverse=True)[0]
        df.loc[df[COL_TRACK] == evicted, COL_PARENT] = -1

    df.loc[df[COL_TRACK] == daughter, COL_PARENT] = mother
    df.loc[df[COL_TRACK] == mother, COL_OUTCOME] = OUTCOME_MITOSIS
    msg = f"{daughter} is now a daughter of {mother}; mother flagged Mitosis."
    if current_mother is not None:
        msg += f" Reassigned from previous mother {current_mother}."
    if evicted is not None:
        msg += f" Daughter {evicted} was detached (mother already had two)."
    return OpResult(True, msg)


def unlink_parent(state, child):
    """Detach a cell from its mother (parent_id -> -1).

    Used by the lineage editor's remove-daughter / detach actions. If the former
    mother is left with no daughters, its auto Mitosis flag is cleared so the
    lineage stays coherent.
    """
    if child == 0:
        return OpResult(False, "Select a cell to detach.")
    df = state.df
    cur = df.loc[df[COL_TRACK] == int(child), COL_PARENT]
    cur = cur[cur > 0]
    if cur.empty:
        return OpResult(False, f"Cell {int(child)} has no parent to remove.")
    mother = int(cur.iloc[0])
    state.save_state(frames=[], label="unlink lineage")
    df.loc[df[COL_TRACK] == int(child), COL_PARENT] = -1
    msg = f"Cell {int(child)} detached from mother {mother}."
    remaining = lineage.classify_lineage(df).children_of.get(mother, [])
    if not remaining:
        sel = (df[COL_TRACK] == mother) & \
              (df[COL_OUTCOME].astype(str) == OUTCOME_MITOSIS)
        if sel.any():
            df.loc[sel, COL_OUTCOME] = ""
            msg += f" Mother {mother} has no other daughters; Mitosis flag cleared."
    return OpResult(True, msg)


# ---------------------------------------------------------------------------
# Merge / swap / relabel / delete
# ---------------------------------------------------------------------------
def _repoint_children(df, old_parent, new_parent):
    """Repoint every cell whose mother is ``old_parent`` to ``new_parent``.

    When a track is renamed (merge / relabel / swap / autotrack), its daughters
    still record the OLD id in their parent_id and would be orphaned (and
    un-relinkable). This keeps the lineage consistent by following the rename.
    A cell is never made its own mother (guards the case where ``new_parent``
    was itself a child of ``old_parent``). ``new_parent`` may be -1 to detach
    (rarely used here). No-op when old == new.
    """
    if int(old_parent) == int(new_parent):
        return
    sel = (df[COL_PARENT] == int(old_parent)) & (df[COL_TRACK] != int(new_parent))
    df.loc[sel, COL_PARENT] = int(new_parent)


def _track_parent(df, t):
    """Return the recorded parent_id of track ``t`` (>0) or -1 if none."""
    p = df.loc[df[COL_TRACK] == int(t), COL_PARENT]
    p = p[p > 0]
    return int(p.iloc[0]) if not p.empty else -1


def _dissolve_division(df, mother):
    """Detach every daughter of ``mother`` and clear its Mitosis flag.

    Used when an edit proves a recorded division was wrong (e.g. a mother merged
    with one of its own daughters): the division no longer exists, so the other
    daughters must lose their parent and the mother must lose the Mitosis flag.
    Returns a short message describing what changed (or "").
    """
    m = int(mother)
    daughters = sorted({int(t) for t in
                        df.loc[df[COL_PARENT] == m, COL_TRACK].dropna().astype(int).unique()
                        if int(t) != m})
    parts = []
    if daughters:
        df.loc[df[COL_PARENT] == m, COL_PARENT] = -1
        parts.append(f"daughters {daughters} detached")
    sel = (df[COL_TRACK] == m) & (df[COL_OUTCOME].astype(str) == OUTCOME_MITOSIS)
    if sel.any():
        df.loc[sel, COL_OUTCOME] = ""
        parts.append(f"Mitosis flag of {m} cleared")
    return ("Division of " + str(m) + " dissolved (" + "; ".join(parts) + ").") \
        if parts else ""


def dissolve_invalid_divisions(df):
    """Repair any division left topologically invalid by an edit (in place).

    A recorded division (mother -> daughters) is dissolved when the mother no
    longer exists, a cell is its own parent, or a daughter is born at/before the
    mother. Dissolving detaches the daughters (parent_id -> -1) and clears the
    mother's Mitosis flag, so a wrongly recorded mitosis disappears from the
    genealogy after the cells are edited. Returns a short message (or "").
    """
    if df is None or df.empty or COL_PARENT not in df.columns:
        return ""
    notes = []
    # Self-loops (a cell pointing at itself) -> detach.
    self_loop = df[COL_TRACK].astype("Int64") == df[COL_PARENT].astype("Int64")
    if self_loop.any():
        df.loc[self_loop, COL_PARENT] = -1
    tracks = set(df[COL_TRACK].dropna().astype(int).unique())
    firsts = (df.dropna(subset=[COL_FRAME])
              .groupby(df[COL_TRACK].astype(int))[COL_FRAME].min().to_dict())
    groups = lineage.classify_lineage(df)
    for mother, daughters in list(groups.children_of.items()):
        m = int(mother)
        invalid = m not in tracks
        if not invalid:
            mf = firsts.get(m)
            for d in daughters:
                d = int(d)
                if d == m:
                    invalid = True
                    break
                df_first = firsts.get(d)
                if df_first is not None and mf is not None and df_first <= mf:
                    invalid = True
                    break
        if invalid:
            msg = _dissolve_division(df, m)
            if msg:
                notes.append(msg)
    return " ".join(notes)


def merge(state, a, b):
    if a == 0 or b == 0:
        return OpResult(False, "Select valid IDs.")
    df = state.df
    # A mother merged with one of its own daughters means the recorded division
    # was a segmentation artefact: the two were the same cell. Detect it BEFORE
    # the rename (afterwards the parent-child relationship is gone) so we can
    # dissolve the division instead of carrying the (now impossible) lineage.
    pa, pb = _track_parent(df, a), _track_parent(df, b)
    mother_daughter = (pa == int(b)) or (pb == int(a))
    grandparent = -1
    div_daughters = []
    if mother_daughter:
        mother = a if pb == int(a) else b      # the one that was the parent
        grandparent = _track_parent(df, mother)
        # Daughters of the false division, captured BEFORE the rename (afterwards
        # they still reference the old mother id). Exclude the cell merged away.
        div_daughters = sorted({int(t) for t in
                                df.loc[df[COL_PARENT] == int(mother), COL_TRACK]
                                .dropna().astype(int).unique() if int(t) != int(a)})

    state.save_state(full=True, label="merge")          # touches all frames
    state.mask[state.mask == a] = b

    if mother_daughter:
        df.loc[df[COL_TRACK] == a, [COL_TRACK, COL_CLABEL]] = b
        # Detach the surviving daughters and anything still pointing at the old
        # mother id or at the merged-away cell, then give b the grandparent.
        if div_daughters:
            df.loc[df[COL_TRACK].astype(int).isin(div_daughters), COL_PARENT] = -1
        df.loc[df[COL_PARENT].isin([int(mother), int(a)]), COL_PARENT] = -1
        gp = int(grandparent) if int(grandparent) != int(b) else -1
        df.loc[df[COL_TRACK] == b, COL_PARENT] = gp
        sel = (df[COL_TRACK] == b) & (df[COL_OUTCOME].astype(str) == OUTCOME_MITOSIS)
        if sel.any():
            df.loc[sel, COL_OUTCOME] = ""
        dissolve_invalid_divisions(df)
        note = (f"division dissolved: daughters {div_daughters} detached, "
                f"Mitosis cleared" if div_daughters else "division dissolved")
        return OpResult(True, f"Track {a} merged into {b}. {note}.")

    mother_of_b = df.loc[df[COL_TRACK] == b, COL_PARENT].max()
    if pd.isna(mother_of_b):
        mother_of_b = -1
    df.loc[df[COL_TRACK] == a, COL_PARENT] = int(mother_of_b)
    df.loc[df[COL_TRACK] == a, [COL_TRACK, COL_CLABEL]] = b
    # Track a no longer exists; any daughters that pointed to a now point to b
    # (a was merged into b, so b inherits a's offspring).
    _repoint_children(df, a, b)
    extra = dissolve_invalid_divisions(df)
    msg = f"Track {a} merged into {b}."
    return OpResult(True, (msg + " " + extra).strip() if extra else msg)


def swap_future(state, a, b, frame):
    if a == 0:
        return OpResult(False, "ID [A] is required.")
    if b == 0:
        b = state.reserve_id()
    n = state.mask.shape[0]
    state.save_state(frames=range(frame, n), label="swap/cut future")
    for f in range(frame, n):
        plane = state.mask[f]
        is_a, is_b = (plane == a), (plane == b)
        plane[is_a] = b
        plane[is_b] = a
    df = state.df
    # Repoint daughters whose birth (first frame) is at/after the swap point:
    # the cell physically called `a` after `frame` is now `b` and vice-versa, so
    # a daughter born from a-after-frame is now a daughter of b, and vice-versa.
    # Daughters born before the swap keep their original mother (that earlier
    # segment kept its name).
    first_frames = df.dropna(subset=[COL_FRAME]).groupby(COL_TRACK)[COL_FRAME].min()
    born_after = set(first_frames[first_frames >= frame].index.astype(int))
    kids_of_a = df[(df[COL_PARENT] == a)][COL_TRACK].astype(int).unique()
    kids_of_b = df[(df[COL_PARENT] == b)][COL_TRACK].astype(int).unique()
    swap_to_b = [k for k in kids_of_a if k in born_after]
    swap_to_a = [k for k in kids_of_b if k in born_after]
    df.loc[df[COL_TRACK].isin(swap_to_b), COL_PARENT] = b
    df.loc[df[COL_TRACK].isin(swap_to_a), COL_PARENT] = a

    mask_a = (df[COL_FRAME] >= frame) & (df[COL_TRACK] == a)
    mask_b = (df[COL_FRAME] >= frame) & (df[COL_TRACK] == b)
    df.loc[mask_a, COL_PARENT] = -1
    df.loc[mask_b, COL_PARENT] = -1
    df.loc[mask_a, COL_OUTCOME] = ""
    df.loc[mask_b, COL_OUTCOME] = ""
    df.loc[mask_a, [COL_TRACK, COL_CLABEL]] = b
    df.loc[mask_b, [COL_TRACK, COL_CLABEL]] = a
    extra = dissolve_invalid_divisions(df)
    return OpResult(True, (f"Future swap: {a} <-> {b}. " + extra).strip())


def swap_local(state, a, b, frame):
    if a == 0:
        return OpResult(False, "Select at least ID [A].")
    if b == 0:
        b = state.reserve_id()
    state.save_state(frames=[frame], label="local swap")
    plane = state.mask[frame]
    is_a, is_b = (plane == a), (plane == b)
    plane[is_a] = b
    plane[is_b] = a
    df = state.df
    idx_a = df[(df[COL_FRAME] == frame) & (df[COL_TRACK] == a)].index
    idx_b = df[(df[COL_FRAME] == frame) & (df[COL_TRACK] == b)].index
    df.loc[idx_a, COL_PARENT] = -1
    df.loc[idx_b, COL_PARENT] = -1
    df.loc[idx_a, COL_OUTCOME] = ""
    df.loc[idx_b, COL_OUTCOME] = ""
    df.loc[idx_a, [COL_TRACK, COL_CLABEL]] = b
    df.loc[idx_b, [COL_TRACK, COL_CLABEL]] = a
    extra = dissolve_invalid_divisions(df)
    return OpResult(True, ("Local swap done. " + extra).strip())


def relabel_new(state, a):
    if a == 0:
        return OpResult(False, "Select the mask in [A].")
    state.save_state(full=True, label="relabel new")
    new_id = state.reserve_id()
    state.mask[state.mask == a] = new_id
    df = state.df
    df.loc[df[COL_TRACK] == a, [COL_TRACK, COL_CLABEL]] = new_id
    df.loc[df[COL_TRACK] == new_id, COL_OUTCOME] = ""
    df.loc[df[COL_TRACK] == new_id, COL_PARENT] = -1
    # The whole track a was renamed to new_id; its daughters follow so the
    # lineage stays intact (they remain children of the same physical cell).
    _repoint_children(df, a, new_id)
    extra = dissolve_invalid_divisions(df)
    return OpResult(True, (f"New track created: {new_id}. " + extra).strip())


def delete_here(state, a, frame):
    if a == 0:
        return OpResult(False, "Select ID in [A].")
    state.save_state(frames=[frame], label="delete (frame)")
    plane = state.mask[frame]
    plane[plane == a] = 0
    df = state.df
    state.df = df[~((df[COL_FRAME] == frame) & (df[COL_TRACK] == a))]
    return OpResult(True, f"Cell {a} deleted from this frame.")


def delete_track(state, a):
    if a == 0:
        return OpResult(False, "Select ID in [A] to exterminate the track.")
    state.save_state(full=True, label="exterminate track")
    state.mask[state.mask == a] = 0
    df = state.df
    df = df[df[COL_TRACK] != a]
    df.loc[df[COL_PARENT] == a, COL_PARENT] = -1
    extra = dissolve_invalid_divisions(df)
    state.df = df
    return OpResult(True, (f"Track {a} exterminated from all frames. " + extra).strip())


# ---------------------------------------------------------------------------
# Unified sync over a frame range
# ---------------------------------------------------------------------------
def sync_masks(state, frames, treatment_config):
    """Reconcile the dataframe with the mask over *frames* (an iterable).

    "This frame" is just ``sync_masks(state, [current], cfg)``; whole-movie is
    ``sync_masks(state, range(n), cfg)``. Uses the vectorized centroid helper.
    """
    frames = list(frames)
    state.save_state(frames=frames, label="sync masks")
    df = state.df
    updated = added = 0
    new_rows = []
    for f in frames:
        plane = state.mask[f]
        cents = analysis.centroids_for_frame(plane)        # {id: (x, y)}
        ids_on_screen = set(cents.keys())
        csv_ids = set(df[df[COL_FRAME] == f][COL_TRACK].tolist())

        for cid, (x, y) in cents.items():
            if cid in csv_ids:
                df.loc[(df[COL_FRAME] == f) & (df[COL_TRACK] == cid),
                       [COL_X, COL_Y]] = [x, y]
                updated += 1
            else:
                new_rows.append({
                    COL_FRAME: f, COL_TRACK: cid, COL_X: x, COL_Y: y,
                    COL_CLABEL: cid, COL_OUTCOME: "", COL_PARENT: -1,
                    COL_TREATMENT: _treatment_label(f, treatment_config),
                })
                added += 1

        to_remove = csv_ids - ids_on_screen
        if to_remove:
            df = df[~((df[COL_FRAME] == f) & (df[COL_TRACK].isin(to_remove)))]

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    state.df = df
    return OpResult(True, f"Synced {len(frames)} frame(s): {updated} updated, {added} added.")


# ---------------------------------------------------------------------------
# Harmonize (split disconnected blobs + force mask value == track_id)
# ---------------------------------------------------------------------------
def harmonize(state, treatment_config):
    state.save_state(full=True, label="harmonize")
    mask = state.mask
    df = state.df
    new_rows = []

    for f in range(mask.shape[0]):
        plane = mask[f]
        ids = np.unique(plane)
        for obj_id in ids[ids > 0]:
            blob = (plane == obj_id)
            labeled, num = cc_label(blob, return_num=True, connectivity=2)
            if num > 1:
                sizes = [int(np.sum(labeled == i)) for i in range(1, num + 1)]
                biggest = int(np.argmax(sizes)) + 1
                for i in range(1, num + 1):
                    if i == biggest:
                        continue
                    new_id = state.reserve_id()
                    plane[labeled == i] = new_id
                    new_rows.append({
                        COL_FRAME: f, COL_TRACK: new_id, COL_CLABEL: new_id,
                        COL_PARENT: -1, COL_OUTCOME: OUTCOME_UNKNOWN,
                        COL_TREATMENT: _treatment_label(f, treatment_config),
                    })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Force mask value to match track_id where the continuous label differs.
    for f in range(mask.shape[0]):
        df_f = df[df[COL_FRAME] == f]
        for _, row in df_f.iterrows():
            c_label = row.get(COL_CLABEL, row[COL_TRACK])
            t_label = row[COL_TRACK]
            if pd.isna(c_label) or pd.isna(t_label):
                continue
            c_label, t_label = int(c_label), int(t_label)
            if c_label != t_label:
                mask[f][mask[f] == c_label] = t_label

    df[COL_CLABEL] = df[COL_TRACK]
    state.df = df
    msg = (f"Harmonization done: {len(new_rows)} ghosts split."
           if new_rows else "Colors harmonized (no ghosts found).")
    return OpResult(True, msg)


# ---------------------------------------------------------------------------
# Rescue orphan masks
# ---------------------------------------------------------------------------
def rescue_orphans(state, treatment_config):
    state.save_state(full=True, label="rescue orphans")
    mask = state.mask
    df = state.df
    rescued = 0
    new_rows = []
    for f in range(mask.shape[0]):
        plane = mask[f]
        cents = analysis.centroids_for_frame(plane)
        ids_csv = set(df[df[COL_FRAME] == f][COL_TRACK].tolist())
        for cid, (x, y) in cents.items():
            if cid not in ids_csv:
                nid = state.reserve_id()
                plane[plane == cid] = nid
                new_rows.append({
                    COL_FRAME: f, COL_TRACK: nid, COL_X: x, COL_Y: y,
                    COL_CLABEL: nid, COL_OUTCOME: "", COL_PARENT: -1,
                    COL_TREATMENT: _treatment_label(f, treatment_config),
                })
                rescued += 1
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    state.df = df
    return OpResult(True, f"{rescued} orphan masks rescued.")


# ---------------------------------------------------------------------------
# Cut post-mitosis ghosts
# ---------------------------------------------------------------------------
def cut_ghosts(state):
    state.save_state(full=True, label="cut ghosts")
    df = state.df
    mask = state.mask
    mothers = df[df[COL_PARENT] > 0][COL_PARENT].unique()
    ghosts_fixed = 0
    for mother_id in mothers:
        division_frame = df[df[COL_PARENT] == mother_id][COL_FRAME].min()
        ghost = (df[COL_TRACK] == mother_id) & (df[COL_FRAME] >= division_frame)
        if ghost.any():
            new_id = state.reserve_id()
            for f in df.loc[ghost, COL_FRAME].unique():
                plane = mask[int(f)]
                plane[plane == mother_id] = new_id
            df.loc[ghost, COL_OUTCOME] = ""
            df.loc[ghost, COL_PARENT] = -1
            df.loc[ghost, [COL_TRACK, COL_CLABEL]] = new_id
            ghosts_fixed += 1
    state.df = df
    if ghosts_fixed:
        return OpResult(True, f"{ghosts_fixed} ghost mothers split into new IDs.")
    return OpResult(True, "All clean: no mother existing after dividing.")


# ---------------------------------------------------------------------------
# Re-sequence lineage tree (uses unified reserved ranges)
# ---------------------------------------------------------------------------
def resequence_tree(state):
    state.save_state(full=True, label="re-sequence tree")
    df = state.df
    mask = state.mask
    groups = lineage.classify_lineage(df)
    children_of = groups.children_of
    roots = groups.roots
    if not roots:
        return OpResult(False, "No lineage or flagged cell found to organize.")

    new_id_map = {}

    def build_ids(node, base_id):
        new_id_map[node] = base_id
        for idx, child in enumerate(children_of.get(node, []), start=1):
            build_ids(child, int(str(base_id) + str(idx)))

    ordered_roots = (df[df[COL_TRACK].isin(roots)]
                     .groupby(COL_TRACK)[COL_FRAME].min().sort_values().index)
    family_counter = 1
    for r in ordered_roots:
        build_ids(int(r), family_counter)
        family_counter += 1

    # Evict unrelated cells occupying target IDs into the quarantine range.
    target_ids = set(new_id_map.values())
    existing_ids = set(df[COL_TRACK].unique())
    intruders = (existing_ids - set(new_id_map.keys())) & target_ids
    if intruders:
        for old in intruders:
            safe = state.reserve_quarantine_id()
            mask[mask == old] = safe
            df.loc[df[COL_TRACK] == old, [COL_TRACK, COL_CLABEL]] = safe
            df.loc[df[COL_PARENT] == old, COL_PARENT] = safe

    # Apply new IDs through a transient offset to avoid clashes.
    offset = config.RESEQ_OFFSET
    for old_id, new_id in new_id_map.items():
        mask[mask == old_id] = new_id + offset
    for new_id in new_id_map.values():
        mask[mask == new_id + offset] = new_id

    total_map = dict(new_id_map)
    df[COL_PARENT] = df[COL_PARENT].map(lambda x: total_map.get(int(x), x) if x > 0 else x)
    df[COL_TRACK] = df[COL_TRACK].map(lambda x: total_map.get(int(x), x))
    df[COL_CLABEL] = df[COL_TRACK]
    state.df = df
    return OpResult(True, "Lineage tree re-sequenced; families numbered.")


# ---------------------------------------------------------------------------
# Apply an approved relink suggestion
# ---------------------------------------------------------------------------
def apply_relink(state, suggestion):
    """Merge the candidate track into the gapped track from the missing frame on."""
    state.save_state(full=True, label="apply relink")
    df = state.df
    tid = suggestion.track_id
    cand = suggestion.candidate_id
    state.mask[state.mask == cand] = tid
    df.loc[df[COL_TRACK] == cand, [COL_TRACK, COL_CLABEL]] = tid
    df.loc[df[COL_PARENT] == cand, COL_PARENT] = tid
    state.df = df.drop_duplicates(subset=[COL_FRAME, COL_TRACK], keep="first")
    return OpResult(True, f"Reconnected {cand} -> {tid} across the gap.")


# ---------------------------------------------------------------------------
# Auto-tracking with conflict resolution
# ---------------------------------------------------------------------------
def autotrack(state, a, treatment_config):
    """Consolidate track *a* across the movie: resolve conflicts, assimilate
    overlaps, then fill gaps and recompute centroids.

    Three phases:
      1. Where the mask of A splits into multiple connected components in a
         frame, try to identify which island belongs to A and which to another
         track B (by the recorded centroids) and auto-swap them; if that is not
         possible, keep only the largest component and drop the rest.
         Then mark any other track whose centroid falls on an A pixel as
         "absorbed".
      2. Merge every absorbed track into A across the whole movie.
      3. Fill missing per-frame rows for A and recompute its centroid.
    """
    if a == 0:
        return OpResult(False, "Select ID [A].")
    state.save_state(full=True, label="auto-track")
    mask = state.mask
    df = state.df
    absorbed_ids = set()

    # Phase 1
    for f in range(mask.shape[0]):
        plane = mask[f]
        if a not in plane:
            continue
        blob_a = (plane == a)
        labeled_a, num = cc_label(blob_a, return_num=True, connectivity=2)

        if num > 1:
            df_frame = df[df[COL_FRAME] == f]
            island_a = island_b = id_b = None
            for i in range(1, num + 1):
                island = (labeled_a == i)
                for _, row in df_frame.iterrows():
                    tid = int(row[COL_TRACK])
                    if pd.isna(row[COL_X]) or pd.isna(row[COL_Y]):
                        continue
                    x, y = int(row[COL_X]), int(row[COL_Y])
                    if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1] and island[y, x]:
                        if tid == a:
                            island_a = i
                        elif tid > 0:
                            island_b = i
                            id_b = tid
            if island_a is not None and island_b is not None and id_b is not None:
                plane[labeled_a == island_a] = id_b
                idx_a = df[(df[COL_FRAME] == f) & (df[COL_TRACK] == a)].index
                idx_b = df[(df[COL_FRAME] == f) & (df[COL_TRACK] == id_b)].index
                df.loc[idx_a, COL_PARENT] = -1
                df.loc[idx_b, COL_PARENT] = -1
                df.loc[idx_a, COL_OUTCOME] = ""
                df.loc[idx_b, COL_OUTCOME] = ""
                df.loc[idx_a, [COL_TRACK, COL_CLABEL]] = id_b
                df.loc[idx_b, [COL_TRACK, COL_CLABEL]] = a
                print(f"Auto local swap (frame {f}): {a} -> {id_b}, painted -> {a}.")
            else:
                sizes = [int(np.sum(labeled_a == i)) for i in range(1, num + 1)]
                biggest = int(np.argmax(sizes)) + 1
                for i in range(1, num + 1):
                    if i != biggest:
                        plane[labeled_a == i] = 0

        df_frame = df[df[COL_FRAME] == f]
        for _, row in df_frame.iterrows():
            tid = int(row[COL_TRACK])
            if tid != a and tid > 0:
                if pd.isna(row[COL_X]) or pd.isna(row[COL_Y]):
                    continue
                x, y = int(row[COL_X]), int(row[COL_Y])
                if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1] and plane[y, x] == a:
                    absorbed_ids.add(tid)

    # Phase 2: merge absorbed tracks into A.
    for old_id in absorbed_ids:
        mask[mask == old_id] = a
        # Daughters of the absorbed track must follow the rename, else their
        # parent_id points to an id that no longer exists (orphaned lineage).
        _repoint_children(df, old_id, a)
        df.loc[df[COL_TRACK] == old_id, COL_PARENT] = -1
        df.loc[df[COL_TRACK] == old_id, COL_OUTCOME] = ""
        df.loc[df[COL_TRACK] == old_id, [COL_TRACK, COL_CLABEL]] = a

    df = df.drop_duplicates(subset=[COL_FRAME, COL_TRACK], keep="first")

    # Phase 3: fill gaps and recompute centroid (vectorized centroid helper).
    new_rows = []
    for f in range(mask.shape[0]):
        plane = mask[f]
        if a not in plane:
            continue
        c = analysis.centroid_of_label(plane, a)
        if c is None:
            continue
        x, y = c
        if a not in df[df[COL_FRAME] == f][COL_TRACK].tolist():
            new_rows.append({
                COL_FRAME: f, COL_TRACK: a, COL_X: x, COL_Y: y,
                COL_CLABEL: a, COL_OUTCOME: "", COL_PARENT: -1,
                COL_TREATMENT: _treatment_label(f, treatment_config),
            })
        else:
            df.loc[(df[COL_FRAME] == f) & (df[COL_TRACK] == a), [COL_X, COL_Y]] = [x, y]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    state.df = df
    n_abs = len(absorbed_ids)
    return OpResult(True, f"Auto-tracking done for ID {a} "
                          f"({n_abs} track(s) absorbed).")
