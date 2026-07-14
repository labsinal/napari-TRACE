"""
Minimal runnable checks for the trickiest curation invariants.

No pytest fixtures, no framework: plain asserts. Run either with
``pytest test_curation_ops.py`` or directly ``python test_curation_ops.py``.
These are the guards a future refactor is most likely to break silently, not
an exhaustive suite: one runnable check per invariant, not a suite per function.
Kept Qt-free by touching only curation_ops / lineage / state (pure pandas).
"""

import numpy as np
import pandas as pd

from curator.state import CuratorState, IDPool
from curator import curation_ops as ops
from curator import lineage
from curator.config import (
    COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME, COL_PARENT, COL_CLABEL,
    COL_TREATMENT, OUTCOME_MITOSIS, TREAT_CONTROL,
)


def _row(track, frame, parent=-1, outcome="", x=1.0, y=1.0):
    return {COL_TRACK: track, COL_FRAME: frame, COL_X: x, COL_Y: y,
            COL_OUTCOME: outcome, COL_PARENT: parent, COL_CLABEL: track,
            COL_TREATMENT: TREAT_CONTROL}


def _state(rows, n_frames, labels_per_frame=None):
    """Build a CuratorState from row dicts and an optional {frame: [ids]} map."""
    df = pd.DataFrame(rows)
    mask = np.zeros((n_frames, 4, 4), dtype=np.int32)
    if labels_per_frame:
        for f, ids in labels_per_frame.items():
            for k, cid in enumerate(ids):
                mask[f, 0, k] = cid  # one distinct pixel per id, enough for relabel
    return CuratorState(df, mask)


def test_idpool_never_repeats():
    pool = IDPool([1, 2, 5])
    got = [pool.new_id() for _ in range(20)]
    assert len(set(got)) == len(got), "pool handed out a duplicate id"
    assert not (set(got) & {1, 2, 5}), "pool reused an occupied id"


def test_predict_path_spline_falls_back_to_linear_on_long_gap():
    frames = np.arange(0, 40)
    xs = frames * 2.0
    ys = frames * 0.0
    q = list(range(15, 35))  # gap length 20 > SPLINE_MAX_GAP_FOR_SPLINE (12)
    px, py = lineage.predict_path(frames, xs, ys, q, method="spline")
    lx, ly = lineage.predict_path(frames, xs, ys, q, method="linear")
    assert np.allclose(px, lx) and np.allclose(py, ly), \
        "spline should fall back to the linear path for a long gap"


def test_merge_mother_into_own_daughter_dissolves_division():
    # Mother 10 divides into daughters 11 and 12; 10 flagged Mitosis.
    rows = [
        _row(10, 0, outcome=OUTCOME_MITOSIS),
        _row(11, 1, parent=10),
        _row(12, 1, parent=10),
    ]
    state = _state(rows, n_frames=2,
                   labels_per_frame={0: [10], 1: [11, 12]})
    # Merge the daughter 11 back into its mother 10: the recorded division was
    # a segmentation artefact and must be dissolved.
    res = ops.merge(state, a=11, b=10)
    assert res.ok
    df = state.df
    assert 11 not in set(df[COL_TRACK]), "merged-away track should be gone"
    other = df.loc[df[COL_TRACK] == 12, COL_PARENT].iloc[0]
    assert int(other) == -1, "the other daughter must be detached"
    oc = set(df.loc[df[COL_TRACK] == 10, COL_OUTCOME].astype(str))
    assert OUTCOME_MITOSIS not in oc, "mother's Mitosis flag must be cleared"


def test_swap_future_only_repoints_daughters_born_after_the_swap():
    # Tracks 1 and 2 both span the movie; daughter 3 is born at/after the swap
    # frame, daughter 4 before it.
    rows = [_row(1, f) for f in range(10)] + [_row(2, f) for f in range(10)]
    rows += [_row(3, f, parent=1) for f in range(6, 10)]   # born frame 6
    rows += [_row(4, f, parent=1) for f in range(1, 10)]   # born frame 1
    state = _state(rows, n_frames=10)
    res = ops.swap_future(state, a=1, b=2, frame=3)
    assert res.ok
    df = state.df
    p3 = int(df.loc[df[COL_TRACK] == 3, COL_PARENT].iloc[0])
    p4 = int(df.loc[df[COL_TRACK] == 4, COL_PARENT].iloc[0])
    assert p3 == 2, "daughter born after the swap should follow to track 2"
    assert p4 == 1, "daughter born before the swap should keep its mother"


def test_setup_working_directory_copies_files_and_subfolders():
    """Bug A: the working copy must include frame-folder subdirs, not just top files."""
    import os, tempfile, shutil
    import numpy as np
    import tifffile
    from curator import data_io
    src = tempfile.mkdtemp()
    try:
        # a top-level file (the tracking CSV) ...
        with open(os.path.join(src, "tracks.csv"), "w") as fh:
            fh.write("track_id,frame,pos_x,pos_y\n1,0,5,5\n")
        # ... and a frame-folder subdirectory (a channel / mask as per-frame TIFs)
        sub = os.path.join(src, "mask_frames")
        os.makedirs(sub)
        for i in range(2):
            tifffile.imwrite(os.path.join(sub, f"f{i}.tif"),
                             np.zeros((4, 4), dtype=np.uint16))

        work_dir, is_first = data_io.setup_working_directory(src)
        assert is_first
        assert os.path.isfile(os.path.join(work_dir, "tracks.csv"))
        copied_sub = os.path.join(work_dir, "mask_frames")
        assert os.path.isdir(copied_sub), "frame-folder subdir was not copied"
        assert len([f for f in os.listdir(copied_sub) if f.endswith(".tif")]) == 2
    finally:
        shutil.rmtree(src, ignore_errors=True)
        cur = src + "_curated"
        shutil.rmtree(cur, ignore_errors=True)


def test_meta_channels_roundtrip():
    """Added channels persist in the meta with their tags, preserving other keys."""
    import os, tempfile, shutil
    from curator import data_io
    wd = tempfile.mkdtemp()
    try:
        data_io.write_meta(wd, {"source_folder": "/src", "version": 6})
        data_io.add_channel_to_meta(wd, "ERK-KTR", "erk.tif", color="green", measure=True)
        data_io.add_channel_to_meta(wd, "53BP1", "dsb.tif", color="red", measure=False)
        chans = data_io.meta_channels(wd)
        assert len(chans) == 2
        erk = next(c for c in chans if c["name"] == "ERK-KTR")
        assert erk["file"] == "erk.tif" and erk["color"] == "green" and erk["measure"] is True
        # other metadata preserved; re-adding same file updates in place (no dup)
        assert data_io.read_meta(wd)["source_folder"] == "/src"
        data_io.add_channel_to_meta(wd, "ERK-KTR", "erk.tif", color="cyan", measure=False)
        chans = data_io.meta_channels(wd)
        assert len(chans) == 2
        assert next(c for c in chans if c["name"] == "ERK-KTR")["color"] == "cyan"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def test_hierarchical_labels_from_parent_id():
    # family root 5 -> daughters 8 (born f2) and 9 (born f3); 8 -> 20 (born f5).
    # a second, later family root 30 -> daughter 31.
    rows = [_row(5, f) for f in range(6)]
    rows += [_row(8, f, parent=5) for f in range(2, 6)]
    rows += [_row(9, f, parent=5) for f in range(3, 6)]
    rows += [_row(20, f, parent=8) for f in range(5, 6)]
    rows += [_row(30, f) for f in range(4, 8)]
    rows += [_row(31, f, parent=30) for f in range(6, 8)]
    labels = lineage.hierarchical_labels(pd.DataFrame(rows))
    assert labels[5] == "1"
    assert labels[8] == "1.1" and labels[9] == "1.2"   # ordered by birth frame
    assert labels[20] == "1.1.1"
    assert labels[30] == "2" and labels[31] == "2.1"   # later root -> family 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
