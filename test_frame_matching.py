"""Pairing a raw channel that has MORE frames on disk than the mask covers.

The raw acquisition keeps every frame; the segmentation runs on the subset that
survived focus checks and interruptions. So the raw folder legitimately holds
more files than the mask has planes, and the two must be paired by FILENAME.
Pairing by position instead looks fine until the first dropped frame and is
wrong for every frame after it -- silently, because the shapes still line up.
"""
import os

import numpy as np

from curator import channels as chmod, io_adapters


def _names(hours):
    out = []
    for h in hours:
        d, rem = divmod(h, 24.0)
        hh, mm = divmod(rem, 1.0)
        out.append(f"VID504_C4_4_{int(d):02d}d{int(hh):02d}h{int(round(mm*60)):02d}m.tif")
    return out


def test_subset_is_picked_in_reference_order():
    raw = _names([0.0, 0.5, 1.0, 1.5, 2.0])       # everything acquired
    ref = _names([0.0, 1.0, 2.0])                  # what got segmented
    picked, missing = chmod.match_frames_by_name(raw, ref)
    assert missing == []
    assert [os.path.basename(p) for p in picked] == ref


def test_positional_pairing_would_have_been_wrong():
    """The failure this exists to prevent, stated as an assertion."""
    raw = _names([0.0, 0.5, 1.0, 1.5, 2.0])
    ref = _names([0.0, 1.0, 2.0])
    positional = raw[:len(ref)]                    # "just take the first N"
    picked, _ = chmod.match_frames_by_name(raw, ref)
    assert positional != [os.path.basename(p) for p in picked]
    assert positional[1] != ref[1]                 # already wrong at frame 1


def test_missing_reference_frame_is_reported_not_silently_dropped():
    raw = _names([0.0, 0.5])
    ref = _names([0.0, 0.5, 1.0])
    picked, missing = chmod.match_frames_by_name(raw, ref)
    assert len(picked) == 2
    assert missing == [_names([1.0])[0]]


def test_full_paths_are_matched_on_basename():
    raw = [os.path.join("E:", "data", "red", n) for n in _names([0.0, 0.5, 1.0])]
    ref = _names([0.0, 1.0])
    picked, missing = chmod.match_frames_by_name(raw, ref)
    assert missing == []
    assert all(os.sep in p or ":" in p for p in picked)   # originals, not basenames


def test_stack_records_its_frame_names(tmp_path):
    import tifffile
    paths = []
    for i, n in enumerate(_names([0.0, 0.5, 1.0])):
        p = tmp_path / n
        tifffile.imwrite(str(p), np.full((4, 4), i, dtype=np.uint16))
        paths.append(str(p))
    stack_path = str(tmp_path / "out_stack.tif")
    stack = io_adapters.build_stack_from_files(paths, stack_path)
    assert stack.shape == (3, 4, 4)
    assert io_adapters.stack_frame_names(stack_path) == _names([0.0, 0.5, 1.0])


def test_no_sidecar_returns_empty_rather_than_raising(tmp_path):
    assert io_adapters.stack_frame_names(str(tmp_path / "nope.tif")) == []


if __name__ == "__main__":
    import tempfile, pathlib
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"ok  {name}")
    print("all checks passed")
