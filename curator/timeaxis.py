"""
Real acquisition time per frame, recovered from frame filenames.

The frame index is a position in a stack, not a time. Those two only agree while
every acquired frame is present: drop the out-of-focus ones, or survive an
interruption of the microscope, and the stack silently closes the gap. Frame 300
is then still "frame 300" but no longer 300 intervals from the start, and every
downstream quantity that divides by an interval -- speed above all -- is wrong by
whatever the gap was, exactly at the frames where something happened worth
dropping a frame for.

This module keeps the two apart. ``build_time_table`` returns one row per frame
carrying both: the stack index (the key everything else joins on) and the real
elapsed time read from the filename, plus the true interval to the previous
frame and a flag marking the frames that sit across a gap.

Timestamp formats are matched by an ordered list of regexes; a dataset whose
names follow none of them falls back to a constant interval, which is the old
behaviour and is stated in the returned table's attrs rather than assumed.

Pure pandas/numpy, no Qt/napari, so it is testable headless.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


# Ordered: the first pattern that matches every name wins. Each entry is
# (name, compiled regex, converter -> hours).
def _h_dhm(m):
    return int(m.group(1)) * 24 + int(m.group(2)) + int(m.group(3)) / 60.0


def _h_hm(m):
    return int(m.group(1)) + int(m.group(2)) / 60.0


def _h_dh(m):
    return int(m.group(1)) * 24 + int(m.group(2))


def _h_minutes(m):
    return int(m.group(1)) / 60.0


def _h_hours(m):
    return float(m.group(1))


TIME_PATTERNS = (
    # 00d12h30m  (IncuCyte and friends)
    ("d_h_m", re.compile(r"(\d+)d(\d+)h(\d+)m"), _h_dhm),
    # 12h30m
    ("h_m", re.compile(r"(?<!\d)(\d+)h(\d+)m"), _h_hm),
    # 02d06h
    ("d_h", re.compile(r"(\d+)d(\d+)h(?!\d)"), _h_dh),
    # t000120min / _120min
    ("minutes", re.compile(r"(\d+)\s*min(?![a-z])", re.I), _h_minutes),
    # t0012h / _12hr
    ("hours", re.compile(r"(\d+)\s*h(?:r|rs|ours?)?(?![a-z0-9])", re.I), _h_hours),
)


def parse_times_from_names(names, pattern=None):
    """Elapsed hours for each filename, or None when no pattern fits them all.

    ``pattern`` overrides the built-in list: either a key from
    :data:`TIME_PATTERNS` or a regex whose groups are (days, hours, minutes),
    (hours, minutes) or (hours). Matching is required for EVERY name -- a
    pattern that fits most of them would produce a time axis that is wrong only
    in places, which is worse than not having one.
    """
    stems = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
    if not stems:
        return None

    candidates = list(TIME_PATTERNS)
    if pattern:
        named = [p for p in TIME_PATTERNS if p[0] == pattern]
        if named:
            candidates = named
        else:
            rx = re.compile(pattern)
            n_groups = rx.groups
            conv = {3: _h_dhm, 2: _h_hm, 1: _h_hours}.get(n_groups)
            if conv is None:
                return None
            candidates = [("custom", rx, conv)]

    for _key, rx, conv in candidates:
        hits = [rx.search(s) for s in stems]
        if all(h is not None for h in hits):
            hours = np.array([conv(h) for h in hits], dtype=float)
            if np.all(np.diff(hours) > 0):     # must be strictly increasing
                return hours
    return None


def build_time_table(names=None, n_frames=None, frame_interval=0.5,
                     pattern=None, gap_tol=1e-6):
    """One row per frame: stack index, real elapsed time, true step, gap flag.

    Columns: ``frame`` (stack index), ``filename``, ``tempo_h`` (elapsed hours
    from the first frame), ``dt_h`` (hours since the previous frame; NaN on the
    first) and ``gap`` (True when ``dt_h`` exceeds the modal interval).

    With parseable ``names`` the times are real. Otherwise every frame is
    ``frame_interval`` apart and no gap is ever flagged -- the assumption the
    tool made implicitly before, now explicit and recorded in
    ``table.attrs["time_source"]`` as "filenames" or "assumed_interval".
    """
    if names is None and n_frames is None:
        raise ValueError("give either names or n_frames")
    names = list(names) if names is not None else [""] * int(n_frames)
    n = len(names)

    hours = parse_times_from_names(names, pattern=pattern) if any(names) else None
    source = "filenames"
    if hours is None:
        hours = np.arange(n, dtype=float) * float(frame_interval)
        source = "assumed_interval"

    hours = hours - hours[0]
    dt = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(hours)

    steps = dt[1:] if n > 1 else np.array([])
    if steps.size:
        vals, counts = np.unique(np.round(steps, 6), return_counts=True)
        modal = float(vals[np.argmax(counts)])
    else:
        modal = float(frame_interval)

    table = pd.DataFrame({
        "frame": np.arange(n, dtype=int),
        "filename": [os.path.basename(str(x)) for x in names],
        "tempo_h": hours,
        "dt_h": dt,
        "gap": np.where(np.isnan(dt), False, dt > modal + gap_tol),
    })
    table.attrs["time_source"] = source
    table.attrs["modal_interval_h"] = modal
    return table


def time_table_from_folder(folder, ext="tif", **kwargs):
    """:func:`build_time_table` over the frame files of a folder (natural order)."""
    from . import io_adapters
    try:
        names = io_adapters.list_frame_files(folder, ext)
    except AttributeError:
        import glob
        names = sorted(glob.glob(os.path.join(folder, f"*.{ext.lstrip('.')}")),
                       key=_natural_key)
    return build_time_table(names=names, **kwargs)


def _natural_key(path):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str(path))]


def summarize(table) -> str:
    """One-line human summary, for the UI banner and the console."""
    n = len(table)
    if n == 0:
        return "no frames"
    span = float(table["tempo_h"].iloc[-1])
    naive = (n - 1) * float(table.attrs.get("modal_interval_h", 0.5))
    n_gap = int(table["gap"].sum())
    src = table.attrs.get("time_source", "?")
    msg = (f"{n} frames, {span:.2f} h real "
           f"(uniform-interval assumption would say {naive:.2f} h), "
           f"{n_gap} gap frame(s), time from {src}")
    if abs(span - naive) > 0.05:
        msg += f"  --  DRIFT {span - naive:+.2f} h"
    return msg


def _selfcheck():
    """Validate parsing, gap detection and the fallback on synthetic names."""
    names = [f"VID_{d:02d}d{h:02d}h{m:02d}m.tif"
             for d in range(1) for h in range(3) for m in (0, 30)]
    t = build_time_table(names)
    assert t.attrs["time_source"] == "filenames"
    assert not t["gap"].any()
    assert abs(t["tempo_h"].iloc[-1] - 2.5) < 1e-9

    # drop two frames in the middle: the stack closes up, the clock must not
    holed = names[:2] + names[4:]
    th = build_time_table(holed)
    assert th["gap"].sum() == 1
    assert abs(th.loc[2, "dt_h"] - 1.5) < 1e-9        # 3 intervals, not 1
    assert abs(th["tempo_h"].iloc[-1] - 2.5) < 1e-9   # real span is unchanged

    # unparseable names fall back and say so
    tf = build_time_table([f"img_{i}.tif" for i in range(4)], frame_interval=0.5)
    assert tf.attrs["time_source"] == "assumed_interval"
    assert abs(tf["tempo_h"].iloc[-1] - 1.5) < 1e-9
    assert not tf["gap"].any()

    tn = build_time_table(n_frames=3, frame_interval=0.25)
    assert abs(tn["tempo_h"].iloc[-1] - 0.5) < 1e-9
    print("timeaxis selfcheck passed")


if __name__ == "__main__":
    _selfcheck()
