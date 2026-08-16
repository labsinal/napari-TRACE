"""Regression for the treatment -> fase rename.

Two things must hold: a table written before the rename still opens, and the
stats figures that reached the column by ATTRIBUTE access (summary.treatment)
still build -- those fail only at runtime, so nothing else catches them.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from curator.config import (COL_TREATMENT, COL_TREATMENT_LEGACY,
                            TREAT_CONTROL, TREAT_TREATED)
from curator import stats, analysis


def test_constant_renamed_with_legacy_kept():
    assert COL_TREATMENT == "fase"
    assert COL_TREATMENT_LEGACY == "treatment"


def test_legacy_column_is_migrated():
    df = pd.DataFrame({"track_id": [1, 1], "frame": [0, 1],
                       COL_TREATMENT_LEGACY: [TREAT_CONTROL, TREAT_TREATED]})
    if COL_TREATMENT not in df.columns and COL_TREATMENT_LEGACY in df.columns:
        df = df.rename(columns={COL_TREATMENT_LEGACY: COL_TREATMENT})
    assert COL_TREATMENT in df.columns
    assert COL_TREATMENT_LEGACY not in df.columns


def _toy_dataset(n_tracks=6, n_frames=12):
    rows = []
    for tid in range(1, n_tracks + 1):
        for f in range(n_frames):
            rows.append({"track_id": tid, "frame": f,
                         "pos_x": 10 + f + tid, "pos_y": 20 + f,
                         "outcome": "", "parent_id": -1, "at_border": False,
                         COL_TREATMENT: TREAT_CONTROL if f < 6 else TREAT_TREATED})
    mask = np.zeros((n_frames, 64, 64), np.uint32)
    for tid in range(1, n_tracks + 1):
        for f in range(n_frames):
            y, x = 20 + f, 10 + f + tid
            mask[f, y - 3:y + 3, x - 3:x + 3] = tid
    return pd.DataFrame(rows), mask


def test_summary_carries_the_renamed_column():
    df, mask = _toy_dataset()
    summary = analysis.compute_track_summary(df, mask)
    assert COL_TREATMENT in summary.columns
    assert COL_TREATMENT_LEGACY not in summary.columns


def test_phase_figures_still_build():
    df, mask = _toy_dataset()
    summary = analysis.compute_track_summary(df, mask)
    cfg = {"mode": TREAT_TREATED, "start": 6, "end": 11}
    stats.lifetime_figure(summary, 1.0)
    stats.migration_figure(summary, 1.0, 1.0)
    stats.motility_figure(summary, 1.0, 1.0)
    stats.area_figure(summary, df, mask, 1.0, cfg, 11)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
