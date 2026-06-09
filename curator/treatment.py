"""
Treatment-phase logic.

The treatment pattern is temporal and shared by every cell in the dish: frames
before the start are control, frames inside the window are treated, and frames
after the end (if any) are washout.
"""

from __future__ import annotations

import numpy as np

from .config import COL_FRAME, COL_TREATMENT, TREAT_CONTROL, TREAT_TREATED, TREAT_WASHOUT


def default_config():
    return {"mode": TREAT_CONTROL, "start": 0, "end": -1}


def treatment_for_frame(frame, config):
    """Label for a single frame given the configuration dict."""
    if config["mode"] == TREAT_CONTROL:
        return TREAT_CONTROL
    if frame < config["start"]:
        return TREAT_CONTROL
    end = config.get("end", -1)
    if end is not None and end >= 0 and frame > end:
        return TREAT_WASHOUT
    return TREAT_TREATED


def apply_treatment(df, config):
    """Assign the treatment label to every row based only on its frame."""
    if config["mode"] == TREAT_CONTROL:
        df[COL_TREATMENT] = TREAT_CONTROL
        return df
    f = df[COL_FRAME].fillna(-1).astype(int).values
    labels = np.where(f < config["start"], TREAT_CONTROL, TREAT_TREATED).astype(object)
    end = config.get("end", -1)
    if end is not None and end >= 0:
        labels[f > end] = TREAT_WASHOUT
    df[COL_TREATMENT] = labels
    return df
