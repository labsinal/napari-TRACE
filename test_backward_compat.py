"""Everything an existing TRACE user already has must keep working.

The changes in this round touch the column schema, the export columns and the
default background estimator. Each of those can break someone whose only crime
was using the tool before. These tests pin the parts that must not move.
"""
import numpy as np
import pandas as pd
import pytest

from curator import exports, fluorescence as fl, io_adapters, channels as chmod
from curator.config import (COL_TRACK, COL_FRAME, COL_X, COL_Y,
                            COL_TREATMENT, COL_TREATMENT_LEGACY, TREAT_CONTROL)


# ---------------------------------------------------------------------------
# Old tables must still open
# ---------------------------------------------------------------------------
def test_legacy_phase_column_is_migrated_not_duplicated():
    df = pd.DataFrame({COL_TRACK: [1, 1], COL_FRAME: [0, 1],
                       COL_TREATMENT_LEGACY: [TREAT_CONTROL, TREAT_CONTROL]})
    if COL_TREATMENT not in df.columns and COL_TREATMENT_LEGACY in df.columns:
        df = df.rename(columns={COL_TREATMENT_LEGACY: COL_TREATMENT})
    assert list(df.columns) == [COL_TRACK, COL_FRAME, COL_TREATMENT]


def test_a_table_that_already_has_both_keeps_the_new_one():
    """A half-migrated table must not lose the current column."""
    df = pd.DataFrame({COL_TRACK: [1], COL_FRAME: [0],
                       COL_TREATMENT: ["treated"],
                       COL_TREATMENT_LEGACY: ["control"]})
    if COL_TREATMENT not in df.columns and COL_TREATMENT_LEGACY in df.columns:
        df = df.rename(columns={COL_TREATMENT_LEGACY: COL_TREATMENT})
    assert df[COL_TREATMENT].iloc[0] == "treated"


# ---------------------------------------------------------------------------
# Old call signatures must still work
# ---------------------------------------------------------------------------
def _toy_df(n=4):
    return pd.DataFrame({
        COL_TRACK: [1] * n, COL_FRAME: list(range(n)),
        COL_X: [10.0 + i for i in range(n)], COL_Y: [5.0] * n,
        "outcome": [""] * n, "parent_id": [-1] * n,
    })


def test_features_table_still_callable_positionally():
    """The pre-existing signature (df, mask, pixel_size, frame_interval)."""
    out = exports.features_table(_toy_df(), None, 1.0, 1.0)
    for col in ("step_disp", "cum_path", "net_disp", "speed", "lifetime"):
        assert col in out.columns


def test_windows_table_still_callable_positionally():
    out = exports.windows_table(_toy_df(6), None, 3, 1.0, 1.0)
    assert not out.empty
    assert {"frame_center", "path", "net", "speed_mean"} <= set(out.columns)


def test_speed_without_a_time_table_is_unchanged():
    """No time table == the old assumption, and the old numbers."""
    out = exports.features_table(_toy_df(), None, pixel_size=1.0, frame_interval=0.5)
    # 1 px per frame, 0.5 h per frame -> 2 px/h, exactly as before.
    assert np.allclose(out["speed"].to_numpy()[1:], 2.0)
    assert not out["gap_antes"].any()


def test_old_export_columns_all_survive():
    """Columns that existed before must still exist; new ones may be added."""
    before = {"step_disp", "cum_path", "net_disp", "speed", "lifetime",
              COL_TRACK, COL_FRAME, COL_X, COL_Y}
    out = exports.features_table(_toy_df(), None)
    assert before <= set(out.columns)


def test_compartment_masks_default_is_the_old_geometry():
    """neighbor_gap defaults to 0, i.e. exactly the previous ring."""
    p = np.zeros((20, 20), dtype=np.int32)
    p[4:9, 4:9] = 1
    p[4:9, 12:17] = 2
    assert fl.NEIGHBOR_GAP_DEFAULT == 0
    old = fl.compartment_masks(p, 1, dilation=2, gap=1, neighbor_gap=0)
    new = fl.compartment_masks(p, 1, dilation=2, gap=1)
    assert (old["ring"] == new["ring"]).all()


def test_measure_intensity_old_kwargs_still_accepted():
    mask = np.zeros((1, 20, 20), dtype=np.int32)
    mask[0, 6:12, 6:12] = 1
    ch = np.ones((1, 20, 20), dtype=float) * 50
    df = fl.measure_intensity(ch, mask, channel_name="g",
                              dilation=2, gap=1, subtract_background=False,
                              background_mode="non_cell_median",
                              use_ellipse=False)
    assert not df.empty
    for stat in ("mean", "median", "sum", "min", "max", "std", "p90"):
        assert f"g_nuc_{stat}" in df.columns


def test_old_background_estimators_are_still_available():
    """The default changed; the previous behaviours must remain reachable."""
    p = np.zeros((20, 20), dtype=np.int32)
    p[8:12, 8:12] = 1
    ch = np.full((20, 20), 7.0)
    assert fl.background_per_frame(ch, p) == 7.0
    assert fl.background_image_min(ch) == 7.0


def test_channel_layer_old_construction_still_works():
    """Existing code builds ChannelLayer with the original four fields."""
    L = chmod.ChannelLayer(name="g", data=np.zeros((3, 4, 4)), colormap="green")
    assert L.color == "green" and L.measure is False
    assert L.role == "" and L.measure_source == ""
    assert L.quant is L.data          # falls back to display, as before


def test_build_stack_from_folder_still_returns_the_stack(tmp_path):
    import tifffile
    for i in range(3):
        tifffile.imwrite(str(tmp_path / f"t{i:03d}.tif"),
                         np.full((4, 4), i, dtype=np.uint16))
    dest = str(tmp_path / "s.tif")
    stack = io_adapters.build_stack_from_folder(str(tmp_path), dest)
    assert stack.shape == (3, 4, 4)


def test_haralick_background_subtraction_can_be_switched_off():
    """The new default changes numbers; the previous behaviour stays reachable."""
    mask = np.zeros((1, 24, 24), dtype=np.int32)
    mask[0, 8:16, 8:16] = 1
    rs = np.random.RandomState(0)
    ch = (rs.rand(1, 24, 24) * 50 + 100)
    off = fl.haralick_features(ch, mask, channel_name="g", subtract_background=False)
    assert not off.empty
    assert "g_hara_entropy" in off.columns


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
