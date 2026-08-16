"""Real elapsed time vs stack index, and what depends on telling them apart.

A stack index is only a clock while every acquired frame is present. Drop the
out-of-focus ones and the stack closes the gap silently, so a step that really
took 2.5 h still looks like one interval -- and speed comes out several times
too high at exactly the frames something was wrong enough to discard.
"""
import numpy as np
import pandas as pd

from curator import timeaxis, exports
from curator.config import COL_TRACK, COL_FRAME, COL_X, COL_Y


def _names(hours):
    out = []
    for h in hours:
        d, rem = divmod(h, 24.0)
        hh, mm = divmod(rem, 1.0)
        out.append(f"VID_{int(d):02d}d{int(hh):02d}h{int(round(mm * 60)):02d}m.tif")
    return out


def test_parses_incucyte_style_names():
    hours = [0.0, 0.5, 1.0, 1.5]
    t = timeaxis.build_time_table(_names(hours))
    assert t.attrs["time_source"] == "filenames"
    assert np.allclose(t["tempo_h"], hours)


def test_gap_is_detected_and_span_preserved():
    # 30 min apart, except a 2.5 h hole between the 2nd and 3rd kept frames
    hours = [0.0, 0.5, 3.0, 3.5]
    t = timeaxis.build_time_table(_names(hours))
    assert t["gap"].tolist() == [False, False, True, False]
    assert abs(t.loc[2, "dt_h"] - 2.5) < 1e-9
    assert abs(t["tempo_h"].iloc[-1] - 3.5) < 1e-9


def test_unparseable_names_fall_back_and_say_so():
    t = timeaxis.build_time_table([f"img_{i}.png" for i in range(4)],
                                  frame_interval=0.5)
    assert t.attrs["time_source"] == "assumed_interval"
    assert not t["gap"].any()


def test_partial_match_is_rejected_rather_than_half_applied():
    names = ["a_00d00h00m.tif", "a_00d00h30m.tif", "no_timestamp_here.tif"]
    assert timeaxis.parse_times_from_names(names) is None


def test_non_monotonic_names_are_rejected():
    names = _names([0.0, 1.0, 0.5])
    assert timeaxis.parse_times_from_names(names) is None


def _straight_line_track(frames, step_px=10.0):
    """One cell moving step_px per OBSERVATION (not per hour)."""
    return pd.DataFrame({
        COL_TRACK: [1] * len(frames),
        COL_FRAME: frames,
        COL_X: [10.0 + i * step_px for i in range(len(frames))],
        COL_Y: [20.0] * len(frames),
        "outcome": [""] * len(frames),
        "parent_id": [-1] * len(frames),
    })


def test_speed_across_a_gap_uses_real_elapsed_time():
    hours = [0.0, 0.5, 3.0, 3.5]          # 2.5 h hole before the 3rd frame
    frames = [0, 1, 2, 3]
    df = _straight_line_track(frames)
    tt = timeaxis.build_time_table(_names(hours))

    naive = exports.features_table(df, mask=None, frame_interval=0.5)
    real = exports.features_table(df, mask=None, time_table=tt)

    # Same distance covered on every step: 10 px.
    assert np.allclose(naive["step_disp"].to_numpy()[1:], 10.0)
    assert np.allclose(real["step_disp"].to_numpy()[1:], 10.0)

    # Assuming a uniform interval, the gap step looks like 20 px/h...
    assert abs(naive["speed"].to_numpy()[2] - 20.0) < 1e-9
    # ...but it really took 2.5 h, so it is 4 px/h: a 5x overestimate.
    assert abs(real["speed"].to_numpy()[2] - 4.0) < 1e-9
    assert real["gap_antes"].tolist() == [False, False, True, False]
    assert np.allclose(real["tempo_h"].to_numpy(), hours)


def test_lifetime_is_real_hours_when_a_time_table_is_given():
    hours = [0.0, 0.5, 3.0, 3.5]
    df = _straight_line_track([0, 1, 2, 3])
    tt = timeaxis.build_time_table(_names(hours))
    real = exports.features_table(df, mask=None, time_table=tt)
    life = real["lifetime"].dropna()
    assert len(life) == 1
    assert abs(float(life.iloc[0]) - (3.5 + 0.5)) < 1e-9   # span + one interval


def test_root_id_is_constant_along_a_daughter_track():
    """parent_id lives only on the daughter's first row; root_id must not."""
    df = pd.DataFrame({
        COL_TRACK: [1, 1, 11, 11, 11],
        COL_FRAME: [0, 1, 2, 3, 4],
        COL_X: [1.0, 2.0, 3.0, 4.0, 5.0],
        COL_Y: [1.0, 1.0, 1.0, 1.0, 1.0],
        "outcome": ["", "Mitosis", "", "", ""],
        "parent_id": [-1, -1, 1, -1, -1],      # link recorded at birth only
    })
    out = exports.features_table(df, mask=None)
    roots = out.groupby(COL_TRACK)["root_id"].nunique()
    assert (roots == 1).all()
    assert set(out.loc[out[COL_TRACK] == 11, "root_id"]) == {1}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
