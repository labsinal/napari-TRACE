import numpy as np
from curator.channels import ChannelLayer
from curator import fluorescence as fl

def test_channel_layer_has_color_and_measure_defaults():
    L = ChannelLayer(name="green", data=np.zeros((3, 4, 4)), colormap="green")
    assert L.color == "green"      # defaults to the colormap
    assert L.measure is False
    L2 = ChannelLayer(name="erk", data=np.zeros((3, 4, 4)), colormap="green",
                      color="green", measure=True)
    assert L2.measure is True


def _two_nuclei_plane():
    # 20x20 with nucleus 1 (rows 4-8, cols 4-8) and nucleus 2 (rows 4-8, cols 12-16)
    p = np.zeros((20, 20), dtype=np.int32)
    p[4:9, 4:9] = 1
    p[4:9, 12:17] = 2
    return p

def test_ring_excludes_neighbor_nucleus():
    p = _two_nuclei_plane()
    m = fl.compartment_masks(p, track_id=1, dilation=2, gap=1)
    assert m["nuc"].sum() == 25                 # 5x5 nucleus
    assert not (m["ring"] & (p == 2)).any()     # ring never covers nucleus 2
    assert not (m["ring"] & (p == 1)).any()     # ring never covers its own nucleus
    assert m["ring"].sum() > 0                   # a ring exists
    assert (m["cell"] == (m["nuc"] | m["ring"])).all()

def test_compartments_absent_id_is_empty():
    p = _two_nuclei_plane()
    m = fl.compartment_masks(p, track_id=99)
    assert m["nuc"].sum() == 0 and m["ring"].sum() == 0 and m["cell"].sum() == 0

def test_background_is_median_of_non_cell():
    p = _two_nuclei_plane()
    ch = np.full((20, 20), 10.0)
    ch[p == 1] = 100.0
    ch[p == 2] = 100.0
    assert fl.background_per_frame(ch, p) == 10.0


import pandas as pd

def test_intensity_median_and_cn_ratio():
    mask = _two_nuclei_plane()[None, ...]           # 1 x 20 x 20
    ch = np.full((1, 20, 20), 5.0)                  # background 5
    ch[0][mask[0] == 1] = 25.0                      # nucleus 1 bright
    # make the cytoplasm ring of nucleus 1 brighter than its nucleus
    m = fl.compartment_masks(mask[0], 1)
    ch[0][m["ring"]] = 45.0
    df = fl.measure_intensity(ch, mask, channel_name="green")
    row = df[df["track_id"] == 1].iloc[0]
    assert abs(row["green_nuc_median"] - 20.0) < 1e-6      # 25 - bg 5
    assert row["green_cn_ratio"] > 1.0                     # cyto brighter -> active
    assert row["green_nuc_sum"] > 0


def test_haralick_contrast_higher_on_textured_than_flat():
    mask = np.zeros((1, 12, 24), dtype=np.int32)
    mask[0, 2:10, 2:10] = 1          # flat nucleus
    mask[0, 2:10, 14:22] = 2         # textured nucleus
    ch = np.zeros((1, 12, 24), dtype=float)
    ch[0][mask[0] == 1] = 100.0                       # uniform
    checker = (np.indices((8, 8)).sum(axis=0) % 2) * 200.0
    ch[0, 2:10, 14:22] = checker                      # high-frequency texture
    df = fl.haralick_features(ch, mask, channel_name="green", levels=16)
    c_flat = df[df["track_id"] == 1]["green_hara_contrast"].iloc[0]
    c_tex = df[df["track_id"] == 2]["green_hara_contrast"].iloc[0]
    assert c_tex > c_flat


def test_ring_preview_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    mask = _two_nuclei_plane()[None, ...]
    ch = np.random.RandomState(0).rand(1, 20, 20) * 50
    fig = fl.ring_preview_figure(mask, ch, track_id=None,
                                 dilations=(1, 2, 3), gap=1, rng=np.random.RandomState(1))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


from curator import analysis


def test_nuclear_morphometry_has_new_shape_columns():
    mask = np.zeros((1, 20, 20), dtype=np.int32)
    mask[0, 4:16, 6:12] = 1                      # a rectangle (elongated)
    nm = analysis.nuclear_morphometry(mask)
    for col in ["eccentricity", "solidity", "extent", "orientation",
                "axis_major_length", "axis_minor_length"]:
        assert col in nm.columns
    assert nm["eccentricity"].iloc[0] > 0.5      # clearly elongated
    assert 0 < nm["extent"].iloc[0] <= 1.0


def test_motility_has_anomalous_exponent():
    frames = np.arange(20, dtype=float)
    xs = frames * 1.0            # ballistic-ish straight motion
    ys = np.zeros_like(frames)
    out = analysis.motility_metrics(frames, xs, ys, frame_interval=1.0)
    assert "anomalous_exponent" in out
    assert np.isfinite(out["anomalous_exponent"])


from curator import exports
from curator.config import (COL_TRACK, COL_FRAME, COL_X, COL_Y, COL_OUTCOME,
                            COL_PARENT, COL_CLABEL, COL_TREATMENT, TREAT_CONTROL)

def _mini_df_and_mask():
    mask = np.zeros((2, 20, 20), dtype=np.int32)
    mask[0, 4:9, 4:9] = 1
    mask[1, 5:10, 5:10] = 1
    rows = []
    for f in (0, 1):
        ys, xs = np.nonzero(mask[f] == 1)
        rows.append({COL_TRACK: 1, COL_FRAME: f, COL_X: float(xs.mean()),
                     COL_Y: float(ys.mean()), COL_OUTCOME: "", COL_PARENT: -1,
                     COL_CLABEL: 1, COL_TREATMENT: TREAT_CONTROL})
    return pd.DataFrame(rows), mask

def test_features_table_merges_channel_columns():
    df, mask = _mini_df_and_mask()
    ch = np.full((2, 20, 20), 5.0)
    ch[mask == 1] = 30.0
    out = exports.features_table(df, mask, channels={"green": ch}, texture=False)
    assert "green_nuc_median" in out.columns
    assert "green_cn_ratio" in out.columns
    # per-frame row for (track 1, frame 0) carries the nucleus median (30-bg5=25)
    r0 = out[(out[COL_TRACK] == 1) & (out[COL_FRAME] == 0)].iloc[0]
    assert abs(r0["green_nuc_median"] - 25.0) < 1e-6

def test_features_table_merges_extra_tables():
    df, mask = _mini_df_and_mask()
    extra = pd.DataFrame({COL_TRACK: [1, 1], COL_FRAME: [0, 1],
                          "erk_cn": [1.5, 2.0]})
    out = exports.features_table(df, mask, extra_tables=[extra])
    assert "erk_cn" in out.columns
    assert out.loc[out[COL_FRAME] == 1, "erk_cn"].iloc[0] == 2.0


from curator import dialogs

def test_apply_channel_config_sets_color_and_measure():
    from curator.channels import ChannelLayer
    layers = [ChannelLayer("green", np.zeros((2, 4, 4)), "green"),
              ChannelLayer("red", np.zeros((2, 4, 4)), "magenta")]
    cfg = [{"name": "ERK-KTR", "color": "green", "measure": True},
           {"name": "red", "color": "red", "measure": False}]
    out = dialogs.apply_channel_config(layers, cfg)
    assert out[0].name == "ERK-KTR" and out[0].colormap == "green" and out[0].measure
    assert out[1].colormap == "red" and out[1].measure is False



# --- Bug B: mask <-> track_id reconciliation -------------------------------
def _mask_df_mismatch():
    """Mask labelled by per-frame seg labels (5,6), df track_ids 101/202."""
    mask = np.zeros((1, 20, 20), dtype=np.int32)
    mask[0, 4:9, 4:9] = 5
    mask[0, 4:9, 12:17] = 6
    df = pd.DataFrame([
        {COL_TRACK: 101, COL_FRAME: 0, COL_X: 6.0, COL_Y: 6.0,
         COL_OUTCOME: "", COL_PARENT: -1, COL_CLABEL: 101, COL_TREATMENT: TREAT_CONTROL},
        {COL_TRACK: 202, COL_FRAME: 0, COL_X: 14.0, COL_Y: 6.0,
         COL_OUTCOME: "", COL_PARENT: -1, COL_CLABEL: 202, COL_TREATMENT: TREAT_CONTROL},
    ])
    return df, mask


def test_mask_correspondence_detects_mismatch():
    from curator import analysis
    df, mask = _mask_df_mismatch()
    assert analysis.mask_track_correspondence(df, mask) == 0.0
    # a matching mask scores 1.0
    good = np.zeros((1, 20, 20), dtype=np.int32)
    good[0, 4:9, 4:9] = 101
    good[0, 4:9, 12:17] = 202
    assert analysis.mask_track_correspondence(df, good) == 1.0


def test_relabel_mask_to_track_ids():
    from curator import analysis
    df, mask = _mask_df_mismatch()
    new, n = analysis.relabel_mask_to_track_ids(df, mask)
    assert n == 2
    assert set(int(v) for v in np.unique(new[0])) == {0, 101, 202}
    assert analysis.mask_track_correspondence(df, new) == 1.0
    # original mask is untouched (returns a copy)
    assert set(int(v) for v in np.unique(mask[0])) == {0, 5, 6}


def test_measure_intensity_uses_background_roi():
    # field intensity 100, nucleus 130, a designated cell-free ROI corner at 20.
    mask = np.zeros((1, 20, 20), dtype=np.int32)
    mask[0, 4:9, 4:9] = 1
    ch = np.full((1, 20, 20), 100.0)
    ch[0][mask[0] == 1] = 130.0
    roi = np.zeros((1, 20, 20), dtype=bool)
    roi[0, 0:3, 0:3] = True
    ch[0, 0:3, 0:3] = 20.0
    # ROI-based background (20) -> nuc median 110; auto (median of non-cell ~100) -> 30
    df_roi = fl.measure_intensity(ch, mask, channel_name="g", background_roi=roi)
    assert abs(df_roi["g_nuc_median"].iloc[0] - 110.0) < 1e-6
    df_auto = fl.measure_intensity(ch, mask, channel_name="g")
    assert abs(df_auto["g_nuc_median"].iloc[0] - 30.0) < 1e-6


def test_features_lifetime_on_middle_frame_only():
    # track 1 lives 5 frames (0..4), track 2 lives 3 frames (0..2)
    mask = np.zeros((5, 20, 20), dtype=np.int32)
    for f in range(5):
        mask[f, 4:9, 4:9] = 1
    for f in range(3):
        mask[f, 4:9, 12:17] = 2
    rows = []
    for f in range(5):
        rows.append({COL_TRACK: 1, COL_FRAME: f, COL_X: 6.0, COL_Y: 6.0,
                     COL_OUTCOME: "", COL_PARENT: -1, COL_CLABEL: 1,
                     COL_TREATMENT: TREAT_CONTROL})
    for f in range(3):
        rows.append({COL_TRACK: 2, COL_FRAME: f, COL_X: 14.0, COL_Y: 6.0,
                     COL_OUTCOME: "", COL_PARENT: -1, COL_CLABEL: 2,
                     COL_TREATMENT: TREAT_CONTROL})
    out = exports.features_table(pd.DataFrame(rows), mask, frame_interval=2.0)
    assert "lifetime" in out.columns
    # exactly one non-NaN lifetime per track
    per = out.dropna(subset=["lifetime"]).groupby(COL_TRACK).size()
    assert set(per.index) == {1, 2} and (per == 1).all()
    # value sits on the middle frame and equals (span)*interval
    r1 = out[(out[COL_TRACK] == 1) & (out["lifetime"].notna())].iloc[0]
    assert r1[COL_FRAME] == 2 and r1["lifetime"] == 10.0     # 5 frames * 2
    r2 = out[(out[COL_TRACK] == 2) & (out["lifetime"].notna())].iloc[0]
    assert r2[COL_FRAME] == 1 and r2["lifetime"] == 6.0      # 3 frames * 2


# --- Deep-lineage IDs must not collide with the reserved quarantine range ----
def test_deep_lineage_id_is_normal_not_quarantined():
    from curator import config
    # A re-sequenced deep-lineage id like 1122111 (7 digits) must stay "normal".
    assert config.is_normal_id(1122111)
    assert not config.is_quarantined_id(1122111)
    # A genuine eviction id is still quarantined.
    assert config.is_quarantined_id(config.QUARANTINE_START + 5)
    # The reserved ranges stay ordered and inside the uint32 mask limit.
    assert config.NORMAL_ID_MAX <= config.QUARANTINE_START
    assert config.QUARANTINE_END <= config.RESEQ_OFFSET
    assert config.RESEQ_OFFSET + config.NORMAL_ID_MAX < 2**32   # no uint32 overflow


def test_features_table_includes_deep_lineage_id():
    mask = np.zeros((2, 20, 20), dtype=np.int32)
    for f in range(2):
        mask[f, 4:9, 4:9] = 1122111
    rows = [{COL_TRACK: 1122111, COL_FRAME: f, COL_X: 6.0, COL_Y: 6.0,
             COL_OUTCOME: "", COL_PARENT: -1, COL_CLABEL: 1122111,
             COL_TREATMENT: TREAT_CONTROL} for f in range(2)]
    out = exports.features_table(pd.DataFrame(rows), mask)
    assert (out[COL_TRACK] == 1122111).any(), "deep-lineage id dropped from features"
    assert out.loc[out[COL_TRACK] == 1122111, "area_px"].notna().all()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
