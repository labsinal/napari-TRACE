"""The warning that stands in for a check that never ran.

``check_measurement_source`` confirms, by phase correlation, that an extra
channel is registered against the movie the mask was built on. That part needs
OpenCV, which is an optional dependency. It used to return with neither an
error nor a warning when OpenCV was missing -- reporting "nothing wrong found"
for a check that never happened, on the one question that decides whether a
measured channel means anything.
"""
import sys

import numpy as np

from curator import channels as chmod


def _stack(n=4, size=16, shift=0):
    """n frames holding one bright square, displaced in x by *shift* px."""
    a = np.zeros((n, size, size), dtype=np.uint16)
    a[:, 5:11, 5 + shift:11 + shift] = 1000
    return a


def test_without_cv2_the_unverified_alignment_is_reported():
    ref, arr = _stack(), _stack(shift=4)        # deliberately misaligned
    saved = sys.modules.get("cv2", "absent")
    sys.modules["cv2"] = None                   # makes `import cv2` raise
    try:
        errors, warnings = chmod.check_measurement_source(
            arr, mask_frames=ref.shape[0], reference=ref)
    finally:
        if saved == "absent":
            del sys.modules["cv2"]
        else:
            sys.modules["cv2"] = saved
    assert errors == []                         # not blocked: we do not know
    assert any("NOT verified" in w for w in warnings), warnings


def test_with_cv2_a_misaligned_channel_is_still_an_error():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("    (skipped: OpenCV not installed)")
        return
    ref, arr = _stack(), _stack(shift=4)
    errors, _ = chmod.check_measurement_source(
        arr, mask_frames=ref.shape[0], reference=ref)
    assert any("misaligned" in e for e in errors), errors


def test_with_cv2_an_aligned_channel_passes():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("    (skipped: OpenCV not installed)")
        return
    ref = _stack()
    errors, warnings = chmod.check_measurement_source(
        ref.copy(), mask_frames=ref.shape[0], reference=ref)
    assert errors == [], errors
    assert warnings == [], warnings


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
