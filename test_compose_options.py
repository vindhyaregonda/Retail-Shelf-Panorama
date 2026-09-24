"""Every selectable compositing option must actually run.

Motivation - a real bug this suite exists to prevent
-----------------------------------------------------
Several `cv2.detail_*` classes expose no constructor to Python. Calling
`cv2.detail_NoSeamFinder()` still *succeeds*: it returns a wrapper object
holding a NULL `Ptr`. The failure only arrives at the first method call, as a
SIGSEGV inside libopencv - which kills the interpreter outright, with no Python
exception, no traceback, and (in a notebook) a dead kernel.

Confirmed on OpenCV 4.13 for `NoExposureCompensator`, `NoSeamFinder` and
`VoronoiSeamFinder`; the fix is to build those via their `*_createDefault`
factory. `--exposure none`, `--seam none` and `--seam voronoi` were all
advertised CLI options that hard-crashed.

Because the crash is a segfault rather than an exception, no amount of
`pytest.raises` would have caught it and no `try/except` can guard it - the
only defence is to *execute* every enumerated option. Each combination runs in
a **subprocess**, so a regression shows up as a non-zero exit code (-11) and a
readable failure instead of taking the whole test session down with it.

These are slow-ish (a subprocess and a small stitch each) but they run on a
tiny synthetic image, so the whole file is a few seconds.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors the `choices=` lists in cli.py. If a new option is added there and not
# here, test_cli_choices_are_all_covered fails.
EXPOSURES = ["blocks", "gain", "none"]
SEAMS = ["graphcut", "dp", "voronoi", "none"]
BLENDS = ["multiband", "feather", "none"]

# Runs one compose() in a fresh interpreter. Kept as source text rather than a
# helper import so the child cannot inherit any state from the test process.
CHILD = r"""
import sys, os
sys.path.insert(0, {repo!r})
import numpy as np
from shelfpano.imageset import StoreImage
from shelfpano.compose import compose

exposure, seam, blend = sys.argv[1], sys.argv[2], sys.argv[3]

# Two synthetic 'photos' that overlap by roughly a third, with enough texture
# that the seam finders have something to cut along.
rng = np.random.default_rng(0)
h, w = 220, 320
base = rng.integers(0, 255, (h, w * 2, 3), dtype=np.uint8)
tiles = [base[:, :w].copy(), base[:, w - 110:2 * w - 110].copy()]

images, tmp = [], {tmpdir!r}
for i, t in enumerate(tiles):
    import cv2
    p = os.path.join(tmp, f"tile_{{i}}.png")
    cv2.imwrite(p, t)
    images.append(StoreImage(path=p, index=i, work=t, full_shape=t.shape[:2], scale=1.0))

# Image 1 sits (w-110) px to the right of image 0 in image 0's frame.
H = {{0: np.eye(3),
     1: np.array([[1.0, 0.0, float(w - 110)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])}}

out = compose(images, H, anchor=0, render_scale=1.0, seam_scale=0.5,
              exposure=exposure, seam=seam, blend=blend,
              max_megapixels=8.0, verbose=False)
assert out.ndim == 3 and out.shape[2] == 3, out.shape
assert out.shape[1] > w, "panorama should be wider than one tile"
assert (out > 0).any(), "panorama is entirely black"
print("OK", out.shape)
"""


def _run(exposure, seam, blend, tmpdir):
    src = CHILD.format(repo=REPO, tmpdir=str(tmpdir))
    script = os.path.join(str(tmpdir), "child.py")
    with open(script, "w") as f:
        f.write(src)
    return subprocess.run([sys.executable, script, exposure, seam, blend],
                          capture_output=True, text=True, timeout=300)


def _assert_ok(proc, label):
    if proc.returncode == 0:
        return
    if proc.returncode < 0:
        pytest.fail(
            f"{label} crashed with signal {-proc.returncode} "
            f"(SIGSEGV=11 means a cv2.detail_* object was built with a NULL Ptr; "
            f"use its *_createDefault factory instead).\n{proc.stderr[-2000:]}")
    pytest.fail(f"{label} exited {proc.returncode}\n{proc.stderr[-2000:]}")


@pytest.mark.parametrize("exposure", EXPOSURES)
def test_every_exposure_option_runs(exposure, tmp_path):
    _assert_ok(_run(exposure, "graphcut", "multiband", tmp_path),
               f"--exposure {exposure}")


@pytest.mark.parametrize("seam", SEAMS)
def test_every_seam_option_runs(seam, tmp_path):
    _assert_ok(_run("blocks", seam, "multiband", tmp_path), f"--seam {seam}")


@pytest.mark.parametrize("blend", BLENDS)
def test_every_blend_option_runs(blend, tmp_path):
    _assert_ok(_run("blocks", "graphcut", blend, tmp_path), f"--blend {blend}")


def test_all_stages_disabled_runs(tmp_path):
    """The 'raw paste' configuration the notebook uses to show what each stage buys."""
    _assert_ok(_run("none", "none", "none", tmp_path), "all stages disabled")


def test_cli_choices_are_all_covered():
    """The lists above must match cli.py, or a new option ships untested."""
    from shelfpano.cli import build_parser
    actions = {a.dest: a for a in build_parser()._actions}
    assert set(actions["exposure"].choices) == set(EXPOSURES)
    assert set(actions["seam"].choices) == set(SEAMS)
    assert set(actions["blend"].choices) == set(BLENDS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
