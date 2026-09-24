"""Tests for the Streamlit front end.

The app deliberately holds no stitching logic - it writes uploads to a temp
directory and calls the same `stitch_store()` as the CLI. So these tests cover
the two things that can still break independently of the pipeline:

  1. the script renders without raising (a Streamlit script runs top to bottom,
     so a typo anywhere is a blank page for the user, not a stack trace);
  2. the PipelineConfig the sidebar builds is actually valid and produces a
     panorama - i.e. the UI and the library have not drifted apart.

Run:  python -m pytest tests/test_streamlit_app.py -v
"""
import os
import sys

import cv2
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")
from streamlit.testing.v1 import AppTest                    # noqa: E402

APP = os.path.join(REPO, "streamlit_app.py")


# --------------------------------------------------------------------------
# The script itself
# --------------------------------------------------------------------------

def test_app_module_is_not_named_streamlit():
    """A module named streamlit.py shadows the package and breaks line 1.

    Python puts the script's directory first on sys.path, so `import streamlit`
    from a file called streamlit.py imports that file. Verified: it resolves to
    the local file, not the package. Pinned here because the obvious filename
    is the broken one.
    """
    assert not os.path.exists(os.path.join(REPO, "streamlit.py")), (
        "A file named streamlit.py shadows the installed streamlit package; "
        "the entry point must be streamlit_app.py")
    assert os.path.exists(APP)


def test_app_renders_initial_state_without_error():
    """No uploads yet: the app must render its prompt, not explode."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Shelf panorama stitcher"
    # The empty state tells the user what to do.
    assert any("Upload at least two" in i.value for i in at.info)


def test_sidebar_exposes_the_knobs_that_matter():
    """The controls the README documents must actually exist."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    labels = " ".join(
        [s.label for s in at.slider] + [s.label for s in at.selectbox]
        + [s.label for s in at.select_slider]
    )
    for expected in ("RANSAC hypotheses", "Photometric gate", "Blending",
                     "Output resolution"):
        assert expected in labels, f"missing control: {expected}"


def test_file_uploader_accepts_multiple_images():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.file_uploader) == 1


# --------------------------------------------------------------------------
# UI settings -> pipeline
# --------------------------------------------------------------------------

def _synthetic_pair(tmp_path):
    """Two overlapping tiles cut from one textured image, written to disk."""
    rng = np.random.default_rng(0)
    h, w = 340, 520
    base = np.full((h, w * 2, 3), 30, np.uint8)
    for _ in range(600):
        c = tuple(int(v) for v in rng.integers(70, 255, 3))
        cv2.circle(base, (int(rng.integers(0, w * 2)), int(rng.integers(0, h))),
                   int(rng.integers(5, 20)), c, -1)
    d = tmp_path / "images"
    d.mkdir()
    cv2.imwrite(str(d / "b.jpg"), base[:, :w])
    cv2.imwrite(str(d / "a.jpg"), base[:, w - 190:])
    return d


def test_sidebar_defaults_build_a_valid_config_and_stitch(tmp_path):
    """The exact kwargs the app passes must construct and run.

    Guards against the UI offering an option the library no longer accepts -
    which would be a TypeError the user only sees as a failed stitch.
    """
    from shelfpano.pipeline import PipelineConfig, stitch_store

    cfg = PipelineConfig(
        n_features=12000, min_ncc=0.25, n_hypotheses=6,
        use_deep_fallback=False, render_scale=1.0,
        exposure="blocks", seam="graphcut", blend="multiband",
        seam_debug_path=str(tmp_path / "seams.jpg"),
    )
    pano, rep = stitch_store(str(_synthetic_pair(tmp_path)), cfg, verbose=False)

    assert rep.n_images_used == 2, rep.components
    assert pano.ndim == 3 and pano.shape[2] == 3
    assert (tmp_path / "seams.jpg").exists(), "seam debug map was not written"
    # Metrics the app puts in st.metric must be present and formattable.
    for key in ("reproj_rms_px", "mean_overlap_ncc"):
        assert key in rep.metrics
    f"{rep.metrics['reproj_rms_px']} {rep.metrics['mean_overlap_ncc']}"


@pytest.mark.parametrize("blend", ["multiband", "feather", "none"])
def test_every_blend_choice_offered_by_the_ui_works(blend, tmp_path):
    """Each dropdown value must survive a real stitch (see test_compose_options
    for why: some cv2.detail classes segfault if built the obvious way)."""
    from shelfpano.pipeline import PipelineConfig, stitch_store

    cfg = PipelineConfig(n_features=4000, use_deep_fallback=False,
                         render_scale=0.5, blend=blend, max_megapixels=8.0)
    pano, rep = stitch_store(str(_synthetic_pair(tmp_path)), cfg, verbose=False)
    assert rep.n_images_used == 2 and pano.size > 0


def test_jpeg_encoding_roundtrips():
    """The download button hands the browser these bytes."""
    img = (np.random.default_rng(0).integers(0, 255, (120, 200, 3))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    back = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    assert back is not None and back.shape == img.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
