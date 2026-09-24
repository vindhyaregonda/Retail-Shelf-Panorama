"""Unit tests for the geometry that the rest of the pipeline trusts.

These target the pieces where a silent sign or scale error would produce a
plausible-looking but wrong panorama - exactly the bugs that are expensive to
find by looking at output images. Each test builds a case with a known answer
so correctness is checked against ground truth rather than against "it ran".

Run:  python -m pytest tests/ -v
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shelfpano.bundle import bundle_adjust                              # noqa: E402
from shelfpano.compose import to_render_frame                           # noqa: E402
from shelfpano.graph import (build_adjacency, connected_components,     # noqa: E402
                             initial_homographies, maximum_spanning_tree)
from shelfpano.imageset import image_corners, rescale_H, warp_points    # noqa: E402
from shelfpano.matching import PairMatch, homography_is_plausible       # noqa: E402


# --------------------------------------------------------------------------
# Coordinate-frame algebra
# --------------------------------------------------------------------------

def test_rescale_H_matches_explicit_point_mapping():
    """Rescaling a homography must agree with scaling points by hand."""
    H = np.array([[1.02, 0.03, 40.0],
                  [-0.01, 0.99, -12.0],
                  [1e-5, 2e-6, 1.0]])
    r = 0.4                                   # work = 0.4 * full
    H_small = rescale_H(H, 1.0, r)

    pts_full = np.array([[100.0, 200.0], [1500.0, 900.0], [0.0, 0.0]])
    # Map at full scale then shrink, vs shrink then map at small scale.
    via_full = warp_points(H, pts_full) * r
    via_small = warp_points(H_small, pts_full * r)
    np.testing.assert_allclose(via_full, via_small, atol=1e-8)


def test_to_render_frame_handles_differing_input_scales():
    """A source and anchor downscaled by different factors must still line up."""
    H_work = np.array([[1.0, 0.0, 250.0], [0.0, 1.0, -30.0], [0.0, 0.0, 1.0]])
    s_i, s_a, rho = 0.25, 0.5, 1.0
    H_r = to_render_frame(H_work, s_i, s_a, rho)

    # A full-res point in image i -> work_i -> work_anchor -> full-res anchor.
    p_full_i = np.array([[800.0, 600.0]])
    p_work_i = p_full_i * s_i
    p_work_a = warp_points(H_work, p_work_i)
    expected = p_work_a / s_a
    np.testing.assert_allclose(warp_points(H_r, p_full_i), expected, atol=1e-9)


def test_to_render_frame_render_scale_is_linear():
    H_work = np.eye(3)
    H_half = to_render_frame(H_work, 0.5, 0.5, 0.5)
    p = np.array([[400.0, 300.0]])
    np.testing.assert_allclose(warp_points(H_half, p), p * 0.5, atol=1e-9)


# --------------------------------------------------------------------------
# Homography plausibility gate
# --------------------------------------------------------------------------

def test_plausible_accepts_a_modest_translation():
    H = np.array([[1.0, 0.0, 300.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
    ok, why = homography_is_plausible(H, (900, 1600), (900, 1600))
    assert ok, why


def test_plausible_rejects_mirroring():
    """A warp that flips handedness can never come from a real viewpoint change."""
    H = np.array([[-1.0, 0.0, 1600.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ok, why = homography_is_plausible(H, (900, 1600), (900, 1600))
    assert not ok and "mirror" in why.lower()


def test_plausible_rejects_extreme_blowup():
    H = np.diag([40.0, 40.0, 1.0])
    ok, why = homography_is_plausible(H, (900, 1600), (900, 1600))
    assert not ok and "area ratio" in why


def test_plausible_rejects_bowtie_fold():
    """Self-intersecting outlines are the classic degenerate RANSAC result."""
    src = image_corners((900, 1600)).astype(np.float32)
    dst = np.array([[0, 0], [1600, 0], [0, 900], [1600, 900]], np.float32)  # swapped
    H = cv2.getPerspectiveTransform(src, dst)
    ok, why = homography_is_plausible(H, (900, 1600), (900, 1600))
    assert not ok


def test_plausible_rejects_nonfinite():
    ok, _ = homography_is_plausible(np.full((3, 3), np.nan), (900, 1600), (900, 1600))
    assert not ok


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def _pm(i, j, H, inliers=100, ncc=0.6):
    return PairMatch(i=i, j=j, H=H, n_matches=inliers * 2, n_inliers=inliers,
                     inlier_ratio=0.5, overlap=0.3, ncc=ncc,
                     pts_i=np.zeros((inliers, 2)), pts_j=np.zeros((inliers, 2)),
                     ok=True)


def _tx(dx):
    return np.array([[1.0, 0, dx], [0, 1.0, 0], [0, 0, 1.0]])


def test_adjacency_inverts_homographies_consistently():
    """adj[j][i] must be the exact inverse of adj[i][j]."""
    adj = build_adjacency(2, [_pm(0, 1, _tx(300))])
    np.testing.assert_allclose(adj[0][1].H @ adj[1][0].H, np.eye(3), atol=1e-9)


def test_connected_components_splits_disjoint_graphs():
    adj = build_adjacency(4, [_pm(0, 1, _tx(100)), _pm(2, 3, _tx(100))])
    comps = connected_components(4, adj)
    assert sorted(map(sorted, comps)) == [[0, 1], [2, 3]]


def test_spanning_tree_prefers_high_confidence_edges():
    """A weak direct edge must lose to a strong two-hop route."""
    pairs = [_pm(0, 1, _tx(300), inliers=1000, ncc=0.8),
             _pm(1, 2, _tx(300), inliers=1000, ncc=0.8),
             _pm(0, 2, _tx(600), inliers=45, ncc=0.3)]
    adj = build_adjacency(3, pairs)
    edges = maximum_spanning_tree([0, 1, 2], adj, anchor=0)
    assert (0, 2) not in edges and (2, 0) not in edges
    assert len(edges) == 2


def test_initial_homographies_compose_along_the_chain():
    """Chained transforms must compose, with the direction convention held.

    A pair (i, j, H) means H maps points of image i into image j. So with
    H_01 = H_12 = translate(+300), a feature at the origin of image 0 is seen
    at x=+300 in image 1 and x=+600 in image 2 - which means image 2's *frame*
    sits 600 px to the LEFT in image 0's coordinates. initial_homographies
    returns image->anchor, so H[2] must map image 2's origin to -600, not +600.
    Getting this sign backwards mirrors the panorama, so it is pinned here.
    """
    pairs = [_pm(0, 1, _tx(300)), _pm(1, 2, _tx(300))]
    adj = build_adjacency(3, pairs)
    H = initial_homographies([0, 1, 2], adj, anchor=0, verbose=False)
    np.testing.assert_allclose(H[0], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(warp_points(H[1], np.array([[0.0, 0.0]])),
                               np.array([[-300.0, 0.0]]), atol=1e-6)
    np.testing.assert_allclose(warp_points(H[2], np.array([[0.0, 0.0]])),
                               np.array([[-600.0, 0.0]]), atol=1e-6)


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------

def _synthetic_problem(seed=0):
    """Three views of a plane with known homographies and noise-free matches."""
    rng = np.random.default_rng(seed)
    H_true = {0: np.eye(3),
              1: np.array([[1.03, 0.02, 420.0], [-0.015, 1.01, 8.0],
                           [2e-5, 1e-6, 1.0]]),
              2: np.array([[1.07, 0.05, 830.0], [-0.03, 1.02, 20.0],
                           [4e-5, 3e-6, 1.0]])}
    H_true = {k: v / v[2, 2] for k, v in H_true.items()}
    pairs = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        pts_i = rng.uniform([0, 0], [1600, 900], size=(300, 2))
        Hij = np.linalg.solve(H_true[j], H_true[i])
        pts_j = warp_points(Hij, pts_i)
        pairs.append(_pm(i, j, Hij, inliers=300))
        pairs[-1].pts_i, pairs[-1].pts_j = pts_i, pts_j
    return H_true, pairs


def test_bundle_adjustment_recovers_a_perturbed_layout():
    """Perturb a known-good solution; BA should pull the error back down."""
    H_true, pairs = _synthetic_problem()
    rng = np.random.default_rng(7)
    H_init = {}
    for k, v in H_true.items():
        if k == 0:
            H_init[k] = np.eye(3)
            continue
        p = v.copy()
        p[0, 2] += rng.normal(0, 12)      # translation drift, as chaining produces
        p[1, 2] += rng.normal(0, 12)
        p[0, 0] += rng.normal(0, 0.01)
        H_init[k] = p / p[2, 2]

    H_ref, stats = bundle_adjust(H_init, pairs, anchor=0, verbose=False)
    assert stats["rms_after"] < stats["rms_before"]
    assert stats["rms_after"] < 0.5          # noise-free data: should nearly vanish
    assert stats["accepted"]


def test_bundle_adjustment_keeps_the_anchor_fixed():
    """The gauge must not drift - anchor stays exactly identity."""
    H_true, pairs = _synthetic_problem()
    H_init = dict(H_true)
    H_ref, _ = bundle_adjust(H_init, pairs, anchor=0, verbose=False)
    np.testing.assert_allclose(H_ref[0], np.eye(3), atol=1e-12)


def test_bundle_adjustment_is_a_noop_on_an_exact_solution():
    """Starting at the optimum, BA must not make things worse."""
    H_true, pairs = _synthetic_problem()
    _, stats = bundle_adjust(dict(H_true), pairs, anchor=0, verbose=False)
    assert stats["rms_before"] < 1e-6
    assert stats["accepted"]


def test_bundle_adjustment_survives_a_single_image():
    H_ref, stats = bundle_adjust({0: np.eye(3)}, [], anchor=0, verbose=False)
    assert list(H_ref) == [0] and stats["n_images"] == 1


# --------------------------------------------------------------------------
# Metric reporting
# --------------------------------------------------------------------------

def test_evaluate_layout_returns_full_schema_when_nothing_scored():
    """An unscoreable layout must still return every key, not a stub.

    Regression: this used to return {"n_pairs_scored": 0}, so any caller that
    formatted the result died with KeyError('reproj_rms_px') - and because the
    metrics are printed several stages after the graph is built, the error
    surfaced far from the real cause (a match graph that had come apart).
    Callers should be able to check one flag and format the rest uniformly.
    """
    from shelfpano.evaluate import evaluate_layout

    scored = evaluate_layout([], {}, [])
    assert scored["n_pairs_scored"] == 0
    for key in ("reproj_rms_px", "reproj_median_px", "reproj_p95_px",
                "mean_overlap_ncc", "min_overlap_ncc", "mean_overlap_frac"):
        assert key in scored, f"missing {key}"
        assert scored[key] is None
    assert scored["per_pair"] == []
    # Must be formattable without a single guard.
    f"{scored['reproj_rms_px']} {scored['mean_overlap_ncc']}"


def test_evaluate_layout_key_set_is_identical_scored_and_unscored():
    """The populated and empty results must agree on their keys.

    Guards the shape contract from both sides: a caller written against a
    successful run must work unchanged on a failed one.
    """
    from shelfpano.evaluate import evaluate_layout
    from shelfpano.imageset import StoreImage

    rng = np.random.default_rng(1)
    imgs = [StoreImage(path=f"/tmp/fake_{k}.png", index=k,
                       work=rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8),
                       full_shape=(900, 1600), scale=1.0)
            for k in range(3)]
    H_true, pairs = _synthetic_problem()

    populated = evaluate_layout(imgs, H_true, pairs)
    empty = evaluate_layout([], {}, [])

    assert populated["n_pairs_scored"] == 3
    assert set(populated) == set(empty)
    # The geometry is exact, so reprojection error must be ~0 regardless of
    # what the (random) pixels do to the NCC term.
    assert populated["reproj_rms_px"] < 1e-6


# --------------------------------------------------------------------------
# End to end, on synthetic data with a known answer
# --------------------------------------------------------------------------

def test_end_to_end_on_a_synthetic_split_image(tmp_path):
    """Cut one textured image into overlapping tiles; stitching must restore it.

    This is the only test that exercises load -> features -> match -> graph ->
    bundle -> compose together, and it has a ground-truth answer: the panorama
    must come back close to the original width, with every tile used.
    """
    from shelfpano.pipeline import PipelineConfig, stitch_store

    rng = np.random.default_rng(3)
    h, w = 700, 1500
    # Structured, non-repeating texture: random blobs give SIFT something
    # unambiguous to lock onto, unlike a periodic pattern.
    src = np.full((h, w, 3), 40, np.uint8)
    for _ in range(700):
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(src, (x, y), int(rng.integers(6, 26)), c, -1)
    src = cv2.GaussianBlur(src, (3, 3), 0)

    d = tmp_path / "images"
    d.mkdir()
    # Three tiles with generous overlap, written in an order unrelated to
    # position so the pipeline has to discover the arrangement.
    spans = [(0, 700), (400, 1100), (800, 1500)]
    for name, (x0, x1) in zip(["c_tile", "a_tile", "b_tile"], spans):
        cv2.imwrite(str(d / f"{name}.png"), src[:, x0:x1])

    cfg = PipelineConfig(work_max_dim=800, n_features=6000,
                         use_deep_fallback=False, max_megapixels=20.0)
    pano, rep = stitch_store(str(d), cfg, verbose=False)

    assert rep.n_images_used == 3, rep.components
    assert rep.metrics["reproj_rms_px"] < 3.0
    # Pure translation between tiles, so the panorama should be about as wide
    # as the original and no taller.
    assert 0.85 * w <= pano.shape[1] <= 1.15 * w
    assert pano.shape[0] <= 1.25 * h


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
