"""LoFTR fallback for pairs that sparse features cannot connect.

SIFT needs repeatable keypoints. A pair whose only overlap is a plain cabinet
front, a blown-out promo header or a stretch of floor tile has very few, and no
amount of ratio-test tuning invents them. LoFTR (Sun et al., CVPR 2021) is
detector-free: it matches dense coarse features with a transformer and refines
them, so it produces correspondences in exactly the low-texture regions where
SIFT produces none.

It is used strictly as a *fallback*, and its output passes through the same
geometric and photometric verification as SIFT's - a deep matcher is not
allowed to bypass the checks, only to supply candidates.

Weights are downloaded by kornia on first use (~45 MB). If kornia or the
weights are unavailable the import fails and the pipeline carries on with SIFT
only; this module is never required for a run to succeed.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .imageset import StoreImage
from .matching import PairMatch, homography_is_plausible, photometric_ncc

_MATCHER = None
_MAX_DIM = 840          # LoFTR is quadratic in token count; 840 fits in memory
                        # comfortably and keeps coarse cells ~8 px on our images


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_matcher():
    global _MATCHER
    if _MATCHER is None:
        import kornia.feature as KF
        _MATCHER = KF.LoFTR(pretrained="outdoor").to(_device()).eval()
    return _MATCHER


def _prep(bgr: np.ndarray) -> tuple[torch.Tensor, float]:
    """Grayscale, downscale to _MAX_DIM, return (1,1,H,W) tensor and the scale."""
    s = min(1.0, _MAX_DIM / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # LoFTR's coarse stage strides by 8, so both sides must be multiples of 8.
    h, w = gray.shape
    gray = gray[:h - h % 8, :w - w % 8]
    t = torch.from_numpy(gray)[None, None].to(_device())
    return t, s


@torch.no_grad()
def estimate_pair_loftr(im_i: StoreImage, im_j: StoreImage,
                        conf_thresh: float = 0.6, ransac_thresh: float = 3.0,
                        min_inliers: int = 40, min_inlier_ratio: float = 0.10,
                        min_ncc: float = 0.25, min_overlap: float = 0.04,
                        verbose: bool = True) -> PairMatch:
    """Estimate image i -> image j with LoFTR, verified like any other pair."""
    matcher = _get_matcher()
    ti, si = _prep(im_i.work)
    tj, sj = _prep(im_j.work)

    out = matcher({"image0": ti, "image1": tj})
    kp_i = out["keypoints0"].cpu().numpy()
    kp_j = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()

    keep = conf >= conf_thresh
    # Coordinates come back in the downscaled frame; lift them to work scale so
    # the resulting homography lives in the same frame as every SIFT pair's.
    src = kp_i[keep] / si
    dst = kp_j[keep] / sj

    fail = lambda why, **kw: PairMatch(  # noqa: E731
        i=im_i.index, j=im_j.index, H=kw.get("H"), n_matches=len(src),
        n_inliers=kw.get("n_inliers", 0), inlier_ratio=kw.get("inlier_ratio", 0.0),
        overlap=kw.get("overlap", 0.0), ncc=kw.get("ncc", 0.0),
        pts_i=np.zeros((0, 2)), pts_j=np.zeros((0, 2)), ok=False,
        reason=f"loftr: {why}")

    if len(src) < min_inliers:
        return fail(f"only {len(src)} confident matches")

    H, inl = cv2.findHomography(src, dst, method=cv2.USAC_MAGSAC,
                                ransacReprojThreshold=ransac_thresh,
                                maxIters=20000, confidence=0.9999)
    if H is None:
        return fail("RANSAC returned no model")

    inl = inl.ravel().astype(bool)
    n_in, ratio_in = int(inl.sum()), int(inl.sum()) / len(src)

    ok, why = homography_is_plausible(H, im_i.work_shape, im_j.work_shape)
    if not ok:
        return fail(why, H=H, n_inliers=n_in, inlier_ratio=ratio_in)

    ncc, overlap = photometric_ncc(im_i.work, im_j.work, H)
    pm = PairMatch(i=im_i.index, j=im_j.index, H=H, n_matches=len(src),
                   n_inliers=n_in, inlier_ratio=ratio_in, overlap=overlap,
                   ncc=ncc, pts_i=src[inl], pts_j=dst[inl], ok=True,
                   reason="loftr")
    if n_in < min_inliers:
        pm.ok, pm.reason = False, f"loftr: {n_in} inliers < {min_inliers}"
    elif ratio_in < min_inlier_ratio:
        pm.ok, pm.reason = False, f"loftr: inlier ratio {ratio_in:.2f} low"
    elif overlap < min_overlap:
        pm.ok, pm.reason = False, f"loftr: overlap {overlap:.3f} low"
    elif ncc < min_ncc:
        pm.ok, pm.reason = False, f"loftr: ncc {ncc:.2f} < {min_ncc}"

    if verbose:
        print(f"  [deep] {im_i.short}->{im_j.short} "
              f"{'OK  ' if pm.ok else 'drop'} matches={len(src)} inliers={n_in} "
              f"({ratio_in:.2f}) ovl={overlap:.2f} ncc={ncc:+.2f} {pm.reason}")
    return pm
