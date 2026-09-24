"""Local feature detection and description.

Shelf photos are an adversarial case for feature matching: a cigarette gantry
is a grid of near-identical rectangular facings, so descriptors repeat across
the image and the usual Lowe ratio test throws away most of the true matches
(the second-nearest neighbour is a *different facing of the same product* and
is nearly as close as the true match).

The response here has three parts:

1. **A lot of keypoints.** We ask SIFT for up to `n_features` (default 12000)
   rather than the usual ~2000. Distinctive detail does exist - price tags,
   promo headers, shelf rail joints, scuffs - it is just sparse relative to the
   repeated product facings, so we need a wide net to collect enough of it.
2. **RootSIFT.** L1-normalise then square-root the descriptor, which makes
   Euclidean distance behave like the Hellinger kernel. It is a free accuracy
   gain over raw SIFT (Arandjelovic & Zisserman, CVPR 2012, "Three things
   everyone should know to improve object retrieval").
3. **A permissive ratio test, compensated by strict geometry.** See
   `matching.py` - we loosen the ratio threshold to keep the true matches that
   repetition would otherwise suppress, and let RANSAC plus a photometric
   check do the rejecting.

SIFT is used rather than ORB because ORB's binary descriptor is markedly less
discriminative on repeated texture: swapping it in drops 2 of 19 images across
the five stores (measured in `experiments/exp05_ablation.py`).

Note that the same ablation shows RootSIFT and the loosened ratio threshold to
be *neutral* on this dataset. They are kept because they cost nothing and guard
failure modes argued for below, but they are not what makes this work - the
wide keypoint budget and the verification in `matching.py` are.
"""
from __future__ import annotations

import cv2
import numpy as np

from .imageset import StoreImage


def _root_sift(desc: np.ndarray | None) -> np.ndarray | None:
    """Hellinger-normalise SIFT descriptors: L1 normalise, then sqrt."""
    if desc is None or len(desc) == 0:
        return None
    desc = desc.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc)


def make_detector(kind: str = "sift", n_features: int = 12000):
    """Build a detector. `kind` is one of sift | rootsift | orb | akaze."""
    kind = kind.lower()
    if kind in ("sift", "rootsift"):
        # contrastThreshold below the 0.04 default: shelf interiors are flat and
        # evenly lit, so genuinely useful low-contrast corners (tag edges, rail
        # seams) fall under the default and never get proposed.
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.03,
                               edgeThreshold=12)
    if kind == "orb":
        return cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=12,
                              fastThreshold=10)
    if kind == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"unknown detector {kind!r}")


def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    """Grayscale with local contrast equalisation.

    Exposure varies a lot between shots (the camera re-meters on each fixture,
    and overhead spots blow out parts of the gantry). CLAHE normalises local
    contrast so the detector fires on the same structures in both images of a
    pair, which matters more than global brightness matching.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def detect_all(images: list[StoreImage], kind: str = "rootsift",
               n_features: int = 12000, verbose: bool = True) -> None:
    """Detect and describe features in-place on every image's work-scale copy."""
    det = make_detector(kind, n_features)
    use_root = kind.lower() == "rootsift"
    for im in images:
        kps, desc = det.detectAndCompute(_clahe_gray(im.work), None)
        if use_root:
            desc = _root_sift(desc)
        im.keypoints = kps
        im.descriptors = desc
        if verbose:
            print(f"  [features] {im.short}  {len(kps):6d} keypoints")


def keypoints_xy(im: StoreImage) -> np.ndarray:
    """Keypoint coordinates as an (N,2) float64 array in work-scale pixels."""
    if not im.keypoints:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in im.keypoints], dtype=np.float64)
