"""Loading a store's images and keeping the two resolutions straight.

The pipeline works at two scales:

  *work* scale  - a downscaled copy (default longest side 1600 px) used for
                  feature detection, matching and bundle adjustment. Matching
                  cost is quadratic in image count and roughly linear in pixel
                  count, and SIFT keypoint localisation is sub-pixel, so the
                  geometry we recover at 1600 px is as good as at 3840 px for
                  far less time.
  *full* scale  - the original pixels, loaded lazily and only for compositing,
                  so the panorama stays legible at full resolution.

Homographies are always stored in *work* pixel coordinates. `rescale_H` moves
one to another scale; keeping a single canonical coordinate frame is the main
source of sign/scale bugs in a stitcher, so it lives in exactly one place.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StoreImage:
    """One captured photo, with its work-scale copy and lazy full-res access."""

    path: str
    index: int
    work: np.ndarray                 # BGR, downscaled
    full_shape: tuple[int, int]      # (h, w) of the original file
    scale: float                     # work_px = scale * full_px
    keypoints: list = field(default_factory=list, repr=False)
    descriptors: np.ndarray | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def short(self) -> str:
        return self.name[:8]

    @property
    def work_shape(self) -> tuple[int, int]:
        return self.work.shape[:2]

    def load_full(self) -> np.ndarray:
        """Read the original pixels from disk (not cached - these are ~25 MP)."""
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"could not read {self.path}")
        return img


def load_store(images_dir: str, work_max_dim: int = 1600) -> list[StoreImage]:
    """Load every jpg/png in `images_dir`, sorted by filename for determinism."""
    paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    if not paths:
        raise FileNotFoundError(f"no images found in {images_dir}")

    out = []
    for i, p in enumerate(paths):
        full = cv2.imread(p, cv2.IMREAD_COLOR)
        if full is None:
            raise IOError(f"could not read {p}")
        h, w = full.shape[:2]
        scale = min(1.0, work_max_dim / max(h, w))
        work = (full if scale == 1.0 else
                cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        # Use the realised scale factors, not the requested one: cv2.resize
        # rounds the output size, so re-deriving from shapes keeps geometry exact.
        out.append(StoreImage(path=p, index=i, work=work, full_shape=(h, w),
                              scale=work.shape[1] / w))
    return out


def rescale_H(H: np.ndarray, src_scale: float, dst_scale: float) -> np.ndarray:
    """Re-express a homography given in `src_scale` pixels at `dst_scale` pixels.

    A homography maps points, so changing the pixel unit conjugates it by the
    scaling matrix S: H' = S H S^-1 with S = diag(r, r, 1), r = dst/src.
    """
    r = dst_scale / src_scale
    S = np.diag([r, r, 1.0])
    Hs = S @ H @ np.linalg.inv(S)
    return Hs / Hs[2, 2]


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N,2) array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def image_corners(shape: tuple[int, int]) -> np.ndarray:
    """Corners of an (h, w) image as (4,2) float, clockwise from top-left."""
    h, w = shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
