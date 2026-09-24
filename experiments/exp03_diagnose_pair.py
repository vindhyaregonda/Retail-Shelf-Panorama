"""Experiment 03 - why is a specific pair being rejected?

Renders a checkerboard overlay and a difference map for one pair so the
homography can be judged by eye instead of by a threshold. Used to decide
whether a low NCC means "wrong homography" or "correct homography, parallax".

Usage:
  python experiments/exp03_diagnose_pair.py store_1 bcd6e94e cbaf4880
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shelfpano.features import detect_all                      # noqa: E402
from shelfpano.imageset import load_store                      # noqa: E402
from shelfpano.matching import estimate_pair, photometric_ncc  # noqa: E402

OUT = "work/debug"


def checkerboard(a, b, mask, cell=120):
    """Interleave two aligned images in squares - misalignment shows as a jump
    in any structure that crosses a cell boundary."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pick = (((yy // cell) + (xx // cell)) % 2).astype(bool)
    out = np.where(pick[..., None], a, b)
    return np.where(mask[..., None] > 0, out, b)


def main():
    store, sa, sb = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    images = load_store(f"stitching_assignment_data/{store}/images", 1600)
    detect_all(images, "rootsift", 12000, verbose=False)

    im_i = next(x for x in images if x.short == sa)
    im_j = next(x for x in images if x.short == sb)

    # min_ncc=-1 so we always get the homography back, whatever it scores.
    pm = estimate_pair(im_i, im_j, min_ncc=-1.0, min_inliers=10, verbose=True)
    if pm.H is None:
        print("no homography at all")
        return

    ncc, ovl = photometric_ncc(im_i.work, im_j.work, pm.H)
    print(f"\nncc={ncc:+.3f}  overlap={ovl:.3f}  inliers={pm.n_inliers}")

    hj, wj = im_j.work.shape[:2]
    warped = cv2.warpPerspective(im_i.work, pm.H, (wj, hj))
    mask = cv2.warpPerspective(np.full(im_i.work.shape[:2], 255, np.uint8),
                               pm.H, (wj, hj), flags=cv2.INTER_NEAREST)

    tag = f"{store}_{sa}_{sb}"
    cv2.imwrite(f"{OUT}/exp03_{tag}_checker.jpg", checkerboard(warped, im_j.work, mask))

    # Per-pixel gradient disagreement, restricted to the overlap. Uniform speckle
    # over product faces = parallax. A coherent shifted copy of the whole shelf
    # structure = the homography is wrong.
    g = lambda x: cv2.magnitude(  # noqa: E731
        cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 1, 0, 3),
        cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 0, 1, 3))
    d = np.abs(g(warped) - g(im_j.work))
    d[mask == 0] = 0
    d = np.clip(d / (d.max() + 1e-6) * 255 * 3, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUT}/exp03_{tag}_diff.jpg", cv2.applyColorMap(d, cv2.COLORMAP_INFERNO))

    # Local NCC map: where in the overlap does agreement break down?
    win = 64
    a32 = g(warped)
    b32 = g(im_j.work)
    ncc_map = np.zeros((hj // win, wj // win), np.float32)
    for r in range(hj // win):
        for c in range(wj // win):
            sl = (slice(r * win, (r + 1) * win), slice(c * win, (c + 1) * win))
            if (mask[sl] > 0).mean() < 0.9:
                ncc_map[r, c] = np.nan
                continue
            u, v = a32[sl].ravel(), b32[sl].ravel()
            u, v = u - u.mean(), v - v.mean()
            den = np.linalg.norm(u) * np.linalg.norm(v)
            ncc_map[r, c] = 0 if den < 1e-6 else np.dot(u, v) / den
    valid = ncc_map[~np.isnan(ncc_map)]
    print(f"local ncc over {len(valid)} windows: "
          f"median={np.median(valid):+.2f} p10={np.percentile(valid,10):+.2f} "
          f"p90={np.percentile(valid,90):+.2f} frac>0.4={np.mean(valid>0.4):.2f}")

    vis = np.nan_to_num(ncc_map, nan=-1)
    vis = ((vis + 1) / 2 * 255).astype(np.uint8)
    vis = cv2.resize(vis, (wj, hj), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{OUT}/exp03_{tag}_nccmap.jpg", cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS))
    print(f"wrote {OUT}/exp03_{tag}_*.jpg")


if __name__ == "__main__":
    main()
