"""Experiment 04 - does multi-hypothesis RANSAC rescue the pairs that fail?

Hypothesis under test
---------------------
RANSAC returns the model with the most inliers. On a shelf of repeated facings
the *wrong* alignment - shifted by one product module - can genuinely have more
inliers than the right one, because every repeated facing votes for it. So the
correct homography is often present as a *secondary* mode that RANSAC never
reports.

Method: sequential RANSAC. Fit, record the model, remove its inliers, refit on
what remains. This enumerates the distinct alignment modes. Then score each
mode photometrically and let pixel agreement pick the winner instead of
inlier count.

Prints a table per pair of every mode found: inliers vs NCC. If the hypothesis
is right we should see pairs where the top-inlier mode has poor NCC and a
lower-inlier mode has clearly better NCC.

Usage: python experiments/exp04_multi_hypothesis.py store_1
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shelfpano.features import detect_all, keypoints_xy          # noqa: E402
from shelfpano.imageset import load_store                        # noqa: E402
from shelfpano.matching import (homography_is_plausible,         # noqa: E402
                                match_descriptors, photometric_ncc)

MIN_INLIERS_PER_MODE = 25


def enumerate_modes(im_i, im_j, n_modes=6, ransac_thresh=3.0):
    """Sequential RANSAC: list the distinct alignment modes between two images."""
    pairs = match_descriptors(im_i.descriptors, im_j.descriptors, 0.85)
    if len(pairs) < MIN_INLIERS_PER_MODE:
        return []
    xy_i, xy_j = keypoints_xy(im_i), keypoints_xy(im_j)
    src_all, dst_all = xy_i[pairs[:, 0]], xy_j[pairs[:, 1]]

    pool = np.arange(len(pairs))
    modes = []
    for _ in range(n_modes):
        if len(pool) < MIN_INLIERS_PER_MODE:
            break
        H, inl = cv2.findHomography(src_all[pool], dst_all[pool],
                                    method=cv2.USAC_MAGSAC,
                                    ransacReprojThreshold=ransac_thresh,
                                    maxIters=20000, confidence=0.9999)
        if H is None:
            break
        inl = inl.ravel().astype(bool)
        if inl.sum() < MIN_INLIERS_PER_MODE:
            break
        ok, why = homography_is_plausible(H, im_i.work_shape, im_j.work_shape)
        ncc, ovl = photometric_ncc(im_i.work, im_j.work, H) if ok else (0.0, 0.0)
        modes.append(dict(H=H, n_inliers=int(inl.sum()), ncc=ncc, overlap=ovl,
                          plausible=ok, why=why))
        pool = pool[~inl]          # remove this mode's support and look again
    return modes


def main():
    store = sys.argv[1] if len(sys.argv) > 1 else "store_1"
    images = load_store(f"stitching_assignment_data/{store}/images", 1600)
    detect_all(images, "rootsift", 12000, verbose=False)

    print(f"\n{store}: sequential-RANSAC modes per pair")
    print(f"{'pair':22s} {'mode':>4s} {'inliers':>8s} {'ovl':>6s} {'ncc':>7s}  note")
    print("-" * 78)
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            modes = enumerate_modes(images[a], images[b])
            tag = f"{images[a].short}->{images[b].short}"
            if not modes:
                print(f"{tag:22s}    -        -      -       -  no matches")
                continue
            best_ncc = max(range(len(modes)), key=lambda k: modes[k]["ncc"])
            for k, m in enumerate(modes):
                note = "" if m["plausible"] else f"implausible: {m['why']}"
                mark = ""
                if k == 0:
                    mark += " [top-inliers]"
                if k == best_ncc and m["ncc"] > 0.25:
                    mark += " [best-ncc]"
                print(f"{tag if k==0 else '':22s} {k:4d} {m['n_inliers']:8d} "
                      f"{m['overlap']:6.2f} {m['ncc']:+7.2f}  {note}{mark}")
            print()


if __name__ == "__main__":
    main()
