"""Objective quality metrics for a solved layout.

There is no pixel-exact ground truth to compare against - the reference
panoramas are in a different coordinate frame, so a direct pixel diff measures
nothing. What we *can* measure is self-consistency: given the final global
homographies, how well do the images agree with each other where they overlap?

Two numbers, computed with the *final* homographies rather than the pairwise
ones, so they score the layout that was actually rendered:

  reproj_px   symmetric transfer error over verified correspondences, in work
              pixels. Low = geometry is consistent. This is what makes shelf
              edges line up and stops products tearing at seams.
  overlap_ncc gradient correlation over each overlap region. Low = the images
              are misaligned or something moved. This is what catches an
              alignment that is self-consistent on features but wrong on
              pixels - the one-facing-shift failure.

Both are cheap enough to run on every stitch, which is the point: they are the
regression signal for a pipeline whose failures are otherwise only visible by
eye.
"""
from __future__ import annotations

import numpy as np

from .imageset import StoreImage
from .matching import PairMatch, photometric_ncc


def evaluate_layout(images: list[StoreImage], H: dict[int, np.ndarray],
                    pairs: list[PairMatch]) -> dict:
    """Score a solved layout by how well overlapping images agree.

    Args:
        H: image index -> homography into the common (anchor) frame.
        pairs: verified pairs, used for their correspondences and to know
               which image pairs actually overlap.
    """
    reproj, nccs, overlaps, per_pair = [], [], [], []

    for pm in pairs:
        if not pm.ok or pm.i not in H or pm.j not in H or len(pm.pts_i) < 4:
            continue
        Hi, Hj = H[pm.i], H[pm.j]
        # The relative transform implied by the *global* solution, which is not
        # the pairwise estimate once bundle adjustment has moved things.
        Hij = np.linalg.solve(Hj, Hi)

        p = np.hstack([pm.pts_i, np.ones((len(pm.pts_i), 1))]) @ Hij.T
        w = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
        err = np.linalg.norm(p[:, :2] / w - pm.pts_j, axis=1)

        ncc, ovl = photometric_ncc(images[pm.i].work, images[pm.j].work, Hij)
        reproj.append(err)
        nccs.append(ncc)
        overlaps.append(ovl)
        per_pair.append({
            "pair": f"{images[pm.i].short}->{images[pm.j].short}",
            "reproj_rms_px": round(float(np.sqrt((err ** 2).mean())), 3),
            "reproj_median_px": round(float(np.median(err)), 3),
            "overlap_ncc": round(ncc, 3), "overlap_frac": round(ovl, 3),
            "n_points": int(len(err)),
        })

    if not reproj:
        # Return the FULL key set even when nothing could be scored, so callers
        # can format the result without guarding every lookup. Returning a
        # short dict here caused a KeyError several stages downstream, far from
        # the real cause (a match graph that had come apart).
        return {"n_pairs_scored": 0, "reproj_rms_px": None,
                "reproj_median_px": None, "reproj_p95_px": None,
                "mean_overlap_ncc": None, "min_overlap_ncc": None,
                "mean_overlap_frac": None, "per_pair": []}

    allerr = np.concatenate(reproj)
    return {
        "n_pairs_scored": len(per_pair),
        # RMS is the headline because large errors are what tear a seam; the
        # median is reported alongside so a few survivors cannot hide behind it.
        "reproj_rms_px": round(float(np.sqrt((allerr ** 2).mean())), 3),
        "reproj_median_px": round(float(np.median(allerr)), 3),
        "reproj_p95_px": round(float(np.percentile(allerr, 95)), 3),
        "mean_overlap_ncc": round(float(np.mean(nccs)), 3),
        "min_overlap_ncc": round(float(np.min(nccs)), 3),
        "mean_overlap_frac": round(float(np.mean(overlaps)), 3),
        "per_pair": per_pair,
    }
