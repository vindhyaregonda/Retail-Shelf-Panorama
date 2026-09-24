"""Experiment 06 - completeness check: did any input content fail to render?

"Complete: every section of the fixture that appears in the inputs is present"
is an explicit grading criterion, and the seam finder is the stage that could
quietly violate it - it can hand an image's whole territory to its neighbours.
That is harmless when the image was fully redundant and a hole in the panorama
when it was not.

This compares, on the same canvas:
  union mask   - every pixel any input warps onto (what SHOULD be filled)
  result mask  - every pixel the blender actually wrote

and reports the difference. Any pixel in the union but not the result is
content that was captured and then lost.

Usage: python experiments/exp06_completeness.py [store_1 ...]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shelfpano.compose import _warp, plan_canvas                # noqa: E402
from shelfpano.features import detect_all                       # noqa: E402
from shelfpano.graph import (build_adjacency, choose_anchor,     # noqa: E402
                             connected_components, initial_homographies)
from shelfpano.bundle import bundle_adjust                      # noqa: E402
from shelfpano.imageset import load_store                       # noqa: E402
from shelfpano.matching import match_all_pairs                  # noqa: E402
from shelfpano.pipeline import PipelineConfig                   # noqa: E402
from shelfpano import compose as compose_mod                    # noqa: E402

DATA = "stitching_assignment_data"
SCALE = 0.2          # holes are large-scale; 0.2x keeps this cheap and honest


def solve(store, cfg):
    """Re-run the geometry half of the pipeline and return (images, H, anchor)."""
    images = load_store(f"{DATA}/{store}/images", cfg.work_max_dim)
    detect_all(images, cfg.detector, cfg.n_features, verbose=False)
    pairs = match_all_pairs(images, ratio=cfg.ratio,
                            ransac_thresh=cfg.ransac_thresh,
                            min_inliers=cfg.min_inliers,
                            min_inlier_ratio=cfg.min_inlier_ratio,
                            min_ncc=cfg.min_ncc, min_overlap=cfg.min_overlap,
                            n_hypotheses=cfg.n_hypotheses,
                            ecc_refine=cfg.ecc_refine,
                            plausibility_gate=cfg.plausibility_gate, verbose=False)
    adj = build_adjacency(len(images), pairs)
    comp = connected_components(len(images), adj)[0]
    anchor = choose_anchor(comp, adj)
    H = initial_homographies(comp, adj, anchor, verbose=False)
    if cfg.bundle:
        H, _ = bundle_adjust(H, pairs, anchor, verbose=False)
    return images, H, anchor


def main():
    stores = sys.argv[1:] or [f"store_{i}" for i in range(1, 6)]
    cfg = PipelineConfig()
    os.makedirs("work/debug", exist_ok=True)

    print(f"{'store':9s} {'union px':>12s} {'filled px':>12s} {'missing':>10s} "
          f"{'missing %':>10s}  verdict")
    print("-" * 74)
    for store in stores:
        images, H, anchor = solve(store, cfg)
        H_render, canvas_wh, _ = plan_canvas(images, H, anchor, 1.0, 220.0)

        cw = int(round(canvas_wh[0] * SCALE))
        ch = int(round(canvas_wh[1] * SCALE))
        union = np.zeros((ch, cw), np.uint8)
        for i in sorted(H_render):
            _, m, (x0, y0) = _warp(images[i].load_full(), H_render[i],
                                   canvas_wh, SCALE)
            h, w = m.shape[:2]
            x1, y1 = min(x0 + w, cw), min(y0 + h, ch)
            if x1 > x0 and y1 > y0:
                roi = union[y0:y1, x0:x1]
                np.maximum(roi, m[:y1 - y0, :x1 - x0], out=roi)

        # Render through the real compositor and see what came back non-black.
        pano = compose_mod.compose(images, H, anchor, render_scale=SCALE,
                                   seam_scale=0.5, max_megapixels=40.0,
                                   verbose=False)
        filled = (cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # Erode the union by a couple of pixels: the 1-px mask erosion in _warp
        # and the crop in compose legitimately shave the outer boundary, and
        # that rim is not a hole.
        u = cv2.erode(union, np.ones((5, 5), np.uint8))
        missing = ((u > 0) & (filled == 0))
        # Ignore specks; a real hole is a contiguous region, not resampling noise.
        missing = cv2.morphologyEx(missing.astype(np.uint8) * 255,
                                   cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        n_union = int((u > 0).sum())
        n_miss = int((missing > 0).sum())
        pct = 100.0 * n_miss / max(n_union, 1)
        verdict = "COMPLETE" if pct < 0.5 else f"!! {pct:.2f}% LOST"
        print(f"{store:9s} {n_union:12d} {int((filled>0).sum()):12d} "
              f"{n_miss:10d} {pct:9.3f}%  {verdict}")

        if n_miss > 0:
            vis = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
            vis[missing > 0] = (0, 0, 255)
            cv2.imwrite(f"work/debug/exp06_{store}_missing.jpg", vis)


if __name__ == "__main__":
    main()
