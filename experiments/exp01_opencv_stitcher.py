"""Experiment 01 - baseline: does stock cv2.Stitcher solve this?

Runs both stock modes on every store and records status + output size.

  PANORAMA mode -> rotation-only camera model (spherical/cylindrical warp,
                   focal length estimated from homographies)
  SCANS mode    -> affine model, planar/flatbed assumption

Usage:  python experiments/exp01_opencv_stitcher.py
Writes: work/logs/exp01_opencv_stitcher.json  and  work/debug/exp01_*.jpg
"""
import glob
import json
import os
import time

import cv2
import numpy as np

DATA = "stitching_assignment_data"
OUT = "work/debug"
LOG = "work/logs/exp01_opencv_stitcher.json"

STATUS = {
    0: "OK",
    1: "ERR_NEED_MORE_IMGS",
    2: "ERR_HOMOGRAPHY_EST_FAIL",
    3: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def load(store, max_dim=1600):
    """Downscale for speed - the baseline's failure modes are scale invariant."""
    imgs = []
    for f in sorted(glob.glob(f"{DATA}/{store}/images/*.jpg")):
        im = cv2.imread(f)
        s = max_dim / max(im.shape[:2])
        imgs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return imgs


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    results = []
    for store in sorted(os.path.basename(d) for d in glob.glob(f"{DATA}/store_*")):
        imgs = load(store)
        for mode_name, mode in [("PANORAMA", cv2.Stitcher_PANORAMA),
                                ("SCANS", cv2.Stitcher_SCANS)]:
            st = cv2.Stitcher_create(mode)
            t0 = time.time()
            try:
                status, pano = st.stitch(imgs)
            except cv2.error as e:
                status, pano = -1, None
                print(f"{store:8s} {mode_name:9s} EXCEPTION {e}")
            dt = time.time() - t0
            # component() is the set of images that survived matching. Stitcher
            # calls leaveBiggestComponent() internally and still returns OK, so
            # status alone hides dropped images - this is the number that matters.
            try:
                comp = [int(c) for c in st.component()]
            except cv2.error:
                comp = []
            rec = {
                "store": store, "mode": mode_name, "n_in": len(imgs),
                "n_used": len(comp), "component": comp,
                "status": int(status), "status_name": STATUS.get(status, "EXCEPTION"),
                "seconds": round(dt, 2),
                "out_shape": None if pano is None else list(pano.shape[:2]),
            }
            # Area ratio > 1 means the panorama is bigger than one input, i.e.
            # something was actually joined rather than a near-copy returned.
            if pano is not None:
                rec["area_ratio"] = round(
                    (pano.shape[0] * pano.shape[1])
                    / (imgs[0].shape[0] * imgs[0].shape[1]), 2)
                cv2.imwrite(f"{OUT}/exp01_{store}_{mode_name}.jpg", pano)
            results.append(rec)
            print(f"{store:8s} {mode_name:9s} {rec['status_name']:24s} "
                  f"used {len(comp)}/{len(imgs)}  {rec['seconds']:6.2f}s  {rec['out_shape']}")
    with open(LOG, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {LOG}")


if __name__ == "__main__":
    main()
