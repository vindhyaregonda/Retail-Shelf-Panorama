"""Experiment 05 - ablation: which pipeline stages actually earn their place?

Every stage in the pipeline was added because something failed without it. This
re-runs all five stores with one stage disabled at a time and reports the
objective metrics, so the design is justified by measurement rather than by
assertion.

Reported per configuration, summed/averaged over the five stores:
  used      images placed in the panorama, out of 19 total
  rms       symmetric reprojection error under the FINAL global homographies
  ncc       mean gradient correlation over overlaps, same homographies
  minncc    worst single overlap - the number that catches one bad fixture
            hiding behind four good ones
  secs      wall clock

Geometry-only ablations skip compositing (--geometry-only) since they cannot
change the rendered pixels; render ablations compose but the metrics are
identical by construction, so they are judged by eye from the written files.

Usage:
  python experiments/exp05_ablation.py            # geometry ablations
  python experiments/exp05_ablation.py --render   # also write render variants
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shelfpano.pipeline import PipelineConfig, stitch_store  # noqa: E402

DATA = "stitching_assignment_data"
STORES = [f"store_{i}" for i in range(1, 6)]

# Each entry: (label, config overrides, what the ablation is testing)
GEOMETRY_ABLATIONS = [
    ("full pipeline", {}, "everything on"),
    ("ORB instead of SIFT", {"detector": "orb"},
     "is a binary descriptor discriminative enough on repeated facings?"),
    ("plain SIFT (no RootSIFT)", {"detector": "sift"},
     "does Hellinger normalisation matter?"),
    ("strict ratio 0.70", {"ratio": 0.70},
     "does the textbook ratio test survive repetition?"),
    ("single-hypothesis RANSAC", {"n_hypotheses": 1},
     "does taking RANSAC's top mode lose correct pairs?"),
    ("no photometric gate", {"min_ncc": -1.0},
     "does NCC verification actually reject anything real?"),
    ("no bundle adjustment", {"bundle": False},
     "how much drift does chaining leave?"),
    ("+ ECC polish (off by default)", {"ecc_refine": True},
     "does direct photometric refinement buy accuracy?"),
    ("no LoFTR fallback", {"use_deep_fallback": False},
     "is the deep matcher load-bearing?"),
    ("fewer features (2000)", {"n_features": 2000},
     "is the wide net necessary or just slow?"),
    ("no plausibility gate", {"plausibility_gate": False},
     "do geometric sanity checks reject anything RANSAC accepts?"),
]

RENDER_ABLATIONS = [
    ("render_full", {}),
    ("render_no_seam_no_blend", {"seam": "none", "blend": "none"}),
    ("render_voronoi_feather", {"seam": "voronoi", "blend": "feather"}),
    ("render_no_exposure", {"exposure": "none"}),
    ("render_graphcut_feather", {"blend": "feather"}),
]


def run_config(overrides, geometry_only=True, out_prefix=None):
    rows = []
    for store in STORES:
        cfg = PipelineConfig(**{**dict(), **overrides})
        if geometry_only:
            # Compose is the expensive part and cannot change the metrics, so
            # ablations that only touch geometry render at a token scale.
            cfg.render_scale = 0.15
            cfg.max_megapixels = 8.0
        t0 = time.time()
        try:
            pano, rep = stitch_store(f"{DATA}/{store}/images", cfg, verbose=False)
        except Exception as e:                            # noqa: BLE001
            rows.append({"store": store, "error": str(e)[:80], "used": 0,
                         "n": 0, "secs": round(time.time() - t0, 1)})
            continue
        if out_prefix is not None:
            os.makedirs("work/ablation", exist_ok=True)
            n = store.split("_")[1]
            cv2.imwrite(f"work/ablation/{out_prefix}_store_{n}.jpg", pano,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        m = rep.metrics
        rows.append({
            "store": store, "n": rep.n_images, "used": rep.n_images_used,
            "rms": m.get("reproj_rms_px"), "ncc": m.get("mean_overlap_ncc"),
            "minncc": m.get("min_overlap_ncc"),
            "pairs": rep.n_pairs_verified,
            "secs": round(time.time() - t0, 1),
        })
    return rows


def summarise(rows):
    used = sum(r.get("used", 0) for r in rows)
    total = sum(r.get("n", 0) for r in rows)
    ok = [r for r in rows if r.get("rms") is not None]
    avg = lambda k: (sum(r[k] for r in ok) / len(ok)) if ok else float("nan")  # noqa: E731
    worst = min((r["minncc"] for r in ok if r.get("minncc") is not None),
                default=float("nan"))
    return {"used": used, "total": total, "rms": avg("rms"), "ncc": avg("ncc"),
            "minncc": worst, "secs": sum(r.get("secs", 0) for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="also write render-stage variants for visual comparison")
    a = ap.parse_args()

    os.makedirs("work/logs", exist_ok=True)
    results = []

    print(f"\n{'configuration':30s} {'used':>7s} {'rms px':>8s} {'ncc':>7s} "
          f"{'minncc':>8s} {'secs':>7s}   what it tests")
    print("-" * 118)
    for label, ov, why in GEOMETRY_ABLATIONS:
        rows = run_config(ov, geometry_only=True)
        s = summarise(rows)
        results.append({"label": label, "overrides": ov, "why": why,
                        "summary": s, "rows": rows})
        print(f"{label:30s} {s['used']:3d}/{s['total']:<3d} {s['rms']:8.2f} "
              f"{s['ncc']:7.3f} {s['minncc']:8.3f} {s['secs']:7.1f}   {why}")

    if a.render:
        print("\nrender variants (metrics identical by construction; judge visually)")
        for label, ov in RENDER_ABLATIONS:
            rows = run_config(ov, geometry_only=False, out_prefix=label)
            print(f"  wrote work/ablation/{label}_store_*.jpg "
                  f"({sum(r.get('secs',0) for r in rows):.0f}s)")

    with open("work/logs/exp05_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote work/logs/exp05_ablation.json")


if __name__ == "__main__":
    main()
