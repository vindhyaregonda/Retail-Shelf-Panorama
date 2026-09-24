"""Command line entry point:  python -m shelfpano ..."""
from __future__ import annotations

import argparse
import glob
import os
import sys

from .pipeline import PipelineConfig, run_and_save


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shelfpano",
        description="Stitch store shelf photos into one panorama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="directory of photos for ONE store")
    src.add_argument("--data-root", help="directory of store_*/images/ folders "
                                         "(stitches every store found)")

    p.add_argument("--out", help="output jpg (with --images) or directory "
                                 "(with --data-root)", default="outputs")
    p.add_argument("--report", help="write a JSON report next to each output",
                   action="store_true", default=True)
    p.add_argument("--no-report", dest="report", action="store_false")

    g = p.add_argument_group("matching")
    g.add_argument("--work-max-dim", type=int, default=1600,
                   help="longest side used for feature matching")
    g.add_argument("--detector", default="rootsift",
                   choices=["rootsift", "sift", "orb", "akaze"])
    g.add_argument("--n-features", type=int, default=12000)
    g.add_argument("--ratio", type=float, default=0.85, help="Lowe ratio threshold")
    g.add_argument("--ransac-thresh", type=float, default=3.0, help="pixels")
    g.add_argument("--min-inliers", type=int, default=40)
    g.add_argument("--min-ncc", type=float, default=0.25,
                   help="photometric gate; lower = more permissive")
    g.add_argument("--n-hypotheses", type=int, default=6,
                   help="sequential-RANSAC modes to enumerate per pair (1 = plain RANSAC)")
    g.add_argument("--ecc", dest="ecc_refine", action="store_true",
                   help="photometrically polish the winning homography (ECC); "
                        "off by default - measured neutral on this data, see WRITEUP")
    g.add_argument("--no-plausibility-gate", dest="plausibility_gate",
                   action="store_false",
                   help="skip the geometric sanity checks on each RANSAC mode")
    g.add_argument("--no-deep", dest="use_deep_fallback", action="store_false",
                   help="disable the LoFTR rescue for unmatched pairs")

    g = p.add_argument_group("geometry")
    g.add_argument("--no-bundle", dest="bundle", action="store_false",
                   help="skip global bundle adjustment (spanning tree only)")
    g.add_argument("--huber-px", type=float, default=3.0)

    g = p.add_argument_group("rendering")
    g.add_argument("--render-scale", type=float, default=1.0,
                   help="1.0 = full input resolution")
    g.add_argument("--seam-scale", type=float, default=0.25)
    g.add_argument("--exposure", default="blocks", choices=["blocks", "gain", "none"])
    g.add_argument("--seam", default="graphcut",
                   choices=["graphcut", "dp", "voronoi", "none"])
    g.add_argument("--blend", default="multiband", choices=["multiband", "feather", "none"])
    g.add_argument("--blend-strength", type=float, default=5.0)
    g.add_argument("--max-megapixels", type=float, default=220.0)
    g.add_argument("--no-crop", dest="crop", action="store_false",
                   help="keep the full canvas including black border")
    g.add_argument("--debug-seams", action="store_true",
                   help="also write a map of which source image won each pixel")
    g.add_argument("--quiet", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        work_max_dim=a.work_max_dim, detector=a.detector, n_features=a.n_features,
        ratio=a.ratio, ransac_thresh=a.ransac_thresh, min_inliers=a.min_inliers,
        min_ncc=a.min_ncc, n_hypotheses=a.n_hypotheses, ecc_refine=a.ecc_refine,
        plausibility_gate=a.plausibility_gate,
        use_deep_fallback=a.use_deep_fallback, bundle=a.bundle,
        huber_px=a.huber_px, render_scale=a.render_scale, seam_scale=a.seam_scale,
        exposure=a.exposure, seam=a.seam, blend=a.blend,
        blend_strength=a.blend_strength, max_megapixels=a.max_megapixels,
        crop=a.crop)


def _run_single(a, cfg, verbose: bool) -> int:
    """Stitch one directory of images."""
    out = a.out
    if os.path.isdir(out) or not out.lower().endswith((".jpg", ".jpeg", ".png")):
        out = os.path.join(out, "stitched.jpg")
    stem = out.rsplit(".", 1)[0]
    if a.debug_seams:
        cfg.seam_debug_path = f"{stem}_seams.jpg"
    rep = run_and_save(a.images, out, cfg,
                       report_path=f"{stem}_report.json" if a.report else None,
                       verbose=verbose)
    return 0 if rep.n_images_used == rep.n_images else 2


def _print_summary(summary) -> None:
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for name, r in summary:
        flag = "OK " if r.n_images_used == r.n_images else "PARTIAL"
        size = f"{r.output_shape[1]}x{r.output_shape[0]}" if r.output_shape else "?"
        rms = r.metrics.get("reproj_rms_px", "-")
        ncc = r.metrics.get("mean_overlap_ncc", "-")
        print(f"{name:9s} {flag:8s} {r.n_images_used}/{r.n_images} images  "
              f"{size:>12s}  rms {rms} px  ncc {ncc}  {r.seconds}s")


def _run_batch(a, cfg, verbose: bool) -> int:
    """Stitch every store_*/images directory under --data-root."""
    stores = sorted(glob.glob(os.path.join(a.data_root, "store_*")))
    if not stores:
        print(f"no store_* directories under {a.data_root}", file=sys.stderr)
        return 1

    rc, summary = 0, []
    for s in stores:
        name = os.path.basename(s)
        images_dir = os.path.join(s, "images")
        if not os.path.isdir(images_dir):
            continue
        n = name.split("_")[-1]
        out = os.path.join(a.out, f"store_{n}_stitched.jpg")
        if a.debug_seams:
            cfg.seam_debug_path = os.path.join(a.out, f"store_{n}_seams.jpg")
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        rep = run_and_save(images_dir, out, cfg,
                           report_path=os.path.join(a.out, f"store_{n}_report.json")
                           if a.report else None, verbose=verbose)
        summary.append((name, rep))
        if rep.n_images_used != rep.n_images:
            rc = 2

    _print_summary(summary)
    return rc


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    verbose = not a.quiet
    return _run_single(a, cfg, verbose) if a.images else _run_batch(a, cfg, verbose)


if __name__ == "__main__":
    raise SystemExit(main())
