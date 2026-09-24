"""End-to-end pipeline: a directory of store photos in, one panorama out."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import compose as compose_mod
from . import graph as graph_mod
from .bundle import bundle_adjust
from .evaluate import evaluate_layout
from .features import detect_all
from .imageset import load_store
from .matching import match_all_pairs


@dataclass
class PipelineConfig:
    work_max_dim: int = 1600
    detector: str = "rootsift"
    n_features: int = 12000
    ratio: float = 0.85
    ransac_thresh: float = 3.0
    min_inliers: int = 40
    min_inlier_ratio: float = 0.10
    min_ncc: float = 0.25
    min_overlap: float = 0.04
    n_hypotheses: int = 6              # sequential-RANSAC modes to consider
    # ECC polish is off by default: ablation (exp05) showed it costs ~25% of
    # runtime and changes reprojection error by 0.01 px on this data. Kept
    # available because it is the right tool if inputs get blurrier.
    ecc_refine: bool = False
    plausibility_gate: bool = True     # geometric sanity checks on each mode
    # LoFTR rescue is only attempted when the match graph comes out
    # disconnected, and only for pairs that would bridge components. On this
    # data it never fires; it exists so an unseen store with a low-texture
    # overlap degrades to "slower" rather than "partial result".
    use_deep_fallback: bool = True
    bundle: bool = True
    huber_px: float = 3.0
    render_scale: float = 1.0
    seam_scale: float = 0.25
    exposure: str = "blocks"
    seam: str = "graphcut"
    blend: str = "multiband"
    blend_strength: float = 5.0
    max_megapixels: float = 220.0
    crop: bool = True
    seam_debug_path: str | None = None   # write a which-image-won-each-pixel map


@dataclass
class PipelineReport:
    """Everything worth knowing about one run - written next to the output."""

    store: str = ""
    n_images: int = 0
    images: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    n_pairs_verified: int = 0
    components: list = field(default_factory=list)
    n_images_used: int = 0
    dropped_images: list = field(default_factory=list)
    anchor: str = ""
    left_to_right_order: list = field(default_factory=list)
    bundle_stats: dict = field(default_factory=dict)
    deep_rescued_pairs: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    output_shape: list = field(default_factory=list)
    seconds: float = 0.0


def stitch_store(images_dir: str, cfg: PipelineConfig | None = None,
                 verbose: bool = True) -> tuple[np.ndarray, PipelineReport]:
    """Stitch one store. Returns (panorama, report)."""
    cfg = cfg or PipelineConfig()
    t0 = time.time()
    rep = PipelineReport(store=os.path.basename(os.path.dirname(images_dir.rstrip("/")))
                         or images_dir)

    # 1. Load ------------------------------------------------------------------
    images = load_store(images_dir, cfg.work_max_dim)
    rep.n_images = len(images)
    rep.images = [im.name for im in images]
    if verbose:
        print(f"[1/6] loaded {len(images)} images "
              f"({images[0].full_shape[1]}x{images[0].full_shape[0]} -> "
              f"work {images[0].work_shape[1]}x{images[0].work_shape[0]})")

    if len(images) == 1:
        rep.n_images_used = 1
        rep.seconds = round(time.time() - t0, 1)
        return images[0].load_full(), rep

    # 2. Features --------------------------------------------------------------
    if verbose:
        print(f"[2/6] detecting {cfg.detector} features")
    detect_all(images, cfg.detector, cfg.n_features, verbose)

    # 3. Pairwise matching + verification ---------------------------------------
    if verbose:
        print(f"[3/6] matching {len(images)*(len(images)-1)//2} candidate pairs")
    pairs = match_all_pairs(
        images, ratio=cfg.ratio, ransac_thresh=cfg.ransac_thresh,
        min_inliers=cfg.min_inliers, min_inlier_ratio=cfg.min_inlier_ratio,
        min_ncc=cfg.min_ncc, min_overlap=cfg.min_overlap,
        n_hypotheses=cfg.n_hypotheses, ecc_refine=cfg.ecc_refine,
        plausibility_gate=cfg.plausibility_gate, verbose=verbose)

    # 3b. Deep rescue, but only where it can change the outcome. A pair that
    #     failed between two images already joined by another route is not worth
    #     re-running an expensive matcher on; a pair that would join two
    #     disconnected components is the only kind that matters.
    if cfg.use_deep_fallback:
        pairs = _deep_rescue_bridges(images, pairs, cfg, rep, verbose)

    rep.pairs = [{"i": images[p.i].short, "j": images[p.j].short, "ok": p.ok,
                  "inliers": p.n_inliers, "inlier_ratio": round(p.inlier_ratio, 3),
                  "overlap": round(p.overlap, 3), "ncc": round(p.ncc, 3),
                  "score": round(p.score, 1), "reason": p.reason} for p in pairs]
    rep.n_pairs_verified = sum(1 for p in pairs if p.ok)

    # 4. Global layout ---------------------------------------------------------
    adj = graph_mod.build_adjacency(len(images), pairs)
    comps = graph_mod.connected_components(len(images), adj)
    rep.components = [[images[i].short for i in c] for c in comps]
    if verbose:
        print(f"[4/6] match graph: {rep.n_pairs_verified} verified pairs, "
              f"{len(comps)} component(s) {rep.components}")

    comp = comps[0]
    if len(comps) > 1:
        # The assignment guarantees a single connected panorama, so a split
        # graph means verification was too strict for some pair. Say so loudly
        # and still deliver the largest component rather than nothing.
        dropped = [images[i].short for c in comps[1:] for i in c]
        rep.dropped_images = dropped
        print(f"  !! WARNING: {len(dropped)} image(s) not connected: {dropped}")

    anchor = graph_mod.choose_anchor(comp, adj)
    rep.anchor = images[anchor].short
    if verbose:
        print(f"  [anchor] {images[anchor].short} (reference frame)")
    H = graph_mod.initial_homographies(comp, adj, anchor, verbose)

    # 5. Bundle adjustment ------------------------------------------------------
    if cfg.bundle:
        if verbose:
            print("[5/6] bundle adjustment")
        H, rep.bundle_stats = bundle_adjust(
            H, pairs, anchor, huber_px=cfg.huber_px, verbose=verbose)
    elif verbose:
        print("[5/6] bundle adjustment SKIPPED")

    order = graph_mod.order_left_to_right(comp, H, images)
    rep.left_to_right_order = [images[i].short for i in order]
    rep.n_images_used = len(comp)
    rep.metrics = evaluate_layout(images, H, pairs)
    if verbose:
        print(f"  [layout] left-to-right: {' -> '.join(rep.left_to_right_order)}")
        m = rep.metrics
        if m.get("n_pairs_scored"):
            print(f"  [metrics] reproj RMS {m['reproj_rms_px']}px "
                  f"(median {m['reproj_median_px']}, p95 {m['reproj_p95_px']}) | "
                  f"overlap NCC mean {m['mean_overlap_ncc']} "
                  f"min {m['min_overlap_ncc']}")

    # 6. Compose ----------------------------------------------------------------
    if verbose:
        print("[6/6] compositing")
    compose_stats: dict = {}
    pano = compose_mod.compose(
        images, H, anchor, render_scale=cfg.render_scale, seam_scale=cfg.seam_scale,
        blend_strength=cfg.blend_strength, exposure=cfg.exposure, seam=cfg.seam,
        blend=cfg.blend, max_megapixels=cfg.max_megapixels,
        seam_debug_path=cfg.seam_debug_path, stats=compose_stats, verbose=verbose)
    rep.metrics.update(compose_stats)
    if cfg.crop:
        pano = compose_mod.crop_to_content(pano)

    rep.output_shape = list(pano.shape[:2])
    rep.seconds = round(time.time() - t0, 1)
    if verbose:
        print(f"done in {rep.seconds}s -> {pano.shape[1]}x{pano.shape[0]}")
    return pano, rep


def _deep_rescue_bridges(images, pairs, cfg, rep, verbose):
    """Retry only the failed pairs that could connect a split match graph.

    SIFT fails when a pair's overlap falls on a low-texture region - a plain
    cabinet front, a blown-out promo header - because there are simply not
    enough repeatable keypoints there. LoFTR is detector-free and finds
    correspondences in exactly those regions.

    It is expensive, so it runs only when it can change the outcome: if the
    graph is already connected, every image will be placed regardless of how
    many non-overlapping pairs "failed", and retrying them is pure cost. Only
    pairs whose endpoints sit in different components are worth attempting, and
    each rescue still has to clear the same geometric and photometric gates -
    a deep matcher supplies candidates, it does not bypass verification.
    """
    if not any(not p.ok for p in pairs):
        return pairs

    comps = graph_mod.connected_components(
        len(images), graph_mod.build_adjacency(len(images), pairs))
    if len(comps) == 1:
        if verbose:
            print("  [deep] graph already connected; rescue not needed")
        return pairs

    try:
        from .deep_matching import estimate_pair_loftr
    except ImportError as e:
        if verbose:
            print(f"  [deep] unavailable ({e}); skipping rescue")
        return pairs

    out = {(p.i, p.j): p for p in pairs}
    # Re-derive components after each success, so once an image is joined we
    # stop paying for the other candidate bridges to it.
    for _ in range(len(images)):
        comp_of = {i: k for k, c in enumerate(comps) for i in c}
        if len(comps) == 1:
            break
        bridges = [p for p in out.values()
                   if not p.ok and comp_of[p.i] != comp_of[p.j]]
        if not bridges:
            break
        # Try the most promising bridge first: the one SIFT got closest on.
        bridges.sort(key=lambda p: -p.ncc)
        progressed = False
        for p in bridges:
            if verbose:
                print(f"  [deep] bridging {images[p.i].short}->{images[p.j].short} "
                      f"(SIFT: {p.reason})")
            try:
                newp = estimate_pair_loftr(
                    images[p.i], images[p.j], min_inliers=cfg.min_inliers,
                    min_inlier_ratio=cfg.min_inlier_ratio, min_ncc=cfg.min_ncc,
                    min_overlap=cfg.min_overlap, ransac_thresh=cfg.ransac_thresh,
                    verbose=verbose)
            except Exception as e:                  # noqa: BLE001 - never fatal
                if verbose:
                    print(f"  [deep] failed: {e}")
                continue
            if newp.ok:
                out[(p.i, p.j)] = newp
                rep.deep_rescued_pairs.append(
                    f"{images[p.i].short}->{images[p.j].short}")
                if verbose:
                    print(f"  [deep] RESCUED {images[p.i].short}->{images[p.j].short}")
                progressed = True
                break
        if not progressed:
            break
        comps = graph_mod.connected_components(
            len(images), graph_mod.build_adjacency(len(images), list(out.values())))
    return list(out.values())


def run_and_save(images_dir: str, out_path: str, cfg: PipelineConfig | None = None,
                 report_path: str | None = None, jpeg_quality: int = 95,
                 verbose: bool = True) -> PipelineReport:
    """Stitch a store and write the panorama plus a JSON report."""
    pano, rep = stitch_store(images_dir, cfg, verbose)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, pano, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if verbose:
        print(f"wrote {out_path}")
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"config": asdict(cfg or PipelineConfig()),
                       "report": asdict(rep)}, f, indent=2)
        if verbose:
            print(f"wrote {report_path}")
    return rep
