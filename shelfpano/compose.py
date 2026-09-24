"""Rendering the final panorama.

Geometry is solved at work scale; pixels are rendered at full scale so price
tags stay readable. Three problems have to be solved between the two:

1. **Exposure.** Each shot is metered independently, so the same shelf is a
   different brightness in two photos and a naive paste leaves a visible step.
   Fixed with per-image, per-block, per-channel gains.
2. **Where to cut.** Every overlap contains the same products twice, at
   slightly different perspective. Averaging them ghosts the facings; picking
   an arbitrary rectangle cuts products in half. A graph-cut seam finder routes
   the boundary along paths where the two images already agree - shelf gaps,
   rail edges - so no facing is duplicated and none is sliced.
3. **How to join.** Even on a good seam there is residual misalignment. A
   multi-band blender merges low frequencies over a wide band (killing the
   exposure step) and high frequencies over a narrow one (keeping text sharp).

Steps 2 and 3 run at reduced scale and full scale respectively, which is the
standard cost split: seam *placement* does not need 25 MP to be right, but seam
*rendering* does.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .imageset import StoreImage, image_corners, warp_points


def to_render_frame(H_work: np.ndarray, scale_i: float, scale_anchor: float,
                    render_scale: float) -> np.ndarray:
    """Re-express a work-scale homography as full-res source -> render canvas.

    `H_work` maps work pixels of image i to work pixels of the anchor. With
    S = diag(s, s, 1) taking full-res pixels to work pixels for each image, the
    full-res-to-render map is  rho * S_anchor^-1 @ H_work @ S_i.

    Images are conjugated by *their own* scale rather than a shared one, so the
    maths still holds if the inputs are not all the same resolution.
    """
    S_i = np.diag([scale_i, scale_i, 1.0])
    S_a_inv = np.diag([1.0 / scale_anchor, 1.0 / scale_anchor, 1.0])
    R = np.diag([render_scale, render_scale, 1.0])
    H = R @ S_a_inv @ H_work @ S_i
    return H / H[2, 2]


def plan_canvas(images: list[StoreImage], H_work: dict[int, np.ndarray], anchor: int,
                render_scale: float = 1.0, max_megapixels: float | None = 220.0
                ) -> tuple[dict[int, np.ndarray], tuple[int, int], float]:
    """Compute per-image render homographies and the canvas size that holds them.

    Returns (H_render, (width, height), effective_render_scale). If the canvas
    would exceed `max_megapixels`, the render scale is reduced - multi-band
    blending allocates several float pyramids over the whole canvas, so an
    unbounded canvas is an out-of-memory crash rather than a slow run.
    """
    s_a = images[anchor].scale

    def extents(rs: float):
        Hs, pts = {}, []
        for i, Hw in H_work.items():
            Hs[i] = to_render_frame(Hw, images[i].scale, s_a, rs)
            pts.append(warp_points(Hs[i], image_corners(images[i].full_shape)))
        allp = np.vstack(pts)
        lo = np.floor(allp.min(axis=0))
        hi = np.ceil(allp.max(axis=0))
        return Hs, lo, hi

    Hs, lo, hi = extents(render_scale)
    w, h = hi - lo
    if max_megapixels is not None and (w * h) / 1e6 > max_megapixels:
        shrink = float(np.sqrt(max_megapixels * 1e6 / (w * h)))
        render_scale *= shrink
        print(f"  [canvas] {w:.0f}x{h:.0f} exceeds {max_megapixels:g} MP; "
              f"rendering at {render_scale:.3f}x")
        Hs, lo, hi = extents(render_scale)
        w, h = hi - lo

    # Shift the canvas so its top-left is the origin.
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], dtype=np.float64)
    return {i: T @ H for i, H in Hs.items()}, (int(round(w)), int(round(h))), render_scale


def _warp(img: np.ndarray, H: np.ndarray, canvas_wh: tuple[int, int],
          scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Warp an image into (a scaled copy of) the canvas, cropped to its own ROI.

    Warping every image across the full canvas wastes most of the work and the
    memory - each image only covers a fraction of it. We compute the image's
    bounding box in canvas space and warp only into that, returning the ROI
    corner alongside, which is exactly the (image, mask, corner) triple that
    OpenCV's seam finders and blenders consume.
    """
    if scale != 1.0:
        S = np.diag([scale, scale, 1.0])
        H = S @ H
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # The source was resized, so undo that scaling on the source side.
        H = H @ np.diag([1.0 / scale, 1.0 / scale, 1.0])
    cw, ch = int(round(canvas_wh[0] * scale)), int(round(canvas_wh[1] * scale))

    corners = warp_points(H, image_corners(img.shape[:2]))
    x0 = max(0, int(np.floor(corners[:, 0].min())))
    y0 = max(0, int(np.floor(corners[:, 1].min())))
    x1 = min(cw, int(np.ceil(corners[:, 0].max())))
    y1 = min(ch, int(np.ceil(corners[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return (np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8), (0, 0))

    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(img, T @ H, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T @ H, size,
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    # Erode by a pixel: bilinear interpolation at the warp boundary mixes in
    # the black border, and that dark fringe is otherwise blended into the seam.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return warped, mask, (x0, y0)


def compose(images: list[StoreImage], H_work: dict[int, np.ndarray], anchor: int,
            render_scale: float = 1.0, seam_scale: float = 0.25,
            blend_strength: float = 5.0, exposure: str = "blocks",
            seam: str = "graphcut", blend: str = "multiband",
            max_megapixels: float | None = 220.0, seam_debug_path: str | None = None,
            stats: dict | None = None, verbose: bool = True) -> np.ndarray:
    """Render the panorama. `H_work` maps each image into the anchor's frame.

    If `seam_debug_path` is given, also writes a false-colour map of which
    source image won each pixel. That map is the direct check on the two
    failure modes the grading cares about: a facing appearing twice shows up as
    an island of one colour inside another, and a seam slicing through a
    product shows up as a boundary crossing the middle of a fixture instead of
    running along its edge.
    """
    idxs = sorted(H_work)
    H_render, canvas_wh, eff = plan_canvas(images, H_work, anchor, render_scale,
                                           max_megapixels)
    if verbose:
        print(f"  [canvas] {canvas_wh[0]}x{canvas_wh[1]} px "
              f"({canvas_wh[0]*canvas_wh[1]/1e6:.1f} MP) at {eff:.3f}x")

    # ---- seam scale: exposure gains and seam masks -------------------------
    sc = seam_scale
    seam_imgs, seam_masks, seam_corners = [], [], []
    for i in idxs:
        w, m, c = _warp(images[i].load_full(), H_render[i], canvas_wh, sc)
        seam_imgs.append(w)
        seam_masks.append(m)
        seam_corners.append(c)

    # NOTE ON CONSTRUCTION: some cv2.detail_* classes expose no constructor to
    # Python, so `cv2.detail_NoSeamFinder()` returns a wrapper holding a NULL
    # Ptr and the first method call dereferences it - a hard SIGSEGV that takes
    # the interpreter (or a Jupyter kernel) down with no Python traceback.
    # Confirmed for NoExposureCompensator, NoSeamFinder and VoronoiSeamFinder
    # on OpenCV 4.13. Those must go through their *_createDefault factory, which
    # returns a properly constructed Ptr. GraphCut/Dp/Blocks/Gain/Feather/
    # MultiBand do declare constructors and are safe to build directly.
    # tests/test_compose_options.py exercises every selectable value so this
    # cannot regress silently.
    if exposure == "blocks":
        comp = cv2.detail_BlocksChannelsCompensator(32, 32, 2)
    elif exposure == "gain":
        comp = cv2.detail_GainCompensator()
    else:
        comp = cv2.detail.ExposureCompensator_createDefault(
            cv2.detail.ExposureCompensator_NO)
    comp.feed(seam_corners, seam_imgs, seam_masks)

    if seam == "graphcut":
        # COST_COLOR_GRAD scores a cut by colour difference *and* gradient
        # difference, which keeps the seam off high-contrast product edges
        # where a small residual misalignment would be obvious.
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    elif seam == "dp":
        finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    elif seam == "voronoi":
        finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
    else:
        finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)

    # Compensate before seam finding so the finder is not tempted to route
    # seams around brightness steps it should ignore.
    for k, i in enumerate(idxs):
        comp.apply(k, seam_corners[k], seam_imgs[k], seam_masks[k])

    seam_float = [im.astype(np.float32) for im in seam_imgs]
    found = finder.find(seam_float, seam_corners, seam_masks)
    seam_masks = list(found) if found is not None else seam_masks
    if verbose:
        print(f"  [seams] {seam} on {len(idxs)} images at {sc:.2f}x")

    # How much of the final panorama each source image actually contributes.
    # A near-zero share means the seam finder gave that image's territory away
    # entirely - harmless if it was fully redundant, but a warning sign that a
    # fixture may be represented by the wrong (e.g. more oblique) shot.
    areas = {}
    for k, i in enumerate(idxs):
        m = seam_masks[k]
        m = np.asarray(m.get() if hasattr(m, "get") else m)
        areas[images[i].name] = int((m > 0).sum())
    total = max(sum(areas.values()), 1)
    shares = {n: round(a / total, 4) for n, a in areas.items()}
    if stats is not None:
        stats["source_pixel_share"] = shares
    if verbose:
        pretty = "  ".join(f"{n[:8]}={s*100:.1f}%" for n, s in shares.items())
        print(f"  [seams] source share: {pretty}")
        for n, s in shares.items():
            if s < 0.01:
                print(f"  !! NOTE: {n[:8]} contributes {s*100:.2f}% of the "
                      f"panorama (fully covered by its neighbours)")

    if seam_debug_path:
        _write_seam_map(seam_debug_path, idxs, images, seam_imgs, seam_masks,
                        seam_corners, canvas_wh, sc)
        if verbose:
            print(f"  [seams] wrote source map to {seam_debug_path}")

    # ---- compose scale: warp full res, apply gains, blend -------------------
    blender = _make_blender(blend, canvas_wh, blend_strength)
    blender.prepare((0, 0, canvas_wh[0], canvas_wh[1]))

    for k, i in enumerate(idxs):
        full, mask, corner = _warp(images[i].load_full(), H_render[i], canvas_wh, 1.0)
        comp.apply(k, corner, full, mask)

        # Upsample the seam mask to full res. Dilating first keeps a 1-pixel
        # margin so that after upsampling neighbouring masks still touch -
        # otherwise rounding leaves hairline gaps along every seam.
        sm = cv2.dilate(seam_masks[k], np.ones((3, 3), np.uint8))
        sm = cv2.resize(sm, (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        blender.feed(full.astype(np.int16), cv2.bitwise_and(mask, sm), corner)

    result, result_mask = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if verbose:
        cov = float((result_mask > 0).mean())
        print(f"  [blend] {blend}, canvas coverage {cov*100:.1f}%")
    return result


def _write_seam_map(path: str, idxs, images, seam_imgs, seam_masks, seam_corners,
                    canvas_wh: tuple[int, int], sc: float) -> None:
    """False-colour map of which source image owns each output pixel."""
    cw, ch = int(round(canvas_wh[0] * sc)), int(round(canvas_wh[1] * sc))
    label = np.zeros((ch, cw, 3), np.uint8)
    grey = np.zeros((ch, cw), np.uint8)

    # Distinct, evenly spaced hues so adjacent regions never look similar.
    hues = np.linspace(0, 179, len(idxs), endpoint=False).astype(np.uint8)
    for k, i in enumerate(idxs):
        x0, y0 = seam_corners[k]
        m = seam_masks[k]
        m = np.asarray(m.get() if hasattr(m, "get") else m)
        h, w = m.shape[:2]
        x1, y1 = min(x0 + w, cw), min(y0 + h, ch)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = m[:y1 - y0, :x1 - x0] > 0
        colour = cv2.cvtColor(np.array([[[hues[k], 200, 255]]], np.uint8),
                              cv2.COLOR_HSV2BGR)[0, 0]
        region = label[y0:y1, x0:x1]
        region[sub] = colour
        # Overlay the actual imagery at low weight so seams can be read against
        # the shelf content rather than in the abstract.
        g = cv2.cvtColor(seam_imgs[k][:y1 - y0, :x1 - x0], cv2.COLOR_BGR2GRAY)
        gsub = grey[y0:y1, x0:x1]
        gsub[sub] = g[sub]

    vis = cv2.addWeighted(label, 0.55, cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), 0.45, 0)
    for k, i in enumerate(idxs):
        x0, y0 = seam_corners[k]
        cv2.putText(vis, images[i].short, (x0 + 12, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, vis)


def _make_blender(kind: str, canvas_wh: tuple[int, int], strength: float):
    """Multi-band blender with the band count matched to the overlap width."""
    if kind == "feather":
        b = cv2.detail_FeatherBlender()
        b.setSharpness(1.0 / max(strength, 1e-3))
        return b
    if kind == "none":
        return cv2.detail_Blender.createDefault(cv2.detail.Blender_NO, False)

    b = cv2.detail_MultiBandBlender()
    # Blend width ~ strength% of the canvas diagonal; bands = log2 of that, so
    # the coarsest band is roughly the width of the transition region. More
    # bands than that just smears colour across the whole panorama.
    width = np.sqrt(canvas_wh[0] * canvas_wh[1]) * strength / 100.0
    bands = int(np.clip(np.ceil(np.log2(max(width, 2))), 1, 7))
    b.setNumBands(bands)
    return b


def crop_to_content(img: np.ndarray, pad: int = 0) -> np.ndarray:
    """Trim fully-black border rows/columns left by the warp."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(ys) == 0:
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y0:y1, x0:x1]
