"""Pairwise matching and verification - the part that repetitive shelves break.

For every unordered pair of images we try to estimate the homography that maps
image A into image B, then decide whether to *believe* it. Believing wrongly is
much worse than rejecting a true pair: one bad edge in the graph drags a whole
fixture to the wrong place, whereas a missing edge usually just routes through
another image.

Three independent gates, cheapest first:

1. **Geometric** - RANSAC (MAGSAC++) inlier count and inlier ratio.
2. **Plausibility** - the warped image outline must stay a sensible convex
   quadrilateral with bounded scale change. Degenerate homographies from
   near-collinear or clustered inliers pass RANSAC happily and produce the
   folded-over garbage that generic stitchers emit.
3. **Photometric** - warp A into B and correlate the actual pixels over the
   overlap. This is the gate that catches the failure specific to this domain:
   a homography shifted by exactly one product facing has *many* RANSAC
   inliers (the facings really do repeat) but misaligns everything that does
   not repeat - price rails, promo headers, shelf furniture, the floor. Pure
   feature-count scoring cannot see that; pixel correlation can.

Gate 3 is deliberately computed on gradient magnitude rather than intensity so
that exposure differences between shots do not depress the score.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import keypoints_xy
from .imageset import StoreImage, image_corners, warp_points


@dataclass
class PairMatch:
    """A verified (or rejected) relative pose between two images.

    `H` maps homogeneous points of image `i` into image `j`:  x_j ~ H x_i.
    """

    i: int
    j: int
    H: np.ndarray | None
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    overlap: float          # fraction of j's frame covered by warped i
    ncc: float              # photometric agreement over the overlap, [-1, 1]
    pts_i: np.ndarray       # (N,2) inlier keypoints in image i
    pts_j: np.ndarray       # (N,2) corresponding inlier keypoints in image j
    ok: bool
    reason: str = ""

    @property
    def score(self) -> float:
        """Edge weight for the spanning tree: confidence in this pair.

        Inlier count is the dominant term (it is what RANSAC actually
        maximises) but is compressed by a square root so that a pair with 4000
        inliers does not outrank a pair with 800 inliers and much better pixel
        agreement. NCC multiplies rather than adds so a photometrically bad
        pair can never win on raw feature count alone.
        """
        if not self.ok:
            return 0.0
        return float(np.sqrt(self.n_inliers) * max(self.ncc, 0.0) ** 2)


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      ratio: float = 0.85) -> np.ndarray:
    """kNN match with a ratio test and a mutual-nearest-neighbour check.

    Returns an (M,2) int array of (index_in_a, index_in_b).

    The ratio threshold is loose (0.85 vs Lowe's usual 0.7-0.75) on purpose:
    with dozens of identical facings the second-nearest neighbour is often a
    genuine sibling of the correct match, so a strict ratio test discards the
    true correspondence along with the ambiguous one. The mutual check restores
    most of the precision this costs, and RANSAC absorbs the rest.
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), dtype=int)

    # FLANN KD-tree: exact brute force on 12k x 12k x 128-D is ~10x slower and
    # the approximation error is far below the ratio-test margin.
    index_params = dict(algorithm=1, trees=5)          # FLANN_INDEX_KDTREE
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=64))

    a = np.asarray(desc_a, dtype=np.float32)
    b = np.asarray(desc_b, dtype=np.float32)

    knn_ab = flann.knnMatch(a, b, k=2)
    good_ab = {}
    for pair in knn_ab:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ab[m.queryIdx] = m.trainIdx

    # Reverse direction, for the mutual consistency test.
    knn_ba = flann.knnMatch(b, a, k=2)
    good_ba = {}
    for pair in knn_ba:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ba[m.queryIdx] = m.trainIdx

    pairs = [(ia, ib) for ia, ib in good_ab.items() if good_ba.get(ib) == ia]
    return np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)


def homography_is_plausible(H: np.ndarray, shape_i: tuple[int, int],
                            shape_j: tuple[int, int],
                            max_scale: float = 6.0) -> tuple[bool, str]:
    """Reject homographies that are geometrically absurd before trusting them.

    RANSAC maximises inliers, not sanity: inliers concentrated in one corner or
    along one shelf rail admit warps that fold the image over itself or blow it
    up by 100x while still scoring well.
    """
    if H is None or not np.all(np.isfinite(H)):
        return False, "non-finite"

    corners = image_corners(shape_i)
    w = warp_points(H, corners)
    if not np.all(np.isfinite(w)):
        return False, "warped corners non-finite"

    # 1. Orientation-preserving and still convex: the cross products of
    #    consecutive edges must all share a sign. A sign flip means the quad is
    #    self-intersecting (bow-tie) or mirrored.
    e = np.roll(w, -1, axis=0) - w
    cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False, "warped outline not convex"
    if cross[0] < 0:
        return False, "warp mirrors the image"

    # 2. Bounded area change. Two shots of the same fixture from a normal
    #    capture distance differ by well under 6x in scale; anything beyond
    #    that is a degenerate fit, not a real viewpoint change.
    area_i = shape_i[0] * shape_i[1]
    area_w = 0.5 * abs(np.dot(w[:, 0], np.roll(w[:, 1], -1))
                       - np.dot(w[:, 1], np.roll(w[:, 0], -1)))
    if area_w <= 0:
        return False, "degenerate warped area"
    ratio = area_w / area_i
    if not (1.0 / max_scale) < ratio < max_scale:
        return False, f"area ratio {ratio:.2f} outside [1/{max_scale:g},{max_scale:g}]"

    # 3. Bounded anisotropy: the quad must not be stretched into a sliver. Very
    #    elongated quads come from inliers lying on a single line.
    side = np.linalg.norm(e, axis=1)
    if side.min() < 1e-6 or side.max() / side.min() > 12.0:
        return False, "warped outline degenerate (sliver)"

    # 4. The perspective row must be sane. |h31|,|h32| far above 1/diagonal
    #    means the horizon crosses the image and pixels invert through infinity.
    diag = np.hypot(*shape_i[:2])
    if np.abs(H[2, :2]).max() * diag > 2.0:
        return False, "extreme perspective (horizon inside frame)"

    return True, ""


def photometric_ncc(img_i: np.ndarray, img_j: np.ndarray, H: np.ndarray,
                    min_overlap_px: int = 2000) -> tuple[float, float]:
    """Correlate image i warped into j against j, over their overlap.

    Returns (ncc, overlap_fraction). Correlation is computed on gradient
    magnitude, which is invariant to the per-shot exposure and white-balance
    shifts that would otherwise dominate an intensity correlation.
    """
    hj, wj = img_j.shape[:2]
    warped = cv2.warpPerspective(img_i, H, (wj, hj), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(img_i.shape[:2], 255, np.uint8), H, (wj, hj),
                               flags=cv2.INTER_NEAREST)

    overlap = float((mask > 0).sum()) / (hj * wj)
    if (mask > 0).sum() < min_overlap_px:
        return 0.0, overlap

    def grad(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.2)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    a, b = grad(warped), grad(img_j)
    # Erode the mask so bilinear bleed at the warp boundary does not create
    # artificial gradient edges that correlate with nothing.
    m = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    if m.sum() < min_overlap_px:
        return 0.0, overlap

    av, bv = a[m], b[m]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-6:
        return 0.0, overlap
    return float(np.dot(av, bv) / denom), overlap


def refine_with_ecc(img_i: np.ndarray, img_j: np.ndarray, H: np.ndarray,
                    max_iters: int = 60, eps: float = 1e-6) -> np.ndarray | None:
    """Polish a homography by direct photometric alignment (ECC).

    Feature correspondences are quantised to keypoint centres, so even a
    correct homography usually sits a pixel or two off the photometric optimum.
    ECC (Evangelidis & Psarakis, PAMI 2008) maximises the enhanced correlation
    coefficient over the warp directly, which is exposure invariant by
    construction. Run at reduced scale for speed and stability, then lifted
    back. Returns None if it fails to converge; the caller keeps the original.
    """
    scale = 0.5
    small_i = cv2.resize(img_i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_j = cv2.resize(img_j, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gi = cv2.cvtColor(small_i, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gj = cv2.cvtColor(small_j, cv2.COLOR_BGR2GRAY).astype(np.float32)

    S = np.diag([scale, scale, 1.0])
    H_small = (S @ H @ np.linalg.inv(S)).astype(np.float32)
    H_small /= H_small[2, 2]

    # ECC only sees where the two images overlap; mask the rest out so empty
    # canvas does not contribute to the correlation.
    mask = cv2.warpPerspective(np.full(gi.shape, 255, np.uint8), H_small,
                               (gj.shape[1], gj.shape[0]), flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if (mask > 0).sum() < 5000:
        return None
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
        _, H_out = cv2.findTransformECC(gj, gi, H_small, cv2.MOTION_HOMOGRAPHY,
                                        crit, mask, 5)
    except cv2.error:
        return None
    H_out = np.asarray(H_out, dtype=np.float64)
    if not np.all(np.isfinite(H_out)) or abs(H_out[2, 2]) < 1e-9:
        return None
    H_full = np.linalg.inv(S) @ H_out @ S
    return H_full / H_full[2, 2]


def estimate_pair(im_i: StoreImage, im_j: StoreImage,
                  ratio: float = 0.85,
                  ransac_thresh: float = 3.0,
                  min_inliers: int = 40,
                  min_inlier_ratio: float = 0.10,
                  min_ncc: float = 0.25,
                  min_overlap: float = 0.04,
                  n_hypotheses: int = 6,
                  ecc_refine: bool = False,
                  plausibility_gate: bool = True,
                  verbose: bool = True) -> PairMatch:
    """Estimate and verify the homography mapping image i into image j.

    Uses *sequential RANSAC* rather than a single fit. RANSAC reports the model
    with the most inliers, but on a gantry of repeated facings the alignment
    shifted by one product module is supported by every repeated facing and can
    genuinely out-vote the correct one - measured on store_1, where the
    219-inlier mode scores NCC 0.21 and the correct 138-inlier mode scores 0.56.

    So we enumerate modes (fit, strip that mode's inliers, refit) and pick the
    winner by *photometric* agreement among modes that clear the geometric
    gates. Inlier count proposes; pixels decide.
    """
    reject = lambda reason, **kw: PairMatch(  # noqa: E731
        i=im_i.index, j=im_j.index, H=kw.get("H"), n_matches=kw.get("n_matches", 0),
        n_inliers=kw.get("n_inliers", 0), inlier_ratio=kw.get("inlier_ratio", 0.0),
        overlap=kw.get("overlap", 0.0), ncc=kw.get("ncc", 0.0),
        pts_i=np.zeros((0, 2)), pts_j=np.zeros((0, 2)), ok=False, reason=reason)

    pairs = match_descriptors(im_i.descriptors, im_j.descriptors, ratio)
    if len(pairs) < min_inliers:
        return reject(f"only {len(pairs)} raw matches", n_matches=len(pairs))

    xy_i, xy_j = keypoints_xy(im_i), keypoints_xy(im_j)
    src_all, dst_all = xy_i[pairs[:, 0]], xy_j[pairs[:, 1]]

    pool = np.arange(len(pairs))
    modes: list[dict] = []
    best_effort = None                     # kept only to explain a rejection

    for _ in range(max(1, n_hypotheses)):
        if len(pool) < min_inliers:
            break
        # MAGSAC++ marginalises over the inlier threshold instead of hard
        # thresholding, which is markedly more stable than plain RANSAC at the
        # low inlier ratios repeated facings produce.
        H, inl = cv2.findHomography(src_all[pool], dst_all[pool],
                                    method=cv2.USAC_MAGSAC,
                                    ransacReprojThreshold=ransac_thresh,
                                    maxIters=20000, confidence=0.9999)
        if H is None:
            break
        inl = inl.ravel().astype(bool)
        n_in = int(inl.sum())
        if n_in < min_inliers:
            break
        sel = pool[inl]
        pool = pool[~inl]                  # strip this mode, look for the next

        ok, why = ((True, "") if not plausibility_gate else
                   homography_is_plausible(H, im_i.work_shape, im_j.work_shape))
        if not ok:
            if best_effort is None:
                best_effort = (H, n_in, n_in / len(pairs), 0.0, 0.0, why)
            continue

        ncc, overlap = photometric_ncc(im_i.work, im_j.work, H)
        modes.append(dict(H=H, sel=sel, n_inliers=n_in, ncc=ncc, overlap=overlap))
        if best_effort is None or ncc > best_effort[4]:
            best_effort = (H, n_in, n_in / len(pairs), overlap, ncc, "")

    viable = [m for m in modes if m["overlap"] >= min_overlap
              and m["n_inliers"] >= min_inliers]
    if not viable:
        if best_effort is None:
            return reject("no plausible model in any mode", n_matches=len(pairs))
        H, n_in, r, ovl, ncc, why = best_effort
        return reject(why or f"ncc {ncc:.2f} < {min_ncc}", H=H, n_matches=len(pairs),
                      n_inliers=n_in, inlier_ratio=r, overlap=ovl, ncc=ncc)

    best = max(viable, key=lambda m: m["ncc"])
    n_modes = len(modes)
    # Flag the case this machinery exists for: the mode RANSAC would have
    # returned is not the mode the pixels prefer.
    overruled = bool(modes and modes[0] is not best)

    H = best["H"]
    ncc, overlap = best["ncc"], best["overlap"]
    if ecc_refine:
        H_ref = refine_with_ecc(im_i.work, im_j.work, H)
        if H_ref is not None:
            ncc_ref, ovl_ref = photometric_ncc(im_i.work, im_j.work, H_ref)
            # Only keep the polish if it actually improved pixel agreement.
            if ncc_ref > ncc and ovl_ref >= min_overlap:
                H, ncc, overlap = H_ref, ncc_ref, ovl_ref

    sel = best["sel"]
    n_in = best["n_inliers"]
    ratio_in = n_in / len(pairs)
    pm = PairMatch(i=im_i.index, j=im_j.index, H=H, n_matches=len(pairs),
                   n_inliers=n_in, inlier_ratio=ratio_in, overlap=overlap,
                   ncc=ncc, pts_i=src_all[sel], pts_j=dst_all[sel], ok=True)

    if ratio_in < min_inlier_ratio:
        pm.ok, pm.reason = False, f"inlier ratio {ratio_in:.2f} < {min_inlier_ratio}"
    elif ncc < min_ncc:
        pm.ok, pm.reason = False, f"ncc {ncc:.2f} < {min_ncc}"
    elif overruled:
        pm.reason = "ncc overruled top-inlier mode"

    if verbose:
        flag = "OK  " if pm.ok else "drop"
        print(f"  [pair] {im_i.short}->{im_j.short} {flag} "
              f"matches={len(pairs):5d} inliers={n_in:5d} ({ratio_in:.2f}) "
              f"ovl={overlap:.2f} ncc={ncc:+.2f} modes={n_modes} "
              f"score={pm.score:6.1f} {pm.reason}")
    return pm


def match_all_pairs(images: list[StoreImage], **kw) -> list[PairMatch]:
    """Estimate every unordered pair. Filenames carry no ordering information,
    so the arrangement has to be discovered from the images themselves."""
    out = []
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            out.append(estimate_pair(images[a], images[b], **kw))
    return out
