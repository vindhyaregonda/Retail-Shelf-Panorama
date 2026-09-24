"""Global refinement of the layout (homography bundle adjustment).

The spanning tree gives each image a homography by *composing* pairwise
estimates. Composition multiplies error: a 1-pixel bias on each of three hops
does not stay 1 pixel, it compounds, and because a homography carries
perspective, the compounded error shows up as a fixture that is progressively
sheared and mis-scaled towards the ends of the run. It also throws away
information - the tree uses n-1 edges and ignores every other verified pair,
including the loop closures that would pin the drift down.

So we refine all homographies jointly against *every* verified correspondence,
minimising symmetric transfer error.

Why not OpenCV's `detail::BundleAdjusterRay`/`BundleAdjusterReproj`
--------------------------------------------------------------------
Those parameterise each camera as a rotation plus a focal length, i.e. they
assume the camera only *rotated* between shots. That is true for a panorama
shot from a tripod and false here: the operator walks sideways along the
aisle, so consecutive shots differ by translation. Under translation there is
no single rotation that aligns the views, and the solver either fails to
converge or converges to a compromise that bows straight shelf edges into
arcs - visibly the case in the stock-Stitcher baseline output.

Store shelves are, however, close to a *plane*. For a planar scene the mapping
between any two views is an exact homography regardless of how the camera
moved. So a homography per image is both the correct model and only 8 degrees
of freedom - the model matches the geometry instead of fighting it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .matching import PairMatch


def _pack(H_by_idx: dict[int, np.ndarray], free: list[int]) -> np.ndarray:
    """Flatten the free images' homographies into a parameter vector.

    Each homography contributes 8 numbers: it is defined only up to scale, so
    we normalise h33 = 1 and drop it. Every plausible homography here has
    h33 != 0 (h33 -> 0 means the source origin maps to infinity, which
    `homography_is_plausible` already rejects).
    """
    return np.concatenate([(H_by_idx[i] / H_by_idx[i][2, 2]).ravel()[:8] for i in free])


def _unpack(x: np.ndarray, free: list[int], fixed: dict[int, np.ndarray]
            ) -> dict[int, np.ndarray]:
    """Inverse of `_pack`, re-inserting the fixed anchor."""
    H = dict(fixed)
    for k, i in enumerate(free):
        H[i] = np.append(x[8 * k:8 * k + 8], 1.0).reshape(3, 3)
    return H


def _transfer(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N,2) points and dehomogenise."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = p[:, 2:3]
    # Guard the divide: during optimisation the solver can transiently probe
    # parameters that send points near the horizon.
    w = np.where(np.abs(w) < 1e-9, np.sign(w) * 1e-9 + 1e-12, w)
    return p[:, :2] / w


def _subsample(pm: PairMatch, max_pts: int, rng: np.random.Generator,
               grid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Cap correspondences per pair, spread spatially rather than at random.

    Inliers cluster heavily on the few textured regions (price rails, promo
    headers). Feeding all of them in would let one busy corner dominate the
    cost and leave the rest of the frame effectively unconstrained, which is
    exactly the configuration that admits a skewed fit. Bucketing on an 8x8
    grid over image i and sampling evenly across occupied buckets spreads the
    constraints over the whole overlap.
    """
    n = len(pm.pts_i)
    if n <= max_pts:
        return pm.pts_i, pm.pts_j

    p = pm.pts_i
    lo, span = p.min(axis=0), np.ptp(p, axis=0) + 1e-9
    cell = np.floor((p - lo) / span * grid).clip(0, grid - 1).astype(int)
    keys = cell[:, 1] * grid + cell[:, 0]

    buckets: dict[int, list[int]] = {}
    for idx, k in enumerate(keys):
        buckets.setdefault(int(k), []).append(idx)

    # Round-robin over buckets so sparse regions are represented too.
    for v in buckets.values():
        rng.shuffle(v)
    chosen: list[int] = []
    order = sorted(buckets)
    depth = 0
    while len(chosen) < max_pts:
        added = False
        for k in order:
            if depth < len(buckets[k]):
                chosen.append(buckets[k][depth])
                added = True
                if len(chosen) >= max_pts:
                    break
        if not added:
            break
        depth += 1
    sel = np.array(chosen)
    return pm.pts_i[sel], pm.pts_j[sel]


def bundle_adjust(H_init: dict[int, np.ndarray], pairs: list[PairMatch],
                  anchor: int, max_pts_per_pair: int = 600,
                  huber_px: float = 3.0, max_nfev: int = 200,
                  seed: int = 0, verbose: bool = True
                  ) -> tuple[dict[int, np.ndarray], dict]:
    """Jointly refine homographies to minimise symmetric transfer error.

    Args:
        H_init: image index -> homography mapping that image into the anchor frame.
        pairs:  verified pairs; only those with both endpoints in `H_init` are used.
        anchor: index held fixed at identity. Fixing one frame removes the global
                gauge freedom - otherwise any global homography applied to every
                image leaves the cost unchanged and the solve is rank deficient.

    Returns (refined homographies, stats dict with before/after RMS in pixels).
    """
    idxs = sorted(H_init)
    free = [i for i in idxs if i != anchor]
    used = [p for p in pairs if p.ok and p.i in H_init and p.j in H_init
            and len(p.pts_i) >= 4]

    stats = {"n_images": len(idxs), "n_pairs": len(used), "anchor": anchor}
    if not free or not used:
        stats.update(rms_before=0.0, rms_after=0.0, n_residual_points=0, converged=True)
        return dict(H_init), stats

    rng = np.random.default_rng(seed)
    data = []
    for pm in used:
        pi, pj = _subsample(pm, max_pts_per_pair, rng)
        # Weight a pair by its verified quality, so a marginal edge cannot pull
        # the layout as hard as a strong one. sqrt because residuals are
        # squared by the least-squares cost.
        w = float(np.sqrt(max(pm.ncc, 0.0)))
        data.append((pm.i, pm.j, pi, pj, w))
    n_pts = sum(len(d[2]) for d in data)

    fixed = {anchor: np.eye(3)}

    def residuals(x: np.ndarray) -> np.ndarray:
        H = _unpack(x, free, fixed)
        out = []
        for i, j, pi, pj, w in data:
            Hi, Hj = H[i], H[j]
            try:
                Hij = np.linalg.solve(Hj, Hi)      # i -> anchor -> j
                Hji = np.linalg.solve(Hi, Hj)      # j -> anchor -> i
            except np.linalg.LinAlgError:
                out.append(np.full(4 * len(pi), 1e6))
                continue
            # Symmetric transfer error: measuring in both image frames avoids
            # the bias of a one-directional cost, which otherwise lets the
            # solver shrink one image to cheaply reduce its own residuals.
            out.append(w * (_transfer(Hij, pi) - pj).ravel())
            out.append(w * (_transfer(Hji, pj) - pi).ravel())
        r = np.concatenate(out)
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=1e6)

    x0 = _pack(H_init, free)
    r0 = residuals(x0)
    rms_before = float(np.sqrt(np.mean(r0.reshape(-1, 2) ** 2) * 2))

    # x_scale='jac' rescales parameters by Jacobian column norms. Essential
    # here: translation entries are ~1e3 px while the perspective entries are
    # ~1e-6, so an unscaled trust region is hopelessly ill-conditioned.
    # Huber loss keeps a handful of surviving mismatches from dominating.
    res = least_squares(residuals, x0, method="trf", loss="huber",
                        f_scale=huber_px, x_scale="jac", max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)

    rms_after = float(np.sqrt(np.mean(res.fun.reshape(-1, 2) ** 2) * 2))
    H_ref = _unpack(res.x, free, fixed)
    H_ref = {i: h / h[2, 2] for i, h in H_ref.items()}

    # Refuse a refinement that made things worse (can happen if a bad pair
    # survived verification and Huber could not fully suppress it).
    improved = rms_after <= rms_before
    stats.update(rms_before=round(rms_before, 3), rms_after=round(rms_after, 3),
                 n_residual_points=n_pts, converged=bool(res.success),
                 accepted=bool(improved), nfev=int(res.nfev))
    if verbose:
        verdict = "accepted" if improved else "REJECTED (worse than init)"
        print(f"  [bundle] {len(free)} free images, {len(used)} pairs, {n_pts} points | "
              f"RMS {rms_before:.2f}px -> {rms_after:.2f}px  {verdict}")
    return (H_ref if improved else dict(H_init)), stats
