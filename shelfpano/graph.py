"""Turning verified pairs into one global layout.

Each verified pair gives a *relative* transform. To composite we need every
image expressed in one common frame. Two steps:

  1. Pick an anchor - the image whose frame the panorama will live in.
  2. Walk a maximum-weight spanning tree of the match graph out from the
     anchor, composing homographies along the way.

Why a maximum spanning tree rather than any path: composing homographies
multiplies their errors, so the route from each image to the anchor should
traverse the most trustworthy edges available and as few of them as possible.
Maximising total edge confidence does the first; the anchor choice below does
the second.

The tree only *initialises* the layout - chained homographies drift, and the
drift is fixed globally in `bundle.py` using every verified pair, including the
ones the tree left out.
"""
from __future__ import annotations

import numpy as np

from .imageset import StoreImage, image_corners, warp_points
from .matching import PairMatch


def build_adjacency(n: int, pairs: list[PairMatch]) -> dict[int, dict[int, PairMatch]]:
    """Symmetric adjacency of verified pairs, with inverted homographies added."""
    adj: dict[int, dict[int, PairMatch]] = {i: {} for i in range(n)}
    for p in pairs:
        if not p.ok or p.H is None:
            continue
        adj[p.i][p.j] = p
        Hinv = np.linalg.inv(p.H)
        adj[p.j][p.i] = PairMatch(
            i=p.j, j=p.i, H=Hinv / Hinv[2, 2], n_matches=p.n_matches,
            n_inliers=p.n_inliers, inlier_ratio=p.inlier_ratio,
            overlap=p.overlap, ncc=p.ncc, pts_i=p.pts_j, pts_j=p.pts_i, ok=True)
    return adj


def connected_components(n: int, adj: dict[int, dict[int, PairMatch]]) -> list[list[int]]:
    """Connected components of the verified-match graph, largest first."""
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def choose_anchor(comp: list[int], adj: dict[int, dict[int, PairMatch]]) -> int:
    """Pick the reference frame: the best-connected, most central image.

    Anchoring on an end image forces every other image to be reached by a long
    chain, and because a homography chain accumulates perspective, the far end
    of the panorama gets stretched into a wedge. A central anchor halves the
    longest chain and keeps the warp budget symmetric - visible in the
    reference outputs, where the middle fixture is the undistorted one.

    Ties on connectivity are broken by summed edge confidence.
    """
    best, best_key = comp[0], None
    for u in comp:
        deg = len(adj[u])
        strength = sum(p.score for p in adj[u].values())
        # Eccentricity within the component: how far the furthest image is.
        dist = _hop_distances(u, comp, adj)
        ecc = max(dist.values())
        key = (-ecc, deg, strength)      # low eccentricity first, then well-connected
        if best_key is None or key > best_key:
            best, best_key = u, key
    return best


def _hop_distances(src: int, comp: list[int],
                   adj: dict[int, dict[int, PairMatch]]) -> dict[int, int]:
    """Unweighted BFS distance from `src` to every node in its component."""
    dist = {src: 0}
    queue = [src]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return {k: v for k, v in dist.items() if k in set(comp)}


def maximum_spanning_tree(comp: list[int], adj: dict[int, dict[int, PairMatch]],
                          anchor: int) -> list[tuple[int, int]]:
    """Prim's algorithm from the anchor, maximising edge confidence.

    Returns parent->child edges in the order they were added, so composing
    homographies in this order always has the parent already resolved.
    """
    in_tree = {anchor}
    edges: list[tuple[int, int]] = []
    comp_set = set(comp)
    while len(in_tree) < len(comp_set):
        best = None
        for u in in_tree:
            for v, p in adj[u].items():
                if v in in_tree or v not in comp_set:
                    continue
                if best is None or p.score > best[2]:
                    best = (u, v, p.score)
        if best is None:
            break                      # component exhausted (shouldn't happen)
        u, v, _ = best
        in_tree.add(v)
        edges.append((u, v))
    return edges


def initial_homographies(comp: list[int], adj: dict[int, dict[int, PairMatch]],
                         anchor: int, verbose: bool = True) -> dict[int, np.ndarray]:
    """Compose tree edges into a homography per image, mapping it to the anchor.

    Result: `H[i]` maps points of image i into the anchor's coordinate frame,
    with `H[anchor] == I`.
    """
    H = {anchor: np.eye(3)}
    for u, v in maximum_spanning_tree(comp, adj, anchor):
        # adj[v][u] maps v -> u; pre-composing with u -> anchor gives v -> anchor.
        Hvu = adj[v][u].H
        Hv = H[u] @ Hvu
        H[v] = Hv / Hv[2, 2]
        if verbose:
            print(f"  [tree] {v} <- {u}   (score {adj[u][v].score:.1f})")
    return H


def order_left_to_right(comp: list[int], H: dict[int, np.ndarray],
                        images: list[StoreImage]) -> list[int]:
    """Recover the capture arrangement: sort images by warped centre x.

    Purely diagnostic - the composite does not need an order - but it is the
    most direct way to show that the layout was discovered from pixels, since
    the UUID filenames carry no ordering.
    """
    cx = {}
    for i in comp:
        c = warp_points(H[i], image_corners(images[i].work_shape)).mean(axis=0)
        cx[i] = c[0]
    return sorted(comp, key=lambda i: cx[i])
