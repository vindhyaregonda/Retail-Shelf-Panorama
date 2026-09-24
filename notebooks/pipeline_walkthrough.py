# %% [markdown]
# # shelfpano — a step-by-step walkthrough on one store
#
# This notebook runs the panorama pipeline **one stage at a time on a single
# store**, showing what each stage receives and what it produces.
#
# Set `STORE` in the setup cell to any of `store_1` … `store_5`. Nothing else is
# hard-coded to a particular store — the pair the notebook dissects in step 3 is
# **chosen automatically**, because the image filenames are capture UUIDs that
# differ per store and carry no meaning.
#
# ```
# load ──> features ──> pairwise match ──> match graph ──> bundle ──> compose
#          RootSIFT      sequential          max spanning   joint      exposure
#          12k/image     RANSAC +            tree from a    homography gains +
#          CLAHE         3 verification      central        refinement graph-cut
#                        gates               anchor                    seams +
#                        (+LoFTR rescue)                               multiband
# ```
#
# `store_1` (the default) is the most interesting one: it contains the failure
# that motivated the whole design — an image pair aligned **more convincingly by
# a wrong homography than by the right one**. That is the point of sections
# 3a–3c. Other stores exercise the same code; some of them simply do not hit
# that failure, and the notebook will say so rather than pretend.
#
# Every cell is independent of Jupyter magics and runs top-to-bottom. Total
# runtime is about 1–2 minutes on a laptop CPU.

# %%
# --- Setup -------------------------------------------------------------------
# Locate the repo root so the notebook works whether it is opened from the repo
# root or from notebooks/.
import os
import sys
from pathlib import Path

_here = Path.cwd()
REPO = next((p for p in (_here, *_here.parents) if (p / "shelfpano").is_dir()), None)
if REPO is None:
    raise RuntimeError("Could not find the repo root (no 'shelfpano/' directory above cwd)")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["figure.figsize"] = (13, 7)
plt.rcParams["image.interpolation"] = "nearest"

# MAGSAC++ is a randomised algorithm, so inlier counts and NCC values move by a
# few percent between runs. Seeding OpenCV's RNG makes this notebook reproducible
# - re-running gives the same tables. The *conclusion* below (the top-inlier mode
# is not the best-aligned mode) held on every unseeded run too; only the exact
# digits move. Comment this out if you want to convince yourself of that.
cv2.setRNGSeed(0)

STORE = "store_1"
IMAGES_DIR = REPO / "stitching_assignment_data" / STORE / "images"
assert IMAGES_DIR.is_dir(), f"missing data at {IMAGES_DIR}"

# Same settings the production pipeline uses, so what you see here is what the
# submitted panorama was built from.
WORK_MAX_DIM = 1600
N_FEATURES = 12000
RATIO = 0.85

# How many sequential-RANSAC alignment hypotheses the MATCHER considers per
# pair. This is the knob that changes the panorama. Set it to 1 to reproduce
# plain single-mode RANSAC (see "Things to try" at the end).
#
# Do not confuse it with the `n_modes` argument of the `enumerate_modes` helper
# defined in step 3: that one only controls the *illustration* - which pair gets
# dissected and how many rows the mode table has - and has no effect on the
# stitched output.
N_HYPOTHESES = 6

print("repo:", REPO)
print("cv2 :", cv2.__version__)


# %%
# --- Display helpers ---------------------------------------------------------
def show(img, title="", ax=None, size=(13, 7), cmap=None):
    """Show a BGR (OpenCV) or single-channel image with matplotlib."""
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap or "gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return ax


def show_row(imgs, titles=None, size=(15, 5), cmap=None):
    """Show several images side by side."""
    titles = titles or [""] * len(imgs)
    fig, axes = plt.subplots(1, len(imgs), figsize=size)
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        show(im, t, ax=ax, cmap=cmap)
    fig.tight_layout()
    return fig


def fit(img, max_dim=900):
    """Downscale for display only."""
    s = min(1.0, max_dim / max(img.shape[:2]))
    return img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s,
                                           interpolation=cv2.INTER_AREA)


# %% [markdown]
# ## Step 1 — Load
#
# The pipeline keeps **two resolutions** of every photo, and this contract is the
# single most bug-prone thing in a stitcher:
#
# * **work scale** (longest side 1600 px) — used for features, matching and
#   bundle adjustment. SIFT localises keypoints to sub-pixel accuracy, so the
#   geometry recovered at 1600 px is as good as at 3840 px for a fraction of the
#   cost.
# * **full scale** (the original 2160×3840) — loaded lazily, used only for the
#   final render so price tags stay readable.
#
# **Every homography in this notebook maps work pixels → work pixels.** There is
# exactly one function (`to_render_frame`) that converts to render coordinates.
# Filenames are capture UUIDs and carry no ordering information — discovering the
# arrangement is part of the job.

# %%
from shelfpano.imageset import load_store, image_corners, warp_points

images = load_store(str(IMAGES_DIR), WORK_MAX_DIM)

print(f"{'#':<3}{'name':<12}{'full (w×h)':<16}{'work (w×h)':<14}{'scale':<8}")
for im in images:
    fh, fw = im.full_shape
    wh, ww = im.work_shape
    print(f"{im.index:<3}{im.short:<12}{f'{fw}×{fh}':<16}{f'{ww}×{wh}':<14}{im.scale:.4f}")

show_row([fit(im.work, 420) for im in images],
         [f"[{im.index}] {im.short}" for im in images], size=(16, 6))
plt.show()

# %% [markdown]
# Try to work out the left-to-right order yourself — it is genuinely not obvious.
# In several stores the same promo header (the orange *"SOFTEST POUCH ON THE
# PLANET"* banner, say) appears two or three times along the run, on different
# fixtures. That repetition is exactly what trips up feature matching in step 3.

# %% [markdown]
# ## Step 2 — Features
#
# Two non-default choices, both forced by the domain:
#
# **CLAHE before detection.** Each shot is metered independently, so the same
# shelf is a different brightness in two photos. Local contrast equalisation
# makes the detector fire on the *same* structures in both.
#
# **12 000 keypoints, not the usual ~2 000.** A cigarette gantry is a grid of
# near-identical facings. The genuinely distinctive detail — price rails, promo
# headers, shelf-rail joints, scuffs — is *sparse* relative to the repetition, so
# we need a wide net to collect enough of it. The ablation shows dropping to
# 2 000 features loses an image.

# %%
# What CLAHE actually does, on a deliberately dark crop.
probe = images[1]
crop = probe.work[300:700, 200:700]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

show_row([gray, clahe], ["grayscale", "after CLAHE (local contrast equalised)"],
         size=(13, 5))
plt.show()

fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(gray.ravel(), bins=64, alpha=0.6, label="grayscale", color="#888")
ax.hist(clahe.ravel(), bins=64, alpha=0.6, label="CLAHE", color="#2a7de1")
ax.set_title("CLAHE spreads the histogram, exposing low-contrast shelf structure")
ax.legend()
plt.show()

# %%
from shelfpano.features import detect_all, keypoints_xy

detect_all(images, "rootsift", N_FEATURES, verbose=False)
for im in images:
    print(f"{im.short}: {len(im.keypoints):,} keypoints, "
          f"descriptors {im.descriptors.shape}")

# Where do the keypoints land on each input?
fig, axes = plt.subplots(1, len(images), figsize=(16, 7))
for ax, im in zip(axes, images):
    xy = keypoints_xy(im)
    resp = np.array([k.response for k in im.keypoints])
    show(im.work, f"{im.short}\n{len(xy):,} keypoints", ax=ax)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=resp, cmap="autumn", alpha=0.40)
fig.suptitle("RootSIFT keypoints on every input (colour = response strength)",
             fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# Notice the keypoints cluster on **text and price rails**, and thin out over the
# flat cabinet doors at the bottom. That density map is the real constraint on
# this problem: alignment information is concentrated in a small fraction of the
# frame.

# %% [markdown]
# ## Step 3 — Pairwise matching
#
# Filenames are capture UUIDs, so **which pair is the interesting one differs per
# store** and cannot be hard-coded. We scan every candidate pair, enumerate its
# alignment modes, and pick the pair that best demonstrates the problem this
# pipeline exists to solve: one where the mode with the most **inliers** is *not*
# the mode the **pixels** prefer.
#
# If no pair in this store shows that disagreement, we fall back to the strongest
# pair and say so — the phenomenon is real but not universal, and the notebook
# should not pretend otherwise.

# %%
from shelfpano.matching import (match_descriptors, homography_is_plausible,
                                photometric_ncc)

MIN_INLIERS = 40

# Force a specific pair by short name, e.g. ("bcd6e94e", "cbaf4880") for store 1.
# Leave as None to auto-select.
PAIR_OVERRIDE = None


def enumerate_modes(im_i, im_j, n_modes=6, thresh=3.0):
    """Sequential RANSAC: list the distinct alignment modes between two images.

    Fit, record the model, strip its inliers, refit on what remains. Plain
    RANSAC only ever reports mode 0.
    """
    pi = match_descriptors(im_i.descriptors, im_j.descriptors, RATIO)
    if len(pi) < MIN_INLIERS:
        return []
    xi, xj = keypoints_xy(im_i), keypoints_xy(im_j)
    s_all, d_all = xi[pi[:, 0]], xj[pi[:, 1]]
    pool = np.arange(len(pi))
    modes = []
    for _ in range(n_modes):
        if len(pool) < MIN_INLIERS:
            break
        H, inl = cv2.findHomography(s_all[pool], d_all[pool],
                                    method=cv2.USAC_MAGSAC,
                                    ransacReprojThreshold=thresh,
                                    maxIters=20000, confidence=0.9999)
        if H is None:
            break
        inl = inl.ravel().astype(bool)
        if inl.sum() < MIN_INLIERS:
            break
        sel, pool = pool[inl], pool[~inl]
        ok, why = homography_is_plausible(H, im_i.work_shape, im_j.work_shape)
        ncc, ovl = photometric_ncc(im_i.work, im_j.work, H) if ok else (0.0, 0.0)
        modes.append(dict(H=H, n_inliers=int(inl.sum()), ncc=ncc, overlap=ovl,
                          plausible=ok, why=why, sel=sel))
    return modes


def pick_demo_pair(imgs, min_gap=0.05):
    """Pick the most instructive pair, and the mode list that goes with it.

    Preference 1: a pair where argmax(inliers) != argmax(NCC) by a clear margin
                  - the repetition failure, which is the point of this section.
    Preference 2: whichever plausible pair has the most inliers.
    """
    best_demo = best_any = None
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            modes = enumerate_modes(imgs[a], imgs[b])
            if not modes or not any(m["plausible"] for m in modes):
                continue
            top_ncc = max(range(len(modes)), key=lambda k: modes[k]["ncc"])
            gap = modes[top_ncc]["ncc"] - modes[0]["ncc"]
            if best_any is None or modes[0]["n_inliers"] > best_any[0]:
                best_any = (modes[0]["n_inliers"], imgs[a], imgs[b], modes)
            if top_ncc != 0 and gap > min_gap:
                if best_demo is None or gap > best_demo[0]:
                    best_demo = (gap, imgs[a], imgs[b], modes)
    if best_demo:
        _, ia, ib, modes = best_demo
        return ia, ib, modes, "inliers and pixels DISAGREE on this pair"
    if best_any:
        _, ia, ib, modes = best_any
        return ia, ib, modes, "no pair shows the disagreement; strongest pair shown"
    raise RuntimeError("no plausible pair found in this store")


if PAIR_OVERRIDE:
    A = next(im for im in images if im.short == PAIR_OVERRIDE[0])
    B = next(im for im in images if im.short == PAIR_OVERRIDE[1])
    modes = enumerate_modes(A, B)
    why_pair = "forced via PAIR_OVERRIDE"
else:
    print(f"scanning {len(images)*(len(images)-1)//2} candidate pairs ...")
    A, B, modes, why_pair = pick_demo_pair(images)

print(f"\ndemo pair: {A.short} -> {B.short}   ({why_pair})")

# %% [markdown]
# ## Step 3a — Raw descriptor matching
#
# We match with a **loose ratio threshold (0.85**, vs Lowe's usual 0.7–0.75**)**
# plus a mutual nearest-neighbour check. Loose on purpose: with dozens of
# identical facings the second-nearest neighbour is often a genuine sibling of
# the correct match, so a strict ratio test throws away the true correspondence
# along with the ambiguous one. Precision is recovered later by geometry and
# pixels, not here.

# %%
pairs_idx = match_descriptors(A.descriptors, B.descriptors, RATIO)
print(f"{A.short} -> {B.short}: {len(pairs_idx):,} mutual matches after ratio test")

xy_a, xy_b = keypoints_xy(A), keypoints_xy(B)
src, dst = xy_a[pairs_idx[:, 0]], xy_b[pairs_idx[:, 1]]


def draw_matches(im_a, im_b, pts_a, pts_b, n=120, seed=0, title=""):
    """Stack two images side by side and draw a random sample of matches."""
    ha, wa = im_a.shape[:2]
    hb, wb = im_b.shape[:2]
    canvas = np.zeros((max(ha, hb), wa + wb, 3), np.uint8)
    canvas[:ha, :wa] = im_a
    canvas[:hb, wa:wa + wb] = im_b
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pts_a), size=min(n, len(pts_a)), replace=False)
    for k in sel:
        pa = tuple(np.round(pts_a[k]).astype(int))
        pb = tuple(np.round(pts_b[k] + [wa, 0]).astype(int))
        colour = tuple(int(v) for v in rng.integers(60, 255, 3))
        cv2.line(canvas, pa, pb, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 3, colour, -1)
        cv2.circle(canvas, pb, 3, colour, -1)
    show(fit(canvas, 1500), title, size=(16, 7))


draw_matches(A.work, B.work, src, dst,
             title=f"{A.short} -> {B.short}: 120 of {len(src):,} raw matches "
                   "(many are wrong — note the crossing lines)")
plt.show()

# %% [markdown]
# The crossing lines are the tell. A correct set of matches between two
# side-by-side photos should be a roughly parallel bundle; every line that
# crosses the others is a facing matched to *a different copy of the same
# facing*. Sorting that out is RANSAC's job — and this is where it can go wrong.

# %% [markdown]
# ## Step 3b — The core problem: RANSAC can pick the wrong answer
#
# RANSAC returns the model with the most **inliers**. On a shelf of repeated
# facings, the alignment shifted by exactly one product module is supported by
# *every repeated facing* — and can genuinely out-vote the correct alignment.
# RANSAC is not malfunctioning; it is correctly maximising a criterion that
# happens to be wrong here.
#
# So instead of accepting one answer we use the **sequential RANSAC** defined
# above to enumerate the distinct alignment *modes*, then score each mode
# **photometrically** — warp one image into the other and correlate the actual
# pixels — and let that decide.

# %%
print(f"Sequential-RANSAC modes for {A.short} -> {B.short}\n")
print(f"{'mode':<6}{'inliers':>9}{'overlap':>10}{'NCC':>8}   note")
print("-" * 62)
for k, m in enumerate(modes):
    note = "" if m["plausible"] else f"rejected: {m['why']}"
    if k == 0:
        note = ("<- what plain RANSAC returns  " + note).strip()
    print(f"{k:<6}{m['n_inliers']:>9}{m['overlap']:>10.2f}{m['ncc']:>8.2f}   {note}")

best = max(range(len(modes)), key=lambda k: modes[k]["ncc"])
print(f"\nMost inliers : mode 0  ({modes[0]['n_inliers']} inliers, "
      f"NCC {modes[0]['ncc']:.2f})")
print(f"Best pixels  : mode {best}  ({modes[best]['n_inliers']} inliers, "
      f"NCC {modes[best]['ncc']:.2f})")
if best != 0:
    print("\n=> The two criteria DISAGREE. Inlier count would have chosen a")
    print("   worse-aligned homography; photometric scoring rescues this pair.")
else:
    print("\n=> They agree on this pair: the top-inlier mode is also the")
    print("   best-aligned one. Mode enumeration costs little and changes")
    print("   nothing here — it is insurance for the pairs where they diverge.")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ks = np.arange(len(modes))
ax[0].bar(ks, [m["n_inliers"] for m in modes], color="#c0392b")
ax[0].set_title("inlier count — what RANSAC maximises")
ax[0].set_xlabel("mode")
ax[1].bar(ks, [m["ncc"] for m in modes], color="#2a7de1")
ax[1].axhline(0.25, ls="--", c="k", lw=1, label="acceptance gate (0.25)")
ax[1].set_title("photometric NCC — what actually measures alignment")
ax[1].set_xlabel("mode")
ax[1].legend()
for a in ax:
    a.set_xticks(ks)
fig.suptitle("The two criteria rank the modes differently — this is the whole problem",
             fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# Note *which* mechanism is doing the work, because it is subtler than "a
# threshold caught it".
#
# The 0.25 NCC gate is a **floor for the winner**; it is not what rejects the
# impostor. Selection is **argmax over modes**. Depending on the run, the
# top-inlier mode may land above or below that floor — MAGSAC++ is randomised, so
# the exact digits in the table move a little — but either way what decides the
# pair is that *a different mode scores higher*.
#
# That distinction matters: a single-hypothesis RANSAC with a photometric gate
# bolted on would not be enough, because whenever the wrong mode happens to clear
# the floor there is nothing to compare it against. **You have to enumerate the
# alternatives before you can prefer one.**

# %% [markdown]
# ## Step 3c — Seeing the difference: checkerboard overlay
#
# Numbers are one thing; here is the same disagreement in pixels. We warp image
# A into image B's frame and interleave the two in squares. **If the alignment is
# right, structures run continuously across every square boundary.** If it is
# wrong, they jump.
#
# This visualisation is the single most useful debugging tool in the project. It
# is what stopped me "fixing" store 1 by lowering the acceptance threshold from
# 0.25 to 0.20 — which would have produced a full 4/4 panorama that was visibly
# broken.

# %%
def checkerboard(imA_warped, imB, mask, cell=110):
    """Interleave two aligned images in squares."""
    h, w = imB.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pick = (((yy // cell) + (xx // cell)) % 2).astype(bool)
    out = np.where(pick[..., None], imA_warped, imB)
    return np.where(mask[..., None] > 0, out, imB)


def overlay(H, imA, imB):
    h, w = imB.shape[:2]
    warped = cv2.warpPerspective(imA, H, (w, h))
    mask = cv2.warpPerspective(np.full(imA.shape[:2], 255, np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST)
    return checkerboard(warped, imB, mask)


if best != 0:
    show_row(
        [fit(overlay(modes[0]["H"], A.work, B.work), 760),
         fit(overlay(modes[best]["H"], A.work, B.work), 760)],
        [f"mode 0 — MOST INLIERS ({modes[0]['n_inliers']}), "
         f"NCC {modes[0]['ncc']:.2f}   ✗ WRONG",
         f"mode {best} — fewer inliers ({modes[best]['n_inliers']}), "
         f"NCC {modes[best]['ncc']:.2f}   ✓ CORRECT"],
        size=(15, 11))
    plt.show()
else:
    # The criteria agreed on this pair, so there is no wrong/right contrast to
    # draw - show the winning alignment on its own. Every runner-up mode is
    # printed below so you can inspect them by hand.
    show(fit(overlay(modes[0]["H"], A.work, B.work), 820),
         f"mode 0 — most inliers AND best NCC ({modes[0]['n_inliers']} inliers, "
         f"NCC {modes[0]['ncc']:.2f})  ✓ correctly aligned", size=(9, 12))
    plt.show()
    for k in range(1, len(modes)):
        print(f"  runner-up mode {k}: {modes[k]['n_inliers']} inliers, "
              f"NCC {modes[k]['ncc']:+.2f}"
              + ("" if modes[k]["plausible"] else f"  ({modes[k]['why']})"))
    print("\n  To see the failure this machinery exists for, set STORE = 'store_1'.")

# %% [markdown]
# On **store 1**, where the two criteria disagree, the contrast is stark:
#
# **Left (what plain RANSAC gives you):** the orange banner text and the product
# rows are *duplicated at a horizontal offset* — you can read fragments of
# "SOFTEST" twice. The shelf rails line up, which is why it scores so many
# inliers, but the contents are shifted by about one module.
#
# **Right (what photometric arbitration gives you):** "SOFTEST POUCH ON THE
# PLANET" reads once, continuously. Rails, cans and price tags all run straight
# through the square boundaries.
#
# On a store where the criteria agree you get a single panel instead, and it
# should read continuously across every square boundary.
#
# Same image pair, same feature set — the only difference is which criterion
# chose the answer.

# %% [markdown]
# ## Step 3d — The three verification gates
#
# Every candidate mode passes through three gates, cheapest first. A pair only
# joins the graph if it clears all three.

# %%
# Gate 2 (plausibility): RANSAC will happily return folded or exploded warps
# when inliers cluster in one corner. These are rejected before we ever look at
# pixels, because the warped outline is geometrically absurd.
shape = A.work_shape
demos = {
    "sane translation": np.array([[1, 0, 300.], [0, 1, 5.], [0, 0, 1.]]),
    "mirrored": np.array([[-1, 0, 1600.], [0, 1, 0.], [0, 0, 1.]]),
    "40x blow-up": np.diag([40., 40., 1.]),
    "horizon in frame": np.array([[1, 0, 0.], [0, 1, 0.], [2e-3, 0, 1.]]),
}
for name, H in demos.items():
    ok, why = homography_is_plausible(H, shape, shape)
    print(f"{name:<20} {'ACCEPT' if ok else 'REJECT':<8}{why}")

# %%
# Gate 3 (photometric): NCC is computed on GRADIENT MAGNITUDE, not intensity,
# so a brightness difference between two shots cannot depress the score.
def grad(bgr):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                         (0, 0), 1.2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))


h, w = B.work_shape
for label, H in [("WRONG mode 0", modes[0]["H"]), (f"CORRECT mode {best}", modes[best]["H"])]:
    warped = cv2.warpPerspective(A.work, H, (w, h))
    m = cv2.erode(cv2.warpPerspective(np.full(A.work_shape, 255, np.uint8), H, (w, h),
                                      flags=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8))
    d = np.abs(grad(warped) - grad(B.work))
    d[m == 0] = 0
    vis = cv2.applyColorMap(
        np.clip(d / (d.max() + 1e-6) * 255 * 3, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO)
    ncc, ovl = photometric_ncc(A.work, B.work, H)
    show(fit(vis, 620), f"{label}: gradient disagreement (bright = mismatch)\n"
                        f"NCC {ncc:+.2f}, overlap {ovl:.2f}", size=(7, 10))
    plt.show()

# %% [markdown]
# The wrong mode lights up in coherent *structured* bands — whole shelf rows
# disagreeing — while the correct one shows only diffuse speckle from parallax on
# protruding products. Structured disagreement means wrong geometry; speckle
# means the model is right and the scene simply is not perfectly planar.

# %%
# Now run the real matcher over ALL pairs. It does everything above internally.
from shelfpano.matching import match_all_pairs

all_pairs = match_all_pairs(images, ratio=RATIO, min_ncc=0.25,
                            n_hypotheses=N_HYPOTHESES, verbose=False)

print(f"{'pair':<24}{'verdict':<9}{'inliers':>8}{'ovl':>7}{'NCC':>7}{'score':>8}  reason")
print("-" * 88)
for p in all_pairs:
    print(f"{images[p.i].short + ' -> ' + images[p.j].short:<24}"
          f"{'ACCEPT' if p.ok else 'reject':<9}{p.n_inliers:>8}{p.overlap:>7.2f}"
          f"{p.ncc:>7.2f}{p.score:>8.1f}  {p.reason}")

n_ok = sum(p.ok for p in all_pairs)
print(f"\n{n_ok} of {len(all_pairs)} candidate pairs verified")

# %% [markdown]
# Most pairs are *correctly* rejected — with 4 images there are 6 possible pairs
# but only 3 fixtures actually overlap. Rejecting a non-overlapping pair is the
# right answer, not a failure.

# %%
# The surviving pairs, drawn on the images themselves. These are the only links
# holding the panorama together — everything downstream is built on them.
for p in all_pairs:
    if not p.ok:
        continue
    draw_matches(images[p.i].work, images[p.j].work, p.pts_i, p.pts_j, n=90,
                 title=f"VERIFIED  {images[p.i].short} -> {images[p.j].short}   "
                       f"{p.n_inliers} inliers, NCC {p.ncc:.2f}, "
                       f"overlap {p.overlap:.2f}")
    plt.show()

# %% [markdown]
# ## Step 4 — The match graph
#
# Verified pairs form a graph. We need every image expressed in one common frame,
# so we pick an **anchor** and walk a **maximum-weight spanning tree** out from
# it, composing homographies along the way.
#
# The anchor is chosen for **low eccentricity** — the image whose furthest
# neighbour is closest. Anchoring on an end image forces long chains, and because
# homographies accumulate perspective, the far end of the panorama gets stretched
# into a wedge. A central anchor halves the longest chain.

# %%
import networkx as nx
from shelfpano.graph import (build_adjacency, connected_components, choose_anchor,
                             maximum_spanning_tree, initial_homographies,
                             order_left_to_right)

adj = build_adjacency(len(images), all_pairs)
comps = connected_components(len(images), adj)
print("connected components:", [[images[i].short for i in c] for c in comps])

# A split graph is a RESULT worth seeing, not a reason to stop - it is exactly
# what the `n_modes=1` experiment below is meant to demonstrate. We carry on
# with the largest component, like the production pipeline does, and say loudly
# what was lost. (Raising here would also strand `comp`/`anchor` at their values
# from a previous run, so every later cell would silently mix fresh pairs with
# stale geometry and fail somewhere far more confusing.)
comp = comps[0]
if len(comps) > 1:
    lost = [images[i].short for c in comps[1:] for i in c]
    print(f"\n  !! {len(lost)} image(s) could NOT be connected: {lost}")
    print(f"  !! Continuing with the largest component ({len(comp)} of "
          f"{len(images)} images) - the panorama below will be PARTIAL.")
    print("  !! If you changed a setting above, this is the setting's effect.")
anchor = choose_anchor(comp, adj)
tree_edges = maximum_spanning_tree(comp, adj, anchor)
print(f"\nanchor: {images[anchor].short}")
print("spanning tree edges (parent -> child):")
for u, v in tree_edges:
    print(f"  {images[u].short} -> {images[v].short}   (score {adj[u][v].score:.1f})")

# %%
G = nx.Graph()
for i in comp:
    G.add_node(i, label=images[i].short)
for p in all_pairs:
    if p.ok:
        G.add_edge(p.i, p.j, weight=p.score, ncc=p.ncc)

pos = nx.spring_layout(G, seed=3, weight="weight")
tree_set = {frozenset(e) for e in tree_edges}
in_tree = [e for e in G.edges() if frozenset(e) in tree_set]
off_tree = [e for e in G.edges() if frozenset(e) not in tree_set]

fig, ax = plt.subplots(figsize=(9, 6))
nx.draw_networkx_edges(G, pos, edgelist=in_tree, width=3.0, edge_color="#2a7de1", ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=off_tree, width=1.2, style="dashed",
                       edge_color="#999", ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=2600, ax=ax,
                       node_color=["#e67e22" if i == anchor else "#dfe6ec" for i in G.nodes()])
nx.draw_networkx_labels(G, pos, {i: images[i].short for i in G.nodes()}, font_size=8, ax=ax)
nx.draw_networkx_edge_labels(
    G, pos, {e: f"{G.edges[e]['weight']:.1f}" for e in G.edges()}, font_size=7, ax=ax)
ax.set_title("Match graph — orange = anchor, solid blue = spanning tree, "
             "dashed = verified but unused by the tree\n"
             "(edge labels are confidence scores)", fontsize=10)
ax.axis("off")
plt.show()

# %% [markdown]
# The **dashed edges matter**: the tree uses only *n−1* edges, but those extra
# verified pairs are extra constraints. Bundle adjustment in the next step uses
# *all* of them, which is how it pins down drift the tree cannot see.

# %%
H_tree = initial_homographies(comp, adj, anchor, verbose=False)
order = order_left_to_right(comp, H_tree, images)
print("Discovered left-to-right arrangement (from pixels, not filenames):")
print("   " + "  ->  ".join(images[i].short for i in order))

show_row([fit(images[i].work, 380) for i in order],
         [f"{images[i].short}" + ("  (anchor)" if i == anchor else "") for i in order],
         size=(16, 6))
plt.show()

# %% [markdown]
# ## Step 5 — Bundle adjustment
#
# The tree gives each image a homography by **composing** pairwise estimates, and
# composition multiplies error. We now refine all homographies jointly against
# **every** verified correspondence, minimising symmetric transfer error.
#
# Why not OpenCV's bundle adjusters: they parameterise each camera as a
# *rotation + focal length*, i.e. they assume the camera only rotated. Here the
# operator walks sideways along the aisle, so shots differ by **translation**.
# A shelf front is near-planar, and for a plane the view-to-view mapping is an
# exact homography regardless of camera motion — so a homography per image is
# both correct and only 8 DOF.

# %%
from shelfpano.bundle import bundle_adjust
from shelfpano.evaluate import evaluate_layout


def all_residuals(H, pairs):
    """Per-correspondence reprojection error under a given global solution."""
    out = []
    for p in pairs:
        if not p.ok or p.i not in H or p.j not in H or len(p.pts_i) < 4:
            continue
        Hij = np.linalg.solve(H[p.j], H[p.i])
        q = np.hstack([p.pts_i, np.ones((len(p.pts_i), 1))]) @ Hij.T
        out.append(np.linalg.norm(q[:, :2] / q[:, 2:3] - p.pts_j, axis=1))
    return np.concatenate(out)


res_before = all_residuals(H_tree, all_pairs)
H_ba, stats = bundle_adjust(H_tree, all_pairs, anchor, verbose=False)
res_after = all_residuals(H_ba, all_pairs)

print(f"spanning tree only : RMS {np.sqrt((res_before**2).mean()):.3f} px   "
      f"median {np.median(res_before):.3f}   p95 {np.percentile(res_before,95):.3f}")
print(f"after bundle adjust: RMS {np.sqrt((res_after**2).mean()):.3f} px   "
      f"median {np.median(res_after):.3f}   p95 {np.percentile(res_after,95):.3f}")
print(f"\nsolver: {stats['n_pairs']} pairs, {stats['n_residual_points']} points, "
      f"accepted={stats['accepted']}")

fig, ax = plt.subplots(figsize=(10, 4))
bins = np.linspace(0, np.percentile(res_before, 99), 60)
ax.hist(res_before, bins=bins, alpha=0.6, label="spanning tree only", color="#c0392b")
ax.hist(res_after, bins=bins, alpha=0.6, label="after bundle adjustment", color="#2a7de1")
ax.set_xlabel("reprojection error (work px)")
ax.set_ylabel("correspondences")
ax.set_title("Bundle adjustment pulls the error distribution left")
ax.legend()
plt.show()

# %% [markdown]
# The gain is modest here (short chains — only 4 images), and the ablation in the
# write-up says the same across all five stores: 1.46 → 1.42 px. Reported
# honestly rather than oversold; its value grows with longer runs and it is
# nearly free.

# %%
# Where does each image land? Draw the warped outlines in the anchor's frame.
fig, ax = plt.subplots(figsize=(12, 7))
colours = plt.cm.tab10(np.linspace(0, 1, 10))
for k, i in enumerate(order):
    for H, style, lab in [(H_tree, "--", "tree"), (H_ba, "-", "bundled")]:
        quad = warp_points(H[i], image_corners(images[i].work_shape))
        quad = np.vstack([quad, quad[:1]])
        ax.plot(quad[:, 0], quad[:, 1], style, color=colours[k], lw=2 if style == "-" else 1,
                label=f"{images[i].short} ({lab})" if k < 99 else None)
    c = warp_points(H_ba[i], image_corners(images[i].work_shape)).mean(axis=0)
    ax.text(c[0], c[1], images[i].short, ha="center", fontsize=8, color=colours[k])
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("Image footprints in the anchor frame — dashed = spanning tree, "
             "solid = after bundle adjustment", fontsize=10)
ax.legend(fontsize=6, ncol=2, loc="upper right")
plt.show()

# %% [markdown]
# Note the **fanning**: the outer images are trapezoids, not rectangles. That is
# real perspective, not a bug — it is the same wedge shape visible in the
# provided reference panoramas, and it is why the anchor is chosen centrally.

# %% [markdown]
# ## Step 5b — What the homography actually does to each photo
#
# Outlines are abstract. Here is each **input photo before and after** its
# homography is applied — the same pixels, re-projected into the anchor's frame.
# This is the transformation that turns four separate snapshots into four pieces
# that fit together.

# %%
from shelfpano.compose import plan_canvas

PREVIEW = 0.28  # render scale for these illustrations


def canvas_for(H_work, render_scale=PREVIEW):
    """Plan the canvas and return per-image *work-pixel* -> canvas homographies."""
    H_render, canvas_wh, _ = plan_canvas(images, H_work, anchor, render_scale, 60.0)
    # H_render maps FULL-res source pixels to the canvas. Our work copies are
    # `scale` times smaller, so undo that on the source side to warp them directly.
    H_from_work = {i: H_render[i] @ np.diag([1 / images[i].scale,
                                             1 / images[i].scale, 1.0])
                   for i in H_render}
    return H_from_work, canvas_wh


def warp_onto_canvas(img, H, canvas_wh):
    """Warp one image onto the full canvas; returns (image, mask)."""
    w, h = canvas_wh
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST)
    return warped, mask


H_canvas, canvas_wh = canvas_for(H_ba)
print(f"canvas: {canvas_wh[0]} x {canvas_wh[1]} px")

for i in order:
    warped, _ = warp_onto_canvas(images[i].work, H_canvas[i], canvas_wh)
    tag = f"{images[i].short}" + ("  (ANCHOR — barely changes)" if i == anchor else "")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    show(fit(images[i].work, 400), f"BEFORE — original photo\n{tag}", ax=axes[0])
    show(fit(warped, 800), "AFTER — warped into the common frame\n"
                           "(black = canvas this photo does not cover)", ax=axes[1])
    fig.tight_layout()
    plt.show()

# %% [markdown]
# The anchor is nearly unchanged (it *defines* the frame). The others are
# squeezed and sheared into trapezoids by increasing amounts the further they sit
# from the anchor — that is the accumulated perspective, and it is precisely why
# anchoring centrally matters.

# %% [markdown]
# ## Step 5c — Watch them stack up
#
# Now place the warped photos onto one canvas, **one at a time**, in the
# discovered left-to-right order. This is the panorama assembling itself.
#
# Deliberately a *dumb* paste — last image wins, no exposure matching, no seam
# finding, no blending — so you can see exactly what the remaining stages have
# left to fix.

# %%
acc = np.zeros((canvas_wh[1], canvas_wh[0], 3), np.uint8)
acc_mask = np.zeros((canvas_wh[1], canvas_wh[0]), np.uint8)

for step, i in enumerate(order, start=1):
    warped, mask = warp_onto_canvas(images[i].work, H_canvas[i], canvas_wh)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

    # Tint the newly added region so it is obvious what this photo brought.
    fresh = (mask > 0) & (acc_mask == 0)
    overlap = (mask > 0) & (acc_mask > 0)
    acc[mask > 0] = warped[mask > 0]
    acc_mask[mask > 0] = 255

    vis = acc.copy()
    tint = np.zeros_like(vis)
    tint[fresh] = (0, 190, 0)        # green  = new ground covered
    tint[overlap] = (0, 120, 255)    # orange = overlapped an earlier photo
    vis = cv2.addWeighted(vis, 0.82, tint, 0.18, 0)

    show(fit(vis, 1250),
         f"after {step}/{len(order)} photos — just added {images[i].short}   "
         f"(green = newly covered, orange = overlaps an earlier photo)", size=(15, 8))
    plt.show()

print(f"canvas coverage: {(acc_mask > 0).mean()*100:.1f}% "
      "(the rest is the black border left by the warp)")

# %% [markdown]
# Three problems are now visible in that raw stack, and they map exactly onto the
# three compositing stages that follow:
#
# 1. **Brightness steps** at the joins — each photo was metered separately.
# 2. **Hard rectangular edges** where the last photo overwrote the previous one,
#    cutting straight through products.
# 3. The orange regions show **every place the same product is present twice** —
#    which is where duplicated facings would come from if we averaged.

# %% [markdown]
# ## Step 6 — Compositing
#
# Geometry is solved. Now we turn four warped images into one coherent picture,
# and there are three separate problems:
#
# 1. **Exposure** — each shot is metered independently, so a naive paste leaves
#    visible brightness steps. Fixed with per-image, per-block, per-channel gains.
# 2. **Where to cut** — every overlap contains the same products twice. Averaging
#    ghosts them; an arbitrary rectangle slices products in half. A **graph-cut**
#    seam finder routes the boundary along places the images already agree.
# 3. **How to join** — **multi-band blending** merges low frequencies over a wide
#    band (killing the exposure step) and high frequencies over a narrow one
#    (keeping text sharp).
#
# We build it up one stage at a time so you can see what each buys.
#
# *(Rendering at 0.30× here for speed; the shipped pipeline renders at 1.0×.)*

# %%
from shelfpano.compose import compose, crop_to_content

RENDER = 0.30
H_render, canvas_wh, eff = plan_canvas(images, H_ba, anchor, RENDER, 60.0)
print(f"canvas: {canvas_wh[0]} x {canvas_wh[1]} px "
      f"({canvas_wh[0]*canvas_wh[1]/1e6:.1f} MP) at {eff:.3f}x")

variants = {}
for label, kw in [
    ("1. raw paste",            dict(exposure="none",   seam="none",     blend="none")),
    ("2. + exposure gains",     dict(exposure="blocks", seam="none",     blend="none")),
    ("3. + graph-cut seams",    dict(exposure="blocks", seam="graphcut", blend="none")),
    ("4. + multi-band blend",   dict(exposure="blocks", seam="graphcut", blend="multiband")),
]:
    out = compose(images, H_ba, anchor, render_scale=RENDER, seam_scale=0.4,
                  max_megapixels=60.0, verbose=False, **kw)
    variants[label] = crop_to_content(out)
    print(f"{label:<26} -> {variants[label].shape[1]}x{variants[label].shape[0]}")

# %%
for label, img in variants.items():
    show(fit(img, 1250), label, size=(15, 8))
    plt.show()

# %% [markdown]
# Step 1 → 2 removes the brightness steps between fixtures. Step 2 → 3 stops the
# rectangular overwrite that sliced through products. Step 3 → 4 removes the
# residual hairline at the seam.
#
# Let's zoom into one seam to see the last two clearly.

# %%
ref = variants["4. + multi-band blend"]
hh, ww = ref.shape[:2]
y0, y1 = int(hh * 0.20), int(hh * 0.55)
x0, x1 = int(ww * 0.42), int(ww * 0.68)
show_row([v[y0:y1, x0:x1] for v in variants.values()],
         list(variants.keys()), size=(17, 7))
plt.show()

# %% [markdown]
# ## Which image won each pixel?
#
# This is the direct check on two of the grading criteria. A **duplicated facing**
# would show up as an island of one colour inside another; a seam **slicing
# through a product** would show up as a boundary crossing the middle of a
# fixture instead of running along its edge.

# %%
seam_map = REPO / "work" / "nb_seams.jpg"
stats_c = {}
_ = compose(images, H_ba, anchor, render_scale=RENDER, seam_scale=0.4,
            max_megapixels=60.0, seam_debug_path=str(seam_map),
            stats=stats_c, verbose=False)
show(cv2.imread(str(seam_map)), "Source attribution — one colour per input image",
     size=(14, 8))
plt.show()

print("Share of the panorama contributed by each image:")
for name, share in stats_c["source_pixel_share"].items():
    print(f"  {name[:8]}: {share*100:5.1f}%")

# %% [markdown]
# Seams follow the vertical fixture uprights — the natural boundaries — and no
# region is duplicated.
#
# If an image shows ~0 %, it was fully covered by its neighbours (a redundant
# re-shot). That is *not* automatically a problem, but it is worth verifying no
# content was lost — `experiments/exp06_completeness.py` does exactly that by
# comparing the union of all warped masks against what the blender actually
# wrote. It reports 0.00 % missing on all five stores.

# %% [markdown]
# ## Step 7 — Evaluate
#
# There is no pixel-exact ground truth (the reference panoramas are in a
# different coordinate frame), so we measure **self-consistency**: given the
# final homographies, how well do the images agree where they overlap?

# %%
metrics = evaluate_layout(images, H_ba, all_pairs)

if not metrics["n_pairs_scored"]:
    print("No overlapping pairs could be scored, so there are no metrics.")
    print("That means the match graph came apart - scroll up to step 4 and")
    print("check the 'connected components' line and any !! warnings.")
else:
    print(f"reprojection RMS : {metrics['reproj_rms_px']} px "
          f"(median {metrics['reproj_median_px']}, p95 {metrics['reproj_p95_px']})")
    print(f"overlap NCC      : mean {metrics['mean_overlap_ncc']}, "
          f"min {metrics['min_overlap_ncc']}")
    print(f"pairs scored     : {metrics['n_pairs_scored']} "
          f"(of {sum(p.ok for p in all_pairs)} verified)")
print()
for pp in metrics["per_pair"]:
    print(f"  {pp['pair']:<22} rms {pp['reproj_rms_px']:>6} px   "
          f"ncc {pp['overlap_ncc']:>6}   overlap {pp['overlap_frac']:>5}")

# %% [markdown]
# **The honest caveat**, and the reason "next steps" in the write-up leads with
# real accuracy evaluation: these are *self-consistency* metrics. A confidently
# wrong layout can score well on them. The ablation found a configuration that
# places all 19 images across the five stores with a *wrong* alignment — and the
# only number that noticed was the worst-overlap NCC, dropping 0.42 → 0.18.
# **Image count alone would have called it a pass.**

# %%
# Side by side with the reference output for this store.
mine = cv2.imread(str(REPO / "outputs" / f"{STORE}_stitched.jpg"))
ref_img = cv2.imread(str(REPO / "stitching_assignment_data" / STORE / "reference_preview.jpg"))
if mine is not None and ref_img is not None:
    Hh = 620
    m = cv2.resize(mine, (int(mine.shape[1] * Hh / mine.shape[0]), Hh))
    r = cv2.resize(ref_img, (int(ref_img.shape[1] * Hh / ref_img.shape[0]), Hh))
    show_row([m, r], ["shelfpano (full-res output)", "reference"], size=(17, 7))
    plt.show()
else:
    print("Run `python -m shelfpano --data-root stitching_assignment_data --out outputs` "
          "first to generate the full-resolution output.")

# %% [markdown]
# ## Things to try
#
# Re-run the notebook with any of these changed near the top, or from the CLI:
#
# | Change | What you should see |
# |---|---|
# | `N_HYPOTHESES = 1` | plain single-mode RANSAC. See the note below — this is the knob that changes the panorama |
# | `match_all_pairs(..., min_ncc=-1)` | disables the photometric gate; the wrong mode is accepted |
# | `RATIO = 0.70` | the textbook ratio test; measured neutral on this data |
# | `N_FEATURES = 2000` | the wide net matters — an image drops out |
# | `detect_all(..., "orb")` | binary descriptors lose 2 of 19 images across the dataset |
# | `STORE = "store_4"` | 5 images, widest run, the reversed capture order |
# | `STORE = "store_3"` | 5 images including two near-duplicate re-shots |
#
# **`N_HYPOTHESES = 1` is stochastic, so run it a few times.** MAGSAC++ is
# randomised; comment out `cv2.setRNGSeed(0)` in the setup cell to vary it.
# Across 8 seeds on store 1's hard pair (`bcd6e94e → cbaf4880`):
#
# | | `N_HYPOTHESES = 6` | `N_HYPOTHESES = 1` |
# |---|---|---|
# | pair kept | 6/8 seeds | 3/8 seeds |
# | NCC when kept | 0.52 – 0.56 (**correct** alignment) | 0.26 – 0.29 (**wrong**, one module off) |
# | failure mode | drops the pair | **silently keeps a misaligned one** |
#
# That second row is the real argument for mode enumeration. With one
# hypothesis, the runs that "succeed" are accepting the shifted alignment — a
# panorama that looks complete but has a fixture visibly wrong. With six, it
# either finds the correct alignment or drops the pair; it never accepted a bad
# one. Multi-hypothesis does not merely recover more images, it **fails safe**
# rather than failing silently.
#
# (Changing `n_modes` on the `enumerate_modes` helper does *not* do this. It
# only changes the step-3 illustration. Any difference you see in the panorama
# after touching it is incidental RNG drift, not the parameter.)
#
# From the command line:
#
# ```bash
# python -m shelfpano --images stitching_assignment_data/store_1/images \
#     --out outputs/store_1_stitched.jpg --debug-seams
#
# python -m shelfpano --images stitching_assignment_data/store_1/images \
#     --out /tmp/plain_ransac.jpg --n-hypotheses 1   # plain RANSAC; run it a few times
# ```
#
# Related reading in the repo: `WRITEUP.md` §1 (why a homography, not a rotating
# camera), §3 (the full ablation, including the stages that turned out *not* to
# matter), and §4 (what didn't work and what it taught me).
