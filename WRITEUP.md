# Panorama generation for store shelves — write-up

**Result: 19/19 input images placed across all 5 stores, one panorama each, no
partial results.** Mean reprojection error 1.4 px at matching scale; every
store rendered at full 3840 px input resolution with price tags legible.
Same settings for every store — nothing tuned per-store.

| Store | Images | Placed | Output | Reproj RMS | Overlap NCC (mean / worst) | Time |
|---|---|---|---|---|---|---|
| 1 | 4 | 4/4 | 7308×4266 | 1.35 px | 0.57 / 0.42 | 12 s |
| 2 | 2 | 2/2 | 3880×4530 | 1.48 px | 0.64 / 0.64 | 5 s |
| 3 | 5 | 5/5 | 5973×4873 | 1.53 px | 0.63 / 0.50 | 25 s |
| 4 | 5 | 5/5 | 7939×4679 | 1.37 px | 0.50 / 0.42 | 14 s |
| 5 | 3 | 3/3 | 4229×4492 | 1.36 px | 0.71 / 0.65 | 14 s |

---

## 1. The pipeline, and why each stage is there

```
load ─> features ─> pairwise match ─> match graph ─> bundle ─> compose
        RootSIFT    sequential        max spanning   joint     exposure gains
        12k/image   RANSAC +          tree from a    homog.    graph-cut seams
        CLAHE       3 gates           central anchor refine    multiband blend
                    (+LoFTR rescue)
```

Geometry is solved on 1600 px copies; pixels are composited at the full 3840 px.
SIFT localises sub-pixel, so the geometry recovered at 1600 px is as good as at
full resolution for a fraction of the cost, while the output stays legible.

**Two-scale contract.** Every homography in the system maps *work* pixels to
*work* pixels, and there is exactly one function (`to_render_frame`) that
converts to render coordinates. Coordinate-frame confusion is the classic way a
stitcher produces a plausible-looking but subtly wrong result, so the conversion
lives in one place and is pinned by unit tests that check it against explicit
point mapping.

### Features — RootSIFT, 12 000 per image, CLAHE first

A cigarette gantry is a grid of near-identical facings: descriptors repeat, and
the distinctive detail (price rails, promo headers, shelf joints, scuffs) is
sparse relative to the repetition. So we cast a wide net — 12 000 keypoints
rather than the usual ~2 000 — and lower SIFT's contrast threshold to 0.03,
because shelf interiors are flat and evenly lit and genuinely useful low-contrast
corners fall below the 0.04 default.

CLAHE before detection matters more than it looks: each shot is metered
independently, so the same shelf is a different brightness in two photos. Local
contrast equalisation makes the detector fire on the *same* structures in both.

The wide net is load-bearing — at 2 000 features the pipeline drops an image
(see §3). RootSIFT, by contrast, turned out to make no measurable difference on
this data; it is kept because it costs nothing, but it is not what makes this
work, and the ablation says so.

### Matching — inlier count proposes, pixels decide

This is the stage that the domain breaks, and where most of the effort went.

**The problem.** RANSAC returns the model with the most inliers. On a shelf of
repeated facings, the alignment shifted by exactly one product module is
supported by *every repeated facing* and can genuinely out-vote the correct
alignment. RANSAC is not malfunctioning — it is correctly maximising a criterion
that happens to be wrong here.

Measured on store 1, pair `bcd6e94e→cbaf4880`, by enumerating modes:

| RANSAC mode | inliers | overlap | NCC | |
|---|---|---|---|---|
| 0 (what RANSAC returns) | 219 | 0.76 | **0.21** | wrong — shifted one module |
| 1 | 208 | 0.40 | 0.28 | |
| 2 | 138 | 0.27 | **0.56** | **correct** |

The wrong answer has 60 % more inliers than the right one.

**The fix.** *Sequential RANSAC*: fit, record the model, strip its inliers, refit.
This enumerates the distinct alignment modes rather than only the largest. Then
score each surviving mode photometrically and pick the winner by pixel
agreement. Inlier count proposes; pixels decide.

**Three verification gates**, cheapest first, applied to every mode:

1. **Geometric** — inlier count and inlier ratio.
2. **Plausibility** — the warped outline must stay a convex, orientation-
   preserving quadrilateral with bounded area change and no horizon inside the
   frame. RANSAC happily returns folded-over or 100×-blown-up homographies when
   inliers cluster in one corner; this is what rejects them.
3. **Photometric** — warp one image into the other and correlate the actual
   pixels over the overlap, on *gradient magnitude* rather than intensity so
   exposure differences do not depress the score.

Gate 3 is the one that catches the shifted-by-one-facing failure, because that
failure is invisible to feature counting and obvious in pixels.

### Graph, anchor, spanning tree

Filenames are capture UUIDs with no ordering information, so the arrangement is
discovered from the images. Verified pairs form a graph; a maximum-weight
spanning tree from a central anchor gives each image an initial homography.

The anchor is chosen for **low eccentricity** — the image whose furthest
neighbour is closest. Anchoring on an end image forces long chains, and because
a homography chain accumulates perspective, the far end of the panorama gets
stretched into a wedge. A central anchor halves the longest chain. (The provided
reference outputs show the same choice: the middle fixture is the undistorted
one.)

### Bundle adjustment — and why *not* OpenCV's

**This is the single most important modelling decision in the project.**

OpenCV's `Stitcher` and its `BundleAdjusterRay`/`BundleAdjusterReproj`
parameterise each camera as a **rotation plus focal length** — they assume the
camera only *rotated* between shots, as on a tripod. That is false here: the
operator walks sideways along the aisle, so consecutive shots differ by
**translation**. Under translation no single rotation aligns the views, and the
solver converges to a compromise that bows straight shelf edges into arcs —
plainly visible in the baseline output (§2).

But a shelf front is close to a **plane**, and for a planar scene the mapping
between any two views is an *exact homography regardless of how the camera
moved*. So a homography per image is both the correct model and only 8 DOF. The
model matches the geometry instead of fighting it.

The refinement minimises **symmetric transfer error** over every verified
correspondence from every verified pair — not just the n−1 tree edges — with:

- **Huber loss** so surviving mismatches cannot dominate;
- **`x_scale='jac'`** because translation entries are ~10³ px while perspective
  entries are ~10⁻⁶, and an unscaled trust region is hopelessly ill-conditioned;
- **grid-bucketed correspondence sampling**, because inliers cluster on the few
  textured regions and letting one busy corner dominate is exactly the
  configuration that admits a skewed fit;
- **the anchor held fixed at identity**, which fixes the gauge — otherwise any
  global homography applied to every image leaves the cost unchanged and the
  solve is rank-deficient;
- **a rejection guard**: if the refined solution scores worse than the input, it
  is discarded rather than shipped.

### Compositing

- **Exposure** — per-image, per-block, per-channel gains
  (`BlocksChannelsCompensator`), fed *before* seam finding so the seam finder is
  not tempted to route around brightness steps it should ignore.
- **Seams** — graph-cut with a colour+gradient cost, so the boundary follows
  places where the images already agree (shelf gaps, rail edges, fixture
  uprights) instead of slicing through a product. This is what makes overlapping
  facings appear **once**.
- **Blending** — multi-band: low frequencies merged over a wide band (killing
  the exposure step), high frequencies over a narrow one (keeping text sharp).
  Band count is derived from the canvas size rather than fixed.

Seam placement runs at 0.25× and rendering at full resolution — the standard
cost split, since seam *placement* does not need 25 MP to be right but seam
*rendering* does.

---

## 2. Baseline: what the stock stitcher actually does

Before building anything I ran `cv2.Stitcher` in both modes on all five stores
(`exp01`). It is worth being precise about how it fails, because "it doesn't
work" is not a diagnosis.

| Store | PANORAMA | SCANS |
|---|---|---|
| 1 | OK, **2/4 images** | OK, **2/4** |
| 2 | OK, 2/2 | OK, 2/2 |
| 3 | OK, 5/5 | OK, **3/5** |
| 4 | OK, **2/5** | OK, **2/5** |
| 5 | OK, 3/3 | OK, 3/3 |

**Two distinct failures, and the first one is a trap.**

1. **It returns `status == OK` while silently discarding images.** `Stitcher`
   calls `leaveBiggestComponent()` internally and reports success on whatever
   survived. Checking the status code tells you nothing. The number that matters
   is `stitcher.component()`, and on store 4 it is 2 of 5. Anything built on
   "status OK" as a health check would have shipped 40 % of that store's shelf
   run and reported success.
2. **Where it does stitch, the geometry is wrong.** PANORAMA mode's spherical
   warp bows straight shelf edges into visible arcs (store 4 and store 5 are the
   clearest). That is the rotation-only camera model asserting itself on
   translational capture — the diagnosis that motivated the homography model.

This is why the pipeline uses OpenCV's *compositing* pieces (they are excellent
and model-agnostic) but not its *camera model or bundle adjuster*.

---

## 3. Ablation: which stages actually earn their place

Every store, one stage disabled at a time (`exp05`). `used` is images placed out
of 19 total; `worst NCC` is the single worst overlap across all five stores —
the number that catches one bad fixture hiding behind four good ones.

| Configuration | Used | Reproj RMS | Mean NCC | **Worst NCC** | Time | Verdict |
|---|---|---|---|---|---|---|
| **full pipeline** | **19/19** | 1.42 px | 0.610 | **0.423** | 25 s | — |
| ORB instead of SIFT | **17/19** | 1.54 | 0.624 | 0.435 | 36 s | **needed** |
| plain SIFT (no RootSIFT) | 19/19 | 1.38 | 0.608 | 0.423 | 26 s | no effect |
| strict ratio 0.70 | 19/19 | 1.36 | 0.614 | 0.423 | 25 s | no effect |
| single-hypothesis RANSAC | **18/19** | 1.40 | 0.613 | 0.424 | 28 s | **needed** |
| no photometric gate | 19/19 | 1.43 | 0.583 | **0.179** | 27 s | **needed** |
| no bundle adjustment | 19/19 | **1.46** | 0.605 | 0.421 | 25 s | small win |
| + ECC polish (off by default) | 19/19 | 1.43 | 0.612 | 0.425 | 35 s | no effect |
| no LoFTR fallback | 19/19 | 1.44 | 0.609 | 0.423 | 25 s | no effect *here* |
| fewer features (2 000) | **18/19** | 1.34 | 0.611 | 0.408 | 16 s | **needed** |
| no plausibility gate | 19/19 | 1.42 | 0.605 | 0.424 | 25 s | no effect *here* |

**What this shows, including several places I was wrong:**

- **Multi-hypothesis RANSAC is load-bearing** — reverting to plain single-mode
  RANSAC loses an image. This was the main engineering effort and it is
  justified.
- **The photometric gate is load-bearing, but not in the way the image count
  suggests.** Disabling it still places 19/19 — it *looks* like a pass. What
  actually happens is that a wrong homography is accepted, and the only signal
  is worst-overlap NCC collapsing from 0.42 to **0.18**. **An image-count
  success metric would have called this a pass.** This is the strongest argument
  in the project for measuring alignment quality rather than completion, and it
  is why §6 puts real accuracy evaluation first.
- **SIFT over ORB is justified** — ORB's binary descriptor loses 2 images, and
  is slower here to boot.
- **The wide feature net is justified** — 2 000 features loses an image. (Its
  RMS looks *better* only because it is averaging over an easier set of pairs
  after dropping the hard one — a good reminder that a mean over a changing
  denominator is not comparable.)
- **Four stages I expected to matter do not: RootSIFT, the loose ratio
  threshold, ECC refinement, and the plausibility gate.** I had written a
  justification for each before measuring. On this data they are neutral. I
  turned ECC **off by default** (~25 % of runtime for a 0.01 px change) rather
  than keep a stage that only looks sophisticated. RootSIFT, the loose ratio and
  the plausibility gate cost nothing measurable and guard against failure modes
  I have argued for but not demonstrated *on these five stores*, so they stay —
  but as hedges, not as contributors, and this table says so.
- **LoFTR is not load-bearing on this dataset either** (see §4.2).
- **Bundle adjustment is a modest win here** (1.46 → 1.42 px). With 2–5 images
  per store the chains are short, so there is little drift to remove; its value
  grows with longer runs and it is nearly free.

The honest summary: of eleven stages tested, **four are provably load-bearing,
one is a small win, and the rest are insurance whose premium I can afford but
whose payout I have not observed.**

---

## 4. What didn't work, and what it taught me

### 4.1 Trusting RANSAC's answer (the central failure)

Covered in §1. The lesson generalises past this assignment: **RANSAC's inlier
count is a proxy for correctness that breaks precisely when the scene is
repetitive** — which is the defining property of retail shelving. Any verification
built only on feature agreement inherits the same blind spot. The fix was not a
better matcher but a *different arbiter*.

### 4.2 LoFTR as the rescue for hard pairs — right idea, wrong failure mode

When store 1 first came out at 3/4 images, my hypothesis was that SIFT lacked
keypoints in the overlap, and the fix was a detector-free deep matcher. I wired
in LoFTR and re-ran.

**LoFTR produced the same wrong homography, with the same NCC of 0.21.**

That was the informative result. The pair was not failing for lack of
correspondences — it was failing because the *scene is genuinely ambiguous*, and
a denser matcher is just as fooled by a shelf that looks the same one module
over. A better matcher cannot resolve an ambiguity that lives in the scene
rather than in the descriptor. That reframing is what led to multi-hypothesis
enumeration, which fixed it.

LoFTR is still in the pipeline, but demoted to what it is actually good for: a
fallback for genuinely *low-texture* overlaps. And I changed *when* it runs —
originally it retried every failed pair, which is wasteful because most failed
pairs are simply non-overlapping images that *should* fail. It now runs only
when the match graph comes out **disconnected**, and only on pairs that would
bridge components. On this dataset it never fires. It is a hedge for unseen
stores, not a contributor here, and the write-up should not pretend otherwise.

### 4.3 Diagnosing by threshold instead of by eye

When store 1 first failed, the rejected pair scored NCC 0.21 against a 0.25
threshold. The tempting move — and I nearly made it — was to lower the
threshold to 0.20 and declare victory: it would have produced 19/19 immediately.

Instead I built `exp03`, which renders a **checkerboard overlay** of the two
images under the candidate homography. The overlay showed the banner text and
product rows duplicated at a horizontal offset: the homography was genuinely
wrong, and the gate was doing its job. Lowering the threshold would have shipped
a visibly broken panorama and hidden the real bug.

**The lesson: when a threshold rejects something you believe is good, look at
the pixels before you move the threshold.** That checkerboard tool paid for
itself several times over and is now the first thing I reach for.

### 4.4 Assuming image count means success

Reported honestly because it nearly fooled me: my first quality gate was "did
all images get placed". The no-photometric-gate ablation places 19/19 with a
*wrong* alignment. Image count measures whether the graph connected, not whether
the result is right. This is why `evaluate.py` exists and why every run now
reports reprojection error and worst-overlap NCC alongside the count.

### 4.5 Chained homographies without global refinement

Straightforward and expected, but worth stating: composing pairwise homographies
along a tree multiplies error, and because homographies carry perspective the
compounded error shows up as progressive shear towards the ends. Bundle
adjustment over all pairs fixes it. The measured gain here is small only because
the runs are short (2–5 images).

### 4.6 Things I checked that turned out fine

- **Duplicated facings.** I added a false-colour source map showing which image
  won each pixel (`--debug-seams`). The seams run along fixture uprights with no
  islands — nothing duplicated.
- **Dropped content.** The source map also revealed that in stores 3 and 4 one
  image contributes ~0 % of pixels. That could have meant a hole in the
  panorama, so I wrote `exp06` to compare the union of all warped masks against
  what the blender actually wrote. Result: **0.00 % missing** on every store —
  those images were genuinely redundant re-shots, fully covered by neighbours.
  Worth verifying rather than assuming.

### 4.7 A domain observation: digital signage cannot be stitched consistently

Stores 3 and 4 have overhead TV panels showing **rotating content**. Different
photos catch different frames, so the same physical panel legitimately shows
different things in two images. No geometric method can reconcile that — there
is no correct answer, only a choice. It produces the faint ghosting visible in
store 4's top banner area (the provided reference has the same artifact for the
same reason). The shelves themselves — what downstream analysis actually needs —
are unaffected. §6 proposes handling it explicitly.

---

## 5. How I used AI coding agents

I used **Claude Code** (Claude Opus) throughout, in a terminal alongside the
repo. What follows is how I actually directed it, including where it was wrong.

**How I decomposed the work.** I did not ask for "a panorama stitcher". I
scoped modules with explicit contracts and had it implement them one at a time:
image loading with a two-scale invariant, feature extraction, pairwise matching
with a defined `PairMatch` return type, graph construction, bundle adjustment,
compositing. Keeping each module small enough to review in one sitting is what
made verification tractable — a single generated 800-line stitcher would have
been unreviewable, and I would have ended up owning code I could not explain.

**Where it helped most.**
- *Boilerplate around a known algorithm.* The OpenCV `detail` compositing API
  (corner/mask/blender triples, the seam-scale-then-compose-scale split) is
  fiddly and poorly documented. It got that right quickly.
- *Instrumentation.* The diagnostic scripts — checkerboard overlay, local NCC
  maps, source-attribution maps, the ablation harness — are the kind of code
  that is tedious to write and enormously valuable to have. Being able to
  produce a diagnostic in two minutes changed how I debugged: I checked things I
  would otherwise have assumed.
- *Speed of experimentation.* The mode-enumeration experiment that produced the
  219-vs-138-inlier table went from idea to table in a few minutes.

**Where it was wrong, and how I caught it.**

1. **It accepted `status == OK` from `cv2.Stitcher` as success.** The first
   baseline script reported all five stores succeeding. That contradicted the
   visual result, so I had it emit output dimensions, then `component()` — which
   exposed 2/5. *Caught by insisting the metric match what I could see.* This is
   the failure I would most expect to slip through unreviewed, because the code
   was correct and the *interpretation* was wrong.
2. **It proposed lowering the NCC threshold to fix store 1.** Locally reasonable,
   and it would have "worked". I rejected it and asked for a visualisation
   instead, which showed the homography was genuinely wrong (§4.3). *Caught by
   refusing to move a threshold without looking at the pixels.*
3. **A sign error — in a test, not the code.** I had it write tests for the
   homography composition convention. One test asserted `+600` where the correct
   answer was `−600`; the implementation was right and the test's expectation was
   wrong. Worth reporting because the agent's *code* was correct and its *test*
   encoded a plausible-sounding misconception. Had I taken the test as ground
   truth and "fixed" the code, I would have mirrored every panorama. I worked
   the convention through by hand to decide which was right, and the test now
   documents it explicitly.
4. **Over-generalised fallback logic.** Its first LoFTR integration retried
   *every* failed pair, including obviously non-overlapping ones. Correct but
   wasteful, and it obscured which failures mattered. I redirected it to run
   only on component-bridging pairs (§4.2).

**My working rule.** The agent is fast at code and unreliable at judgement about
whether a result is *good*. So I kept judgement — what to measure, whether a
number is believable, when a threshold is hiding a bug — and delegated
implementation. Every non-obvious claim in this write-up is backed by a script
in `experiments/` that I can re-run, which is the standard I held generated code
to: **not "does it run" but "does it produce a number I can defend".**

---

## 6. Next steps, in priority order

**1. Replace self-consistency metrics with real accuracy evaluation.** *(highest
value)* Everything I currently measure — reprojection error, overlap NCC — is
self-consistency. A confidently wrong layout can score well. The ablation in §3
showed a configuration that places every image with a wrong alignment, which is
exactly the failure a self-consistency metric under-weights.

What I would build: annotate a few dozen corresponding points per store by hand
(or harvest them from a product detector run independently on each input and on
the panorama), then measure **facing-level precision/recall** — does each product
facing appear exactly once, in the right place? That is the metric that matches
what downstream analysis consumes, and it makes regressions detectable without
looking at images. I would prioritise this above any accuracy improvement,
because right now I cannot prove an improvement *is* one.

**2. Parallax-aware compositing.** The current model is a single global plane per
image. Real fixtures have depth — products protrude, end-caps turn corners,
store 5 has a genuine wall corner. Where depth is significant, one homography
cannot align everything and the seam finder is left to hide the residual. Two
concrete steps, cheapest first: (a) *local warp refinement* — as-projective-as-
possible / mesh-based warping, which relaxes the single-plane assumption without
needing 3D; (b) detect multi-plane structure and fit a homography per plane.
(a) is a contained change to the compose stage and would measurably reduce the
residual misalignment visible at store 4's seams.

**3. Explicit handling of non-static content.** Digital signage (§4.7) and
transient occluders (staff, customers, the delivery boxes in store 2) currently
get blended, producing ghosts. The fix is a *disagreement mask*: where warped
images disagree far more than the local exposure difference explains, stop
blending and take a single source, choosing the one with the best focus/most
frontal view. This is a well-scoped change to `compose.py` and removes the most
visible remaining artifact class.

**4. Robustness work driven by failure injection.** Five stores is a small
sample and all five now pass; I do not know where the edge is. I would build a
synthetic stress harness — programmatically vary overlap fraction, exposure
step, blur, and viewing angle on held-out captures — to find the operating limits
and turn them into regression tests. Concretely: *at what overlap fraction does
the pipeline start failing?* I cannot answer that today, and in production that
number determines the capture instructions given to store staff — which is
probably the highest-leverage lever on end-to-end quality, since fixing capture
is cheaper than fixing stitching.

**5. Scale and speed.** Currently O(n²) pairwise matching, fine at n≤5. For a
full store (dozens of fixtures) I would add a vocabulary-tree or global-
descriptor pre-filter to propose candidate pairs, and cache features. Runtime is
5–30 s per store and the assignment imposes no constraint, so this is
deliberately last — but it is the first thing that breaks when the input grows
from one fixture run to a whole store.

**6. Productionisation.** Confidence score per output so low-quality stitches can
be flagged for re-capture rather than silently entering the pipeline; structured
failure reporting (already partly there via the JSON reports); pinned
dependencies and a container.

---

## 7. Sources

- **RootSIFT** — Arandjelović & Zisserman, *Three things everyone should know to
  improve object retrieval*, CVPR 2012.
- **MAGSAC++** — Barath et al., CVPR 2020; used via OpenCV `USAC_MAGSAC`.
- **ECC alignment** — Evangelidis & Psarakis, PAMI 2008; via `cv2.findTransformECC`.
- **Graph-cut seams** — Kwatra et al., *Graphcut Textures*, SIGGRAPH 2003; via
  `cv2.detail_GraphCutSeamFinder`.
- **Multi-band blending** — Burt & Adelson, 1983; via `cv2.detail_MultiBandBlender`.
- **LoFTR** — Sun et al., CVPR 2021; pretrained weights via `kornia.feature.LoFTR`.
- **Bundle adjustment framing** — Triggs et al., *Bundle Adjustment — A Modern
  Synthesis*, 1999; and Brown & Lowe, *Automatic Panoramic Image Stitching using
  Invariant Features*, IJCV 2007 (whose rotation-only model is the one this
  pipeline deliberately departs from).

Exposure compensation, seam finding and blending use OpenCV's `cv2.detail`
implementations. Feature extraction, matching and verification, graph
construction, bundle adjustment, canvas planning and all evaluation code are
written for this project.
