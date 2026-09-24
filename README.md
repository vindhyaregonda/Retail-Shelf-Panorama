<div align="center">

# 🛒 shelfpano

**Panorama stitching that survives retail shelves.**

Turns a handful of overlapping shelf photos — in any order, with no filename hints —
into one coherent, full-resolution panorama.

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-40%20passing-3fb950)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

```
 ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            ┌──────────────────────────────┐
 │ photo  │ │ photo  │ │ photo  │ │ photo  │    ───►     │        one panorama          │
 │   ?    │ │   ?    │ │   ?    │ │   ?    │             │   ordered · aligned · clean  │
 └────────┘ └────────┘ └────────┘ └────────┘            └──────────────────────────────┘
        unordered, overlapping, uneven exposure
```

</div>

---

## Why shelves break ordinary stitchers

Point `cv2.Stitcher` at a shelf run and it reports **success** while quietly throwing
half your photos away. Measured on a 5-store benchmark:

| Store | Stock `cv2.Stitcher` (PANORAMA) | shelfpano |
|---|---|---|
| 1 | `OK` — **2 of 4** images used | **4 / 4** |
| 2 | `OK` — 2 of 2 | **2 / 2** |
| 3 | `OK` — 5 of 5, shelves bowed into arcs | **5 / 5** |
| 4 | `OK` — **2 of 5** images used | **5 / 5** |
| 5 | `OK` — 3 of 3, curved | **3 / 3** |

Three things make this domain hard:

- **Repetition.** A cigarette gantry is a grid of near-identical facings. The alignment
  shifted by *one product module* is supported by every repeated facing — and can
  genuinely out-vote the correct one inside RANSAC.
- **Translation, not rotation.** The operator walks sideways along the aisle. Panorama
  stitchers assume a camera that only *rotated*, so they bow straight shelf edges into
  arcs.
- **Silent failure.** `status == OK` tells you nothing; `Stitcher` drops images and still
  reports success. Nothing in the output says a fixture is missing.

---

## ⚡ Quick start

### Option 1 — the web app (no code)

```bash
pip install -r requirements.txt -r requirements-app.txt
streamlit run streamlit_app.py
```

Drag in your photos, press **Stitch**, download the full-resolution JPEG.
The sidebar exposes the matching and blending controls, and the result panel shows the
discovered left-to-right order, the pairwise match table, and a colour map of which
photo won each pixel.

> ⚠️ The entry point is `streamlit_app.py`, **not** `streamlit.py`. Python puts the
> script's own directory first on `sys.path`, so a file named `streamlit.py` shadows the
> installed package and `import streamlit` imports the file itself.

### Option 2 — the CLI

```bash
pip install -r requirements.txt

# one folder of photos -> one panorama
python -m shelfpano --images path/to/photos --out panorama.jpg

# batch: every store_*/images/ folder under a root
python -m shelfpano --data-root path/to/data --out outputs/
```

Writes the panorama plus a `*_report.json` with the match graph, the recovered ordering,
bundle-adjustment residuals and quality metrics.

Exit codes: `0` all photos placed · `2` partial panorama · `1` nothing produced.

### Option 3 — the interactive walkthrough

```bash
pip install -r requirements.txt -r requirements-notebook.txt
jupyter lab notebooks/pipeline_walkthrough.ipynb
```

Runs the pipeline **one stage at a time**, visualising what happens to the input photos:
keypoints, raw matches, the checkerboard overlay that exposes a wrong homography, the
match graph, each photo before/after its warp, the panorama assembling one photo at a
time, and each compositing stage added in turn.

---

## 🧠 How it works

```
load ──> features ──> pairwise match ──> match graph ──> bundle ──> compose
         RootSIFT      sequential          max spanning   joint      exposure gains
         12k/image     RANSAC +            tree from a    homography graph-cut seams
         CLAHE         3 verification      central anchor refinement multiband blend
                       gates
```

Geometry is solved on 1600 px copies; pixels are rendered at full resolution so price
tags stay readable.

### The two decisions that matter

<table>
<tr><th width="50%">1 · A homography per image, not a rotating camera</th>
<th width="50%">2 · Pixels arbitrate, not inlier counts</th></tr>
<tr valign="top"><td>

Shelf photos differ by **translation**, so a rotation-only camera model cannot align
them and bends straight shelf edges into arcs.

A shelf front is near-**planar**, and for a plane the view-to-view mapping is an *exact
homography regardless of how the camera moved* — 8 DOF, correct by construction.

Refined by a custom bundle adjustment (`scipy.least_squares`, Huber loss, symmetric
transfer error over **every** verified pair, anchor fixed to pin the gauge).

</td><td>

RANSAC returns the model with the most inliers. On repeated facings that is often the
**wrong** alignment. Measured on one real pair:

| mode | inliers | pixel agreement |
|---|---|---|
| 0 (what RANSAC returns) | **219** | 0.21 ❌ |
| 2 | 138 | **0.56** ✅ |

So we enumerate modes with *sequential RANSAC* and pick by photometric agreement.
Inlier count proposes; pixels decide.

</td></tr></table>

### Three verification gates

Every candidate alignment must clear all three, cheapest first:

1. **Geometric** — inlier count and ratio.
2. **Plausibility** — the warped outline must stay a convex, orientation-preserving quad
   with bounded scale. Rejects the folded-over and 100×-blown-up homographies RANSAC
   happily returns when inliers cluster in a corner.
3. **Photometric** — correlate actual pixels over the overlap, on *gradient magnitude* so
   exposure differences don't depress the score. This is the gate that catches the
   shifted-by-one-facing failure, which is invisible to feature counting.

### Compositing

Per-image, per-block, per-channel **exposure gains** → **graph-cut seams** routed along
places the photos already agree (so no facing appears twice and none is sliced) →
**multi-band blending** (low frequencies over a wide band to kill exposure steps, high
frequencies over a narrow one to keep text sharp).

---

## 📊 What actually earns its place

Every stage ablated across the benchmark. Negative results included, because most
write-ups quietly omit them:

| Configuration | Images placed | Reproj RMS | Worst overlap NCC | Verdict |
|---|---|---|---|---|
| **full pipeline** | **19/19** | 1.42 px | 0.42 | — |
| ORB instead of SIFT | 17/19 | 1.54 | 0.44 | **needed** |
| single-hypothesis RANSAC | 18/19 | 1.40 | 0.42 | **needed** |
| no photometric gate | 19/19 | 1.43 | **0.18** | **needed** ⚠️ |
| fewer features (2 000) | 18/19 | 1.34 | 0.41 | **needed** |
| no bundle adjustment | 19/19 | 1.46 | 0.42 | small win |
| plain SIFT (no RootSIFT) | 19/19 | 1.38 | 0.42 | no effect |
| strict ratio 0.70 | 19/19 | 1.36 | 0.42 | no effect |
| ECC refinement | 19/19 | 1.43 | 0.43 | no effect |

⚠️ **Read the "no photometric gate" row carefully.** It still places every image — it
*looks* like a pass. What actually happens is that a wrong homography is accepted, and
the only number that notices is the worst-overlap NCC collapsing from 0.42 to 0.18.
An image-count success metric would have shipped it. That is the single strongest
argument in this project for measuring alignment quality rather than completion.

ECC refinement cost ~25% of runtime for a 0.01 px change, so it is **off by default**
rather than left in to look sophisticated.

---

## 🗂 Project structure

```
shelfpano/
  imageset.py       loading; the work/full two-scale contract; coordinate algebra
  features.py       RootSIFT extraction with CLAHE preprocessing
  matching.py       sequential RANSAC, the three verification gates, ECC refinement
  deep_matching.py  optional LoFTR fallback for low-texture overlaps
  graph.py          connectivity, anchor selection, maximum spanning tree
  bundle.py         joint homography refinement (scipy least_squares)
  compose.py        canvas planning, exposure compensation, seam finding, blending
  evaluate.py       objective self-consistency metrics
  pipeline.py       orchestration + JSON reporting
  cli.py            argument parsing
streamlit_app.py    web front end (thin: no stitching logic of its own)
notebooks/          stage-by-stage walkthrough (.py source of truth + .ipynb)
experiments/        the numbered experiments behind every claim above
tests/              40 tests
tools/              py_to_ipynb.py — regenerates the notebook from its .py
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

**40 tests.** Highlights of what they guard:

- **Coordinate-frame algebra** — the scale and sign conventions that silently *mirror* a
  panorama when wrong. One test pins the homography composition direction after a real
  sign bug.
- **Bundle adjustment** against a synthetic problem with a known answer.
- **End-to-end** on a deliberately shuffled split image, where the correct output is known.
- **Every selectable compositing option**, each in a **subprocess** — because several
  `cv2.detail_*` classes expose no constructor to Python: `cv2.detail_NoSeamFinder()`
  *succeeds* but returns a NULL-`Ptr` wrapper, and the first method call is a **SIGSEGV**
  that no `try`/`except` can catch. The only defence is to execute every option.

---

## 🗺 Roadmap

1. **Facing-level accuracy evaluation.** Everything measured today is *self-consistency*;
   a confidently wrong layout can still score well (see the ⚠️ row above). Annotate
   corresponding points and measure whether each product facing appears exactly once, in
   the right place.
2. **Parallax-aware compositing.** One global plane per image is an approximation;
   products protrude and end-caps turn corners. Mesh-based / as-projective-as-possible
   warping relaxes it without needing 3D.
3. **Explicit non-static content handling.** Digital signage shows different frames in
   different shots and can never photometrically agree; transient occluders (staff,
   boxes) currently get blended into ghosts. Detect disagreement and take a single source.
4. **Failure-injection harness.** Vary overlap, exposure, blur and angle synthetically to
   find the operating limits — *at what overlap fraction does this start failing?* That
   number drives capture instructions, which is cheaper to fix than stitching.
5. **Scale.** O(n²) pairwise matching is fine at n ≤ 5; a vocabulary-tree pre-filter and
   striped compositing would take it to whole-store runs.

---

## 📚 References

| Technique | Source |
|---|---|
| RootSIFT | Arandjelović & Zisserman, *Three things everyone should know to improve object retrieval*, CVPR 2012 |
| MAGSAC++ | Barath et al., CVPR 2020 — via OpenCV `USAC_MAGSAC` |
| ECC alignment | Evangelidis & Psarakis, PAMI 2008 |
| Graph-cut seams | Kwatra et al., *Graphcut Textures*, SIGGRAPH 2003 |
| Multi-band blending | Burt & Adelson, 1983 |
| LoFTR | Sun et al., CVPR 2021 — weights via `kornia` |
| Rotation-model framing (deliberately departed from) | Brown & Lowe, *Automatic Panoramic Image Stitching using Invariant Features*, IJCV 2007 |

Exposure compensation, seam finding and blending use OpenCV's `cv2.detail`
implementations. Feature extraction, matching and verification, graph construction,
bundle adjustment, canvas planning and all evaluation code are written for this project.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

**No sample imagery is included in this repository.** Bring your own shelf photos, or
point the CLI at any folder of overlapping images.
