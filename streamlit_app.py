"""Browser front end for shelfpano — upload shelf photos, download a panorama.

    streamlit run streamlit_app.py

Why this file is not called `streamlit.py`: Python puts the script's own
directory first on sys.path, so a module named `streamlit.py` shadows the
installed `streamlit` package and `import streamlit as st` imports *this file*
instead. The app would fail on its first line.

Design note: this module deliberately contains no stitching logic. It writes
the uploads to a temp directory and calls the same `stitch_store()` the CLI and
the test-suite use, so the web app cannot drift from the pipeline the rest of
the project validates. It only adds I/O, progress reporting and presentation.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shelfpano.pipeline import PipelineConfig, stitch_store   # noqa: E402

MAX_UPLOAD_MB = 60
PREVIEW_MAX_DIM = 1600

st.set_page_config(page_title="shelfpano — shelf panorama stitcher",
                   page_icon="🛒", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def downscale(img: np.ndarray, max_dim: int = PREVIEW_MAX_DIM) -> np.ndarray:
    """Shrink for on-screen preview. The download always gets full resolution."""
    s = min(1.0, max_dim / max(img.shape[:2]))
    if s == 1.0:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def save_uploads(files, target: Path) -> tuple[list[Path], list[str]]:
    """Write uploads to disk, skipping anything OpenCV cannot decode.

    Returning the rejects rather than raising keeps one bad file from sinking
    an otherwise fine batch - the user can see what was skipped and re-upload.
    """
    saved, rejected = [], []
    for f in files:
        raw = f.getbuffer()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            rejected.append(f"{f.name} — larger than {MAX_UPLOAD_MB} MB")
            continue
        probe = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if probe is None:
            rejected.append(f"{f.name} — not a readable image")
            continue
        # Normalise the extension: load_store filters on it, and phone uploads
        # arrive with every casing of .JPG/.jpeg imaginable.
        out = target / f"{len(saved):03d}_{Path(f.name).stem}.jpg"
        cv2.imwrite(str(out), probe, [cv2.IMWRITE_JPEG_QUALITY, 97])
        saved.append(out)
    return saved, rejected


# --------------------------------------------------------------------------
# Sidebar — settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 shelfpano")
    st.caption("Panorama stitching built for retail shelf photography.")
    st.divider()

    st.subheader("Quality")
    render_scale = st.select_slider(
        "Output resolution",
        options=[0.25, 0.4, 0.6, 0.8, 1.0],
        value=1.0,
        format_func=lambda v: {0.25: "25% (fast preview)", 0.4: "40%",
                               0.6: "60%", 0.8: "80%",
                               1.0: "100% (full, recommended)"}[v],
        help="Geometry is always solved at full accuracy; this only scales the "
             "rendered pixels. Lower it if you are just checking the layout.")

    blend = st.selectbox(
        "Blending", ["multiband", "feather", "none"], index=0,
        help="multiband keeps text sharp while hiding exposure steps. "
             "'none' shows the raw seams — useful for debugging.")

    st.subheader("Matching")
    n_hypotheses = st.slider(
        "RANSAC hypotheses per pair", 1, 8, 6,
        help="Shelves are full of near-identical product facings, so the "
             "alignment shifted by one facing can out-vote the correct one. "
             "Enumerating several hypotheses and picking by pixel agreement "
             "fixes that. 1 = plain RANSAC.")

    min_ncc = st.slider(
        "Photometric gate (NCC)", -1.0, 0.8, 0.25, 0.05,
        help="Minimum pixel agreement to accept a pair. Lower accepts more "
             "pairs but risks a wrong alignment; raise it to be strict.")

    with st.expander("Advanced"):
        n_features = st.select_slider("Features per image",
                                      options=[2000, 5000, 8000, 12000, 20000],
                                      value=12000)
        seam = st.selectbox("Seam finder",
                            ["graphcut", "dp", "voronoi", "none"], index=0)
        exposure = st.selectbox("Exposure compensation",
                                ["blocks", "gain", "none"], index=0)
        use_deep = st.checkbox(
            "LoFTR fallback for unmatched pairs", value=False,
            help="Only runs when the match graph comes apart. Needs torch + "
                 "kornia installed; slow on CPU.")
        show_seams = st.checkbox("Show which photo won each pixel", value=True)

    st.divider()
    st.caption("Tip: photos must overlap. Aim for 30–50% between consecutive "
               "shots, and keep the camera roughly the same distance from the "
               "fixture.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Shelf panorama stitcher")
st.markdown(
    "Upload the photos of **one shelf run** — overlapping shots of the same "
    "fixture, in any order — and get a single stitched panorama back. "
    "Filenames are ignored; the arrangement is discovered from the images."
)

uploaded = st.file_uploader(
    "Shelf photos (2 or more, JPG/PNG)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)

if not uploaded:
    st.info("👆 Upload at least two overlapping photos to begin.")
    with st.expander("What makes a good capture?"):
        st.markdown("""
- **Overlap 30–50%** between consecutive shots. Below ~15% the pipeline may
  not find a reliable link and will report a partial panorama.
- **One fixture run per batch.** Photos from a different aisle will simply
  fail to match and be reported as unconnected.
- **Keep roughly the same distance and height.** Large scale changes between
  shots make matching harder.
- **Order does not matter.** Nor do filenames.
- Avoid heavy motion blur — it destroys the keypoints matching relies on.
        """)
    st.stop()

if len(uploaded) < 2:
    st.warning("At least two photos are needed to make a panorama.")
    st.stop()

with st.expander(f"📷 {len(uploaded)} photo(s) selected", expanded=False):
    cols = st.columns(min(len(uploaded), 5))
    for k, f in enumerate(uploaded):
        arr = cv2.imdecode(np.frombuffer(f.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            cols[k % len(cols)].image(bgr_to_rgb(downscale(arr, 300)),
                                      caption=f.name[:22], use_container_width=True)

go = st.button("🧵 Stitch panorama", type="primary", use_container_width=True)

if go:
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = Path(tmp) / "images"
        images_dir.mkdir()
        saved, rejected = save_uploads(uploaded, images_dir)

        for r in rejected:
            st.warning(f"Skipped {r}")
        if len(saved) < 2:
            st.error("Fewer than two usable images. Nothing to stitch.")
            st.stop()

        cfg = PipelineConfig(
            n_features=n_features, min_ncc=min_ncc, n_hypotheses=n_hypotheses,
            use_deep_fallback=use_deep, render_scale=render_scale,
            exposure=exposure, seam=seam, blend=blend,
            seam_debug_path=str(Path(tmp) / "seams.jpg") if show_seams else None,
        )

        status = st.status("Stitching…", expanded=True)
        t0 = time.time()
        try:
            with status:
                st.write(f"Loading {len(saved)} images, detecting features, "
                         "matching every pair…")
                pano, rep = stitch_store(str(images_dir), cfg, verbose=False)
            status.update(label=f"Done in {time.time()-t0:.1f}s", state="complete")
        except Exception as exc:                        # noqa: BLE001
            status.update(label="Stitching failed", state="error")
            st.error(f"**{type(exc).__name__}:** {exc}")
            st.info("Most often this means the photos do not overlap enough, "
                    "or they are not all of the same fixture. Try the capture "
                    "guidance in the sidebar.")
            st.stop()

        # ---- outcome banner ------------------------------------------------
        complete = rep.n_images_used == rep.n_images
        if complete:
            st.success(f"✅ Stitched all {rep.n_images} photos into one panorama.")
        else:
            st.warning(
                f"⚠️ **Partial panorama** — {rep.n_images_used} of "
                f"{rep.n_images} photos placed. Not connected: "
                f"`{', '.join(rep.dropped_images)}`.\n\n"
                "Those photos did not overlap the rest enough to be matched "
                "reliably. The panorama below contains everything that did.")

        # ---- metrics -------------------------------------------------------
        m = rep.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Photos placed", f"{rep.n_images_used}/{rep.n_images}")
        c2.metric("Output size", f"{pano.shape[1]}×{pano.shape[0]}")
        c3.metric("Alignment error",
                  f"{m['reproj_rms_px']} px" if m.get("reproj_rms_px") else "—",
                  help="RMS reprojection error at matching scale. Under ~3 px "
                       "means shelf edges line up.")
        c4.metric("Overlap agreement",
                  f"{m['mean_overlap_ncc']:.2f}" if m.get("mean_overlap_ncc") else "—",
                  help="Mean pixel correlation where photos overlap. Higher is "
                       "better; below ~0.3 suggests a misalignment.")

        # ---- the panorama --------------------------------------------------
        st.subheader("Panorama")
        st.image(bgr_to_rgb(downscale(pano)), use_container_width=True,
                 caption=f"Preview (downscaled). The download is full "
                         f"resolution: {pano.shape[1]}×{pano.shape[0]}.")

        full_jpeg = encode_jpeg(pano)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download panorama (JPG)", data=full_jpeg,
                           file_name="panorama.jpg", mime="image/jpeg",
                           type="primary", use_container_width=True)

        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("panorama.jpg", full_jpeg)
            z.writestr("report.json", __import__("json").dumps(
                {"config": cfg.__dict__, "report": rep.__dict__}, indent=2,
                default=str))
            seam_file = Path(tmp) / "seams.jpg"
            if show_seams and seam_file.exists():
                z.writestr("seam_map.jpg", seam_file.read_bytes())
        d2.download_button("⬇️ Download everything (ZIP)", data=bundle.getvalue(),
                           file_name="shelfpano_result.zip", mime="application/zip",
                           use_container_width=True)

        # ---- diagnostics ---------------------------------------------------
        with st.expander("🔍 How it was assembled"):
            st.markdown(f"**Discovered left-to-right order** — recovered from "
                        f"pixels, not filenames:")
            st.code(" → ".join(rep.left_to_right_order) or "—")
            st.markdown(f"**Reference frame:** `{rep.anchor}` "
                        "(chosen as the most central photo, which keeps "
                        "perspective distortion symmetric)")

            if rep.pairs:
                st.markdown("**Pairwise matching** — every pair was tested; "
                            "non-overlapping pairs *should* be rejected:")
                st.dataframe(
                    [{"pair": f"{p['i']} → {p['j']}",
                      "accepted": "✅" if p["ok"] else "—",
                      "inliers": p["inliers"], "overlap": p["overlap"],
                      "pixel agreement": p["ncc"],
                      "reason": p["reason"] or ""} for p in rep.pairs],
                    use_container_width=True, hide_index=True)

            seam_file = Path(tmp) / "seams.jpg"
            if show_seams and seam_file.exists():
                st.markdown("**Which photo won each pixel** — one colour per "
                            "input. Seams should follow fixture edges, and no "
                            "colour should appear as an island inside another:")
                smap = cv2.imread(str(seam_file))
                if smap is not None:
                    st.image(bgr_to_rgb(downscale(smap, 1400)),
                             use_container_width=True)
                shares = m.get("source_pixel_share", {})
                if shares:
                    st.caption("Share of the panorama contributed by each photo: "
                               + ", ".join(f"`{k[:10]}` {v*100:.1f}%"
                                           for k, v in shares.items()))
