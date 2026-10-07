"""Streamlit dashboard for IMAGE AUTOPSY."""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.acquisition import (
    downscale_for_analysis,
    get_image_metadata,
    load_image,
    quantize_image,
    sample_image,
)
from src.copy_move import detect_copy_move
from src.demo_generator import demo_cases
from src.edges import structural_analysis
from src.evidence_fusion import DEFAULT_WEIGHTS, fuse_evidence
from src.frequency_analysis import fft_magnitude_spectrum, frequency_anomaly_map, frequency_statistics
from src.morphology import morphology_panel, refine_mask
from src.preprocessing import calculate_histogram, preprocessing_gallery
from src.report import generate_report
from src.segmentation import connected_components, draw_regions, otsu_threshold, percentile_anomaly_threshold
from src.splicing import splicing_anomaly_map
from src.texture_analysis import texture_feature_panel
from src.wavelet_analysis import haar_decomposition, wavelet_anomaly_map


st.set_page_config(
    page_title="IMAGE AUTOPSY",
    page_icon="IA",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
  --ia-bg: #0e1117;
  --ia-panel: #151a23;
  --ia-soft: #1f2633;
  --ia-text: #e8edf5;
  --ia-muted: #9ba7b6;
  --ia-red: #ff4b5c;
  --ia-cyan: #61d9ff;
}
.main { background: var(--ia-bg); color: var(--ia-text); }
.block-container { padding-top: 1.4rem; }
h1, h2, h3 { letter-spacing: 0 !important; }
.ia-title {
  padding: 1.2rem 1.3rem;
  border: 1px solid #2c3544;
  background: linear-gradient(135deg, #121822 0%, #171f2d 58%, #211820 100%);
  border-radius: 8px;
  margin-bottom: 1rem;
}
.ia-title h1 { margin: 0; font-size: 2.2rem; }
.ia-title p { color: var(--ia-muted); margin: .25rem 0 0 0; }
.ia-pill {
  display: inline-block;
  padding: .25rem .55rem;
  margin: .15rem .25rem .15rem 0;
  border-radius: 999px;
  border: 1px solid #344154;
  background: #121722;
  color: #dce7f6;
  font-size: .78rem;
}
.ia-timeline {
  font-size: .83rem;
  line-height: 1.85;
  color: #d7e0ec;
  padding: .8rem;
  border: 1px solid #2b3544;
  border-radius: 8px;
  background: #111722;
}
.small-note { color: var(--ia-muted); font-size: .86rem; }
</style>
"""


def show_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ia-title">
          <h1>IMAGE AUTOPSY</h1>
          <p>Explainable Digital Forensics for Image Manipulation Detection</p>
          <p><strong>Every edit leaves evidence.</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def histogram_figure(gray_or_rgb: np.ndarray):
    hist = calculate_histogram(gray_or_rgb)
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ax.plot(hist, color="#61d9ff", linewidth=1.4)
    ax.fill_between(np.arange(256), hist, color="#61d9ff", alpha=0.15)
    ax.set_xlim(0, 255)
    ax.set_title("Intensity Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def image_grid(items: dict[str, np.ndarray], columns: int = 3) -> None:
    names = list(items.keys())
    for start in range(0, len(names), columns):
        cols = st.columns(columns)
        for col, name in zip(cols, names[start : start + columns]):
            col.image(items[name], caption=name, use_container_width=True)


def region_table(regions) -> pd.DataFrame:
    rows = []
    for idx, region in enumerate(regions, start=1):
        x, y, w, h = region.bbox
        rows.append(
            {
                "Region": idx,
                "Area %": round(region.area_percent, 2),
                "Position": f"x={x}, y={y}",
                "Size": f"{w} x {h}",
                "Score": round(region.score, 2),
                "Severity": region.severity,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(image: np.ndarray, filename: str, file_size: int) -> dict:
    metadata = get_image_metadata(image, filename=filename, file_size_bytes=file_size)
    analysis_image, scale = downscale_for_analysis(image, max_side=760)

    preprocessing = preprocessing_gallery(analysis_image)
    structural = structural_analysis(analysis_image)
    texture = texture_feature_panel(analysis_image)
    _, spectrum = fft_magnitude_spectrum(analysis_image)
    frequency_map = frequency_anomaly_map(analysis_image)
    wavelets = haar_decomposition(analysis_image)
    wavelet_map = wavelet_anomaly_map(analysis_image)
    copy_move = detect_copy_move(analysis_image)
    splice_map = splicing_anomaly_map(analysis_image)

    evidence_maps = {
        "texture": texture["Texture Anomaly"],
        "edge": structural["Edge Inconsistency"],
        "frequency": frequency_map,
        "wavelet": wavelet_map,
        "copy_move": copy_move.mask,
        "local_statistical": splice_map,
    }
    fusion = fuse_evidence(
        analysis_image,
        evidence_maps,
        weights=DEFAULT_WEIGHTS,
        copy_move_score=copy_move.evidence_score,
    )

    raw_mask = percentile_anomaly_threshold(fusion.fusion_map, percentile=88)
    refined_mask = refine_mask(raw_mask, min_area=max(50, int(0.0012 * raw_mask.size)))
    regions = connected_components(refined_mask, fusion.fusion_map, min_area_percent=0.12)
    region_overlay = draw_regions(analysis_image, regions)
    report = generate_report(metadata, fusion, regions, copy_move.message)

    return {
        "metadata": metadata,
        "analysis_image": analysis_image,
        "analysis_scale": scale,
        "preprocessing": preprocessing,
        "structural": structural,
        "texture": texture,
        "spectrum": spectrum,
        "frequency_map": frequency_map,
        "frequency_stats": frequency_statistics(analysis_image),
        "wavelets": wavelets,
        "wavelet_map": wavelet_map,
        "copy_move": copy_move,
        "splice_map": splice_map,
        "fusion": fusion,
        "raw_mask": raw_mask,
        "morphology": morphology_panel(raw_mask),
        "refined_mask": refined_mask,
        "regions": regions,
        "region_overlay": region_overlay,
        "report": report,
    }


def acquisition_controls(image: np.ndarray) -> None:
    st.subheader("CO1 Acquisition: Sampling and Quantization")
    col_a, col_b = st.columns(2)
    sample_percent = col_a.slider("Resolution sampling", 25, 100, 70, step=5)
    levels = col_b.select_slider("Intensity quantization levels", options=[16, 32, 64, 128, 256], value=64)
    sampled = sample_image(image, sample_percent)
    quantized = quantize_image(sampled, levels)
    image_grid({"Original": image, "Sampled": sampled, "Quantized": quantized}, columns=3)


def sidebar_input() -> tuple[np.ndarray | None, str, int]:
    st.sidebar.header("Case Input")
    demo_mode = st.sidebar.toggle("DEMO MODE", value=True)

    if demo_mode:
        cases = demo_cases()
        selected = st.sidebar.selectbox("Presentation case", list(cases.keys()))
        st.sidebar.caption("Demo images are generated locally and do not require internet access.")
        image = cases[selected]
        return image, selected.replace(" ", "_").lower() + ".png", int(image.nbytes)

    uploaded = st.sidebar.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return None, "", 0
    image = load_image(uploaded)
    return image, uploaded.name, int(uploaded.size)


def main() -> None:
    show_header()
    try:
        image, filename, file_size = sidebar_input()
    except ValueError as exc:
        st.error(str(exc))
        return

    if image is None:
        st.info("Upload an image or enable DEMO MODE to begin a forensic analysis.")
        return

    if min(image.shape[:2]) < 32:
        st.error("This image is too small for meaningful forensic analysis.")
        return

    with st.spinner("Running forensic examination..."):
        results = run_pipeline(image, filename, file_size)

    metadata = results["metadata"]
    fusion = results["fusion"]

    top_cols = st.columns([1.15, 1, 1])
    top_cols[0].image(image, caption="Case image", use_container_width=True)
    top_cols[1].metric("Forensic Suspicion Score", f"{fusion.score} / 100")
    top_cols[1].metric("Verdict", fusion.verdict)
    top_cols[1].metric("Likely Manipulation", fusion.likely_manipulation)
    top_cols[2].markdown(
        """
        <div class="ia-timeline">
        CASE RECEIVED<br>
        &darr; IMAGE ACQUIRED<br>
        &darr; PREPROCESSING<br>
        &darr; STRUCTURAL EXAMINATION<br>
        &darr; FREQUENCY EXAMINATION<br>
        &darr; WAVELET EXAMINATION<br>
        &darr; DUPLICATION SEARCH<br>
        &darr; SEGMENTATION<br>
        &darr; MORPHOLOGICAL REFINEMENT<br>
        &darr; EVIDENCE FUSION<br>
        &darr; FINAL VERDICT
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(fusion.disclaimer)

    tabs = st.tabs(
        [
            "CASE FILE",
            "VISUAL ANALYSIS",
            "EDGE & SEGMENTATION",
            "FREQUENCY ANALYSIS",
            "WAVELET ANALYSIS",
            "COPY-MOVE ANALYSIS",
            "EVIDENCE MAP",
            "FINAL VERDICT",
        ]
    )

    with tabs[0]:
        st.subheader("Image Information")
        metrics = st.columns(5)
        metrics[0].metric("Width", metadata.width)
        metrics[1].metric("Height", metadata.height)
        metrics[2].metric("Channels", metadata.channels)
        metrics[3].metric("File Size", f"{metadata.file_size_kb:.1f} KB")
        metrics[4].metric("Type", metadata.image_type)
        stats = st.columns(4)
        stats[0].metric("Mean", f"{metadata.mean_intensity:.2f}")
        stats[1].metric("Std Dev", f"{metadata.std_intensity:.2f}")
        stats[2].metric("Min", metadata.min_intensity)
        stats[3].metric("Max", metadata.max_intensity)
        st.markdown(
            f"<span class='ia-pill'>{'Grayscale' if metadata.is_grayscale else 'Color image'}</span>"
            f"<span class='ia-pill'>Analysis scale: {results['analysis_scale']:.2f}</span>",
            unsafe_allow_html=True,
        )
        acquisition_controls(results["analysis_image"])

    with tabs[1]:
        st.subheader("CO2 Preprocessing and Histogram Analysis")
        st.pyplot(histogram_figure(results["analysis_image"]), use_container_width=True)
        image_grid(results["preprocessing"], columns=3)

        st.subheader("Texture / Local Statistical Anomaly Analysis")
        image_grid(results["texture"], columns=3)
        st.caption(
            "Texture evidence compares local variance, standard deviation and gradient energy "
            "against surrounding neighborhoods. It is supporting evidence, not standalone proof."
        )

    with tabs[2]:
        st.subheader("Structural Examination: Point, Line and Edge Detection")
        structural_images = {
            "Point Response": results["structural"]["Point Response"],
            "Sobel": results["structural"]["Sobel"],
            "Scharr": results["structural"]["Scharr"],
            "Prewitt": results["structural"]["Prewitt"],
            "Laplacian": results["structural"]["Laplacian"],
            "Canny": results["structural"]["Canny"],
            "Hough Lines": results["structural"]["Line Overlay"],
            "Edge Inconsistency": results["structural"]["Edge Inconsistency"],
        }
        ecols = st.columns(2)
        ecols[0].metric("Edge density", f"{results['structural']['Edge Density']:.4f}")
        ecols[1].metric("Hough lines", int(results["structural"]["Line Count"]))
        image_grid(structural_images, columns=4)

        st.subheader("Segmentation and Morphological Refinement")
        image_grid(
            {
                "Raw Fusion Map": results["fusion"].fusion_map,
                "Otsu Mask": otsu_threshold(results["fusion"].fusion_map),
                "Thresholded Anomaly Mask": results["raw_mask"],
                "Refined Suspicious Regions": results["refined_mask"],
                "Connected Components": results["region_overlay"],
            },
            columns=3,
        )
        st.subheader("Morphological Operations")
        image_grid(results["morphology"], columns=3)

    with tabs[3]:
        st.subheader("CO3 Fourier-Domain Analysis")
        fcols = st.columns(3)
        fcols[0].image(results["analysis_image"], caption="Analysis Image", use_container_width=True)
        fcols[1].image(results["spectrum"], caption="FFT Magnitude Spectrum", use_container_width=True)
        fcols[2].image(results["frequency_map"], caption="Frequency Anomaly Map", use_container_width=True)
        st.dataframe(pd.DataFrame([results["frequency_stats"]]), use_container_width=True)
        st.caption("Compression and resizing can create frequency anomalies; FFT evidence is fused with other channels.")

    with tabs[4]:
        st.subheader("CO3 Wavelet-Domain Analysis")
        image_grid(results["wavelets"], columns=4)
        st.image(results["wavelet_map"], caption="Wavelet Detail-Energy Anomaly Map", use_container_width=True)
        st.markdown(
            """
            <span class='ia-pill'>LL: low-frequency approximation</span>
            <span class='ia-pill'>LH / HL / HH: horizontal, vertical and diagonal detail evidence</span>
            """,
            unsafe_allow_html=True,
        )

    with tabs[5]:
        st.subheader("Copy-Move / Duplicated Region Search")
        copy_move = results["copy_move"]
        ccols = st.columns(3)
        ccols[0].image(results["analysis_image"], caption="Analysis Image", use_container_width=True)
        ccols[1].image(copy_move.mask, caption="Copy-Move Mask", use_container_width=True)
        ccols[2].image(copy_move.overlay, caption="Matched Regions Overlay", use_container_width=True)
        st.info(copy_move.message)
        if copy_move.matches:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Match": idx,
                            "Box A": match.box_a,
                            "Box B": match.box_b,
                            "Score": round(match.score, 3),
                            "Descriptor Distance": round(match.distance, 3),
                        }
                        for idx, match in enumerate(copy_move.matches[:20], start=1)
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.write("No strong internal duplication pattern detected.")

    with tabs[6]:
        st.subheader("Evidence Fusion: Suspicious-Region Heatmap")
        hcols = st.columns(3)
        hcols[0].image(results["analysis_image"], caption="Original / Analysis Image", use_container_width=True)
        hcols[1].image(fusion.heatmap, caption="Forensic Heatmap", use_container_width=True)
        hcols[2].image(fusion.overlay, caption="Original + Red Suspicion Overlay", use_container_width=True)
        st.image(results["region_overlay"], caption="Segmented Suspicious Regions", use_container_width=True)

        score_df = pd.DataFrame(
            [
                {
                    "Evidence Channel": name.replace("_", " ").title(),
                    "Channel Score": round(value, 3),
                    "Weight": DEFAULT_WEIGHTS.get(name, 0.0),
                }
                for name, value in fusion.channel_scores.items()
            ]
        )
        st.dataframe(score_df, use_container_width=True)
        if results["regions"]:
            st.dataframe(region_table(results["regions"]), use_container_width=True)
        else:
            st.write("No connected suspicious region passed the reporting threshold.")

    with tabs[7]:
        st.subheader("Final Forensic Report")
        st.code(results["report"], language="text")
        report_bytes = results["report"].encode("utf-8")
        st.download_button(
            "Download Report",
            data=io.BytesIO(report_bytes),
            file_name=Path(filename).stem + "_image_autopsy_report.txt",
            mime="text/plain",
        )
        st.warning(
            "This is an explainable academic image-processing system. The output is not a certified "
            "legal determination of authenticity."
        )


if __name__ == "__main__":
    main()
