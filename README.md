# IMAGE AUTOPSY

## Explainable Digital Forensics Framework for Detecting and Localizing Image Manipulation

IMAGE AUTOPSY is a classical image-processing and computer-vision application for academic digital image forensics. It examines an uploaded image, creates visual evidence maps, localizes suspicious regions, and generates a readable forensic report.

It does not use pretrained CNNs, cloud APIs, or black-box deep-learning classifiers. The system is designed to show why an image looks suspicious, where the evidence appears, and which image-processing tests contributed to the final heuristic score.

## Problem Statement

Manipulated images are common in misinformation, document fraud, social media editing, and legal/cybersecurity investigations. A useful forensic tool should do more than label an image as fake or real. It should expose visible evidence, isolate suspicious regions, and explain which signals support the conclusion.

## Motivation

Many beginner image-forensics demos collapse the problem into a single binary prediction. IMAGE AUTOPSY instead behaves like a digital forensic investigator:

- inspect the image metadata and intensity statistics
- analyze histograms, contrast, edge structure, texture and local statistics
- examine Fourier and wavelet evidence
- search for internal duplicated regions
- segment and refine suspicious regions
- fuse evidence into a transparent report

## Objectives

- Build a polished Streamlit forensic dashboard.
- Demonstrate core image-processing concepts from CO1 through CO6.
- Detect and localize possible manipulation evidence using classical methods.
- Provide deterministic demo cases for presentations.
- Generate an explainable forensic report with limitations clearly stated.

## CO Mapping

CO1:
Image acquisition, grayscale conversion, sampling and quantization

CO2:
Histogram analysis, histogram equalization, contrast enhancement, log transformation and image negatives

CO3:
2-D Fourier transform and wavelet decomposition

CO4:
Erosion, dilation, opening, closing and connected components

CO5:
Point/line/edge detection, segmentation and suspicious-object/region extraction

CO6:
Digital forensics, cybersecurity, misinformation analysis, legal/document verification, system limitations and application-specific technique selection

## System Architecture

```text
image upload / demo case
        |
        v
acquisition and validation
        |
        v
preprocessing + histogram analysis
        |
        v
edge, texture, FFT, wavelet, splicing, copy-move evidence maps
        |
        v
evidence fusion
        |
        v
segmentation + morphology + connected components
        |
        v
heatmap, suspicious-region overlay, final report
```

## Algorithms Used

- Sampling and quantization for CO1 acquisition demonstration.
- Histogram equalization, CLAHE, log transform and negative transform for CO2 preprocessing.
- Sobel, Scharr, Prewitt, Laplacian, Canny, Harris response and Hough lines for CO5 structural examination.
- Local mean, local variance, local standard deviation, entropy and gradient energy for texture anomalies.
- 2-D FFT with shifted log-magnitude spectrum and high-frequency residual mapping for CO3 frequency analysis.
- Haar wavelet LL, LH, HL and HH subbands for CO3 multi-resolution analysis.
- Patch descriptor + KD-tree nearest-neighbor copy-move detection.
- Otsu, adaptive and percentile thresholding for segmentation.
- Erosion, dilation, opening and closing for CO4 morphological cleanup.
- Connected-component analysis for suspicious-region extraction.
- Weighted evidence fusion for a heuristic forensic suspicion score.

## Why Each Algorithm Is Relevant

- Histogram methods reveal global and local contrast changes that may accompany editing.
- Edge and line detectors expose structural discontinuities around inserted or modified objects.
- Texture statistics highlight patches whose noise, variance or gradient behavior differs from their neighborhood.
- FFT analysis can reveal unusual high-frequency residuals, periodic artifacts or compression-related inconsistencies.
- Wavelets separate approximation and detail information, making localized high-frequency artifacts easier to inspect.
- Copy-move analysis targets a common manipulation pattern: duplicated regions inside the same image.
- Segmentation and morphology convert fuzzy evidence into clean, reportable suspicious regions.

## Screenshots

Add screenshots after running the dashboard:

- Case file and metadata
- Preprocessing comparison
- Edge and segmentation tab
- Fourier and wavelet analysis
- Copy-move analysis
- Final evidence heatmap
- Report view

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

```bash
streamlit run app.py
```

The app opens in the browser. Use DEMO MODE for reliable built-in cases, or upload a JPG, JPEG, PNG or BMP image.

## Demo Instructions

1. Start the app with `streamlit run app.py`.
2. Keep DEMO MODE enabled in the sidebar.
3. Select one of the built-in cases:
   - CASE 001 - Original
   - CASE 002 - Copy-Move
   - CASE 003 - Local Contrast Manipulation
   - CASE 004 - Spliced Image
   - CASE 005 - Local Brightness Manipulation
4. Review the evidence tabs.
5. Open FINAL VERDICT and download the report.

The demo images are generated locally, so the presentation does not depend on internet access.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── acquisition.py
│   ├── preprocessing.py
│   ├── edges.py
│   ├── texture_analysis.py
│   ├── frequency_analysis.py
│   ├── wavelet_analysis.py
│   ├── copy_move.py
│   ├── splicing.py
│   ├── segmentation.py
│   ├── morphology.py
│   ├── evidence_fusion.py
│   ├── report.py
│   └── demo_generator.py
├── demo_images/
├── outputs/
└── tests/
```

## Limitations

- Classical image-processing methods can produce false positives.
- Compression can create frequency anomalies.
- Resizing can alter texture statistics.
- Lighting differences can resemble manipulation.
- Copy-move detection depends on patch similarity.
- Very sophisticated manipulation may evade classical methods.
- Results are not legally certified forensic evidence.
- Different image formats and camera pipelines affect evidence.

## Future Scope

- Add EXIF metadata parsing and camera-pipeline consistency checks.
- Export reports as PDF.
- Add richer copy-move clustering and affine-invariant descriptors.
- Add noise-level estimation maps.
- Save complete case folders in `outputs/`.
- Add more controlled demo images and benchmark examples.

## Real-World Applications

- Cybersecurity and misinformation triage.
- Social-media image authenticity screening.
- Legal and document-verification support.
- Academic teaching of image processing and digital forensics.
- Newsroom and public-interest investigative workflows.

## Important Disclaimer

IMAGE AUTOPSY produces a heuristic evidence-aggregation score for academic demonstration. It is not a certified forensic probability and must not be treated as legally binding proof of manipulation.
