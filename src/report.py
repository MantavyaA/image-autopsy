"""Forensic report generation."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from src.acquisition import ImageMetadata
from src.evidence_fusion import FusionResult
from src.segmentation import Region


def generate_case_id() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    return f"IA-{date_part}-{uuid4().hex[:8].upper()}"


def generate_report(
    metadata: ImageMetadata,
    fusion: FusionResult,
    regions: list[Region],
    copy_move_message: str,
    case_id: str | None = None,
) -> str:
    """Generate a readable IMAGE AUTOPSY report."""

    case_id = case_id or generate_case_id()
    largest = max((region.area_percent for region in regions), default=0.0)
    evidence_lines = "\n".join(f"[+] {flag}" for flag in fusion.evidence_flags)
    if not evidence_lines:
        evidence_lines = "[ ] No strong isolated evidence channel"

    region_lines = []
    for index, region in enumerate(regions[:8], start=1):
        x, y, w, h = region.bbox
        region_lines.append(
            f"Region {index}: area={region.area_percent:.2f}% "
            f"bbox=(x={x}, y={y}, w={w}, h={h}) score={region.score:.2f} severity={region.severity}"
        )
    if not region_lines:
        region_lines.append("No connected suspicious region passed the reporting threshold.")

    return f"""------------------------------------
IMAGE AUTOPSY REPORT
------------------------------------

Case ID: {case_id}

Image:
{metadata.filename}

Resolution:
{metadata.width} x {metadata.height}

Channels:
{metadata.channels}

Mean / Std intensity:
{metadata.mean_intensity:.2f} / {metadata.std_intensity:.2f}

Overall suspicion score:
{fusion.score} / 100

Verdict:
{fusion.verdict}

Likely manipulation:
{fusion.likely_manipulation}

Evidence:
{evidence_lines}

Copy-move analysis:
{copy_move_message}

Suspicious regions:
{len(regions)}

Largest suspicious region:
{largest:.2f}% of image

Region details:
{chr(10).join(region_lines)}

Important:
This system provides algorithmic forensic evidence for academic demonstration and must not be treated as a certified forensic verification tool.

Limitations:
- Classical image-processing methods can produce false positives.
- Compression, resizing, lighting changes and camera pipelines can alter evidence channels.
- Sophisticated manipulations may evade these tests.
- Results are not legally certified forensic evidence.

------------------------------------
"""
