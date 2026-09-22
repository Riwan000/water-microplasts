"""
Step 4: ReportLab-based PDF laboratory report generator.

Produces a self-contained lab report PDF containing:
  - Sample metadata (ID, date, volume)
  - Tier 1 summary (total count, particles/L)
  - Tier 2 particle table (class, dimensions, confidence)
  - Annotated microscope image (if provided as base64 PNG)
"""

import base64
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_PAGE_W, _PAGE_H = A4
_MARGIN = 2 * cm


def build_pdf_report(
    sample_id: str,
    tier1: dict,
    tier2: list[dict],
    annotated_image_b64: str | None = None,
) -> bytes:
    """Generate a PDF lab report from detection results.

    Args:
        sample_id: Unique sample identifier printed on every page.
        tier1: Dict with keys ``total_plastic_count`` and ``particles_per_litre``.
        tier2: List of per-particle dicts as returned by ``compute_tier2()``.
        annotated_image_b64: Optional base64-encoded PNG of the annotated
            microscope image. Embedded in the report if provided.

    Returns:
        PDF as raw bytes, ready for ``Response(content=..., media_type='application/pdf')``.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Microplastic Detector — Laboratory Report", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Sample ID: <b>{sample_id}</b>", styles["Normal"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.6 * cm))

    # ── Tier 1 Summary ───────────────────────────────────────────────────────
    story.append(Paragraph("Tier 1 — Overall Contamination", styles["Heading2"]))
    t1_data = [
        ["Metric", "Value"],
        ["Total plastic particle count", str(tier1.get("total_plastic_count", 0))],
        ["Concentration", f"{tier1.get('particles_per_litre', 0.0):.4f} particles/L"],
    ]
    t1_table = Table(t1_data, colWidths=[9 * cm, 7 * cm])
    t1_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2E4057")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(t1_table)
    story.append(Spacer(1, 0.6 * cm))

    # ── Annotated Image ──────────────────────────────────────────────────────
    if annotated_image_b64:
        story.append(Paragraph("Annotated Filter Paper Image", styles["Heading2"]))
        try:
            img_bytes = base64.b64decode(annotated_image_b64)
            img_buf   = io.BytesIO(img_bytes)
            img       = Image(img_buf, width=14 * cm, height=10 * cm, kind="proportional")
            story.append(img)
        except Exception:
            story.append(Paragraph("(Image could not be embedded.)", styles["Normal"]))
        story.append(Spacer(1, 0.6 * cm))

    # ── Tier 2 Particle Table ────────────────────────────────────────────────
    story.append(Paragraph("Tier 2 — Per-Particle Morphotype Breakdown", styles["Heading2"]))

    if not tier2:
        story.append(Paragraph("No plastic particles detected above the confidence threshold.", styles["Normal"]))
    else:
        headers = ["ID", "Class", "Conf", "Length µm", "Width µm", "AR", "Area µm²"]
        t2_data = [headers] + [
            [
                str(p.get("particle_id", "")),
                str(p.get("class_name", "")),
                f"{float(p.get('confidence', 0)):.3f}",
                f"{float(p.get('length_um', 0)):.1f}",
                f"{float(p.get('width_um', 0)):.1f}",
                f"{float(p.get('aspect_ratio', 0)):.2f}",
                f"{float(p.get('surface_area_um2', 0)):.1f}",
            ]
            for p in tier2
        ]

        col_widths = [1.5*cm, 3*cm, 1.8*cm, 2.5*cm, 2.5*cm, 1.8*cm, 2.9*cm]
        t2_table = Table(t2_data, colWidths=col_widths, repeatRows=1)
        t2_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#048A81")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0FAFA"), colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.4, colors.grey),
            ("PADDING",        (0, 0), (-1, -1), 5),
            ("ALIGN",          (2, 1), (-1, -1), "RIGHT"),
        ]))
        story.append(t2_table)

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "Generated by Microplastic Detector v0.1 — Phase 1 (static image upload). "
        "For research use only.",
        styles["Italic"],
    ))

    doc.build(story)
    return buf.getvalue()

