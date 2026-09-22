"""Step 4: PDF/CSV laboratory report export endpoints."""

from fastapi import APIRouter
from fastapi.responses import Response

from src.backend.routes.inference import get_session
from src.backend.services.csv_exporter import build_csv_report
from src.backend.services.pdf_exporter import build_pdf_report

router = APIRouter()


@router.get("/pdf/{sample_id}")
async def export_pdf(sample_id: str) -> Response:
    """Download a PDF lab report for a completed inference session.

    The sample_id is returned by POST /inference/image. The report
    includes the Tier 1 summary, annotated image, and Tier 2 particle
    table with µm dimensions.
    """
    session = get_session(sample_id)
    pdf_bytes = build_pdf_report(
        sample_id=sample_id,
        tier1=session["tier1"],
        tier2=session["tier2"],
        annotated_image_b64=session.get("annotated_image_b64"),
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="microplastic_{sample_id}.pdf"'},
    )


@router.get("/csv/{sample_id}")
async def export_csv(sample_id: str) -> Response:
    """Download a CSV audit trail for a completed inference session.

    Returns the Tier 2 per-particle data as a UTF-8 CSV file. The same
    CSV is also persisted to data/sessions/{sample_id}.csv on the server.
    """
    session = get_session(sample_id)
    csv_bytes = build_csv_report(
        sample_id=sample_id,
        tier2=session["tier2"],
    )
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="microplastic_{sample_id}.csv"'},
    )

