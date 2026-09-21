"""PDF/CSV laboratory report export endpoints (Step 4)."""

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


@router.get("/pdf/{sample_id}")
async def export_pdf(sample_id: str) -> Response:
    raise NotImplementedError("Wire up ReportLab PDF export here (Step 4).")


@router.get("/csv/{sample_id}")
async def export_csv(sample_id: str) -> Response:
    raise NotImplementedError("Wire up pandas CSV export here (Step 4).")
