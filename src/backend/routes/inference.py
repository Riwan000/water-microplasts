"""Image-upload inference endpoint (Step 4). Live streaming is Phase 2."""

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/image")
async def infer_image(file: UploadFile) -> dict:
    """Run the champion segmentation model on an uploaded filter-paper image."""
    raise NotImplementedError("Wire up champion model inference here (Step 4).")
