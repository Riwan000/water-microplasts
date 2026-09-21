"""
Step 4: FastAPI Backend Entrypoint
Run with: uvicorn src.backend.main:app --reload

Wires up the image-upload inference endpoint, OpenRouter AI diagnostic
endpoint, and PDF/CSV export endpoints behind a single ASGI app. Live
camera streaming is a Phase 2 addition — see
docs/revised-microplastic-detector-plan.md.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.config import load_settings
from src.backend.routes import assistant, export, inference

settings = load_settings()

app = FastAPI(title="Microplastic Detector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(assistant.router, prefix="/assistant", tags=["assistant"])
app.include_router(export.router, prefix="/export", tags=["export"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
