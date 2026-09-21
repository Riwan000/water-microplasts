"""OpenRouter AI diagnostic chat endpoint (Step 4)."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AssistantRequest(BaseModel):
    question: str
    particle_counts: dict[str, int]
    filtered_volume_litres: float


@router.post("/chat")
async def ask_assistant(payload: AssistantRequest) -> dict:
    """Forward a context-aware water-safety question to OpenRouter."""
    raise NotImplementedError("Wire up the OpenRouter client call here (Step 4).")
