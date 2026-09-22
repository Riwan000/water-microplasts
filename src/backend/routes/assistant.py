"""Step 4: OpenRouter AI diagnostic chat endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.backend.config import load_settings
from src.backend.services.openrouter_client import ask

router = APIRouter()


class AssistantRequest(BaseModel):
    question: str
    particle_counts: dict[str, int]
    filtered_volume_litres: float
    total_plastic_count: int = 0
    particles_per_litre: float = 0.0
    quarantine_count: int = 0


class AssistantResponse(BaseModel):
    reply: str
    model: str


@router.post("/chat", response_model=AssistantResponse)
async def ask_assistant(payload: AssistantRequest) -> AssistantResponse:
    """Forward a context-aware water-safety question to OpenRouter.

    The particle counts and concentration are automatically injected into
    the system prompt so the assistant's answer is grounded in real data.
    """
    settings = load_settings()

    context = {
        "particle_counts":       payload.particle_counts,
        "total_plastic_count":   payload.total_plastic_count,
        "particles_per_litre":   payload.particles_per_litre,
        "filtered_volume_litres": payload.filtered_volume_litres,
        "quarantine_count":      payload.quarantine_count,
    }

    reply = ask(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        question=payload.question,
        context=context,
    )

    return AssistantResponse(reply=reply, model=settings.openrouter_model)

