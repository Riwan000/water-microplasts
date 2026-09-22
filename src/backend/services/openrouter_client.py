"""
Step 4: OpenRouter chat-completions client.

Sends a context-aware water-safety question to OpenRouter and returns
the assistant reply text. The system prompt automatically injects the
current sample's particle counts, morphotype breakdown, and concentration
so every answer is grounded in real detection data.
"""

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT_TEMPLATE = """You are an expert environmental scientist specialising in microplastic contamination analysis and water safety.

You have just analysed a water sample filtered through a membrane filter paper using an instance segmentation AI model (YOLO11s-seg). Here are the current detection results:

Sample volume filtered: {volume_litres:.4f} L
Total plastic particle count (Tier 1): {total_count}
Concentration: {concentration:.2f} particles/L

Morphotype breakdown (Tier 2):
{morphotype_summary}

Quarantine / Unclassified debris count: {quarantine_count}

Guidelines:
- Answer concisely and in plain language suitable for a lab technician.
- When discussing contamination levels, reference WHO / EFSA / GESAMP microplastic thresholds where relevant.
- Identify likely plastic sources based on morphotype distribution (e.g. fibres → synthetic textiles, fragments → bottle shards, pellets → nurdles).
- Suggest practical filtration or mitigation strategies if contamination is elevated.
- Do NOT invent particle counts or measurements beyond what is provided above.
"""


def _build_system_prompt(context: dict) -> str:
    """Render the system prompt with live detection context."""
    particle_counts: dict[str, int] = context.get("particle_counts", {})
    total_count: int = context.get("total_plastic_count", 0)
    concentration: float = context.get("particles_per_litre", 0.0)
    volume_litres: float = context.get("filtered_volume_litres", 0.0)
    quarantine_count: int = context.get("quarantine_count", 0)

    if particle_counts:
        morphotype_summary = "\n".join(
            f"  - {cls}: {count} particle(s)"
            for cls, count in sorted(particle_counts.items())
        )
    else:
        morphotype_summary = "  - No plastic particles detected."

    return _SYSTEM_PROMPT_TEMPLATE.format(
        volume_litres=volume_litres,
        total_count=total_count,
        concentration=concentration,
        morphotype_summary=morphotype_summary,
        quarantine_count=quarantine_count,
    )


def ask(api_key: str, model: str, question: str, context: dict) -> str:
    """Send a context-aware question to OpenRouter and return the reply text.

    Args:
        api_key: OpenRouter API key (from OPENROUTER_API_KEY env var).
        model: OpenRouter model string e.g. ``"google/gemini-2.0-flash-exp:free"``.
        question: The user's natural-language question about the sample.
        context: Dict with detection results to inject into the system prompt.
            Expected keys: ``particle_counts``, ``total_plastic_count``,
            ``particles_per_litre``, ``filtered_volume_litres``, ``quarantine_count``.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        RuntimeError: If the OpenRouter API returns a non-200 status.
    """
    system_prompt = _build_system_prompt(context)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
    }

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]
