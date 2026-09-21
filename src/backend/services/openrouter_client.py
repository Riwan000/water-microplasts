"""Thin client for the OpenRouter chat-completions API (Step 4)."""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def ask(api_key: str, model: str, question: str, context: dict) -> str:
    """Send a context-aware question to OpenRouter and return the reply text."""
    raise NotImplementedError("Wire up the OpenRouter chat-completions call here (Step 4).")
