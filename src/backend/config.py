"""
Runtime settings loaded from environment variables (see .env.example).
Fails fast at import time if a required secret is missing, per the
project's "validate required secrets at startup" security rule.
"""

import os
from dataclasses import dataclass


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    champion_weights_path: str
    cors_origins: list[str]


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=_require_env("OPENROUTER_API_KEY"),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        champion_weights_path=os.environ.get("CHAMPION_WEIGHTS_PATH", "models/champion.pt"),
        cors_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(","),
    )
