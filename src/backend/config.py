"""
Runtime settings loaded from environment variables (see .env.example).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    champion_weights_path: str
    cors_origins: list[str]


def load_settings() -> Settings:
    return Settings(
        # Optional for now (see GitHub issue: re-enable AI assistant once a key
        # is configured) — main.py only mounts the /assistant router when set.
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        champion_weights_path=os.environ.get("CHAMPION_WEIGHTS_PATH", "models/champion.pt"),
        cors_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(","),
    )
