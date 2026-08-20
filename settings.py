"""Configuracion centralizada, leida desde variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_DB_PATH = "checkpoints.db"
DEFAULT_RECURSION_LIMIT = 10


@dataclass(frozen=True)
class Settings:
    """Configuracion resuelta del agente."""

    openai_api_key: str
    openai_model: str
    agent_db_path: str
    recursion_limit: int


def _load_dotenv_if_enabled() -> None:
    if os.getenv("PYTHON_DOTENV_DISABLED") == "1":
        return
    from dotenv import load_dotenv

    load_dotenv()


def get_settings() -> Settings:
    """Arma la configuracion desde el entorno. Nunca loguea ni expone la key."""
    _load_dotenv_if_enabled()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Definila como variable de entorno o en un "
            "archivo .env (ver .env.example)."
        )

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        agent_db_path=os.getenv("AGENT_DB_PATH", DEFAULT_DB_PATH),
        recursion_limit=int(os.getenv("RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT))),
    )
