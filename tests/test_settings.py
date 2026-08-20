"""Tests de settings.py: defaults, overrides y validacion de la API key."""

from __future__ import annotations

import pytest

import settings


def test_get_settings_falla_sin_api_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        settings.get_settings()


def test_error_de_api_key_no_filtra_ningun_valor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    try:
        settings.get_settings()
        pytest.fail("se esperaba RuntimeError")
    except RuntimeError as exc:
        assert "sk-" not in str(exc)


def test_get_settings_usa_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    resultado = settings.get_settings()

    assert resultado.openai_model == settings.DEFAULT_MODEL
    assert resultado.agent_db_path == settings.DEFAULT_DB_PATH
    assert resultado.recursion_limit == settings.DEFAULT_RECURSION_LIMIT
    assert resultado.openai_api_key == "sk-test-fake"


def test_get_settings_respeta_overrides_de_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_DB_PATH", "otra.db")
    monkeypatch.setenv("RECURSION_LIMIT", "25")

    resultado = settings.get_settings()

    assert resultado.openai_model == "gpt-4o"
    assert resultado.agent_db_path == "otra.db"
    assert resultado.recursion_limit == 25


def test_recursion_limit_invalido_lanza_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("RECURSION_LIMIT", "no-es-un-numero")

    with pytest.raises(ValueError, match="no-es-un-numero"):
        settings.get_settings()


def test_dotenv_deshabilitado_no_requiere_archivo_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PYTHON_DOTENV_DISABLED ya esta seteado por el fixture autouse de conftest.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    resultado = settings.get_settings()

    assert resultado.openai_api_key == "sk-test-fake"


def test_dotenv_habilitado_no_rompe_sin_archivo_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sin PYTHON_DOTENV_DISABLED, get_settings intenta cargar un .env; no hay
    # ninguno en el repo (solo .env.example), asi que debe seguir funcionando.
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    resultado = settings.get_settings()

    assert resultado.openai_api_key == "sk-test-fake"
