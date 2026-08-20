"""Fixtures compartidas: sin red, sin dotenv, sin API key real filtrada entre tests."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest


def _blocked_connect(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Acceso de red bloqueado en tests")


@pytest.fixture(autouse=True)
def _no_network_no_leaked_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AGENT_DB_PATH", raising=False)
    monkeypatch.delenv("RECURSION_LIMIT", raising=False)
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect)
    yield
