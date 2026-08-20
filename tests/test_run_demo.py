"""Tests del serializador de trazas de run_demo.py, con mensajes armados a mano."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agent
import run_demo
from run_demo import THREAD_ID, TURNOS, message_to_dict
from tests.fakes import ScriptedChatModel, tool_call_message


def test_human_message_se_serializa_con_type_y_content() -> None:
    entrada = message_to_dict(HumanMessage(content="¿Cuánto gastó Juan?"))

    assert entrada == {"type": "human", "content": "¿Cuánto gastó Juan?"}


def test_ai_message_sin_tool_calls_no_incluye_la_clave() -> None:
    entrada = message_to_dict(AIMessage(content="Gastó $875.00."))

    assert entrada["type"] == "ai"
    assert "tool_calls" not in entrada


def test_ai_message_con_tool_calls_los_incluye_completos() -> None:
    mensaje = AIMessage(
        content="",
        tool_calls=[
            {"name": "buscar_cliente", "args": {"nombre": "Juan Pérez"}, "id": "call_1"}
        ],
    )

    entrada = message_to_dict(mensaje)

    assert entrada["tool_calls"] == [
        {"name": "buscar_cliente", "args": {"nombre": "Juan Pérez"}, "id": "call_1"}
    ]


def test_tool_message_incluye_name_y_tool_call_id() -> None:
    mensaje = ToolMessage(
        content="Cliente encontrado: Juan Pérez (cliente_id=1)",
        name="buscar_cliente",
        tool_call_id="call_1",
    )

    entrada = message_to_dict(mensaje)

    assert entrada["type"] == "tool"
    assert entrada["name"] == "buscar_cliente"
    assert entrada["tool_call_id"] == "call_1"


def test_system_message_no_agrega_claves_extra() -> None:
    entrada = message_to_dict(SystemMessage(content="prompt del sistema"))

    assert set(entrada.keys()) == {"type", "content"}


def test_resultado_es_serializable_a_json() -> None:
    mensajes = [
        HumanMessage(content="hola"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "buscar_cliente", "args": {"nombre": "Juan"}, "id": "1"}
            ],
        ),
        ToolMessage(content="ok", name="buscar_cliente", tool_call_id="1"),
    ]

    dump = json.dumps([message_to_dict(m) for m in mensajes], ensure_ascii=False)

    assert "buscar_cliente" in dump


def test_turnos_y_thread_id_configurados_para_la_demo() -> None:
    assert len(TURNOS) == 3
    assert THREAD_ID
    assert all(isinstance(t, str) and t for t in TURNOS)


@pytest.mark.asyncio
async def test_main_corre_los_tres_turnos_y_escribe_traza(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corre run_demo.main() de punta a punta con un LLM scripteado, sin red.

    Las respuestas siguen el mismo guion que la demo real: multi-paso,
    memoria (sin tool call) y ciclo de retorno ante nombre ambiguo.
    """
    respuestas = [
        tool_call_message("buscar_cliente", {"nombre": "Juan Pérez"}, "call_1"),
        tool_call_message("buscar_pedidos", {"cliente_id": 1}, "call_2"),
        AIMessage(content="Juan Pérez tuvo 2 pedidos y gastó $875.00 en total."),
        AIMessage(content="Su último pedido fue un Mouse por $25.00."),
        tool_call_message("buscar_cliente", {"nombre": "López"}, "call_3"),
        AIMessage(content="Hay más de un López. ¿Podés confirmar el nombre completo?"),
    ]
    fake_llm = ScriptedChatModel(responses=respuestas)
    monkeypatch.setattr(agent, "build_llm", lambda: fake_llm)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(run_demo, "TRACE_DIR", tmp_path / "traces")

    await run_demo.main()

    trace_json = json.loads((tmp_path / "traces" / "react_trace.json").read_text())
    assert (tmp_path / "traces" / "react_trace.log").exists()
    assert [t["turno"] for t in trace_json] == [1, 2, 3, "persistencia"]
    # turno 1: human + (ai tool_call, tool) x2 + respuesta final = 6 mensajes nuevos
    assert len(trace_json[0]["mensajes"]) == 6
    # turno 2: human + respuesta final, sin tool call nuevo (memoria)
    assert len(trace_json[1]["mensajes"]) == 2
    # turno 3: human + ai tool_call + tool + respuesta final = 4 mensajes nuevos
    assert len(trace_json[2]["mensajes"]) == 4
    assert trace_json[-1]["mensajes_recuperados"] == 12
