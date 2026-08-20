"""Tests del grafo ReAct: ciclo agent<->tools, memoria y ciclo de retorno.

Usan ScriptedChatModel (ver tests/fakes.py) para no depender de red ni de una
API key real; el checkpointer SQLite si es el real, apuntando a un archivo
temporal, para probar persistencia de verdad.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import MessagesState

from agent import AgentState, build_graph, build_llm, make_config
from tests.fakes import ScriptedChatModel, tool_call_message


def test_build_llm_arma_un_chat_openai_sin_llamar_a_la_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    modelo = build_llm()

    assert isinstance(modelo, ChatOpenAI)
    assert modelo.model_name == "gpt-4o-mini"


def test_agent_state_hereda_de_messages_state() -> None:
    # TypedDict aplana su __mro__ en runtime, asi que la herencia declarada
    # se verifica via __orig_bases__ (lo que usa el chequeo estatico de tipos).
    assert MessagesState in AgentState.__orig_bases__
    assert "messages" in AgentState.__annotations__
    assert "tool_call_count" in AgentState.__annotations__


def test_make_config_incluye_thread_id_y_recursion_limit() -> None:
    config = make_config("thread-123", recursion_limit=7)

    assert (config.get("configurable") or {}).get("thread_id") == "thread-123"
    assert config.get("recursion_limit") == 7


def test_make_config_usa_recursion_limit_de_settings_por_defecto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("RECURSION_LIMIT", "42")

    config = make_config("thread-abc")

    assert config.get("recursion_limit") == 42


@pytest.mark.asyncio
async def test_grafo_encadena_dos_herramientas_para_resolver_la_pregunta(
    tmp_path: Path,
) -> None:
    respuestas = [
        tool_call_message("buscar_cliente", {"nombre": "Juan Pérez"}, "call_1"),
        tool_call_message("buscar_pedidos", {"cliente_id": 1}, "call_2"),
        AIMessage(content="Juan Pérez tuvo 2 pedidos y gastó $875.00 en total."),
    ]
    fake_llm = ScriptedChatModel(responses=respuestas)
    db_path = str(tmp_path / "test.db")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer, llm=fake_llm)
        config = make_config("thread-multi-paso", recursion_limit=10)

        result = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo Juan Pérez?")]},
            config=config,
        )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert "cliente_id=1" in tool_messages[0].content
    assert "875" in tool_messages[1].content
    assert (
        result["messages"][-1].content
        == "Juan Pérez tuvo 2 pedidos y gastó $875.00 en total."
    )
    assert result["tool_call_count"] == 2


@pytest.mark.asyncio
async def test_grafo_termina_sin_loop_si_no_hay_tool_calls(tmp_path: Path) -> None:
    fake_llm = ScriptedChatModel(
        responses=[AIMessage(content="Hola, ¿en qué te ayudo?")]
    )
    db_path = str(tmp_path / "test.db")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer, llm=fake_llm)
        config = make_config("thread-sin-tools", recursion_limit=10)

        result = await app.ainvoke(
            {"messages": [HumanMessage(content="hola")]}, config=config
        )

    assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert result["messages"][-1].content == "Hola, ¿en qué te ayudo?"


@pytest.mark.asyncio
async def test_ciclo_de_retorno_pide_aclaracion_ante_nombre_ambiguo(
    tmp_path: Path,
) -> None:
    respuestas = [
        tool_call_message("buscar_cliente", {"nombre": "López"}, "call_1"),
        AIMessage(
            content="Hay más de un cliente López. ¿Podés confirmar el nombre completo?"
        ),
    ]
    fake_llm = ScriptedChatModel(responses=respuestas)
    db_path = str(tmp_path / "test.db")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer, llm=fake_llm)
        config = make_config("thread-ambiguo", recursion_limit=10)

        result = await app.ainvoke(
            {"messages": [HumanMessage(content="dame los pedidos de López")]},
            config=config,
        )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "ambiguo" in str(tool_messages[0].content).lower()
    assert "confirmar" in result["messages"][-1].content.lower()


@pytest.mark.asyncio
async def test_misma_thread_id_acumula_mensajes_de_ambos_turnos(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test.db")
    config = make_config("thread-memoria", recursion_limit=10)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(
            checkpointer,
            llm=ScriptedChatModel(responses=[AIMessage(content="Gastó $875.00.")]),
        )
        await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuánto gastó Juan Pérez?")]},
            config=config,
        )

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(
            checkpointer,
            llm=ScriptedChatModel(
                responses=[AIMessage(content="Su último pedido fue un Mouse.")]
            ),
        )
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Y cuál fue su último pedido?")]},
            config=config,
        )

    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "¿Cuánto gastó Juan Pérez?"
    assert result["messages"][2].content == "¿Y cuál fue su último pedido?"


@pytest.mark.asyncio
async def test_thread_id_distinto_no_comparte_estado(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test.db")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(
            checkpointer, llm=ScriptedChatModel(responses=[AIMessage(content="ok")])
        )
        await app.ainvoke(
            {"messages": [HumanMessage(content="hola")]},
            config=make_config("thread-a", recursion_limit=10),
        )

        estado_b = await app.aget_state(make_config("thread-b", recursion_limit=10))

    assert estado_b.values == {} or not estado_b.values.get("messages")


@pytest.mark.asyncio
async def test_reapertura_del_checkpointer_desde_disco_preserva_el_estado(
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "persistente.db")
    config = make_config("thread-persistencia", recursion_limit=10)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(
            checkpointer, llm=ScriptedChatModel(responses=[AIMessage(content="listo")])
        )
        await app.ainvoke(
            {"messages": [HumanMessage(content="hola, quedate en la memoria")]},
            config=config,
        )

    # Contexto nuevo, checkpointer nuevo, mismo archivo en disco.
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer_nuevo:
        app_nuevo = build_graph(
            checkpointer_nuevo,
            llm=ScriptedChatModel(responses=[AIMessage(content="no debería usarse")]),
        )
        estado = await app_nuevo.aget_state(config)

    assert len(estado.values["messages"]) == 2
    assert estado.values["messages"][0].content == "hola, quedate en la memoria"


@pytest.mark.asyncio
async def test_build_graph_expone_los_nodos_agent_y_tools(tmp_path: Path) -> None:
    db_path = str(tmp_path / "estructura.db")
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(
            checkpointer, llm=ScriptedChatModel(responses=[AIMessage(content="ok")])
        )
        nodos = set(app.get_graph().nodes)

    assert {"agent", "tools", "__start__", "__end__"} <= nodos
