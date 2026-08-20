"""Genera un trace de ejemplo para el repo SIN necesitar una OPENAI_API_KEY real.

Corre el mismo grafo y las mismas herramientas de agent.py/tools.py, pero con
un chat model scripteado (tests/fakes.py) en lugar de OpenAI. Sirve para dejar
un ejemplo de traza commiteado; correr run_demo.py con una key real produce
una traza equivalente pero con razonamiento genuino del modelo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage

import run_demo
from agent import agent_app, make_config
from tests.fakes import ScriptedChatModel, tool_call_message

EXAMPLE_DB_PATH = "traces/_example_checkpoints.db"

RESPUESTAS = [
    tool_call_message("buscar_cliente", {"nombre": "Juan Pérez"}, "call_1"),
    tool_call_message("buscar_pedidos", {"cliente_id": 1}, "call_2"),
    AIMessage(content="Juan Pérez tuvo 2 pedidos y gastó un total de $875.00."),
    AIMessage(content="Su último pedido fue un Mouse por $25.00 el 2026-05-14."),
    tool_call_message("buscar_cliente", {"nombre": "López"}, "call_3"),
    AIMessage(
        content=(
            "Encontré más de un cliente con ese apellido: María López y Marta "
            "López. ¿Podés confirmarme el nombre completo para poder buscar "
            "sus pedidos?"
        )
    ),
]


async def main() -> None:
    fake_llm = ScriptedChatModel(responses=RESPUESTAS)
    config = make_config(run_demo.THREAD_ID, recursion_limit=10)
    trace: list[dict[str, object]] = [
        {
            "nota": (
                "Traza generada offline con un modelo scripteado "
                "(generate_example_trace.py), sin llamar a OpenAI. El grafo, "
                "el checkpointer y las herramientas son los reales del "
                "repo. Correr run_demo.py con OPENAI_API_KEY real produce "
                "una traza equivalente con razonamiento genuino del modelo."
            )
        }
    ]

    async with agent_app(EXAMPLE_DB_PATH, llm=fake_llm) as app:
        mensajes_previos = 0
        for i, texto in enumerate(run_demo.TURNOS, start=1):
            entrada, mensajes_previos = await run_demo.run_turno(
                app, config, texto, i, mensajes_previos
            )
            trace.append(entrada)

    # Mismo fake_llm: aget_state no invoca al modelo, pero build_graph igual
    # necesita construir uno (aunque no se termine llamando) para bindear tools.
    async with agent_app(EXAMPLE_DB_PATH, llm=fake_llm) as app_nuevo:
        estado = await app_nuevo.aget_state(config)
        trace.append(
            {
                "turno": "persistencia",
                "mensajes_recuperados": len(estado.values["messages"]),
            }
        )

    run_demo.write_trace(trace)
    Path(EXAMPLE_DB_PATH).unlink(missing_ok=True)
    print("Trace de ejemplo (modelo scripteado, offline) escrito en traces/.")


if __name__ == "__main__":
    asyncio.run(main())
