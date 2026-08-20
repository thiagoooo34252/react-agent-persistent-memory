"""Demo end-to-end: razonamiento multi-paso, memoria persistente y ciclo de retorno.

Requiere OPENAI_API_KEY real (ver .env.example). Escribe traces/react_trace.json
y traces/react_trace.log, y al final reabre el checkpointer desde disco en un
contexto nuevo para probar que la memoria sobrevive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agent import agent_app, make_config
from settings import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

TRACE_DIR = Path(__file__).parent / "traces"
THREAD_ID = "demo-cliente-juan-perez"

TURNOS = [
    "¿Cuántos pedidos tuvo Juan Pérez y cuánto gastó en total?",
    "¿Y cuál fue su último pedido?",
    "Ahora quiero lo mismo pero para López.",
]


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Serializa un mensaje de LangChain a un dict apto para JSON."""
    entry: dict[str, Any] = {"type": message.type, "content": message.content}

    if isinstance(message, AIMessage) and message.tool_calls:
        entry["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
            for tc in message.tool_calls
        ]

    if isinstance(message, ToolMessage):
        entry["name"] = message.name
        entry["tool_call_id"] = message.tool_call_id

    return entry


async def run_turno(
    app: CompiledStateGraph,
    config: RunnableConfig,
    texto: str,
    indice: int,
    mensajes_previos: int,
) -> tuple[dict[str, Any], int]:
    """Corre un turno y devuelve su entrada de traza + el total de mensajes acumulados.

    `ainvoke` devuelve el estado completo del thread (todo lo persistido hasta
    ahora), asi que solo tomamos los mensajes nuevos de este turno para la traza.
    """
    print(f"\n--- Turno {indice} ---\nUsuario: {texto}")
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=texto)]}, config=config
    )

    todos = result["messages"]
    nuevos = todos[mensajes_previos:]
    for m in nuevos:
        resumen = m.content if m.content else getattr(m, "tool_calls", None)
        print(f"[{m.type}] {resumen}")

    entrada = {
        "turno": indice,
        "input": texto,
        "mensajes": [message_to_dict(m) for m in nuevos],
    }
    return entrada, len(todos)


def write_trace(trace: list[dict[str, Any]]) -> None:
    """Escribe la traza en traces/react_trace.json y traces/react_trace.log."""
    TRACE_DIR.mkdir(exist_ok=True)
    (TRACE_DIR / "react_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (TRACE_DIR / "react_trace.log").open("w", encoding="utf-8") as fh:
        for turno in trace:
            fh.write(json.dumps(turno, ensure_ascii=False, indent=2) + "\n")


async def main() -> None:
    settings = get_settings()
    logger.info(
        "Iniciando demo con thread_id=%s, modelo=%s", THREAD_ID, settings.openai_model
    )
    config = make_config(THREAD_ID, settings.recursion_limit)
    trace: list[dict[str, Any]] = []

    async with agent_app(settings.agent_db_path) as app:
        mensajes_previos = 0
        for i, texto in enumerate(TURNOS, start=1):
            entrada, mensajes_previos = await run_turno(
                app, config, texto, i, mensajes_previos
            )
            trace.append(entrada)

    # Reabrir el checkpointer desde disco en un contexto/proceso nuevo:
    # la prueba mas fuerte de que la memoria persiste de verdad.
    async with agent_app(settings.agent_db_path) as app_nuevo:
        estado = await app_nuevo.aget_state(config)
        cantidad = len(estado.values["messages"])
        print(f"\n--- Estado recuperado desde disco: {cantidad} mensajes ---")
        trace.append({"turno": "persistencia", "mensajes_recuperados": cantidad})

    write_trace(trace)
    print("\nTraza escrita en traces/react_trace.json y traces/react_trace.log")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
