"""Grafo ReAct (StateGraph) con memoria persistente via checkpointer SQLite.

Ver la nota sobre el checkpointer en el README: se usa AsyncSqliteSaver
(no SqliteSaver) porque el codigo corre con ainvoke/astream.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

from settings import get_settings
from tools import buscar_cliente, buscar_pedidos

logger = logging.getLogger(__name__)

TOOLS = [buscar_cliente, buscar_pedidos]

SYSTEM_PROMPT = (
    "Sos un asistente de atencion al cliente. Para responder preguntas sobre "
    "pedidos de un cliente necesitas primero su cliente_id: usa buscar_cliente "
    "con el nombre que te dieron, y recien despues buscar_pedidos con el id "
    "que obtuviste. Si ya conoces el cliente_id de un turno anterior de esta "
    "misma conversacion, no vuelvas a buscarlo. "
    "Si una herramienta te devuelve un resultado ambiguo, incompleto o un "
    "error, no inventes datos: pedile al usuario que precise la informacion "
    "o reintenta con un dato mejor."
)


class AgentState(MessagesState):
    """Estado del grafo. Hereda `messages` de MessagesState.

    `tool_call_count` es informativo: cuenta cuantas herramientas ya se
    ejecutaron en la conversacion, util para inspeccionar el razonamiento
    multi-paso en la traza sin tener que contar mensajes a mano.
    """

    tool_call_count: int


def build_llm() -> BaseChatModel:
    """Crea el chat model real (OpenAI) a partir de la configuracion del entorno."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=0,
    )


def build_graph(
    checkpointer: BaseCheckpointSaver, llm: BaseChatModel | None = None
) -> CompiledStateGraph:
    """Arma y compila el grafo ReAct: agent <-> tools, con checkpointer inyectado."""
    model = (llm or build_llm()).bind_tools(TOOLS)

    async def agent_node(state: AgentState) -> dict[str, object]:
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = await model.ainvoke(messages)
        tool_call_count = sum(
            1 for m in state["messages"] if isinstance(m, ToolMessage)
        )
        return {"messages": [response], "tool_call_count": tool_call_count}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def agent_app(
    db_path: str, llm: BaseChatModel | None = None
) -> AsyncIterator[CompiledStateGraph]:
    """Abre el checkpointer SQLite y entrega el grafo compilado listo para usar."""
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        yield build_graph(checkpointer, llm=llm)


def make_config(thread_id: str, recursion_limit: int | None = None) -> RunnableConfig:
    """Arma el RunnableConfig con thread_id (memoria) y recursion_limit."""
    limit = (
        recursion_limit
        if recursion_limit is not None
        else get_settings().recursion_limit
    )
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=limit,
    )
