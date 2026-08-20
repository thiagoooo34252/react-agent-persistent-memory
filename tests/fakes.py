"""Chat model fake y deterministico para testear el grafo sin llamar a OpenAI."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Devuelve, en orden, una respuesta pre-armada por cada invocacion.

    BaseChatModel.bind_tools() esta sin implementar (raise NotImplementedError)
    porque cada proveedor real lo formatea distinto; para el fake alcanza con
    devolverse a si mismo, ya que las respuestas ya vienen con tool_calls listos.
    """

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._index >= len(self.responses):
            raise AssertionError(
                "ScriptedChatModel se quedo sin respuestas programadas"
            )
        response = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def tool_call_message(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """Arma un AIMessage con un unico tool_call, mas legible en los tests."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
