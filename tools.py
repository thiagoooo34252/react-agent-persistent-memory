"""Herramientas del agente sobre un dataset ficticio de clientes y pedidos.

Simulan una consulta a una base de datos real. Para resolver "cuantos pedidos
tuvo <nombre> y cuanto gasto" el agente necesita encadenar buscar_cliente
(nombre -> id) y despues buscar_pedidos (id -> metricas).
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Cliente(TypedDict):
    id: int
    nombre: str


class Pedido(TypedDict):
    id: int
    producto: str
    monto: float
    fecha: str


_CLIENTES: list[Cliente] = [
    {"id": 1, "nombre": "Juan Pérez"},
    {"id": 2, "nombre": "Ana Gómez"},
    {"id": 3, "nombre": "María López"},
    {"id": 4, "nombre": "Marta López"},
    {"id": 5, "nombre": "Carlos Fernández"},
]

_PEDIDOS: dict[int, list[Pedido]] = {
    1: [
        {"id": 101, "producto": "Notebook", "monto": 850.0, "fecha": "2026-03-02"},
        {"id": 108, "producto": "Mouse", "monto": 25.0, "fecha": "2026-05-14"},
    ],
    2: [
        {"id": 110, "producto": "Monitor", "monto": 300.0, "fecha": "2026-06-01"},
    ],
    5: [],
}


class BuscarClienteInput(BaseModel):
    """Parametros para buscar un cliente por nombre parcial o completo."""

    model_config = ConfigDict(extra="forbid")

    nombre: str = Field(
        min_length=2,
        max_length=100,
        description="Nombre o parte del nombre del cliente a buscar.",
    )


class BuscarPedidosInput(BaseModel):
    """Parametros para consultar los pedidos de un cliente ya identificado."""

    model_config = ConfigDict(extra="forbid")

    cliente_id: int = Field(
        ge=1,
        le=10_000,
        description="ID numerico del cliente, obtenido previamente con buscar_cliente.",
    )


@tool(args_schema=BuscarClienteInput)
async def buscar_cliente(nombre: str) -> str:
    """Resuelve un nombre (parcial o completo) a un cliente_id exacto.

    Usala SIEMPRE antes de buscar_pedidos cuando el usuario menciona un
    cliente por nombre: buscar_pedidos necesita el id numerico, no el nombre.
    Si el nombre es ambiguo o no coincide con nadie, la herramienta lo dice
    en el propio resultado en vez de fallar en silencio.
    No la uses si ya tenes el cliente_id de un turno anterior.
    """
    try:
        nombre_normalizado = nombre.strip().lower()

        exactos = [c for c in _CLIENTES if c["nombre"].lower() == nombre_normalizado]
        if exactos:
            cliente = exactos[0]
            return (
                f"Cliente encontrado: {cliente['nombre']} (cliente_id={cliente['id']})"
            )

        nombres = [c["nombre"] for c in _CLIENTES]
        por_substring = [n for n in nombres if nombre_normalizado in n.lower()]
        parecidos = get_close_matches(nombre, nombres, n=5, cutoff=0.5)
        candidatos = list(dict.fromkeys(por_substring + parecidos))

        if not candidatos:
            return (
                f"No se encontró ningún cliente parecido a '{nombre}'. "
                "Pedile al usuario que confirme el nombre."
            )

        if len(candidatos) == 1 and por_substring:
            cliente = next(c for c in _CLIENTES if c["nombre"] == candidatos[0])
            return (
                f"Cliente encontrado: {cliente['nombre']} (cliente_id={cliente['id']})"
            )

        lista = ", ".join(candidatos)
        return (
            f"El nombre '{nombre}' es ambiguo o no es exacto. Candidatos posibles: "
            f"{lista}. Pedile al usuario que precise cual de estos es, o reintentá "
            "con el nombre completo."
        )
    except Exception:
        logger.exception("Fallo inesperado buscando cliente")
        return (
            "Ocurrió un error interno buscando el cliente. Reintentá con otro nombre."
        )


@tool(args_schema=BuscarPedidosInput)
async def buscar_pedidos(cliente_id: int) -> str:
    """Devuelve cantidad de pedidos, total gastado y ultimo pedido de un cliente.

    Requiere el cliente_id numerico exacto (usá buscar_cliente primero si solo
    tenés el nombre). No la uses para buscar por nombre.
    """
    try:
        cliente = next((c for c in _CLIENTES if c["id"] == cliente_id), None)
        if cliente is None:
            return (
                f"No existe ningún cliente con cliente_id={cliente_id}. "
                "Verificá el id con buscar_cliente."
            )

        pedidos = _PEDIDOS.get(cliente_id, [])
        if not pedidos:
            return (
                f"{cliente['nombre']} (cliente_id={cliente_id}) no tiene "
                "pedidos registrados."
            )

        total = sum(p["monto"] for p in pedidos)
        ultimo = max(pedidos, key=lambda p: p["fecha"])
        return (
            f"{cliente['nombre']} tiene {len(pedidos)} pedido(s), gastó un total de "
            f"${total:,.2f}. Último pedido: {ultimo['producto']} por "
            f"${ultimo['monto']:,.2f} el {ultimo['fecha']}."
        )
    except Exception:
        logger.exception("Fallo inesperado buscando pedidos")
        return (
            "Ocurrió un error interno consultando los pedidos. "
            "Reintentá en unos segundos."
        )
