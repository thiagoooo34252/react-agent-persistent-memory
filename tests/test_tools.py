"""Tests de tools.py: caminos felices, ambiguedad, no-encontrado y errores internos."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tools


class TestBuscarClienteInput:
    def test_rechaza_nombre_demasiado_corto(self) -> None:
        with pytest.raises(ValidationError):
            tools.BuscarClienteInput(nombre="a")

    def test_rechaza_campos_extra(self) -> None:
        with pytest.raises(ValidationError):
            tools.BuscarClienteInput.model_validate(
                {"nombre": "Juan", "campo_extra": "no-deberia-existir"}
            )

    def test_acepta_nombre_valido(self) -> None:
        modelo = tools.BuscarClienteInput(nombre="Juan Pérez")
        assert modelo.nombre == "Juan Pérez"


class TestBuscarPedidosInput:
    def test_rechaza_id_menor_a_uno(self) -> None:
        with pytest.raises(ValidationError):
            tools.BuscarPedidosInput(cliente_id=0)

    def test_rechaza_id_fuera_de_rango(self) -> None:
        with pytest.raises(ValidationError):
            tools.BuscarPedidosInput(cliente_id=999_999)

    def test_rechaza_campos_extra(self) -> None:
        with pytest.raises(ValidationError):
            tools.BuscarPedidosInput.model_validate({"cliente_id": 1, "otro": "no"})


class TestBuscarClienteTool:
    pytestmark = pytest.mark.asyncio

    async def test_nombre_exacto(self) -> None:
        resultado = await tools.buscar_cliente.ainvoke({"nombre": "Juan Pérez"})
        assert "cliente_id=1" in resultado

    async def test_nombre_parcial_con_match_unico(self) -> None:
        resultado = await tools.buscar_cliente.ainvoke({"nombre": "Pérez"})
        assert "cliente_id=1" in resultado

    async def test_nombre_ambiguo_devuelve_candidatos(self) -> None:
        resultado = await tools.buscar_cliente.ainvoke({"nombre": "López"})
        assert "ambiguo" in resultado.lower()
        assert "María López" in resultado
        assert "Marta López" in resultado

    async def test_nombre_inexistente(self) -> None:
        resultado = await tools.buscar_cliente.ainvoke({"nombre": "Rodríguez"})
        assert "no se encontró" in resultado.lower()

    async def test_invoke_sincronico_no_soportado(self) -> None:
        with pytest.raises(NotImplementedError):
            tools.buscar_cliente.invoke({"nombre": "Juan Pérez"})

    async def test_maneja_error_interno_sin_filtrar_detalles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tools, "_CLIENTES", None)

        resultado = await tools.buscar_cliente.ainvoke({"nombre": "Juan Pérez"})

        assert "error interno" in resultado.lower()
        assert "NoneType" not in resultado
        assert "Traceback" not in resultado


class TestBuscarPedidosTool:
    pytestmark = pytest.mark.asyncio

    async def test_cliente_con_pedidos(self) -> None:
        resultado = await tools.buscar_pedidos.ainvoke({"cliente_id": 1})
        assert "2 pedido" in resultado
        assert "875" in resultado
        assert "Mouse" in resultado

    async def test_cliente_sin_pedidos(self) -> None:
        resultado = await tools.buscar_pedidos.ainvoke({"cliente_id": 5})
        assert "no tiene pedidos registrados" in resultado

    async def test_cliente_id_inexistente(self) -> None:
        resultado = await tools.buscar_pedidos.ainvoke({"cliente_id": 999})
        assert "no existe" in resultado.lower()

    async def test_invoke_sincronico_no_soportado(self) -> None:
        with pytest.raises(NotImplementedError):
            tools.buscar_pedidos.invoke({"cliente_id": 1})

    async def test_maneja_error_interno_sin_filtrar_detalles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tools, "_PEDIDOS", None)

        resultado = await tools.buscar_pedidos.ainvoke({"cliente_id": 1})

        assert "error interno" in resultado.lower()
        assert "NoneType" not in resultado
