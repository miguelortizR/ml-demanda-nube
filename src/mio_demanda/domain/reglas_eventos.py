"""Motor de reglas de calendario (dominio puro).

Traduce **reglas declarativas** (definidas en ``config/eventos_base.json``) a
rangos de fechas concretos para cualquier año. Así los eventos especiales del
MIO (Semana Santa, Feria de Cali, fechas comerciales, recesos escolares,
elecciones) no se "queman" en el código y funcionan en cualquier horizonte.

Tipos de regla soportados:

===============  ==========================================================
``fecha_fija``   Un día del año (mes, día), con ventana opcional.
``rango_fijo``   Un rango de días del año (mes/día inicial y final).
``pascua``       Rango relativo al Domingo de Resurrección (días de desplazamiento).
``n_esimo_dia_semana``  El n-ésimo día de la semana de un mes.
``ultimo_dia_semana``   El último día de la semana de un mes.
``fechas_puntuales``    Fechas explícitas (dependen de resoluciones o del Gobierno).
``fechas_por_anio``     Un rango explícito **por año** (eventos cuya fecha cambia
                        cada año, como el Festival Petronio Álvarez).
===============  ==========================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from mio_demanda.domain.calendario import (
    fecha_pascua,
    n_esimo_dia_semana,
    ultimo_dia_semana,
)
from mio_demanda.domain.entidades import CategoriaEvento, EventoCalendario, PerfilImpacto

#: Tipos de regla admitidos en la configuración.
TIPOS_REGLA = (
    "fecha_fija",
    "rango_fijo",
    "pascua",
    "n_esimo_dia_semana",
    "ultimo_dia_semana",
    "fechas_puntuales",
    "fechas_por_anio",
)


@dataclass(frozen=True, slots=True)
class ReglaEvento:
    """Regla declarativa que resuelve un rango de fechas para un año dado.

    Attributes:
        tipo: uno de :data:`TIPOS_REGLA`.
        parametros: parámetros específicos del tipo de regla.
    """

    tipo: str
    parametros: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS_REGLA:
            raise ValueError(f"Tipo de regla no soportado: {self.tipo}")

    def resolver(self, anio: int) -> tuple[date, date] | None:
        """Calcula el rango ``(inicio, fin)`` del evento en un año.

        Args:
            anio: año a resolver.

        Returns:
            Tupla ``(fecha_inicio, fecha_fin)`` con la ventana ya aplicada, o
            ``None`` si la regla no aplica a ese año.
        """
        anios = self.parametros.get("anios")
        if anios is not None and anio not in anios:
            return None

        rango = self._rango_base(anio)
        if rango is None:
            return None

        inicio, fin = rango
        antes = int(self.parametros.get("ventana_antes", 0))
        despues = int(self.parametros.get("ventana_despues", 0))
        return inicio - timedelta(days=antes), fin + timedelta(days=despues)

    def _rango_base(self, anio: int) -> tuple[date, date] | None:
        """Calcula el rango sin ventana, según el tipo de regla.

        Args:
            anio: año a resolver.

        Returns:
            El rango base, o ``None`` si la regla no aplica.
        """
        p = self.parametros

        if self.tipo == "fecha_fija":
            fecha = date(anio, int(p["mes"]), int(p["dia"]))
            return fecha, fecha

        if self.tipo == "rango_fijo":
            return date(anio, int(p["mes_ini"]), int(p["dia_ini"])), date(
                anio, int(p["mes_fin"]), int(p["dia_fin"])
            )

        if self.tipo == "pascua":
            base = fecha_pascua(anio)
            return (
                base + timedelta(days=int(p.get("offset_ini", 0))),
                base + timedelta(days=int(p.get("offset_fin", 0))),
            )

        if self.tipo == "n_esimo_dia_semana":
            fecha = n_esimo_dia_semana(anio, int(p["mes"]), int(p["dia_semana"]), int(p["n"]))
            return fecha, fecha

        if self.tipo == "ultimo_dia_semana":
            fecha = ultimo_dia_semana(anio, int(p["mes"]), int(p["dia_semana"]))
            return fecha, fecha

        # fechas_puntuales: solo aplica si el año consultado cae dentro del rango.
        if self.tipo == "fechas_puntuales":
            inicio = date.fromisoformat(str(p["fecha_ini"]))
            fin = date.fromisoformat(str(p["fecha_fin"]))
            if not (inicio.year <= anio <= fin.year):
                return None
            return inicio, fin

        # fechas_por_anio: cada año tiene su propio rango explícito.
        if self.tipo == "fechas_por_anio":
            fechas = p.get("fechas", {})
            rango = fechas.get(str(anio))
            if rango is None:
                return None
            return date.fromisoformat(str(rango[0])), date.fromisoformat(str(rango[1]))

        return None


def crear_regla(datos: Mapping[str, Any]) -> ReglaEvento:
    """Construye una :class:`ReglaEvento` a partir de su definición declarativa.

    Args:
        datos: diccionario con las claves ``tipo`` y los parámetros de la regla.

    Returns:
        La regla creada.

    Raises:
        KeyError: si falta la clave ``tipo``.
        ValueError: si el tipo no está soportado.
    """
    if "tipo" not in datos:
        raise KeyError("La regla debe indicar 'tipo'")
    parametros = {clave: valor for clave, valor in datos.items() if clave != "tipo"}
    return ReglaEvento(tipo=str(datos["tipo"]), parametros=parametros)


def expandir_evento(definicion: Mapping[str, Any], anios: Iterable[int]) -> list[EventoCalendario]:
    """Expande la definición de un evento a un evento por año aplicable.

    Args:
        definicion: definición declarativa del evento (``id``, ``nombre``,
            ``categoria``, ``impacto``, ``fuente`` y ``regla``).
        anios: años a resolver.

    Returns:
        Lista de :class:`EventoCalendario`, uno por año en el que la regla aplica.
    """
    regla = crear_regla(definicion["regla"])
    identificador = str(definicion["id"])
    eventos: list[EventoCalendario] = []

    for anio in sorted(set(anios)):
        rango = regla.resolver(int(anio))
        if rango is None:
            continue
        eventos.append(
            EventoCalendario(
                id=identificador,
                nombre=str(definicion.get("nombre", identificador)),
                fecha_inicio=rango[0],
                fecha_fin=rango[1],
                impacto=float(definicion.get("impacto", 0.5)),
                categoria=CategoriaEvento(definicion.get("categoria", CategoriaEvento.OTRO.value)),
                fuente=str(definicion.get("fuente", "")),
                perfil_impacto=PerfilImpacto(
                    definicion.get("perfil_impacto", PerfilImpacto.CONSTANTE.value)
                ),
            )
        )
    return eventos


def expandir_definiciones(
    definiciones: Sequence[Mapping[str, Any]], anios: Iterable[int]
) -> list[EventoCalendario]:
    """Expande una lista completa de definiciones de eventos.

    Args:
        definiciones: definiciones declarativas de eventos.
        anios: años a resolver.

    Returns:
        Lista de :class:`EventoCalendario` con todos los eventos aplicables.
    """
    anios_lista = list(anios)
    eventos: list[EventoCalendario] = []
    for definicion in definiciones:
        eventos.extend(expandir_evento(definicion, anios_lista))
    return eventos


def ids_definiciones(definiciones: Sequence[Mapping[str, Any]]) -> list[str]:
    """Devuelve los identificadores de todas las definiciones, en orden estable.

    Se usa para garantizar que el conjunto de columnas de eventos del dataset sea
    el mismo en entrenamiento y en proyección (contrato estable de variables).

    Args:
        definiciones: definiciones declarativas de eventos.

    Returns:
        Lista de identificadores sin duplicados, ordenada.
    """
    return sorted({str(definicion["id"]) for definicion in definiciones})
