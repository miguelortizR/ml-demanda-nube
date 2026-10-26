"""Entidades y enumeraciones del dominio de demanda del MIO.

Define los conceptos del negocio (tipo de día, tipo de entidad, evento de
calendario) y la **regla de clasificación** que distingue una estación de una
ruta. Nada aquí depende de un framework ni de una fuente de datos.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import IntEnum, StrEnum


class TipoDia(IntEnum):
    """Tipo de día operativo según el calendario oficial del MIO."""

    HABIL = 0
    SABADO = 1
    FESTIVO = 2

    @property
    def nombre(self) -> str:
        """Nombre legible del tipo de día (para UI y reportes)."""
        return _NOMBRES_TIPO_DIA[self]


_NOMBRES_TIPO_DIA = {
    TipoDia.HABIL: "Hábil",
    TipoDia.SABADO: "Sábado",
    TipoDia.FESTIVO: "Festivo",
}


class TipoEntidad(StrEnum):
    """Nivel de agregación de una proyección de demanda."""

    SISTEMA = "sistema"
    ESTACION = "estacion"
    RUTA = "ruta"


class CategoriaEvento(StrEnum):
    """Categoría de un evento de calendario personalizado."""

    FERIA = "feria"
    FESTIVO_LOCAL = "festivo_local"
    EVENTO_MASIVO = "evento_masivo"
    OBRA = "obra"
    PARO = "paro"
    CLIMA = "clima"
    DESASTRE = "desastre"
    OTRO = "otro"


class PerfilImpacto(StrEnum):
    """Cómo evoluciona el impacto de un evento a lo largo de sus días.

    Un sismo, por ejemplo, golpea con fuerza el primer día y su efecto se
    desvanece con el tiempo (``DECRECIENTE``); una feria puede crecer hasta su
    día central (``CRECIENTE``); un cierre programado mantiene su efecto
    (``CONSTANTE``).
    """

    CONSTANTE = "constante"
    DECRECIENTE = "decreciente"
    CRECIENTE = "creciente"

    def factor(self, posicion: int, total_dias: int) -> float:
        """Peso relativo del día ``posicion`` (base 0) dentro del evento.

        Args:
            posicion: índice del día dentro del evento (0 = primer día).
            total_dias: duración total del evento en días.

        Returns:
            Factor entre 0.1 y 1.0 que multiplica el impacto declarado.
        """
        if total_dias <= 1:
            return 1.0
        avance = posicion / (total_dias - 1)
        if self is PerfilImpacto.DECRECIENTE:
            # Del 100 % del impacto al 10 % en el último día.
            return 1.0 - 0.9 * avance
        if self is PerfilImpacto.CRECIENTE:
            return 0.1 + 0.9 * avance
        return 1.0


# Prefijos oficiales de las rutas del MIO (ver guía de datos):
#   A -> Alimentadora, P/C -> Pretroncal, T/E -> Troncal/Expresa.
_PREFIJOS_RUTA = ("A", "T", "P", "C", "E")
_PATRON_RUTA = re.compile(rf"^[{'|'.join(_PREFIJOS_RUTA)}]-?\d{{1,3}}[A-Z]?$", re.IGNORECASE)


def _normalizar(texto: str) -> str:
    """Quita acentos, espacios y mayúsculas para comparar identificadores."""
    sin_acentos = unicodedata.normalize("NFKD", str(texto))
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return sin_acentos.strip().upper().replace(" ", "")


def clasificar_entidad(nombre: str) -> TipoEntidad:
    """Determina si un identificador pertenece a una ruta o a una estación.

    La columna ``Nombre_estacion`` de los usos mezcla estaciones (nombres
    largos) y rutas (códigos cortos con prefijo alfabético). Esta regla
    centraliza el criterio para no repetirlo en cada consulta.

    Args:
        nombre: identificador crudo leído de la fuente de usos.

    Returns:
        TipoEntidad.RUTA si el identificador sigue el patrón de ruta,
        TipoEntidad.ESTACION en caso contrario.
    """
    limpio = _normalizar(nombre)
    if not limpio:
        return TipoEntidad.ESTACION
    return TipoEntidad.RUTA if _PATRON_RUTA.match(limpio) else TipoEntidad.ESTACION


@dataclass(frozen=True, slots=True)
class EventoCalendario:
    """Evento de calendario definido por el profesional.

    Los eventos personalizados permiten explicar picos o caídas de demanda
    (ferias, obras, paros, sismos) que el calendario oficial no contempla, y se
    convierten en variables del modelo.

    Attributes:
        id: identificador estable en formato slug.
        nombre: nombre legible del evento.
        fecha_inicio: primer día del evento (inclusive).
        fecha_fin: último día del evento (inclusive).
        impacto: peso esperado del ajuste homólogo, entre 0.0 y 1.0.
        categoria: categoría del evento.
        fuente: justificación o fuente del evento.
        perfil_impacto: cómo evoluciona el impacto a lo largo de los días.
    """

    id: str
    nombre: str
    fecha_inicio: date
    fecha_fin: date
    impacto: float = 0.5
    categoria: CategoriaEvento = CategoriaEvento.OTRO
    fuente: str = ""
    perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE

    def __post_init__(self) -> None:
        if self.fecha_fin < self.fecha_inicio:
            raise ValueError("fecha_fin no puede ser anterior a fecha_inicio")
        if not 0.0 <= self.impacto <= 1.0:
            raise ValueError("impacto debe estar entre 0.0 y 1.0")

    @property
    def dias(self) -> int:
        """Duración del evento en días (inclusive)."""
        return (self.fecha_fin - self.fecha_inicio).days + 1

    def fechas(self) -> list[date]:
        """Lista todas las fechas que cubre el evento."""
        return [self.fecha_inicio + timedelta(days=i) for i in range(self.dias)]

    def peso_en(self, fecha: date) -> float:
        """Peso del impacto del evento en una fecha concreta.

        Args:
            fecha: día a evaluar.

        Returns:
            ``impacto`` multiplicado por el factor del perfil; 0.0 si la fecha
            queda fuera del evento.
        """
        if not self.fecha_inicio <= fecha <= self.fecha_fin:
            return 0.0
        posicion = (fecha - self.fecha_inicio).days
        return self.impacto * self.perfil_impacto.factor(posicion, self.dias)

    def se_solapa(self, otra_inicio: date, otra_fin: date) -> bool:
        """Indica si el evento se solapa con un rango de fechas dado."""
        return not (self.fecha_fin < otra_inicio or self.fecha_inicio > otra_fin)


@dataclass(slots=True)
class ResumenProyeccion:
    """Resultado agregado de una proyección de demanda.

    Attributes:
        etiqueta: descripción del alcance (p. ej. "Sistema MIO" o "Ruta T31").
        dias: cantidad de días proyectados.
        usos_totales: suma de usos estimados en el horizonte.
        usos_diarios: promedio diario de usos.
        margen_error: error esperado del horizonte en usos (semibanda P10-P90 estimada
            a partir del RMSE del modelo).
        detalle_por_tipo_dia: mapa tipo de día -> métricas parciales.
    """

    etiqueta: str
    dias: int
    usos_totales: float
    usos_diarios: float
    margen_error: float = 0.0
    detalle_por_tipo_dia: dict[str, dict[str, float]] = field(default_factory=dict)
