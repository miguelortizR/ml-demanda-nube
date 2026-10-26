"""Métricas de confiabilidad del dominio (puro).

Los umbrales de negocio viven aquí como **constantes con nombre** y se pueden
inyectar en las funciones, de modo que ninguna cifra quede "quemada" dentro de la
lógica: si el negocio cambia el criterio, se ajusta en un solo lugar.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Confiabilidad mínima y máxima que se reporta (política de negocio).
CONFIABILIDAD_MINIMA = 0.50
CONFIABILIDAD_MAXIMA = 0.98

#: Umbrales de las etiquetas cualitativas: (desde, etiqueta), de mayor a menor.
UMBRALES_CONFIABILIDAD: tuple[tuple[float, str], ...] = (
    (0.90, "Alta"),
    (0.80, "Media"),
    (0.70, "Baja"),
)

#: Etiqueta cuando la confiabilidad está por debajo del último umbral.
ETIQUETA_CONFIABILIDAD_MINIMA = "Muy baja"


#: Confiabilidad cuando el error no es calculable (punto medio del rango válido).
CONFIABILIDAD_POR_DEFECTO = (CONFIABILIDAD_MINIMA + CONFIABILIDAD_MAXIMA) / 2


def confiabilidad_desde_error(
    error_porcentual: float,
    minimo: float = CONFIABILIDAD_MINIMA,
    maximo: float = CONFIABILIDAD_MAXIMA,
) -> float:
    """Convierte un error porcentual en un índice de confiabilidad.

    Se le pasa el **WAPE** (error porcentual absoluto ponderado), que es el
    indicador robusto para demanda, aunque acepta cualquier error porcentual.

    Args:
        error_porcentual: error en tanto por ciento (7.5 = 7.5 % de error).
        minimo: confiabilidad mínima a reportar.
        maximo: confiabilidad máxima a reportar.

    Returns:
        Confiabilidad en tanto por uno, acotada a ``[minimo, maximo]``.
    """
    if error_porcentual != error_porcentual:  # NaN
        return CONFIABILIDAD_POR_DEFECTO
    return max(minimo, min(maximo, 1.0 - error_porcentual / 100.0))


def clasificar_confiabilidad(
    valor: float,
    umbrales: Sequence[tuple[float, str]] = UMBRALES_CONFIABILIDAD,
) -> str:
    """Traduce un índice de confiabilidad a una etiqueta cualitativa.

    Args:
        valor: confiabilidad en [0, 1].
        umbrales: pares ``(desde, etiqueta)`` ordenados de mayor a menor.

    Returns:
        La etiqueta correspondiente (``"Alta"``, ``"Media"``, ``"Baja"`` o la de
        :data:`ETIQUETA_CONFIABILIDAD_MINIMA`).
    """
    for desde, etiqueta in umbrales:
        if valor >= desde:
            return etiqueta
    return ETIQUETA_CONFIABILIDAD_MINIMA
