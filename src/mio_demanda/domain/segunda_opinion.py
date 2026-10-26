"""Segunda opinión de demanda: traducción de la rúbrica a usos.

Cuando se le pide a un modelo externo (Jev) que estime la demanda, la respuesta no es
un número libre sino una **puntuación sobre una rúbrica ordenada de niveles**. Aquí se
define esa rúbrica y se traduce la respuesta a usos por día, incluida su banda.

Todo este módulo es **puro**: no hay red, ni configuración, ni estado.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Límite de niveles que acepta la API de la segunda opinión.
MAX_NIVELES_API = 10
#: Niveles por defecto de la rúbrica (impar, para que quede centrada en el ancla).
NIVELES_POR_DEFECTO = 9
#: Amplitud por defecto a cada lado del ancla (±15 %).
AMPLITUD_POR_DEFECTO = 0.15
#: Cuantiles de la banda que se reporta junto a la predicción.
CUANTIL_INFERIOR = 0.10
CUANTIL_SUPERIOR = 0.90


def construir_rubrica(
    ancla: float,
    niveles: int = NIVELES_POR_DEFECTO,
    amplitud: float = AMPLITUD_POR_DEFECTO,
) -> list[int]:
    """Niveles de demanda (en usos) con los que se consulta al modelo externo.

    La rúbrica se centra en el nivel reciente y se abre ``amplitud`` a cada lado, de
    modo que sirve igual para el sistema completo (cientos de miles de usos) que para
    una ruta pequeña (decenas de usos).

    Args:
        ancla: nivel reciente de demanda en usos (el centro de la rúbrica).
        niveles: número de niveles; se acota al máximo que admite la API.
        amplitud: fracción a cada lado del ancla (0.15 = ±15 %).

    Returns:
        Lista ordenada y creciente de niveles en usos (enteros, sin repetidos).
    """
    niveles = min(max(3, int(niveles)), MAX_NIVELES_API)
    centro = max(float(ancla), 1.0)
    limites = np.linspace(1.0 - float(amplitud), 1.0 + float(amplitud), niveles)
    valores = np.maximum(np.round(centro * limites), 0.0).astype(int)

    # Sin repetidos (con demandas muy pequeñas los niveles pueden colapsar).
    unicos: list[int] = []
    for valor in valores.tolist():
        if not unicos or valor > unicos[-1]:
            unicos.append(int(valor))
    return unicos if len(unicos) >= 3 else [0, int(centro), int(centro) * 2 + 1]


def usos_desde_puntuacion(puntuacion: float, rubrica: Sequence[int]) -> float:
    """Convierte la puntuación de la rúbrica en usos por interpolación lineal.

    La puntuación es el promedio ponderado de los niveles (puede caer entre dos).

    Args:
        puntuacion: puntuación devuelta por el modelo, en el rango de índices.
        rubrica: niveles en usos.

    Returns:
        La demanda estimada en usos.
    """
    valores = np.asarray(list(rubrica), dtype=float)
    if valores.size == 0:
        return 0.0
    if valores.size == 1:
        return float(valores[0])
    indice = float(np.clip(puntuacion, 0.0, valores.size - 1))
    return float(np.interp(indice, np.arange(valores.size), valores))


def banda_desde_probabilidades(
    probabilidades: Sequence[float],
    rubrica: Sequence[int],
    inferior: float = CUANTIL_INFERIOR,
    superior: float = CUANTIL_SUPERIOR,
) -> tuple[float, float]:
    """Banda (P10-P90 por defecto) de la distribución que devolvió el modelo.

    Se usa el **cuantil discreto**: el primer nivel cuya probabilidad acumulada
    alcanza el cuantil. Es la lectura honesta de una distribución sobre niveles
    (interpolar con probabilidades repetidas daría valores engañosos).

    Args:
        probabilidades: probabilidad de cada nivel, en el orden de la rúbrica.
        rubrica: niveles en usos.
        inferior: cuantil inferior de la banda.
        superior: cuantil superior de la banda.

    Returns:
        Tupla ``(usos_p10, usos_p90)``; si las probabilidades no sirven, el rango
        completo de la rúbrica.
    """
    valores = np.asarray(list(rubrica), dtype=float)
    if valores.size == 0:
        return 0.0, 0.0
    extremos = (float(valores[0]), float(valores[-1]))
    pesos = np.asarray(list(probabilidades), dtype=float)
    if pesos.size != valores.size or not np.isfinite(pesos).all() or pesos.sum() <= 0:
        return extremos

    acumulada = np.cumsum(pesos / pesos.sum())
    limite = valores.size - 1
    return (
        float(valores[min(int(np.searchsorted(acumulada, inferior)), limite)]),
        float(valores[min(int(np.searchsorted(acumulada, superior)), limite)]),
    )
