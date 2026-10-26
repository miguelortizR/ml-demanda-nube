"""Calidad de los datos de demanda: detección del día en curso.

La extracción puede correr a mitad de jornada y dejar el **último día a medias**
(por ejemplo, 77.939 usos un jueves que suele mover ~245.000). Ese día contamina
cualquier referencia calculada hacia atrás: la coherencia, el contexto que se le
entrega a la segunda opinión y el ancla.

Regla (pura y configurable): un día es **parcial** si su demanda queda por debajo de
una fracción de la mediana de su mismo día de la semana.
"""

from __future__ import annotations


def es_dia_parcial(
    usos: float,
    mediana_mismo_dia_semana: float,
    umbral: float = 0.5,
) -> bool:
    """Indica si un día parece estar a medio cargar.

    Args:
        usos: demanda registrada ese día.
        mediana_mismo_dia_semana: mediana de ese mismo día de la semana en la ventana
            reciente.
        umbral: fracción de la mediana por debajo de la cual se considera parcial.

    Returns:
        ``True`` si el día parece incompleto. Sin mediana válida nunca lo marca (no se
        descarta información por una duda).
    """
    if usos != usos or mediana_mismo_dia_semana != mediana_mismo_dia_semana:
        return False
    if mediana_mismo_dia_semana <= 0:
        return False
    return float(usos) < float(umbral) * float(mediana_mismo_dia_semana)
