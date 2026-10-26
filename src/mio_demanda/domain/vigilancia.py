"""Vigilancia del modelo: ¿sigue sirviendo lo que entrenamos?

Reglas **puras**: traducen el error medido contra la demanda real (la retrospectiva)
en un veredicto que el equipo entiende. La verdad la da la demanda observada; ningún
modelo externo decide el veredicto.

Umbrales (todos configurables con ``MIO_*``):

* el WAPE reciente no debería superar ``factor_alerta`` veces el de validación;
* el sesgo sostenido (sobreestimar o subestimar) tampoco debería pasar de
  ``sesgo_alerta``;
* con menos de ``min_dias`` días reales no se emite juicio.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class EstadoModelo(StrEnum):
    """Veredicto del estado de un modelo entrenado."""

    AL_DIA = "al_dia"
    DESAJUSTADO = "desajustado"
    SIN_DATOS = "sin_datos"

    @property
    def etiqueta(self) -> str:
        """Nombre legible del estado."""
        return {
            EstadoModelo.AL_DIA: "Al día",
            EstadoModelo.DESAJUSTADO: "Desajustado",
            EstadoModelo.SIN_DATOS: "Sin datos suficientes",
        }[self]


def sesgo_relativo(predicho: Sequence[float], real: Sequence[float]) -> float:
    """Sesgo relativo: cuánto se desvía la predicción de la demanda real, en conjunto.

    Un valor positivo indica que el modelo **sobreestima** de forma sostenida (el caso
    más peligroso: planifica de más todos los días en la misma dirección).

    Args:
        predicho: valores proyectados.
        real: valores observados, alineados con los anteriores.

    Returns:
        El sesgo en tanto por uno; ``0.0`` si no hay demanda real.
    """
    total_real = sum(float(valor) for valor in real)
    if total_real == 0:
        return 0.0
    total_predicho = sum(float(valor) for valor in predicho)
    return (total_predicho - total_real) / total_real


def evaluar_desajuste(
    *,
    wape_reciente: float | None,
    wape_validacion: float | None,
    sesgo: float | None = None,
    dias_evaluados: int,
    min_dias: int,
    factor_alerta: float,
    sesgo_alerta: float,
) -> tuple[EstadoModelo, str]:
    """Veredicto sobre el modelo a partir de su error reciente.

    Args:
        wape_reciente: WAPE de los días ya observados.
        wape_validacion: WAPE de la validación del modelo guardado.
        sesgo: sesgo relativo de la ventana (positivo = sobreestima).
        dias_evaluados: días reales con los que se juzga.
        min_dias: mínimo de días para emitir juicio.
        factor_alerta: veces que el WAPE reciente puede superar al de validación.
        sesgo_alerta: sesgo absoluto que dispara la alerta.

    Returns:
        Tupla ``(estado, motivo)`` con el texto explícito para el equipo.
    """
    if dias_evaluados < min_dias or wape_reciente is None:
        return (
            EstadoModelo.SIN_DATOS,
            f"Solo hay {dias_evaluados} día(s) con demanda real posterior al corte; "
            f"se necesitan al menos {min_dias} para juzgar el modelo.",
        )

    if wape_validacion and wape_validacion > 0:
        limite = factor_alerta * wape_validacion
        if wape_reciente > limite:
            return (
                EstadoModelo.DESAJUSTADO,
                f"El error reciente (WAPE {wape_reciente:.1f} %) es "
                f"{wape_reciente / wape_validacion:.1f} veces el de la validación "
                f"({wape_validacion:.1f} %). Conviene reentrenar con la historia reciente.",
            )

    if sesgo is not None and sesgo_alerta and abs(sesgo) > sesgo_alerta:
        direccion = "sobreestima" if sesgo > 0 else "subestima"
        return (
            EstadoModelo.DESAJUSTADO,
            f"El modelo {direccion} la demanda en {abs(sesgo):.1%} de forma sostenida "
            f"({dias_evaluados} días). Revisa eventos no registrados o reentrena.",
        )

    detalle = f"WAPE {wape_reciente:.1f} %"
    if wape_validacion:
        detalle += f" frente a {wape_validacion:.1f} % de la validación"
    return (
        EstadoModelo.AL_DIA,
        f"El error de los últimos días ({detalle}) está en línea con lo esperado.",
    )
