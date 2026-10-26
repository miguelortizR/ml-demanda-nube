"""Reglas de calendario del dominio (cálculo de fechas y tipo de día).

Contiene únicamente matemática de fechas, sin dependencias de framework:
cálculo litúrgico de la Pascua, n-ésimo día de la semana, último día de la
semana de un mes y la clasificación *por defecto* de un día cuando el
calendario oficial no lo cubre.
"""

from __future__ import annotations

from datetime import date, timedelta

from mio_demanda.domain.entidades import TipoDia

# Festivos colombianos de fecha fija (mes, día) usados como respaldo cuando el
# calendario oficial no cubre el año consultado.
FESTIVOS_FIJOS: tuple[tuple[int, int], ...] = (
    (1, 1),  # Año Nuevo
    (5, 1),  # Día del Trabajo
    (7, 20),  # Independencia
    (8, 7),  # Batalla de Boyacá
    (12, 8),  # Inmaculada Concepción
    (12, 25),  # Navidad
)


def fecha_pascua(anio: int) -> date:
    """Calcula el Domingo de Resurrección (algoritmo gregoriano de Meeus/Butcher).

    Args:
        anio: año gregoriano.

    Returns:
        La fecha del Domingo de Resurrección de ese año.
    """
    a = anio % 19
    b = anio // 100
    c = anio % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    mes = (h + ll - 7 * m + 114) // 31
    dia = ((h + ll - 7 * m + 114) % 31) + 1
    return date(anio, mes, dia)


def n_esimo_dia_semana(anio: int, mes: int, dia_semana: int, n: int) -> date:
    """Devuelve el n-ésimo día de la semana de un mes.

    Args:
        anio: año.
        mes: mes (1-12).
        dia_semana: día de la semana con lunes=0 y domingo=6.
        n: ordinal (1 = primero, 2 = segundo, ...).

    Returns:
        La fecha correspondiente.
    """
    primero = date(anio, mes, 1)
    offset = (dia_semana - primero.weekday()) % 7
    return primero + timedelta(days=offset + 7 * (n - 1))


def ultimo_dia_semana(anio: int, mes: int, dia_semana: int) -> date:
    """Devuelve el último día de la semana de un mes.

    Args:
        anio: año.
        mes: mes (1-12).
        dia_semana: día de la semana con lunes=0 y domingo=6.

    Returns:
        La fecha correspondiente.
    """
    ultimo = date(anio, 12, 31) if mes == 12 else date(anio, mes + 1, 1) - timedelta(days=1)
    offset = (ultimo.weekday() - dia_semana) % 7
    return ultimo - timedelta(days=offset)


def es_festivo_fijo(fecha: date) -> bool:
    """Indica si una fecha corresponde a un festivo colombiano de fecha fija."""
    return (fecha.month, fecha.day) in FESTIVOS_FIJOS


def clasificar_tipo_dia(fecha: date) -> TipoDia:
    """Clasifica un día cuando el calendario oficial no lo cubre.

    Regla de respaldo: domingo y festivos de fecha fija -> FESTIVO; sábado ->
    SÁBADO; en otro caso -> HÁBIL.

    Args:
        fecha: fecha a clasificar.

    Returns:
        El :class:`TipoDia` correspondiente.
    """
    if fecha.weekday() == 6 or es_festivo_fijo(fecha):
        return TipoDia.FESTIVO
    if fecha.weekday() == 5:
        return TipoDia.SABADO
    return TipoDia.HABIL


def rango_fechas(inicio: date, fin: date) -> list[date]:
    """Genera la lista de fechas de un rango cerrado [inicio, fin].

    Args:
        inicio: fecha inicial (inclusive).
        fin: fecha final (inclusive).

    Returns:
        Lista de fechas en orden ascendente.
    """
    if fin < inicio:
        raise ValueError("fin no puede ser anterior a inicio")
    return [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]
