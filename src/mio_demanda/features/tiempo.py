"""Características de tiempo: anclas históricas y estacionalidad sin fuga.

El "ancla histórica" resume el comportamiento pasado de una serie (por ejemplo,
la demanda media reciente de una estación). Se calcula con **desplazamiento**
(``shift``) para que cada fila use únicamente información anterior a su propia
fecha; así se evita el *data leakage* al entrenar y al proyectar.
"""

from __future__ import annotations

import pandas as pd


def mediana_movil_pasada(
    df: pd.DataFrame,
    columna_valor: str,
    ventana: int,
    min_periodos: int = 7,
    columnas_grupo: list[str] | None = None,
) -> pd.Series:
    """Mediana móvil desplazada, global o por grupo.

    Args:
        df: DataFrame de entrada (debe venir ordenado por fecha).
        columna_valor: columna a suavizar.
        ventana: tamaño de la ventana en observaciones.
        min_periodos: observaciones mínimas para emitir un valor.
        columnas_grupo: si se indica, la ventana se calcula dentro de cada grupo.

    Returns:
        Serie alineada con ``df`` con la mediana de las observaciones anteriores.
    """

    def _ventana(serie: pd.Series) -> pd.Series:
        return serie.shift(1).rolling(window=ventana, min_periods=min_periodos).median()

    if columnas_grupo:
        return df.groupby(columnas_grupo, dropna=False)[columna_valor].transform(_ventana)
    return _ventana(df[columna_valor])


def factor_mes_tipo(df: pd.DataFrame, columna_valor: str) -> pd.DataFrame:
    """Calcula el factor estacional mes/tipo de día respecto a la mediana.

    Args:
        df: DataFrame con columnas ``mes_num``, ``dia_tipo_num`` y la métrica.
        columna_valor: columna de demanda.

    Returns:
        DataFrame con columnas ``mes_num``, ``dia_tipo_num`` y ``factor_mes_tipo``
        (acotado entre 0.70 y 1.30).
    """
    copia = df.copy()
    mediana_tipo = copia.groupby("dia_tipo_num")[columna_valor].transform("median")
    copia["_ratio"] = copia[columna_valor] / mediana_tipo.replace(0, pd.NA)
    return (
        copia.groupby(["mes_num", "dia_tipo_num"])["_ratio"]
        .median()
        .clip(0.70, 1.30)
        .reset_index()
        .rename(columns={"_ratio": "factor_mes_tipo"})
    )
