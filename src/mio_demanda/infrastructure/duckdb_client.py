"""Cliente DuckDB para consultas analíticas sobre Parquet.

DuckDB es el motor de consulta del proyecto: permite leer cientos de archivos
Parquet como si fueran una tabla, sin cargarlos completos en memoria. Cada
consulta abre una conexión efímera, lo que la hace segura frente a los *reruns*
concurrentes de Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)


def consultar(sql: str, parametros: Sequence[Any] | None = None) -> pd.DataFrame:
    """Ejecuta una consulta SQL y devuelve el resultado como DataFrame.

    Args:
        sql: sentencia SQL (admite parámetros ``?``).
        parametros: valores de los parámetros, en orden de aparición.

    Returns:
        El resultado de la consulta como :class:`pandas.DataFrame`.

    Raises:
        duckdb.Error: si la consulta es inválida.
    """
    with duckdb.connect() as conexion:
        return conexion.execute(sql, list(parametros) if parametros else []).df()


def patron_parquet(carpeta: Path) -> str:
    """Construye el patrón glob recursivo de Parquet de una carpeta.

    DuckDB requiere separadores ``/`` incluso en Windows.

    Args:
        carpeta: carpeta raíz que contiene los parquet.

    Returns:
        El patrón glob, por ejemplo ``C:/datos/**/*.parquet``.
    """
    return str(carpeta / "**" / "*.parquet").replace("\\", "/")


def hay_parquets(carpeta: Path) -> bool:
    """Indica si una carpeta contiene al menos un archivo Parquet.

    Args:
        carpeta: carpeta a inspeccionar.

    Returns:
        ``True`` si existe la carpeta y contiene ``.parquet`` en cualquier nivel.
    """
    return carpeta.is_dir() and any(carpeta.rglob("*.parquet"))
