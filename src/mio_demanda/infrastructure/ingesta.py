"""Ingesta: extrae y normaliza las fuentes crudas a Parquet dentro del proyecto.

Las fuentes originales viven en el directorio de Power BI de la Oficina de
Evaluación y se leen **en modo solo lectura**. Este módulo las normaliza
(nombres de columnas, tipos, deduplicación) y las deja en ``data/raw`` para que
el resto de la aplicación trabaje siempre contra un contrato estable.

Alcance actual: **demanda (usos)** y **calendario/dimensiones**. Los kilómetros y
la puntualidad quedaron fuera del alcance de la aplicación.

Ejecución típica::

    uv run python -m mio_demanda.cli extract
    # o
    make extract
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.infrastructure.duckdb_client import consultar, hay_parquets, patron_parquet
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)

# Columnas de salida del contrato de usos normalizados.
COLUMNAS_USOS = ["fecha", "entidad", "hora", "usos", "integraciones"]
# Columnas de salida del calendario normalizado.
COLUMNAS_CALENDARIO = [
    "fecha",
    "anio",
    "semana_iso",
    "mes_num",
    "mes_nombre",
    "dia_num",
    "dia_orden",
    "dia_nombre",
    "dia_tipo_num",
    "dia_tipo_nom",
]

# Mapa del calendario oficial: Hab -> día hábil, Sab -> sábado, Fes -> festivo.
_MAPA_DIA_TIPO = {
    "HAB": (0, "Hábil"),
    "SAB": (1, "Sábado"),
    "FES": (2, "Festivo"),
    "FESTIVO": (2, "Festivo"),
}


@dataclass(slots=True)
class ResultadoIngesta:
    """Resumen de la extracción: registros escritos por fuente."""

    usos: int = 0
    calendario: int = 0

    def como_diccionario(self) -> dict[str, int]:
        """Devuelve el resumen como diccionario plano."""
        return asdict(self)


# ----------------------------------------------------------------------
# USOS (demanda)
# ----------------------------------------------------------------------
def extraer_usos(settings: Settings) -> int:
    """Normaliza los usos diarios/horarios y los guarda en ``data/raw/usos.parquet``.

    Args:
        settings: configuración de la aplicación.

    Returns:
        Número de registros escritos (0 si no hay fuente).
    """
    if not hay_parquets(settings.ruta_usos):
        logger.warning("No se encontraron parquets de usos en %s", settings.ruta_usos)
        return 0

    sql = """
        SELECT
            "fecha"::DATE                            AS fecha,
            trim(cast("Nombre_estacion" AS VARCHAR)) AS entidad,
            cast("hora" AS INTEGER)                  AS hora,
            coalesce(TRY_CAST("Uso_pago" AS DOUBLE), 0)     AS usos,
            coalesce(TRY_CAST("Integracion" AS DOUBLE), 0)  AS integraciones
        FROM read_parquet(?, union_by_name = true)
        WHERE "fecha" IS NOT NULL
    """
    df = consultar(sql, [patron_parquet(settings.ruta_usos)])
    df = df[COLUMNAS_USOS].dropna(subset=["fecha", "entidad"])
    df["entidad"] = df["entidad"].astype(str).str.strip()

    settings.parquet_usos.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.parquet_usos, index=False)
    logger.info("Usos normalizados: %s registros -> %s", f"{len(df):,}", settings.parquet_usos)
    return len(df)


# ----------------------------------------------------------------------
# CALENDARIO OFICIAL
# ----------------------------------------------------------------------
def extraer_calendario(settings: Settings) -> int:
    """Normaliza ``dim_Calendario.xlsx`` a ``data/raw/calendario.parquet``.

    Args:
        settings: configuración de la aplicación.

    Returns:
        Número de filas escritas (0 si no existe el archivo).
    """
    origen = settings.ruta_dimensiones / settings.archivo_calendario
    if not origen.exists():
        logger.warning("No se encontró el calendario oficial en %s", origen)
        return 0

    crudo = pd.read_excel(origen, sheet_name=0)
    crudo.columns = [str(c).strip() for c in crudo.columns]

    df = pd.DataFrame()
    df["fecha"] = pd.to_datetime(crudo["Fecha"], errors="coerce").dt.normalize()
    df["anio"] = pd.to_numeric(crudo.get("Año"), errors="coerce").astype("Int32")
    df["semana_iso"] = pd.to_numeric(crudo.get("Semana.iso"), errors="coerce").astype("Int32")
    df["mes_num"] = pd.to_numeric(crudo.get("Mes.num"), errors="coerce").astype("Int32")
    df["mes_nombre"] = crudo.get("Mes.nombre", pd.Series(dtype="object")).astype("string")
    df["dia_num"] = pd.to_numeric(crudo.get("Dia.num"), errors="coerce").astype("Int32")
    df["dia_orden"] = pd.to_numeric(crudo.get("Dia.orden"), errors="coerce").astype("Int32")
    df["dia_nombre"] = crudo.get("Dia.nombre", pd.Series(dtype="object")).astype("string")

    tipo = crudo["Dia.tipo"].astype(str).str.strip().str.upper()
    df["dia_tipo_num"] = tipo.map(lambda v: _MAPA_DIA_TIPO.get(v, (0, "Hábil"))[0]).astype("Int32")
    df["dia_tipo_nom"] = tipo.map(lambda v: _MAPA_DIA_TIPO.get(v, (0, "Hábil"))[1]).astype("string")

    df = df[COLUMNAS_CALENDARIO].dropna(subset=["fecha"]).drop_duplicates(subset=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    settings.parquet_calendario.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.parquet_calendario, index=False)
    logger.info(
        "Calendario normalizado: %s filas -> %s", f"{len(df):,}", settings.parquet_calendario
    )
    return len(df)




# ----------------------------------------------------------------------
# ORQUESTADOR
# ----------------------------------------------------------------------
def extraer_todo(settings: Settings) -> ResultadoIngesta:
    """Ejecuta todas las extracciones y devuelve el resumen.

    Args:
        settings: configuración de la aplicación.

    Returns:
        Un :class:`ResultadoIngesta` con el conteo por fuente.
    """
    settings.ensure_directorios()
    resultado = ResultadoIngesta(
        usos=extraer_usos(settings),
        calendario=extraer_calendario(settings),
    )
    logger.info("Ingesta finalizada: %s", resultado.como_diccionario())
    return resultado


def archivos_disponibles(settings: Settings) -> dict[str, Path]:
    """Lista los parquet normalizados presentes y sus rutas.

    Args:
        settings: configuración de la aplicación.

    Returns:
        Mapa nombre -> ruta de los archivos que existen.
    """
    candidatos = {
        "usos": settings.parquet_usos,
        "calendario": settings.parquet_calendario,
    }
    return {nombre: ruta for nombre, ruta in candidatos.items() if ruta.exists()}
