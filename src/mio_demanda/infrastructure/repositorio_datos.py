"""Repositorio de datos normalizados (usos del sistema MIO).

Encapsula **todas** las consultas al Parquet normalizado de usos en
``data/raw``. Las capas superiores nunca escriben SQL: piden series ya tipadas.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.calidad_datos import es_dia_parcial
from mio_demanda.domain.entidades import TipoEntidad, clasificar_entidad
from mio_demanda.infrastructure.duckdb_client import consultar
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)


class RepositorioDatos:
    """Acceso de solo lectura a los usos normalizados del proyecto.

    Args:
        settings: configuración con la ruta del Parquet de usos.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Disponibilidad
    # ------------------------------------------------------------------
    @property
    def disponible(self) -> bool:
        """Indica si el Parquet de usos existe en disco."""
        return self._settings.parquet_usos.exists()

    def rango_fechas_usos(self) -> tuple[date, date] | None:
        """Devuelve el rango de fechas disponible en los usos.

        Returns:
            Tupla (mínima, máxima) o ``None`` si no hay datos.
        """
        if not self.disponible:
            return None
        sql = "SELECT min(fecha) AS minimo, max(fecha) AS maximo FROM read_parquet(?)"
        df = consultar(sql, [str(self._settings.parquet_usos)])
        if df.empty or pd.isna(df.iloc[0]["minimo"]):
            return None
        return (
            pd.Timestamp(df.iloc[0]["minimo"]).date(),
            pd.Timestamp(df.iloc[0]["maximo"]).date(),
        )

    # ------------------------------------------------------------------
    # Usos (demanda)
    # ------------------------------------------------------------------
    def usos_diarios(
        self,
        desde: date,
        hasta: date,
        incluir_integraciones: bool = False,
    ) -> pd.DataFrame:
        """Serie diaria de usos del sistema completo.

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).
            incluir_integraciones: si es ``True``, suma también las integraciones.

        Returns:
            DataFrame con columnas ``fecha``, ``usos`` e ``integraciones``.
        """
        expresion_usos = "sum(usos)"
        if incluir_integraciones:
            expresion_usos = "sum(usos + integraciones)"
        sql = f"""
            SELECT fecha, {expresion_usos} AS usos, sum(integraciones) AS integraciones
            FROM read_parquet(?)
            WHERE fecha BETWEEN ?::DATE AND ?::DATE
            GROUP BY fecha
            ORDER BY fecha
        """
        df = consultar(sql, [str(self._settings.parquet_usos), desde, hasta])
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def usos_diarios_por_entidad(
        self,
        desde: date,
        hasta: date,
        entidades: list[str] | None = None,
    ) -> pd.DataFrame:
        """Serie diaria de usos por entidad (suma de todas las horas).

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).
            entidades: subconjunto de entidades; ``None`` para todas.

        Returns:
            DataFrame con ``fecha``, ``entidad`` y ``usos``.
        """
        filtro = ""
        parametros: list[object] = [str(self._settings.parquet_usos), desde, hasta]
        if entidades:
            marcadores = ", ".join(["?"] * len(entidades))
            filtro = f" AND entidad IN ({marcadores})"
            parametros.extend(entidades)

        sql = f"""
            SELECT fecha, entidad, sum(usos) AS usos
            FROM read_parquet(?)
            WHERE fecha BETWEEN ?::DATE AND ?::DATE {filtro}
            GROUP BY fecha, entidad
            ORDER BY fecha, entidad
        """
        df = consultar(sql, parametros)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def usos_horarios_perfil(
        self,
        desde: date,
        hasta: date,
        entidades: list[str] | None = None,
    ) -> pd.DataFrame:
        """Serie horaria de usos por entidad, sin agregar (para el perfil horario).

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).
            entidades: subconjunto de entidades; ``None`` para todas.

        Returns:
            DataFrame con ``fecha``, ``entidad``, ``hora`` y ``usos``.
        """
        filtro = ""
        parametros: list[object] = [str(self._settings.parquet_usos), desde, hasta]
        if entidades:
            marcadores = ", ".join(["?"] * len(entidades))
            filtro = f" AND entidad IN ({marcadores})"
            parametros.extend(entidades)

        sql = f"""
            SELECT fecha, entidad, hora, sum(usos) AS usos
            FROM read_parquet(?)
            WHERE fecha BETWEEN ?::DATE AND ?::DATE {filtro}
            GROUP BY fecha, entidad, hora
        """
        df = consultar(sql, parametros)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def ultima_fecha_confiable(self, umbral: float, dias: int = 60) -> date | None:
        """Última fecha con datos que se consideran completos.

        La extracción puede correr a mitad de jornada y dejar el último día a medias.
        Si ese día queda por debajo del ``umbral`` de la mediana de su mismo día de la
        semana, se descarta y se devuelve el día anterior.

        Args:
            umbral: fracción de la mediana por debajo de la cual el día es parcial.
            dias: ventana reciente usada para estimar la mediana del día de la semana.

        Returns:
            La fecha confiable, o ``None`` si no hay datos.
        """
        rango = self.rango_fechas_usos()
        if rango is None:
            return None
        _, fin = rango
        usos = self.usos_diarios(fin - timedelta(days=max(dias, 2) - 1), fin)
        if usos.empty:
            return fin

        usos = usos.sort_values("fecha")
        usos["dia_semana"] = usos["fecha"].dt.weekday
        ultimo = usos.iloc[-1]
        mediana = usos.loc[usos["dia_semana"] == ultimo["dia_semana"], "usos"].median()
        if not es_dia_parcial(float(ultimo["usos"]), float(mediana), umbral):
            return fin

        anteriores = usos[usos["fecha"] < ultimo["fecha"]]
        if anteriores.empty:
            return fin
        logger.info("El último día (%s) parece incompleto; se usa el anterior", fin)
        return anteriores.iloc[-1]["fecha"].date()

    def listar_entidades(self, tipo: TipoEntidad) -> list[str]:
        """Lista las entidades disponibles del tipo solicitado.

        Args:
            tipo: :class:`TipoEntidad` deseado (estación o ruta).

        Returns:
            Lista ordenada de identificadores de entidad.
        """
        sql = "SELECT DISTINCT entidad FROM read_parquet(?) ORDER BY entidad"
        df = consultar(sql, [str(self._settings.parquet_usos)])
        entidades = sorted(
            {
                str(valor)
                for valor in df["entidad"].tolist()
                if clasificar_entidad(str(valor)) is tipo
            }
        )
        return entidades

    def entidades_con_datos(self, tipo: TipoEntidad, desde: date, hasta: date) -> list[str]:
        """Entidades del tipo con usos registrados en el rango indicado.

        El histórico acumula entidades retiradas o con nombres en desuso que ya no
        tienen demanda; ofrecerlas como proyectables solo produce errores. Este
        método devuelve las que **sí** tienen datos en la ventana pedida.

        Args:
            tipo: :class:`TipoEntidad` deseado (estación o ruta).
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            Lista ordenada de identificadores con datos en el rango.
        """
        if not self.disponible:
            return []
        sql = """
            SELECT DISTINCT entidad FROM read_parquet(?)
            WHERE fecha BETWEEN ?::DATE AND ?::DATE
            ORDER BY entidad
        """
        df = consultar(sql, [str(self._settings.parquet_usos), desde, hasta])
        return sorted(
            {
                str(valor)
                for valor in df["entidad"].tolist()
                if clasificar_entidad(str(valor)) is tipo
            }
        )
