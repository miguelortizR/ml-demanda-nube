"""Proyección de demanda a futuro con un paquete de modelo guardado.

Carga el modelo campeón, reconstruye las variables del horizonte solicitado
(usando solo historia pasada para las anclas) y genera la proyección:

* **Sistema**: demanda diaria total.
* **Entidad**: demanda diaria por estación/ruta, desagregada al detalle horario
  con el perfil histórico de la entidad.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.domain.metricas import confiabilidad_desde_error
from mio_demanda.infrastructure.almacen_modelos import AlmacenModelos, PaqueteModelo
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_calendario import RepositorioCalendario
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos
from mio_demanda.ml.datasets import (
    Dataset,
    PerfilHorario,
    construir_dataset_entidad,
    construir_dataset_sistema,
    construir_perfil_horario,
)

logger = obtener_logger(__name__)


def margen_error_usos(metricas: dict[str, float], factor: float) -> float:
    """Margen de error esperado (en usos) a partir del RMSE del modelo.

    Se interpreta como la semibanda del intervalo P10-P90: en promedio, la demanda
    real caerá dentro de ``predicción ± margen`` en 8 de cada 10 días. Es una
    estimación normal basada en el RMSE (el error cuadrático medio del modelo).

    Args:
        metricas: métricas del paquete de modelo (debe incluir ``rmse``).
        factor: factor del intervalo P10-P90 (obligatorio; viene de la configuración).

    Returns:
        El margen en usos (``factor · RMSE``).
    """
    try:
        rmse = float(metricas.get("rmse", 0) or 0)
    except (TypeError, ValueError):
        rmse = 0.0
    return float(factor) * max(rmse, 0.0)


def _anexar_banda(datos: pd.DataFrame, margen: float) -> pd.DataFrame:
    """Añade las columnas ``usos_p10`` y ``usos_p90`` a un detalle de proyección.

    Args:
        datos: DataFrame con la columna ``usos_predicho``.
        margen: semibanda en usos.

    Returns:
        El DataFrame con la banda (nunca negativa).
    """
    if margen <= 0:
        return datos
    datos["usos_p10"] = (datos["usos_predicho"] - margen).clip(lower=0)
    datos["usos_p90"] = datos["usos_predicho"] + margen
    return datos


@dataclass(slots=True)
class ProyeccionEntidad:
    """Resultado de proyectar una entidad (estación o ruta).

    Attributes:
        detalle_horario: DataFrame con ``fecha``, ``hora``, ``entidad`` y
            ``usos_predicho``.
        serie_diaria: DataFrame con el agregado diario.
        confiabilidad: confiabilidad estimada del modelo usado.
        modelo: nombre del paquete de modelo consumido.
    """

    detalle_horario: pd.DataFrame
    serie_diaria: pd.DataFrame
    confiabilidad: float
    modelo: str


def alinear_features(datos: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Ajusta una matriz de entrada al contrato de variables del modelo.

    Protege al modelo guardado frente a cambios de esquema (por ejemplo, si el
    profesional agrega o quita eventos del calendario después de entrenar): las
    variables faltantes se rellenan con 0 y las sobrantes se descartan.

    Args:
        datos: matriz de entrada construida por el dataset.
        features: lista ordenada de variables que espera el modelo.

    Returns:
        DataFrame con exactamente las columnas de ``features``, en ese orden.
    """
    faltantes = [columna for columna in features if columna not in datos.columns]
    if faltantes:
        logger.warning(
            "El dataset no contiene %s variable(s) del modelo (%s); se rellenan con 0",
            len(faltantes),
            ", ".join(faltantes[:5]),
        )
    sobrantes = [columna for columna in datos.columns if columna not in features]
    if sobrantes:
        logger.info("Se descartan %s variable(s) no usadas por el modelo", len(sobrantes))
    return datos.reindex(columns=features, fill_value=0.0).astype(float)


def congelar_anclas(dataset: Dataset, corte: date) -> Dataset:
    """Congela las anclas históricas al último valor disponible en ``corte``.

    Al evaluar (backtest) días posteriores a un corte, las anclas de esos días no
    deben usar información de otros días de la misma ventana de evaluación: se
    reemplazan por el valor vigente en el corte. Así la comparación es honesta.

    Args:
        dataset: dataset con las filas de entrenamiento y de evaluación.
        corte: último día permitido para alimentar las anclas.

    Returns:
        El dataset con las anclas congeladas a partir del corte.
    """
    fechas = pd.to_datetime(dataset.fechas)
    posteriores = (fechas > pd.Timestamp(corte)).to_numpy()
    columnas = [
        columna
        for columna in dataset.X.columns
        if columna.startswith(("ancla", "nivel", "patron", "factor_mes"))
    ]
    if not posteriores.any() or not columnas:
        return dataset

    X = dataset.X.copy()
    if "entidad" in dataset.auxiliares.columns:
        entidades = dataset.auxiliares["entidad"].to_numpy()
        for columna in columnas:
            valores = X[columna].to_numpy(dtype=float, copy=True)
            vigente: dict[object, float] = {}
            for indice in range(len(fechas)):
                if not posteriores[indice]:
                    vigente[entidades[indice]] = valores[indice]
            for indice in np.flatnonzero(posteriores):
                valores[indice] = vigente.get(entidades[indice], valores[indice])
            X[columna] = valores
    else:
        for columna in columnas:
            valores = X[columna].to_numpy(dtype=float, copy=True)
            previos = valores[~posteriores]
            if len(previos):
                valores[posteriores] = previos[-1]
            X[columna] = valores

    return replace(dataset, X=X)


class ProyectorDemanda:
    """Proyecta demanda usando los paquetes de modelo disponibles.

    Args:
        settings: configuración de la aplicación.
        almacen: almacén de modelos (se crea si no se inyecta).
        repo: repositorio de datos (se crea si no se inyecta).
        repo_cal: repositorio del calendario (se crea si no se inyecta).
    """

    def __init__(
        self,
        settings: Settings,
        almacen: AlmacenModelos | None = None,
        repo: RepositorioDatos | None = None,
        repo_cal: RepositorioCalendario | None = None,
    ) -> None:
        self._settings = settings
        self._almacen = almacen or AlmacenModelos(settings)
        self._repo = repo or RepositorioDatos(settings)
        self._repo_cal = repo_cal or RepositorioCalendario(settings)
        # El perfil horario depende solo de la historia usada para entrenar: se guarda
        # en memoria para no recalcularlo en cada proyección (tardaba ~1 s).
        self._perfiles: dict[tuple[str | None, date, date], Any] = {}

    # ------------------------------------------------------------------
    # Opciones de ingeniería de características (desde la configuración)
    # ------------------------------------------------------------------
    def _opciones_features(self) -> dict[str, int]:
        """Ventanas de las anclas tomadas de la configuración.

        Returns:
            Diccionario con ``dias_calentamiento``, ``ventana_reciente`` y ``min_obs_ancla``.
        """
        return {
            "dias_calentamiento": self._settings.dias_calentamiento,
            "ventana_reciente": self._settings.ventana_reciente_dias,
            "min_obs_ancla": self._settings.min_obs_ancla,
        }

    def _margen(self, metricas: dict[str, float]) -> float:
        """Semibanda del rango esperado con el factor configurado.

        Args:
            metricas: métricas del paquete de modelo.

        Returns:
            El margen en usos.
        """
        return margen_error_usos(metricas, factor=self._settings.factor_banda_error)

    # ------------------------------------------------------------------
    # Disponibilidad
    # ------------------------------------------------------------------
    @staticmethod
    def nombre_para(tipo: TipoEntidad | None) -> str:
        """Devuelve el nombre del paquete de modelo según el alcance.

        Args:
            tipo: ``None`` para el sistema; tipo de entidad en otro caso.

        Returns:
            El nombre del paquete (``sistema``/``estaciones``/``rutas``).
        """
        if tipo is None:
            return "sistema"
        return "estaciones" if tipo is TipoEntidad.ESTACION else "rutas"

    def modelo_disponible(self, tipo: TipoEntidad | None = None) -> bool:
        """Indica si el modelo requerido está entrenado y guardado.

        Args:
            tipo: alcance del modelo (``None`` = sistema).

        Returns:
            ``True`` si el paquete existe.
        """
        return self._almacen.existe(self.nombre_para(tipo))

    def metadatos(self, tipo: TipoEntidad | None = None) -> PaqueteModelo:
        """Carga el paquete de modelo solicitado.

        Args:
            tipo: alcance del modelo (``None`` = sistema).

        Returns:
            El :class:`PaqueteModelo` cargado.
        """
        return self._almacen.cargar(self.nombre_para(tipo))

    # ------------------------------------------------------------------
    # Sistema
    # ------------------------------------------------------------------
    def proyectar_sistema(self, desde: date, hasta: date) -> pd.DataFrame:
        """Proyecta la demanda diaria del sistema completo.

        Args:
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            DataFrame con ``fecha``, ``dia_tipo_nom``, ``usos_predicho`` y
            ``es_proyectado``.

        Raises:
            FileNotFoundError: si el modelo del sistema no está entrenado.
        """
        paquete = self.metadatos(None)
        dataset = construir_dataset_sistema(
            self._repo, self._repo_cal, desde, hasta, **self._opciones_features()
        )
        entrada = alinear_features(dataset.X, paquete.features)

        resultado = dataset.auxiliares.copy()
        resultado["usos_predicho"] = np.maximum(
            paquete.modelo.predict(entrada.to_numpy(dtype=float)), 0.0
        )
        # Demanda real observada (NaN en las fechas futuras).
        resultado["demanda_real"] = dataset.tabla["usos"].to_numpy()
        resultado["es_proyectado"] = ~dataset.en_entrenamiento.to_numpy()
        resultado = _anexar_banda(resultado, self._margen(paquete.metricas))
        return resultado.sort_values("fecha").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Backtest con un modelo dado
    # ------------------------------------------------------------------
    def detalle_con_modelo(
        self,
        modelo: Any,
        features: list[str],
        desde: date,
        hasta: date,
        tipo: TipoEntidad | None = None,
        entidad: str | None = None,
    ) -> pd.DataFrame:
        """Detalle diario proyectado con un modelo y contrato de variables dados.

        Se usa en el backtest cuando hay que reentrenar una copia del modelo con un
        corte anterior. Las anclas se **congelan** en el corte para que la
        comparación no tenga fuga de información.

        Args:
            modelo: modelo o ensamble ya entrenado.
            features: variables que espera el modelo.
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.
            tipo: ``None`` para el sistema; tipo de entidad en otro caso.
            entidad: entidad a proyectar (obligatoria si ``tipo`` no es ``None``).

        Returns:
            DataFrame con ``fecha``, tipo de día, ``usos_predicho`` y ``demanda_real``.
        """
        if tipo is None:
            dataset = construir_dataset_sistema(
                self._repo, self._repo_cal, desde, hasta, **self._opciones_features()
            )
        else:
            if not entidad:
                raise ValueError("Se requiere la entidad para proyectar un alcance por entidad")
            dataset = construir_dataset_entidad(
                self._repo,
                self._repo_cal,
                tipo,
                desde,
                hasta,
                entidades=[entidad],
                min_cobertura=self._settings.min_cobertura_entidad,
                **self._opciones_features(),
            )

        dataset = congelar_anclas(dataset, desde - timedelta(days=1))
        entrada = alinear_features(dataset.X, features)
        detalle = dataset.auxiliares.copy()
        detalle["usos_predicho"] = np.maximum(modelo.predict(entrada.to_numpy(dtype=float)), 0.0)
        detalle["demanda_real"] = dataset.tabla["usos"].to_numpy()
        return detalle.drop(columns=["entidad"], errors="ignore")

    def detalle_horario(
        self,
        diaria: pd.DataFrame,
        paquete: PaqueteModelo,
        margen_diario: float = 0.0,
        entidad: str | None = None,
        entidades: list[str] | None = None,
    ) -> pd.DataFrame:
        """Reparte una proyección diaria en usos por hora con el perfil histórico.

        Args:
            diaria: proyección diaria (``fecha``, ``dia_tipo_num``, ``usos_predicho``).
            paquete: paquete de modelo usado.
            margen_diario: semibanda diaria que se reparte igual que la demanda.
            entidad: una entidad concreta (usa **su** perfil).
            entidades: grupo de entidades (usa el perfil del grupo); ``None`` y sin
                ``entidad`` = sistema completo.

        Returns:
            El detalle horario listo para graficar o exportar.
        """
        return self._desagregar_horario(
            diaria, entidad, paquete, margen_diario, entidades=entidades
        )

    def proyectar_entidades(
        self,
        tipo: TipoEntidad,
        desde: date,
        hasta: date,
        entidades: list[str] | None = None,
    ) -> pd.DataFrame:
        """Proyecta **todas** las entidades de un tipo en una sola pasada.

        Construye el dataset una vez y predice para todas juntas: es lo que permite
        proyectar el período completo de todas las rutas o estaciones y después filtrar,
        en lugar de repetir el trabajo entidad por entidad.

        Args:
            tipo: tipo de entidad (estación o ruta).
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.
            entidades: subconjunto de entidades; ``None`` para todas.

        Returns:
            DataFrame con ``entidad``, ``fecha``, ``dia_tipo_nom``, ``usos_predicho`` y
            ``demanda_real`` (NaN en el futuro).

        Raises:
            FileNotFoundError: si el modelo del tipo no está entrenado.
        """
        paquete = self.metadatos(tipo)
        dataset = construir_dataset_entidad(
            self._repo,
            self._repo_cal,
            tipo,
            desde,
            hasta,
            entidades=entidades,
            min_cobertura=self._settings.min_cobertura_entidad,
            **self._opciones_features(),
        )
        entrada = alinear_features(dataset.X, paquete.features)
        resultado = dataset.auxiliares.copy()
        resultado["usos_predicho"] = np.maximum(
            paquete.modelo.predict(entrada.to_numpy(dtype=float)), 0.0
        )
        resultado["demanda_real"] = dataset.tabla["usos"].to_numpy()
        resultado["es_proyectado"] = ~dataset.en_entrenamiento.to_numpy()
        return resultado.sort_values(["entidad", "fecha"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Entidad (estación / ruta)
    # ------------------------------------------------------------------
    def proyectar_entidad(
        self,
        entidad: str,
        tipo: TipoEntidad,
        desde: date,
        hasta: date,
    ) -> ProyeccionEntidad:
        """Proyecta la demanda de una estación o ruta, con detalle horario.

        Args:
            entidad: identificador de la estación o ruta.
            tipo: tipo de entidad.
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            Una :class:`ProyeccionEntidad` con el detalle horario y el agregado diario.

        Raises:
            FileNotFoundError: si el modelo del tipo solicitado no está entrenado.
        """
        paquete = self.metadatos(tipo)
        dataset = construir_dataset_entidad(
            self._repo,
            self._repo_cal,
            tipo,
            desde,
            hasta,
            entidades=[entidad],
            min_cobertura=self._settings.min_cobertura_entidad,
            **self._opciones_features(),
        )
        entrada = alinear_features(dataset.X, paquete.features)

        diaria = dataset.auxiliares.copy()
        diaria["usos_predicho"] = np.maximum(
            paquete.modelo.predict(entrada.to_numpy(dtype=float)), 0.0
        )
        # Demanda real observada (NaN en las fechas futuras).
        diaria["demanda_real"] = dataset.tabla["usos"].to_numpy()
        diaria["es_proyectado"] = ~dataset.en_entrenamiento.to_numpy()
        diaria = diaria.drop(columns=["entidad"])
        margen_diario = self._margen(paquete.metricas)
        diaria = _anexar_banda(diaria, margen_diario)

        horario = self._desagregar_horario(diaria, entidad, paquete, margen_diario)
        confiabilidad = confiabilidad_desde_error(paquete.metricas.get("wape", 15.0))
        return ProyeccionEntidad(
            detalle_horario=horario,
            serie_diaria=diaria,
            confiabilidad=confiabilidad,
            modelo=paquete.nombre,
        )

    def _perfil_horario(self, entidades: list[str] | None, paquete: PaqueteModelo) -> PerfilHorario:
        """Devuelve el perfil horario histórico, cacheado por conjunto de entidades.

        Args:
            entidades: entidades del perfil; ``None`` para el sistema completo.
            paquete: paquete de modelo (para conocer el fin de la historia real).

        Returns:
            El :class:`PerfilHorario` con ``por_entidad`` y ``global_``.
        """
        hasta_historia = pd.Timestamp(paquete.hasta_entrenamiento).date()
        desde_historia = (
            pd.Timestamp(paquete.hasta_entrenamiento)
            - pd.Timedelta(days=self._settings.dias_historia_perfil)
        ).date()
        conjunto = tuple(sorted(entidades)) if entidades else None
        clave = (conjunto, desde_historia, hasta_historia)
        perfil = self._perfiles.get(clave)
        if perfil is None:
            perfil = construir_perfil_horario(
                self._repo,
                self._repo_cal,
                entidades=list(conjunto) if conjunto else None,
                desde=desde_historia,
                hasta=hasta_historia,
            )
            self._perfiles[clave] = perfil
        return perfil

    def detalle_horario_por_entidad(
        self, detalle: pd.DataFrame, paquete: PaqueteModelo
    ) -> pd.DataFrame:
        """Reparte la serie diaria de **cada** entidad con el perfil horario de esa entidad.

        Se usa para exportar el detalle por entidad con sus horas. Aprovecha el perfil del
        grupo ya cacheado, así que no consulta la base de datos otra vez.

        Args:
            detalle: serie diaria por entidad (``entidad``, ``fecha``, ``dia_tipo_num``,
                ``usos_predicho``).
            paquete: paquete de modelo del alcance.

        Returns:
            DataFrame con ``fecha``, ``entidad``, ``hora`` y ``usos_hora``.
        """
        entidades = sorted(detalle["entidad"].unique().tolist())
        perfil = self._perfil_horario(entidades, paquete)
        por_entidad = perfil.por_entidad[["entidad", "hora", "dia_tipo_num", "participacion"]]
        global_ = perfil.global_[["hora", "dia_tipo_num", "participacion"]].rename(
            columns={"participacion": "participacion_global"}
        )
        unido = detalle[["entidad", "fecha", "dia_tipo_num", "usos_predicho"]].merge(
            por_entidad, on=["entidad", "dia_tipo_num"], how="left"
        )
        unido = unido.merge(global_, on=["hora", "dia_tipo_num"], how="left")
        unido["participacion"] = (
            unido["participacion"].fillna(unido["participacion_global"]).fillna(0.0)
        )
        unido["usos_hora"] = unido["usos_predicho"] * unido["participacion"]
        return (
            unido[["fecha", "entidad", "hora", "usos_hora"]]
            .sort_values(["fecha", "entidad", "hora"])
            .reset_index(drop=True)
        )

    def _desagregar_horario(
        self,
        diaria: pd.DataFrame,
        entidad: str | None,
        paquete: PaqueteModelo,
        margen_diario: float = 0.0,
        entidades: list[str] | None = None,
    ) -> pd.DataFrame:
        """Reparte la demanda diaria entre las horas según el perfil histórico.

        El modelo predice **por día**; esta función traduce ese total diario a usos por
        hora con la participación histórica de cada hora en ese tipo de día.

        Args:
            diaria: proyección diaria (del sistema, de una entidad o de un grupo).
            entidad: identificador de la entidad; ``None`` para el sistema o el grupo.
            paquete: paquete de modelo (para conocer el fin de la historia real).
            margen_diario: semibanda de error diaria en usos (se reparte igual que la
                demanda).
            entidades: grupo de entidades cuyo perfil se usa (todas las rutas, por
                ejemplo). Si se indica, se usa el perfil agregado del grupo.

        Returns:
            DataFrame horario con ``fecha``, ``hora``, ``entidad``, ``usos_predicho``
            y la banda ``usos_p10``/``usos_p90``.
        """
        perfil = self._perfil_horario([entidad] if entidad is not None else entidades, paquete)

        if entidad is not None:
            perfil_usado = perfil.por_entidad[perfil.por_entidad["entidad"] == entidad]
            if perfil_usado.empty:
                perfil_usado = perfil.global_
            etiqueta = entidad
        else:
            perfil_usado = perfil.global_
            etiqueta = "sistema" if not entidades else "grupo"

        expandido = diaria.merge(
            perfil_usado[["hora", "dia_tipo_num", "participacion"]],
            on="dia_tipo_num",
            how="left",
        )
        expandido["participacion"] = expandido["participacion"].fillna(0.0)
        expandido["usos_predicho"] = expandido["usos_predicho"] * expandido["participacion"]
        if margen_diario > 0:
            margen_hora = margen_diario * expandido["participacion"]
            expandido["usos_p10"] = (expandido["usos_predicho"] - margen_hora).clip(lower=0)
            expandido["usos_p90"] = expandido["usos_predicho"] + margen_hora
        expandido["entidad"] = etiqueta
        columnas = [
            "fecha",
            "hora",
            "entidad",
            "usos_predicho",
            "usos_p10",
            "usos_p90",
            "dia_tipo_num",
            "dia_tipo_nom",
            "es_proyectado",
        ]
        return expandido[columnas].sort_values(["fecha", "hora"]).reset_index(drop=True)
