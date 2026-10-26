"""Servicio de proyección: casos de uso de las pestañas de negocio.

Convierte los resultados del modelo en información accionable:

* **Proyección general** (sistema): demanda diaria total + resumen.
* **Proyección por entidad**: demanda horaria de una estación o ruta + resumen.

Todas las cifras son de **demanda (usos)**; el alcance del proyecto no incluye
kilómetros, cumplimiento ni resultados económicos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import ResumenProyeccion, TipoEntidad
from mio_demanda.domain.metricas import confiabilidad_desde_error
from mio_demanda.infrastructure.almacen_modelos import PaqueteModelo
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos
from mio_demanda.ml.entrenamiento import EntrenadorModelo
from mio_demanda.ml.evaluacion import calcular_metricas
from mio_demanda.ml.prediccion import ProyectorDemanda, margen_error_usos

logger = obtener_logger(__name__)


@dataclass(slots=True)
class ProyeccionGeneral:
    """Proyección de demanda del sistema con su resumen.

    Attributes:
        detalle: DataFrame diario con la demanda proyectada y la real observada.
        resumen: métricas agregadas del horizonte.
        modelo: paquete de modelo consumido.
        confiabilidad: confiabilidad del modelo (1 - WAPE).
        metricas: métricas de validación del modelo consumido.
    """

    detalle: pd.DataFrame
    resumen: ResumenProyeccion
    modelo: str
    confiabilidad: float
    metricas: dict[str, float] = field(default_factory=dict)
    detalle_horario: pd.DataFrame = field(default_factory=pd.DataFrame)
    fecha_entrenamiento: datetime | None = None
    desde_entrenamiento: date | None = None
    hasta_entrenamiento: date | None = None


@dataclass(slots=True)
class ProyeccionPorEntidad:
    """Proyección de demanda de una estación o ruta.

    Attributes:
        detalle_horario: DataFrame horario con la demanda proyectada.
        serie_diaria: DataFrame diario agregado.
        resumen: métricas agregadas del horizonte.
        confiabilidad: confiabilidad del modelo (1 - WAPE).
        modelo: paquete de modelo consumido.
        etiqueta: descripción del alcance (p. ej. ``Ruta T31``).
        por_tipo_dia: desglose por tipo de día.
        metricas: métricas de validación del modelo consumido.
    """

    detalle_horario: pd.DataFrame
    serie_diaria: pd.DataFrame
    resumen: ResumenProyeccion
    confiabilidad: float
    modelo: str
    etiqueta: str
    por_tipo_dia: pd.DataFrame = field(default_factory=pd.DataFrame)
    metricas: dict[str, float] = field(default_factory=dict)
    fecha_entrenamiento: datetime | None = None
    desde_entrenamiento: date | None = None
    hasta_entrenamiento: date | None = None
    detalle_por_entidad: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class ProyeccionConjunta:
    """Proyección de **todas** las estaciones y **todas** las rutas de un período.

    Sirve a la opción «Todas» de la interfaz: cada grupo se proyecta una sola vez y
    después se filtra por entidad sin volver a calcular.

    Attributes:
        estaciones: proyección agregada de las estaciones (``None`` si no hay modelo).
        rutas: proyección agregada de las rutas (``None`` si no hay modelo).
    """

    estaciones: ProyeccionPorEntidad | None = None
    rutas: ProyeccionPorEntidad | None = None

    def grupos(self) -> dict[TipoEntidad, ProyeccionPorEntidad]:
        """Devuelve solo los grupos disponibles, por tipo de entidad.

        Returns:
            Mapa ``tipo -> proyección`` con los grupos que tengan modelo.
        """
        disponibles: dict[TipoEntidad, ProyeccionPorEntidad] = {}
        if self.estaciones is not None:
            disponibles[TipoEntidad.ESTACION] = self.estaciones
        if self.rutas is not None:
            disponibles[TipoEntidad.RUTA] = self.rutas
        return disponibles


@dataclass(slots=True)
class EvaluacionProyeccion:
    """Comparación del modelo contra la demanda real de un periodo reciente.

    Attributes:
        comparacion: DataFrame con fecha, demanda real, predicción, error y APE.
        metricas: MAE, RMSE, MAPE, WAPE y R² del periodo evaluado.
        confiabilidad: confiabilidad estimada a partir del WAPE.
        dias_evaluados: número de días comparados.
        dias_fuera_de_muestra: días evaluados posteriores al corte de entrenamiento.
        desde: primera fecha evaluada.
        hasta: última fecha evaluada.
        modelo: paquete de modelo evaluado.
        etiqueta: alcance evaluado.
        corte_entrenamiento: último día de historia del modelo evaluado.
        reentrenado: si hubo que entrenar una copia temporal (el modelo guardado ya
            estaba entrenado hasta el último dato disponible).
    """

    comparacion: pd.DataFrame
    metricas: dict[str, float]
    confiabilidad: float
    dias_evaluados: int
    dias_fuera_de_muestra: int
    desde: date
    hasta: date
    modelo: str
    etiqueta: str = ""
    corte_entrenamiento: date | None = None
    reentrenado: bool = False


class ServicioProyeccion:
    """Casos de uso de proyección y evaluación de demanda.

    Args:
        settings: configuración de la aplicación.
        proyector: proyector de demanda (se crea si no se inyecta).
        repo: repositorio de datos (se crea si no se inyecta).
    """

    def __init__(
        self,
        settings: Settings,
        proyector: ProyectorDemanda | None = None,
        repo: RepositorioDatos | None = None,
        entrenador: EntrenadorModelo | None = None,
    ) -> None:
        self._settings = settings
        self._proyector = proyector or ProyectorDemanda(settings)
        self._repo = repo or RepositorioDatos(settings)
        self._entrenador = entrenador or EntrenadorModelo(settings)

    def modelos_disponibles(self) -> dict[str, bool]:
        """Indica qué modelos están entrenados y disponibles.

        Returns:
            Mapa alcance -> disponibilidad (``sistema``, ``estaciones``, ``rutas``).
        """
        return {
            "sistema": self._proyector.modelo_disponible(None),
            "estaciones": self._proyector.modelo_disponible(TipoEntidad.ESTACION),
            "rutas": self._proyector.modelo_disponible(TipoEntidad.RUTA),
        }

    def modelo_en_uso(self, tipo: TipoEntidad | None = None) -> dict[str, Any]:
        """Describe el modelo que se usará para proyectar un alcance.

        Aunque se entrenen varios candidatos, **por alcance se guarda un solo
        campeón**; si ese campeón es el ensamble, internamente combina los
        candidatos con los pesos optimizados.

        Args:
            tipo: alcance del modelo (``None`` = sistema).

        Returns:
            Diccionario con el paquete, el campeón, su composición, las métricas de
            validación y el periodo de entrenamiento.
        """
        paquete = self._proyector.metadatos(tipo)
        extras = paquete.extras or {}
        nombres = list(extras.get("nombres_ensamble", []))
        pesos = [float(peso) for peso in extras.get("pesos_ensamble", [])]
        etiquetas = {
            None: "Sistema",
            TipoEntidad.ESTACION: "Estaciones",
            TipoEntidad.RUTA: "Rutas",
        }
        return {
            "alcance": etiquetas.get(tipo, str(tipo)),
            "paquete": paquete.nombre,
            "campeon": str(extras.get("campeon", "modelo")),
            "algoritmo": paquete.algoritmo,
            "es_ensamble": str(extras.get("campeon")) == "ensamble",
            "componentes": dict(zip(nombres, pesos, strict=False)),
            "comparativa": list(extras.get("comparativa", [])),
            "activado_manualmente": bool(extras.get("activado_manualmente")),
            "metricas": dict(paquete.metricas),
            "confiabilidad": confiabilidad_desde_error(paquete.metricas.get("wape", 15.0)),
            "fecha_entrenamiento": paquete.fecha_entrenamiento,
            "desde_entrenamiento": paquete.desde_entrenamiento,
            "hasta_entrenamiento": paquete.hasta_entrenamiento,
            "n_variables": len(paquete.features),
        }

    def modelos_en_uso(self) -> list[dict[str, Any]]:
        """Describe los modelos disponibles en los tres alcances.

        Returns:
            Lista de descripciones (solo de los alcances que tienen modelo).
        """
        descripciones = []
        for tipo in (None, TipoEntidad.ESTACION, TipoEntidad.RUTA):
            if self._proyector.modelo_disponible(tipo):
                descripciones.append(self.modelo_en_uso(tipo))
        return descripciones

    def umbrales_coherencia(self) -> tuple[float, float]:
        """Razones aceptables entre lo proyectado y lo real (control de coherencia).

        Returns:
            Tupla ``(minimo, maximo)`` tomada de la configuración.
        """
        return self._settings.coherencia_min, self._settings.coherencia_max

    def retrospectiva(self, dias: int | None = None) -> EvaluacionProyeccion:
        """Proyección del sistema hecha **hacia atrás** y su comparación con lo real.

        Es el corazón del vigilante: en vez de avisar en una sección aparte, devuelve
        lo que el modelo habría proyectado para los últimos días (con un modelo que no
        los vio) al lado de la demanda real. Se dibuja en la misma gráfica y se exporta
        en la misma hoja del Excel.

        Args:
            dias: días hacia atrás; ``None`` usa la configuración.

        Returns:
            La :class:`EvaluacionProyeccion` con la comparación real vs. proyectado.
        """
        return self.evaluar_sistema(dias or self._settings.retrospectiva_dias)

    def retrospectiva_entidad(
        self, entidad: str, tipo: TipoEntidad, dias: int | None = None
    ) -> EvaluacionProyeccion:
        """Retrospectiva de una estación o ruta (misma idea que :meth:`retrospectiva`).

        Args:
            entidad: identificador de la estación o ruta.
            tipo: tipo de entidad.
            dias: días hacia atrás; ``None`` usa la configuración.

        Returns:
            La :class:`EvaluacionProyeccion` de la entidad.
        """
        return self.evaluar_entidad(entidad, tipo, dias or self._settings.retrospectiva_dias)

    def referencia_reciente(self, dias: int | None = None) -> float:
        """Demanda diaria real promedio de los últimos días (control de coherencia).

        Sirve para comparar lo proyectado con lo observado recientemente y detectar
        proyecciones absurdamente bajas o altas.

        Args:
            dias: número de días recientes a promediar; ``None`` usa la configuración.

        Returns:
            El promedio diario real; ``0.0`` si no hay datos.
        """
        dias = dias or self._settings.dias_referencia_reciente
        fin = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin is None:
            return 0.0
        desde = fin - timedelta(days=max(1, dias) - 1)
        usos = self._repo.usos_diarios(desde, fin)
        if usos.empty:
            return 0.0
        return float(usos["usos"].sum()) / max(usos["fecha"].nunique(), 1)

    def referencia_reciente_entidad(self, entidad: str, dias: int | None = None) -> float:
        """Demanda diaria real promedio reciente de una entidad.

        Args:
            entidad: identificador de la estación o ruta.
            dias: número de días recientes a promediar; ``None`` usa la configuración.

        Returns:
            El promedio diario real de la entidad; ``0.0`` si no hay datos.
        """
        dias = dias or self._settings.dias_referencia_reciente
        fin = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin is None:
            return 0.0
        desde = fin - timedelta(days=max(1, dias) - 1)
        usos = self._repo.usos_diarios_por_entidad(desde, fin, [entidad])
        if usos.empty:
            return 0.0
        return float(usos["usos"].sum()) / max(usos["fecha"].nunique(), 1)

    # ------------------------------------------------------------------
    # Proyección general (sistema)
    # ------------------------------------------------------------------
    def proyectar_sistema(self, desde: date, hasta: date) -> ProyeccionGeneral:
        """Proyecta la demanda diaria del sistema completo.

        Args:
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            Una :class:`ProyeccionGeneral` con detalle y resumen.
        """
        detalle = self._proyector.proyectar_sistema(desde, hasta)
        usos = float(detalle["usos_predicho"].sum())
        dias = int(detalle["fecha"].nunique())
        paquete = self._proyector.metadatos(None)
        margen = margen_error_usos(paquete.metricas, factor=self._settings.factor_banda_error)

        resumen = ResumenProyeccion(
            etiqueta="Sistema MIO",
            dias=dias,
            usos_totales=usos,
            usos_diarios=usos / max(dias, 1),
            margen_error=margen,
            detalle_por_tipo_dia=self._resumen_por_tipo(detalle),
        )
        return ProyeccionGeneral(
            detalle=detalle,
            resumen=resumen,
            modelo=paquete.nombre,
            confiabilidad=confiabilidad_desde_error(paquete.metricas.get("wape", 15.0)),
            metricas=dict(paquete.metricas),
            detalle_horario=self._proyector.detalle_horario(detalle, paquete, margen),
            fecha_entrenamiento=paquete.fecha_entrenamiento,
            desde_entrenamiento=paquete.desde_entrenamiento,
            hasta_entrenamiento=paquete.hasta_entrenamiento,
        )

    # ------------------------------------------------------------------
    # Proyección por entidad (estación / ruta)
    # ------------------------------------------------------------------
    def proyectar_entidad(
        self,
        entidad: str,
        tipo: TipoEntidad,
        desde: date,
        hasta: date,
    ) -> ProyeccionPorEntidad:
        """Proyecta la demanda de una estación o ruta.

        Args:
            entidad: identificador de la estación o ruta.
            tipo: tipo de entidad.
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            Una :class:`ProyeccionPorEntidad` con detalle horario y resumen.
        """
        proyeccion = self._proyector.proyectar_entidad(entidad, tipo, desde, hasta)
        detalle = proyeccion.detalle_horario
        diaria = proyeccion.serie_diaria
        paquete = self._proyector.metadatos(tipo)

        usos = float(detalle["usos_predicho"].sum())
        dias = int(diaria["fecha"].nunique())
        etiqueta = f"{'Ruta' if tipo is TipoEntidad.RUTA else 'Estación'} {entidad}"

        peso = paquete.metricas.get("wape", 15.0)
        resumen = ResumenProyeccion(
            etiqueta=etiqueta,
            dias=dias,
            usos_totales=usos,
            usos_diarios=usos / max(dias, 1),
            margen_error=margen_error_usos(
                paquete.metricas, factor=self._settings.factor_banda_error
            ),
            detalle_por_tipo_dia=self._resumen_por_tipo(detalle),
        )
        por_tipo = pd.DataFrame(resumen.detalle_por_tipo_dia).T.reset_index(names="Tipo de día")
        return ProyeccionPorEntidad(
            detalle_horario=detalle,
            serie_diaria=diaria,
            resumen=resumen,
            confiabilidad=confiabilidad_desde_error(peso),
            modelo=paquete.nombre,
            etiqueta=etiqueta,
            por_tipo_dia=por_tipo,
            metricas=dict(paquete.metricas),
            fecha_entrenamiento=paquete.fecha_entrenamiento,
            desde_entrenamiento=paquete.desde_entrenamiento,
            hasta_entrenamiento=paquete.hasta_entrenamiento,
        )

    # ------------------------------------------------------------------
    # Todas las entidades (una sola pasada, luego se filtra)
    # ------------------------------------------------------------------
    def proyectar_grupo(self, tipo: TipoEntidad, desde: date, hasta: date) -> ProyeccionPorEntidad:
        """Proyecta **todas** las entidades de un tipo y devuelve el agregado.

        Es «todas las rutas» (o todas las estaciones) tratado como si fuera una sola, para
        que el resto de la vista (ficha, coherencia, gráfica, hora y Excel) funcione
        exactamente igual que con una entidad.

        Args:
            tipo: tipo de entidad (estación o ruta).
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            La :class:`ProyeccionPorEntidad` del grupo (``etiqueta = "Todas las rutas"``).
        """
        paquete = self._proyector.metadatos(tipo)
        detalle = self._proyector.proyectar_entidades(tipo, desde, hasta)
        entidades = sorted(detalle["entidad"].unique().tolist())
        margen = margen_error_usos(paquete.metricas, factor=self._settings.factor_banda_error)

        agregado = (
            detalle.groupby("fecha", as_index=False)
            .agg(usos_predicho=("usos_predicho", "sum"), dia_tipo_num=("dia_tipo_num", "first"))
            .sort_values("fecha")
            .reset_index(drop=True)
        )
        tipos = detalle.groupby("fecha")["dia_tipo_nom"].first().reindex(agregado["fecha"])
        agregado["dia_tipo_nom"] = tipos.to_numpy()
        agregado["usos_p10"] = (agregado["usos_predicho"] - margen).clip(lower=0)
        agregado["usos_p90"] = agregado["usos_predicho"] + margen
        agregado["demanda_real"] = float("nan")
        agregado["es_proyectado"] = True

        usos = float(agregado["usos_predicho"].sum())
        dias = int(agregado["fecha"].nunique())
        nombre = "Todas las estaciones" if tipo is TipoEntidad.ESTACION else "Todas las rutas"
        resumen = ResumenProyeccion(
            etiqueta=nombre,
            dias=dias,
            usos_totales=usos,
            usos_diarios=usos / max(dias, 1),
            margen_error=margen,
            detalle_por_tipo_dia=self._resumen_por_tipo(agregado),
        )
        horario = self._proyector.detalle_horario(agregado, paquete, margen, entidades=entidades)
        return ProyeccionPorEntidad(
            detalle_horario=horario,
            serie_diaria=agregado,
            resumen=resumen,
            confiabilidad=confiabilidad_desde_error(paquete.metricas.get("wape", 15.0)),
            modelo=paquete.nombre,
            etiqueta=nombre,
            metricas=dict(paquete.metricas),
            fecha_entrenamiento=paquete.fecha_entrenamiento,
            desde_entrenamiento=paquete.desde_entrenamiento,
            hasta_entrenamiento=paquete.hasta_entrenamiento,
            detalle_por_entidad=detalle[
                [
                    "entidad",
                    "fecha",
                    "dia_tipo_num",
                    "dia_tipo_nom",
                    "usos_predicho",
                    "es_proyectado",
                ]
            ].reset_index(drop=True),
        )

    def horario_por_entidad(
        self, detalle_por_entidad: pd.DataFrame, tipo: TipoEntidad
    ) -> pd.DataFrame:
        """Detalle horario de cada entidad del grupo (para exportarlo a Excel).

        Args:
            detalle_por_entidad: serie diaria por entidad de una proyección de grupo.
            tipo: tipo de entidad.

        Returns:
            DataFrame con ``fecha``, ``entidad``, ``hora`` y ``usos_hora``.
        """
        paquete = self._proyector.metadatos(tipo)
        return self._proyector.detalle_horario_por_entidad(detalle_por_entidad, paquete)

    def detalle_horario_entidad(
        self, tipo: TipoEntidad, entidad: str, serie_diaria: pd.DataFrame
    ) -> pd.DataFrame:
        """Detalle horario de una entidad a partir de su serie diaria ya calculada.

        Se usa al filtrar dentro de una proyección de grupo: no vuelve a predecir, solo
        reparte la serie con el perfil horario de esa entidad.

        Args:
            tipo: tipo de entidad.
            entidad: identificador de la estación o ruta.
            serie_diaria: serie diaria de la entidad (``fecha``, ``dia_tipo_num``,
                ``usos_predicho``).

        Returns:
            El detalle horario de la entidad.
        """
        paquete = self._proyector.metadatos(tipo)
        margen = margen_error_usos(paquete.metricas, factor=self._settings.factor_banda_error)
        return self._proyector.detalle_horario(serie_diaria, paquete, margen, entidad=entidad)

    def referencia_reciente_grupo(self, tipo: TipoEntidad, dias: int | None = None) -> float:
        """Demanda diaria real promedio reciente de **todas** las entidades de un tipo.

        Se usa como referencia de coherencia cuando se proyecta el grupo completo: la
        suma de las rutas no se parece al sistema, sino a la suma de las rutas.

        Args:
            tipo: tipo de entidad (estación o ruta).
            dias: días a promediar; ``None`` usa la configuración.

        Returns:
            El promedio diario real del grupo; ``0.0`` si no hay datos.
        """
        dias = dias or self._settings.dias_referencia_reciente
        fin = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin is None:
            return 0.0
        desde = fin - timedelta(days=max(1, dias) - 1)
        usos = self._repo.usos_diarios_por_entidad(desde, fin, self._repo.listar_entidades(tipo))
        if usos.empty:
            return 0.0
        diaria = usos.groupby("fecha")["usos"].sum()
        return float(diaria.sum()) / max(len(diaria), 1)

    def proyectar_conjunta(self, desde: date, hasta: date) -> ProyeccionConjunta:
        """Proyecta todas las estaciones y todas las rutas del mismo período.

        Args:
            desde: primera fecha a proyectar.
            hasta: última fecha a proyectar.

        Returns:
            La :class:`ProyeccionConjunta` con los grupos que tengan modelo entrenado.

        Raises:
            ValueError: si no hay ningún modelo de estación ni de ruta.
        """
        disponibles = self.modelos_disponibles()
        conjunto = ProyeccionConjunta()
        if disponibles["estaciones"]:
            conjunto.estaciones = self.proyectar_grupo(TipoEntidad.ESTACION, desde, hasta)
        if disponibles["rutas"]:
            conjunto.rutas = self.proyectar_grupo(TipoEntidad.RUTA, desde, hasta)
        if not conjunto.grupos():
            raise ValueError("No hay modelos de estaciones ni de rutas entrenados.")
        return conjunto

    def agregado_conjunto(self, conjunto: ProyeccionConjunta) -> pd.DataFrame:
        """Suma diaria de estaciones y rutas (equivale a la demanda del sistema).

        Args:
            conjunto: proyección conjunta.

        Returns:
            DataFrame con ``fecha``, ``estaciones``, ``rutas``, ``usos_predicho`` (suma),
            ``usos_p10``, ``usos_p90``, ``dia_tipo_num`` y ``dia_tipo_nom``.
        """
        partes: list[pd.DataFrame] = []
        tipos: pd.DataFrame | None = None
        for nombre, proyeccion in (
            ("estaciones", conjunto.estaciones),
            ("rutas", conjunto.rutas),
        ):
            if proyeccion is None:
                continue
            datos = proyeccion.serie_diaria.set_index("fecha")
            partes.append(datos[["usos_predicho"]].rename(columns={"usos_predicho": nombre}))
            if tipos is None:
                tipos = datos[["dia_tipo_num", "dia_tipo_nom"]]
        if not partes:
            return pd.DataFrame()
        unido = pd.concat(partes, axis=1).sort_index()
        nombres = [columna for columna in ("estaciones", "rutas") if columna in unido.columns]
        unido["usos_predicho"] = unido[nombres].sum(axis=1, min_count=1)
        margen = sum(
            proyeccion.resumen.margen_error
            for proyeccion in (conjunto.estaciones, conjunto.rutas)
            if proyeccion is not None
        )
        unido["usos_p10"] = (unido["usos_predicho"] - margen).clip(lower=0)
        unido["usos_p90"] = unido["usos_predicho"] + margen
        if tipos is not None:
            unido = unido.join(tipos)
        return unido.reset_index()

    def horario_conjunto(self, conjunto: ProyeccionConjunta) -> pd.DataFrame:
        """Detalle horario de la suma de estaciones y rutas.

        Args:
            conjunto: proyección conjunta.

        Returns:
            DataFrame horario con ``fecha``, ``hora``, ``entidad``, ``usos_predicho`` y
            la banda ``usos_p10``/``usos_p90``.
        """
        partes = [
            proyeccion.detalle_horario
            for proyeccion in (conjunto.estaciones, conjunto.rutas)
            if proyeccion is not None
        ]
        if not partes:
            return pd.DataFrame()
        horario = pd.concat(partes, ignore_index=True)
        agregado = (
            horario.groupby(["fecha", "hora"], as_index=False)
            .agg(
                usos_predicho=("usos_predicho", "sum"),
                usos_p10=("usos_p10", "sum"),
                usos_p90=("usos_p90", "sum"),
                dia_tipo_num=("dia_tipo_num", "first"),
                dia_tipo_nom=("dia_tipo_nom", "first"),
            )
            .sort_values(["fecha", "hora"])
            .reset_index(drop=True)
        )
        agregado["entidad"] = "Estaciones + rutas"
        agregado["es_proyectado"] = True
        return agregado

    # ------------------------------------------------------------------
    # Evaluación (backtest contra datos reales)
    # ------------------------------------------------------------------
    def evaluar_sistema(self, dias: int = 10) -> EvaluacionProyeccion:
        """Compara la proyección del sistema con la demanda real de los últimos días.

        Si el modelo guardado ya está entrenado hasta el último dato disponible, se
        entrena una **copia temporal** con corte anterior (no se guarda) para poder
        comparar sin fuga de información.

        Args:
            dias: número de días reales a comparar.

        Returns:
            Una :class:`EvaluacionProyeccion` con la comparación y las métricas.

        Raises:
            ValueError: si no hay datos suficientes.
        """
        paquete = self._proyector.metadatos(None)
        desde, hasta, reentrenado, corte = self._ventana_evaluacion(dias, paquete)
        if reentrenado:
            modelo, features = self._entrenador.modelo_con_corte("sistema", corte)
            detalle = self._proyector.detalle_con_modelo(modelo, features, desde, hasta)
        else:
            detalle = self._proyector.proyectar_sistema(desde, hasta)

        comparacion = self._comparar(detalle, desde, hasta, corte)
        return self._empaquetar(
            comparacion,
            desde,
            hasta,
            paquete,
            etiqueta="Sistema MIO",
            corte=corte,
            reentrenado=reentrenado,
        )

    def evaluar_entidad(
        self, entidad: str, tipo: TipoEntidad, dias: int = 10
    ) -> EvaluacionProyeccion:
        """Compara la proyección de una entidad con su demanda real reciente.

        Args:
            entidad: identificador de la estación o ruta.
            tipo: tipo de entidad.
            dias: número de días reales a comparar.

        Returns:
            Una :class:`EvaluacionProyeccion` con la comparación y las métricas.

        Raises:
            ValueError: si no hay datos suficientes.
        """
        paquete = self._proyector.metadatos(tipo)
        desde, hasta, reentrenado, corte = self._ventana_evaluacion(dias, paquete)
        alcance = "estaciones" if tipo is TipoEntidad.ESTACION else "rutas"

        if reentrenado:
            modelo, features = self._entrenador.modelo_con_corte(alcance, corte)
            detalle = self._proyector.detalle_con_modelo(
                modelo, features, desde, hasta, tipo=tipo, entidad=entidad
            )
        else:
            proyeccion = self._proyector.proyectar_entidad(entidad, tipo, desde, hasta)
            detalle = proyeccion.serie_diaria

        comparacion = self._comparar(detalle, desde, hasta, corte)
        etiqueta = f"{'Ruta' if tipo is TipoEntidad.RUTA else 'Estación'} {entidad}"
        return self._empaquetar(
            comparacion,
            desde,
            hasta,
            paquete,
            etiqueta=etiqueta,
            corte=corte,
            reentrenado=reentrenado,
        )

    def _ventana_evaluacion(
        self, dias: int, paquete: PaqueteModelo
    ) -> tuple[date, date, bool, date]:
        """Calcula la ventana de evaluación y si hay que reentrenar una copia.

        Si el modelo ya está entrenado hasta el último dato, la ventana quedaría sin
        días fuera de muestra: en ese caso se marca ``reentrenado=True`` para entrenar
        una copia temporal con corte anterior.

        Args:
            dias: número de días a evaluar.
            paquete: paquete de modelo del alcance.

        Returns:
            Tupla ``(desde, hasta, reentrenado, corte)``.

        Raises:
            ValueError: si no hay datos disponibles.
        """
        rango = self._repo.rango_fechas_usos()
        if rango is None:
            raise ValueError("No hay datos de usos para evaluar. Ejecuta la extracción.")
        fin_datos = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin_datos is None:
            raise ValueError("No hay datos de usos para evaluar. Ejecuta la extracción.")

        desde = fin_datos - timedelta(days=max(1, dias) - 1)
        corte_modelo = paquete.hasta_entrenamiento
        if desde > corte_modelo:
            return desde, fin_datos, False, corte_modelo
        # El modelo ya vio esos días: se reentrena una copia con corte anterior.
        return desde, fin_datos, True, desde - timedelta(days=1)

    @staticmethod
    def _comparar(detalle: pd.DataFrame, desde: date, hasta: date, corte: date) -> pd.DataFrame:
        """Construye la tabla real vs. predicho del periodo evaluado.

        Args:
            detalle: detalle diario con ``demanda_real`` y ``usos_predicho``.
            desde: primera fecha evaluada.
            hasta: última fecha evaluada.
            corte: último día de historia del modelo evaluado.

        Returns:
            DataFrame con fecha, demanda real, predicha, error, error absoluto y APE.
        """
        datos = detalle.copy()
        datos["fecha"] = pd.to_datetime(datos["fecha"])
        datos = datos[
            (datos["fecha"].dt.date >= desde)
            & (datos["fecha"].dt.date <= hasta)
            & datos["demanda_real"].notna()
        ].sort_values("fecha")
        if datos.empty:
            raise ValueError("No hay demanda real en el periodo de evaluación seleccionado.")

        real = datos["demanda_real"].astype(float)
        predicha = datos["usos_predicho"].astype(float)
        error = real - predicha

        comparacion = pd.DataFrame(
            {
                "Fecha": datos["fecha"].dt.strftime("%Y-%m-%d"),
                "Dia.tipo": datos.get("dia_tipo_nom", ""),
                "Demanda_Real": real.round(0).astype("Int64"),
                "Demanda_Predicha": predicha.round(0).astype("Int64"),
                "error": error.round(0).astype("Int64"),
                "error_abs": error.abs().round(0).astype("Int64"),
                "APE_%": np.where(real > 0, (error.abs() / real * 100).round(2), np.nan),
            }
        )
        denominador = real.abs() + predicha.abs()
        comparacion["SAPE_%"] = np.where(
            denominador > 0, (200 * error.abs() / denominador).round(2), np.nan
        )
        comparacion["Fuera_de_muestra"] = [
            pd.Timestamp(fecha).date() > corte for fecha in datos["fecha"]
        ]
        return comparacion.reset_index(drop=True)

    @staticmethod
    def _empaquetar(
        comparacion: pd.DataFrame,
        desde: date,
        hasta: date,
        paquete: PaqueteModelo,
        etiqueta: str,
        corte: date,
        reentrenado: bool,
    ) -> EvaluacionProyeccion:
        """Calcula las métricas de la comparación y arma el resultado.

        Args:
            comparacion: tabla real vs. predicho.
            desde: primera fecha evaluada.
            hasta: última fecha evaluada.
            paquete: paquete de modelo evaluado.
            etiqueta: alcance evaluado.
            corte: último día de historia del modelo evaluado.
            reentrenado: si hubo que reentrenar una copia temporal.

        Returns:
            La :class:`EvaluacionProyeccion` lista para la interfaz.
        """
        metricas = calcular_metricas(
            comparacion["Demanda_Real"].astype(float).to_numpy(),
            comparacion["Demanda_Predicha"].astype(float).to_numpy(),
        )
        return EvaluacionProyeccion(
            comparacion=comparacion,
            metricas=metricas,
            confiabilidad=confiabilidad_desde_error(metricas.get("wape", 15.0)),
            dias_evaluados=len(comparacion),
            dias_fuera_de_muestra=int(comparacion["Fuera_de_muestra"].sum()),
            desde=desde,
            hasta=hasta,
            modelo=paquete.nombre,
            etiqueta=etiqueta,
            corte_entrenamiento=corte,
            reentrenado=reentrenado,
        )

    # ------------------------------------------------------------------
    # Apoyo
    # ------------------------------------------------------------------
    @staticmethod
    def _resumen_por_tipo(
        detalle: pd.DataFrame, columna_fecha: str = "fecha"
    ) -> dict[str, dict[str, float]]:
        """Agrupa el detalle por tipo de día.

        Args:
            detalle: detalle con ``dia_tipo_nom`` y ``usos_predicho``.
            columna_fecha: nombre de la columna de fecha.

        Returns:
            Mapa tipo de día -> métricas parciales (días, demanda total y diaria).
        """
        if "dia_tipo_nom" not in detalle.columns:
            return {}
        resumen: dict[str, dict[str, float]] = {}
        for tipo, grupo in detalle.groupby("dia_tipo_nom"):
            usos = float(grupo["usos_predicho"].sum())
            dias = int(grupo[columna_fecha].nunique())
            resumen[str(tipo)] = {
                "dias": dias,
                "usos_totales": usos,
                "usos_diarios": usos / max(dias, 1),
            }
        return resumen


def clasificar_tendencia(serie: pd.Series) -> str:
    """Clasifica la tendencia de una serie por su pendiente normalizada.

    Args:
        serie: valores de la serie (en orden temporal).

    Returns:
        ``"creciente"``, ``"decreciente"`` o ``"estable"``.
    """
    valores = np.asarray(serie, dtype=float)
    if len(valores) < 2 or valores.mean() == 0:
        return "estable"
    pendiente = np.polyfit(np.arange(len(valores)), valores, 1)[0]
    cambio_relativo = pendiente * len(valores) / valores.mean()
    if cambio_relativo > 0.05:
        return "creciente"
    if cambio_relativo < -0.05:
        return "decreciente"
    return "estable"
