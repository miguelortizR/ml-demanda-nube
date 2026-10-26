"""Orquestador del ciclo de entrenamiento (comparación + afinación + ensamble).

Ciclo completo, replicable y trazable:

1. Construir el dataset (sin fuga de datos).
2. Para **cada** candidato: buscar hiperparámetros con Optuna sobre validación
   cruzada temporal y conservar sus métricas.
3. Obtener predicciones *out-of-fold* de cada candidato afinado y **optimizar
   los pesos del ensamble** minimizando el MAE (SciPy).
4. Elegir el **campeón** (mejor MAE de validación) entre candidatos y ensamble.
5. Reentrenar el campeón con toda la historia disponible.
6. Registrar parámetros, métricas y artefactos en **MLflow**.
7. Guardar el paquete de modelo para que la proyección lo consuma.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.infrastructure.almacen_modelos import AlmacenModelos, PaqueteModelo
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_calendario import RepositorioCalendario
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos
from mio_demanda.infrastructure.tracking import RastreadorMLflow
from mio_demanda.ml.datasets import (
    Dataset,
    construir_dataset_entidad,
    construir_dataset_sistema,
)
from mio_demanda.ml.evaluacion import calcular_metricas, validacion_cruzada_temporal
from mio_demanda.ml.modelos import (
    EnsamblePonderado,
    crear_pipeline,
    nombres_candidatos,
    optimizar_pesos_ensamble,
    reentrenar,
)
from mio_demanda.ml.prediccion import alinear_features
from mio_demanda.ml.tuning import optimizar_hiperparametros

logger = obtener_logger(__name__)

# Límite por defecto de filas para la afinación (las más recientes) en datasets grandes.
MAX_FILAS_TUNING_ENTIDAD = 25_000


@dataclass(slots=True)
class ResultadoEntrenamiento:
    """Resultado completo de un ciclo de entrenamiento.

    Attributes:
        paquete: paquete de modelo campeón (listo para guardar/usar).
        comparativa: tabla con las métricas de cada candidato y del ensamble.
        pesos_ensamble: pesos optimizados del ensamble.
        campeon: nombre del modelo ganador (``"ensamble"`` o un candidato).
        historial_tuning: historial de *trials* por candidato.
        metricas_cv_campeon: métricas de validación cruzada del campeón.
    """

    paquete: PaqueteModelo
    comparativa: pd.DataFrame
    pesos_ensamble: dict[str, float]
    campeon: str
    historial_tuning: dict[str, list[dict[str, Any]]]
    metricas_cv_campeon: dict[str, float]


class EntrenadorModelo:
    """Entrena y compara modelos de demanda (sistema y por entidad).

    Args:
        settings: configuración de la aplicación.
        repo: repositorio de datos (se crea si no se inyecta).
        repo_cal: repositorio del calendario (se crea si no se inyecta).
        tracking: rastreador de MLflow (se crea si no se inyecta).
        notificar: callback de progreso ``(mensaje: str) -> None`` para la UI.
    """

    def __init__(
        self,
        settings: Settings,
        repo: RepositorioDatos | None = None,
        repo_cal: RepositorioCalendario | None = None,
        tracking: RastreadorMLflow | None = None,
        notificar: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo or RepositorioDatos(settings)
        self._repo_cal = repo_cal or RepositorioCalendario(settings)
        self._tracking = tracking or RastreadorMLflow(settings)
        # Copias temporales del backtest ya entrenadas: reentrenar el mismo corte es
        # costoso (decenas de segundos) y el vigilante evalúa repetidamente.
        self._temporales: dict[tuple[str, date], tuple[Any, list[str]]] = {}
        self._almacen = AlmacenModelos(settings)
        self._notificar = notificar or (lambda _mensaje: None)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def entrenar_sistema(
        self,
        desde: date,
        hasta: date,
        n_trials: int | None = None,
        n_splits: int | None = None,
        observacion: str = "",
    ) -> ResultadoEntrenamiento:
        """Entrena el modelo de demanda diaria del sistema completo.

        Args:
            desde: fecha inicial de la historia de entrenamiento.
            hasta: fecha final de la historia de entrenamiento.
            n_trials: presupuesto de Optuna por candidato.
            n_splits: particiones de la validación cruzada.
            observacion: nota del usuario sobre por qué entrenó así (queda guardada).

        Returns:
            El :class:`ResultadoEntrenamiento` del sistema.
        """
        dataset = construir_dataset_sistema(
            self._repo, self._repo_cal, desde, hasta, **self._opciones_features()
        )
        return self._entrenar(
            dataset.solo_entrenamiento(),
            nombre="sistema",
            alcance="sistema",
            n_trials=n_trials,
            n_splits=n_splits,
            observacion=observacion,
        )

    def entrenar_entidad(
        self,
        tipo: TipoEntidad,
        desde: date,
        hasta: date,
        entidades: list[str] | None = None,
        n_trials: int | None = None,
        n_splits: int | None = None,
        max_filas_tuning: int | None = None,
        observacion: str = "",
    ) -> ResultadoEntrenamiento:
        """Entrena el modelo de demanda diaria por entidad (estaciones o rutas).

        Args:
            tipo: tipo de entidad a modelar.
            desde: fecha inicial de la historia de entrenamiento.
            hasta: fecha final de la historia de entrenamiento.
            entidades: subconjunto de entidades (por defecto, todas del tipo).
            n_trials: presupuesto de Optuna por candidato.
            n_splits: particiones de la validación cruzada.
            max_filas_tuning: límite de filas para la afinación (acelera la búsqueda).
            observacion: nota del usuario sobre por qué entrenó así (queda guardada).

        Returns:
            El :class:`ResultadoEntrenamiento` de las entidades.
        """
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
        nombre = "estaciones" if tipo is TipoEntidad.ESTACION else "rutas"
        return self._entrenar(
            dataset.solo_entrenamiento(),
            nombre=nombre,
            alcance="entidad",
            n_trials=n_trials,
            n_splits=n_splits,
            max_filas_tuning=max_filas_tuning,
            etiquetas={"tipo_entidad": tipo.value},
            observacion=observacion,
        )

    # ------------------------------------------------------------------
    # Opciones de ingeniería de características (desde la configuración)
    # ------------------------------------------------------------------
    def _opciones_features(self) -> dict[str, int]:
        """Ventanas de las anclas tomadas de la configuración.

        Returns:
            Diccionario con ``dias_calentamiento``, ``ventana_reciente`` y
            ``min_obs_ancla``.
        """
        return {
            "dias_calentamiento": self._settings.dias_calentamiento,
            "ventana_reciente": self._settings.ventana_reciente_dias,
            "min_obs_ancla": self._settings.min_obs_ancla,
        }

    # ------------------------------------------------------------------
    # Activación manual del modelo a proyectar
    # ------------------------------------------------------------------
    def activar_modelo(self, alcance: str, modelo: str) -> PaqueteModelo:
        """Reentrena y guarda el candidato elegido como modelo de proyección.
        Permite que el profesional decida **con qué modelo se proyecta** en lugar de
        usar siempre el campeón automático. Se reutilizan los hiperparámetros ya
        afinados (no se vuelve a buscar con Optuna) y las métricas de validación del
        candidato elegido.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.
            modelo: nombre del candidato (``"ensamble"`` o el de un candidato).

        Returns:
            El paquete actualizado, con el modelo elegido como activo.

        Raises:
            ValueError: si el alcance o el modelo no son válidos.
            FileNotFoundError: si el paquete no existe.
        """
        paquete = self._almacen.cargar(alcance)
        comparativa = {str(fila["modelo"]): fila for fila in paquete.extras.get("comparativa", [])}
        if modelo not in comparativa:
            raise ValueError(f"El modelo '{modelo}' no está entre los candidatos de '{alcance}'")

        dataset = self._dataset_para(alcance, paquete)
        # Se respeta el contrato de variables del paquete: si el catálogo de eventos
        # cambió después de entrenar, las variables nuevas se rellenan con 0.
        entrada = alinear_features(dataset.X, paquete.features)
        X = entrada.to_numpy(dtype=float)
        y = dataset.y.to_numpy(dtype=float)
        parametros = paquete.extras.get("mejores_parametros", {})

        if modelo == "ensamble":
            nombres = list(paquete.extras.get("nombres_ensamble", []))
            pesos = [float(peso) for peso in paquete.extras.get("pesos_ensamble", [])]
            if not nombres:
                raise ValueError("El ensamble no tiene componentes guardados")
            modelos = {
                nombre: reentrenar(crear_pipeline(nombre, parametros.get(nombre, {})), X, y)
                for nombre in nombres
            }
            modelo_ajustado: Any = EnsamblePonderado(modelos=modelos, pesos=pesos)
        else:
            modelo_ajustado = reentrenar(crear_pipeline(modelo, parametros.get(modelo, {})), X, y)

        fila = comparativa[modelo]
        paquete.modelo = modelo_ajustado
        paquete.metricas = {
            clave: float(valor)
            for clave, valor in fila.items()
            if clave not in {"modelo", "tipo"} and isinstance(valor, (int, float))
        }
        paquete.extras = {**paquete.extras, "campeon": modelo, "activado_manualmente": True}
        self._almacen.guardar(paquete)
        logger.info("Modelo activado para '%s': %s", alcance, modelo)
        self._notificar(f"Modelo activo de {alcance}: {modelo}")
        return paquete

    def eliminar_modelo(self, alcance: str) -> bool:
        """Elimina el paquete de modelo de un alcance.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.

        Returns:
            ``True`` si el paquete existía y fue eliminado.
        """
        eliminado = self._almacen.eliminar(alcance)
        if eliminado:
            logger.warning("Modelo eliminado: %s", alcance)
        return eliminado

    # ------------------------------------------------------------------
    # Historial de entrenamientos
    # ------------------------------------------------------------------
    def listar_versiones(self, alcance: str) -> list[dict[str, Any]]:
        """Entrenamientos archivados de un alcance, el más reciente primero.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.

        Returns:
            Lista de metadatos con ``version``, ``activo`` y ``tamano_mb``.
        """
        return self._almacen.listar_versiones(alcance)

    def activar_version(self, version: str) -> PaqueteModelo:
        """Pone a proyectar un entrenamiento archivado, sin reentrenar.

        Args:
            version: nombre identificable del entrenamiento.

        Returns:
            El paquete restaurado.

        Raises:
            FileNotFoundError: si la versión no existe.
        """
        paquete = self._almacen.activar_version(version)
        logger.info("Versión activada para %s: %s", paquete.nombre, version)
        self._notificar(f"Versión activa de {paquete.nombre}: {version}")
        return paquete

    def listar_papelera(self) -> list[dict[str, Any]]:
        """Lista lo eliminado que se puede recuperar.

        Returns:
            Lista de diccionarios con ``archivo``, ``origen`` y ``mb``.
        """
        return self._almacen.listar_papelera()

    def restaurar_de_papelera(self, archivo: str) -> bool:
        """Devuelve un archivo de la papelera a su carpeta original.

        Args:
            archivo: nombre del archivo en la papelera.

        Returns:
            ``True`` si se restauró.
        """
        return self._almacen.restaurar_de_papelera(archivo)

    def eliminar_version(self, version: str) -> bool:
        """Elimina un entrenamiento del historial.

        Args:
            version: nombre identificable del entrenamiento.

        Returns:
            ``True`` si existía y se eliminó.
        """
        return self._almacen.eliminar_version(version)

    def _dataset_para(
        self, alcance: str, paquete: PaqueteModelo, hasta: date | None = None
    ) -> Dataset:
        """Reconstruye el dataset de entrenamiento de un alcance.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.
            paquete: paquete guardado (define el inicio del periodo).
            hasta: fecha final a usar; por defecto, la del paquete.

        Returns:
            El dataset con las filas de entrenamiento.

        Raises:
            ValueError: si el alcance no es válido.
        """
        desde = paquete.desde_entrenamiento
        fin = hasta or paquete.hasta_entrenamiento
        if alcance == "sistema":
            dataset = construir_dataset_sistema(
                self._repo, self._repo_cal, desde, fin, **self._opciones_features()
            )
        elif alcance == "estaciones":
            dataset = construir_dataset_entidad(
                self._repo,
                self._repo_cal,
                TipoEntidad.ESTACION,
                desde,
                fin,
                min_cobertura=self._settings.min_cobertura_entidad,
                **self._opciones_features(),
            )
        elif alcance == "rutas":
            dataset = construir_dataset_entidad(
                self._repo,
                self._repo_cal,
                TipoEntidad.RUTA,
                desde,
                fin,
                min_cobertura=self._settings.min_cobertura_entidad,
                **self._opciones_features(),
            )
        else:
            raise ValueError(f"Alcance no soportado: {alcance}")
        return dataset.solo_entrenamiento()

    def modelo_con_corte(self, alcance: str, corte: date) -> tuple[Any, list[str]]:
        """Entrena una **copia temporal** del campeón con la historia hasta ``corte``.

        Se usa para el backtest cuando el modelo guardado ya está entrenado hasta el
        último dato disponible. Reutiliza los hiperparámetros afinalados y los pesos
        del ensamble; **no** guarda nada ni ejecuta Optuna.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.
            corte: último día de historia permitido para entrenar.

        Returns:
            Tupla ``(modelo, variables)`` lista para predecir.

        Raises:
            FileNotFoundError: si el paquete no existe.
            ValueError: si no hay historia suficiente antes del corte.
        """
        paquete = self._almacen.cargar(alcance)
        clave = (alcance, corte)
        if clave in self._temporales:
            return self._temporales[clave]

        dataset = self._dataset_para(alcance, paquete, hasta=corte)
        if len(dataset) == 0:
            raise ValueError(
                f"No hay historia antes del {corte} para reentrenar '{alcance}' y evaluar."
            )

        X = alinear_features(dataset.X, paquete.features).to_numpy(dtype=float)
        y = dataset.y.to_numpy(dtype=float)
        parametros = paquete.extras.get("mejores_parametros", {})
        campeon = str(paquete.extras.get("campeon", "ensamble"))

        if campeon == "ensamble":
            nombres = list(paquete.extras.get("nombres_ensamble", []))
            pesos = [float(peso) for peso in paquete.extras.get("pesos_ensamble", [])]
            componentes = {
                nombre: reentrenar(crear_pipeline(nombre, parametros.get(nombre, {})), X, y)
                for nombre in nombres
            }
            modelo: Any = EnsamblePonderado(modelos=componentes, pesos=pesos)
        else:
            modelo = reentrenar(crear_pipeline(campeon, parametros.get(campeon, {})), X, y)

        self._notificar(f"Copia temporal de '{alcance}' entrenada hasta {corte} para el backtest")
        self._temporales[clave] = (modelo, list(paquete.features))
        return self._temporales[clave]

    # ------------------------------------------------------------------
    # Núcleo
    # ------------------------------------------------------------------
    def _entrenar(
        self,
        dataset: Dataset,
        nombre: str,
        alcance: str,
        n_trials: int | None = None,
        n_splits: int | None = None,
        max_filas_tuning: int | None = None,
        etiquetas: dict[str, str] | None = None,
        observacion: str = "",
    ) -> ResultadoEntrenamiento:
        """Ejecuta el ciclo de entrenamiento sobre un dataset ya construido."""
        n_splits = n_splits or self._settings.n_splits_cv
        n_trials = n_trials or self._settings.n_trials_optuna
        X = dataset.X.to_numpy(dtype=float)
        y = dataset.y.to_numpy(dtype=float)

        comparativa: list[dict[str, Any]] = []
        historial_tuning: dict[str, list[dict[str, Any]]] = {}
        oof_por_modelo: dict[str, np.ndarray] = {}
        pipelines: dict[str, Any] = {}
        mejores_params: dict[str, dict[str, Any]] = {}

        for nombre_modelo in nombres_candidatos():
            self._notificar(f"Optimizando {nombre_modelo} ({n_trials} trials)...")
            tuning = optimizar_hiperparametros(
                nombre_modelo,
                X,
                y,
                n_splits=n_splits,
                n_trials=n_trials,
                random_state=self._settings.random_state,
                max_muestras=max_filas_tuning,
                callback=lambda r, nm=nombre_modelo: self._notificar(
                    f"  {nm}: trial {r.n_trials} (mejor MAE {r.metricas_cv.get('mae', float('nan')):,.0f})"
                ),
            )
            historial_tuning[nombre_modelo] = tuning.historial
            mejores_params[nombre_modelo] = tuning.mejores_parametros
            pipeline = crear_pipeline(nombre_modelo, tuning.mejores_parametros)
            pipelines[nombre_modelo] = pipeline

            resultado = validacion_cruzada_temporal(
                pipeline, X, y, n_splits=n_splits, recolectar_oof=True
            )
            oof_por_modelo[nombre_modelo] = resultado.oof_predicciones
            comparativa.append(
                {"modelo": nombre_modelo, "tipo": "individual", **resultado.metricas}
            )

        # --- Ensamble: pesos óptimos sobre predicciones out-of-fold ---
        pesos, metricas_ensamble, indices = self._construir_ensamble(oof_por_modelo, y)
        nombres_validos = list(oof_por_modelo)
        comparativa.append({"modelo": "ensamble", "tipo": "ensamble", **metricas_ensamble})

        tabla = pd.DataFrame(comparativa).sort_values("mae").reset_index(drop=True)
        campeon = str(tabla.iloc[0]["modelo"])
        metricas_campeon = {
            k: float(v) for k, v in tabla.iloc[0].items() if k not in {"modelo", "tipo"}
        }

        self._notificar(f"Campeón: {campeon} (MAE {metricas_campeon['mae']:,.0f})")

        modelo_final = self._construir_campeon(
            campeon, pipelines, pesos, nombres_validos, X, y, indices
        )

        paquete = PaqueteModelo(
            nombre=nombre,
            alcance=alcance,
            algoritmo=self._describir_campeon(campeon, pesos, nombres_validos, mejores_params),
            features=list(dataset.columnas_features),
            metricas=metricas_campeon,
            desde_entrenamiento=dataset.fechas.min().date(),
            hasta_entrenamiento=dataset.fechas.max().date(),
            extras={
                "comparativa": json.loads(tabla.to_json(orient="records")),
                "pesos_ensamble": [float(peso) for peso in pesos],
                "nombres_ensamble": nombres_validos,
                "mejores_parametros": mejores_params,
                "campeon": campeon,
                "activado_manualmente": False,
                "observacion": observacion.strip(),
            },
            modelo=modelo_final,
        )
        self._registrar_en_mlflow(paquete, tabla, mejores_params, pesos, nombres_validos, etiquetas)
        self._almacen.guardar(paquete)

        return ResultadoEntrenamiento(
            paquete=paquete,
            comparativa=tabla,
            pesos_ensamble=dict(zip(nombres_validos, pesos, strict=True)),
            campeon=campeon,
            historial_tuning=historial_tuning,
            metricas_cv_campeon=metricas_campeon,
        )

    def _construir_ensamble(
        self, oof_por_modelo: dict[str, np.ndarray], y: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        """Optimiza los pesos del ensamble y calcula sus métricas.

        Args:
            oof_por_modelo: predicciones out-of-fold por candidato.
            y: vector objetivo.

        Returns:
            Tupla ``(pesos, metricas, indices_comunes)``.
        """
        matrices = [np.asarray(pred, dtype=float) for pred in oof_por_modelo.values()]
        validez = np.ones(len(y), dtype=bool)
        for pred in matrices:
            validez &= ~np.isnan(pred)
        indices = np.flatnonzero(validez)

        P = np.vstack([pred[indices] for pred in matrices])
        pesos = optimizar_pesos_ensamble(P, y[indices])
        pred_ensamble = pesos @ P
        return pesos, calcular_metricas(y[indices], pred_ensamble), indices

    def _construir_campeon(
        self,
        campeon: str,
        pipelines: dict[str, Any],
        pesos: np.ndarray,
        nombres_validos: list[str],
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> Any:
        """Reentrena el campeón (modelo individual o ensamble) con toda la historia."""
        if campeon == "ensamble":
            modelos = {nombre: reentrenar(pipelines[nombre], X, y) for nombre in nombres_validos}
            return EnsamblePonderado(modelos=modelos, pesos=[float(p) for p in pesos])
        return reentrenar(pipelines[campeon], X, y)

    @staticmethod
    def _describir_campeon(
        campeon: str,
        pesos: np.ndarray,
        nombres_validos: list[str],
        mejores_params: dict[str, dict[str, Any]],
    ) -> str:
        """Genera una descripción legible del modelo campeón."""
        if campeon == "ensamble":
            partes = ", ".join(
                f"{nombre} (w={peso:.2f})"
                for nombre, peso in zip(nombres_validos, pesos, strict=True)
            )
            return f"Ensamble ponderado optimizado: {partes}"
        return f"{campeon} con hiperparámetros {mejores_params.get(campeon, {})}"

    def _registrar_en_mlflow(
        self,
        paquete: PaqueteModelo,
        comparativa: pd.DataFrame,
        mejores_params: dict[str, dict[str, Any]],
        pesos: np.ndarray,
        nombres_validos: list[str],
        etiquetas: dict[str, str] | None,
    ) -> None:
        """Registra el entrenamiento completo en MLflow (si está disponible)."""
        etiquetas_run = {"modelo": paquete.nombre, "campeon": str(paquete.extras.get("campeon"))}
        if etiquetas:
            etiquetas_run.update(etiquetas)

        with self._tracking.run(f"entrenamiento_{paquete.nombre}", etiquetas=etiquetas_run):
            for tabla_row in comparativa.to_dict(orient="records"):
                nombre_modelo = str(tabla_row["modelo"])
                prefijo = f"{nombre_modelo}_"
                self._tracking.registrar_metricas_por_grupo(
                    f"cv_{prefijo}",
                    {k: v for k, v in tabla_row.items() if isinstance(v, (int, float))},
                )
            for nombre_modelo, parametros in mejores_params.items():
                self._tracking.registrar_parametros(
                    {f"{nombre_modelo}__{k}": v for k, v in parametros.items()}
                )
            for nombre_modelo, peso in zip(nombres_validos, pesos, strict=True):
                self._tracking.registrar_metricas({f"peso_{nombre_modelo}": float(peso)})
            self._tracking.registrar_metricas_por_grupo("campeon_", paquete.metricas)
            self._tracking.registrar_artefacto(self._settings.models_dir / f"{paquete.nombre}.json")
            # Se guarda la corrida para poder enlazarla desde la app (el análisis
            # detallado se hace en MLflow, no dentro de la app).
            paquete.extras["mlflow_run_id"] = self._tracking.run_id
            paquete.extras["mlflow_experiment_id"] = self._tracking.experiment_id
