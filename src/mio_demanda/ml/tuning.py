"""Búsqueda de hiperparámetros con Optuna sobre validación cruzada temporal.

Cada candidato se optimiza minimizando el MAE promedio de una
``TimeSeriesSplit``. El resultado (mejores parámetros + métricas de la CV) se
registra después en MLflow y se usa para construir el ensamble.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.ml.evaluacion import validacion_cruzada_temporal
from mio_demanda.ml.modelos import crear_pipeline, espacio_por_modelo

logger = obtener_logger(__name__)

# Silencia el ruido de Optuna; el progreso se reporta por nuestra propia callback.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class ResultadoTuning:
    """Resultado de la búsqueda de hiperparámetros de un candidato.

    Attributes:
        nombre_modelo: candidato optimizado.
        mejores_parametros: hiperparámetros del mejor *trial*.
        metricas_cv: métricas de validación cruzada del mejor *trial*.
        n_trials: número de *trials* ejecutados.
        historial: métricas de todos los *trials* (para gráficas de convergencia).
    """

    nombre_modelo: str
    mejores_parametros: dict[str, Any]
    metricas_cv: dict[str, float]
    n_trials: int
    historial: list[dict[str, Any]] = field(default_factory=list)


def _sugerir(trial: optuna.Trial, nombre: str, definicion: tuple[str, Any]) -> Any:
    """Traduce una definición de espacio a una sugerencia de Optuna.

    Args:
        trial: *trial* activo.
        nombre: nombre del parámetro (con prefijo ``modelo__``).
        definicion: tupla ``(tipo, rango)`` del espacio de búsqueda.

    Returns:
        El valor sugerido.
    """
    tipo, rango = definicion
    if tipo == "int":
        return trial.suggest_int(nombre, int(rango[0]), int(rango[1]), log=False)
    if tipo == "float":
        minimo, maximo, log = rango
        return trial.suggest_float(nombre, float(minimo), float(maximo), log=bool(log))
    raise ValueError(f"Tipo de parámetro no soportado: {tipo}")


def optimizar_hiperparametros(
    nombre_modelo: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    n_trials: int = 25,
    random_state: int = 42,
    max_muestras: int | None = None,
    callback: Callable[[ResultadoTuning], None] | None = None,
) -> ResultadoTuning:
    """Optimiza los hiperparámetros de un candidato con Optuna.

    Args:
        nombre_modelo: nombre del candidato (ver :data:`mio_demanda.ml.modelos.CANDIDATOS`).
        X: matriz de variables.
        y: vector objetivo.
        n_splits: particiones de la validación cruzada temporal.
        n_trials: presupuesto de búsqueda.
        random_state: semilla para reproducibilidad.
        max_muestras: si se indica, usa solo las últimas ``max_muestras`` filas
            para la búsqueda (acelera la afinación en datasets grandes).
        callback: función opcional invocada al terminar cada *trial*.

    Returns:
        Un :class:`ResultadoTuning` con los mejores parámetros y sus métricas.
    """
    X_uso, y_uso = _recortar(X, y, max_muestras)
    espacio = espacio_por_modelo(nombre_modelo)

    def objetivo(trial: optuna.Trial) -> float:
        parametros = {
            nombre: _sugerir(trial, nombre, definicion) for nombre, definicion in espacio.items()
        }
        try:
            pipeline = crear_pipeline(nombre_modelo, parametros)
            resultado = validacion_cruzada_temporal(pipeline, X_uso, y_uso, n_splits=n_splits)
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            raise
        trial.set_user_attr("metricas", resultado.metricas)
        return resultado.metricas["mae"]

    estudio = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        study_name=f"{nombre_modelo}",
    )

    def _callback(estudio_interno: optuna.Study, _trial: optuna.FrozenTrial) -> None:
        if callback is None:
            return
        completados = [t for t in estudio_interno.trials if t.value is not None]
        if not completados:
            return
        mejor = min(completados, key=lambda t: t.value)
        callback(
            ResultadoTuning(
                nombre_modelo=nombre_modelo,
                mejores_parametros=dict(mejor.params),
                metricas_cv=dict(mejor.user_attrs.get("metricas", {})),
                n_trials=len(completados),
            )
        )

    estudio.optimize(objetivo, n_trials=n_trials, callbacks=[_callback], catch=(Exception,))

    completados = [t for t in estudio.trials if t.value is not None]
    if not completados:
        motivos = {t.user_attrs.get("error", "error desconocido") for t in estudio.trials}
        raise RuntimeError(
            f"Ningún trial de '{nombre_modelo}' terminó correctamente. Motivos: {sorted(motivos)}"
        )

    historial = [
        {"trial": t.number, "mae": t.value, **t.user_attrs.get("metricas", {})} for t in completados
    ]
    mejor = min(completados, key=lambda t: t.value)
    resultado = ResultadoTuning(
        nombre_modelo=nombre_modelo,
        mejores_parametros=dict(mejor.params),
        metricas_cv=dict(mejor.user_attrs.get("metricas", {})),
        n_trials=len(completados),
        historial=historial,
    )
    logger.info(
        "%s -> MAE CV %.2f (MAE en la partición mejor) con %s trials",
        nombre_modelo,
        resultado.metricas_cv.get("mae", float("nan")),
        resultado.n_trials,
    )
    return resultado


def _recortar(
    X: np.ndarray, y: np.ndarray, max_muestras: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Toma las últimas ``max_muestras`` filas (las más recientes)."""
    if max_muestras is None or len(X) <= max_muestras:
        return X, y
    return X[-max_muestras:], y[-max_muestras:]
