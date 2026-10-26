"""Fábrica de modelos candidatos y ensamble ponderado.

Estrategia (problema de **regresión**):

1. Se definen varios candidatos con sesgos distintos (boosting, bosque, lineal).
2. Cada candidato se afina con Optuna sobre validación cruzada temporal.
3. Se construye un **ensamble** como combinación lineal convexa de los
   candidatos, con pesos **optimizados empíricamente** por SciPy para minimizar
   el MAE sobre las predicciones *out-of-fold* (se elimina así cualquier peso
   heurístico fijado "a ojo").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)


# ======================================================================
# Fábrica de candidatos
# ======================================================================
def _espacio_hist_gb() -> dict[str, tuple[str, Any]]:
    """Espacio de búsqueda de ``HistGradientBoostingRegressor``."""
    return {
        "learning_rate": ("float", (0.01, 0.2, True)),
        "max_iter": ("int", (200, 1200)),
        "max_leaf_nodes": ("int", (15, 63)),
        "min_samples_leaf": ("int", (5, 40)),
        "l2_regularization": ("float", (1e-3, 1.0, True)),
    }


def _espacio_random_forest() -> dict[str, tuple[str, Any]]:
    """Espacio de búsqueda de ``RandomForestRegressor``.

    Se acota el número de árboles: por encima de ~300 el bosque deja de mejorar de
    forma apreciable y el coste de entrenamiento se dispara (es el candidato más
    lento del conjunto).
    """
    return {
        "n_estimators": ("int", (100, 300)),
        "max_depth": ("int", (5, 30)),
        "min_samples_leaf": ("int", (1, 20)),
        "max_features": ("float", (0.3, 1.0, False)),
    }


def _espacio_ridge() -> dict[str, tuple[str, Any]]:
    """Espacio de búsqueda de ``Ridge`` (modelo lineal de referencia)."""
    return {"alpha": ("float", (0.01, 100.0, True))}


CANDIDATOS: dict[str, Callable[[], tuple[Any, dict[str, tuple[str, Any]]]]] = {
    "hist_gradient_boosting": lambda: (
        HistGradientBoostingRegressor(random_state=42, early_stopping=False),
        _espacio_hist_gb(),
    ),
    "random_forest": lambda: (
        # n_jobs=1 a propósito: en Windows + OneDrive el pool de procesos de joblib
        # puede quedar bloqueado; con un solo proceso el entrenamiento es determinista.
        RandomForestRegressor(random_state=42, n_jobs=1),
        _espacio_random_forest(),
    ),
    "ridge": lambda: (Ridge(), _espacio_ridge()),
}


def nombres_candidatos() -> list[str]:
    """Devuelve los nombres de los modelos candidatos disponibles."""
    return list(CANDIDATOS)


def crear_pipeline(
    nombre: str, parametros: dict[str, Any] | None = None, escalar: bool = False
) -> Pipeline:
    """Crea un pipeline (imputación + [escalado] + estimador) para un candidato.

    Args:
        nombre: nombre del candidato (ver :data:`CANDIDATOS`).
        parametros: hiperparámetros del estimador.
        escalar: si es ``True``, añade estandarización (necesaria para modelos lineales).

    Returns:
        Un :class:`sklearn.pipeline.Pipeline` listo para entrenar.

    Raises:
        KeyError: si el candidato no existe.
    """
    if nombre not in CANDIDATOS:
        raise KeyError(f"Candidato desconocido: {nombre}")

    estimador, _ = CANDIDATOS[nombre]()

    pasos: list[tuple[str, Any]] = [("imputar", SimpleImputer(strategy="median"))]
    if escalar or nombre == "ridge":
        pasos.append(("escalar", StandardScaler()))
    pasos.append(("modelo", estimador))

    pipeline = Pipeline(pasos)
    # Los hiperparámetros llegan con el prefijo de paso (``modelo__...``), que es
    # el nombre válido para ``Pipeline.set_params``.
    if parametros:
        pipeline.set_params(**parametros)
    return pipeline


def espacio_por_modelo(nombre: str) -> dict[str, tuple[str, Any]]:
    """Devuelve el espacio de búsqueda de un candidato, prefijado para el pipeline.

    Args:
        nombre: nombre del candidato.

    Returns:
        Diccionario ``modelo__parametro -> (tipo, rango)``.
    """
    _, espacio = CANDIDATOS[nombre]()
    return {f"modelo__{clave}": valor for clave, valor in espacio.items()}


# ======================================================================
# Ensamble ponderado
# ======================================================================
def optimizar_pesos_ensamble(
    predicciones: np.ndarray,
    y_real: np.ndarray,
    pesos_iniciales: np.ndarray | None = None,
) -> np.ndarray:
    """Calcula los pesos óptimos de un ensamble lineal sobre predicciones dadas.

    Resuelve ``min_w MAE(w · P, y)`` sujeto a ``sum(w) = 1`` y ``w >= 0``.

    Args:
        predicciones: matriz ``(n_modelos, n_observaciones)`` con las predicciones
            *out-of-fold* de cada candidato.
        y_real: vector de valores observados.
        pesos_iniciales: punto de partida; por defecto uniforme.

    Returns:
        Vector de pesos (longitud = número de modelos) que suma 1.
    """
    predicciones = np.asarray(predicciones, dtype=float)
    y_real = np.asarray(y_real, dtype=float)
    n_modelos = predicciones.shape[0]

    if pesos_iniciales is None or len(pesos_iniciales) != n_modelos:
        pesos_iniciales = np.full(n_modelos, 1.0 / n_modelos)

    def perdida(pesos: np.ndarray) -> float:
        return float(np.mean(np.abs(pesos @ predicciones - y_real)))

    resultado = minimize(
        perdida,
        x0=pesos_iniciales,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_modelos,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"maxiter": 200, "ftol": 1e-8},
    )

    if not resultado.success or np.any(~np.isfinite(resultado.x)):
        logger.warning(
            "No se pudo optimizar el ensamble (%s); se usan pesos uniformes", resultado.message
        )
        return pesos_iniciales

    pesos = np.clip(resultado.x, 0.0, 1.0)
    total = pesos.sum()
    return pesos / total if total > 0 else pesos_iniciales


class EnsamblePonderado(BaseEstimator, RegressorMixin):
    """Ensamble que combina varios modelos con pesos fijos optimizados.

    Args:
        modelos: mapa nombre -> estimador ya entrenado.
        pesos: vector de pesos alineado con ``modelos``.
    """

    def __init__(
        self, modelos: dict[str, Any] | None = None, pesos: list[float] | None = None
    ) -> None:
        self.modelos = modelos or {}
        self.pesos = pesos or []

    def fit(self, X: Any, y: Any) -> EnsamblePonderado:
        """No entrena: el ensamble se construye con modelos ya ajustados.

        Args:
            X: ignorado.
            y: ignorado.

        Returns:
            El propio ensamble.
        """
        if not self.modelos:
            raise ValueError("El ensamble requiere al menos un modelo entrenado")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Combina las predicciones de los modelos con los pesos del ensamble.

        Args:
            X: matriz de variables.

        Returns:
            Predicción combinada (nunca negativa).
        """
        nombres = list(self.modelos)
        pesos = np.asarray(self.pesos, dtype=float)
        if len(pesos) != len(nombres):
            pesos = np.full(len(nombres), 1.0 / len(nombres))

        acumulado = np.zeros(len(X), dtype=float)
        for nombre, peso in zip(nombres, pesos, strict=True):
            acumulado += peso * np.asarray(self.modelos[nombre].predict(X), dtype=float)
        return np.maximum(acumulado, 0.0)

    def nombres(self) -> list[str]:
        """Devuelve los nombres de los modelos del ensamble."""
        return list(self.modelos)


def reentrenar(pipeline: Pipeline, X: Any, y: Any) -> Pipeline:
    """Reentrena una copia del pipeline con todos los datos disponibles.

    Args:
        pipeline: pipeline ya configurado (o entrenado).
        X: matriz de variables.
        y: objetivo.

    Returns:
        El pipeline ajustado.
    """
    modelo = clone(pipeline)
    modelo.fit(X, y)
    return modelo
