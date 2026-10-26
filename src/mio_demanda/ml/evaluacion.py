"""Evaluación de modelos de regresión con validación cruzada temporal.

Se usa ``TimeSeriesSplit`` (no ``KFold``) porque el problema es temporal: el
conjunto de prueba siempre debe ser **posterior** al de entrenamiento. Se
reportan MAE, RMSE, MAPE y R², y se generan predicciones *out-of-fold* que luego
permiten ajustar los pesos del ensamble sin fuga de datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(slots=True)
class ResultadoCV:
    """Resultado de una validación cruzada temporal.

    Attributes:
        metricas: métricas promedio entre particiones (``mae``, ``rmse``, ``mape``, ``r2``).
        por_particion: métricas detalladas de cada partición.
        oof_predicciones: predicciones fuera de muestra alineadas con ``y``.
        oof_indices: índices de ``y`` cubiertos por las predicciones out-of-fold.
    """

    metricas: dict[str, float]
    por_particion: list[dict[str, float]] = field(default_factory=list)
    oof_predicciones: np.ndarray | None = None
    oof_indices: np.ndarray | None = None


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión.

    Se reportan los errores de **negocio** y los **técnicos** por separado:

    * ``mae`` y ``rmse``: error medio y error cuadrático medio, **en usos** (la unidad
      de la demanda). Son los que se comunican al negocio.
    * ``wape``: error porcentual absoluto **ponderado** (``Σ|error| / Σreal·100``).
      Indicador de referencia del proyecto; con él se calcula la confiabilidad. No se
      dispara con demandas pequeñas porque pondera por volumen.
    * ``smape``: error porcentual **simétrico** (``200·|error| / (|real|+|pred|)``),
      robusto por observación. Complemento técnico.
    * ``mape``: solo **referencia técnica**; se infla con demandas cercanas a cero.

    Args:
        y_real: valores observados.
        y_pred: valores predichos.

    Returns:
        Diccionario con ``mae``, ``rmse``, ``wape``, ``smape`` y ``mape`` (porcentajes
        en escala 0-100, salvo ``smape`` que puede llegar a 200) y ``r2``.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    r2 = float(r2_score(y_real, y_pred)) if len(y_real) > 1 else float("nan")

    error_abs = np.abs(y_real - y_pred)

    # WAPE: error porcentual ponderado por volumen (indicador de referencia).
    total_real = float(np.sum(np.abs(y_real)))
    wape = float(np.sum(error_abs) / total_real * 100) if total_real > 0 else float("nan")

    # sMAPE: error porcentual simétrico (robusto por observación).
    denominador = np.abs(y_real) + np.abs(y_pred)
    valido = denominador > 0
    smape = (
        float(np.mean(200 * error_abs[valido] / denominador[valido]))
        if valido.any()
        else float("nan")
    )

    # MAPE: solo referencia técnica (se infla con demandas cercanas a cero).
    mascara = y_real > 0
    if mascara.any():
        mape = float(mean_absolute_percentage_error(y_real[mascara], y_pred[mascara]) * 100)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "wape": wape, "smape": smape, "mape": mape, "r2": r2}


def validacion_cruzada_temporal(
    estimador: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    recolectar_oof: bool = False,
    max_muestras_oof: int | None = None,
) -> ResultadoCV:
    """Evalúa un estimador con validación cruzada temporal.

    Args:
        estimador: estimador de scikit-learn (o compatible) sin entrenar.
        X: matriz de variables (``numpy``).
        y: vector objetivo.
        n_splits: número de particiones temporales.
        recolectar_oof: si es ``True``, guarda las predicciones fuera de muestra.
        max_muestras_oof: límite de filas para recolectar OOF (las más recientes).

    Returns:
        Un :class:`ResultadoCV` con las métricas y, opcionalmente, las OOF.
    """
    from sklearn.base import clone

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    particiones = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    oof = np.full(len(y), np.nan) if recolectar_oof else None
    por_particion: list[dict[str, float]] = []

    for numero, (indices_train, indices_test) in enumerate(particiones, start=1):
        modelo = clone(estimador)
        modelo.fit(X[indices_train], y[indices_train])
        prediccion = np.maximum(modelo.predict(X[indices_test]), 0.0)

        metricas = calcular_metricas(y[indices_test], prediccion)
        metricas["particion"] = numero
        por_particion.append(metricas)

        if oof is not None:
            oof[indices_test] = prediccion

    promedio = {
        clave: float(np.nanmean([p[clave] for p in por_particion]))
        for clave in ("mae", "rmse", "wape", "smape", "mape", "r2")
    }

    oof_indices = None
    if oof is not None:
        validos = ~np.isnan(oof)
        oof_indices = np.flatnonzero(validos)
        if max_muestras_oof is not None and len(oof_indices) > max_muestras_oof:
            oof_indices = oof_indices[-max_muestras_oof:]

    return ResultadoCV(
        metricas=promedio,
        por_particion=por_particion,
        oof_predicciones=oof,
        oof_indices=oof_indices,
    )
