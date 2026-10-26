"""Retrospectiva del vigilante: lo que proyectamos hace N días contra lo real.

Se usa igual en las dos pestañas de proyección:

* :func:`anexar` mete en el detalle diario la demanda real ya observada y la
  proyección que el modelo hizo para esas mismas fechas, de modo que la gráfica
  quede continua: pasado (real vs. proyectado entonces) y futuro (proyección actual).
* :func:`render` muestra el veredicto: ¿el modelo sigue sirviendo?
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.domain.vigilancia import EstadoModelo, evaluar_desajuste, sesgo_relativo
from mio_demanda.ui import componentes as c
from mio_demanda.ui.estado import obtener_proyeccion, obtener_settings

_COLUMNAS = [
    "fecha",
    "dia_tipo_nom",
    "usos_predicho",
    "retro_proyectado",
    "demanda_real",
    "es_proyectado",
    "etiqueta",
]


def anexar(detalle: pd.DataFrame, retrospectiva) -> pd.DataFrame:
    """Antepone la retrospectiva (real + proyectado de entonces) al detalle futuro.

    Args:
        detalle: detalle de la proyección futura.
        retrospectiva: :class:`EvaluacionProyeccion` de los últimos días.

    Returns:
        La unión de las dos partes, ordenada por fecha.
    """
    comparado = retrospectiva.comparacion.rename(columns={"Fecha": "fecha"})
    atras = pd.DataFrame(
        {
            "fecha": pd.to_datetime(comparado["fecha"]),
            "dia_tipo_nom": comparado.get("Dia.tipo"),
            "usos_predicho": float("nan"),
            "retro_proyectado": pd.to_numeric(comparado["Demanda_Predicha"], errors="coerce"),
            "demanda_real": pd.to_numeric(comparado["Demanda_Real"], errors="coerce"),
            "es_proyectado": False,
            "etiqueta": "",
        }
    )
    futuro = detalle.copy()
    futuro["retro_proyectado"] = float("nan")
    if "etiqueta" not in futuro.columns:
        futuro["etiqueta"] = ""
    if "dia_tipo_nom" not in futuro.columns:
        futuro["dia_tipo_nom"] = None
    unido = pd.concat([atras[_COLUMNAS], futuro[_COLUMNAS]], ignore_index=True)
    return unido.sort_values("fecha").reset_index(drop=True)


def render(retrospectiva, tipo: TipoEntidad | None = None) -> None:
    """Muestra el veredicto del vigilante sobre la proyección hecha hace N días.

    Args:
        retrospectiva: :class:`EvaluacionProyeccion` con la comparación.
        tipo: alcance del modelo (``None`` para el sistema).
    """
    settings = obtener_settings()
    info = obtener_proyeccion().modelo_en_uso(tipo)
    validacion = float(info["metricas"].get("wape", 0.0))
    comparado = retrospectiva.comparacion

    st.caption(
        f"Ventana fija: los últimos {retrospectiva.dias_evaluados} días con demanda real "
        f"({retrospectiva.desde} a {retrospectiva.hasta}). No cambia con el horizonte elegido."
    )

    estado, motivo = evaluar_desajuste(
        wape_reciente=float(retrospectiva.metricas.get("wape", 0.0)),
        wape_validacion=validacion,
        sesgo=sesgo_relativo(
            comparado["Demanda_Predicha"].astype(float),
            comparado["Demanda_Real"].astype(float),
        ),
        dias_evaluados=retrospectiva.dias_evaluados,
        min_dias=settings.vigilancia_min_dias,
        factor_alerta=settings.vigilancia_factor_alerta,
        sesgo_alerta=settings.vigilancia_sesgo_alerta,
    )

    st.markdown("#### ¿Sigue sirviendo el modelo?")
    c.fila_kpis(
        [
            {"etiqueta": "Estado", "valor": estado.etiqueta},
            {
                "etiqueta": "WAPE reciente",
                "valor": f"{retrospectiva.metricas.get('wape', 0):.2f} %",
            },
            {"etiqueta": "WAPE de validación", "valor": f"{validacion:.2f} %"},
            {"etiqueta": "Días comparados", "valor": str(retrospectiva.dias_evaluados)},
        ]
    )
    if estado is EstadoModelo.DESAJUSTADO:
        st.error(motivo)
    elif estado is EstadoModelo.SIN_DATOS:
        st.info(motivo)
    else:
        st.success(motivo)
    st.caption(
        f"Proyección hecha con el modelo entrenado hasta el "
        f"{retrospectiva.corte_entrenamiento} y comparada con la demanda real del "
        f"{retrospectiva.desde} al {retrospectiva.hasta}."
        + (
            " Para poder comparar se entrenó una copia temporal con ese corte."
            if retrospectiva.reentrenado
            else ""
        )
    )
