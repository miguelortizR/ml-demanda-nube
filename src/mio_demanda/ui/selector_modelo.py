"""Selector del modelo de proyección, reutilizable en cualquier pestaña.

Permite **escoger el modelo por panel**: muestra la comparación de los candidatos
con sus métricas, resalta el que está en uso y permite cambiarlo sin salir de la
pestaña. Se apoya en los servicios (Facade); aquí solo hay presentación y el
encadenado de la acción.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from mio_demanda.ui import componentes as c

#: Métricas que se grafican al comparar los modelos (el WAPE va primero: es el
#: indicador de referencia del negocio). El MAPE no se grafica a propósito.
_METRICAS_GRAFICADAS: list[tuple[str, str]] = [
    ("wape", "WAPE (%) — menor es mejor"),
    ("mae", "MAE (usos) — menor es mejor"),
    ("r2", "R² — mayor es mejor"),
]

#: Columnas de la tabla técnica de comparación (con su formato).
_COLUMNAS_TABLA: list[str] = ["Modelo", "MAE", "RMSE", "WAPE %", "sMAPE %", "MAPE %", "R²"]


def render_comparacion_modelos(info: dict[str, Any], clave: str, enlace: str | None = None) -> None:
    """Muestra la comparación **compacta** de candidatos de un alcance.

    La app solo muestra lo necesario para elegir; el análisis detallado (parámetros,
    curvas y todas las corridas) se consulta en MLflow mediante ``enlace``.

    Args:
        info: descripción del modelo (salida de ``modelo_en_uso``).
        clave: sufijo único de los widgets de este panel.
        enlace: dirección de la corrida en MLflow, si se conoce.
    """
    comparativa = list(info.get("comparativa", []))
    if not comparativa:
        st.info("Este modelo no tiene candidatos registrados para comparar.")
        return

    tabla = c.tabla_comparativa_candidatos(comparativa)
    presentes = [
        columna for columna in ("Modelo", "WAPE %", "MAE", "R²") if columna in tabla.columns
    ]
    st.dataframe(tabla[presentes], width="stretch", hide_index=True)
    st.caption(
        "En verde, el modelo que está proyectando. El **WAPE** es el indicador de "
        "referencia (pondera por volumen)."
    )
    _enlace_mlflow(enlace)


def _enlace_mlflow(enlace: str | None) -> None:
    """Ofrece el enlace a MLflow con la corrida completa del entrenamiento.

    Args:
        enlace: dirección de la corrida en MLflow, si se conoce.
    """
    if not enlace:
        return
    st.markdown(
        f"**[Ver el entrenamiento completo en MLflow]({enlace})** "
        "— parámetros, curvas y todas las corridas."
    )
    st.caption(
        "La interfaz de MLflow se levanta con `make mlflow-ui` (queda en "
        f"{enlace.split('/#/')[0]}). Si no está encendida, el enlace no abrirá."
    )


def render_selector_modelo(
    info: dict[str, Any],
    servicio_entrenamiento,
    alcance: str,
    clave: str,
    etiqueta_expander: str | None = None,
    invalidar=None,
    enlace_mlflow: str | None = None,
) -> None:
    """Bloque completo: comparación compacta + selector + botón de activación.

    Args:
        info: descripción del modelo (salida de ``modelo_en_uso``).
        servicio_entrenamiento: servicio para activar el candidato elegido.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.
        clave: sufijo único de los widgets (una por panel).
        etiqueta_expander: título del desplegable; ``None`` para mostrarlo abierto.
        invalidar: callback opcional que limpia cachés tras activar.
        enlace_mlflow: dirección de la corrida en MLflow, si se conoce.
    """
    titulo = etiqueta_expander or f"Modelos de {alcance}: comparar y elegir"
    if etiqueta_expander:
        contexto = st.expander(titulo, expanded=False)
    else:
        contexto = st.container()
        st.markdown(f"##### {titulo}")

    with contexto:
        _contenido_selector(info, servicio_entrenamiento, alcance, clave, invalidar, enlace_mlflow)


def _contenido_selector(
    info: dict[str, Any],
    servicio_entrenamiento,
    alcance: str,
    clave: str,
    invalidar,
    enlace_mlflow: str | None = None,
) -> None:
    """Dibuja la comparación y el control de activación."""
    render_comparacion_modelos(info, clave, enlace_mlflow)

    comparativa = list(info.get("comparativa", []))
    if not comparativa:
        return

    etiquetas = {
        f"{fila['modelo']} — MAE {float(fila['mae']):,.0f} · WAPE {float(fila['wape']):.2f} %": str(
            fila["modelo"]
        )
        for fila in comparativa
    }
    activo = info.get("campeon")
    opciones = list(etiquetas)
    indice = next((i for i, texto in enumerate(opciones) if etiquetas[texto] == activo), 0)

    col1, col2 = st.columns([3, 1])
    seleccion = col1.selectbox(
        "Candidato que proyectará",
        opciones,
        index=indice,
        key=f"cand_{alcance}_{clave}",
    )
    col2.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    if col2.button("Usar este modelo", key=f"btn_cand_{alcance}_{clave}"):
        elegido = etiquetas[str(seleccion)]
        if elegido == activo:
            st.info(f"«{activo}» ya está proyectando.")
        else:
            with st.spinner(f"Reentrenando «{elegido}» con toda la historia..."):
                try:
                    servicio_entrenamiento.activar_modelo(alcance, elegido)
                except (ValueError, FileNotFoundError) as error:
                    st.error(f"{error}")
                    return
            if invalidar is not None:
                invalidar()
            st.success(f"Listo: {alcance} proyectará con «{elegido}».")
            st.rerun()

    st.caption(
        f"Proyectando con **{activo}**"
        + (
            " (elegido manualmente)."
            if info.get("activado_manualmente")
            else " (campeón automático por MAE)."
        )
        + f" · entrenado {str(info.get('fecha_entrenamiento'))[:10]}"
        + f" · historia {info.get('desde_entrenamiento')} a {info.get('hasta_entrenamiento')}"
    )
