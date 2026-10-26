"""Pestaña 2 — Proyección general (sistema completo).

Proyecta la demanda diaria de todo el sistema MIO en un horizonte definido por el
usuario, muestra los gráficos de tendencia y permite exportar el resultado a
Excel.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from mio_demanda.ui import componentes as c
from mio_demanda.ui import retrospectiva as retrospectiva_ui
from mio_demanda.ui import segunda_opinion
from mio_demanda.ui.estado import (
    obtener_calendario,
    obtener_proyeccion,
    obtener_reporte,
    obtener_settings,
)

_HORIZONTES = {
    "Próximos 7 días": 7,
    "Próximos 14 días": 14,
    "Próximo mes": 30,
    "Próximo trimestre": 90,
}


def render() -> None:
    """Dibuja la pestaña de proyección general."""
    c.encabezado(
        "Proyección de demanda general",
        "Demanda diaria del sistema MIO. Proyecta un horizonte futuro o evalúa el modelo contra los últimos días reales.",
        variante="exito",
    )
    servicio = obtener_proyeccion()

    if not servicio.modelos_disponibles()["sistema"]:
        c.aviso_sin_modelo("sistema")
        return

    c.panel_modelo_en_uso(servicio.modelo_en_uso(None))

    modo = st.radio(
        "Modo de trabajo",
        ["Proyección a futuro", "Evaluación con datos reales (backtest)"],
        horizontal=True,
        key="sist_modo",
    )

    if modo == "Proyección a futuro":
        _flujo_proyeccion(servicio)
    else:
        _flujo_evaluacion(servicio)


def _flujo_proyeccion(servicio) -> None:
    """Ejecuta la proyección de un horizonte futuro."""
    desde, hasta = _controles_horizonte()
    dias_retro = obtener_settings().retrospectiva_dias
    comparar = st.checkbox(
        f"Comparar con la demanda real (proyección a {dias_retro} días)",
        value=True,
        key="sist_retro",
        help=(
            "Proyecta los últimos días con un modelo que no los vio y los pone al lado de "
            "la demanda real, en la misma gráfica y en el mismo Excel."
        ),
    )

    if st.button("Generar proyección", type="primary", key="btn_proy_sistema"):
        with st.spinner("Calculando la proyección del sistema..."):
            try:
                st.session_state["proy_sistema"] = servicio.proyectar_sistema(desde, hasta)
                st.session_state["proy_sistema_rango"] = (desde, hasta)
                st.session_state["proy_sistema_retro"] = (
                    servicio.retrospectiva() if comparar else None
                )
            except Exception as error:
                st.error(f"No se pudo generar la proyección: {error}")
                return

    resultado = st.session_state.get("proy_sistema")
    if resultado is None:
        st.info("Define el horizonte y pulsa **Generar proyección**.")
        return

    desde, hasta = st.session_state.get("proy_sistema_rango", (desde, hasta))
    _mostrar_resultado(resultado, desde, hasta, st.session_state.get("proy_sistema_retro"))


def _flujo_evaluacion(servicio) -> None:
    """Compara el modelo contra los últimos días con demanda real."""
    st.caption(
        "El modelo se compara contra la demanda real de los últimos días **posteriores** "
        "a su fecha de entrenamiento (comparación fuera de muestra)."
    )
    dias = st.slider("Últimos días reales a comparar", 5, 90, 10, 1, key="sist_dias_eval")

    if st.button("Evaluar modelo", type="primary", key="btn_eval_sistema"):
        with st.spinner("Comparando la proyección con la demanda real..."):
            try:
                st.session_state["eval_sistema"] = servicio.evaluar_sistema(dias)
            except Exception as error:
                st.error(f"No se pudo evaluar: {error}")
                return

    evaluacion = st.session_state.get("eval_sistema")
    if evaluacion is None:
        st.info("Elige cuántos días comparar y pulsa **Evaluar modelo**.")
        return
    _mostrar_evaluacion(evaluacion)


def _mostrar_evaluacion(evaluacion) -> None:
    """Presenta la comparación real vs. proyectado con sus métricas."""
    metricas = evaluacion.metricas
    st.divider()
    st.markdown(f"### Evaluación: {evaluacion.etiqueta} · modelo `{evaluacion.modelo}`")
    st.caption(
        f"Periodo evaluado: {evaluacion.desde} a {evaluacion.hasta} · "
        f"{evaluacion.dias_evaluados} días ({evaluacion.dias_fuera_de_muestra} fuera de muestra)"
    )
    if evaluacion.reentrenado:
        st.info(
            "Tu modelo guardado ya está entrenado hasta el último dato, así que no quedaban días "
            f"para comparar. Para poder evaluarlo se entrenó una **copia temporal con corte al "
            f"{evaluacion.corte_entrenamiento}** (no se guardó): los últimos "
            f"{evaluacion.dias_evaluados} días quedaron fuera del entrenamiento y las anclas se "
            "congelaron en ese corte para que la comparación sea honesta."
        )

    c.fila_kpis(
        [
            {"etiqueta": "WAPE", "valor": f"{metricas.get('wape', 0):.2f} %"},
            {"etiqueta": "MAE (usos)", "valor": c.formato_numero(metricas.get("mae", 0))},
            {"etiqueta": "RMSE (usos)", "valor": c.formato_numero(metricas.get("rmse", 0))},
            {"etiqueta": "R²", "valor": f"{metricas.get('r2', 0):.3f}"},
            {"etiqueta": "Confiabilidad", "valor": f"{evaluacion.confiabilidad:.0%}"},
        ]
    )
    st.caption(
        "El **WAPE** es el indicador de referencia. El MAPE se omite del tablero porque se "
        "infla con demandas cercanas a cero."
    )

    st.plotly_chart(
        c.grafico_comparacion(
            evaluacion.comparacion,
            f"Demanda real vs. proyectada · últimos {evaluacion.dias_evaluados} días",
        ),
        width="stretch",
    )
    c.tabla_detalle(evaluacion.comparacion, "Detalle de la comparación")

    st.divider()
    c.boton_descarga_excel(
        obtener_reporte().libro_evaluacion(evaluacion),
        nombre_archivo=f"evaluacion_general_{evaluacion.desde}_{evaluacion.hasta}",
        etiqueta="Exportar evaluación a Excel",
        clave="excel_eval_general",
    )


def _controles_horizonte() -> tuple[date, date]:
    """Dibuja los controles del horizonte de proyección.

    Returns:
        Tupla ``(desde, hasta)`` seleccionada.
    """
    manana = date.today() + timedelta(days=1)
    col1, col2, col3 = st.columns([1, 1, 2])
    desde = col1.date_input("Desde", value=manana, key="sist_desde")
    hasta = col2.date_input("Hasta", value=manana + timedelta(days=13), key="sist_hasta")

    preset = col3.selectbox(
        "Atajos de horizonte", ["Personalizado", *_HORIZONTES], key="sist_preset"
    )
    if preset != "Personalizado":
        dias = _HORIZONTES[preset]
        desde, hasta = manana, manana + timedelta(days=dias - 1)
        st.caption(f"Horizonte seleccionado: {desde} a {hasta} ({dias} días)")
    return desde, hasta


def _mostrar_resultado(resultado, desde: date, hasta: date, retrospectiva=None) -> None:
    """Presenta el resultado en una tabla compacta, la tendencia y el detalle por hora."""
    resumen = resultado.resumen
    st.divider()
    info = obtener_proyeccion().modelo_en_uso(None)
    st.markdown(f"### {resumen.etiqueta} · {resumen.dias} días")

    # Ficha del modelo: lo que hay que pegar en un informe para justificar la
    # proyección (con qué modelo se hizo, con qué historia y con qué error).
    metricas = info["metricas"]
    desde_ent = info.get("desde_entrenamiento")
    hasta_ent = info.get("hasta_entrenamiento")
    dias_historia = (hasta_ent - desde_ent).days + 1 if desde_ent and hasta_ent else 0
    composicion = (
        " + ".join(f"{nombre} {peso:.2f}" for nombre, peso in info["componentes"].items())
        if info.get("es_ensamble")
        else ""
    )
    filas_ficha = [
        {
            "Métrica": "Modelo",
            "Valor": str(info["campeon"]) + (f"  ({composicion})" if composicion else ""),
        },
        {
            "Métrica": "Entrenado",
            "Valor": str(info.get("fecha_entrenamiento"))[:10],
        },
        {
            "Métrica": "Periodo de entrenamiento",
            "Valor": (
                f"{desde_ent} a {hasta_ent}  ({dias_historia:,} días)".replace(",", ".")
                if desde_ent and hasta_ent
                else "—"
            ),
        },
        {"Métrica": "WAPE", "Valor": f"{float(metricas.get('wape', 0)):.2f} %"},
        {"Métrica": "MAE", "Valor": f"{c.formato_numero(metricas.get('mae', 0))} usos/día"},
        {"Métrica": "RMSE", "Valor": f"{c.formato_numero(metricas.get('rmse', 0))} usos"},
        {"Métrica": "R²", "Valor": f"{float(metricas.get('r2', 0)):.3f}"},
        {
            "Métrica": "Error esperado por día",
            "Valor": f"± {c.formato_numero(resumen.margen_error)} usos",
        },
        {"Métrica": "Confiabilidad", "Valor": f"{info['confiabilidad']:.0%}"},
    ]
    st.dataframe(pd.DataFrame(filas_ficha), width="stretch", hide_index=True)

    umbrales = obtener_proyeccion().umbrales_coherencia()
    c.aviso_coherencia(
        resumen.usos_diarios,
        obtener_proyeccion().referencia_reciente(),
        "de la demanda del sistema",
        minimo=umbrales[0],
        maximo=umbrales[1],
    )

    if retrospectiva is not None:
        retrospectiva_ui.render(retrospectiva)

    detalle = _anexar_eventos(resultado.detalle, desde, hasta)
    if retrospectiva is not None:
        detalle = retrospectiva_ui.anexar(detalle, retrospectiva)
    segunda = segunda_opinion.render(desde, hasta, resumen.usos_diarios)
    if segunda is not None:
        detalle = _anexar_segunda_opinion(detalle, segunda.detalle)

    st.plotly_chart(
        c.grafico_tendencia(
            detalle,
            "fecha",
            "usos_predicho",
            "Tendencia diaria",
            columna_etiqueta="etiqueta",
            columna_inferior="usos_p10",
            columna_superior="usos_p90",
            columna_real="demanda_real" if "retro_proyectado" in detalle.columns else None,
            columna_retrospectiva="retro_proyectado"
            if "retro_proyectado" in detalle.columns
            else None,
            nombre_retrospectiva=f"Proyectado a {obtener_settings().retrospectiva_dias} días",
            columna_comparacion="usos_jev" if "usos_jev" in detalle.columns else None,
            nombre_comparacion="Segunda opinión (Jev)",
        ),
        width="stretch",
    )

    st.markdown("#### Un día · por hora")
    _perfil_horario_de_un_dia(resultado.detalle_horario)

    st.divider()
    c.boton_descarga_excel(
        obtener_reporte().libro_general(
            resultado,
            segunda_opinion=segunda.detalle if segunda else None,
            retrospectiva=retrospectiva,
        ),
        nombre_archivo=f"proyeccion_demanda_general_{desde}_{hasta}",
        etiqueta="Exportar proyección a Excel",
        clave="excel_general",
    )


def _perfil_horario_de_un_dia(horario: pd.DataFrame) -> None:
    """Muestra la demanda por hora de un día proyectado concreto.

    El modelo predice el total del día; las horas son ese total repartido con el perfil
    histórico del tipo de día.

    Args:
        horario: detalle horario con ``fecha``, ``hora`` y ``usos_predicho``.
    """
    if horario is None or horario.empty:
        st.info("Esta proyección no tiene detalle horario.")
        return

    fechas = sorted(horario["fecha"].dt.date.unique())
    elegida = st.selectbox(
        "Día proyectado",
        fechas,
        index=0,
        key="sist_dia_horario",
        format_func=lambda fecha: fecha.strftime("%Y-%m-%d (%A)"),
    )
    del_dia = horario[horario["fecha"].dt.date == elegida]
    st.plotly_chart(
        c.grafico_perfil_horario(del_dia, f"Usos por hora · {elegida}"),
        width="stretch",
    )


def _anexar_segunda_opinion(detalle: pd.DataFrame, jev: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas de la segunda opinión al detalle diario.

    Args:
        detalle: detalle diario del modelo entrenado.
        jev: detalle diario de la segunda opinión.

    Returns:
        El detalle con ``usos_jev``, ``usos_jev_p10``, ``usos_jev_p90`` y ``confianza``.
    """
    columnas = [
        columna
        for columna in ("fecha", "usos_jev", "usos_jev_p10", "usos_jev_p90", "confianza")
        if columna in jev.columns
    ]
    return detalle.merge(jev[columnas], on="fecha", how="left")


def _anexar_eventos(detalle: pd.DataFrame, desde: date, hasta: date) -> pd.DataFrame:
    """Añade la etiqueta de evento de calendario a cada fecha.

    Args:
        detalle: detalle diario proyectado.
        desde: fecha inicial del horizonte.
        hasta: fecha final del horizonte.

    Returns:
        El detalle con la columna ``etiqueta``.
    """
    etiquetas = obtener_calendario().etiquetas_barras(desde, hasta)
    combinado = detalle.merge(etiquetas[["fecha", "etiqueta"]], on="fecha", how="left")
    combinado["etiqueta"] = combinado["etiqueta"].fillna("")
    return combinado
