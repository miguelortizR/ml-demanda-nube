"""Pestaña 3 — Proyección por estaciones o rutas.

Permite elegir el tipo de entidad (estación o ruta), la entidad concreta y el
horizonte de tiempo, y muestra la tendencia diaria, el perfil horario, el resumen
de demanda, oferta y rentabilidad, con exportación a Excel.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.ui import componentes as c
from mio_demanda.ui import retrospectiva as retrospectiva_ui
from mio_demanda.ui import segunda_opinion
from mio_demanda.ui.estado import (
    entidades_activas,
    invalidar_cache,
    limpiar_entidades,
    obtener_calendario,
    obtener_proyeccion,
    obtener_reporte,
    obtener_settings,
)

_TIPOS = {"Estaciones": TipoEntidad.ESTACION, "Rutas": TipoEntidad.RUTA}

#: Opción del filtro que muestra la suma de estaciones y rutas.
_AMBAS = "Ambas (agregado)"

#: Claves de estado donde se guarda el último cálculo **con** su identificador.
_CLAVE_PROYECCION = "proy_entidad"
_CLAVE_EVALUACION = "eval_entidad"
_CLAVE_CONJUNTA = "proy_conjunta"


def render() -> None:
    """Dibuja la pestaña de proyección por estaciones, rutas o todas."""
    servicio = obtener_proyeccion()
    disponibles = servicio.modelos_disponibles()

    col_alcance, col_detalle, col_accion = st.columns([2.0, 1.7, 2.0])
    alcance = col_alcance.radio(
        "Entidad", ["Estaciones", "Rutas", "Todas"], horizontal=True, key="ent_tipo"
    )
    modo = col_accion.radio(
        "Acción",
        ["Proyectar", "Evaluación con datos reales"],
        horizontal=True,
        key="ent_modo",
    )

    if alcance == "Todas":
        if not (disponibles["estaciones"] or disponibles["rutas"]):
            c.aviso_sin_modelo("estaciones")
            return
        col_detalle.caption("Estaciones + rutas")
        if modo != "Proyectar":
            st.info(
                "La evaluación con datos reales es por entidad: elige **Estaciones** o "
                "**Rutas** y después una estación o ruta concreta."
            )
            return
        _flujo_todas(servicio)
        return

    tipo = _TIPOS[alcance]
    nombre_alcance = "estaciones" if tipo is TipoEntidad.ESTACION else "rutas"
    if not disponibles[nombre_alcance]:
        c.aviso_sin_modelo(nombre_alcance)
        return

    entidades = entidades_activas(tipo.value)
    if not entidades:
        st.warning(
            "No hay entidades con datos recientes. Puede ser que la base de datos se esté "
            "actualizando o que el proceso de la app tenga datos viejos en memoria."
        )
        if st.button("Reintentar", key="btn_reintentar_entidades"):
            limpiar_entidades()
            invalidar_cache()
            st.rerun()
        return

    todas = "Todas las estaciones" if tipo is TipoEntidad.ESTACION else "Todas las rutas"
    entidad = col_detalle.selectbox(
        "Estación o ruta",
        [todas, *entidades],
        index=0,
        key=f"ent_entidad_{tipo.value}",
        help="«Todas» proyecta el grupo completo y permite filtrarlo después.",
    )

    if modo == "Proyectar":
        _flujo_proyeccion(servicio, entidad, tipo)
        return

    if entidad == todas:
        st.info(
            "La evaluación con datos reales es por entidad: elige una estación o ruta "
            "concreta para compararla con su demanda observada."
        )
        return
    _flujo_evaluacion(servicio, entidad, tipo)


def _identificador(tipo: TipoEntidad, entidad: str) -> str:
    """Identificador del resultado guardado (tipo de entidad + entidad).

    Evita que se muestre la proyección de una entidad cuando el usuario ya eligió
    otra: el resultado se guarda con su identificador y solo se pinta si coincide.

    Args:
        tipo: tipo de entidad.
        entidad: identificador de la estación o ruta.

    Returns:
        Cadena única, por ejemplo ``ruta::T31``.
    """
    return f"{tipo.value}::{entidad}"


def _flujo_proyeccion(servicio, entidad: str, tipo: TipoEntidad) -> None:
    """Proyecta un horizonte futuro para la entidad (o para todas).

    El mismo flujo sirve para una estación/ruta concreta y para el agregado de todas: el
    selector es el único filtro.
    """
    desde, hasta, pulsado = _controles_horizonte("btn_proy_entidad")
    identificador = _identificador(tipo, entidad)
    es_grupo = entidad.startswith("Todas")

    if pulsado:
        with st.spinner(f"Proyectando {entidad}..."):
            try:
                resultado = (
                    servicio.proyectar_grupo(tipo, desde, hasta)
                    if es_grupo
                    else servicio.proyectar_entidad(entidad, tipo, desde, hasta)
                )
                st.session_state[_CLAVE_PROYECCION] = {
                    "identificador": identificador,
                    "resultado": resultado,
                    "desde": desde,
                    "hasta": hasta,
                }
            except Exception as error:
                st.session_state.pop(_CLAVE_PROYECCION, None)
                st.error(f"No se pudo generar la proyección: {error}")
                return

    guardado = st.session_state.get(_CLAVE_PROYECCION)
    if guardado is None:
        st.caption("Define el horizonte y pulsa **Generar proyección**.")
        return
    if guardado["identificador"] != identificador:
        anterior = str(guardado["identificador"]).split("::")[-1]
        st.info(
            f"El último resultado calculado es de **{anterior}**. Pulsa "
            f"**Generar proyección** para calcular el de **{entidad}**."
        )
        return

    _mostrar_resultado(
        guardado["resultado"],
        guardado["desde"],
        guardado["hasta"],
        tipo,
        entidad,
    )


def _flujo_evaluacion(servicio, entidad: str, tipo: TipoEntidad) -> None:
    """Compara el modelo de la entidad contra sus últimos días con demanda real."""
    st.caption(
        "El modelo se compara contra la demanda real de los últimos días **posteriores** "
        "a su fecha de entrenamiento (comparación fuera de muestra)."
    )
    dias = st.slider("Últimos días reales a comparar", 5, 90, 10, 1, key="ent_dias_eval")
    identificador = _identificador(tipo, entidad)

    if st.button("Evaluar modelo", type="primary", key="btn_eval_entidad"):
        with st.spinner(f"Comparando la proyección de {entidad} con la demanda real..."):
            try:
                st.session_state[_CLAVE_EVALUACION] = {
                    "identificador": identificador,
                    "evaluacion": servicio.evaluar_entidad(entidad, tipo, dias),
                }
            except Exception as error:
                st.session_state.pop(_CLAVE_EVALUACION, None)
                st.error(f"No se pudo evaluar: {error}")
                return

    guardado = st.session_state.get(_CLAVE_EVALUACION)
    if guardado is None:
        st.info("Elige cuántos días comparar y pulsa **Evaluar modelo**.")
        return
    if guardado["identificador"] != identificador:
        anterior = str(guardado["identificador"]).split("::")[-1]
        st.info(
            f"La última evaluación es de **{anterior}**. Pulsa **Evaluar modelo** para "
            f"comparar **{entidad}** con su demanda real."
        )
        return

    _mostrar_evaluacion(guardado["evaluacion"])


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
        nombre_archivo=f"evaluacion_{evaluacion.etiqueta}_{evaluacion.desde}_{evaluacion.hasta}".replace(
            " ", "_"
        ),
        etiqueta="Exportar evaluación a Excel",
        clave="excel_eval_entidad",
    )


def _controles_horizonte(clave_boton: str) -> tuple[date, date, bool]:
    """Dibuja fechas y botón de generar en una misma línea.

    Args:
        clave_boton: clave del botón, distinta por alcance.

    Returns:
        Tupla ``(desde, hasta, pulsado)``.
    """
    manana = date.today() + timedelta(days=1)
    col_desde, col_hasta, col_boton = st.columns([1, 1, 0.95], vertical_alignment="bottom")
    desde = col_desde.date_input("Desde", value=manana, key="ent_desde")
    hasta = col_hasta.date_input("Hasta", value=manana + timedelta(days=6), key="ent_hasta")
    pulsado = col_boton.button("Generar proyección", type="primary", key=clave_boton)
    return desde, hasta, pulsado


def _mostrar_resultado(
    resultado, desde: date, hasta: date, tipo: TipoEntidad, entidad: str
) -> None:
    """Presenta ficha, coherencia, tendencia, hora y exportación (una entidad o todas)."""
    resumen = resultado.resumen
    st.divider()
    info = obtener_proyeccion().modelo_en_uso(tipo)
    st.markdown(f"### {resultado.etiqueta} · {resumen.dias} días")
    with st.expander("Datos técnicos del modelo"):
        c.ficha_modelo(info, resumen.margen_error)

    es_grupo = entidad.startswith("Todas")
    umbrales = obtener_proyeccion().umbrales_coherencia()
    referencia = (
        obtener_proyeccion().referencia_reciente_grupo(tipo)
        if es_grupo
        else obtener_proyeccion().referencia_reciente_entidad(entidad)
    )
    c.aviso_coherencia(
        resumen.usos_diarios,
        referencia,
        f"del promedio real reciente de {entidad}",
        minimo=umbrales[0],
        maximo=umbrales[1],
    )

    diaria = _anexar_eventos(resultado.serie_diaria, desde, hasta)
    segunda = segunda_opinion.render(desde, hasta, resumen.usos_diarios, entidad=entidad, tipo=tipo)
    if segunda is not None:
        diaria = _anexar_segunda_opinion(diaria, segunda.detalle)

    st.plotly_chart(
        c.grafico_tendencia(
            diaria,
            "fecha",
            "usos_predicho",
            f"Tendencia diaria · {resultado.etiqueta}",
            columna_etiqueta="etiqueta",
            columna_inferior="usos_p10",
            columna_superior="usos_p90",
            columna_real="demanda_real" if "retro_proyectado" in diaria.columns else None,
            columna_retrospectiva="retro_proyectado"
            if "retro_proyectado" in diaria.columns
            else None,
            nombre_retrospectiva=f"Proyectado a {obtener_settings().retrospectiva_dias} días",
            columna_comparacion="usos_jev" if "usos_jev" in diaria.columns else None,
            nombre_comparacion="Segunda opinión (Jev)",
        ),
        width="stretch",
    )

    if not resultado.detalle_por_entidad.empty:
        _bloque_grupo(resultado, tipo)
    else:
        _revision_con_demanda_real(tipo, entidad, es_grupo)
        st.markdown("#### Un día · por hora")
        c.selector_dia_y_horario(resultado.detalle_horario, clave=f"entidad_{tipo.value}_{entidad}")

    st.divider()
    horario_entidades = None
    if not resultado.detalle_por_entidad.empty:
        horario_entidades = obtener_proyeccion().horario_por_entidad(
            resultado.detalle_por_entidad, tipo
        )
    c.boton_descarga_excel(
        obtener_reporte().libro_entidad(
            resultado,
            segunda_opinion=segunda.detalle if segunda else None,
            horario_entidades=horario_entidades,
        ),
        nombre_archivo=f"proyeccion_{resultado.etiqueta}_{desde}_{hasta}".replace(" ", "_"),
        etiqueta="Exportar proyección a Excel",
        clave="excel_entidad",
    )


def _bloque_grupo(resultado, tipo: TipoEntidad) -> None:
    """Filtro por entidad sobre una proyección de grupo ya calculada.

    Args:
        resultado: proyección del grupo (trae ``detalle_por_entidad``).
        tipo: tipo de entidad.
    """
    st.markdown("#### Filtro por entidad")
    st.caption(
        "El período completo ya está proyectado: al elegir una entidad no se vuelve a "
        "calcular nada, solo se filtra."
    )
    opciones = sorted(resultado.detalle_por_entidad["entidad"].unique().tolist())
    elegida = st.selectbox(
        "Ver el detalle de",
        ["Todas (agregado)", *opciones],
        key=f"filtro_entidad_{tipo.value}",
    )

    if elegida == "Todas (agregado)":
        serie = resultado.serie_diaria
        horario = resultado.detalle_horario
        titulo = f"Suma de todas · {resultado.etiqueta}"
    else:
        serie = _serie_de_la_entidad(resultado, elegida)
        horario = obtener_proyeccion().detalle_horario_entidad(tipo, elegida, serie)
        titulo = f"{elegida} (dentro del grupo)"

    st.plotly_chart(
        c.grafico_tendencia(
            serie,
            "fecha",
            "usos_predicho",
            titulo,
            columna_inferior="usos_p10",
            columna_superior="usos_p90",
        ),
        width="stretch",
    )
    st.markdown("#### Un día · por hora")
    c.selector_dia_y_horario(horario, clave=f"grupo_{tipo.value}_{elegida}")


def _serie_de_la_entidad(resultado, entidad: str) -> pd.DataFrame:
    """Extrae la serie diaria de una entidad a partir del detalle del grupo.

    Args:
        resultado: proyección del grupo (con ``detalle_por_entidad``).
        entidad: identificador de la entidad.

    Returns:
        Serie diaria con ``fecha``, ``dia_tipo_num`` y ``usos_predicho``.
    """
    detalle = resultado.detalle_por_entidad
    columnas = ["fecha", "dia_tipo_num", "dia_tipo_nom", "usos_predicho"]
    if "es_proyectado" in detalle.columns:
        columnas.append("es_proyectado")
    serie = detalle[detalle["entidad"] == entidad][columnas].sort_values("fecha")
    serie = serie.reset_index(drop=True)
    if "es_proyectado" not in serie.columns:
        serie["es_proyectado"] = True
    margen = resultado.resumen.margen_error
    serie["usos_p10"] = (serie["usos_predicho"] - margen).clip(lower=0)
    serie["usos_p90"] = serie["usos_predicho"] + margen
    return serie


def _revision_con_demanda_real(tipo: TipoEntidad, entidad: str, es_grupo: bool) -> None:
    """Ofrece revisar si el modelo sigue sirviendo, solo cuando el usuario lo pide.

    La revisión reentrena una copia temporal del modelo del alcance (~1 min para rutas y
    estaciones la primera vez; después queda en caché), por eso no se hace sola.

    Args:
        tipo: tipo de entidad.
        entidad: entidad o etiqueta del grupo.
        es_grupo: si la vista es el agregado de todas.
    """
    if es_grupo:
        return
    clave = f"retro_{tipo.value}_{entidad}"
    guardada = st.session_state.get(clave)
    st.markdown("#### ¿Sigue sirviendo el modelo?")
    if guardada is None:
        if st.button("Revisar con la demanda real", key=f"btn_retro_{tipo.value}"):
            with st.spinner(
                "Entrenando una copia temporal del modelo. La primera vez tarda cerca de "
                "un minuto; después queda en caché."
            ):
                try:
                    st.session_state[clave] = obtener_proyeccion().retrospectiva_entidad(
                        entidad, tipo
                    )
                except (ValueError, FileNotFoundError) as error:
                    st.error(f"No se pudo revisar: {error}")
                    return
            st.rerun()
        st.caption(
            "Compara lo proyectado con la demanda real de los últimos días. La primera "
            "revisión reentrena una copia temporal con un corte anterior; después es "
            "instantánea."
        )
        return
    retrospectiva_ui.render(guardada, tipo)


def _anexar_segunda_opinion(diaria: pd.DataFrame, jev: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas de la segunda opinión a la serie diaria.

    Args:
        diaria: serie diaria del modelo entrenado.
        jev: detalle diario de la segunda opinión.

    Returns:
        La serie con ``usos_jev``, ``usos_jev_p10``, ``usos_jev_p90`` y ``confianza``.
    """
    columnas = [
        columna
        for columna in ("fecha", "usos_jev", "usos_jev_p10", "usos_jev_p90", "confianza")
        if columna in jev.columns
    ]
    return diaria.merge(jev[columnas], on="fecha", how="left")


def _anexar_eventos(datos: pd.DataFrame, desde: date, hasta: date) -> pd.DataFrame:
    """Añade la etiqueta de evento de calendario a cada fecha.

    Args:
        datos: serie diaria.
        desde: fecha inicial del horizonte.
        hasta: fecha final del horizonte.

    Returns:
        El DataFrame con la columna ``etiqueta``.
    """
    etiquetas = obtener_calendario().etiquetas_barras(desde, hasta)
    combinado = datos.merge(etiquetas[["fecha", "etiqueta"]], on="fecha", how="left")
    combinado["etiqueta"] = combinado["etiqueta"].fillna("")
    return combinado


# ======================================================================
# Vista «Todas» (estaciones + rutas)
# ======================================================================
def _flujo_todas(servicio) -> None:
    """Proyecta estaciones y rutas juntas y deja filtrar las gráficas por entidad."""
    desde, hasta, pulsado = _controles_horizonte("btn_proy_todas")
    if pulsado:
        with st.spinner("Proyectando estaciones y rutas..."):
            try:
                st.session_state[_CLAVE_CONJUNTA] = {
                    "conjunto": servicio.proyectar_conjunta(desde, hasta),
                    "desde": desde,
                    "hasta": hasta,
                }
            except Exception as error:
                st.session_state.pop(_CLAVE_CONJUNTA, None)
                st.error(f"No se pudo generar la proyección: {error}")
                return

    guardado = st.session_state.get(_CLAVE_CONJUNTA)
    if guardado is None:
        st.caption("Define el horizonte y pulsa **Generar proyección**.")
        return
    _mostrar_conjunta(guardado["conjunto"], guardado["desde"], guardado["hasta"])


def _opciones_filtro(grupos: dict) -> list[str]:
    """Opciones del filtro de la vista «Todas».

    Args:
        grupos: mapa ``tipo -> proyección`` con los grupos disponibles.

    Returns:
        Lista con «Ambas (agregado)» y cada estación y ruta disponible.
    """
    opciones = [_AMBAS]
    for tipo, proyeccion in grupos.items():
        prefijo = "Estación" if tipo is TipoEntidad.ESTACION else "Ruta"
        for entidad in sorted(proyeccion.detalle_por_entidad["entidad"].unique().tolist()):
            opciones.append(f"{prefijo} · {entidad}")
    return opciones


def _partir_opcion(opcion: str) -> tuple[TipoEntidad, str]:
    """Separa una opción del filtro en tipo de entidad y nombre.

    Args:
        opcion: cadena con formato ``"Estación · A01A"``.

    Returns:
        Tupla ``(tipo, entidad)``.
    """
    prefijo, entidad = opcion.split(" · ", maxsplit=1)
    tipo = TipoEntidad.ESTACION if prefijo == "Estación" else TipoEntidad.RUTA
    return tipo, entidad


def _tabla_grupos(grupos: dict) -> pd.DataFrame:
    """Tabla compacta con los usos diarios de cada grupo y su suma.

    Args:
        grupos: mapa ``tipo -> proyección``.

    Returns:
        DataFrame con una fila por grupo y el total.
    """
    filas = [
        {
            "Grupo": "Estaciones" if tipo is TipoEntidad.ESTACION else "Rutas",
            "Usos/día": c.formato_numero(proyeccion.resumen.usos_diarios),
            "Días": proyeccion.resumen.dias,
            "Confiabilidad": f"{proyeccion.confiabilidad:.0%}",
        }
        for tipo, proyeccion in grupos.items()
    ]
    if len(filas) > 1:
        suma = sum(proyeccion.resumen.usos_diarios for proyeccion in grupos.values())
        filas.append(
            {
                "Grupo": "Estaciones + rutas",
                "Usos/día": c.formato_numero(suma),
                "Días": next(iter(grupos.values())).resumen.dias,
                "Confiabilidad": "",
            }
        )
    return pd.DataFrame(filas)


def _mostrar_conjunta(conjunto, desde: date, hasta: date) -> None:
    """Muestra la suma de estaciones y rutas con filtro por entidad.

    Args:
        conjunto: proyección conjunta de ambos grupos.
        desde: primera fecha proyectada.
        hasta: última fecha proyectada.
    """
    servicio = obtener_proyeccion()
    grupos = conjunto.grupos()
    st.divider()
    st.markdown("### Estaciones + rutas")
    st.dataframe(_tabla_grupos(grupos), width="stretch", hide_index=True)
    st.caption(
        "La suma de los dos grupos describe la demanda del sistema: una estación y una "
        "ruta son dos formas de mirar el mismo viaje."
    )

    elegida = st.selectbox(
        "Filtrar las gráficas por estación o ruta",
        _opciones_filtro(grupos),
        key="conj_filtro",
    )
    if elegida == _AMBAS:
        serie = servicio.agregado_conjunto(conjunto)
        horario = servicio.horario_conjunto(conjunto)
        titulo = "Suma de estaciones y rutas · tendencia diaria"
        clave_hora = "conj_ambas"
        if len(grupos) == 2:
            referencia = servicio.referencia_reciente()
            descripcion = "del promedio real del sistema"
        else:
            referencia = servicio.referencia_reciente_grupo(next(iter(grupos)))
            descripcion = "del promedio real reciente del grupo"
    else:
        tipo, entidad = _partir_opcion(elegida)
        proyeccion = grupos[tipo]
        serie = _serie_de_la_entidad(proyeccion, entidad)
        horario = servicio.detalle_horario_entidad(tipo, entidad, serie)
        titulo = f"{entidad} · tendencia diaria"
        clave_hora = f"conj_{tipo.value}_{entidad}"
        referencia = servicio.referencia_reciente_entidad(entidad)
        descripcion = f"del promedio real reciente de {entidad}"

    usos_diarios = float(serie["usos_predicho"].mean())
    umbrales = servicio.umbrales_coherencia()
    c.aviso_coherencia(
        usos_diarios,
        referencia,
        descripcion,
        minimo=umbrales[0],
        maximo=umbrales[1],
    )

    st.plotly_chart(
        c.grafico_tendencia(
            serie,
            "fecha",
            "usos_predicho",
            titulo,
            columna_inferior="usos_p10",
            columna_superior="usos_p90",
        ),
        width="stretch",
    )

    st.markdown("#### Un día · por hora")
    c.selector_dia_y_horario(horario, clave=clave_hora)

    with st.expander("Datos técnicos de los modelos"):
        for tipo, proyeccion in grupos.items():
            nombre = "Estaciones" if tipo is TipoEntidad.ESTACION else "Rutas"
            st.markdown(f"**Modelo de {nombre.lower()}**")
            c.ficha_modelo(servicio.modelo_en_uso(tipo), proyeccion.resumen.margen_error)
            st.caption(
                f"{proyeccion.detalle_por_entidad['entidad'].nunique()} entidades · "
                f"{proyeccion.resumen.dias} días proyectados"
            )

    st.divider()
    horario_entidades = pd.concat(
        [
            servicio.horario_por_entidad(proyeccion.detalle_por_entidad, tipo).assign(
                grupo="Estaciones" if tipo is TipoEntidad.ESTACION else "Rutas"
            )
            for tipo, proyeccion in grupos.items()
        ],
        ignore_index=True,
    )
    c.boton_descarga_excel(
        obtener_reporte().libro_conjunta(
            conjunto, horario_entidades=horario_entidades, serie=serie
        ),
        nombre_archivo=f"proyeccion_todas_{desde}_{hasta}",
        etiqueta="Exportar proyección a Excel",
        clave="excel_conjunta",
    )
