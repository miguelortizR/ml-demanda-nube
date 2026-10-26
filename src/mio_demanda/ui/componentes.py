"""Encabezados, KPIs, gráficos y exportación reutilizables.

Todas las funciones de este módulo reciben datos ya calculados y solo los
**presentan**: formato, color, disposición y exportación. Ninguna consulta ni
calcula reglas de negocio.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mio_demanda.config import get_settings

# Paleta institucional del proyecto.
COLOR_PRIMARIO = "#2563eb"
COLOR_EXITO = "#16a34a"
COLOR_ALERTA = "#d97706"
COLOR_PELIGRO = "#dc2626"
COLOR_COMPARACION = "#7c3aed"
COLOR_REAL = "#0f172a"
COLORES_TIPO_DIA = {"Hábil": "#22c55e", "Sábado": "#f59e0b", "Festivo": "#ef4444"}

_TIPO_EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_CARACTERES_INVALIDOS_HOJA = re.compile(r"[\[\]:*?/\\]")
_LARGO_MAX_HOJA = 31


def inyectar_estilos() -> None:
    """Inyecta el CSS global de la aplicación."""
    st.markdown(
        """
        <style>
            /* Encabezado de bloque: plano, sin tarjeta ni sombra. La jerarquía la
               hacen el tamaño y una regla fina, no un fondo de color. */
            .bloque-titulo {
                padding: 0 0 10px 0; margin: 2px 0 18px 0;
                border-bottom: 1px solid #e2e8f0;
            }
            .bloque-titulo h3 {
                margin: 0; font-size: 1.05rem; font-weight: 600;
                letter-spacing: -0.01em; color: #0f172a;
            }
            .bloque-titulo p {
                margin: 4px 0 0 0; color: #64748b; font-size: 0.875rem;
            }
            .bloque-titulo.exito { border-bottom-color: #86efac; }
            .bloque-titulo.alerta { border-bottom-color: #fcd34d; }
            /* Identidad: una línea, sin caja. */
            .barra-identidad {
                display: flex; align-items: baseline; gap: 10px;
                padding: 0 0 10px 0; margin: 0 0 14px 0;
                border-bottom: 1px solid #e2e8f0;
            }
            .barra-marca {
                font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;
                color: #0f172a;
            }
            .barra-alcance { font-size: 0.8125rem; color: #64748b; }
            /* Menos aire arriba: la pantalla empieza donde empieza el trabajo. */
            [data-testid="stMainBlockContainer"], .block-container {
                padding-top: 2.5rem; padding-bottom: 3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def barra_identidad(marca: str, alcance: str) -> None:
    """Muestra la identidad de la aplicación en una sola línea discreta.

    Sustituye al título grande: nombra el producto y su alcance sin ocupar media
    pantalla (regla de pantallas de aplicación: sin cabecera hero).

    Args:
        marca: nombre del producto.
        alcance: para qué sirve y para quién.
    """
    st.markdown(
        f'<div class="barra-identidad"><span class="barra-marca">{marca}</span>'
        f'<span class="barra-alcance">{alcance}</span></div>',
        unsafe_allow_html=True,
    )


def encabezado(titulo: str, subtitulo: str, variante: str = "") -> None:
    """Muestra un encabezado con estilo institucional.

    Args:
        titulo: título del bloque.
        subtitulo: descripción breve.
        variante: ``""``, ``"exito"`` o ``"alerta"`` (color del borde).
    """
    clase = f"bloque-titulo {variante}".strip()
    st.markdown(
        f'<div class="{clase}"><h3>{titulo}</h3><p>{subtitulo}</p></div>',
        unsafe_allow_html=True,
    )


def fila_kpis(metricas: list[dict[str, str]]) -> None:
    """Muestra una fila de indicadores (``st.metric``).

    Args:
        metricas: lista de diccionarios ``{"etiqueta", "valor", "delta"?}``.
    """
    columnas = st.columns(len(metricas))
    for columna, metrica in zip(columnas, metricas, strict=True):
        columna.metric(metrica["etiqueta"], metrica["valor"], metrica.get("delta"))


def grafico_tendencia(
    datos: pd.DataFrame,
    columna_x: str,
    columna_y: str,
    titulo: str,
    color: str = COLOR_PRIMARIO,
    columna_tipo_dia: str | None = "dia_tipo_nom",
    columna_etiqueta: str | None = None,
    nombre_y: str = "Usos",
    columna_inferior: str | None = None,
    columna_superior: str | None = None,
    columna_comparacion: str | None = None,
    nombre_comparacion: str = "Segunda opinión",
    columna_real: str | None = None,
    columna_retrospectiva: str | None = None,
    nombre_retrospectiva: str = "Proyectado hace N días",
) -> go.Figure:
    """Construye una gráfica de tendencia con marcadores por tipo de día.

    Args:
        datos: DataFrame con los datos.
        columna_x: columna de fechas.
        columna_y: columna de valores.
        titulo: título de la gráfica.
        color: color de la línea.
        columna_tipo_dia: columna de tipo de día para colorear los puntos.
        columna_etiqueta: columna con etiquetas de evento (tooltip).
        nombre_y: nombre del eje Y.
        columna_inferior: columna con el límite inferior del rango esperado (banda).
        columna_superior: columna con el límite superior del rango esperado (banda).
        columna_comparacion: columna con una segunda serie (por ejemplo, la segunda
            opinión) que se dibuja punteada junto a la principal.
        nombre_comparacion: nombre de esa segunda serie en la leyenda.
        columna_real: columna con la demanda real ya observada (para la retrospectiva).
        columna_retrospectiva: columna con lo que el modelo proyectó hace N días.
        nombre_retrospectiva: nombre de esa serie en la leyenda.

    Returns:
        La figura de Plotly.
    """
    figura = px.line(datos, x=columna_x, y=columna_y, title=titulo, markers=False)
    figura.update_traces(line_color=color, line_width=2.5, name=nombre_y)

    if (
        columna_inferior
        and columna_superior
        and {columna_inferior, columna_superior} <= set(datos.columns)
    ):
        x = pd.concat([datos[columna_x], datos[columna_x][::-1]])
        y = pd.concat([datos[columna_superior], datos[columna_inferior][::-1]])
        figura.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="Rango esperado (P10-P90)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if columna_real and columna_real in datos.columns:
        figura.add_trace(
            go.Scatter(
                x=datos[columna_x],
                y=datos[columna_real],
                mode="lines+markers",
                line={"color": COLOR_REAL, "width": 2.5},
                marker={"size": 7, "color": COLOR_REAL},
                name="Demanda real observada",
                connectgaps=False,
            )
        )

    if columna_retrospectiva and columna_retrospectiva in datos.columns:
        figura.add_trace(
            go.Scatter(
                x=datos[columna_x],
                y=datos[columna_retrospectiva],
                mode="lines+markers",
                line={"color": COLOR_COMPARACION, "width": 2, "dash": "dash"},
                marker={"size": 7, "color": COLOR_COMPARACION, "symbol": "diamond"},
                name=nombre_retrospectiva,
                connectgaps=False,
            )
        )

    if columna_comparacion and columna_comparacion in datos.columns:
        figura.add_trace(
            go.Scatter(
                x=datos[columna_x],
                y=datos[columna_comparacion],
                mode="lines+markers",
                line={"color": COLOR_COMPARACION, "width": 2, "dash": "dot"},
                marker={"size": 7, "color": COLOR_COMPARACION},
                name=nombre_comparacion,
            )
        )

    if columna_tipo_dia and columna_tipo_dia in datos.columns:
        for tipo, color_tipo in COLORES_TIPO_DIA.items():
            subconjunto = datos[datos[columna_tipo_dia] == tipo]
            if subconjunto.empty:
                continue
            personalizado = None
            if columna_etiqueta and columna_etiqueta in subconjunto.columns:
                personalizado = subconjunto[[columna_etiqueta]].where(
                    subconjunto[columna_etiqueta] != "", None
                )
            figura.add_trace(
                go.Scatter(
                    x=subconjunto[columna_x],
                    y=subconjunto[columna_y],
                    mode="markers",
                    marker={"color": color_tipo, "size": 9},
                    name=tipo,
                    customdata=personalizado,
                    hovertemplate="%{x|%Y-%m-%d}<br>"
                    + nombre_y
                    + ": %{y:,.0f}<br>%{customdata[0]}<extra></extra>"
                    if personalizado is not None
                    else None,
                )
            )

    figura.update_layout(height=420, legend_title="Tipo de día", hovermode="x unified")
    return figura


def grafico_comparacion(
    datos: pd.DataFrame,
    titulo: str,
    columna_x: str = "Fecha",
    columna_real: str = "Demanda_Real",
    columna_predicha: str = "Demanda_Predicha",
) -> go.Figure:
    """Construye una gráfica que compara la demanda real contra la predicha.

    Args:
        datos: DataFrame con las columnas de fecha, real y predicha.
        titulo: título de la gráfica.
        columna_x: columna de fechas.
        columna_real: columna con los valores observados.
        columna_predicha: columna con los valores proyectados.

    Returns:
        La figura de Plotly.
    """
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=datos[columna_x],
            y=datos[columna_real],
            mode="lines+markers",
            name="Real",
            line={"color": COLOR_PRIMARIO, "width": 2.5},
            marker={"size": 7},
        )
    )
    figura.add_trace(
        go.Scatter(
            x=datos[columna_x],
            y=datos[columna_predicha],
            mode="lines+markers",
            name="Proyectado",
            line={"color": COLOR_ALERTA, "width": 2.5, "dash": "dash"},
            marker={"size": 7, "symbol": "diamond"},
        )
    )
    figura.update_layout(
        height=420,
        title=titulo,
        hovermode="x unified",
        yaxis_title="Usos",
        legend_title="Serie",
    )
    return figura


def grafico_perfil_horario(
    datos: pd.DataFrame, titulo: str, columna_y: str = "usos_predicho"
) -> go.Figure:
    """Construye una gráfica de barras con los usos de cada hora del día.

    Args:
        datos: DataFrame con columnas ``hora`` y la métrica (usos).
        titulo: título de la gráfica.
        columna_y: columna de valores.

    Returns:
        La figura de Plotly.
    """
    ajustes = get_settings()
    inicio, fin = ajustes.hora_inicio_jornada, ajustes.hora_fin_jornada
    copia = datos.copy()
    copia["hora"] = copia["hora"].astype(int)
    copia = copia[(copia["hora"] >= inicio) & (copia["hora"] <= fin)]
    copia[columna_y] = copia[columna_y].round(0)
    copia = copia.sort_values("hora")

    figura = px.bar(
        copia,
        x="hora",
        y=columna_y,
        text_auto=",.0f",
        title=titulo,
        labels={"hora": "Hora del día", columna_y: "Usos por hora"},
    )
    figura.update_traces(
        marker_color=COLOR_PRIMARIO,
        texttemplate="%{y:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="Hora %{x}:00 · %{y:,.0f} usos<extra></extra>",
    )
    # Eje X con una marca por hora (sin decimales ni huecos) y eje Y en usos enteros.
    figura.update_xaxes(
        title="Hora del día",
        tickmode="linear",
        tick0=inicio,
        dtick=1,
        range=[inicio - 0.6, fin + 0.6],
    )
    figura.update_yaxes(title="Usos por hora", tickformat=",.0f")
    figura.update_layout(height=400, margin={"t": 50, "b": 40})
    return figura


def grafico_barras_por_tipo(
    datos: pd.DataFrame, columna_x: str, columna_y: str, titulo: str
) -> go.Figure:
    """Construye una gráfica de barras simple.

    Args:
        datos: DataFrame con los datos.
        columna_x: columna de categorías.
        columna_y: columna de valores.
        titulo: título de la gráfica.

    Returns:
        La figura de Plotly.
    """
    figura = px.bar(datos, x=columna_x, y=columna_y, title=titulo, color=columna_x)
    figura.update_layout(height=360, showlegend=False)
    return figura


def grafico_metricas_modelos(
    comparativa: list[dict[str, Any]],
    metrica: str,
    titulo: str,
    activo: str | None = None,
) -> go.Figure:
    """Gráfica de barras que compara los candidatos según una métrica.

    Resalta el modelo que se está usando y ordena de mejor a peor, de modo que se
    vea de un vistazo qué candidato conviene.

    Args:
        comparativa: filas con las métricas de cada candidato (``modelo`` y la métrica).
        metrica: nombre de la métrica a graficar (``mae``, ``wape``, ``mape`` o ``r2``).
        titulo: título de la gráfica.
        activo: nombre del modelo en uso (se resalta).

    Returns:
        La figura de Plotly.
    """
    inversa = metrica == "r2"
    datos = [
        {"modelo": str(fila["modelo"]), "valor": float(fila.get(metrica, 0) or 0)}
        for fila in comparativa
    ]
    datos.sort(key=lambda fila: fila["valor"], reverse=inversa)
    if not datos:
        return go.Figure()

    etiquetas = {
        "mae": "MAE (usos)",
        "wape": "WAPE (%)",
        "mape": "MAPE (%)",
        "r2": "R²",
    }
    colores = [COLOR_EXITO if fila["modelo"] == activo else COLOR_PRIMARIO for fila in datos]
    figura = go.Figure(
        go.Bar(
            x=[fila["modelo"] for fila in datos],
            y=[fila["valor"] for fila in datos],
            marker_color=colores,
            text=[
                f"{fila['valor']:,.3f}" if inversa else f"{fila['valor']:,.1f}" for fila in datos
            ],
            textposition="outside",
        )
    )
    figura.update_layout(
        height=340,
        title=titulo,
        yaxis_title=etiquetas.get(metrica, metrica),
        showlegend=False,
        margin={"t": 60, "b": 10},
    )
    return figura


def ficha_modelo(info: dict[str, Any], margen_error: float) -> None:
    """Muestra la ficha del modelo: con qué se proyectó, con qué historia y con qué error.

    La misma tabla sirve para el sistema y para estaciones/rutas, de modo que se pueda
    copiar en un informe y justificar la proyección.

    Args:
        info: descripción del modelo (salida de ``modelo_en_uso``).
        margen_error: semibanda diaria del rango esperado, en usos.
    """
    metricas = info.get("metricas", {})
    desde = info.get("desde_entrenamiento")
    hasta = info.get("hasta_entrenamiento")
    dias_historia = (hasta - desde).days + 1 if desde and hasta else 0
    composicion = (
        " + ".join(f"{nombre} {peso:.2f}" for nombre, peso in info.get("componentes", {}).items())
        if info.get("es_ensamble")
        else ""
    )

    filas = [
        {
            "Métrica": "Modelo",
            "Valor": str(info.get("campeon", "-")) + (f"  ({composicion})" if composicion else ""),
        },
        {"Métrica": "Entrenado", "Valor": str(info.get("fecha_entrenamiento"))[:10]},
        {
            "Métrica": "Periodo de entrenamiento",
            "Valor": (
                f"{desde} a {hasta}  ({dias_historia:,} días)".replace(",", ".")
                if desde and hasta
                else "—"
            ),
        },
        {"Métrica": "WAPE", "Valor": f"{float(metricas.get('wape', 0)):.2f} %"},
        {"Métrica": "MAE", "Valor": f"{formato_numero(metricas.get('mae', 0))} usos/día"},
        {"Métrica": "RMSE", "Valor": f"{formato_numero(metricas.get('rmse', 0))} usos"},
        {"Métrica": "R²", "Valor": f"{float(metricas.get('r2', 0)):.3f}"},
        {
            "Métrica": "Error esperado por día",
            "Valor": f"± {formato_numero(margen_error)} usos",
        },
        {"Métrica": "Confiabilidad", "Valor": f"{float(info.get('confiabilidad', 0)):.0%}"},
    ]
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)


def selector_dia_y_horario(horario: pd.DataFrame, clave: str) -> None:
    """Un día proyectado con sus usos hora a hora (misma vista en las dos pestañas).

    Args:
        horario: detalle horario con ``fecha``, ``hora`` y ``usos_predicho``.
        clave: sufijo único del selector (una vista por panel).
    """
    if horario is None or horario.empty:
        st.info("Esta proyección no tiene detalle horario.")
        return

    fechas = sorted(horario["fecha"].dt.date.unique())
    elegida = st.selectbox(
        "Día proyectado",
        fechas,
        index=0,
        key=f"dia_horario_{clave}",
        format_func=lambda fecha: fecha.strftime("%Y-%m-%d"),
    )
    st.plotly_chart(
        grafico_perfil_horario(
            horario[horario["fecha"].dt.date == elegida], f"Usos por hora · {elegida}"
        ),
        width="stretch",
    )


def panel_modelo_en_uso(info: dict[str, Any]) -> None:
    """Muestra, en un desplegable, **con qué modelo** se está proyectando.

    Aclara que, aunque se entrenen varios candidatos, por alcance se guarda un
    único campeón (que puede ser el ensamble de los candidatos).

    Args:
        info: descripción del modelo devuelta por
            :meth:`~mio_demanda.services.proyeccion_service.ServicioProyeccion.modelo_en_uso`.
    """
    metricas = info.get("metricas", {})
    campeon = info.get("campeon", "modelo")
    if info.get("es_ensamble"):
        composicion = " + ".join(
            f"{nombre} ({peso:.2f})" for nombre, peso in info.get("componentes", {}).items()
        )
        etiqueta_campeon = f"ensamble ponderado · {composicion}"
    else:
        etiqueta_campeon = campeon

    with st.expander(
        f"Modelo en uso: {info.get('alcance')} → {campeon} (paquete «{info.get('paquete')}»)",
        expanded=False,
    ):
        st.markdown(
            f"- **Alcance:** {info.get('alcance')}\n"
            f"- **Paquete guardado:** `{info.get('paquete')}`\n"
            f"- **Campeón:** {etiqueta_campeon}\n"
            f"- **Variables del modelo:** {info.get('n_variables')}\n"
            f"- **Entrenado el:** {str(info.get('fecha_entrenamiento'))[:19]}\n"
            f"- **Historia usada:** {info.get('desde_entrenamiento')} a "
            f"{info.get('hasta_entrenamiento')}\n"
            f"- **Confiabilidad:** {float(info.get('confiabilidad', 0)):.0%}"
        )
        st.caption(
            "Validación cruzada temporal — "
            f"MAE {float(metricas.get('mae', 0)):,.0f} · "
            f"WAPE {float(metricas.get('wape', 0)):.2f} % · "
            f"MAPE {float(metricas.get('mape', 0)):.2f} % · "
            f"R² {float(metricas.get('r2', 0)):.3f}"
        )
        st.markdown(
            "**¿Cómo se eligió?** Se evaluaron varios candidatos (boosting, bagging y "
            "lineal) con validación cruzada temporal y se guardó el de **menor MAE**. Si el "
            "campeón es el **ensamble**, combina esos candidatos con los pesos indicados "
            "(optimizados para minimizar el MAE), así que puede haber varios modelos "
            "trabajando juntos. Para cambiarlo, reentrena en la pestaña 1."
        )


def tabla_detalle(datos: pd.DataFrame, titulo: str = "Ver detalle") -> None:
    """Muestra una tabla de detalle en un desplegable.

    Args:
        datos: DataFrame a mostrar.
        titulo: etiqueta del desplegable.
    """
    with st.expander(titulo):
        st.dataframe(datos, width="stretch")


def boton_descarga_excel(
    hojas: dict[str, pd.DataFrame],
    nombre_archivo: str,
    etiqueta: str = "Exportar a Excel",
    clave: str | None = None,
) -> None:
    """Ofrece la descarga de un libro de Excel con una hoja por tabla.

    Args:
        hojas: mapa ``nombre de hoja -> DataFrame``.
        nombre_archivo: nombre del archivo sin extensión.
        etiqueta: texto del botón.
        clave: clave única del widget (necesaria si hay varios botones iguales).
    """
    contenido = _libro_excel(hojas)
    st.download_button(
        etiqueta,
        contenido,
        file_name=f"{nombre_archivo}.xlsx",
        mime=_TIPO_EXCEL,
        key=clave,
    )


def _libro_excel(hojas: dict[str, pd.DataFrame]) -> bytes:
    """Construye un libro de Excel en memoria.

    Args:
        hojas: mapa ``nombre de hoja -> DataFrame``.

    Returns:
        Los bytes del archivo ``.xlsx``.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as escritor:
        for nombre, tabla in hojas.items():
            hoja = _nombre_hoja_valido(nombre)
            (tabla if tabla is not None else pd.DataFrame()).to_excel(
                escritor, sheet_name=hoja, index=False
            )
    return buffer.getvalue()


def _nombre_hoja_valido(nombre: str) -> str:
    """Sanea el nombre de una hoja de Excel (sin caracteres inválidos ni exceso)."""
    limpio = _CARACTERES_INVALIDOS_HOJA.sub("-", str(nombre)).strip()
    return (limpio or "Datos")[:_LARGO_MAX_HOJA]


def aviso_coherencia(
    proyectado: float,
    referencia: float,
    etiqueta: str = "de la demanda",
    minimo: float = 0.5,
    maximo: float = 2.0,
) -> None:
    """Compara lo proyectado con el promedio real reciente y avisa si no cuadra.

    Evita que una proyección absurda (por un modelo mal alineado, datos insuficientes,
    etc.) pase desapercibida.

    Args:
        proyectado: promedio diario proyectado.
        referencia: promedio diario real de los últimos días.
        etiqueta: texto para el mensaje.
        minimo: razón mínima aceptable (configurable).
        maximo: razón máxima aceptable (configurable).
    """
    if referencia <= 0 or proyectado <= 0:
        return
    razon = proyectado / referencia
    if minimo <= razon <= maximo:
        st.caption(
            f"Control de coherencia: el promedio proyectado es **{razon:.0%}** del promedio "
            f"real reciente ({referencia:,.0f} usos/día)."
        )
        return
    st.warning(
        f"⚠️ **Revisa el resultado:** el promedio proyectado ({proyectado:,.0f} usos/día) es "
        f"{razon:.0%} {etiqueta} ({referencia:,.0f} usos/día). "
        "Suele indicar un modelo entrenado con otro alcance o datos insuficientes: "
        "revisa el *Modelo en uso* y la pestaña correcta."
    )


def aviso_sin_modelo(alcance: str) -> None:
    """Muestra un aviso cuando el modelo requerido no está entrenado.

    Args:
        alcance: descripción del alcance (``"sistema"``, ``"estaciones"``, ...).
    """
    st.warning(
        f"El modelo de **{alcance}** no está entrenado. "
        "Ve a la pestaña **1. Entrenamiento** (zona del profesional), entrénalo y guárdalo."
    )


def formato_numero(valor: float) -> str:
    """Formatea un número con separadores de miles.

    Args:
        valor: valor numérico.

    Returns:
        Cadena formateada.
    """
    return f"{valor:,.0f}"


def tabla_comparativa_candidatos(comparativa: list[dict[str, Any]]) -> pd.DataFrame:
    """Tabla técnica de los candidatos de un alcance.

    Incluye el **MAPE** solo como referencia (se infla con demandas cercanas a cero) y
    el **sMAPE** cuando el modelo se entrenó con la versión que lo calcula.

    Args:
        comparativa: filas con las métricas de cada candidato.

    Returns:
        DataFrame ordenado por MAE, con las columnas disponibles.
    """
    columnas = [
        ("MAE", "mae", 1),
        ("RMSE", "rmse", 1),
        ("WAPE %", "wape", 2),
        ("sMAPE %", "smape", 2),
        ("MAPE %", "mape", 2),
        ("R²", "r2", 3),
    ]
    filas = []
    for fila in comparativa:
        registro: dict[str, Any] = {
            "Modelo": fila.get("modelo"),
            "Tipo": fila.get("tipo", "-"),
        }
        for etiqueta, sufijo, decimales in columnas:
            if sufijo in fila and fila[sufijo] is not None:
                registro[etiqueta] = round(float(fila[sufijo] or 0), decimales)
        filas.append(registro)
    tabla = pd.DataFrame(filas)
    if "MAE" in tabla.columns:
        tabla = tabla.sort_values("MAE").reset_index(drop=True)
    return tabla


def etiqueta_modelo(metadatos: dict[str, Any]) -> str:
    """Construye el nombre identificable de un modelo guardado.

    Formato: ``alcance-fecha-hora-r2`` (por ejemplo
    ``sistema-2026-10-15_11-03-27-r2-0.945``), de modo que al elegir o eliminar se
    sepa exactamente de qué modelo se trata.

    Args:
        metadatos: metadatos del modelo (``nombre``, ``fecha_entrenamiento``,
            ``metricas``).

    Returns:
        La etiqueta legible del modelo.
    """
    crudo = str(metadatos.get("fecha_entrenamiento", ""))
    try:
        momento = datetime.fromisoformat(crudo)
        sello = momento.strftime("%Y-%m-%d_%H-%M-%S")
    except ValueError:
        sello = crudo[:19].replace(":", "-").replace(" ", "_") or "sin-fecha"

    metricas = metadatos.get("metricas", {})
    r2 = float(metricas.get("r2", 0.0) or 0.0)
    return f"{metadatos.get('nombre')}-{sello}-r2-{r2:.3f}"


def tabla_modelos(metadatos: list[dict[str, Any]]) -> pd.DataFrame:
    """Construye la tabla amigable de modelos guardados.

    Args:
        metadatos: lista de metadatos de modelos.

    Returns:
        DataFrame con una fila por modelo: nombre identificable, alcance, con qué
        se proyecta, si la elección fue manual, métricas y fecha de entrenamiento.
    """
    filas = []
    for meta in metadatos:
        metricas = meta.get("metricas", {})
        extras = meta.get("extras", {})
        desde = str(meta.get("desde_entrenamiento", ""))
        hasta = str(meta.get("hasta_entrenamiento", ""))
        try:
            dias = (date.fromisoformat(hasta) - date.fromisoformat(desde)).days + 1
        except ValueError:
            dias = 0
        filas.append(
            {
                "Modelo": etiqueta_modelo(meta),
                "Alcance": meta.get("nombre"),
                "Proyecta con": extras.get("campeon", "-"),
                "Elección": "Manual" if extras.get("activado_manualmente") else "Automática",
                "MAE": round(float(metricas.get("mae", 0) or 0), 1),
                "WAPE %": round(float(metricas.get("wape", 0) or 0), 2),
                "R²": round(float(metricas.get("r2", 0) or 0), 3),
                "Días historia": dias,
                "Desde": desde,
                "Hasta": hasta,
                "Entrenado": str(meta.get("fecha_entrenamiento", ""))[:19].replace("T", " "),
            }
        )
    return pd.DataFrame(filas)
