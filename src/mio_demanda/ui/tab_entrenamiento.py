"""Pestaña 1 — Entrenamiento (zona del profesional).

Permite: verificar/extraer datos, consultar el **calendario** (eventos estándar
calculados por regla y eventos del usuario), escoger el **periodo de
entrenamiento**, entrenar los modelos candidatos con validación cruzada y Optuna,
ver la comparativa y guardar el campeón. La UI solo llama a los servicios.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from mio_demanda.domain.entidades import CategoriaEvento, PerfilImpacto
from mio_demanda.domain.metricas import confiabilidad_desde_error
from mio_demanda.services.calendario_service import OPCIONES_CATEGORIA, OPCIONES_PERFIL
from mio_demanda.services.entrenamiento_service import ALCANCES, ServicioEntrenamiento
from mio_demanda.ui import componentes as c
from mio_demanda.ui.estado import (
    estado_datos,
    invalidar_cache,
    obtener_calendario,
    obtener_catalogo,
    obtener_settings,
    rango_datos_disponibles,
)

_TITULO_ALCANCE = {"sistema": "Sistema completo", "estaciones": "Estaciones", "rutas": "Rutas"}

#: Categorías para el formulario, con «otro» primero (el valor neutro por defecto).
_OPCIONES_CATEGORIA_EVENTO = [
    CategoriaEvento.OTRO.value,
    *(categoria.value for categoria in CategoriaEvento if categoria is not CategoriaEvento.OTRO),
]


def _anios_calendario() -> list[int]:
    """Años disponibles para consultar el calendario.

    Se derivan de los datos y se abren varios años hacia adelante: los eventos
    estándar se calculan por regla para **cualquier** año, así que no hay razón para
    quedarse corto al planificar.

    Returns:
        Años desde uno antes del primer dato hasta varios después del último.
    """
    minimo, maximo = rango_datos_disponibles()
    return list(
        range(minimo.year - 1, maximo.year + obtener_settings().calendario_anios_futuros + 1)
    )


def render() -> None:
    """Dibuja la pestaña de entrenamiento (solo administradores)."""
    _seccion_datos()
    st.divider()
    _seccion_calendario()
    st.divider()
    _seccion_entrenamiento()
    st.divider()
    _seccion_modelos_guardados()


# ======================================================================
# Acceso
# ======================================================================


# ======================================================================
# Datos
# ======================================================================
def _seccion_datos() -> None:
    """Muestra el estado de los parquet y permite actualizar la base de datos."""
    st.subheader("1. Datos")
    estado = estado_datos()

    if estado["hay_datos"]:
        minimo, maximo = estado["rango"] or (None, None)
        st.success(f"Base de datos disponible. Usos: {minimo} a {maximo}")
    else:
        st.warning("Todavía no hay datos en `data/raw`. Actualiza la base de datos.")

    archivos = estado.get("archivos", {})
    st.caption(
        "La aplicación trabaja con los archivos **parquet** de `data/raw`: "
        + (", ".join(archivos) if archivos else "ninguno todavía")
        + "."
    )

    with st.expander("Rutas de datos"):
        ajustes = obtener_settings()
        st.caption(
            "Todo se cambia en el archivo `.env` y se comprueba con "
            "`uv run python -m mio_demanda.cli rutas`. No hay rutas escritas en el código."
        )
        st.dataframe(
            pd.DataFrame(ajustes.rutas_de_datos()),
            width="stretch",
            hide_index=True,
        )
        if not ajustes.ruta_usos.exists():
            st.error(
                "No se encuentra la carpeta de origen. Revisa `MIO_DIRECTORIO_BASE_PBI` "
                "en `.env`: debe apuntar a la carpeta de los reportes."
            )

    if st.button("Actualizar base de datos", type="primary"):
        with st.status("Actualizando Base de Datos...", expanded=True) as salida:
            conteos = obtener_catalogo().extraer_datos()
            for fuente, total in conteos.items():
                salida.write(f"{fuente}: {total:,} registros")
            salida.update(label="Base de datos actualizada", state="complete")
        invalidar_cache()
        st.rerun()


# ======================================================================
# Calendario
# ======================================================================
def _seccion_calendario() -> None:
    """Muestra los eventos estándar y gestiona los del usuario."""
    st.subheader("2. Calendario de eventos")
    st.caption(
        "Los eventos explican picos o caídas de demanda. Los **estándar** son nacionales y los "
        "calcula el motor de reglas; los **del usuario** los carga la aplicación (Feria de Cali, "
        "recesos escolares, Black Friday, elecciones, sismos, obras) y también puedes crear los tuyos."
    )
    servicio = obtener_calendario()
    anio = _selector_anio_calendario()
    _tabla_eventos_estandar(servicio, anio)
    _gestion_eventos_usuario(servicio, anio)
    _vista_mensual(servicio)
    _editor_eventos(servicio, anio)


def _editor_eventos(servicio, anio: int) -> None:
    """Editor tipo Excel para retoques menores de todos los eventos."""
    st.markdown("#### Edición rápida de todos los eventos (tipo Excel)")
    st.caption(
        "Retoques menores sin tocar los archivos: se puede cambiar el **impacto**, el **perfil**, "
        "la **categoría** y la **fuente**. Los eventos del usuario se modifican directamente; los "
        "estándar se guardan como **ajustes** (el catálogo base no se altera). Las fechas solo se "
        "aplican a los eventos de fechas fijas. Tras guardar, hay que **reentrenar**."
    )

    columnas = [
        "Id",
        "Evento",
        "Inicio",
        "Fin",
        "Impacto",
        "Perfil",
        "Categoria",
        "Fuente",
        "Origen",
    ]
    original = servicio.catalogo_editable(date(anio, 1, 1), date(anio, 12, 31))
    if original.empty:
        st.info("No hay eventos para editar en el año seleccionado.")
        return
    original = original[columnas].reset_index(drop=True)

    editado = st.data_editor(
        original,
        key="editor_eventos",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Id", "Origen"],
        column_config={
            "Evento": st.column_config.TextColumn("Evento", width="large"),
            "Inicio": st.column_config.TextColumn("Inicio (solo fechas fijas)"),
            "Fin": st.column_config.TextColumn("Fin (solo fechas fijas)"),
            "Impacto": st.column_config.NumberColumn(
                "Impacto", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
            "Perfil": st.column_config.SelectboxColumn(
                "Perfil", options=OPCIONES_PERFIL, required=True
            ),
            "Categoria": st.column_config.SelectboxColumn(
                "Categoría", options=OPCIONES_CATEGORIA, required=True
            ),
            "Fuente": st.column_config.TextColumn("Fuente", width="large"),
        },
    )

    if st.button("Guardar cambios de eventos", type="primary", key="btn_guardar_eventos"):
        cambiados = _filas_cambiadas(original, editado)
        if not cambiados:
            st.info("No hay cambios por guardar.")
            return
        try:
            total = servicio.actualizar_eventos(cambiados)
        except RuntimeError as error:
            st.error(f"{error}")
            return
        invalidar_cache()
        st.success(f"Se actualizaron {total} evento(s). Reentrena para aplicar los cambios.")
        st.rerun()


def _filas_cambiadas(original: pd.DataFrame, editado: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las filas que el usuario modificó en el editor.

    Args:
        original: tabla antes de editar.
        editado: tabla devuelta por ``st.data_editor``.

    Returns:
        DataFrame con las filas modificadas (vacío si no hubo cambios).
    """
    if editado is None or len(editado) != len(original):
        return editado if editado is not None else pd.DataFrame()
    iguales = (
        original.astype(str).reset_index(drop=True) == editado.astype(str).reset_index(drop=True)
    ).all(axis=1)
    return editado[~iguales.to_numpy()].reset_index(drop=True)


def _indice_anio_actual() -> int:
    """Posición del año actual en la lista de años disponibles."""
    anios = _anios_calendario()
    anio = date.today().year
    return anios.index(anio) if anio in anios else 0


def _selector_anio_calendario() -> int:
    """Selector del año de referencia para el calendario.

    Returns:
        El año seleccionado.
    """
    col_anio = st.columns([1, 3])[0]
    return int(
        col_anio.selectbox(
            "Año de referencia", _anios_calendario(), index=_indice_anio_actual(), key="cal_anio"
        )
    )


def _tabla_eventos_estandar(servicio, anio: int) -> None:
    """Muestra las fechas de los eventos estándar (nacionales)."""
    st.markdown("#### Eventos estándar (nacionales y de calendario)")
    st.caption(
        f"{servicio.total_eventos_base()} eventos en `config/eventos_base.json`, calculados por "
        "regla para cualquier año: festivos, Semana Santa y fechas comerciales."
    )
    tabla = servicio.eventos_base(anio)
    if tabla.empty:
        st.info("No hay eventos estándar definidos para el año seleccionado.")
        return
    st.dataframe(tabla.drop(columns=["Id"]), width="stretch", hide_index=True, height=300)


def _gestion_eventos_usuario(servicio, anio: int) -> None:
    """Muestra y gestiona los eventos del usuario (recurrentes y de fechas fijas)."""
    st.markdown("#### Eventos del usuario (no estándar)")
    ruta = servicio.ruta_archivo_eventos()
    st.caption(
        f"{servicio.total_eventos_usuario()} eventos en `{ruta}`. Aquí viven los que no son "
        "estándar (Feria de Cali, recesos escolares, Black Friday, elecciones, sismos, obras). "
        "Pueden ser **recurrentes** (se repiten cada año) o de **fechas fijas**. Se guardan de "
        "forma atómica: agregar uno no borra los anteriores."
    )

    tabla = servicio.eventos_usuario(anio)
    if tabla.empty:
        st.info("No hay eventos del usuario registrados.")
    else:
        columnas = [
            "Evento",
            "Inicio",
            "Fin",
            "Dias",
            "Impacto",
            "Perfil",
            "Definicion",
            "Categoria",
        ]
        st.dataframe(tabla[columnas], width="stretch", hide_index=True, height=280)

    acciones = st.columns([1, 1, 3])
    with acciones[0]:
        agregar = st.toggle("Agregar evento", key="ev_mostrar_formulario")
    with acciones[1]:
        eliminar = st.toggle("Eliminar evento", key="ev_mostrar_eliminar", disabled=tabla.empty)
    with acciones[2]:
        st.caption("Usa las dos opciones de arriba para agregar o eliminar eventos del usuario.")

    if agregar:
        _formulario_evento(servicio)

    if eliminar and not tabla.empty:
        _eliminar_evento(servicio, tabla)


def _eliminar_evento(servicio, tabla: pd.DataFrame) -> None:
    """Elimina un evento del usuario, con confirmación.

    Args:
        servicio: servicio de calendario.
        tabla: eventos del usuario vigentes en el año.
    """
    opciones = {f"{fila['Evento']} ({fila['Id']})": fila["Id"] for _, fila in tabla.iterrows()}
    seleccionado = st.selectbox("Evento a eliminar", list(opciones), key="ev_eliminar")
    confirmar = st.checkbox("Confirmo la eliminación", key="ev_confirmar_eliminar")
    if st.button(
        "Eliminar evento",
        type="secondary",
        disabled=not confirmar,
        key="btn_eliminar_evento",
    ):
        try:
            servicio.eliminar_evento(str(opciones[str(seleccionado)]))
        except RuntimeError as error:
            st.error(f"{error}")
            return
        invalidar_cache()
        st.rerun()


def _formulario_evento(servicio) -> None:
    """Formulario para agregar un evento del usuario (recurrente o de fechas fijas)."""
    st.markdown("##### Nuevo evento")

    with st.form("form_evento", clear_on_submit=True):
        nombre = st.text_input("Nombre del evento", placeholder="Ej. Sismo del 10 de agosto")

        modo = st.radio(
            "¿Cómo se define?",
            ["Fechas fijas", "Se repite cada año", "Fecha central con días antes y después"],
            horizontal=True,
            key="ev_modo",
        )

        inicio = fin = date.today()
        central = date.today()
        dias_antes = dias_despues = 0
        if modo == "Fechas fijas":
            col1, col2 = st.columns(2)
            inicio = col1.date_input("Desde", value=date.today(), key="ev_inicio")
            fin = col2.date_input("Hasta", value=date.today(), key="ev_fin")
        elif modo == "Se repite cada año":
            col1, col2 = st.columns(2)
            inicio = col1.date_input(
                "Desde (solo se usa mes y día)", value=date.today(), key="ev_rec_ini"
            )
            fin = col2.date_input(
                "Hasta (solo se usa mes y día)", value=date.today(), key="ev_rec_fin"
            )
            st.caption("Se repetirá todos los años en ese rango de mes y día.")
        else:
            col1, col2, col3 = st.columns([2, 1, 1])
            central = col1.date_input("Fecha central", value=date.today(), key="ev_central")
            dias_antes = int(col2.number_input("Días antes", 0, 180, 0, key="ev_antes"))
            dias_despues = int(col3.number_input("Días después", 0, 365, 0, key="ev_despues"))
            inicio = central - timedelta(days=dias_antes)
            fin = central + timedelta(days=dias_despues)
            st.caption(f"Afectará del {inicio} al {fin} ({(fin - inicio).days + 1} días en total).")

        col4, col5, col6 = st.columns([1, 1, 1])
        categoria = col4.selectbox(
            "Categoría",
            _OPCIONES_CATEGORIA_EVENTO,
            index=0,
            key="ev_categoria",
            help=(
                "Clasificación del evento. Por defecto «otro»; usa «desastre» para sismos o "
                "emergencias, «feria» solo para ferias, «obra» para cierres por obras, etc."
            ),
        )
        perfil = col5.selectbox(
            "Perfil del impacto",
            [perfil.value for perfil in PerfilImpacto],
            index=0,
            key="ev_perfil",
            help=(
                "Constante: el efecto se mantiene. Decreciente: empieza fuerte y se desvanece "
                "(sismos). Creciente: el efecto crece con los días."
            ),
        )
        impacto = col6.slider(
            "Impacto esperado (0-1)",
            0.0,
            1.0,
            0.5,
            0.05,
            key="ev_impacto",
            help=(
                "Importancia relativa del evento (no es un porcentaje de cambio en la demanda). "
                "Guía: 0.9-1.0 el servicio se suspende o cae drásticamente; 0.7-0.9 afectación "
                "fuerte y clara; 0.5-0.7 notable pero parcial; 0.3-0.5 leve-moderada; 0.1-0.3 "
                "marginal. Con perfil decreciente, el primer día pesa este valor y el último el 10 %."
            ),
        )
        fuente = st.text_input(
            "Fuente / justificación",
            placeholder="Ej. Alcaldía de Cali, Resolución 4143",
            key="ev_fuente",
        )

        _guia_impacto()

        if st.form_submit_button("Agregar evento", type="primary"):
            if not nombre.strip():
                st.error("El nombre del evento es obligatorio.")
            else:
                datos = {
                    "nombre": nombre.strip(),
                    "impacto": impacto,
                    "categoria": CategoriaEvento(categoria),
                    "fuente": fuente.strip(),
                    "perfil_impacto": PerfilImpacto(perfil),
                }
                try:
                    if modo == "Se repite cada año":
                        servicio.crear_evento_recurrente(
                            mes_inicio=inicio.month,
                            dia_inicio=inicio.day,
                            mes_fin=fin.month,
                            dia_fin=fin.day,
                            **datos,
                        )
                    elif modo == "Fechas fijas":
                        servicio.crear_evento(fecha_inicio=inicio, fecha_fin=fin, **datos)
                    else:
                        servicio.crear_evento_con_ventana(
                            fecha_central=central,
                            dias_antes=dias_antes,
                            dias_despues=dias_despues,
                            **datos,
                        )
                except (RuntimeError, ValueError) as error:
                    st.error(f"{error}")
                    return
                invalidar_cache()
                st.success(f"Evento «{nombre}» guardado. Recuerda reentrenar para aplicarlo.")
                st.rerun()


def _guia_impacto() -> None:
    """Muestra la guía de valores del impacto dentro del formulario."""
    with st.expander("¿Qué valor de impacto pongo?"):
        st.markdown(
            """
            El **impacto** es la *importancia relativa* que le das al evento, entre 0 y 1.
            **No** es un porcentaje de cambio en la demanda: es el peso con el que el evento
            entra al modelo (variables `impacto_evento` y `peso_eventos`).

            | Impacto | Cuándo usarlo | Ejemplos |
            | :--- | :--- | :--- |
            | **0.90 - 1.00** | El servicio se suspende o cae drásticamente | Sismo fuerte con vías cerradas, paro total del sistema |
            | **0.70 - 0.89** | Afectación fuerte y clara, varios días | Semana Santa, Feria de Cali, elecciones, sismo con afectación parcial |
            | **0.50 - 0.69** | Afectación notable pero parcial | Obra que cierra un carril, evento masivo en un sector |
            | **0.30 - 0.49** | Afectación leve o moderada | Receso escolar, fecha comercial, lluvia intensa |
            | **0.10 - 0.29** | Afectación marginal | Evento local pequeño, ajuste fino |
            | **0.00** | No usar: equivale a no registrar el evento | - |

            **Cómo se usa el valor**

            - **Perfil Constante:** todos los días del evento pesan el valor indicado.
            - **Perfil Decreciente** (sismos, paros largos): el primer día pesa el valor
              indicado y el último un 10 % de ese valor, de forma lineal.
            - **Perfil Creciente:** al contrario; el último día pesa el valor indicado.

            El impacto no cambia la demanda por sí mismo: el modelo aprende el efecto real
            durante el entrenamiento.
            """
        )


def _vista_mensual(servicio) -> None:
    """Muestra el calendario de un mes con sus eventos."""
    with st.expander("Ver calendario mensual", expanded=False):
        hoy = date.today()
        col1, col2 = st.columns(2)
        anio = col1.number_input("Año", min_value=2015, max_value=2035, value=hoy.year, step=1)
        mes = col2.number_input("Mes", min_value=1, max_value=12, value=hoy.month, step=1)

        calendario = servicio.calendario_mes(int(anio), int(mes))
        calendario = calendario.rename(
            columns={
                "fecha": "Fecha",
                "dia_tipo_nom": "Tipo de día",
                "eventos": "Eventos",
                "es_fin_semana": "Fin de semana",
            }
        )
        calendario["Fecha"] = pd.to_datetime(calendario["Fecha"]).dt.strftime("%Y-%m-%d")
        st.dataframe(calendario, width="stretch", hide_index=True, height=340)


# ======================================================================
# Entrenamiento
# ======================================================================
def _seccion_entrenamiento() -> None:
    """Panel de entrenamiento (plegado hasta que el usuario lo pide)."""
    st.subheader("3. Entrenamiento del modelo")
    abrir = st.toggle(
        "Abrir panel de entrenamiento",
        key="train_abrir_panel",
        help="Configura el periodo, los alcances y la observación, y entrena.",
    )
    if not abrir:
        st.caption(
            "El panel está plegado. Ábrelo para ajustar el periodo, los alcances y "
            "la observación de por qué entrenas así."
        )
        return

    settings = obtener_settings()
    minimo, maximo = rango_datos_disponibles()

    desde, hasta = _selector_periodo(minimo, maximo)
    if desde is None or hasta is None:
        return

    col1, col2 = st.columns(2)
    alcances = col1.multiselect(
        "Alcance", ALCANCES, default=["sistema"], format_func=lambda a: _TITULO_ALCANCE[a]
    )
    n_trials = col2.slider("Trials de Optuna por candidato", 3, 60, settings.n_trials_optuna, 1)
    n_splits = st.slider("Particiones de validación cruzada", 2, 6, settings.n_splits_cv, 1)

    observacion = st.text_area(
        "Observación del entrenamiento",
        key="train_observacion",
        placeholder=(
            "¿Por qué entrenas con este periodo y estos parámetros? "
            "Ej.: se excluye el sismo de agosto; se prueba ventana de 36 meses por el cambio de oferta."
        ),
        help="Queda guardada junto al modelo y se ve en el historial de entrenamientos.",
    )

    st.caption(
        "Cada alcance se guarda por separado y **conserva el periodo con el que se entrenó**, "
        "así que puedes darle a Estaciones o Rutas una fecha distinta: entrena de a uno."
    )

    if desde >= hasta:
        st.error("La fecha inicial debe ser anterior a la final.")
        return
    if not alcances:
        st.info("Selecciona al menos un alcance para entrenar.")
        return

    if st.button("Entrenar y guardar modelos", type="primary"):
        _ejecutar_entrenamiento(desde, hasta, alcances, n_trials, n_splits, observacion)


def _selector_periodo(minimo: date, maximo: date) -> tuple[date | None, date | None]:
    """Permite elegir el periodo de entrenamiento: inicio por atajo y fin explícito.

    El **fin del periodo lo eliges siempre tú**; los atajos solo calculan el inicio
    (hacia atrás desde ese fin). Así «Últimos 36 meses» termina donde tú decidas y no
    forzosamente en el último dato disponible.

    Args:
        minimo: primera fecha con datos disponibles.
        maximo: última fecha con datos disponibles.

    Returns:
        Tupla ``(desde, hasta)``; ``(None, None)`` si el rango es inválido.
    """
    opciones = [
        "Últimos 12 meses",
        "Últimos 24 meses",
        "Últimos 36 meses",
        "Todo el histórico",
        "Personalizado",
    ]
    col1, col2, col3 = st.columns(3)
    preset = col1.selectbox("Periodo de entrenamiento", opciones, index=1, key="train_periodo")
    hasta = col2.date_input(
        "Hasta (fin del entrenamiento)",
        value=maximo,
        min_value=minimo,
        max_value=maximo,
        key="train_hasta",
        help="Puede ser anterior al último dato: así dejas días reales para evaluar el modelo.",
    )

    if preset == "Personalizado":
        desde = col3.date_input(
            "Desde (inicio)",
            value=minimo,
            min_value=minimo,
            max_value=hasta,
            key="train_desde",
        )
    elif preset == "Todo el histórico":
        desde = minimo
        col3.metric("Desde", str(desde), help="Todo el histórico disponible")
    else:
        meses = {"Últimos 12 meses": 12, "Últimos 24 meses": 24, "Últimos 36 meses": 36}[preset]
        desde = max(_restar_meses(hasta, meses), minimo)
        col3.metric("Desde", str(desde), help=f"{meses} meses antes del fin elegido")

    if desde is None or hasta is None or hasta <= desde:
        st.error("El periodo seleccionado no es válido: el inicio debe ser anterior al fin.")
        return None, None

    dias = (hasta - desde).days + 1
    st.info(
        f"Se entrenará del **{desde}** al **{hasta}** ({dias:,} días). El modelo queda guardado "
        "con ese periodo y lo consumen las pestañas 2 y 3."
    )
    if hasta >= maximo:
        st.caption(
            "⚠️ Con «Hasta» en el último dato no quedan días reales para evaluar: el backtest "
            "reentrenará una copia temporal. Si prefieres evaluar sin reentrenar, pon un «Hasta» "
            "de 2 a 4 semanas atrás."
        )
    return desde, hasta


def _restar_meses(fecha: date, meses: int) -> date:
    """Resta meses a una fecha.

    Args:
        fecha: fecha de referencia.
        meses: número de meses a restar.

    Returns:
        La fecha resultante (ajustada al último día del mes si hace falta).
    """
    return (pd.Timestamp(fecha) - pd.DateOffset(months=meses)).date()


def _ejecutar_entrenamiento(
    desde: date,
    hasta: date,
    alcances: list[str],
    n_trials: int,
    n_splits: int,
    observacion: str = "",
) -> None:
    """Orquesta el entrenamiento y muestra el progreso y los resultados."""
    settings = obtener_settings()
    with st.status("Entrenando modelos...", expanded=True) as salida:
        servicio = ServicioEntrenamiento(settings, notificar=salida.write)
        resultados = {}
        for alcance in alcances:
            salida.write(f"### {_TITULO_ALCANCE[alcance]}")
            try:
                resultados[alcance] = servicio.entrenar(
                    alcance,
                    desde,
                    hasta,
                    n_trials=n_trials,
                    n_splits=n_splits,
                    observacion=observacion,
                )
            except Exception as error:
                salida.write(f"Error en {alcance}: {error}")
        salida.update(label="Entrenamiento finalizado", state="complete")

    invalidar_cache()
    servicio = ServicioEntrenamiento(settings)
    for alcance, resultado in resultados.items():
        _mostrar_resultado(alcance, resultado, servicio)


def _mostrar_resultado(alcance: str, resultado, servicio) -> None:
    """Presenta la comparativa, el campeón y el selector del modelo a proyectar."""
    st.markdown(f"#### {_TITULO_ALCANCE[alcance]} — campeón: `{resultado.campeon}`")
    metricas = resultado.metricas_cv_campeon
    c.fila_kpis(
        [
            {"etiqueta": "WAPE", "valor": f"{metricas.get('wape', 0):.2f} %"},
            {"etiqueta": "MAE (usos)", "valor": c.formato_numero(metricas.get("mae", 0))},
            {"etiqueta": "RMSE (usos)", "valor": c.formato_numero(metricas.get("rmse", 0))},
            {"etiqueta": "R²", "valor": f"{metricas.get('r2', 0):.3f}"},
            {
                "etiqueta": "Confiabilidad",
                "valor": f"{confiabilidad_desde_error(metricas.get('wape', 15.0)):.0%}",
            },
        ]
    )
    st.caption(
        "El **WAPE** es el indicador de referencia del negocio (error porcentual ponderado "
        "por volumen). El **MAPE** queda en la tabla solo como referencia técnica."
    )

    comparativa = resultado.comparativa.copy()
    st.dataframe(
        c.tabla_comparativa_candidatos(comparativa.to_dict(orient="records")),
        width="stretch",
        hide_index=True,
    )

    if resultado.campeon == "ensamble" and resultado.pesos_ensamble:
        st.caption(
            "Pesos del ensamble: "
            + " · ".join(
                f"{nombre} = {peso:.2f}" for nombre, peso in resultado.pesos_ensamble.items()
            )
        )

    _selector_modelo_activo(
        servicio,
        alcance,
        resultado.comparativa.to_dict(orient="records"),
        resultado.paquete.extras.get("campeon"),
        bool(resultado.paquete.extras.get("activado_manualmente")),
        clave="reciente",
    )


def _selector_modelo_activo(
    servicio,
    alcance: str,
    comparativa: list[dict],
    activo: str | None,
    elegido_manualmente: bool,
    clave: str,
) -> None:
    """Permite elegir con qué candidato se proyectará el alcance.

    Args:
        servicio: servicio de entrenamiento.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.
        comparativa: filas con las métricas de cada candidato.
        activo: nombre del modelo actualmente activo.
        elegido_manualmente: si el activo fue elegido a mano.
        clave: sufijo de las claves de los widgets.
    """
    if not comparativa:
        return

    st.markdown("##### Modelo que se usará para proyectar")
    etiquetas = {
        f"{fila['modelo']} — MAE {float(fila['mae']):,.0f} · WAPE {float(fila['wape']):.2f} %": str(
            fila["modelo"]
        )
        for fila in comparativa
    }
    opciones = list(etiquetas)
    indice = next((i for i, texto in enumerate(opciones) if etiquetas[texto] == activo), 0)

    col1, col2 = st.columns([3, 1])
    seleccion = col1.radio(
        "Candidato",
        opciones,
        index=indice,
        key=f"activo_{alcance}_{clave}",
        label_visibility="collapsed",
    )
    if col2.button("Usar para proyectar", key=f"btn_activo_{alcance}_{clave}"):
        if etiquetas[seleccion] == activo:
            st.info(f"«{activo}» ya es el modelo activo de {alcance}.")
        else:
            with st.spinner(f"Reentrenando «{etiquetas[seleccion]}» con toda la historia..."):
                try:
                    servicio.activar_modelo(alcance, etiquetas[seleccion])
                except (ValueError, FileNotFoundError) as error:
                    st.error(f"{error}")
                    return
            invalidar_cache()
            st.success(
                f"Listo: las pestañas 2 y 3 proyectarán {alcance} con «{etiquetas[seleccion]}»."
            )

    st.caption(
        f"Activo actualmente: **{activo}**"
        + (" (elegido manualmente)." if elegido_manualmente else " (campeón automático por MAE).")
        + " Al reentrenar vuelve a fijarse el campeón automático."
    )


# ======================================================================
# Modelos guardados
# ======================================================================
def _seccion_modelos_guardados() -> None:
    """Modelos guardados, **separados por alcance**, con su historial de entrenamientos.

    Cada alcance tiene su propia pestaña: lo que se elija en Sistema no toca a Rutas ni
    a Estaciones. Dentro de cada una se ve qué está proyectando, qué candidatos se
    evaluaron y qué entrenamientos anteriores se pueden volver a poner activos.
    """
    st.subheader("4. Modelos por alcance")
    servicio = ServicioEntrenamiento(obtener_settings())
    metadatos = obtener_catalogo().modelos_entrenados()
    guardados = {str(meta["nombre"]): meta for meta in metadatos}

    st.caption(
        "Cada alcance se elige por separado. **Cada entrenamiento queda archivado** con su "
        "nombre `alcance-fecha-hora-r2`, así que puedes volver a cualquiera sin reentrenar."
    )

    pestanas = st.tabs([_TITULO_ALCANCE[alcance] for alcance in ALCANCES])
    for pestana, alcance in zip(pestanas, ALCANCES, strict=True):
        with pestana:
            _panel_de_alcance(servicio, alcance, guardados.get(alcance))

    # Una sola papelera para los tres alcances: lo eliminado se puede recuperar.
    _papelera(servicio)


def _panel_de_alcance(servicio, alcance: str, meta: dict | None) -> None:
    """Muestra y permite cambiar el modelo de un alcance concreto.

    Args:
        servicio: servicio de entrenamiento.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.
        meta: metadatos del paquete activo de ese alcance (o ``None`` si no hay).
    """
    if meta is None:
        st.info(
            f"Todavía no hay modelo para **{_TITULO_ALCANCE[alcance]}**. "
            "Ve a la sección 3 y entrena este alcance."
        )
        return

    _resumen_activo(meta)
    versiones = _historial_de_entrenamientos(servicio, alcance)
    if versiones:
        _elegir_modelo(servicio, alcance, versiones)
        _eliminar_modelo(servicio, alcance, versiones)


def _resumen_activo(meta: dict) -> None:
    """Muestra, arriba y claro, qué modelo está proyectando un alcance."""
    c.fila_kpis(
        [
            {"etiqueta": "Modelo en uso", "valor": c.etiqueta_modelo(meta)},
            {"etiqueta": "Proyecta con", "valor": str(meta.get("extras", {}).get("campeon", "-"))},
            {"etiqueta": "WAPE", "valor": f"{float(meta['metricas'].get('wape', 0)):.2f} %"},
            {"etiqueta": "R²", "valor": f"{float(meta['metricas'].get('r2', 0)):.3f}"},
        ]
    )
    st.caption(
        f"Historia de entrenamiento: {meta.get('desde_entrenamiento')} a "
        f"{meta.get('hasta_entrenamiento')} · entrenado "
        f"{str(meta.get('fecha_entrenamiento'))[:16].replace('T', ' ')}"
    )


def _historial_de_entrenamientos(servicio, alcance: str) -> list[dict]:
    """Lista los entrenamientos archivados del alcance (solo lectura).

    Args:
        servicio: servicio de entrenamiento.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.

    Returns:
        Las versiones archivadas del alcance (lista vacía si no hay).
    """
    versiones = servicio.versiones(alcance)
    st.markdown("###### Historial de entrenamientos")
    if not versiones:
        st.caption("Aún no hay entrenamientos archivados para este alcance.")
        return []

    tabla = pd.DataFrame(
        [
            {
                "Entrenamiento": version["version"],
                "En uso": "Sí" if version["activo"] else "",
                "Entrenado": str(version.get("fecha_entrenamiento"))[:16].replace("T", " "),
                "Historia": f"{version.get('desde_entrenamiento')} a {version.get('hasta_entrenamiento')}",
                "Proyecta con": version.get("extras", {}).get("campeon", "-"),
                "WAPE %": round(float(version["metricas"].get("wape", 0)), 2),
                "R²": round(float(version["metricas"].get("r2", 0)), 4),
                "MB": version.get("tamano_mb", 0),
                "Observación": version.get("extras", {}).get("observacion", "") or "—",
            }
            for version in versiones
        ]
    )
    st.dataframe(tabla, width="stretch", hide_index=True)

    total = sum(float(version.get("tamano_mb", 0)) for version in versiones)
    st.caption(
        f"{len(versiones)} entrenamiento(s) · {total:,.1f} MB en disco. "
        "Elige abajo el entrenamiento y el candidato que debe proyectar."
    )
    return versiones


def _elegir_modelo(servicio, alcance: str, versiones: list[dict]) -> None:
    """Dos listas encadenadas: el entrenamiento y el candidato de ese entrenamiento.

    La estrella marca el campeón (menor MAE) de la versión elegida. Poner en uso
    restaura esa versión y, si se eligió otro candidato, lo ajusta con la historia de
    ese mismo entrenamiento.

    Args:
        servicio: servicio de entrenamiento.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.
        versiones: versiones archivadas del alcance.
    """
    st.markdown("###### Elegir qué proyecta")
    por_etiqueta = {
        (
            f"{version['version']}   ← en uso" if version["activo"] else str(version["version"])
        ): version
        for version in versiones
    }

    col1, col2, col3 = st.columns([3, 2, 1])
    etiqueta_version = col1.selectbox(
        "Entrenamiento", list(por_etiqueta), key=f"sel_version_{alcance}"
    )
    version = por_etiqueta[str(etiqueta_version)]
    comparativa = list(version.get("extras", {}).get("comparativa", []))
    campeon = str(version.get("extras", {}).get("campeon", ""))

    if not comparativa:
        st.info("Ese entrenamiento no tiene candidatos registrados.")
        return

    por_candidato = {
        (
            f"★ {fila['modelo']}" if str(fila["modelo"]) == campeon else f"     {fila['modelo']}"
        ): str(fila["modelo"])
        for fila in comparativa
    }
    etiqueta_candidato = col2.selectbox(
        "Candidato (★ = campeón)",
        list(por_candidato),
        key=f"sel_candidato_{alcance}",
    )
    candidato = por_candidato[str(etiqueta_candidato)]

    ya_esta = bool(version["activo"]) and candidato == campeon
    col3.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    if col3.button("Poner en uso", key=f"btn_usar_{alcance}", disabled=ya_esta, type="primary"):
        with st.spinner("Preparando el modelo..."):
            try:
                if not version["activo"]:
                    servicio.activar_version(str(version["version"]))
                if candidato != campeon:
                    servicio.activar_modelo(alcance, candidato)
            except (ValueError, FileNotFoundError) as error:
                st.error(f"{error}")
                return
        invalidar_cache()
        st.success(
            f"Listo: {_TITULO_ALCANCE[alcance]} proyectará con **{candidato}** "
            f"de `{version['version']}`."
        )
        st.rerun()

    observacion = str(version.get("extras", {}).get("observacion", "") or "").strip()
    if observacion:
        st.info(f"**Observación del entrenamiento:** {observacion}")

    filas = [
        {
            "Candidato": f"★ {fila['modelo']}"
            if str(fila["modelo"]) == campeon
            else str(fila["modelo"]),
            "WAPE %": round(float(fila.get("wape", 0)), 2),
            "MAE": round(float(fila.get("mae", 0)), 1),
            "R²": round(float(fila.get("r2", 0)), 4),
        }
        for fila in comparativa
    ]
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
    enlace = obtener_catalogo().enlace_mlflow(version)
    if enlace:
        st.caption(f"Detalle completo del entrenamiento en [MLflow]({enlace}).")


def _papelera(servicio) -> None:
    """Muestra lo eliminado y permite devolverlo a su sitio.

    Args:
        servicio: servicio de entrenamiento.
    """
    filas = servicio.papelera()
    with st.expander(f"Papelera ({len(filas)})", expanded=False):
        if not filas:
            st.caption("No hay nada eliminado.")
            return
        st.caption(
            "Lo eliminado se conserva en `data/models/papelera`. Restaurar lo devuelve a "
            "su carpeta original."
        )
        for fila in filas:
            col_info, col_boton = st.columns([4, 1])
            col_info.caption(f"`{fila['archivo']}` · {fila['mb']} MB")
            if col_boton.button("Restaurar", key=f"restaurar_{fila['archivo']}"):
                servicio.restaurar_de_papelera(fila["archivo"])
                invalidar_cache()
                st.rerun()


def _eliminar_modelo(servicio, alcance: str, versiones: list[dict]) -> None:
    """Elimina lo que el usuario elija de un alcance: la versión o el modelo activo.

    Un único desplegable con **todo lo que existe** para ese alcance: la versión en uso
    y las archivadas. Si elige la que está en uso, se borra el modelo del alcance (deja
    de proyectar); si elige una archivada, solo se libera ese archivo.

    Args:
        servicio: servicio de entrenamiento.
        alcance: ``sistema``, ``estaciones`` o ``rutas``.
        versiones: versiones archivadas del alcance (incluye la que está en uso).
    """
    etiquetas = {
        f"{version['version']}   ← en uso"
        if version["activo"]
        else str(version["version"]): version
        for version in versiones
    }
    with st.expander("Eliminar…", expanded=False):
        elegido = st.selectbox(
            "¿Qué quieres eliminar de este alcance?",
            list(etiquetas),
            key=f"del_sel_{alcance}",
            help="Se listan todos los entrenamientos guardados de este alcance.",
        )
        fila = etiquetas[str(elegido)]
        confirmar = st.checkbox("Confirmo la eliminación", key=f"chk_borrar_{alcance}")

        if fila["activo"]:
            st.warning(
                f"Vas a eliminar **{fila['version']}**, que es el modelo **en uso** de "
                f"{_TITULO_ALCANCE[alcance]}.\n\n"
                f"- {_TITULO_ALCANCE[alcance]} dejará de poder proyectar (pestañas 2 y 3) "
                "hasta que actives otra versión o reentrenes.\n"
                f"- `data/models/{alcance}.joblib` y `.json` se mueven a la papelera "
                "(`data/models/papelera`), no se destruyen.\n"
                "- Las demás versiones del historial, los eventos, los datos y MLflow "
                "**no** se tocan."
            )
        else:
            st.caption(
                f"`{fila['version']}` ({fila.get('tamano_mb', 0)} MB) se mueve a la "
                "papelera y se puede recuperar. El modelo en uso y los datos no se tocan."
            )

        if st.button(
            "Eliminar",
            type="secondary",
            disabled=not confirmar,
            key=f"btn_borrar_{alcance}",
        ):
            if fila["activo"]:
                try:
                    eliminado = servicio.eliminar_modelo(alcance)
                except ValueError as error:
                    st.error(f"{error}")
                    return
                mensaje = (
                    f"Modelo de {_TITULO_ALCANCE[alcance]} eliminado. "
                    "Activa otra versión del historial o reentrena."
                )
            else:
                eliminado = servicio.eliminar_version(str(fila["version"]))
                mensaje = f"Entrenamiento «{fila['version']}» eliminado."
            invalidar_cache()
            if eliminado:
                st.success(mensaje)
            else:
                st.info("Eso ya no existía.")
            st.rerun()
        if not confirmar:
            st.caption("Marca la casilla de confirmación para habilitar el botón.")
