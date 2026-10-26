"""Punto de entrada de la aplicación Streamlit de ML-Demanda.

Ejecución::

    make app                 # o
    uv run streamlit run app.py

La aplicación tiene tres pestañas:

1. **Entrenamiento** (profesional): extrae datos, gestiona el calendario, entrena
   y guarda el modelo campeón.
2. **Proyección general**: demanda diaria del sistema completo.
3. **Proyección por estaciones/rutas**: demanda de una entidad concreta.

Toda la lógica vive en los servicios (Facade); este archivo solo dispone la
interfaz.
"""

from __future__ import annotations

import streamlit as st

from mio_demanda import config as _config
from mio_demanda.ui import componentes as c
from mio_demanda.ui.acceso import cerrar_sesion, exigir_acceso
from mio_demanda.ui.estado import (
    obtener_acceso,
    obtener_proyeccion,
    obtener_settings,
    recargar_aplicacion,
)
from mio_demanda.ui.tab_entrenamiento import render as render_entrenamiento
from mio_demanda.ui.tab_proyeccion_entidad import render as render_proyeccion_entidad
from mio_demanda.ui.tab_proyeccion_general import render as render_proyeccion_general

st.set_page_config(
    page_title="ML-Demanda | MIO Metrocali",
    page_icon=":material/directions_bus:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#: Versión del esquema de configuración que espera **este** archivo. Si el módulo
#: cargado en memoria es más viejo, Streamlit dejó código anterior: se avisa y se
#: para, en vez de reventar a mitad de una pestaña con un AttributeError.
CONFIG_VERSION_ESPERADA = 4
_config_cargada = getattr(_config, "CONFIG_VERSION", 0)

if _config_cargada != CONFIG_VERSION_ESPERADA:
    st.error(
        "La aplicación tiene **código anterior cargado en memoria** "
        f"(configuración v{_config_cargada}, se esperaba v{CONFIG_VERSION_ESPERADA}).\n\n"
        "Cierra la ventana de Streamlit y ejecuta `make app` de nuevo."
    )
    st.stop()

c.inyectar_estilos()
usuario = exigir_acceso()
ajustes = obtener_settings()
c.barra_identidad("ML-Demanda", "Proyección de demanda · MIO Metrocali")

with st.sidebar:
    st.markdown("## ML-Demanda")
    st.caption("Proyección de demanda del MIO con aprendizaje automático y ensambles.")
    rol_sesion = obtener_acceso().rol(usuario)
    st.caption(f"Sesión: **{usuario}** · {rol_sesion} · Modo: **{ajustes.modo_efectivo}**")
    if st.button("Cerrar sesión", key="btn_cerrar_sesion"):
        cerrar_sesion()
        st.rerun()
    st.divider()
    st.markdown("**Modelos en uso**")
    modelos = obtener_proyeccion().modelos_en_uso()
    if not modelos:
        st.caption("Todavía no hay modelos entrenados.")
    else:
        for info in modelos:
            st.caption(
                f"**{info['alcance']}** · WAPE {float(info['metricas'].get('wape', 0)):.2f} %"
            )
    st.divider()
    st.caption("Aplicación interna de la Oficina de Evaluación · Metrocali")
    st.caption("Los datos crudos se leen en modo solo lectura.")
    if st.button("Recargar aplicación", key="btn_recargar"):
        recargar_aplicacion()
        st.rerun()

# En la nube solo se proyecta: el entrenamiento se hace en el equipo del profesional
# y viaja como paquete de modelo. Cada pestaña aparece solo si su modelo existe, para
# que la app funcione igual con uno, con dos o con los tres.
en_nube = ajustes.es_nube
disponibles = obtener_proyeccion().modelos_disponibles()

pestanas: list[tuple[str, callable]] = []
rol = obtener_acceso().rol(usuario)
if not en_nube and rol == "administrador":
    pestanas.append(("1. Entrenamiento", render_entrenamiento))
if disponibles["sistema"]:
    pestanas.append(("2. Proyección general", render_proyeccion_general))
if disponibles["estaciones"] or disponibles["rutas"]:
    pestanas.append(("3. Proyección por estaciones/rutas", render_proyeccion_entidad))

if not pestanas:
    st.error(
        "No hay modelos entrenados en `data/models`. Entrena desde la app local o "
        "activa una versión del historial."
    )
    st.stop()

for contenedor, (_, dibujar) in zip(
    st.tabs([etiqueta for etiqueta, _ in pestanas]), pestanas, strict=True
):
    with contenedor:
        dibujar()
