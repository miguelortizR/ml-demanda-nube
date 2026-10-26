"""Recursos compartidos y cacheados de la interfaz.

Aquí se crean (una sola vez) la configuración y los servicios, y se cachean las
consultas costosas. La UI **no** construye lógica: solo obtiene servicios listos.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from mio_demanda import __version__
from mio_demanda.config import Settings, get_settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.services.acceso_service import ServicioAcceso
from mio_demanda.services.calendario_service import ServicioCalendario
from mio_demanda.services.catalogo_service import ServicioCatalogo
from mio_demanda.services.proyeccion_service import ServicioProyeccion
from mio_demanda.services.reporte_service import ServicioReporte
from mio_demanda.services.segunda_opinion_service import ServicioSegundaOpinion

# Las funciones de servicio se cachean con la versión del paquete como clave, de
# modo que al cambiar el código de un servicio la caché se invalida sola (si no,
# Streamlit reutilizaría instancias de la clase anterior).


@st.cache_resource
def obtener_settings(version: str = __version__) -> Settings:
    """Devuelve la configuración única de la aplicación (cacheada)."""
    return get_settings()


@st.cache_resource
def obtener_catalogo(version: str = __version__) -> ServicioCatalogo:
    """Devuelve el servicio de catálogo (cacheado)."""
    return ServicioCatalogo(obtener_settings())


@st.cache_resource
def obtener_calendario(version: str = __version__) -> ServicioCalendario:
    """Devuelve el servicio de calendario (cacheado)."""
    return ServicioCalendario(obtener_settings())


@st.cache_resource
def obtener_proyeccion(version: str = __version__) -> ServicioProyeccion:
    """Devuelve el servicio de proyección (cacheado)."""
    return ServicioProyeccion(obtener_settings())


@st.cache_resource
def obtener_reporte(version: str = __version__) -> ServicioReporte:
    """Devuelve el servicio de reporte de Excel (cacheado)."""
    return ServicioReporte(obtener_settings())


@st.cache_resource
def obtener_segunda_opinion(version: str = __version__) -> ServicioSegundaOpinion:
    """Devuelve el servicio de segunda opinión (Jev), cacheado."""
    return ServicioSegundaOpinion(obtener_settings())


@st.cache_data(ttl=300, show_spinner=False)
def listar_entidades(tipo_valor: str) -> list[str]:
    """Lista entidades (caché de 5 minutos).

    Args:
        tipo_valor: valor del enum ``TipoEntidad``.

    Returns:
        Lista de entidades disponibles.
    """
    return obtener_catalogo().listar_entidades(TipoEntidad(tipo_valor))


@st.cache_data(ttl=300, show_spinner=False)
def _entidades_activas_cache(tipo_valor: str) -> list[str]:
    """Consulta cacheada de entidades vigentes."""
    return obtener_catalogo().entidades_activas(TipoEntidad(tipo_valor))


def entidades_activas(tipo_valor: str) -> list[str]:
    """Entidades con datos recientes, aptas para proyectar.

    Una respuesta **vacía no se cachea**: si la consulta se hizo mientras los datos se
    estaban actualizando, el siguiente render vuelve a intentarlo en vez de mostrar la
    lista vacía durante cinco minutos.

    Args:
        tipo_valor: valor del enum ``TipoEntidad``.

    Returns:
        Lista de entidades vigentes.
    """
    resultado = _entidades_activas_cache(tipo_valor)
    if not resultado:
        _entidades_activas_cache.clear()
    return resultado


def limpiar_entidades() -> None:
    """Descarta la caché de entidades para forzar una consulta nueva."""
    _entidades_activas_cache.clear()


@st.cache_data(ttl=60, show_spinner=False)
def estado_datos() -> dict[str, Any]:
    """Devuelve el estado de la ingesta (caché de 1 minuto)."""
    return obtener_catalogo().estado_datos()


def rango_datos_disponibles() -> tuple[date, date]:
    """Devuelve el rango de fechas con datos o un rango por defecto.

    Returns:
        Tupla ``(fecha_minima, fecha_maxima)``.
    """
    estado = estado_datos()
    if estado.get("rango"):
        return estado["rango"]
    return date(date.today().year - 1, 1, 1), date.today()


def invalidar_cache() -> None:
    """Limpia las cachés de datos (tras extraer o entrenar)."""
    st.cache_data.clear()


def recargar_aplicacion() -> None:
    """Limpia todas las cachés (datos y servicios) y vuelve a ejecutar la app."""
    st.cache_data.clear()
    st.cache_resource.clear()


def _usuarios_de_secretos() -> dict[str, str]:
    """Usuarios definidos en ``st.secrets`` (los de la nube).

    Returns:
        Mapa ``usuario -> hash``; vacío si no hay secretos configurados.
    """
    try:
        usuarios = st.secrets.get("usuarios", {})
    except Exception:  # sin archivo de secretos (entorno local)
        return {}
    return {str(usuario): str(clave) for usuario, clave in dict(usuarios).items()}


def obtener_acceso(version: str = __version__) -> ServicioAcceso:
    """Devuelve el servicio de acceso, con los usuarios de ``st.secrets`` si existen.

    Args:
        version: versión de la app (invalida la caché al cambiar de código).

    Returns:
        El servicio de acceso listo para autenticar.
    """
    return ServicioAcceso(obtener_settings(), _usuarios_de_secretos())
