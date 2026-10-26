"""Puerta de acceso de la aplicación.

Sin usuario autenticado no se dibuja ninguna pestaña: la app puede estar publicada
en una dirección abierta de Internet y el contenido es interno.
"""

from __future__ import annotations

import streamlit as st

from mio_demanda.ui.estado import obtener_acceso

#: Clave de sesión con el usuario autenticado.
CLAVE_SESION = "usuario_autenticado"
#: Intentos fallidos antes de bloquear el formulario en esta sesión.
MAX_INTENTOS = 5


def usuario_actual() -> str | None:
    """Usuario autenticado en esta sesión.

    Returns:
        El nombre del usuario, o ``None`` si nadie ha entrado.
    """
    return st.session_state.get(CLAVE_SESION)


def cerrar_sesion() -> None:
    """Olvida el usuario de esta sesión."""
    st.session_state.pop(CLAVE_SESION, None)


def exigir_acceso() -> str:
    """Devuelve el usuario autenticado y, si no lo hay, muestra el formulario.

    Detiene la ejecución (`st.stop`) hasta que alguien entre con credenciales válidas.

    Returns:
        El nombre del usuario autenticado.
    """
    usuario = usuario_actual()
    if usuario:
        return str(usuario)

    servicio = obtener_acceso()
    _, centro, _ = st.columns([1, 1.1, 1])
    with centro:
        st.markdown("#### Acceso")
        st.caption("Aplicación interna de la Oficina de Evaluación. Ingresa con tu usuario.")

        if not servicio.hay_usuarios():
            st.error(
                "No hay usuarios configurados. Define `MIO_USUARIOS_ACCESO` en `.env` o "
                "la sección `[usuarios]` en los secretos de Streamlit."
            )
            st.stop()

        intentos = int(st.session_state.get("intentos_acceso", 0))
        if intentos >= MAX_INTENTOS:
            st.error("Demasiados intentos fallidos. Recarga la página para volver a intentar.")
            st.stop()

        with st.form("acceso_app"):
            nombre = st.text_input("Usuario", key="acceso_usuario")
            clave = st.text_input("Contraseña", type="password", key="acceso_clave")
            entrar = st.form_submit_button("Entrar", type="primary")

        if entrar:
            if servicio.autenticar(nombre, clave):
                st.session_state[CLAVE_SESION] = nombre.strip()
                st.session_state.pop("intentos_acceso", None)
                st.rerun()
            else:
                st.session_state["intentos_acceso"] = intentos + 1
                st.error("Usuario o contraseña incorrectos.")

    st.stop()
