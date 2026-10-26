"""Servicio de acceso: valida usuarios y contraseñas de la aplicación.

Las claves nunca se guardan en claro: se comparan contra su hash PBKDF2, venga de
``Settings`` (local) o de ``st.secrets`` (Streamlit Community Cloud).
"""

from __future__ import annotations

import hmac

from mio_demanda.config import Settings
from mio_demanda.infrastructure.secretos import verificar_hash


class ServicioAcceso:
    """Comprueba las credenciales de quien entra a la aplicación.

    Args:
        settings: configuración con los usuarios y sus hashes.
        usuarios_extra: usuarios adicionales, por ejemplo los de ``st.secrets`` en la
            nube. Tienen prioridad sobre los de la configuración.
    """

    def __init__(
        self, settings: Settings, usuarios_extra: dict[str, str] | None = None
    ) -> None:
        extra = {str(usuario): str(clave) for usuario, clave in (usuarios_extra or {}).items()}
        self._usuarios = {**settings.credenciales(), **extra}
        self._admins = {str(usuario).strip() for usuario in settings.usuarios_admin}

    def usuarios(self) -> list[str]:
        """Usuarios habilitados, en orden alfabético (nunca sus hashes).

        Returns:
            Lista de nombres de usuario.
        """
        return sorted(self._usuarios)

    def hay_usuarios(self) -> bool:
        """Indica si la aplicación tiene al menos un usuario configurado.

        Returns:
            ``True`` si hay algún acceso definido.
        """
        return bool(self._usuarios)

    def rol(self, usuario: str | None) -> str:
        """Rol del usuario en la aplicación.

        Args:
            usuario: nombre del usuario autenticado.

        Returns:
            ``"administrador"`` o ``"consulta"``.
        """
        nombre = str(usuario or "").strip()
        return "administrador" if nombre in self._admins else "consulta"

    def es_admin(self, usuario: str | None) -> bool:
        """Indica si un usuario puede entrar a la pestaña de entrenamiento.

        Args:
            usuario: nombre del usuario autenticado.

        Returns:
            ``True`` solo si está en la lista de administradores.
        """
        return bool(usuario) and str(usuario).strip() in self._admins

    def autenticar(self, usuario: str, clave: str) -> bool:
        """Comprueba un par usuario/contraseña.

        Args:
            usuario: nombre escrito en el formulario.
            clave: contraseña escrita en el formulario.

        Returns:
            ``True`` solo si el usuario existe y la clave coincide con su hash.
        """
        guardado = self._usuarios.get(usuario.strip())
        if not guardado:
            return False
        if guardado.startswith("pbkdf2"):
            return verificar_hash(clave, guardado)
        return hmac.compare_digest(clave, guardado)
