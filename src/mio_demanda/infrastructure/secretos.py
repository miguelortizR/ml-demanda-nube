"""Secretos locales: hash de contraseñas y cifrado de tokens.

Dos problemas distintos, dos herramientas distintas:

* **Contraseña** (se verifica, nunca se necesita el original): se guarda un **hash**
  PBKDF2-SHA256 con sal aleatoria. No se puede revertir.
* **Token de API** (hay que enviarlo tal cual al proveedor): no se puede hashear; se
  **cifra en reposo** con DPAPI de Windows, que lo ata al usuario de la máquina. Si
  alguien copia el archivo a otro equipo, no puede descifrarlo.

Nada de esto sustituye a la higiene básica: `.env` no se versiona y el archivo se
protege con los permisos del sistema.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import os
import sys
from ctypes import wintypes

#: Iteraciones de PBKDF2 (más lento = más caro de atacar por fuerza bruta).
ITERACIONES_PBKDF2 = 600_000
_ALGORITMO = "pbkdf2_sha256"
_PREFIJO_DPAPI = "dpapi:"


# ----------------------------------------------------------------------
# Contraseñas: hash con sal
# ----------------------------------------------------------------------
def generar_hash(clave: str, iteraciones: int = ITERACIONES_PBKDF2) -> str:
    """Genera el hash de una contraseña, listo para guardar en configuración.

    Args:
        clave: contraseña en claro.
        iteraciones: costo del PBKDF2.

    Returns:
        Cadena ``pbkdf2_sha256$<iteraciones>$<sal>$<huella>``.
    """
    sal = os.urandom(16)
    huella = hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), sal, iteraciones)
    return "$".join(
        (
            _ALGORITMO,
            str(iteraciones),
            base64.b64encode(sal).decode("ascii"),
            base64.b64encode(huella).decode("ascii"),
        )
    )


def verificar_hash(clave: str, guardado: str) -> bool:
    """Comprueba una contraseña contra un hash guardado.

    Args:
        clave: contraseña escrita por el usuario.
        guardado: hash generado por :func:`generar_hash`.

    Returns:
        ``True`` si coinciden. Un hash mal formado devuelve ``False`` (no revienta).
    """
    partes = str(guardado or "").split("$")
    if len(partes) != 4 or partes[0] != _ALGORITMO:
        return False
    try:
        iteraciones = int(partes[1])
        sal = base64.b64decode(partes[2])
        esperado = base64.b64decode(partes[3])
    except (ValueError, TypeError):
        return False
    calculado = hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), sal, iteraciones)
    return hmac.compare_digest(calculado, esperado)


# ----------------------------------------------------------------------
# Tokens: cifrado en reposo con DPAPI (Windows)
# ----------------------------------------------------------------------
class _Blob(ctypes.Structure):
    """Estructura DATA_BLOB de la API de Windows."""

    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _a_blob(datos: bytes) -> tuple[_Blob, ctypes.Array]:
    """Empaqueta bytes en un DATA_BLOB, conservando el búfer vivo."""
    buffer = ctypes.create_string_buffer(datos, len(datos))
    return _Blob(len(datos), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _leer_blob(blob: _Blob) -> bytes:
    """Extrae los bytes de un DATA_BLOB y libera la memoria de Windows."""
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob.pbData)


def dpapi_disponible() -> bool:
    """Indica si este sistema puede cifrar con DPAPI.

    DPAPI existe solo en Windows y está atado al usuario del equipo. Su ausencia es la
    señal más fiable de que la aplicación no corre en el equipo del profesional (por
    ejemplo, en Streamlit Community Cloud).

    Returns:
        ``True`` si ``crypt32`` está disponible en este sistema.
    """
    if sys.platform != "win32":
        return False
    try:
        return hasattr(ctypes, "windll") and hasattr(ctypes.windll, "crypt32")
    except (AttributeError, OSError):
        return False


def cifrar(texto: str) -> str | None:
    """Cifra un texto con DPAPI (atado al usuario actual de Windows).

    Args:
        texto: valor a proteger (por ejemplo, un token de API).

    Returns:
        Cadena ``dpapi:<base64>`` o ``None`` si el sistema no lo permite.
    """
    if sys.platform != "win32" or not texto:
        return None
    try:
        entrada, _ = _a_blob(texto.encode("utf-8"))
        salida = _Blob()
        correcto = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(entrada), None, None, None, None, 0, ctypes.byref(salida)
        )
        if not correcto:
            return None
        return _PREFIJO_DPAPI + base64.b64encode(_leer_blob(salida)).decode("ascii")
    except (AttributeError, OSError):
        return None


def descifrar(guardado: str) -> str | None:
    """Descifra un valor generado por :func:`cifrar`.

    Args:
        guardado: cadena ``dpapi:<base64>``.

    Returns:
        El texto original, o ``None`` si no se puede descifrar (otro usuario, otro
        equipo o datos corruptos).
    """
    if sys.platform != "win32" or not str(guardado or "").startswith(_PREFIJO_DPAPI):
        return None
    try:
        datos = base64.b64decode(str(guardado)[len(_PREFIJO_DPAPI) :])
        entrada, _ = _a_blob(datos)
        salida = _Blob()
        correcto = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(entrada), None, None, None, None, 0, ctypes.byref(salida)
        )
        if not correcto:
            return None
        return _leer_blob(salida).decode("utf-8")
    except (AttributeError, OSError, ValueError):
        return None
