"""Configuración centralizada de logging.

Un único punto para inicializar el logging de la aplicación, de modo que tanto
la CLI como la UI compartan el mismo formato y nivel.
"""

from __future__ import annotations

import logging
import sys

_FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_inicializado = False


def configurar_logging(nivel: str = "INFO") -> None:
    """Inicializa el logging raíz de la aplicación (idempotente).

    Args:
        nivel: nivel de logging (``DEBUG``, ``INFO``, ``WARNING``, ...).
    """
    global _inicializado
    if _inicializado:
        return
    logging.basicConfig(
        level=getattr(logging, str(nivel).upper(), logging.INFO),
        format=_FORMATO,
        stream=sys.stdout,
        force=True,
    )
    _inicializado = True


def obtener_logger(nombre: str) -> logging.Logger:
    """Devuelve un logger con el nombre del módulo solicitante.

    Args:
        nombre: nombre del logger (usualmente ``__name__``).

    Returns:
        El logger configurado.
    """
    configurar_logging()
    return logging.getLogger(nombre)
