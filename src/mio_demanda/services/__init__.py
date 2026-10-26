"""Capa de servicios (Facade): casos de uso que consume la interfaz.

La UI **no** ejecuta tareas: solo llama a estos servicios y muestra el resultado.
Cada servicio recibe sus dependencias por inyección (constructor).
"""

from __future__ import annotations
