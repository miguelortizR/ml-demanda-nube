"""Paquete raíz de la aplicación ML-Demanda.

Arquitectura por capas (ver ``docs/desacople.md``)::

    UI (Streamlit) -> Servicios/Facade -> Dominio -> Infraestructura

El dominio es puro (sin framework, sin IO). La infraestructura conoce los
frameworks y las fuentes de datos. La UI solo captura entrada y muestra salida.
"""

from __future__ import annotations

__version__ = "0.3.0"
