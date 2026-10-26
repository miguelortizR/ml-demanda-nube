"""Punto de entrada para Streamlit Community Cloud.

Deja el paquete `mio_demanda` en el camino de importación y ejecuta la aplicación.
"""

import runpy
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

runpy.run_path(str(RAIZ / "app.py"), run_name="__main__")
