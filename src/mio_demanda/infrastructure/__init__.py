"""Capa de infraestructura: acceso a datos, persistencia y servicios externos.

Aquí viven las dependencias tecnológicas (DuckDB, Parquet, MLflow, sistema de
archivos). El dominio y los servicios no conocen estas implementaciones.
"""

from __future__ import annotations
