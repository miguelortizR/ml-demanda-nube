"""Capa de aprendizaje automático: datasets, modelos, evaluación y entrenamiento.

Separación interna:

* ``datasets``    — construye matrices de diseño (X) y objetivos (y).
* ``modelos``     — fábrica de candidatos y ensamble por pesos.
* ``evaluacion``  — validación cruzada temporal y métricas de regresión.
* ``tuning``      — búsqueda de hiperparámetros con Optuna.
* ``entrenamiento`` — orquestador del ciclo completo.
* ``prediccion``  — proyecta demanda a futuro con un paquete guardado.
"""

from __future__ import annotations
