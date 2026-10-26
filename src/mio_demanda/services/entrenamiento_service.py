"""Servicio de entrenamiento: casos de uso que expone la pestaña profesional.

Envuelve al :class:`~mio_demanda.ml.entrenamiento.EntrenadorModelo` y traduce el
alcance solicitado (sistema / estaciones / rutas / todo) a llamadas concretas.
La UI solo invoca ``entrenar`` y muestra el resultado.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.infrastructure.almacen_modelos import PaqueteModelo
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.ml.entrenamiento import EntrenadorModelo, ResultadoEntrenamiento

logger = obtener_logger(__name__)

ALCANCES = ("sistema", "estaciones", "rutas")


class ServicioEntrenamiento:
    """Ejecuta los entrenamientos disponibles y devuelve sus resultados.

    Args:
        settings: configuración de la aplicación.
        entrenador: entrenador inyectable (se crea si no se pasa).
        notificar: callback de progreso ``(mensaje: str) -> None``.
    """

    def __init__(
        self,
        settings: Settings,
        entrenador: EntrenadorModelo | None = None,
        notificar: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._entrenador = entrenador or EntrenadorModelo(settings, notificar=notificar)

    def entrenar(
        self,
        alcance: str,
        desde: date,
        hasta: date,
        n_trials: int | None = None,
        n_splits: int | None = None,
        observacion: str = "",
    ) -> ResultadoEntrenamiento:
        """Entrena el alcance solicitado.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.
            desde: fecha inicial de la historia de entrenamiento.
            hasta: fecha final de la historia de entrenamiento.
            n_trials: presupuesto de Optuna por candidato.
            n_splits: particiones de la validación cruzada.
            observacion: nota del usuario sobre por qué entrenó así.

        Returns:
            El :class:`ResultadoEntrenamiento` correspondiente.

        Raises:
            ValueError: si el alcance no es válido.
        """
        if alcance == "sistema":
            return self._entrenador.entrenar_sistema(
                desde, hasta, n_trials=n_trials, n_splits=n_splits, observacion=observacion
            )
        if alcance == "estaciones":
            return self._entrenador.entrenar_entidad(
                TipoEntidad.ESTACION,
                desde,
                hasta,
                n_trials=n_trials,
                n_splits=n_splits,
                max_filas_tuning=self._settings.max_filas_tuning_entidad,
                observacion=observacion,
            )
        if alcance == "rutas":
            return self._entrenador.entrenar_entidad(
                TipoEntidad.RUTA,
                desde,
                hasta,
                n_trials=n_trials,
                n_splits=n_splits,
                max_filas_tuning=self._settings.max_filas_tuning_entidad,
                observacion=observacion,
            )
        raise ValueError(f"Alcance no soportado: {alcance}")

    def activar_modelo(self, alcance: str, modelo: str) -> PaqueteModelo:
        """Elige qué modelo guardado se usará para proyectar un alcance.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.
            modelo: nombre del candidato (``"ensamble"`` o el de un candidato).

        Returns:
            El paquete actualizado.

        Raises:
            ValueError: si el alcance no es válido.
        """
        if alcance not in ALCANCES:
            raise ValueError(f"Alcance no soportado: {alcance}")
        return self._entrenador.activar_modelo(alcance, modelo)

    def eliminar_modelo(self, alcance: str) -> bool:
        """Elimina el modelo guardado de un alcance.

        Las pestañas de proyección dejarán de poder proyectar ese alcance hasta que
        se vuelva a entrenar.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.

        Returns:
            ``True`` si el modelo existía y fue eliminado.

        Raises:
            ValueError: si el alcance no es válido.
        """
        if alcance not in ALCANCES:
            raise ValueError(f"Alcance no soportado: {alcance}")
        return self._entrenador.eliminar_modelo(alcance)

    def versiones(self, alcance: str) -> list[dict[str, Any]]:
        """Entrenamientos archivados de un alcance, el más reciente primero.

        Args:
            alcance: ``"sistema"``, ``"estaciones"`` o ``"rutas"``.

        Returns:
            Lista de metadatos con ``version``, ``activo``, ``tamano_mb`` y métricas.

        Raises:
            ValueError: si el alcance no es válido.
        """
        if alcance not in ALCANCES:
            raise ValueError(f"Alcance no soportado: {alcance}")
        return self._entrenador.listar_versiones(alcance)

    def activar_version(self, version: str) -> PaqueteModelo:
        """Pone a proyectar un entrenamiento archivado (sin reentrenar).

        Args:
            version: nombre identificable del entrenamiento
                (``sistema-2026-10-16_14-33-r2-0.950``).

        Returns:
            El paquete restaurado.

        Raises:
            FileNotFoundError: si la versión no existe.
        """
        return self._entrenador.activar_version(version)

    def papelera(self) -> list[dict[str, Any]]:
        """Lista lo eliminado que aún se puede recuperar.

        Returns:
            Lista de diccionarios con ``archivo``, ``origen`` y ``mb``.
        """
        return self._entrenador.listar_papelera()

    def restaurar_de_papelera(self, archivo: str) -> bool:
        """Devuelve un archivo de la papelera a su sitio.

        Args:
            archivo: nombre del archivo en la papelera.

        Returns:
            ``True`` si se restauró.
        """
        return self._entrenador.restaurar_de_papelera(archivo)

    def eliminar_version(self, version: str) -> bool:
        """Elimina un entrenamiento del historial (no toca el que está activo).

        Args:
            version: nombre identificable del entrenamiento.

        Returns:
            ``True`` si existía y se eliminó.
        """
        return self._entrenador.eliminar_version(version)
