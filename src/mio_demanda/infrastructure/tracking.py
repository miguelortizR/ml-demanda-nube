"""Seguimiento de experimentos con MLflow (tracking local).

Envuelve MLflow tras una interfaz mínima y **degradable**: si el servidor de
tracking no está disponible (o una DLL nativa está bloqueada por Windows), la
aplicación sigue funcionando y solo se registra una advertencia.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mio_demanda.config import Settings
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)


class RastreadorMLflow:
    """Cliente de MLflow con degradación elegante.

    Args:
        settings: configuración con el *tracking URI* y el nombre del experimento.
    """

    def __init__(self, settings: Settings) -> None:
        self._uri = settings.mlflow_tracking_uri
        self._experimento = settings.mlflow_experiment
        self._mlflow: Any = None
        self._disponible = self._inicializar()

    def _inicializar(self) -> bool:
        """Configura MLflow; devuelve ``False`` si no está disponible."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self._experimento)
            self._mlflow = mlflow
            logger.info(
                "MLflow listo. Tracking: %s | Experimento: %s", self._uri, self._experimento
            )
            return True
        except Exception as error:
            logger.warning("MLflow no disponible (%s). Se continúa sin tracking.", error)
            return False

    @property
    def disponible(self) -> bool:
        """Indica si el tracking con MLflow está operativo."""
        return self._disponible

    @property
    def run_id(self) -> str | None:
        """Identificador del *run* activo (``None`` si no hay ninguno abierto)."""
        if not self._disponible:
            return None
        activo = self._mlflow.active_run()
        return str(activo.info.run_id) if activo is not None else None

    @property
    def experiment_id(self) -> str | None:
        """Identificador numérico del experimento configurado.

        Se usa para construir el enlace a la interfaz web de MLflow.
        """
        if not self._disponible:
            return None
        try:
            experimento = self._mlflow.get_experiment_by_name(self._experimento)
        except Exception as error:
            logger.warning("No se pudo resolver el experimento de MLflow: %s", error)
            return None
        return str(experimento.experiment_id) if experimento is not None else None

    @contextmanager
    def run(
        self,
        nombre: str,
        parametros: dict[str, Any] | None = None,
        etiquetas: dict[str, str] | None = None,
    ) -> Iterator[None]:
        """Abre un *run* de MLflow (o un contexto vacío si no está disponible).

        Args:
            nombre: nombre del run.
            parametros: hiperparámetros a registrar.
            etiquetas: etiquetas del run.

        Yields:
            ``None``; se usa solo por su efecto de contexto.
        """
        if not self._disponible:
            yield
            return

        with self._mlflow.start_run(run_name=nombre):
            if parametros:
                self.registrar_parametros(parametros)
            if etiquetas:
                self._mlflow.set_tags(etiquetas)
            yield

    def registrar_parametros(self, parametros: dict[str, Any]) -> None:
        """Registra hiperparámetros en el run activo."""
        if not self._disponible:
            return
        self._mlflow.log_params({clave: _serializar(valor) for clave, valor in parametros.items()})

    def registrar_metricas(self, metricas: dict[str, float], paso: int | None = None) -> None:
        """Registra métricas numéricas en el run activo.

        Args:
            metricas: mapa nombre -> valor.
            paso: paso opcional (para curvas de aprendizaje).
        """
        if not self._disponible:
            return
        limpias = {clave: float(valor) for clave, valor in metricas.items() if _es_numero(valor)}
        self._mlflow.log_metrics(limpias, step=paso)

    def registrar_artefacto(self, ruta: Path) -> None:
        """Sube un archivo como artefacto del run activo."""
        if not self._disponible or not Path(ruta).exists():
            return
        self._mlflow.log_artifact(str(ruta))

    def registrar_metricas_por_grupo(self, prefijo: str, metricas: dict[str, float]) -> None:
        """Registra métricas aplicando un prefijo (p. ej. ``cv_`` o ``test_``).

        Args:
            prefijo: prefijo a anteponer al nombre de cada métrica.
            metricas: mapa nombre -> valor.
        """
        self.registrar_metricas({f"{prefijo}{clave}": valor for clave, valor in metricas.items()})


def _es_numero(valor: Any) -> bool:
    """Indica si un valor es numérico y no nulo."""
    return isinstance(valor, (int, float)) and valor == valor  # descarta NaN


def _serializar(valor: Any) -> Any:
    """Convierte un valor a un tipo aceptado por MLflow (str/int/float/bool)."""
    if isinstance(valor, (str, int, float, bool)):
        return valor
    return str(valor)
