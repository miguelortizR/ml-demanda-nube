"""Servicio de catálogo: estado de datos, entidades y modelos disponibles.

Es el caso de uso que consulta la UI para saber qué puede mostrar: si hay datos
extraídos, qué estaciones/rutas existen y qué modelos están entrenados.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.infrastructure.almacen_modelos import AlmacenModelos
from mio_demanda.infrastructure.ingesta import archivos_disponibles, extraer_todo
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos

logger = obtener_logger(__name__)


class ServicioCatalogo:
    """Expone el inventario de datos y modelos a la interfaz.

    Args:
        settings: configuración de la aplicación.
        repo: repositorio de datos (inyectable para pruebas).
        almacen: almacén de modelos (inyectable para pruebas).
    """

    def __init__(
        self,
        settings: Settings,
        repo: RepositorioDatos | None = None,
        almacen: AlmacenModelos | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo or RepositorioDatos(settings)
        self._almacen = almacen or AlmacenModelos(settings)

    def estado_datos(self) -> dict[str, Any]:
        """Resume el estado de la ingesta de datos.

        Returns:
            Diccionario con ``archivos`` presentes, ``hay_datos`` y ``rango``
            (fechas mínima y máxima de usos).
        """
        archivos = archivos_disponibles(self._settings)
        rango = self._repo.rango_fechas_usos()
        return {
            "archivos": {nombre: str(ruta) for nombre, ruta in archivos.items()},
            "hay_datos": self._repo.disponible,
            "rango": rango,
        }

    def listar_entidades(self, tipo: TipoEntidad) -> list[str]:
        """Lista las entidades disponibles del tipo solicitado.

        Args:
            tipo: tipo de entidad.

        Returns:
            Lista ordenada de identificadores.
        """
        if not self._repo.disponible:
            return []
        return self._repo.listar_entidades(tipo)

    def entidades_activas(self, tipo: TipoEntidad) -> list[str]:
        """Entidades que **hoy se pueden proyectar** porque tienen datos recientes.

        El histórico contiene estaciones y rutas retiradas (o con nombres en desuso)
        que ya no registran demanda: proyectarlas no tiene sentido y antes producía
        un error confuso. Solo se ofrecen las que tienen usos en los últimos
        ``dias_actividad_entidad`` días.

        Args:
            tipo: tipo de entidad.

        Returns:
            Lista ordenada de identificadores vigentes.
        """
        rango = self._repo.rango_fechas_usos()
        if rango is None:
            return []
        _, ultimo = rango
        desde = ultimo - timedelta(days=max(1, self._settings.dias_actividad_entidad) - 1)
        return self._repo.entidades_con_datos(tipo, desde, ultimo)

    def modelos_entrenados(self) -> list[dict[str, Any]]:
        """Lista los metadatos de los modelos disponibles.

        Returns:
            Lista de diccionarios de metadatos.
        """
        return self._almacen.listar()

    def enlace_mlflow(self, metadatos: dict[str, Any]) -> str | None:
        """Enlace a la corrida de MLflow de un entrenamiento.

        El análisis detallado (parámetros, curvas y comparación completa) vive en MLflow:
        la app solo ofrece el enlace, para mantenerse ligera. Se usan **solo** los
        identificadores guardados con el modelo: resolverlos aquí obligaría a cargar
        MLflow en cada interacción (costaba casi un segundo por render).

        Args:
            metadatos: metadatos del modelo (``extras`` con los identificadores).

        Returns:
            La dirección web, o ``None`` si no hay interfaz configurada.
        """
        base = str(self._settings.mlflow_ui_url or "").rstrip("/")
        if not base:
            return None

        extras = metadatos.get("extras", {}) or {}
        corrida = extras.get("mlflow_run_id")
        experimento = extras.get("mlflow_experiment_id")

        if corrida and experimento:
            return f"{base}/#/experiments/{experimento}/runs/{corrida}"
        if experimento:
            return f"{base}/#/experiments/{experimento}"
        return f"{base}/#/experiments"

    def extraer_datos(self) -> dict[str, int]:
        """Extrae y normaliza las fuentes crudas a ``data/raw``.

        Returns:
            Diccionario con el número de registros escritos por fuente.
        """
        resultado = extraer_todo(self._settings)
        return resultado.como_diccionario()
