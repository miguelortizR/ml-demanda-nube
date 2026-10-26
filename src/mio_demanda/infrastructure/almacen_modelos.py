"""Persistencia de modelos entrenados (paquetes y metadatos).

Guarda cada modelo como un *paquete* autocontenido: el objeto entrenado (joblib)
más un JSON de metadatos que permite listar y auditar modelos **sin** cargarlos.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib

from mio_demanda.config import Settings
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)


def _serializar_json(valor: Any) -> Any:
    """Convierte tipos de NumPy a tipos serializables por JSON.

    Args:
        valor: valor no serializable hallado por ``json.dumps``.

    Returns:
        Una representación serializable (escalar o lista).

    Raises:
        TypeError: si el tipo no se puede convertir.
    """
    if hasattr(valor, "item"):
        return valor.item()
    if hasattr(valor, "tolist"):
        return valor.tolist()
    raise TypeError(f"Tipo no serializable: {type(valor).__name__}")


@dataclass(slots=True)
class PaqueteModelo:
    """Modelo entrenado con su contrato de entrada y sus metadatos.

    Attributes:
        nombre: identificador del paquete (p. ej. ``sistema`` o ``entidad_T31``).
        alcance: ``"sistema"`` o ``"entidad"``.
        algoritmo: descripción legible del modelo o ensamble.
        features: lista ordenada de variables de entrada que espera el modelo.
        metricas: métricas de evaluación cruzada (MAE, MAPE, R², ...).
        desde_entrenamiento: fecha mínima de la historia usada.
        hasta_entrenamiento: fecha máxima de la historia usada.
        fecha_entrenamiento: momento en que se entrenó.
        extras: información adicional específica del modelo (pesos, parámetros).
        modelo: objeto entrenado (se serializa con joblib, no aparece en el JSON).
    """

    nombre: str
    alcance: str
    algoritmo: str
    features: list[str]
    metricas: dict[str, float]
    desde_entrenamiento: date
    hasta_entrenamiento: date
    fecha_entrenamiento: datetime = field(default_factory=datetime.now)
    extras: dict[str, Any] = field(default_factory=dict)
    modelo: Any = None

    def metadatos(self) -> dict[str, Any]:
        """Devuelve los metadatos serializables (sin el objeto modelo)."""
        datos = asdict(self)
        datos.pop("modelo", None)
        datos["desde_entrenamiento"] = self.desde_entrenamiento.isoformat()
        datos["hasta_entrenamiento"] = self.hasta_entrenamiento.isoformat()
        datos["fecha_entrenamiento"] = self.fecha_entrenamiento.isoformat()
        return datos


class AlmacenModelos:
    """Repositorio de paquetes de modelo en ``data/models``.

    Cada alcance tiene un paquete **activo** (``sistema.joblib``, ``estaciones.joblib``,
    ``rutas.joblib``): es el que consumen las pestañas de proyección. Además, **cada
    entrenamiento se archiva** en ``data/models/historial`` con un nombre
    identificable (``rutas-2026-10-16_14-33-r2-0.957``), de modo que se pueda volver a
    cualquiera de ellos sin reentrenar. Entrenar de nuevo ya **no** pierde lo anterior.

    Args:
        settings: configuración con la carpeta de modelos.
    """

    _HISTORIAL = "historial"

    def __init__(self, settings: Settings) -> None:
        self._carpeta = settings.models_dir
        self._carpeta.mkdir(parents=True, exist_ok=True)

    def _ruta_modelo(self, nombre: str) -> Path:
        return self._carpeta / f"{nombre}.joblib"

    def _ruta_metadatos(self, nombre: str) -> Path:
        return self._carpeta / f"{nombre}.json"

    def _carpeta_historial(self) -> Path:
        """Carpeta donde se archiva cada entrenamiento (se crea si no existe)."""
        ruta = self._carpeta / self._HISTORIAL
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    @staticmethod
    def nombre_version(metadatos: dict[str, Any]) -> str:
        """Nombre identificable de un entrenamiento: ``alcance-fecha-hora-r2``.

        Args:
            metadatos: metadatos del modelo (``nombre``, ``fecha_entrenamiento``,
                ``metricas``).

        Returns:
            El nombre de la versión (también válido como nombre de archivo).
        """
        crudo = str(metadatos.get("fecha_entrenamiento", ""))
        try:
            sello = datetime.fromisoformat(crudo).strftime("%Y-%m-%d_%H-%M-%S")
        except ValueError:
            sello = crudo[:19].replace(":", "-").replace(" ", "_") or "sin-fecha"
        metricas = metadatos.get("metricas") or {}
        r2 = float(metricas.get("r2", 0.0) or 0.0)
        return f"{metadatos.get('nombre')}-{sello}-r2-{r2:.3f}"

    def guardar(self, paquete: PaqueteModelo) -> Path:
        """Guarda el modelo y sus metadatos de forma **atómica**, y lo archiva.

        Escribe primero archivos temporales y luego los reemplaza, de modo que una
        lectura simultánea (p. ej. la app proyectando mientras se entrena) nunca vea
        un paquete a medio escribir. Además copia el paquete al historial para no
        perder el entrenamiento.

        Args:
            paquete: paquete de modelo a persistir.

        Returns:
            La ruta del archivo de modelo guardado.
        """
        ruta_modelo = self._ruta_modelo(paquete.nombre)
        ruta_metadatos = self._ruta_metadatos(paquete.nombre)
        self._carpeta.mkdir(parents=True, exist_ok=True)
        metadatos = paquete.metadatos()

        temporal_modelo = ruta_modelo.with_name(f"{ruta_modelo.name}.tmp")
        temporal_metadatos = ruta_metadatos.with_name(f"{ruta_metadatos.name}.tmp")

        joblib.dump(paquete.modelo, temporal_modelo)
        temporal_metadatos.write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False, default=_serializar_json),
            encoding="utf-8",
        )
        # Los metadatos primero (llevan el nombre y las métricas) y luego el modelo.
        temporal_metadatos.replace(ruta_metadatos)
        temporal_modelo.replace(ruta_modelo)
        self._archivar(metadatos, ruta_modelo)
        logger.info("Modelo guardado: %s -> %s", paquete.nombre, ruta_modelo)
        return ruta_modelo

    def _archivar(self, metadatos: dict[str, Any], ruta_modelo: Path) -> Path:
        """Copia un paquete al historial de entrenamientos.

        Args:
            metadatos: metadatos del paquete (incluyen alcance, métricas y extras).
            ruta_modelo: ruta del joblib activo que se va a copiar.

        Returns:
            La ruta de los metadatos archivados.
        """
        destino_metadatos = self._carpeta_historial() / f"{self.nombre_version(metadatos)}.json"
        destino_modelo = destino_metadatos.with_suffix(".joblib")
        shutil.copy2(ruta_modelo, destino_modelo)
        temporal = destino_metadatos.with_name(f"{destino_metadatos.name}.tmp")
        temporal.write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False, default=_serializar_json),
            encoding="utf-8",
        )
        temporal.replace(destino_metadatos)
        return destino_metadatos

    # ------------------------------------------------------------------
    # Historial de entrenamientos
    # ------------------------------------------------------------------
    def listar_versiones(self, alcance: str | None = None) -> list[dict[str, Any]]:
        """Lista los entrenamientos archivados, el más reciente primero.

        Args:
            alcance: ``sistema``, ``estaciones`` o ``rutas``; ``None`` para todos.

        Returns:
            Lista de metadatos, cada uno con ``version`` (nombre identificable),
            ``activo`` (si es el que está proyectando) y ``tamano_mb``.
        """
        carpeta = self._carpeta_historial()
        # La identidad de una versión es (alcance, momento de entrenamiento): así se
        # reconoce la que está en uso aunque cambie el formato del nombre.
        activos = {
            (str(meta.get("nombre")), str(meta.get("fecha_entrenamiento")))
            for meta in self.listar()
        }
        entradas: list[dict[str, Any]] = []
        for ruta in carpeta.glob("*.json"):
            try:
                meta = dict(json.loads(ruta.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as error:
                logger.warning("Metadatos de historial ilegibles %s: %s", ruta, error)
                continue
            modelo = ruta.with_suffix(".joblib")
            meta["version"] = ruta.stem
            meta["activo"] = (
                str(meta.get("nombre")),
                str(meta.get("fecha_entrenamiento")),
            ) in activos
            meta["tamano_mb"] = (
                round((ruta.stat().st_size + modelo.stat().st_size) / 1_048_576, 1)
                if modelo.exists()
                else round(ruta.stat().st_size / 1_048_576, 1)
            )
            entradas.append(meta)

        if alcance:
            entradas = [entrada for entrada in entradas if entrada.get("nombre") == alcance]
        return sorted(
            entradas, key=lambda meta: str(meta.get("fecha_entrenamiento", "")), reverse=True
        )

    def activar_version(self, version: str) -> PaqueteModelo:
        """Convierte una versión del historial en el paquete activo del alcance.

        Args:
            version: nombre identificable de la versión (``sistema-2026-....``).

        Returns:
            El :class:`PaqueteModelo` restaurado.

        Raises:
            FileNotFoundError: si la versión o sus metadatos no existen.
        """
        carpeta = self._carpeta_historial()
        origen_modelo = carpeta / f"{version}.joblib"
        origen_metadatos = carpeta / f"{version}.json"
        if not origen_modelo.exists() or not origen_metadatos.exists():
            raise FileNotFoundError(f"No existe la versión '{version}' en el historial")

        metadatos = json.loads(origen_metadatos.read_text(encoding="utf-8"))
        nombre = str(metadatos["nombre"])
        destino_modelo = self._ruta_modelo(nombre)
        destino_metadatos = self._ruta_metadatos(nombre)

        temporal_modelo = destino_modelo.with_name(f"{destino_modelo.name}.tmp")
        temporal_metadatos = destino_metadatos.with_name(f"{destino_metadatos.name}.tmp")
        shutil.copy2(origen_modelo, temporal_modelo)
        temporal_metadatos.write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False, default=_serializar_json),
            encoding="utf-8",
        )
        temporal_metadatos.replace(destino_metadatos)
        temporal_modelo.replace(destino_modelo)
        logger.info("Versión activada: %s -> %s", version, destino_modelo)
        return self.cargar(nombre)

    def eliminar_version(self, version: str) -> bool:
        """Elimina una versión del historial (no toca el paquete activo).

        Args:
            version: nombre identificable de la versión.

        Returns:
            ``True`` si se eliminó algo.
        """
        eliminado = False
        for ruta in (
            self._carpeta_historial() / f"{version}.joblib",
            self._carpeta_historial() / f"{version}.json",
        ):
            if self._a_papelera(ruta, "historial"):
                eliminado = True
        return eliminado

    def registrar_activos_en_historial(self) -> list[str]:
        """Archiva los paquetes activos que aún no estén en el historial.

        Sirve para no perder los modelos guardados antes de que existiera el
        historial.

        Returns:
            Nombres de las versiones archivadas.
        """
        creadas: list[str] = []
        archivadas = {
            (str(meta.get("nombre")), str(meta.get("fecha_entrenamiento")))
            for meta in self.listar_versiones()
        }
        for meta in self.listar():
            identidad = (str(meta.get("nombre")), str(meta.get("fecha_entrenamiento")))
            if identidad in archivadas:
                continue
            self._archivar(dict(meta), self._ruta_modelo(str(meta.get("nombre"))))
            creadas.append(self.nombre_version(meta))
        return creadas

    def cargar(self, nombre: str) -> PaqueteModelo:
        """Carga un paquete de modelo desde disco.

        Args:
            nombre: identificador del paquete.

        Returns:
            El :class:`PaqueteModelo` reconstruido.

        Raises:
            FileNotFoundError: si el paquete no existe.
        """
        ruta_modelo = self._ruta_modelo(nombre)
        ruta_metadatos = self._ruta_metadatos(nombre)
        if not ruta_modelo.exists() or not ruta_metadatos.exists():
            raise FileNotFoundError(f"No existe el paquete de modelo '{nombre}'")

        metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
        return PaqueteModelo(
            nombre=metadatos["nombre"],
            alcance=metadatos["alcance"],
            algoritmo=metadatos["algoritmo"],
            features=list(metadatos["features"]),
            metricas=dict(metadatos["metricas"]),
            desde_entrenamiento=date.fromisoformat(metadatos["desde_entrenamiento"]),
            hasta_entrenamiento=date.fromisoformat(metadatos["hasta_entrenamiento"]),
            fecha_entrenamiento=datetime.fromisoformat(metadatos["fecha_entrenamiento"]),
            extras=dict(metadatos.get("extras", {})),
            modelo=joblib.load(ruta_modelo),
        )

    def listar(self) -> list[dict[str, Any]]:
        """Lista los metadatos de todos los modelos disponibles.

        Returns:
            Lista de diccionarios (uno por modelo), ordenada por fecha descendente.
        """
        disponibles: list[dict[str, Any]] = []
        for ruta in self._carpeta.glob("*.json"):
            try:
                disponibles.append(json.loads(ruta.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as error:
                logger.warning("Metadatos ilegibles %s: %s", ruta, error)
        return sorted(disponibles, key=lambda m: m.get("fecha_entrenamiento", ""), reverse=True)

    def existe(self, nombre: str) -> bool:
        """Indica si un paquete de modelo existe.

        Args:
            nombre: identificador del paquete.

        Returns:
            ``True`` si están presentes el modelo y sus metadatos.
        """
        return self._ruta_modelo(nombre).exists() and self._ruta_metadatos(nombre).exists()

    # ------------------------------------------------------------------
    # Papelera: lo eliminado se mueve, no se destruye
    # ------------------------------------------------------------------
    def _carpeta_papelera(self) -> Path:
        """Carpeta donde se guarda lo eliminado, por si hay que recuperarlo.

        Returns:
            La ruta ``data/models/papelera``.
        """
        return self._carpeta / "papelera"

    def _a_papelera(self, ruta: Path, origen: str) -> bool:
        """Mueve un archivo a la papelera con la fecha y su origen en el nombre.

        Args:
            ruta: archivo a mover.
            origen: ``"activo"`` o ``"historial"``, para saber a dónde vuelve.

        Returns:
            ``True`` si se movió algo.
        """
        if not ruta.exists():
            return False
        destino = self._carpeta_papelera() / f"{origen}-{datetime.now():%Y%m%d_%H%M%S}-{ruta.name}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        ruta.replace(destino)
        return True

    def listar_papelera(self) -> list[dict[str, Any]]:
        """Lista lo que hay en la papelera.

        Returns:
            Lista de diccionarios con ``archivo``, ``origen`` y ``mb``.
        """
        carpeta = self._carpeta_papelera()
        if not carpeta.exists():
            return []
        filas = []
        for ruta in sorted(carpeta.glob("*.joblib")) + sorted(carpeta.glob("*.json")):
            origen = ruta.name.split("-", 1)[0]
            filas.append(
                {
                    "archivo": ruta.name,
                    "origen": origen,
                    "mb": round(ruta.stat().st_size / (1024 * 1024), 1),
                }
            )
        return filas

    def restaurar_de_papelera(self, archivo: str) -> bool:
        """Devuelve un archivo de la papelera a su carpeta original.

        Args:
            archivo: nombre del archivo tal como aparece en :meth:`listar_papelera`.

        Returns:
            ``True`` si se restauró.
        """
        ruta = self._carpeta_papelera() / archivo
        if not ruta.exists():
            return False
        partes = archivo.split("-", 2)
        if len(partes) < 3:
            return False
        origen, _, nombre = partes
        destino = (self._carpeta if origen == "activo" else self._carpeta_historial()) / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        ruta.replace(destino)
        return True

    def eliminar(self, nombre: str) -> bool:
        """Elimina un paquete de modelo (modelo + metadatos).

        Args:
            nombre: identificador del paquete.

        Returns:
            ``True`` si se eliminó algo.
        """
        eliminado = False
        for ruta in (self._ruta_modelo(nombre), self._ruta_metadatos(nombre)):
            if self._a_papelera(ruta, "activo"):
                eliminado = True
        return eliminado
