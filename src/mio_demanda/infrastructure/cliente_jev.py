"""Cliente HTTP de la API de TypeSafe (modelo Jev).

Adaptador de infraestructura: solo habla HTTP y traduce errores. No decide nada de
negocio. La clave viaja en la cabecera ``Authorization``, se lee como ``SecretStr`` y
**nunca** se escribe en un log ni en un mensaje de error.

La API expone:

* ``GET /v1/models``: modelos y alias disponibles para la cuenta.
* ``POST /v1/systemone``: un estado y varias preguntas tipadas (sí/no, elección,
  puntuación sobre una rúbrica).

Se usa la librería estándar para no añadir dependencias al proyecto.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from mio_demanda.config import Settings
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)

#: Tipo del transporte inyectable (en las pruebas se sustituye por un doble).
Transporte = Callable[[dict[str, Any]], dict[str, Any]]


class ErrorSegundaOpinion(RuntimeError):
    """No se pudo obtener la segunda opinión del modelo externo."""


class ClienteJev:
    """Cliente de la API de TypeSafe.

    Args:
        settings: configuración (clave, URL base, modelo y espera máxima).
        transporte: función que envía el cuerpo y devuelve la respuesta. Se inyecta
            en las pruebas para no tocar la red.
    """

    def __init__(self, settings: Settings, transporte: Transporte | None = None) -> None:
        self._settings = settings
        self._transporte = transporte or self._enviar

    @property
    def configurado(self) -> bool:
        """Indica si hay token configurado para consultar la API."""
        return self._settings.clave_jev() is not None

    def modelos(self) -> list[dict[str, Any]]:
        """Lista los modelos y alias disponibles para la cuenta.

        Returns:
            Lista de diccionarios con ``name``, ``description`` y ``release_date``.

        Raises:
            ErrorSegundaOpinion: si no hay clave o la API falla.
        """
        respuesta = self._peticion("GET", "/v1/models")
        return list(respuesta.get("models", []))

    def preguntar(self, estado: Any, preguntas: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Envía un estado y varias preguntas tipadas a ``POST /v1/systemone``.

        Args:
            estado: contenido al que se refieren todas las preguntas.
            preguntas: preguntas por nombre (``{"dia_2026-11-03": {...}}``).

        Returns:
            La respuesta completa: ``model``, ``answers`` y ``usage``.

        Raises:
            ErrorSegundaOpinion: si no hay clave, si la API responde con error o si
                no se puede contactar.
        """
        cuerpo = {
            "model": self._settings.jev_modelo,
            "state": estado,
            "questions": preguntas,
        }
        return self._transporte(cuerpo)

    # ------------------------------------------------------------------
    # Transporte HTTP (sin dependencias externas)
    # ------------------------------------------------------------------
    def _peticion(self, metodo: str, ruta: str) -> dict[str, Any]:
        """Ejecuta una petición simple sin cuerpo.

        Args:
            metodo: verbo HTTP.
            ruta: ruta relativa a la URL base.

        Returns:
            El JSON de la respuesta.

        Raises:
            ErrorSegundaOpinion: si no hay clave o la petición falla.
        """
        self._exigir_clave()
        url = f"{self._settings.jev_base_url.rstrip('/')}{ruta}"
        peticion = urllib.request.Request(url, method=metodo, headers=self._cabeceras())
        return self._leer(peticion)

    def _enviar(self, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """Envía el cuerpo a ``POST /v1/systemone``.

        Args:
            cuerpo: petición completa (modelo, estado y preguntas).

        Returns:
            El JSON de la respuesta.

        Raises:
            ErrorSegundaOpinion: si no hay clave o la petición falla.
        """
        self._exigir_clave()
        url = f"{self._settings.jev_base_url.rstrip('/')}/v1/systemone"
        datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        peticion = urllib.request.Request(url, data=datos, method="POST", headers=self._cabeceras())
        return self._leer(peticion)

    def _cabeceras(self) -> dict[str, str]:
        """Cabeceras de la petición, con el token como portador."""
        clave = self._settings.clave_jev() or ""
        return {
            "Authorization": f"Bearer {clave}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _exigir_clave(self) -> None:
        """Comprueba que exista clave antes de gastar una llamada.

        Raises:
            ErrorSegundaOpinion: si no hay clave configurada.
        """
        if not self.configurado:
            raise ErrorSegundaOpinion(
                "No hay clave de la segunda opinión. Define MIO_JEV_API_KEY en el archivo .env."
            )

    def _leer(self, peticion: urllib.request.Request) -> dict[str, Any]:
        """Ejecuta la petición y devuelve el JSON, traduciendo los fallos.

        Args:
            peticion: petición preparada.

        Returns:
            El JSON de la respuesta.

        Raises:
            ErrorSegundaOpinion: si la API responde con error o no se puede contactar.
        """
        try:
            with urllib.request.urlopen(
                peticion, timeout=self._settings.jev_timeout_segundos
            ) as respuesta:
                return dict(json.loads(respuesta.read().decode("utf-8")))
        except urllib.error.HTTPError as error:
            detalle = error.read().decode("utf-8", errors="replace")[:400]
            raise ErrorSegundaOpinion(
                f"La API respondió {error.code}. Detalle: {detalle}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise ErrorSegundaOpinion(
                f"No se pudo contactar la API de la segunda opinión: {error}"
            ) from error
