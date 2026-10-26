"""Repositorio del calendario: eventos estándar, eventos del usuario y tipo de día.

Fusiona tres fuentes en un único calendario operativo:

1. **Calendario oficial** normalizado (``data/raw/calendario.parquet``): tipo de día.
2. **Eventos estándar** (``config/eventos_base.json``): nacionales y de calendario
   (festivos, Semana Santa, fechas comerciales). Se expanden por regla a cualquier año.
3. **Eventos del usuario** (``config/eventos_personalizados.json``): Feria de Cali,
   recesos escolares, Black Friday, elecciones, sismos, obras… Cada uno puede ser
   **recurrente** (con ``regla``) o de **fechas fijas**.

**Contrato de columnas estable**: el calendario enriquecido siempre genera una
columna ``evento_<id>`` por *cada* evento configurado (aunque en el rango no
ocurra). Así el conjunto de variables del modelo es idéntico en entrenamiento y en
proyección, y el modelo guardado no se rompe.

**Persistencia segura**: el archivo del usuario se lee antes de escribir, se guarda
de forma atómica y **nunca se sobrescribe** si no se puede leer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.calendario import clasificar_tipo_dia, rango_fechas
from mio_demanda.domain.entidades import CategoriaEvento, EventoCalendario, PerfilImpacto
from mio_demanda.domain.reglas_eventos import (
    expandir_definiciones,
    expandir_evento,
    ids_definiciones,
)
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)

_PREFIJO_EVENTO = "evento_"
_CLAVE_REGLA = "regla"


def _aplicar_ajuste(evento: EventoCalendario, ajuste: dict[str, Any] | None) -> EventoCalendario:
    """Devuelve el evento con los ajustes del usuario aplicados.

    Args:
        evento: evento estándar expandido.
        ajuste: campos ajustados (``impacto``, ``perfil_impacto``, ``categoria``,
            ``fuente``, ``nombre``) o ``None``.

    Returns:
        El evento original o una copia con los ajustes aplicados.
    """
    if not ajuste:
        return evento

    cambios: dict[str, Any] = {}
    if "impacto" in ajuste:
        cambios["impacto"] = float(ajuste["impacto"])
    if ajuste.get("perfil_impacto"):
        cambios["perfil_impacto"] = PerfilImpacto(str(ajuste["perfil_impacto"]))
    if ajuste.get("categoria"):
        cambios["categoria"] = CategoriaEvento(str(ajuste["categoria"]))
    if "fuente" in ajuste:
        cambios["fuente"] = str(ajuste["fuente"])
    if ajuste.get("nombre"):
        cambios["nombre"] = str(ajuste["nombre"])
    return replace(evento, **cambios) if cambios else evento


def _campos_ajustables(cambio: dict[str, Any]) -> dict[str, Any]:
    """Extrae los campos editables de una fila del editor tipo Excel.

    Args:
        cambio: fila con las claves del editor (``Impacto``, ``Perfil``,
            ``Categoria``, ``Fuente``, ``Nombre``).

    Returns:
        Diccionario con las claves del modelo de datos (``impacto``,
        ``perfil_impacto``, ``categoria``, ``fuente``, ``nombre``).
    """
    campos: dict[str, Any] = {}
    impacto = cambio.get("Impacto")
    if impacto is not None and str(impacto).strip() not in {"", "nan", "<NA>"}:
        campos["impacto"] = float(impacto)
    if cambio.get("Perfil"):
        campos["perfil_impacto"] = str(cambio["Perfil"]).strip()
    if cambio.get("Categoria"):
        campos["categoria"] = str(cambio["Categoria"]).strip()
    if cambio.get("Nombre"):
        campos["nombre"] = str(cambio["Nombre"]).strip()
    if "Fuente" in cambio:
        campos["fuente"] = str(cambio.get("Fuente") or "").strip()
    return campos


def _slug(texto: str) -> str:
    """Convierte un texto libre en un identificador estable (slug)."""
    sin_acentos = unicodedata.normalize("NFKD", texto)
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    limpio = re.sub(r"[^a-zA-Z0-9]+", "_", sin_acentos).strip("_").lower()
    return limpio or "evento"


class RepositorioCalendario:
    """Lectura/escritura del calendario operativo y de los eventos.

    Args:
        settings: configuración con las rutas del calendario y de los eventos.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._definiciones: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Calendario oficial
    # ------------------------------------------------------------------
    def calendario_oficial(self) -> pd.DataFrame:
        """Carga el calendario oficial normalizado.

        Returns:
            DataFrame con ``fecha``, ``dia_tipo_num`` y ``dia_tipo_nom``; vacío si
            no se ha ejecutado la ingesta.
        """
        ruta = self._settings.parquet_calendario
        if not ruta.exists():
            return pd.DataFrame(columns=["fecha", "dia_tipo_num", "dia_tipo_nom"])
        df = pd.read_parquet(ruta)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def _calendario_para(self, desde: date, hasta: date) -> pd.DataFrame:
        """Calendario oficial recortado al rango, rellenando lo que falte.

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            DataFrame con ``fecha``, ``dia_tipo_num`` y ``dia_tipo_nom``.
        """
        oficial = self.calendario_oficial()
        rango = pd.DataFrame({"fecha": pd.to_datetime(rango_fechas(desde, hasta))})

        if not oficial.empty:
            recorte = oficial[
                (oficial["fecha"] >= pd.Timestamp(desde))
                & (oficial["fecha"] <= pd.Timestamp(hasta))
            ]
            rango = rango.merge(
                recorte[["fecha", "dia_tipo_num", "dia_tipo_nom"]], on="fecha", how="left"
            )
        else:
            rango["dia_tipo_num"] = pd.NA
            rango["dia_tipo_nom"] = pd.NA

        faltantes = rango["dia_tipo_num"].isna()
        if faltantes.any():
            respaldo = [
                clasificar_tipo_dia(pd.Timestamp(f).date()) for f in rango.loc[faltantes, "fecha"]
            ]
            rango.loc[faltantes, "dia_tipo_num"] = [int(t) for t in respaldo]
            rango.loc[faltantes, "dia_tipo_nom"] = [t.nombre for t in respaldo]

        rango["dia_tipo_num"] = rango["dia_tipo_num"].astype(int)
        return rango

    # ------------------------------------------------------------------
    # Eventos estándar (catálogo base)
    # ------------------------------------------------------------------
    def definiciones_base(self) -> list[dict[str, Any]]:
        """Carga el catálogo estándar de ``config/eventos_base.json``.

        Returns:
            Lista de definiciones; vacía si el archivo no existe o es inválido.
        """
        if self._definiciones is not None:
            return self._definiciones

        ruta = self._settings.config_dir / "eventos_base.json"
        if not ruta.exists():
            logger.warning("No existe el catálogo de eventos estándar en %s", ruta)
            self._definiciones = []
            return self._definiciones

        try:
            contenido = json.loads(ruta.read_text(encoding="utf-8"))
            self._definiciones = list(contenido.get("eventos", []))
        except (json.JSONDecodeError, OSError) as error:
            logger.error("No se pudo leer el catálogo estándar de eventos: %s", error)
            self._definiciones = []
        return self._definiciones

    def invalidar_cache(self) -> None:
        """Descarta las definiciones estándar en memoria (tras editar el catálogo)."""
        self._definiciones = None

    def eventos_base(self, desde: date, hasta: date) -> list[EventoCalendario]:
        """Expande los eventos estándar a fechas reales dentro del rango.

        Aplica los **ajustes** que el usuario haya guardado sobre los eventos
        estándar (impacto, perfil, categoría, fuente), sin tocar el catálogo base.

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            Lista de :class:`EventoCalendario` (uno por evento y año aplicable).
        """
        eventos = expandir_definiciones(self.definiciones_base(), range(desde.year, hasta.year + 1))
        ajustes = self.ajustes_estandar()
        if not ajustes:
            return eventos
        return [_aplicar_ajuste(evento, ajustes.get(evento.id)) for evento in eventos]

    # ------------------------------------------------------------------
    # Eventos del usuario
    # ------------------------------------------------------------------
    def ruta_eventos(self) -> Path:
        """Devuelve la ruta del archivo de eventos del usuario.

        Returns:
            Ruta del JSON donde se persisten los eventos.
        """
        return self._settings.archivo_eventos_personalizados

    def _leer_archivo(self) -> dict[str, Any]:
        """Lee el archivo del usuario completo (eventos y ajustes).

        Returns:
            El contenido del archivo, o un diccionario vacío si no existe.

        Raises:
            ValueError: si el archivo existe pero su contenido es ilegible.
        """
        ruta = self.ruta_eventos()
        if not ruta.exists():
            return {}
        try:
            return dict(json.loads(ruta.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"El archivo de eventos {ruta} no se pudo leer: {error}") from error

    def _leer_definiciones(self) -> list[dict[str, Any]]:
        """Lee las definiciones de eventos del usuario, **sin ocultar errores**.

        Returns:
            Lista de definiciones tal como están en disco.

        Raises:
            ValueError: si el archivo existe pero su contenido es ilegible.
        """
        return list(self._leer_archivo().get("eventos", []))

    def ajustes_estandar(self) -> dict[str, dict[str, Any]]:
        """Ajustes del usuario sobre los eventos **estándar** (impacto, perfil, …).

        Returns:
            Mapa ``id de evento estándar -> campos ajustados``; vacío si no hay.
        """
        try:
            return dict(self._leer_archivo().get("ajustes_estandar", {}))
        except ValueError as error:
            logger.error("%s. No se aplican ajustes estándar.", error)
            return {}

    def definiciones_usuario(self) -> list[dict[str, Any]]:
        """Definiciones del usuario sin lanzar excepción (para consultas de lectura).

        Returns:
            Lista de definiciones; vacía si el archivo es ilegible.
        """
        try:
            return self._leer_definiciones()
        except ValueError as error:
            logger.error("%s. No se cargan eventos y NO se sobrescribirá el archivo.", error)
            return []

    @staticmethod
    def _es_recurrente(definicion: dict[str, Any]) -> bool:
        """Indica si una definición se expande por regla (recurrente)."""
        return _CLAVE_REGLA in definicion

    @staticmethod
    def _parsear_concretos(
        definiciones: Sequence[dict[str, Any]],
    ) -> list[EventoCalendario]:
        """Convierte las definiciones de fechas fijas en eventos.

        Args:
            definiciones: definiciones del usuario (se ignoran las recurrentes).

        Returns:
            Lista de :class:`EventoCalendario` con fechas explícitas.
        """
        eventos: list[EventoCalendario] = []
        for item in definiciones:
            if RepositorioCalendario._es_recurrente(item):
                continue
            try:
                eventos.append(
                    EventoCalendario(
                        id=str(item["id"]),
                        nombre=str(item["nombre"]),
                        fecha_inicio=pd.Timestamp(item["fecha_inicio"]).date(),
                        fecha_fin=pd.Timestamp(item["fecha_fin"]).date(),
                        impacto=float(item.get("impacto", 0.5)),
                        categoria=CategoriaEvento(
                            item.get("categoria", CategoriaEvento.OTRO.value)
                        ),
                        fuente=str(item.get("fuente", "")),
                        perfil_impacto=PerfilImpacto(
                            item.get("perfil_impacto", PerfilImpacto.CONSTANTE.value)
                        ),
                    )
                )
            except (KeyError, ValueError) as error:
                logger.warning("Evento con fechas fijas inválido, se omite: %s", error)
        return eventos

    @staticmethod
    def _expandir_recurrentes(
        definiciones: Sequence[dict[str, Any]], anios: Sequence[int]
    ) -> list[EventoCalendario]:
        """Expande las definiciones recurrentes del usuario a fechas reales.

        Args:
            definiciones: definiciones del usuario.
            anios: años a resolver.

        Returns:
            Lista de :class:`EventoCalendario` expandidos.
        """
        eventos: list[EventoCalendario] = []
        for item in definiciones:
            if not RepositorioCalendario._es_recurrente(item):
                continue
            try:
                eventos.extend(expandir_evento(item, anios))
            except (KeyError, ValueError) as error:
                logger.warning("Evento recurrente inválido, se omite: %s", error)
        return eventos

    def eventos_personalizados(self) -> list[EventoCalendario]:
        """Eventos de **fechas fijas** del usuario (los recurrentes se expanden por rango).

        Returns:
            Lista de :class:`EventoCalendario`.
        """
        return self._parsear_concretos(self.definiciones_usuario())

    def eventos_usuario(self, desde: date, hasta: date) -> list[EventoCalendario]:
        """Todos los eventos del usuario vigentes en el rango (fijos + recurrentes).

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            Lista combinada de :class:`EventoCalendario`.
        """
        definiciones = self.definiciones_usuario()
        anios = list(range(desde.year, hasta.year + 1))
        return self._parsear_concretos(definiciones) + self._expandir_recurrentes(
            definiciones, anios
        )

    def agregar_evento(
        self,
        nombre: str,
        fecha_inicio: date,
        fecha_fin: date,
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Crea y persiste un evento del usuario (fechas fijas).

        Args:
            nombre: nombre legible del evento.
            fecha_inicio: primer día del evento.
            fecha_fin: último día del evento.
            impacto: importancia relativa en [0, 1].
            categoria: categoría del evento.
            fuente: justificación del evento.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El :class:`EventoCalendario` creado.

        Raises:
            RuntimeError: si el archivo de eventos existente no se puede leer (para
                no sobrescribir y perder eventos ya guardados).
        """
        try:
            crudos = self._leer_definiciones()
        except ValueError as error:
            raise RuntimeError(
                f"No se agregó el evento: el archivo de eventos no se puede leer ({error}). "
                "Corrige o elimina el archivo y vuelve a intentarlo. No se modificó nada."
            ) from error

        reservados = {str(d["id"]) for d in self.definiciones_base()}
        existentes = reservados | {str(d["id"]) for d in crudos}
        identificador = _slug(nombre)
        candidato, sufijo = identificador, 1
        while candidato in existentes:
            sufijo += 1
            candidato = f"{identificador}_{sufijo}"

        evento = EventoCalendario(
            id=candidato,
            nombre=nombre,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            impacto=impacto,
            categoria=categoria,
            fuente=fuente,
            perfil_impacto=perfil_impacto,
        )
        crudos.append(
            {
                "id": evento.id,
                "nombre": evento.nombre,
                "fecha_inicio": evento.fecha_inicio.isoformat(),
                "fecha_fin": evento.fecha_fin.isoformat(),
                "impacto": evento.impacto,
                "categoria": evento.categoria.value,
                "fuente": evento.fuente,
                "perfil_impacto": evento.perfil_impacto.value,
            }
        )
        self._guardar_definiciones(crudos)
        return evento

    def _siguiente_id(self, nombre: str, crudos: Sequence[dict[str, Any]]) -> str:
        """Genera un identificador libre a partir del nombre.

        Args:
            nombre: nombre legible del evento.
            crudos: definiciones ya existentes en el archivo.

        Returns:
            Identificador único (``slug`` con sufijo si hace falta).
        """
        reservados = {str(d["id"]) for d in self.definiciones_base()}
        existentes = reservados | {str(d["id"]) for d in crudos}
        identificador = _slug(nombre)
        candidato, sufijo = identificador, 1
        while candidato in existentes:
            sufijo += 1
            candidato = f"{identificador}_{sufijo}"
        return candidato

    def agregar_evento_con_regla(
        self,
        nombre: str,
        regla: dict[str, Any],
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Crea y persiste un evento **recurrente** a partir de una regla.

        La regla usa el mismo motor declarativo del catálogo estándar, de modo que
        el evento puede repetirse en el mismo mes y día, o calcular sus fechas
        (por ejemplo, relativo a la Pascua o al n-ésimo día de la semana de un mes).

        Args:
            nombre: nombre legible del evento.
            regla: definición de la regla (``tipo`` y sus parámetros).
            impacto: importancia relativa en [0, 1].
            categoria: categoría del evento.
            fuente: justificación del evento.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El :class:`EventoCalendario` resuelto para el año actual.

        Raises:
            RuntimeError: si el archivo de eventos existente no se puede leer.
            ValueError: si la regla no produce fechas válidas.
        """
        try:
            crudos = self._leer_definiciones()
        except ValueError as error:
            raise RuntimeError(
                f"No se agregó el evento: el archivo de eventos no se puede leer ({error}). "
                "Corrige o elimina el archivo y vuelve a intentarlo. No se modificó nada."
            ) from error

        definicion: dict[str, Any] = {
            "id": self._siguiente_id(nombre, crudos),
            "nombre": nombre,
            "categoria": categoria.value,
            "impacto": impacto,
            "fuente": fuente,
            "perfil_impacto": perfil_impacto.value,
            "regla": dict(regla),
        }
        resueltos = expandir_evento(definicion, [date.today().year])
        if not resueltos:
            raise ValueError("La regla del evento no produce fechas válidas")

        crudos.append(definicion)
        self._guardar_definiciones(crudos)
        return resueltos[0]

    def agregar_evento_recurrente(
        self,
        nombre: str,
        mes_inicio: int,
        dia_inicio: int,
        mes_fin: int,
        dia_fin: int,
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Crea un evento que se repite **el mismo mes y día** todos los años.

        Args:
            nombre: nombre legible del evento.
            mes_inicio: mes de inicio (1-12).
            dia_inicio: día de inicio.
            mes_fin: mes de fin (1-12).
            dia_fin: día de fin.
            impacto: importancia relativa en [0, 1].
            categoria: categoría del evento.
            fuente: justificación del evento.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El :class:`EventoCalendario` resuelto para el año actual.
        """
        return self.agregar_evento_con_regla(
            nombre=nombre,
            regla={
                "tipo": "rango_fijo",
                "mes_ini": int(mes_inicio),
                "dia_ini": int(dia_inicio),
                "mes_fin": int(mes_fin),
                "dia_fin": int(dia_fin),
            },
            impacto=impacto,
            categoria=categoria,
            fuente=fuente,
            perfil_impacto=perfil_impacto,
        )

    def agregar_rango_anual(
        self,
        nombre: str,
        anio: int,
        fecha_inicio: date,
        fecha_fin: date,
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Registra el rango de **un año** para un evento que cambia de fechas.

        Pensado para eventos recurrentes cuyas fechas varían cada año (por ejemplo,
        el Festival Petronio Álvarez): la primera vez crea el evento y las veces
        siguientes **añaden o actualizan el rango de ese año** en el mismo evento,
        de modo que el modelo tenga una única bandera para todos los años.

        Args:
            nombre: nombre legible del evento (identifica al evento entre años).
            anio: año del rango.
            fecha_inicio: primer día del evento ese año.
            fecha_fin: último día del evento ese año.
            impacto: importancia relativa en [0, 1].
            categoria: categoría del evento.
            fuente: justificación del evento.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El :class:`EventoCalendario` del año indicado.

        Raises:
            RuntimeError: si el archivo de eventos existente no se puede leer.
            ValueError: si el rango no es válido.
        """
        if fecha_fin < fecha_inicio:
            raise ValueError("La fecha final no puede ser anterior a la inicial")

        try:
            crudos = self._leer_definiciones()
        except ValueError as error:
            raise RuntimeError(
                f"No se agregó el evento: el archivo de eventos no se puede leer ({error}). "
                "Corrige o elimina el archivo y vuelve a intentarlo. No se modificó nada."
            ) from error

        identificador = _slug(nombre)
        definicion = next(
            (
                item
                for item in crudos
                if str(item.get("id")) == identificador
                and str(item.get("regla", {}).get("tipo")) == "fechas_por_anio"
            ),
            None,
        )
        if definicion is None:
            definicion = {
                "id": self._siguiente_id(nombre, crudos),
                "nombre": nombre,
                "categoria": categoria.value,
                "impacto": impacto,
                "fuente": fuente,
                "perfil_impacto": perfil_impacto.value,
                "regla": {"tipo": "fechas_por_anio", "fechas": {}},
            }
            crudos.append(definicion)
        else:
            definicion["categoria"] = categoria.value
            definicion["impacto"] = impacto
            definicion["fuente"] = fuente
            definicion["perfil_impacto"] = perfil_impacto.value

        definicion["regla"]["fechas"][str(anio)] = [
            fecha_inicio.isoformat(),
            fecha_fin.isoformat(),
        ]
        self._guardar_definiciones(crudos)

        return EventoCalendario(
            id=str(definicion["id"]),
            nombre=nombre,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            impacto=impacto,
            categoria=categoria,
            fuente=fuente,
            perfil_impacto=perfil_impacto,
        )

    def eliminar_evento(self, evento_id: str) -> bool:
        """Elimina un evento del usuario (recurrente o de fechas fijas).

        Args:
            evento_id: identificador del evento a eliminar.

        Returns:
            ``True`` si el evento existía y fue eliminado.

        Raises:
            RuntimeError: si el archivo de eventos no se puede leer.
        """
        try:
            crudos = self._leer_definiciones()
        except ValueError as error:
            raise RuntimeError(
                f"No se eliminó el evento: el archivo de eventos no se puede leer ({error})."
            ) from error

        restantes = [item for item in crudos if str(item.get("id")) != evento_id]
        if len(restantes) == len(crudos):
            return False
        self._guardar_definiciones(restantes)
        return True

    def _guardar_definiciones(self, crudos: list[dict[str, Any]]) -> None:
        """Escribe las definiciones del usuario preservando los ajustes estándar.

        Args:
            crudos: lista completa de definiciones a persistir.
        """
        self._guardar_archivo(crudos, self.ajustes_estandar())

    def _guardar_archivo(
        self, crudos: list[dict[str, Any]], ajustes: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Escribe el archivo del usuario de forma **atómica**.

        Se preservan tal cual las definiciones recurrentes (con ``regla``), las de
        fechas fijas y los ajustes sobre eventos estándar, de modo que agregar,
        editar o eliminar un evento nunca altera los demás.

        Args:
            crudos: lista completa de definiciones a persistir.
            ajustes: ajustes sobre los eventos estándar (``None`` para conservarlos).
        """
        ruta = self.ruta_eventos()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        if ajustes is None:
            ajustes = self.ajustes_estandar()

        payload = dict(_AYUDA_ARCHIVO)
        payload["ajustes_estandar"] = ajustes
        payload["eventos"] = crudos

        temporal = ruta.with_name(f"{ruta.name}.tmp")
        temporal.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(ruta)
        logger.info(
            "Eventos del usuario guardados: %s evento(s) y %s ajuste(s) en %s",
            len(crudos),
            len(ajustes),
            ruta,
        )

    def actualizar_eventos(self, cambios: Sequence[dict[str, Any]]) -> int:
        """Aplica cambios masivos a los eventos (estándar y del usuario).

        Es la operación que respalda el editor tipo Excel. Para los eventos del
        usuario se actualiza su definición; para los estándar se guarda un
        **ajuste** que se aplica al expandirlos, sin tocar el catálogo base.

        Args:
            cambios: lista de diccionarios con ``Id`` y los campos a modificar
                (``Impacto``, ``Perfil``, ``Categoria``, ``Fuente`` y, si aplica,
                ``Inicio``/``Fin`` para eventos de fechas fijas).

        Returns:
            Número de eventos actualizados.

        Raises:
            RuntimeError: si el archivo de eventos no se puede leer.
        """
        try:
            contenido = self._leer_archivo()
        except ValueError as error:
            raise RuntimeError(
                f"No se guardaron los cambios: el archivo de eventos no se puede leer ({error}). "
                "No se modificó nada."
            ) from error

        crudos = list(contenido.get("eventos", []))
        ajustes = dict(contenido.get("ajustes_estandar", {}))
        ids_base = {str(d["id"]) for d in self.definiciones_base()}
        actualizados = 0

        for cambio in cambios:
            identificador = str(cambio.get("Id", "")).strip()
            if not identificador:
                continue

            if identificador in ids_base:
                ajuste = dict(ajustes.get(identificador, {}))
                ajuste.update(_campos_ajustables(cambio))
                ajustes[identificador] = ajuste
                actualizados += 1
                continue

            definicion = next(
                (item for item in crudos if str(item.get("id")) == identificador), None
            )
            if definicion is None:
                continue
            definicion.update(_campos_ajustables(cambio))
            if not self._es_recurrente(definicion) and cambio.get("Inicio") and cambio.get("Fin"):
                definicion["fecha_inicio"] = str(cambio["Inicio"])
                definicion["fecha_fin"] = str(cambio["Fin"])
            actualizados += 1

        self._guardar_archivo(crudos, ajustes)
        return actualizados

    # ------------------------------------------------------------------
    # Catálogo unificado
    # ------------------------------------------------------------------
    def ids_eventos(self) -> list[str]:
        """Identificadores de **todos** los eventos configurados (estándar y usuario).

        Returns:
            Lista ordenada y sin duplicados de identificadores.
        """
        propios = sorted({str(d["id"]) for d in self.definiciones_usuario()})
        return ids_definiciones(self.definiciones_base()) + propios

    def eventos_vigentes(self, desde: date, hasta: date) -> list[EventoCalendario]:
        """Eventos estándar expandidos más los del usuario vigentes en el rango.

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            Lista combinada de eventos.
        """
        return self.eventos_base(desde, hasta) + self.eventos_usuario(desde, hasta)

    # ------------------------------------------------------------------
    # Calendario enriquecido
    # ------------------------------------------------------------------
    def calendario_enriquecido(self, desde: date, hasta: date) -> pd.DataFrame:
        """Calendario listo para el modelo: tipo de día, días de pago y eventos.

        Genera una columna binaria ``evento_<id>`` por **cada** evento configurado
        (esquema estable), más el conteo y el impacto ponderado del día, y las
        banderas de **día de pago** (15 y último día del mes).

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            DataFrame con ``fecha``, ``dia_tipo_num``, ``dia_tipo_nom``, las
            variables de calendario, las banderas ``evento_<id>``, ``n_eventos``,
            ``impacto_evento`` y ``peso_eventos``.
        """
        cal = self._calendario_para(desde, hasta)
        cal["dia_semana"] = cal["fecha"].dt.weekday
        cal["es_habil"] = (cal["dia_tipo_num"] == 0).astype(int)
        cal["es_sabado"] = (cal["dia_tipo_num"] == 1).astype(int)
        cal["es_festivo"] = (cal["dia_tipo_num"] == 2).astype(int)
        cal["es_fin_semana"] = (cal["dia_semana"] >= 5).astype(int)
        cal["dia_del_mes"] = cal["fecha"].dt.day
        cal["mes_num"] = cal["fecha"].dt.month
        cal["semana_iso"] = cal["fecha"].dt.isocalendar().week.astype(int)
        cal["anio"] = cal["fecha"].dt.year
        cal["es_inicio_mes"] = (cal["dia_del_mes"] <= 5).astype(int)
        cal["es_fin_mes"] = (cal["dia_del_mes"] >= 25).astype(int)
        cal["es_quincena"] = cal["dia_del_mes"].isin([14, 15, 16, 29, 30, 31]).astype(int)
        cal = self._agregar_dias_de_pago(cal)

        # Columnas estables: todos los eventos configurados existen siempre.
        for identificador in self.ids_eventos():
            cal[f"{_PREFIJO_EVENTO}{identificador}"] = 0

        cal["n_eventos"] = 0
        cal["impacto_evento"] = 0.0
        cal["peso_eventos"] = 0.0
        fechas = cal["fecha"].dt.date
        for evento in self.eventos_vigentes(desde, hasta):
            columna = f"{_PREFIJO_EVENTO}{evento.id}"
            dentro = fechas.between(evento.fecha_inicio, evento.fecha_fin).to_numpy()
            if not dentro.any():
                continue
            cal[columna] = np.maximum(cal[columna].to_numpy(), dentro.astype(int))
            cal.loc[dentro, "n_eventos"] = cal.loc[dentro, "n_eventos"] + 1

            # El impacto puede decaer o crecer a lo largo de los días del evento.
            pesos = self._pesos_por_dia(cal.loc[dentro, "fecha"], evento)
            cal.loc[dentro, "peso_eventos"] = cal.loc[dentro, "peso_eventos"].to_numpy() + pesos
            cal.loc[dentro, "impacto_evento"] = np.maximum(
                cal.loc[dentro, "impacto_evento"].to_numpy(), pesos
            )

        return cal

    @staticmethod
    def _agregar_dias_de_pago(cal: pd.DataFrame) -> pd.DataFrame:
        """Marca los **días de pago** (15 y último día de cada mes) y sus vecinos.

        Los días de pago mueven la demanda del sistema, por lo que se marcan de
        forma explícita y también el día anterior y el posterior.

        Args:
            cal: calendario con ``fecha`` y ``dia_del_mes``.

        Returns:
            El calendario con ``es_dia_15``, ``es_ultimo_dia_mes``, ``es_dia_pago``,
            ``es_pre_pago`` y ``es_post_pago``.
        """
        ultimo = cal["fecha"].dt.is_month_end
        cal["es_dia_15"] = (cal["dia_del_mes"] == 15).astype(int)
        cal["es_ultimo_dia_mes"] = ultimo.astype(int)
        cal["es_dia_pago"] = ((cal["dia_del_mes"] == 15) | ultimo).astype(int)
        cal["es_pre_pago"] = (
            (cal["dia_del_mes"] == 14) | ultimo.shift(-1, fill_value=False)
        ).astype(int)
        cal["es_post_pago"] = ((cal["dia_del_mes"] == 16) | cal["fecha"].dt.is_month_start).astype(
            int
        )
        return cal

    @staticmethod
    def _pesos_por_dia(fechas: pd.Series, evento: EventoCalendario) -> np.ndarray:
        """Calcula el peso del impacto de un evento en cada una de sus fechas.

        Args:
            fechas: fechas del calendario incluidas en el evento.
            evento: evento con su impacto y perfil.

        Returns:
            Arreglo de pesos, uno por fecha.
        """
        posiciones = (fechas - pd.Timestamp(evento.fecha_inicio)).dt.days.to_numpy()
        factores = np.array(
            [evento.perfil_impacto.factor(int(posicion), evento.dias) for posicion in posiciones]
        )
        return evento.impacto * factores

    def columnas_eventos(self) -> list[str]:
        """Nombres de las columnas de eventos del calendario enriquecido.

        Returns:
            Lista de nombres ``evento_<id>``.
        """
        return [f"{_PREFIJO_EVENTO}{identificador}" for identificador in self.ids_eventos()]


_AYUDA_ARCHIVO: dict[str, Any] = {
    "version": 2,
    "_ayuda": (
        "Eventos agregados por el equipo (los carga la aplicación y también se pueden crear "
        "desde la pestaña Entrenamiento). A diferencia de config/eventos_base.json, aquí puede "
        "vivir cualquier cosa que NO sea estándar: Feria de Cali, recesos escolares (dependen de "
        "resoluciones), Black Friday, elecciones, fechas puntuales. Cada evento puede definirse "
        "de dos formas: (1) RECURRENTE con 'regla' — igual que en el catálogo base, se expande a "
        "fechas de cualquier año — o (2) FECHAS FIJAS con 'fecha_inicio' y 'fecha_fin'. Campos: "
        "id, nombre, categoria, impacto (0-1), perfil_impacto (constante|decreciente|creciente), "
        "fuente. La aplicación escribe este archivo de forma atómica y NO lo sobrescribe si no "
        "puede leerlo."
    ),
    "categorias_validas": [categoria.value for categoria in CategoriaEvento],
    "perfiles_validos": [perfil.value for perfil in PerfilImpacto],
}
