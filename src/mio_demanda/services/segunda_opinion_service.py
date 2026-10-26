"""Segunda opinión de demanda con un modelo externo (Jev, de TypeSafe).

Caso de uso: cuando el equipo **desconfía** de la proyección del modelo entrenado,
pide una segunda opinión. El modelo externo recibe el estado del sistema —**sin**
nuestra predicción, para no sesgarlo— y devuelve una estimación por día, traducida
desde una rúbrica ordenada de niveles de demanda.

Punto clave del diseño: **la rúbrica es distinta para cada día**. Se centra en el
nivel típico de su *tipo de día* (hábil, sábado, festivo), porque un domingo mueve
una fracción de la demanda de un día hábil. Con una sola rúbrica global, el domingo
real quedaría fuera de las opciones ofrecidas y ninguna respuesta podría acertar.

Reglas de diseño:

* La segunda opinión **nunca sustituye** al modelo entrenado: se muestra al lado para
  comparar. La verdad la da la demanda real de los días siguientes.
* Solo se consulta **cuando el equipo lo pide**: no se gastan llamadas automáticas.
* Si no hay clave o la API falla, todo lo demás sigue funcionando.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.domain.segunda_opinion import (
    banda_desde_probabilidades,
    construir_rubrica,
    usos_desde_puntuacion,
)
from mio_demanda.infrastructure.cliente_jev import ClienteJev, ErrorSegundaOpinion
from mio_demanda.infrastructure.repositorio_calendario import RepositorioCalendario
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos

_PREFIJO_EVENTO = "evento_"
_DIAS_SEMANA = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_DIAS_CONTEXTO = 14


@dataclass(slots=True)
class PrediccionJev:
    """Estimación de demanda pedida a la segunda opinión.

    Attributes:
        detalle: DataFrame con ``fecha``, ``usos_jev``, ``usos_jev_p10``,
            ``usos_jev_p90`` y ``confianza``.
        modelo: nombre del modelo que respondió.
        rubricas: niveles ofrecidos a cada día, por fecha ISO.
        confianza: confianza media de los días estimados.
        tokens_entrada: tokens facturables de la consulta.
        generado: momento de la consulta.
        etiqueta: descripción del alcance.
    """

    detalle: pd.DataFrame
    modelo: str
    rubricas: dict[str, list[int]] = field(default_factory=dict)
    confianza: float = 0.0
    tokens_entrada: int = 0
    generado: datetime = field(default_factory=datetime.now)
    etiqueta: str = ""

    @property
    def usos_diarios(self) -> float:
        """Promedio diario estimado por la segunda opinión."""
        if self.detalle.empty:
            return 0.0
        return float(self.detalle["usos_jev"].mean())

    @property
    def rango_rubrica(self) -> tuple[int, int]:
        """Rango completo cubierto por las rúbricas consultadas."""
        niveles = [valor for lista in self.rubricas.values() for valor in lista]
        if not niveles:
            return 0, 0
        return min(niveles), max(niveles)


class ServicioSegundaOpinion:
    """Pide al modelo externo su propia estimación de la demanda.

    Args:
        settings: configuración de la aplicación.
        cliente: cliente de la API (se crea si no se inyecta).
        repo: repositorio de datos (se crea si no se inyecta).
        repo_cal: repositorio del calendario (se crea si no se inyecta).
    """

    def __init__(
        self,
        settings: Settings,
        cliente: ClienteJev | None = None,
        repo: RepositorioDatos | None = None,
        repo_cal: RepositorioCalendario | None = None,
    ) -> None:
        self._settings = settings
        self._cliente = cliente or ClienteJev(settings)
        self._repo = repo or RepositorioDatos(settings)
        self._repo_cal = repo_cal or RepositorioCalendario(settings)

    @property
    def disponible(self) -> bool:
        """Indica si hay clave configurada para pedir la segunda opinión."""
        return self._cliente.configurado

    def predecir_sistema(self, desde: date, hasta: date) -> PrediccionJev:
        """Pide la estimación de la demanda del sistema completo.

        Args:
            desde: primera fecha a estimar.
            hasta: última fecha a estimar.

        Returns:
            La :class:`PrediccionJev` con el detalle diario.
        """
        return self._pedir(desde, hasta, entidad=None, tipo=None)

    def predecir_entidad(
        self, entidad: str, tipo: TipoEntidad, desde: date, hasta: date
    ) -> PrediccionJev:
        """Pide la estimación de la demanda de una estación o ruta.

        Args:
            entidad: identificador de la entidad.
            tipo: tipo de entidad.
            desde: primera fecha a estimar.
            hasta: última fecha a estimar.

        Returns:
            La :class:`PrediccionJev` con el detalle diario.
        """
        return self._pedir(desde, hasta, entidad=entidad, tipo=tipo)

    # ------------------------------------------------------------------
    # Núcleo
    # ------------------------------------------------------------------
    def _pedir(
        self,
        desde: date,
        hasta: date,
        entidad: str | None,
        tipo: TipoEntidad | None,
    ) -> PrediccionJev:
        """Arma el estado, pregunta y traduce la respuesta.

        Args:
            desde: primera fecha a estimar.
            hasta: última fecha a estimar.
            entidad: entidad a estimar (``None`` para el sistema).
            tipo: tipo de entidad (``None`` para el sistema).

        Returns:
            La :class:`PrediccionJev` lista para la interfaz.

        Raises:
            ValueError: si el horizonte supera el máximo de una consulta.
            ErrorSegundaOpinion: si no hay clave, datos o la API falla.
        """
        dias = self._contexto_dias(desde, hasta)
        if not dias:
            raise ValueError("El horizonte de la segunda opinión está vacío")

        ancla = self._nivel_reciente(entidad)
        perfiles = self._perfil_por_tipo_dia(entidad)
        etiqueta = self._etiqueta(entidad, tipo)

        # Un horizonte largo se trocea en varias consultas y luego se une: la API
        # responde mejor con grupos cortos de preguntas.
        tamano = max(1, self._settings.jev_max_dias)
        lotes = [dias[indice : indice + tamano] for indice in range(0, len(dias), tamano)]

        parciales: list[PrediccionJev] = []
        for lote in lotes:
            rubricas = {
                dia["fecha"]: construir_rubrica(
                    perfiles.get(str(dia["tipo_de_dia"]), ancla),
                    self._settings.jev_niveles,
                    self._settings.jev_amplitud,
                )
                for dia in lote
            }
            estado = self._estado(entidad, etiqueta, ancla, perfiles, lote)
            preguntas = self._preguntas(lote, rubricas, etiqueta)
            respuesta = self._cliente.preguntar(estado, preguntas)
            parciales.append(self._interpretar(respuesta, lote, rubricas, etiqueta))

        return self._combinar(parciales, etiqueta)

    @staticmethod
    def _combinar(parciales: list[PrediccionJev], etiqueta: str) -> PrediccionJev:
        """Une las respuestas de varios lotes en una sola predicción.

        Args:
            parciales: predicciones de cada lote, en orden temporal.
            etiqueta: descripción del alcance.

        Returns:
            La :class:`PrediccionJev` completa, con los tokens acumulados.
        """
        detalle = (
            pd.concat([parcial.detalle for parcial in parciales], ignore_index=True)
            .sort_values("fecha")
            .reset_index(drop=True)
        )
        rubricas = {
            fecha: niveles for parcial in parciales for fecha, niveles in parcial.rubricas.items()
        }
        return PrediccionJev(
            detalle=detalle,
            modelo=parciales[-1].modelo,
            rubricas=rubricas,
            confianza=float(detalle["confianza"].mean()) if not detalle.empty else 0.0,
            tokens_entrada=sum(parcial.tokens_entrada for parcial in parciales),
            etiqueta=etiqueta,
        )

    def _contexto_dias(self, desde: date, hasta: date) -> list[dict[str, Any]]:
        """Datos de calendario de cada día a estimar.

        Args:
            desde: primera fecha a estimar.
            hasta: última fecha a estimar.

        Returns:
            Lista de diccionarios con fecha, día de la semana, tipo de día, festivo,
            día de pago y eventos activos.
        """
        calendario = self._repo_cal.calendario_enriquecido(desde, hasta)
        dias: list[dict[str, Any]] = []
        for _, fila in calendario.iterrows():
            momento = pd.Timestamp(fila["fecha"])
            dias.append(
                {
                    "fecha": momento.date().isoformat(),
                    "dia_semana": _DIAS_SEMANA[momento.weekday()],
                    "tipo_de_dia": str(fila["dia_tipo_nom"]),
                    "es_festivo": bool(fila["es_festivo"]),
                    "es_dia_de_pago": bool(fila["es_dia_pago"]),
                    "eventos": self._eventos(fila),
                }
            )
        return dias

    def _nivel_reciente(self, entidad: str | None) -> float:
        """Nivel reciente de demanda (mediana) de todo el período.

        Args:
            entidad: entidad consultada (``None`` para el sistema).

        Returns:
            La mediana de los últimos días configurados.

        Raises:
            ErrorSegundaOpinion: si no hay datos de usos.
        """
        usos = self._usos_recientes(entidad)
        if usos.empty:
            sujeto = entidad or "el sistema"
            raise ErrorSegundaOpinion(f"Sin demanda reciente para {sujeto}.")
        return float(usos["usos"].median())

    def _usos_recientes(self, entidad: str | None) -> pd.DataFrame:
        """Demanda diaria de la ventana reciente, para el sistema o una entidad.

        Args:
            entidad: entidad consultada (``None`` para el sistema).

        Returns:
            DataFrame con ``fecha`` y ``usos``.
        """
        rango = self._repo.rango_fechas_usos()
        if rango is None:
            return pd.DataFrame(columns=["fecha", "usos"])
        fin = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin is None:
            return pd.DataFrame(columns=["fecha", "usos"])
        desde = fin - timedelta(days=self._settings.ventana_reciente_dias - 1)
        if entidad is None:
            usos = self._repo.usos_diarios(desde, fin)
        else:
            usos = self._repo.usos_diarios_por_entidad(desde, fin, [entidad])
        if usos.empty:
            return pd.DataFrame(columns=["fecha", "usos"])
        return usos.groupby("fecha", as_index=False)["usos"].sum()

    def _perfil_por_tipo_dia(self, entidad: str | None) -> dict[str, int]:
        """Mediana de demanda real por tipo de día (hábil, sábado, festivo).

        Es la base de cada rúbrica: el domingo y el festivo mueven una fracción de la
        demanda de un día hábil, así que sus opciones tienen que ser distintas.

        Args:
            entidad: entidad consultada (``None`` para el sistema).

        Returns:
            Mapa ``{"Hábil": usos, "Sábado": usos, "Festivo": usos}``.
        """
        usos = self._usos_recientes(entidad)
        if usos.empty:
            return {}
        rango = self._repo.rango_fechas_usos()
        if rango is None:
            return {}
        calendario = self._repo_cal.calendario_enriquecido(
            usos["fecha"].min().date(), usos["fecha"].max().date()
        )
        combinado = usos.merge(calendario[["fecha", "dia_tipo_nom"]], on="fecha", how="left")
        medianas = combinado.groupby("dia_tipo_nom")["usos"].median()
        return {str(nombre): round(float(valor)) for nombre, valor in medianas.items()}

    def _perfil_semanal(self, entidad: str | None) -> dict[str, int]:
        """Mediana de demanda real por día de la semana.

        Args:
            entidad: entidad consultada (``None`` para el sistema).

        Returns:
            Mapa ``{"lunes": usos, ..., "domingo": usos}``.
        """
        usos = self._usos_recientes(entidad)
        if usos.empty:
            return {}
        usos = usos.assign(dia=pd.to_datetime(usos["fecha"]).dt.weekday)
        medianas = usos.groupby("dia")["usos"].median()
        return {_DIAS_SEMANA[int(dia)]: round(float(valor)) for dia, valor in medianas.items()}

    def _estado(
        self,
        entidad: str | None,
        etiqueta: str,
        ancla: float,
        perfiles: dict[str, int],
        dias: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Construye el estado que se le entrega al modelo externo.

        No incluye nuestra predicción: la segunda opinión debe ser **a ciegas** para
        que su comparación aporte información.

        Args:
            entidad: entidad consultada (``None`` para el sistema).
            etiqueta: descripción legible del alcance.
            ancla: nivel reciente de demanda.
            perfiles: mediana de demanda por tipo de día.
            dias: contexto de calendario de los días a estimar.

        Returns:
            Diccionario con el contexto, la historia reciente y los días a estimar.
        """
        return {
            "contexto": (
                "Demanda diaria de pasajeros del sistema de transporte masivo MIO "
                "(Metrocali, Cali, Colombia). La unidad es «usos por día»."
            ),
            "alcance": etiqueta,
            "nivel_reciente_mediano": round(ancla),
            "perfil_tipico_por_tipo_de_dia": perfiles,
            "perfil_tipico_por_dia_semana": self._perfil_semanal(entidad),
            "ultimos_dias_reales": self._historia_reciente(entidad),
            "dias_a_estimar": dias,
        }

    def _historia_reciente(self, entidad: str | None) -> list[dict[str, Any]]:
        """Últimos días con demanda real, para dar contexto al modelo externo.

        Args:
            entidad: entidad consultada (``None`` para el sistema).

        Returns:
            Lista de ``{"fecha", "usos"}``.
        """
        rango = self._repo.rango_fechas_usos()
        if rango is None:
            return []
        fin = self._repo.ultima_fecha_confiable(self._settings.umbral_dia_parcial)
        if fin is None:
            return []
        desde = fin - timedelta(days=_DIAS_CONTEXTO - 1)
        if entidad is None:
            usos = self._repo.usos_diarios(desde, fin)
        else:
            usos = self._repo.usos_diarios_por_entidad(desde, fin, [entidad])
        if usos.empty:
            return []
        agrupado = usos.groupby("fecha", as_index=False)["usos"].sum().sort_values("fecha")
        return [
            {"fecha": fecha.date().isoformat(), "usos": round(float(valor))}
            for fecha, valor in zip(agrupado["fecha"], agrupado["usos"], strict=False)
        ]

    @staticmethod
    def _eventos(fila: pd.Series) -> list[str]:
        """Identificadores de los eventos de calendario activos ese día."""
        return [
            str(columna)[len(_PREFIJO_EVENTO) :]
            for columna in fila.index
            if str(columna).startswith(_PREFIJO_EVENTO) and float(fila[columna] or 0) > 0
        ]

    @staticmethod
    def _etiqueta(entidad: str | None, tipo: TipoEntidad | None) -> str:
        """Descripción legible del alcance consultado."""
        if entidad is None:
            return "el sistema completo (todos los viajes)"
        if tipo is TipoEntidad.RUTA:
            return f"la ruta {entidad}"
        if tipo is TipoEntidad.ESTACION:
            return f"la estación {entidad}"
        return f"la entidad {entidad}"

    def _preguntas(
        self,
        dias: list[dict[str, Any]],
        rubricas: dict[str, list[int]],
        etiqueta: str,
    ) -> dict[str, dict[str, Any]]:
        """Una pregunta de puntuación por día, con la rúbrica de ese día.

        Args:
            dias: contexto de los días a estimar.
            rubricas: niveles de demanda por fecha ISO.
            etiqueta: descripción del alcance.

        Returns:
            Preguntas tipadas por nombre (``dia_<fecha>``).
        """
        return {
            f"dia_{dia['fecha']}": {
                "type": "score",
                "instructions": (
                    f"¿Cuántos usos tendrá {etiqueta} el día {dia['fecha']} "
                    f"({dia['dia_semana']}, tipo de día: {dia['tipo_de_dia']})? "
                    "Elige el nivel más probable según la historia reciente, el perfil "
                    "típico de ese tipo de día, el día de pago y los eventos."
                ),
                "criteria": [f"{valor} usos" for valor in rubricas[dia["fecha"]]],
            }
            for dia in dias
        }

    def _interpretar(
        self,
        respuesta: dict[str, Any],
        dias: list[dict[str, Any]],
        rubricas: dict[str, list[int]],
        etiqueta: str,
    ) -> PrediccionJev:
        """Traduce la respuesta de la API a una tabla diaria de usos.

        Args:
            respuesta: respuesta completa de la API.
            dias: contexto de los días estimados.
            rubricas: niveles de demanda por fecha ISO.
            etiqueta: descripción del alcance.

        Returns:
            La :class:`PrediccionJev`.

        Raises:
            ErrorSegundaOpinion: si la API no devolvió ninguna estimación utilizable.
        """
        respuestas = respuesta.get("answers") or {}
        filas: list[dict[str, Any]] = []
        for dia in dias:
            contestada = respuestas.get(f"dia_{dia['fecha']}")
            rubrica = rubricas[dia["fecha"]]
            if not contestada:
                continue
            probabilidades = self._probabilidades(contestada, len(rubrica))
            p10, p90 = banda_desde_probabilidades(probabilidades, rubrica)
            filas.append(
                {
                    "fecha": pd.Timestamp(dia["fecha"]),
                    "usos_jev": usos_desde_puntuacion(float(contestada.get("score", 0.0)), rubrica),
                    "usos_jev_p10": p10,
                    "usos_jev_p90": p90,
                    "confianza": float(contestada.get("confidence") or 0.0),
                }
            )

        if not filas:
            raise ErrorSegundaOpinion("La segunda opinión no devolvió estimaciones utilizables.")

        detalle = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)
        uso = respuesta.get("usage") or {}
        return PrediccionJev(
            detalle=detalle,
            modelo=str(respuesta.get("model", self._settings.jev_modelo)),
            rubricas=dict(rubricas),
            confianza=float(detalle["confianza"].mean()),
            tokens_entrada=int(uso.get("input_tokens", 0) or 0),
            etiqueta=etiqueta,
        )

    @staticmethod
    def _probabilidades(contestada: dict[str, Any], niveles: int) -> list[float]:
        """Probabilidades por nivel, en el orden de la rúbrica.

        Args:
            contestada: respuesta de un día.
            niveles: número de niveles de la rúbrica.

        Returns:
            Lista de probabilidades alineada con la rúbrica.
        """
        crudas = dict(contestada.get("probabilities") or {})
        return [float(crudas.get(str(indice), 0.0) or 0.0) for indice in range(niveles)]
