"""Servicio de calendario: eventos base, eventos personalizados y vista mensual.

Caso de uso que consume la pestaña de Entrenamiento (zona del profesional):

* **Eventos base**: el catálogo declarativo de ``config/eventos_base.json``
  expandido a fechas reales (Semana Santa, Feria de Cali, comerciales, recesos
  escolares, elecciones). Son de solo lectura desde la aplicación.
* **Eventos personalizados**: los que registra el profesional (ferias locales,
  obras, paros). Se persisten y alimentan el siguiente reentrenamiento.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.entidades import CategoriaEvento, EventoCalendario, PerfilImpacto
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_calendario import RepositorioCalendario

logger = obtener_logger(__name__)

# Etiquetas legibles para las categorías de evento.
_ETIQUETAS_CATEGORIA = {
    CategoriaEvento.FERIA: "Feria",
    CategoriaEvento.FESTIVO_LOCAL: "Festivo local",
    CategoriaEvento.EVENTO_MASIVO: "Evento masivo",
    CategoriaEvento.OBRA: "Obra",
    CategoriaEvento.PARO: "Paro",
    CategoriaEvento.CLIMA: "Clima",
    CategoriaEvento.DESASTRE: "Desastre",
    CategoriaEvento.OTRO: "Otro",
}

#: Etiquetas legibles para los perfiles de impacto.
ETIQUETAS_PERFIL = {
    PerfilImpacto.CONSTANTE: "Constante",
    PerfilImpacto.DECRECIENTE: "Decreciente",
    PerfilImpacto.CRECIENTE: "Creciente",
}

#: Inversos: etiqueta legible -> valor del modelo (para el editor tipo Excel).
_VALOR_PERFIL = {etiqueta: perfil.value for perfil, etiqueta in ETIQUETAS_PERFIL.items()}
_VALOR_CATEGORIA = {
    etiqueta: categoria.value for categoria, etiqueta in _ETIQUETAS_CATEGORIA.items()
}

#: Perfiles válidos para el editor (etiquetas).
OPCIONES_PERFIL = list(ETIQUETAS_PERFIL.values())
#: Categorías válidas para el editor (etiquetas).
OPCIONES_CATEGORIA = list(_ETIQUETAS_CATEGORIA.values())


class ServicioCalendario:
    """Gestiona el calendario operativo y los eventos del profesional.

    Args:
        settings: configuración de la aplicación.
        repo_cal: repositorio del calendario (inyectable para pruebas).
    """

    def __init__(self, settings: Settings, repo_cal: RepositorioCalendario | None = None) -> None:
        self._settings = settings
        self._repo_cal = repo_cal or RepositorioCalendario(settings)

    # ------------------------------------------------------------------
    # Eventos personalizados
    # ------------------------------------------------------------------
    def eventos(self) -> list[dict[str, object]]:
        """Lista los eventos del usuario de **fechas fijas** (serializables).

        Returns:
            Lista de diccionarios ordenada por fecha de inicio.
        """
        datos = [asdict(evento) for evento in self._repo_cal.eventos_personalizados()]
        datos.sort(key=lambda e: str(e["fecha_inicio"]))
        return datos

    def eventos_usuario(self, anio: int) -> pd.DataFrame:
        """Tabla de **todos** los eventos del usuario vigentes en un año.

        Incluye los recurrentes (expandidos a fechas del año) y los de fechas fijas.

        Args:
            anio: año a mostrar.

        Returns:
            DataFrame con ``Id``, ``Evento``, ``Inicio``, ``Fin``, ``Dias``,
            ``Impacto``, ``Perfil``, ``Categoria``, ``Fuente`` y ``Definicion``
            (``Recurrente``, ``Puntual`` o ``Fechas fijas``), ordenado por fecha de inicio.
        """
        eventos = self._repo_cal.eventos_usuario(date(anio, 1, 1), date(anio, 12, 31))
        recurrente_ids: set[str] = set()
        puntual_ids: set[str] = set()
        for definicion in self._repo_cal.definiciones_usuario():
            if "regla" not in definicion:
                continue
            identificador = str(definicion["id"])
            if str(definicion["regla"].get("tipo")) == "fechas_puntuales":
                puntual_ids.add(identificador)
            else:
                recurrente_ids.add(identificador)
        tabla = self._tabla_eventos(eventos)
        if tabla.empty:
            return tabla

        def _etiqueta(identificador: str) -> str:
            if identificador in puntual_ids:
                return "Puntual"
            if identificador in recurrente_ids:
                return "Recurrente"
            return "Fechas fijas"

        tabla["Definicion"] = [_etiqueta(identificador) for identificador in tabla["Id"]]
        return tabla

    def total_eventos_usuario(self) -> int:
        """Cuántas definiciones de eventos tiene el usuario (recurrentes y fijas).

        Returns:
            Número de definiciones en ``config/eventos_personalizados.json``.
        """
        return len(self._repo_cal.definiciones_usuario())

    def crear_evento(
        self,
        nombre: str,
        fecha_inicio: date,
        fecha_fin: date,
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Registra un nuevo evento personalizado.

        Args:
            nombre: nombre legible del evento.
            fecha_inicio: primer día del evento.
            fecha_fin: último día del evento.
            impacto: peso del ajuste homólogo en [0, 1].
            categoria: categoría del evento.
            fuente: justificación o fuente.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El evento creado.
        """
        return self._repo_cal.agregar_evento(
            nombre, fecha_inicio, fecha_fin, impacto, categoria, fuente, perfil_impacto
        )

    def crear_evento_recurrente(
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
        """Registra un evento que se repite **todos los años** (mismo mes y día).

        Args:
            nombre: nombre legible del evento.
            mes_inicio: mes de inicio (1-12).
            dia_inicio: día de inicio.
            mes_fin: mes de fin (1-12).
            dia_fin: día de fin.
            impacto: importancia relativa en [0, 1].
            categoria: categoría del evento.
            fuente: justificación o fuente.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El evento resuelto para el año actual.
        """
        return self._repo_cal.agregar_evento_recurrente(
            nombre=nombre,
            mes_inicio=mes_inicio,
            dia_inicio=dia_inicio,
            mes_fin=mes_fin,
            dia_fin=dia_fin,
            impacto=impacto,
            categoria=categoria,
            fuente=fuente,
            perfil_impacto=perfil_impacto,
        )

    def crear_evento_con_ventana(
        self,
        nombre: str,
        fecha_central: date,
        dias_antes: int,
        dias_despues: int,
        impacto: float = 0.5,
        categoria: CategoriaEvento = CategoriaEvento.OTRO,
        fuente: str = "",
        perfil_impacto: PerfilImpacto = PerfilImpacto.CONSTANTE,
    ) -> EventoCalendario:
        """Registra un evento que afecta días antes y después de una fecha central.

        Útil para eventos cuyo efecto no se limita al día en que ocurren (una
        feria, un paro, una jornada electoral).

        Args:
            nombre: nombre legible del evento.
            fecha_central: fecha de referencia del evento.
            dias_antes: días de afectación previos a la fecha central.
            dias_despues: días de afectación posteriores a la fecha central.
            impacto: peso del ajuste homólogo en [0, 1].
            categoria: categoría del evento.
            fuente: justificación o fuente.
            perfil_impacto: evolución del impacto a lo largo de los días.

        Returns:
            El evento creado.
        """
        inicio = fecha_central - timedelta(days=max(0, dias_antes))
        fin = fecha_central + timedelta(days=max(0, dias_despues))
        return self.crear_evento(
            nombre=nombre,
            fecha_inicio=inicio,
            fecha_fin=fin,
            impacto=impacto,
            categoria=categoria,
            fuente=fuente,
            perfil_impacto=perfil_impacto,
        )

    def eliminar_evento(self, evento_id: str) -> bool:
        """Elimina un evento personalizado.

        Args:
            evento_id: identificador del evento.

        Returns:
            ``True`` si existía y fue eliminado.
        """
        return self._repo_cal.eliminar_evento(evento_id)

    # ------------------------------------------------------------------
    # Eventos base (parametrizados)
    # ------------------------------------------------------------------
    def eventos_base(self, anio: int) -> pd.DataFrame:
        """Expande los eventos base de un año a una tabla legible.

        Args:
            anio: año a mostrar.

        Returns:
            DataFrame con ``Evento``, ``Inicio``, ``Fin``, ``Dias``, ``Impacto``,
            ``Categoria`` y ``Fuente``, ordenado por fecha de inicio.
        """
        eventos = self._repo_cal.eventos_base(date(anio, 1, 1), date(anio, 12, 31))
        return self._tabla_eventos(eventos)

    def total_eventos_base(self) -> int:
        """Devuelve cuántos eventos base están configurados.

        Returns:
            Número de definiciones en ``config/eventos_base.json``.
        """
        return len(self._repo_cal.definiciones_base())

    def ruta_archivo_eventos(self) -> str:
        """Devuelve la ruta del archivo de eventos personalizados.

        Returns:
            Ruta del JSON donde se persisten los eventos del profesional.
        """
        return str(self._repo_cal.ruta_eventos())

    def actualizar_eventos(self, tabla: pd.DataFrame) -> int:
        """Guarda los cambios hechos en el editor tipo Excel.

        Traduce las etiquetas de la tabla (Perfil, Categoría) a los valores del
        modelo y delega en el repositorio.

        Args:
            tabla: tabla editada con las columnas ``Id``, ``Impacto``, ``Perfil``,
                ``Categoria``, ``Fuente`` y, opcionalmente, ``Inicio``/``Fin``.

        Returns:
            Número de eventos actualizados.

        Raises:
            RuntimeError: si el archivo de eventos no se puede leer.
        """
        cambios = []
        for fila in tabla.to_dict(orient="records"):
            cambios.append(
                {
                    "Id": fila.get("Id"),
                    "Nombre": fila.get("Evento"),
                    "Impacto": fila.get("Impacto"),
                    "Perfil": _VALOR_PERFIL.get(str(fila.get("Perfil", "")), ""),
                    "Categoria": _VALOR_CATEGORIA.get(str(fila.get("Categoria", "")), ""),
                    "Fuente": fila.get("Fuente"),
                    "Inicio": fila.get("Inicio"),
                    "Fin": fila.get("Fin"),
                }
            )
        return self._repo_cal.actualizar_eventos(cambios)

    def catalogo_editable(self, desde: date, hasta: date) -> pd.DataFrame:
        """Tabla con **todos** los eventos (estándar y del usuario) para editar.

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            DataFrame con ``Id``, ``Evento``, ``Inicio``, ``Fin``, ``Impacto``,
            ``Perfil``, ``Categoria``, ``Fuente``, ``Definicion`` y ``Origen``.
        """
        base = self._repo_cal.eventos_base(desde, hasta)
        propios = self._repo_cal.eventos_usuario(desde, hasta)
        tabla_base = self._tabla_eventos(base)
        tabla_propios = self._tabla_eventos(propios)
        if not tabla_base.empty:
            tabla_base["Origen"] = "Estándar"
        if not tabla_propios.empty:
            tabla_propios["Origen"] = "Usuario"
        return pd.concat([tabla_base, tabla_propios], ignore_index=True).sort_values("Inicio")

    def catalogo_eventos(self, desde: date, hasta: date) -> pd.DataFrame:
        """Tabla con los eventos vigentes (base expandidos + personalizados).

        Args:
            desde: fecha inicial (inclusive).
            hasta: fecha final (inclusive).

        Returns:
            DataFrame con las mismas columnas que :meth:`eventos_base` más
            ``Origen`` (``Estándar`` o ``Usuario``).
        """
        base = self._repo_cal.eventos_base(desde, hasta)
        propios = self._repo_cal.eventos_usuario(desde, hasta)
        tabla_base = self._tabla_eventos(base)
        tabla_propios = self._tabla_eventos(propios)
        if not tabla_base.empty:
            tabla_base["Origen"] = "Estándar"
        if not tabla_propios.empty:
            tabla_propios["Origen"] = "Usuario"
        return pd.concat([tabla_base, tabla_propios], ignore_index=True).sort_values("Inicio")

    @staticmethod
    def _tabla_eventos(eventos: list[EventoCalendario]) -> pd.DataFrame:
        """Convierte una lista de eventos en una tabla legible."""
        columnas = [
            "Id",
            "Evento",
            "Inicio",
            "Fin",
            "Dias",
            "Impacto",
            "Perfil",
            "Categoria",
            "Fuente",
        ]
        if not eventos:
            return pd.DataFrame(columns=columnas)
        tabla = pd.DataFrame(
            [
                {
                    "Id": evento.id,
                    "Evento": evento.nombre,
                    "Inicio": evento.fecha_inicio.isoformat(),
                    "Fin": evento.fecha_fin.isoformat(),
                    "Dias": evento.dias,
                    "Impacto": round(evento.impacto, 2),
                    "Perfil": ETIQUETAS_PERFIL.get(
                        evento.perfil_impacto, evento.perfil_impacto.value
                    ),
                    "Categoria": _ETIQUETAS_CATEGORIA.get(evento.categoria, evento.categoria.value),
                    "Fuente": evento.fuente,
                }
                for evento in eventos
            ]
        )
        return tabla.sort_values("Inicio").reset_index(drop=True)[columnas]

    # ------------------------------------------------------------------
    # Vista mensual
    # ------------------------------------------------------------------
    def calendario_mes(self, anio: int, mes: int) -> pd.DataFrame:
        """Devuelve el calendario de un mes con los eventos que lo afectan.

        Args:
            anio: año del mes solicitado.
            mes: mes (1-12).

        Returns:
            DataFrame con ``fecha``, ``dia_tipo_nom``, ``es_fin_semana`` y
            ``eventos`` (nombres de los eventos activos ese día).
        """
        inicio = date(anio, mes, 1)
        fin = (pd.Timestamp(inicio) + pd.offsets.MonthEnd(0)).date()
        calendario = self._repo_cal.calendario_enriquecido(inicio, fin)

        columnas_eventos = [c for c in calendario.columns if c.startswith("evento_")]
        nombres = self._nombres_eventos(inicio, fin)

        def _etiquetas(fila: pd.Series) -> str:
            activos = [nombres.get(c, c) for c in columnas_eventos if fila.get(c, 0) == 1]
            return ", ".join(activos)

        resultado = calendario[["fecha", "dia_tipo_nom", "es_fin_semana"]].copy()
        resultado["es_fin_semana"] = resultado["es_fin_semana"].astype(bool)
        resultado["eventos"] = calendario.apply(_etiquetas, axis=1)
        return resultado

    def etiquetas_barras(self, desde: date, hasta: date) -> pd.DataFrame:
        """Devuelve, por fecha, la etiqueta de evento y su impacto (para gráficas).

        Args:
            desde: fecha inicial.
            hasta: fecha final.

        Returns:
            DataFrame con ``fecha``, ``etiqueta`` e ``impacto``.
        """
        calendario = self._repo_cal.calendario_enriquecido(desde, hasta)
        nombres = self._nombres_eventos(desde, hasta)

        def _etiqueta(fila: pd.Series) -> str:
            activos = [nombres[columna] for columna in nombres if fila.get(columna, 0) == 1]
            return ", ".join(activos)

        return pd.DataFrame(
            {
                "fecha": calendario["fecha"],
                "etiqueta": calendario.apply(_etiqueta, axis=1),
                "impacto": calendario["impacto_evento"].astype(float),
            }
        )

    def _nombres_eventos(self, desde: date, hasta: date) -> dict[str, str]:
        """Mapa ``evento_<id>`` -> nombre legible para los eventos vigentes."""
        nombres = {
            self._columna_evento(evento.id): evento.nombre
            for evento in self._repo_cal.eventos_vigentes(desde, hasta)
        }
        # Los eventos configurados que no ocurren en el rango igual deben tener nombre.
        for identificador in self._repo_cal.ids_eventos():
            nombres.setdefault(self._columna_evento(identificador), identificador)
        return nombres

    @staticmethod
    def _columna_evento(evento_id: str) -> str:
        return f"evento_{evento_id}"
