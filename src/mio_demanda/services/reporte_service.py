"""Servicio de reporte: construye el libro de Excel de la proyección.

Replica la estructura del archivo institucional
``proyeccion_demanda_2026_final.xlsx``:

======================  ==========================================================
``Diario_<año>``        Detalle diario con el calendario, la demanda real, la
                        proyectada, la final, el tipo de registro y la confiabilidad.
``Evaluacion_Modelo``   Comparación real vs. predicho (solo si el horizonte
                        contiene días con demanda observada).
``Resumen_Mensual``     Totales y promedios por mes.
``Resumen_Confiabilidad``  Desempeño por tipo de día (días con demanda real).
``Metricas``            Métricas de validación del modelo consumido.
======================  ==========================================================

El servicio devuelve un diccionario ``nombre de hoja -> DataFrame``; la interfaz
solo lo entrega a ``ExcelWriter``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mio_demanda.config import Settings
from mio_demanda.domain.metricas import (
    CONFIABILIDAD_MAXIMA,
    CONFIABILIDAD_MINIMA,
    clasificar_confiabilidad,
)
from mio_demanda.domain.vigilancia import evaluar_desajuste, sesgo_relativo
from mio_demanda.infrastructure.logging_utils import obtener_logger

logger = obtener_logger(__name__)

_MESES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]
_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

_TIPOS_DIA = {0: "Hábil", 1: "Sábado", 2: "Festivo"}


class ServicioReporte:
    """Construye los libros de Excel de las proyecciones.

    Args:
        settings: configuración de la aplicación.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Libro de la proyección general (sistema)
    # ------------------------------------------------------------------
    def libro_general(
        self,
        resultado,
        segunda_opinion: pd.DataFrame | None = None,
        retrospectiva=None,
    ) -> dict[str, pd.DataFrame]:
        """Construye las hojas del Excel para la proyección del sistema.

        Args:
            resultado: :class:`~mio_demanda.services.proyeccion_service.ProyeccionGeneral`.
            segunda_opinion: detalle diario de la segunda opinión (Jev), si se pidió.
                Sus columnas se añaden al lado de las del modelo.
            retrospectiva: comparación real vs. proyectado de los últimos días. Sus
                filas van delante en la primera hoja y su veredicto a ``Metricas``.

        Returns:
            Diccionario ``nombre de hoja -> DataFrame``.
        """
        diario = self._tabla_diaria(resultado.detalle, resultado.confiabilidad, segunda_opinion)
        hojas = self._hojas_comunes(
            diario,
            resultado.modelo,
            resultado.metricas,
            resultado.confiabilidad,
            retrospectiva,
            entrenamiento={
                "fecha": resultado.fecha_entrenamiento,
                "desde": resultado.desde_entrenamiento,
                "hasta": resultado.hasta_entrenamiento,
            },
        )
        hojas["Proyeccion_por_tipo_dia"] = pd.DataFrame(
            resultado.resumen.detalle_por_tipo_dia
        ).T.reset_index(names="Dia.tipo")
        horaria = self._tabla_horaria(resultado.detalle_horario, resultado.detalle, segunda_opinion)
        if not horaria.empty:
            hojas["Diario_Horario"] = horaria
        return hojas

    # ------------------------------------------------------------------
    # Libro de la proyección por entidad (estación / ruta)
    # ------------------------------------------------------------------
    def libro_entidad(
        self,
        resultado,
        segunda_opinion: pd.DataFrame | None = None,
        retrospectiva=None,
        horario_entidades: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Construye las hojas del Excel de una entidad o de un grupo de entidades.

        El libro es corto a propósito: el diario, su resumen mensual, las métricas del
        modelo y —cuando se proyecta un grupo— el detalle por entidad con sus horas.

        Args:
            resultado: :class:`~mio_demanda.services.proyeccion_service.ProyeccionPorEntidad`.
            segunda_opinion: detalle diario de la segunda opinión (Jev), si se pidió.
            retrospectiva: comparación real vs. proyectado de los últimos días.
            horario_entidades: detalle horario por entidad (``fecha``, ``entidad``,
                ``hora``, ``usos_hora``) para la hoja ``Diario_Entidades``.

        Returns:
            Diccionario ``nombre de hoja -> DataFrame``.
        """
        diario = self._tabla_diaria(
            resultado.serie_diaria, resultado.confiabilidad, segunda_opinion
        )
        anio = int(pd.to_datetime(diario["Fecha"]).dt.year.max())
        compacto = diario
        if retrospectiva is not None:
            compacto = pd.concat(
                [
                    self._tabla_retrospectiva(retrospectiva, resultado.confiabilidad),
                    diario,
                ],
                ignore_index=True,
            )

        hojas: dict[str, pd.DataFrame] = {
            f"Diario_{anio}": self._tabla_compacta(compacto),
            "Resumen_Mensual": self._tabla_resumen_mensual(diario),
            "Metricas": self._tabla_metricas(
                str(resultado.modelo).capitalize(),
                resultado.metricas,
                resultado.confiabilidad,
                entrenamiento={
                    "fecha": resultado.fecha_entrenamiento,
                    "desde": resultado.desde_entrenamiento,
                    "hasta": resultado.hasta_entrenamiento,
                },
            ),
        }
        if retrospectiva is not None:
            hojas["Metricas"] = pd.concat(
                [
                    hojas["Metricas"],
                    self._tabla_vigilancia(retrospectiva, resultado.metricas),
                ],
                ignore_index=True,
            )
        if horario_entidades is not None and not horario_entidades.empty:
            hojas["Diario_Entidades"] = self._tabla_horaria_por_entidad(horario_entidades)
        return hojas

    # ------------------------------------------------------------------
    # Libro de «Todas» (estaciones + rutas)
    # ------------------------------------------------------------------
    def libro_conjunta(
        self,
        conjunto,
        horario_entidades: pd.DataFrame | None = None,
        serie: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Construye el libro de «Todas»: estaciones y rutas en el mismo archivo.

        Lleva el diario de la suma, su resumen mensual, las métricas de **los dos
        modelos** y el detalle por entidad con sus horas.

        Args:
            conjunto: :class:`~mio_demanda.services.proyeccion_service.ProyeccionConjunta`.
            horario_entidades: detalle horario por entidad de ambos grupos, con la
                columna ``grupo`` (``Estaciones`` / ``Rutas``).
            serie: suma diaria ya calculada en la vista (``fecha``, ``usos_predicho``).

        Returns:
            Diccionario ``nombre de hoja -> DataFrame``.
        """
        grupos = conjunto.grupos()
        if serie is None:
            serie = self._agregado_conjunto(conjunto).rename(
                columns={"Fecha": "fecha", "Suma": "usos_predicho"}
            )
        confiabilidad = min(proyeccion.confiabilidad for proyeccion in grupos.values())
        diario = self._tabla_diaria(serie, confiabilidad, None)
        anio = int(pd.to_datetime(diario["Fecha"]).dt.year.max())

        hojas: dict[str, pd.DataFrame] = {
            f"Diario_{anio}": self._tabla_compacta(diario),
            "Resumen_Mensual": self._tabla_resumen_mensual(diario),
            "Metricas": self._metricas_conjunta(conjunto),
        }
        if horario_entidades is not None and not horario_entidades.empty:
            hojas["Diario_Entidades"] = self._tabla_horaria_por_entidad(horario_entidades)
        return hojas

    def _agregado_conjunto(self, conjunto) -> pd.DataFrame:
        """Hoja ``Agregado``: suma diaria de estaciones y rutas.

        Args:
            conjunto: proyección conjunta.

        Returns:
            DataFrame con fecha, estaciones, rutas y suma.
        """
        partes = [
            proyeccion.serie_diaria[["fecha", "usos_predicho"]]
            .rename(columns={"usos_predicho": nombre})
            .set_index("fecha")
            for nombre, proyeccion in (
                ("Estaciones", conjunto.estaciones),
                ("Rutas", conjunto.rutas),
            )
            if proyeccion is not None
        ]
        if not partes:
            return pd.DataFrame()
        unido = pd.concat(partes, axis=1).sort_index()
        unido["Suma"] = unido.sum(axis=1, min_count=1)
        unido = unido.round(0).astype("Int64").reset_index()
        unido["fecha"] = pd.to_datetime(unido["fecha"]).dt.strftime("%Y-%m-%d")
        return unido.rename(columns={"fecha": "Fecha"})

    def _metricas_conjunta(self, conjunto) -> pd.DataFrame:
        """Hoja ``Metricas``: una fila por modelo (estaciones y rutas).

        Args:
            conjunto: proyección conjunta.

        Returns:
            DataFrame con las métricas de cada modelo disponible.
        """
        filas = []
        for nombre, proyeccion in (
            ("Estaciones", conjunto.estaciones),
            ("Rutas", conjunto.rutas),
        ):
            if proyeccion is None:
                continue
            metricas = proyeccion.metricas
            filas.append(
                {
                    "Modelo": nombre,
                    "WAPE (%)": round(float(metricas.get("wape", float("nan"))), 2),
                    "MAE": round(float(metricas.get("mae", float("nan"))), 0),
                    "RMSE": round(float(metricas.get("rmse", float("nan"))), 0),
                    "R2": round(float(metricas.get("r2", float("nan"))), 4),
                    "Confiabilidad": f"{proyeccion.confiabilidad:.0%}",
                    "Entrenado": proyeccion.fecha_entrenamiento,
                    "Desde": proyeccion.desde_entrenamiento,
                    "Hasta": proyeccion.hasta_entrenamiento,
                }
            )
        return pd.DataFrame(filas)

    # ------------------------------------------------------------------
    # Hojas comunes
    # ------------------------------------------------------------------
    def _hojas_comunes(
        self,
        diario: pd.DataFrame,
        modelo: str,
        metricas: dict[str, float],
        confiabilidad: float,
        retrospectiva=None,
        entrenamiento: dict[str, object] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Ensambla las hojas compartidas por ambos libros.

        Args:
            diario: tabla diaria ya normalizada (columnas del Excel).
            modelo: nombre del paquete de modelo.
            metricas: métricas de validación del modelo.
            confiabilidad: confiabilidad global (1 - WAPE).
            retrospectiva: comparación real vs. proyectado de los últimos días, si se
                pidió. Sus filas se añaden delante en la primera hoja (compacta) y su
                veredicto va a la hoja ``Metricas``.

        Returns:
            Diccionario con ``Diario_<año>`` y, cuando aplica,
            ``Evaluacion_Modelo``, ``Resumen_Mensual``, ``Resumen_Confiabilidad``
            y ``Metricas``.
        """
        anio = int(pd.to_datetime(diario["Fecha"]).dt.year.max())
        compacto = diario
        if retrospectiva is not None:
            compacto = pd.concat(
                [self._tabla_retrospectiva(retrospectiva, confiabilidad), diario],
                ignore_index=True,
            )
        hojas: dict[str, pd.DataFrame] = {
            f"Diario_{anio}": self._tabla_compacta(compacto),
            f"Diario_Detallado_{anio}": diario,
        }

        evaluacion = self._tabla_evaluacion(diario)
        if not evaluacion.empty:
            hojas["Evaluacion_Modelo"] = evaluacion
            hojas["Resumen_Confiabilidad"] = self._tabla_confiabilidad(diario)
        else:
            hojas["Resumen_Confiabilidad"] = self._tabla_confiabilidad_estimada(
                diario, confiabilidad
            )

        hojas["Resumen_Mensual"] = self._tabla_resumen_mensual(diario)
        hojas["Metricas"] = self._tabla_metricas(modelo, metricas, confiabilidad, entrenamiento)
        if retrospectiva is not None:
            hojas["Metricas"] = pd.concat(
                [hojas["Metricas"], self._tabla_vigilancia(retrospectiva, metricas)],
                ignore_index=True,
            )
        return hojas

    def _tabla_horaria_por_entidad(self, horario: pd.DataFrame) -> pd.DataFrame:
        """Hoja por entidad con una columna por hora (misma lectura que ``Diario_Horario``).

        Args:
            horario: detalle con ``fecha``, ``entidad``, ``hora`` y ``usos_hora``; si
                trae la columna ``grupo`` (Estaciones / Rutas) se conserva.

        Returns:
            Una fila por entidad y día, con las 24 horas como columnas.
        """
        tabla = horario.copy()
        tabla["Fecha"] = pd.to_datetime(tabla["fecha"]).dt.strftime("%Y-%m-%d")
        tabla["Entidad"] = tabla["entidad"]
        tabla["Hora"] = pd.to_numeric(tabla["hora"], errors="coerce").astype("Int64")
        tabla["Usos"] = pd.to_numeric(tabla["usos_hora"], errors="coerce").round(0)

        indices = ["Fecha"]
        if "grupo" in tabla.columns:
            tabla["Grupo"] = tabla["grupo"]
            indices.append("Grupo")
        indices.append("Entidad")

        pivote = tabla.pivot_table(
            index=indices, columns="Hora", values="Usos", aggfunc="sum"
        ).fillna(0)
        pivote = self._horas_de_la_jornada(pivote)
        pivote.columns = [str(int(columna)) for columna in pivote.columns]
        return pivote.round(0).astype("Int64").reset_index()

    def _horas_de_la_jornada(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Recorta las columnas de hora al rango operativo configurado.

        Args:
            tabla: tabla con las horas (numéricas) como columnas.

        Returns:
            La misma tabla, solo con las horas entre ``hora_inicio_jornada`` y
            ``hora_fin_jornada``.
        """
        inicio = self._settings.hora_inicio_jornada
        fin = self._settings.hora_fin_jornada
        horas = [
            columna
            for columna in tabla.columns
            if isinstance(columna, (int, np.integer)) and inicio <= int(columna) <= fin
        ]
        return tabla.loc[:, horas] if horas else tabla

    def _tabla_horaria(
        self,
        horario: pd.DataFrame,
        diaria: pd.DataFrame,
        segunda_opinion: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Hoja simple: una fila por día y una columna por hora con los usos previstos.

        El modelo predice el **total del día**; las horas son ese total repartido según
        el perfil histórico del tipo de día.

        Args:
            horario: detalle horario (``fecha``, ``hora``, ``usos_predicho``).
            diaria: detalle diario (``fecha``, ``usos_predicho``).
            segunda_opinion: detalle de la segunda opinión, si se pidió.

        Returns:
            La tabla diaria-horaria, o un DataFrame vacío si no hay detalle horario.
        """
        if horario is None or horario.empty:
            return pd.DataFrame()

        diaria = diaria.copy()
        diaria["fecha"] = pd.to_datetime(diaria["fecha"])
        pivote = horario.copy()
        pivote["fecha"] = pd.to_datetime(pivote["fecha"])
        tabla = pivote.pivot_table(
            index="fecha", columns="hora", values="usos_predicho", aggfunc="sum"
        )
        tabla = self._horas_de_la_jornada(tabla)

        salida = diaria[["fecha"]].copy()
        salida["Fecha"] = salida["fecha"].dt.strftime("%Y-%m-%d")
        if "dia_tipo_nom" in diaria.columns:
            salida["Dia.tipo"] = diaria["dia_tipo_nom"].to_numpy()
        salida["Modelo"] = pd.to_numeric(diaria["usos_predicho"], errors="coerce").round(0)
        if segunda_opinion is not None and not segunda_opinion.empty:
            jev = segunda_opinion[["fecha", "usos_jev"]].copy()
            jev["fecha"] = pd.to_datetime(jev["fecha"])
            salida = salida.merge(jev, on="fecha", how="left")
            salida["Jev"] = pd.to_numeric(salida["usos_jev"], errors="coerce").round(0)
        for hora in tabla.columns:
            valores = tabla[hora].reindex(salida["fecha"]).to_numpy()
            salida[str(int(hora))] = np.round(valores, 0)

        return salida.drop(columns=["fecha", "usos_jev"], errors="ignore")

    def _tabla_retrospectiva(self, retrospectiva, confiabilidad: float) -> pd.DataFrame:
        """Normaliza la retrospectiva (proyección hecha N días antes) como filas del diario.

        La demanda real manda en esas fechas; la proyección de entonces va a una
        columna aparte (``Demanda_Retro``) para poder compararlas lado a lado.

        Args:
            retrospectiva: comparación real vs. proyectado de los últimos días.
            confiabilidad: confiabilidad global del modelo.

        Returns:
            Tabla normalizada con ``Demanda_Retro``.
        """
        comparado = retrospectiva.comparacion.rename(columns={"Demanda_Predicha": "retro"})
        preparado = pd.DataFrame(
            {
                "fecha": pd.to_datetime(comparado["Fecha"]),
                "dia_tipo_num": comparado["Dia.tipo"]
                .map({"Hábil": 0, "Sábado": 1, "Festivo": 2})
                .fillna(0)
                .astype(int),
                "usos_predicho": float("nan"),
                "demanda_real": pd.to_numeric(comparado["Demanda_Real"], errors="coerce"),
                "es_proyectado": False,
            }
        )
        normalizado = self._tabla_diaria(preparado, confiabilidad)
        normalizado["Demanda_Retro"] = pd.to_numeric(comparado["retro"], errors="coerce").to_numpy()
        return normalizado

    def _tabla_vigilancia(self, retrospectiva, metricas: dict[str, float]) -> pd.DataFrame:
        """Filas del veredicto de vigilancia para la hoja ``Metricas``.

        Args:
            retrospectiva: comparación real vs. proyectado de los últimos días.
            metricas: métricas de validación del modelo guardado.

        Returns:
            DataFrame con ``Metrica`` y ``Valor`` listo para concatenar.
        """
        settings = self._settings
        estado, motivo = evaluar_desajuste(
            wape_reciente=float(retrospectiva.metricas.get("wape", 0.0)),
            wape_validacion=float(metricas.get("wape", 0.0)),
            sesgo=sesgo_relativo(
                retrospectiva.comparacion["Demanda_Predicha"].astype(float),
                retrospectiva.comparacion["Demanda_Real"].astype(float),
            ),
            dias_evaluados=retrospectiva.dias_evaluados,
            min_dias=settings.vigilancia_min_dias,
            factor_alerta=settings.vigilancia_factor_alerta,
            sesgo_alerta=settings.vigilancia_sesgo_alerta,
        )
        return pd.DataFrame(
            {
                "Metrica": [
                    "Vigilancia: dias comparados",
                    "Vigilancia: rango comparado",
                    "Vigilancia: estado",
                    "Vigilancia: motivo",
                ],
                "Valor": [
                    retrospectiva.dias_evaluados,
                    f"{retrospectiva.desde} a {retrospectiva.hasta}",
                    estado.etiqueta,
                    motivo,
                ],
            }
        )

    def _tabla_compacta(self, diario: pd.DataFrame) -> pd.DataFrame:
        """Primera hoja del libro: una fila por día con las dos predicciones.

        El detalle institucional completo se conserva en ``Diario_Detallado_<año>``;
        esta primera hoja es la que se mira: fecha, modelo local, segunda opinión (si
        se pidió) y las métricas clave de cada día.

        Args:
            diario: tabla diaria completa, ya normalizada.

        Returns:
            La tabla reducida y con nombres legibles.
        """
        nombres = {
            "Fecha": "Fecha",
            "Dia.tipo": "Dia tipo",
            "Demanda_Real": "Real (usos)",
            "Demanda_Retro": "Proyectado hace N dias",
            "Demanda_Proyectada": "Modelo local",
            "Demanda_P10": "Modelo P10",
            "Demanda_P90": "Modelo P90",
            "Demanda_Jev": "Jev",
            "Jev_P10": "Jev P10",
            "Jev_P90": "Jev P90",
            "Diferencia_Jev_%": "Dif. modelo vs Jev (%)",
            "Confiabilidad_%": "Confiabilidad (%)",
        }
        presentes = [columna for columna in nombres if columna in diario.columns]
        return diario[presentes].rename(columns=nombres)

    def _tabla_diaria(
        self,
        detalle: pd.DataFrame,
        confiabilidad: float,
        segunda_opinion: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Normaliza el detalle diario al formato del Excel institucional.

        Args:
            detalle: detalle diario con ``fecha``, ``usos_predicho``,
                ``demanda_real``, ``dia_tipo_num`` y ``es_proyectado``.
            confiabilidad: confiabilidad global del modelo.
            segunda_opinion: detalle diario de la segunda opinión (Jev), si se pidió.
                Se añade como columnas adicionales, al lado de la proyección.

        Returns:
            DataFrame con las columnas del archivo legado.
        """
        tabla = detalle.sort_values("fecha").copy()
        fecha = pd.to_datetime(tabla["fecha"])
        tipo_num = (
            tabla["dia_tipo_num"].astype(int)
            if "dia_tipo_num" in tabla.columns
            else fecha.dt.weekday.map(lambda d: 2 if d == 6 else (1 if d == 5 else 0))
        )

        salida = pd.DataFrame(
            {
                "Fecha": tabla["fecha"].dt.strftime("%Y-%m-%d"),
                "Año": fecha.dt.year,
                "Mes.num": fecha.dt.month,
                "Mes.nombre": fecha.dt.month.map(lambda m: _MESES[m - 1]),
                "Semana.iso": fecha.dt.isocalendar().week.astype(int),
                "Dia.num": fecha.dt.day,
                "Dia.orden": fecha.dt.weekday + 1,
                "Dia.nombre": fecha.dt.weekday.map(lambda d: _DIAS[d]),
                "Dia.tipo": tipo_num.map(_TIPOS_DIA).fillna("Hábil"),
            }
        )

        salida["es_inicio_mes"] = (salida["Dia.num"] <= 5).astype(int)
        salida["es_mitad_mes"] = salida["Dia.num"].between(13, 17).astype(int)
        salida["es_fin_mes"] = (salida["Dia.num"] >= 25).astype(int)
        salida["es_quincena_operativa"] = (
            salida["Dia.num"].isin([14, 15, 16, 29, 30, 31]).astype(int)
        )

        es_festivo = salida["Dia.tipo"].eq("Festivo").astype(int)
        es_habil = salida["Dia.tipo"].eq("Hábil").astype(int)
        salida["festivo_anterior"] = es_festivo.shift(1, fill_value=0).astype(int)
        salida["festivo_siguiente"] = es_festivo.shift(-1, fill_value=0).astype(int)
        salida["habil_despues_festivo"] = (es_habil & salida["festivo_anterior"]).astype(int)
        salida["habil_antes_festivo"] = (es_habil & salida["festivo_siguiente"]).astype(int)
        salida["es_puente"] = (
            es_habil
            & (
                (salida["festivo_anterior"].eq(1) & salida["Dia.orden"].isin([1, 5]))
                | (salida["festivo_siguiente"].eq(1) & salida["Dia.orden"].isin([1, 5]))
            )
        ).astype(int)
        salida["fin_mes_habil"] = (salida["es_fin_mes"].eq(1) & es_habil.eq(1)).astype(int)
        salida["fin_mes_festivo"] = (salida["es_fin_mes"].eq(1) & es_festivo.eq(1)).astype(int)
        salida["quincena_habil"] = (salida["es_quincena_operativa"].eq(1) & es_habil.eq(1)).astype(
            int
        )

        proyectado = tabla.get("es_proyectado", pd.Series(False, index=tabla.index)).fillna(True)
        real = pd.to_numeric(
            tabla["demanda_real"]
            if "demanda_real" in tabla.columns
            else pd.Series(np.nan, index=tabla.index),
            errors="coerce",
        )
        predicha = pd.to_numeric(tabla["usos_predicho"], errors="coerce")

        salida["Demanda_Real"] = real.round(0).astype("Int64")
        salida["Demanda_Proyectada"] = predicha.round(0).astype("Int64")
        salida["Demanda_Final"] = np.where(proyectado, predicha, real)
        salida["Tipo_Registro"] = np.where(proyectado, "Proyectado", "Real")

        conf = np.where(proyectado, round(confiabilidad * 100, 2), 100.0)
        salida["Confiabilidad_%"] = np.round(conf, 2)
        salida["Nivel_Confiabilidad"] = [
            clasificar_confiabilidad(valor / 100.0) for valor in salida["Confiabilidad_%"]
        ]

        # Días de pago (15 y último día del mes) y sus vecinos.
        ultimo = fecha.dt.is_month_end
        salida["es_dia_15"] = (salida["Dia.num"] == 15).astype(int)
        salida["es_ultimo_dia_mes"] = ultimo.astype(int).to_numpy()
        salida["es_dia_pago"] = ((salida["Dia.num"] == 15) | ultimo).astype(int).to_numpy()

        # Rango esperado (P10-P90 estimado del RMSE del modelo).
        if {"usos_p10", "usos_p90"} <= set(tabla.columns):
            inferior = pd.to_numeric(tabla["usos_p10"], errors="coerce")
            superior = pd.to_numeric(tabla["usos_p90"], errors="coerce")
            salida["Demanda_P10"] = inferior.round(0).astype("Int64")
            salida["Demanda_P90"] = superior.round(0).astype("Int64")
            salida["Error_Esperado_+/-_Usos"] = ((superior - inferior) / 2).round(0).astype("Int64")

        salida = self._anexar_segunda_opinion(salida, tabla, segunda_opinion, predicha)
        return salida

    @staticmethod
    def _anexar_segunda_opinion(
        salida: pd.DataFrame,
        tabla: pd.DataFrame,
        segunda_opinion: pd.DataFrame | None,
        predicha: pd.Series,
    ) -> pd.DataFrame:
        """Añade la segunda opinión (Jev) al lado de la proyección del modelo.

        Args:
            salida: tabla diaria ya normalizada.
            tabla: detalle original (para tomar la fecha).
            segunda_opinion: detalle diario de la segunda opinión, si existe.
            predicha: proyección del modelo (para la diferencia relativa).

        Returns:
            La tabla con ``Demanda_Jev``, ``Jev_P10``, ``Jev_P90``, ``Confianza_Jev``
            y ``Diferencia_Jev_%`` cuando hay segunda opinión.
        """
        if segunda_opinion is None or segunda_opinion.empty:
            return salida

        columnas = [
            columna
            for columna in ("fecha", "usos_jev", "usos_jev_p10", "usos_jev_p90", "confianza")
            if columna in segunda_opinion.columns
        ]
        jev = segunda_opinion[columnas].copy()
        jev["fecha"] = pd.to_datetime(jev["fecha"])
        base = tabla.reset_index(drop=True)
        unido = base[["fecha"]].merge(jev, on="fecha", how="left")

        salida = salida.reset_index(drop=True)
        usos_jev = pd.to_numeric(unido.get("usos_jev"), errors="coerce")
        salida["Demanda_Jev"] = usos_jev.round(0).astype("Int64")
        if "usos_jev_p10" in unido.columns:
            salida["Jev_P10"] = (
                pd.to_numeric(unido["usos_jev_p10"], errors="coerce").round(0).astype("Int64")
            )
            salida["Jev_P90"] = (
                pd.to_numeric(unido["usos_jev_p90"], errors="coerce").round(0).astype("Int64")
            )
        if "confianza" in unido.columns:
            salida["Confianza_Jev"] = pd.to_numeric(unido["confianza"], errors="coerce").round(2)
        referencia = predicha.reset_index(drop=True)
        salida["Diferencia_Jev_%"] = (
            (referencia - usos_jev) / usos_jev.replace(0, np.nan) * 100
        ).round(1)
        return salida

    @staticmethod
    def _tabla_evaluacion(diario: pd.DataFrame) -> pd.DataFrame:
        """Compara demanda real contra la predicha en los días observados.

        Args:
            diario: tabla diaria normalizada.

        Returns:
            DataFrame de evaluación (vacío si no hay demanda real en el rango).
        """
        observados = diario[diario["Demanda_Real"].notna()].copy()
        if observados.empty:
            return pd.DataFrame()

        real = pd.to_numeric(observados["Demanda_Real"], errors="coerce").astype(float)
        predicha = pd.to_numeric(observados["Demanda_Proyectada"], errors="coerce").astype(float)
        error = real - predicha
        error_abs = error.abs()

        columnas = [
            "Fecha",
            "Año",
            "Mes.num",
            "Mes.nombre",
            "Semana.iso",
            "Dia.num",
            "Dia.orden",
            "Dia.nombre",
            "Dia.tipo",
        ]
        evaluacion = observados[columnas].copy()
        evaluacion["Demanda_Real"] = real.round(0).astype("Int64")
        evaluacion["Demanda_Predicha"] = predicha.round(0).astype("Int64")
        evaluacion["error"] = error.round(0).astype("Int64")
        evaluacion["error_abs"] = error_abs.round(0).astype("Int64")
        evaluacion["APE_%"] = np.where(real > 0, (error_abs / real * 100).round(2), np.nan)
        # Error porcentual simétrico (robusto con demandas pequeñas); la confiabilidad
        # se deriva de él para no castigar días de demanda casi nula.
        denominador = real.abs() + predicha.abs()
        evaluacion["SAPE_%"] = np.where(
            denominador > 0, (200 * error_abs / denominador).round(2), np.nan
        )
        evaluacion["Confiabilidad_%"] = (
            (100 - evaluacion["SAPE_%"] / 2)
            .clip(CONFIABILIDAD_MINIMA * 100, CONFIABILIDAD_MAXIMA * 100)
            .round(2)
        )
        evaluacion["Nivel_Confiabilidad"] = evaluacion["Confiabilidad_%"].map(
            lambda valor: clasificar_confiabilidad(float(valor) / 100.0)
        )
        return evaluacion.reset_index(drop=True)
        evaluacion["Nivel_Confiabilidad"] = evaluacion["Confiabilidad_%"].map(
            lambda valor: clasificar_confiabilidad(float(valor) / 100.0)
        )
        return evaluacion.reset_index(drop=True)

    @staticmethod
    def _tabla_resumen_mensual(diario: pd.DataFrame) -> pd.DataFrame:
        """Agrega la proyección por mes.

        Args:
            diario: tabla diaria normalizada.

        Returns:
            DataFrame con los totales y promedios mensuales.
        """
        resumen = diario.groupby(["Año", "Mes.num", "Mes.nombre"], as_index=False).agg(
            Demanda_Total=("Demanda_Final", "sum"),
            Promedio_Diario=("Demanda_Final", "mean"),
            Dias=("Fecha", "count"),
            Dias_Reales=("Tipo_Registro", lambda s: int((s == "Real").sum())),
            Dias_Proyectados=("Tipo_Registro", lambda s: int((s == "Proyectado").sum())),
            Confiabilidad_Promedio=("Confiabilidad_%", "mean"),
        )
        resumen["Demanda_Total"] = resumen["Demanda_Total"].round(0).astype("Int64")
        resumen["Promedio_Diario"] = resumen["Promedio_Diario"].round(0).astype("Int64")
        resumen["Confiabilidad_Promedio"] = resumen["Confiabilidad_Promedio"].round(2)
        return resumen

    @staticmethod
    def _tabla_confiabilidad(diario: pd.DataFrame) -> pd.DataFrame:
        """Desempeño por tipo de día sobre los días con demanda observada.

        Args:
            diario: tabla diaria normalizada.

        Returns:
            DataFrame con el desempeño por tipo de día (vacío si no hay reales).
        """
        observados = diario[diario["Demanda_Real"].notna()].copy()
        if observados.empty:
            return pd.DataFrame()

        real = pd.to_numeric(observados["Demanda_Real"], errors="coerce").astype(float)
        predicha = pd.to_numeric(observados["Demanda_Proyectada"], errors="coerce").astype(float)
        observados["_error_abs"] = (real - predicha).abs()
        observados["_ape"] = np.where(real > 0, observados["_error_abs"] / real * 100, np.nan)

        resumen = observados.groupby("Dia.tipo", as_index=False).agg(
            Dias=("Fecha", "count"),
            Demanda_Real_Prom=("Demanda_Real", "mean"),
            Prediccion_Prom=("Demanda_Proyectada", "mean"),
            Error_Abs_Prom=("_error_abs", "mean"),
            MAPE_Prom=("_ape", "mean"),
        )
        resumen["Demanda_Real_Prom"] = resumen["Demanda_Real_Prom"].round(0).astype("Int64")
        resumen["Prediccion_Prom"] = resumen["Prediccion_Prom"].round(0).astype("Int64")
        resumen["Error_Abs_Prom"] = resumen["Error_Abs_Prom"].round(0).astype("Int64")
        resumen["MAPE_Prom"] = resumen["MAPE_Prom"].round(2)
        resumen["Confiabilidad_Prom"] = (
            (100 - resumen["MAPE_Prom"])
            .clip(CONFIABILIDAD_MINIMA * 100, CONFIABILIDAD_MAXIMA * 100)
            .round(2)
        )
        resumen["Nivel_Confiabilidad"] = resumen["Confiabilidad_Prom"].map(
            lambda valor: clasificar_confiabilidad(float(valor) / 100.0)
        )
        return resumen

    @staticmethod
    def _tabla_confiabilidad_estimada(diario: pd.DataFrame, confiabilidad: float) -> pd.DataFrame:
        """Confiabilidad estimada por tipo de día cuando no hay demanda real.

        Args:
            diario: tabla diaria normalizada.
            confiabilidad: confiabilidad global del modelo (1 - WAPE).

        Returns:
            DataFrame con los días proyectados y la confiabilidad estimada.
        """
        resumen = (
            diario.groupby("Dia.tipo", as_index=False)
            .agg(
                Dias=("Fecha", "count"),
                Prediccion_Prom=("Demanda_Proyectada", "mean"),
            )
            .rename(columns={"Prediccion_Prom": "Prediccion_Prom"})
        )
        resumen["Prediccion_Prom"] = resumen["Prediccion_Prom"].round(0).astype("Int64")
        resumen["Confiabilidad_Prom"] = round(confiabilidad * 100, 2)
        resumen["Nivel_Confiabilidad"] = resumen["Confiabilidad_Prom"].map(
            lambda valor: clasificar_confiabilidad(float(valor) / 100.0)
        )
        return resumen

    # ------------------------------------------------------------------
    # Libro de evaluación (backtest)
    # ------------------------------------------------------------------
    def libro_evaluacion(self, evaluacion) -> dict[str, pd.DataFrame]:
        """Construye las hojas del Excel para una evaluación contra datos reales.

        Args:
            evaluacion: :class:`~mio_demanda.services.proyeccion_service.EvaluacionProyeccion`.

        Returns:
            Diccionario ``nombre de hoja -> DataFrame``.
        """
        comparacion = evaluacion.comparacion
        hojas: dict[str, pd.DataFrame] = {"Evaluacion_Modelo": comparacion}

        resumen_tipo = self._resumen_por_tipo_evaluado(comparacion)
        if not resumen_tipo.empty:
            hojas["Resumen_Confiabilidad"] = resumen_tipo

        resumen_mensual = self._resumen_mensual_evaluacion(comparacion)
        if not resumen_mensual.empty:
            hojas["Resumen_Mensual"] = resumen_mensual

        hojas["Metricas"] = self._tabla_metricas(
            evaluacion.modelo, evaluacion.metricas, evaluacion.confiabilidad
        ).copy()
        extras = {
            "Dias evaluados": evaluacion.dias_evaluados,
            "Dias fuera de muestra": evaluacion.dias_fuera_de_muestra,
            "Periodo evaluado": f"{evaluacion.desde} a {evaluacion.hasta}",
            "Corte de entrenamiento": str(evaluacion.corte_entrenamiento),
            "Copia temporal (no guardada)": "Si" if evaluacion.reentrenado else "No",
        }
        hojas["Metricas"] = pd.concat(
            [
                hojas["Metricas"],
                pd.DataFrame(
                    [{"Metrica": clave, "Valor": valor} for clave, valor in extras.items()]
                ),
            ],
            ignore_index=True,
        )
        return hojas

    @staticmethod
    def _resumen_por_tipo_evaluado(comparacion: pd.DataFrame) -> pd.DataFrame:
        """Resume el desempeño por tipo de día en la evaluación.

        Args:
            comparacion: tabla real vs. predicho.

        Returns:
            DataFrame por tipo de día (vacío si no hay columna de tipo).
        """
        if "Dia.tipo" not in comparacion.columns:
            return pd.DataFrame()
        agrupado = comparacion.groupby("Dia.tipo", as_index=False).agg(
            Dias=("Fecha", "count"),
            Demanda_Real_Prom=("Demanda_Real", "mean"),
            Prediccion_Prom=("Demanda_Predicha", "mean"),
            Error_Abs_Prom=("error_abs", "mean"),
            MAPE_Prom=("APE_%", "mean"),
        )
        if "SAPE_%" in comparacion.columns:
            agrupado["sMAPE_Prom"] = comparacion.groupby("Dia.tipo")["SAPE_%"].mean().to_numpy()
        agrupado["Demanda_Real_Prom"] = agrupado["Demanda_Real_Prom"].round(0).astype("Int64")
        agrupado["Prediccion_Prom"] = agrupado["Prediccion_Prom"].round(0).astype("Int64")
        agrupado["Error_Abs_Prom"] = agrupado["Error_Abs_Prom"].round(0).astype("Int64")
        agrupado["MAPE_Prom"] = agrupado["MAPE_Prom"].round(2)
        referencia = (
            agrupado["sMAPE_Prom"] if "sMAPE_Prom" in agrupado.columns else agrupado["MAPE_Prom"]
        )
        agrupado["Confiabilidad_Prom"] = (
            (100 - referencia / 2)
            .clip(CONFIABILIDAD_MINIMA * 100, CONFIABILIDAD_MAXIMA * 100)
            .round(2)
        )
        agrupado["Nivel_Confiabilidad"] = agrupado["Confiabilidad_Prom"].map(
            lambda valor: clasificar_confiabilidad(float(valor) / 100.0)
        )
        return agrupado

    @staticmethod
    def _resumen_mensual_evaluacion(comparacion: pd.DataFrame) -> pd.DataFrame:
        """Resume la evaluación por mes.

        Args:
            comparacion: tabla real vs. predicho.

        Returns:
            DataFrame con totales y error promedio por mes.
        """
        if comparacion.empty:
            return pd.DataFrame()
        fechas = pd.to_datetime(comparacion["Fecha"])
        datos = comparacion.assign(
            **{
                "Año": fechas.dt.year,
                "Mes.num": fechas.dt.month,
                "Mes.nombre": fechas.dt.month.map(lambda m: _MESES[m - 1]),
            }
        )
        resumen = datos.groupby(["Año", "Mes.num", "Mes.nombre"], as_index=False).agg(
            Demanda_Real_Total=("Demanda_Real", "sum"),
            Demanda_Predicha_Total=("Demanda_Predicha", "sum"),
            Dias=("Fecha", "count"),
            MAPE_Prom=("APE_%", "mean"),
        )
        resumen["Demanda_Real_Total"] = resumen["Demanda_Real_Total"].round(0).astype("Int64")
        resumen["Demanda_Predicha_Total"] = (
            resumen["Demanda_Predicha_Total"].round(0).astype("Int64")
        )
        resumen["MAPE_Prom"] = resumen["MAPE_Prom"].round(2)
        return resumen

    @staticmethod
    def _tabla_metricas(
        modelo: str,
        metricas: dict[str, float],
        confiabilidad: float,
        entrenamiento: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Resume las métricas de validación del modelo consumido.

        Incluye cuándo y con qué historia se entrenó, para que el Excel exportado se
        explique solo (con qué modelo se proyectó y con qué error).

        Args:
            modelo: nombre del paquete de modelo.
            metricas: métricas de validación cruzada.
            confiabilidad: confiabilidad global (1 - WAPE).
            entrenamiento: fechas del entrenamiento (``fecha``, ``desde``, ``hasta``).

        Returns:
            DataFrame con ``Metrica`` y ``Valor``.
        """
        filas = [
            {"Metrica": "Modelo", "Valor": modelo},
            {"Metrica": "WAPE % (referencia)", "Valor": round(float(metricas.get("wape", 0)), 2)},
            {"Metrica": "MAE (usos)", "Valor": round(float(metricas.get("mae", 0)), 1)},
            {"Metrica": "RMSE (usos)", "Valor": round(float(metricas.get("rmse", 0)), 1)},
            {"Metrica": "sMAPE %", "Valor": round(float(metricas.get("smape", 0)), 2)},
            {
                "Metrica": "MAPE % (solo referencia)",
                "Valor": round(float(metricas.get("mape", 0)), 2),
            },
            {"Metrica": "R2", "Valor": round(float(metricas.get("r2", 0)), 4)},
            {"Metrica": "Confiabilidad global %", "Valor": round(confiabilidad * 100, 2)},
            {"Metrica": "Nivel de confiabilidad", "Valor": clasificar_confiabilidad(confiabilidad)},
        ]
        entrenamiento = entrenamiento or {}
        desde = entrenamiento.get("desde")
        hasta = entrenamiento.get("hasta")
        if entrenamiento.get("fecha"):
            filas.append({"Metrica": "Entrenado", "Valor": str(entrenamiento["fecha"])[:10]})
        if desde and hasta:
            filas.append({"Metrica": "Periodo de entrenamiento", "Valor": f"{desde} a {hasta}"})
        return pd.DataFrame(filas)
