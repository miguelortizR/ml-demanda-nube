"""Construcción de los datasets de modelado (sistema y por entidad).

Reglas de diseño:

* **Sin fuga de datos**: todas las anclas (nivel reciente, media por entidad,
  factor estacional) se calculan con **desplazamiento** (``shift``), de modo que
  cada fila usa únicamente observaciones **anteriores** a su propia fecha.
* **Pronóstico directo**: la demanda futura no existe, por lo que las anclas se
  arrastran hacia adelante (``ffill``) con su último valor conocido. Así el mismo
  dataset sirve para entrenar y para proyectar.
* **Variables disponibles en producción**: kilómetros y otras variables de oferta
  **no** son variables del modelo (no se conocen a futuro); se proyectan aparte.
* **Todo numérico**: sin variables categóricas de alta cardinalidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.features.tiempo import factor_mes_tipo, mediana_movil_pasada
from mio_demanda.infrastructure.logging_utils import obtener_logger
from mio_demanda.infrastructure.repositorio_calendario import RepositorioCalendario
from mio_demanda.infrastructure.repositorio_datos import RepositorioDatos

logger = obtener_logger(__name__)

# Variables de calendario comunes a todos los modelos.
COLUMNAS_CALENDARIO = [
    "dia_tipo_num",
    "dia_semana",
    "es_habil",
    "es_sabado",
    "es_festivo",
    "es_fin_semana",
    "mes_num",
    "dia_del_mes",
    "semana_iso",
    "es_inicio_mes",
    "es_fin_mes",
    "es_quincena",
    # Días de pago: el 15 y el último día de cada mes mueven la demanda.
    "es_dia_15",
    "es_ultimo_dia_mes",
    "es_dia_pago",
    "es_pre_pago",
    "es_post_pago",
]

# Variables de eventos del profesional.
COLUMNAS_EVENTOS_BASE = ["n_eventos", "impacto_evento", "peso_eventos"]

# Las ventanas de las anclas (calentamiento, ancla reciente, mínimo de observaciones y
# cobertura mínima de una entidad) **no** se declaran aquí: son parametros
# obligatorios que el llamador toma de ``mio_demanda.config.Settings`` (``MIO_*``).
# Duplicarlas como constantes dejaría dos fuentes de verdad para el mismo valor.


@dataclass(slots=True)
class Dataset:
    """Matriz de diseño lista para entrenar o proyectar.

    Attributes:
        X: variables de entrada (numéricas).
        y: objetivo (usos); contiene ``NaN`` en las filas de proyección.
        fechas: fecha de cada fila (alineada con ``X``).
        columnas_features: orden exacto de las columnas de ``X``.
        auxiliares: columnas de apoyo (entidad, tipo de día) alineadas por índice.
        tabla: fila completa (X + y + calendario) para trazabilidad.
        en_entrenamiento: máscara booleana de filas con objetivo observado.
    """

    X: pd.DataFrame
    y: pd.Series
    fechas: pd.Series
    columnas_features: list[str]
    auxiliares: pd.DataFrame = field(default_factory=pd.DataFrame)
    tabla: pd.DataFrame = field(default_factory=pd.DataFrame)
    en_entrenamiento: pd.Series | None = None

    def __len__(self) -> int:
        return len(self.X)

    def solo_entrenamiento(self) -> Dataset:
        """Devuelve una copia restringida a las filas con objetivo observado."""
        if self.en_entrenamiento is None:
            return self
        mascara = self.en_entrenamiento.to_numpy()
        return Dataset(
            X=self.X.loc[mascara].reset_index(drop=True),
            y=self.y.loc[mascara].reset_index(drop=True),
            fechas=self.fechas.loc[mascara].reset_index(drop=True),
            columnas_features=self.columnas_features,
            auxiliares=self.auxiliares.loc[mascara].reset_index(drop=True),
            tabla=self.tabla.loc[mascara].reset_index(drop=True),
            en_entrenamiento=self.en_entrenamiento.loc[mascara].reset_index(drop=True),
        )


@dataclass(slots=True)
class PerfilHorario:
    """Curva de participación horaria de la demanda.

    Attributes:
        por_entidad: participación de cada hora por entidad y tipo de día.
        global_: participación por hora y tipo de día (respaldo).
    """

    por_entidad: pd.DataFrame
    global_: pd.DataFrame


def _columnas_eventos(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas de eventos personalizados presentes en el DataFrame."""
    return [c for c in df.columns if c.startswith("evento_")]


def _entidades_con_cobertura(
    usos: pd.DataFrame, min_cobertura: float, columna_entidad: str = "entidad"
) -> list[str]:
    """Selecciona las entidades con suficiente presencia de datos.

    Las entidades que solo aparecen unos pocos días (rutas retiradas, nombres
    temporales, errores de digitación) ensucian el modelo y distorsionan las
    métricas. Se conservan aquellas cuyo número de días con datos alcanza la
    cobertura mínima exigida.

    Args:
        usos: usos diarios por entidad (``fecha``, ``entidad``, ``usos``).
        min_cobertura: fracción mínima de días con datos (0 a 1).
        columna_entidad: nombre de la columna identificadora de la entidad.

    Returns:
        Lista de entidades que superan la cobertura mínima.
    """
    dias_ventana = usos["fecha"].nunique()
    if dias_ventana == 0:
        return []
    presencia = usos.groupby(columna_entidad)["usos"].apply(lambda s: (s > 0).sum())
    umbral = max(1, int(min_cobertura * dias_ventana))
    return sorted(presencia[presencia >= umbral].index.astype(str).tolist())


def _inicio_de_carga(repo: RepositorioDatos, desde: date, dias_calentamiento: int) -> date:
    """Fecha desde la que se carga historia para estimar las anclas.

    Lo habitual es ``desde - dias_calentamiento``. Pero si el período pedido cae muy
    adelante en el futuro (más de ``dias_calentamiento`` días después del último dato
    real), se retrocede hasta incluir ese último dato: las anclas se calculan con la
    historia disponible y se **arrastran** hacia adelante.

    Sin este ajuste, un horizonte lejano dejaba las anclas en **cero** (el modelo
    proyectaba sin nivel de demanda) y, en el alcance por entidad, no había ninguna
    fila y la proyección fallaba.

    Args:
        repo: repositorio de datos.
        desde: primera fecha a modelar/proyectar.
        dias_calentamiento: días de historia previa deseados.

    Returns:
        La fecha de inicio de carga, garantizando historia si existen datos.
    """
    candidato = desde - timedelta(days=dias_calentamiento)
    rango = repo.rango_fechas_usos()
    if rango is None:
        return candidato
    _, ultimo = rango
    if candidato <= ultimo:
        return candidato
    return ultimo - timedelta(days=dias_calentamiento)


def _factor_estacional(observado: pd.DataFrame) -> pd.DataFrame:
    """Estima el factor estacional mes/tipo de día con datos observados.

    Args:
        observado: filas con ``usos`` observado.

    Returns:
        DataFrame con ``mes_num``, ``dia_tipo_num`` y ``factor_mes_tipo``.
    """
    if observado.empty or observado["usos"].sum() == 0:
        return pd.DataFrame(columns=["mes_num", "dia_tipo_num", "factor_mes_tipo"])
    return factor_mes_tipo(observado, "usos")


# ======================================================================
# DATASET DEL SISTEMA (demanda diaria total)
# ======================================================================
def construir_dataset_sistema(
    repo: RepositorioDatos,
    repo_cal: RepositorioCalendario,
    desde: date,
    hasta: date,
    *,
    dias_calentamiento: int,
    ventana_reciente: int,
    min_obs_ancla: int,
) -> Dataset:
    """Construye el dataset diario de demanda del sistema completo.

    Args:
        repo: repositorio de datos.
        repo_cal: repositorio del calendario.
        desde: fecha inicial del período a modelar/proyectar.
        hasta: fecha final del período.
        dias_calentamiento: días de historia previa para estimar las anclas.
        ventana_reciente: ventana del ancla reciente (nivel de la demanda).
        min_obs_ancla: observaciones mínimas para emitir un ancla.

    Returns:
        Un :class:`Dataset` con la demanda diaria del sistema. Las filas sin
        demanda observada (futuro) quedan marcadas en ``en_entrenamiento``.
    """
    desde_carga = _inicio_de_carga(repo, desde, dias_calentamiento)
    calendario = repo_cal.calendario_enriquecido(desde_carga, hasta)
    usos = repo.usos_diarios(desde_carga, hasta)
    df = calendario.merge(usos[["fecha", "usos"]], on="fecha", how="left").sort_values("fecha")

    # Ancla pasada por fila; se arrastra hacia el futuro con el último valor conocido.
    df["nivel_reciente"] = mediana_movil_pasada(
        df, "usos", ventana=ventana_reciente, min_periodos=min_obs_ancla
    ).ffill()

    factor = _factor_estacional(df[df["usos"].notna()])
    df = df.merge(factor, on=["mes_num", "dia_tipo_num"], how="left").fillna(
        {"factor_mes_tipo": 1.0}
    )
    df["patron_estacional"] = df["nivel_reciente"].fillna(0.0) * df["factor_mes_tipo"]
    df["nivel_reciente"] = df["nivel_reciente"].fillna(df["usos"]).fillna(0.0)

    modelado = df[df["fecha"] >= pd.Timestamp(desde)].copy()
    columnas = (
        COLUMNAS_CALENDARIO
        + COLUMNAS_EVENTOS_BASE
        + _columnas_eventos(df)
        + ["nivel_reciente", "factor_mes_tipo", "patron_estacional"]
    )
    logger.info(
        "Dataset sistema: %s filas (%s con objetivo)",
        len(modelado),
        int(modelado["usos"].notna().sum()),
    )

    return Dataset(
        X=modelado[columnas].astype(float).reset_index(drop=True),
        y=modelado["usos"].astype(float).reset_index(drop=True),
        fechas=modelado["fecha"].reset_index(drop=True),
        columnas_features=columnas,
        auxiliares=modelado[["fecha", "dia_tipo_num", "dia_tipo_nom"]].reset_index(drop=True),
        tabla=modelado.reset_index(drop=True),
        en_entrenamiento=modelado["usos"].notna().reset_index(drop=True),
    )


# ======================================================================
# DATASET POR ENTIDAD (estaciones / rutas, demanda diaria por entidad)
# ======================================================================
def construir_dataset_entidad(
    repo: RepositorioDatos,
    repo_cal: RepositorioCalendario,
    tipo: TipoEntidad,
    desde: date,
    hasta: date,
    entidades: list[str] | None = None,
    *,
    min_cobertura: float,
    dias_calentamiento: int,
    ventana_reciente: int,
    min_obs_ancla: int,
) -> Dataset:
    """Construye el dataset diario de demanda para estaciones o rutas.

    Args:
        repo: repositorio de datos.
        repo_cal: repositorio del calendario.
        tipo: :class:`TipoEntidad` (``ESTACION`` o ``RUTA``).
        desde: fecha inicial del período a modelar/proyectar.
        hasta: fecha final del período.
        entidades: subconjunto de entidades; ``None`` para todas las del tipo.
        min_cobertura: fracción mínima de días con datos para conservar una entidad.
        dias_calentamiento: días de historia previa para estimar las anclas.
        ventana_reciente: ventana del ancla reciente (nivel de la demanda).
        min_obs_ancla: observaciones mínimas para emitir un ancla.

    Returns:
        Un :class:`Dataset` con una fila por (entidad, fecha).

    Raises:
        ValueError: si no hay entidades o datos para el tipo solicitado.
    """
    entidades = entidades or repo.listar_entidades(tipo)
    if not entidades:
        raise ValueError(f"No hay entidades de tipo {tipo.value} en los datos normalizados")

    desde_carga = _inicio_de_carga(repo, desde, dias_calentamiento)
    usos = repo.usos_diarios_por_entidad(desde_carga, hasta, entidades)
    if usos.empty:
        nombres = ", ".join(sorted(str(entidad) for entidad in entidades)[:10])
        raise ValueError(
            f"Sin usos de {tipo.value} entre {desde_carga} y {hasta} para: {nombres}. "
            "Suele ser una entidad retirada o renombrada, sin datos en ese período."
        )

    entidades = _entidades_con_cobertura(usos, min_cobertura)
    if not entidades:
        raise ValueError(
            f"Ninguna entidad de tipo {tipo.value} alcanza la cobertura mínima "
            f"({min_cobertura:.0%}) de días con datos"
        )

    calendario = repo_cal.calendario_enriquecido(desde_carga, hasta)
    # Producto cartesiano fecha/entidad para que existan las filas futuras.
    grilla = (
        calendario.assign(_union=1)
        .merge(pd.DataFrame({"entidad": entidades, "_union": 1}), on="_union")
        .drop(columns="_union")
    )
    df = grilla.merge(usos, on=["fecha", "entidad"], how="left").sort_values(["entidad", "fecha"])

    # Anclas pasadas por entidad (se arrastran hacia el futuro).
    df["ancla_reciente"] = mediana_movil_pasada(
        df,
        "usos",
        ventana=ventana_reciente,
        min_periodos=min_obs_ancla,
        columnas_grupo=["entidad"],
    )
    df["ancla_entidad"] = df.groupby("entidad", dropna=False)["usos"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).median()
    )
    df["ancla_tipo_dia"] = df.groupby(["entidad", "dia_tipo_num"], dropna=False)["usos"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).median()
    )
    # El relleno se hace **dentro de cada entidad**: el marco apila todas las
    # entidades, así que un ``ffill()`` sobre la columna completa arrastraría el
    # ancla de la entidad anterior a los días sin historia de la siguiente.
    df["ancla_entidad"] = df.groupby("entidad", dropna=False)["ancla_entidad"].ffill().fillna(0.0)
    df["ancla_reciente"] = (
        df.groupby("entidad", dropna=False)["ancla_reciente"].ffill().fillna(df["ancla_entidad"])
    )
    df["ancla_tipo_dia"] = (
        df.groupby(["entidad", "dia_tipo_num"], dropna=False)["ancla_tipo_dia"]
        .ffill()
        .fillna(df["ancla_entidad"])
    )

    factor = _factor_estacional(df[df["usos"].notna()])
    df = df.merge(factor, on=["mes_num", "dia_tipo_num"], how="left").fillna(
        {"factor_mes_tipo": 1.0}
    )
    df["patron_entidad"] = df["ancla_reciente"].fillna(0.0) * df["factor_mes_tipo"]

    modelado = df[df["fecha"] >= pd.Timestamp(desde)].copy()
    columnas = (
        COLUMNAS_CALENDARIO
        + COLUMNAS_EVENTOS_BASE
        + _columnas_eventos(df)
        + ["ancla_reciente", "ancla_entidad", "ancla_tipo_dia", "factor_mes_tipo", "patron_entidad"]
    )
    logger.info(
        "Dataset %s: %s filas (%s con objetivo), %s variables",
        tipo.value,
        len(modelado),
        int(modelado["usos"].notna().sum()),
        len(columnas),
    )

    return Dataset(
        X=modelado[columnas].astype(float).reset_index(drop=True),
        y=modelado["usos"].astype(float).reset_index(drop=True),
        fechas=modelado["fecha"].reset_index(drop=True),
        columnas_features=columnas,
        auxiliares=modelado[["fecha", "entidad", "dia_tipo_num", "dia_tipo_nom"]].reset_index(
            drop=True
        ),
        tabla=modelado.reset_index(drop=True),
        en_entrenamiento=modelado["usos"].notna().reset_index(drop=True),
    )


# ======================================================================
# PERFIL HORARIO (para desagregar la proyección diaria en horas)
# ======================================================================
def construir_perfil_horario(
    repo: RepositorioDatos,
    repo_cal: RepositorioCalendario,
    entidades: list[str] | None,
    desde: date,
    hasta: date,
) -> PerfilHorario:
    """Calcula la participación de cada hora en la demanda diaria.

    Args:
        repo: repositorio de datos.
        repo_cal: repositorio del calendario.
        entidades: entidades cuyo perfil se desea; ``None`` para todas (sistema).
        desde: fecha inicial de la historia usada para el perfil.
        hasta: fecha final de la historia usada para el perfil.

    Returns:
        Un :class:`PerfilHorario` con la participación por entidad y el respaldo global.
    """
    horario = repo.usos_horarios_perfil(desde, hasta, entidades)
    vacio = pd.DataFrame(columns=["entidad", "hora", "dia_tipo_num", "participacion"])
    if horario.empty:
        return PerfilHorario(
            por_entidad=vacio,
            global_=pd.DataFrame(columns=["hora", "dia_tipo_num", "participacion"]),
        )

    calendario = repo_cal.calendario_enriquecido(desde, hasta)[["fecha", "dia_tipo_num"]]
    horario = horario.merge(calendario, on="fecha", how="left")

    totales_dia = horario.groupby(["entidad", "fecha"])["usos"].transform("sum")
    horario["participacion"] = (horario["usos"] / totales_dia.replace(0, pd.NA)).fillna(0.0)
    por_entidad = (
        horario.groupby(["entidad", "hora", "dia_tipo_num"])["participacion"].mean().reset_index()
    )
    # Se normaliza para que cada (entidad, tipo de día) sume exactamente 1: así la suma
    # de las horas reproduce el total diario sin desfase por redondeo.
    suma_entidad = por_entidad.groupby(["entidad", "dia_tipo_num"])["participacion"].transform(
        "sum"
    )
    por_entidad["participacion"] = (
        por_entidad["participacion"] / suma_entidad.replace(0, pd.NA)
    ).fillna(0.0)

    # Perfil global: primero se suman **todas las entidades** por hora y luego se divide
    # por el total del día. Sin ese paso la participación quedaba dividida entre el
    # número de entidades y no sumaba 1 (el reparto horario salía plano y diminuto).
    por_hora = horario.groupby(["fecha", "hora", "dia_tipo_num"], as_index=False)["usos"].sum()
    totales_global = por_hora.groupby("fecha")["usos"].transform("sum")
    por_hora["participacion"] = (por_hora["usos"] / totales_global.replace(0, pd.NA)).fillna(0.0)
    global_ = por_hora.groupby(["hora", "dia_tipo_num"])["participacion"].mean().reset_index()
    suma_tipo = global_.groupby("dia_tipo_num")["participacion"].transform("sum")
    global_["participacion"] = (global_["participacion"] / suma_tipo.replace(0, pd.NA)).fillna(0.0)

    return PerfilHorario(por_entidad=por_entidad, global_=global_)
