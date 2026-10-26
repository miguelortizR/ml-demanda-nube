"""Bloque de segunda opinión (Jev) para las pestañas de proyección.

La interfaz solo muestra: el caso de uso vive en
:class:`~mio_demanda.services.segunda_opinion_service.ServicioSegundaOpinion`.

Reglas del bloque:

* Se consulta **solo cuando el equipo lo pide**: nunca hay llamadas automáticas, para
  no gastar saldo sin motivo.
* Se guarda en el estado de la sesión **con su identificador** (entidad y horizonte),
  de modo que no se mezcle con otra proyección.
* Devuelve el detalle diario para que las gráficas y las tablas puedan mostrar las dos
  series juntas: la del modelo entrenado y la de la segunda opinión.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from mio_demanda.domain.entidades import TipoEntidad
from mio_demanda.infrastructure.cliente_jev import ErrorSegundaOpinion
from mio_demanda.services.segunda_opinion_service import PrediccionJev
from mio_demanda.ui.estado import obtener_segunda_opinion, obtener_settings

_CLAVE = "segunda_opinion"


def identificador(desde: date, hasta: date, entidad: str | None) -> str:
    """Clave del bloque: distingue entidad y horizonte.

    Args:
        desde: primera fecha del horizonte.
        hasta: última fecha del horizonte.
        entidad: entidad consultada (``None`` para el sistema).

    Returns:
        Cadena única, por ejemplo ``T31::2026-10-16::2026-10-22``.
    """
    return f"{entidad or 'sistema'}::{desde}::{hasta}"


def render(
    desde: date,
    hasta: date,
    usos_modelo: float,
    entidad: str | None = None,
    tipo: TipoEntidad | None = None,
) -> PrediccionJev | None:
    """Pide (si el usuario lo decide) y muestra la segunda opinión.

    Args:
        desde: primera fecha del horizonte.
        hasta: última fecha del horizonte.
        usos_modelo: promedio diario que proyectó el modelo entrenado.
        entidad: entidad consultada (``None`` para el sistema).
        tipo: tipo de entidad (``None`` para el sistema).

    Returns:
        La :class:`PrediccionJev` si hay una disponible para este horizonte,
        ``None`` en caso contrario.
    """
    servicio = obtener_segunda_opinion()
    settings = obtener_settings()
    st.divider()
    st.markdown("#### Segunda opinión (Jev)")

    if settings.es_nube:
        st.button("Pedir predicción a Jev", key=f"btn_jev_nube_{desde}_{hasta}", disabled=True)
        st.caption(
            "Solo la app local puede consultar a Jev: la clave de la API no se sube a la "
            "nube. Esta opción está desactivada aquí a propósito."
        )
        return None

    if not servicio.disponible:
        st.caption("Sin clave configurada. Define `MIO_JEV_API_KEY_CIFRADA` en el archivo `.env`.")
        return None

    clave = identificador(desde, hasta, entidad)
    dias = (hasta - desde).days + 1
    llamadas = max(1, -(-dias // settings.jev_max_dias))
    if st.button("Pedir predicción a Jev", key=f"btn_jev_{clave}"):
        with st.spinner(
            f"Consultando la segunda opinión ({llamadas} llamada(s) para {dias} días)..."
        ):
            try:
                prediccion = (
                    servicio.predecir_sistema(desde, hasta)
                    if entidad is None
                    else servicio.predecir_entidad(entidad, tipo, desde, hasta)
                )
                st.session_state[_CLAVE] = {"clave": clave, "prediccion": prediccion}
            except (ErrorSegundaOpinion, ValueError) as error:
                st.session_state.pop(_CLAVE, None)
                st.error(f"No se pudo obtener la segunda opinión: {error}")
                return None

    guardado = st.session_state.get(_CLAVE)
    if guardado is None:
        st.info("Pulsa **Pedir predicción a Jev** para contrastar con otra fuente.")
        return None
    if guardado["clave"] != clave:
        st.info(
            "La última segunda opinión es de otro horizonte o entidad. Vuelve a pedirla para este."
        )
        return None

    prediccion: PrediccionJev = guardado["prediccion"]
    _resumen(prediccion, usos_modelo)
    return prediccion


def _resumen(prediccion: PrediccionJev, usos_modelo: float) -> None:
    """Muestra la comparación de promedios y la confianza de la segunda opinión.

    Args:
        prediccion: respuesta del modelo externo.
        usos_modelo: promedio diario del modelo entrenado.
    """
    usos_jev = prediccion.usos_diarios
    diferencia = (usos_jev - usos_modelo) / usos_modelo * 100 if usos_modelo else 0.0
    tabla = pd.DataFrame(
        [
            {
                "Fuente": "Modelo entrenado",
                "Promedio diario (usos)": round(usos_modelo),
                "Diferencia": "—",
                "Confianza": "—",
            },
            {
                "Fuente": f"Segunda opinión ({prediccion.modelo})",
                "Promedio diario (usos)": round(usos_jev),
                "Diferencia": f"{diferencia:+.1f} %",
                "Confianza": f"{prediccion.confianza:.0%}",
            },
        ]
    )
    st.dataframe(tabla, width="stretch", hide_index=True)
    minimo, maximo = prediccion.rango_rubrica
    st.caption(
        f"Cada día se pregunta en el rango de su tipo de día ({minimo:,} a {maximo:,} usos)".replace(
            ",", "."
        )
        + f" · {prediccion.tokens_entrada:,} tokens · a ciegas, no se le envía nuestra predicción."
    )
