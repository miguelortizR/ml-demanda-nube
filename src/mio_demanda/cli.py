"""Interfaz de línea de comandos de ML-Demanda.

Punto de entrada único para las tareas operativas::

    uv run python -m mio_demanda.cli extract
    uv run python -m mio_demanda.cli train --alcance todo --desde 2022-01-01 --hasta 2026-09-30
    uv run python -m mio_demanda.cli estado

La CLI es un *cliente* de los servicios, igual que la UI: no contiene lógica de
negocio.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime

from mio_demanda.config import get_settings
from mio_demanda.infrastructure.ingesta import extraer_todo
from mio_demanda.infrastructure.logging_utils import configurar_logging, obtener_logger
from mio_demanda.services.catalogo_service import ServicioCatalogo
from mio_demanda.services.entrenamiento_service import ALCANCES, ServicioEntrenamiento

logger = obtener_logger(__name__)


def _parse_fecha(texto: str) -> date:
    """Convierte una fecha ISO (``YYYY-MM-DD``) a ``date``.

    Args:
        texto: fecha en formato ISO.

    Returns:
        La fecha convertida.

    Raises:
        argparse.ArgumentTypeError: si el formato es inválido.
    """
    try:
        return datetime.strptime(texto, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Fecha inválida '{texto}' (use YYYY-MM-DD)") from error


def _comando_extract(_args: argparse.Namespace) -> int:
    """Ejecuta la extracción de datos crudos hacia ``data/raw``."""
    settings = get_settings()
    resultado = extraer_todo(settings)
    print(json.dumps(resultado.como_diccionario(), indent=2, ensure_ascii=False))
    return 0


def _comando_train(args: argparse.Namespace) -> int:
    """Entrena los modelos del alcance solicitado."""
    settings = get_settings()
    servicio = ServicioEntrenamiento(settings, notificar=lambda m: print(f"  {m}"))

    alcances = ALCANCES if args.alcance == "todo" else (args.alcance,)
    for alcance in alcances:
        print(f"\n=== Entrenando: {alcance} ===")
        resultado = servicio.entrenar(
            alcance, args.desde, args.hasta, n_trials=args.trials, n_splits=args.splits
        )
        print(resultado.comparativa.to_string(index=False))
        print(
            f"Campeón: {resultado.campeon} | MAE CV: {resultado.metricas_cv_campeon.get('mae', 0):,.0f}"
        )
    return 0


def _comando_estado(_args: argparse.Namespace) -> int:
    """Muestra el estado de datos y modelos."""
    settings = get_settings()
    catalogo = ServicioCatalogo(settings)
    estado = catalogo.estado_datos()
    print("Datos disponibles:", ", ".join(estado["archivos"]) or "(ninguno)")
    if estado["rango"]:
        print(f"Rango de usos: {estado['rango'][0]} -> {estado['rango'][1]}")
    print("\nModelos entrenados:")
    for meta in catalogo.modelos_entrenados():
        print(
            f"  - {meta['nombre']}: MAE={meta['metricas'].get('mae', 0):,.0f} ({meta['fecha_entrenamiento']})"
        )
    return 0


def _comando_secreto(args: argparse.Namespace) -> int:
    """Genera el hash de la clave y el token cifrado para el archivo ``.env``.

    Nunca imprime los valores en claro.

    Args:
        args: argumentos con ``clave`` (contraseña) y ``token`` (opcional).

    Returns:
        Código de salida.
    """
    from mio_demanda.infrastructure.secretos import cifrar, generar_hash

    if args.clave:
        print("MIO_ACCESO_CLAVE_HASH=" + generar_hash(args.clave))
    if args.token:
        cifrado = cifrar(args.token)
        if cifrado:
            print("MIO_JEV_API_KEY_CIFRADA=" + cifrado)
        else:
            print(
                "No se pudo cifrar el token (DPAPI no disponible). "
                "Déjalo en MIO_JEV_API_KEY y no compartas el archivo .env."
            )
            return 1
    if not args.clave and not args.token:
        print("Nada que generar: usa --clave y/o --token.")
        return 2
    print("\nCopia esas líneas en .env y borra MIO_ACCESO_CLAVE / MIO_JEV_API_KEY.")
    return 0


def _comando_usuario(args: argparse.Namespace) -> int:
    """Genera el bloque de usuario con su clave hasheada, listo para pegar.

    Args:
        args: argumentos con ``nombre`` y ``clave``.

    Returns:
        Código de salida.
    """
    from mio_demanda.infrastructure.secretos import generar_hash

    nombre = args.nombre.strip()
    if not nombre:
        print("El nombre de usuario no puede estar vacío.")
        return 2
    huella = generar_hash(args.clave)
    print("# Para .streamlit/secrets.toml (nube) o para el área de secretos de Streamlit:")
    print("")
    print("[usuarios]")
    print(f'"{nombre}" = "{huella}"')
    print("")
    print("# Para .env (local), todo en una línea:")
    print(f'MIO_USUARIOS_ACCESO={{"{nombre}": "{huella}"}}')
    print("")
    print("La contraseña no se guarda ni se imprime.")
    return 0


def _comando_rutas(args: argparse.Namespace) -> int:
    """Imprime las rutas de datos configuradas y si cada una existe.

    Args:
        args: argumentos de la CLI (no se usan).

    Returns:
        Código de salida: 0 si toda la cadena de origen existe, 1 si falta algo.
    """
    ajustes = get_settings()
    print(f"Directorio del proyecto: {ajustes.data_raw_dir.parent.parent}")
    print(f"Origen configurado:      {ajustes.directorio_base_pbi}")
    print()
    faltan = 0
    for fila in ajustes.rutas_de_datos():
        estado = "OK     " if fila["Existe"] else "FALTA  "
        if not fila["Existe"]:
            faltan += 1
        print(f"{estado}{fila['Nombre']:<26} {fila['Ruta']}")
    print()
    print(
        "Cambia cualquier ruta en .env (MIO_DIRECTORIO_BASE_PBI, MIO_DATA_RAW_DIR, "
        "MIO_MODELS_DIR, MIO_CONFIG_DIR, MIO_SUBCARPETA_USOS, MIO_SUBCARPETA_DIMENSIONES) "
        "y vuelve a ejecutar el comando."
    )
    return 0 if faltan == 0 else 1


def construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de la CLI.

    Returns:
        El parser configurado con todos los subcomandos.
    """
    parser = argparse.ArgumentParser(prog="ml-demanda", description="Proyección de demanda del MIO")
    parser.add_argument("--log-level", default=None, help="Nivel de logging (INFO, DEBUG, ...)")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    subparsers.add_parser("extract", help="Extrae y normaliza los datos crudos a data/raw")
    subparsers.add_parser("estado", help="Muestra el estado de datos y modelos")
    subparsers.add_parser("rutas", help="Muestra de dónde lee y dónde escribe la aplicación")

    usuario = subparsers.add_parser(
        "usuario", help="Genera el bloque [usuarios] (con hash) para .env o st.secrets"
    )
    usuario.add_argument("--nombre", required=True, help="Nombre de usuario")
    usuario.add_argument("--clave", required=True, help="Contraseña a hashear")

    secreto = subparsers.add_parser(
        "secreto", help="Genera el hash de la clave y el token cifrado para .env"
    )
    secreto.add_argument("--clave", default=None, help="Contraseña a hashear")
    secreto.add_argument("--token", default=None, help="Token de API a cifrar con DPAPI")

    train = subparsers.add_parser("train", help="Entrena los modelos")
    train.add_argument("--alcance", choices=[*ALCANCES, "todo"], default="todo")
    train.add_argument("--desde", type=_parse_fecha, default=date(get_settings().anio_inicio, 1, 1))
    train.add_argument("--hasta", type=_parse_fecha, default=date.today())
    train.add_argument("--trials", type=int, default=None, help="Trials de Optuna por candidato")
    train.add_argument(
        "--splits", type=int, default=None, help="Particiones de la validación cruzada"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la CLI.

    Args:
        argv: argumentos de línea de comandos (por defecto, ``sys.argv``).

    Returns:
        Código de salida del proceso.
    """
    parser = construir_parser()
    args = parser.parse_args(argv)
    configurar_logging(args.log_level or get_settings().log_level)

    manejadores = {
        "extract": _comando_extract,
        "train": _comando_train,
        "estado": _comando_estado,
        "rutas": _comando_rutas,
        "usuario": _comando_usuario,
        "secreto": _comando_secreto,
    }
    return manejadores[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
