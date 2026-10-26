"""Configuración central del proyecto ML-Demanda.

Este módulo es la **única** fuente de configuración de la aplicación. Ninguna
otra capa debe leer variables de entorno ni construir rutas por su cuenta:
todas reciben una instancia de :class:`Settings` inyectada.

Reglas respetadas:
    * Prohibido "quemar" rutas, credenciales o parámetros: todo proviene de
      variables de entorno (``.env``) con valores por defecto razonables.
    * La configuración es inmutable y cacheada (``get_settings``).
"""

from __future__ import annotations

import hmac
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mio_demanda.infrastructure.secretos import descifrar, dpapi_disponible, verificar_hash

# Raíz del proyecto: <raiz>/src/mio_demanda/config.py  ->  parents[2]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Versión del esquema de configuración. **Súbela** al añadir o quitar campos o
#: métodos de :class:`Settings`: la aplicación la compara con la que espera su código
#: para detectar módulos viejos que Streamlit dejó en memoria.
CONFIG_VERSION = 4


class Settings(BaseSettings):
    """Configuración tipada y validada de ML-Demanda.

    Todos los campos pueden sobrescribirse con variables de entorno prefijadas
    con ``MIO_`` (por ejemplo ``MIO_ANIO_INICIO=2023``) o desde el archivo
    ``.env`` en la raíz del proyecto.
    """

    model_config = SettingsConfigDict(
        env_prefix="MIO_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Origen de datos crudos (Power BI / Metrocali) ---
    directorio_base_pbi: Path = Field(
        default=Path("data/origen"),
        description=(
            "Carpeta raíz de los reportes de la Oficina de Evaluación (solo lectura). "
            "Por defecto `data/origen` dentro del proyecto; en esta máquina se apunta a "
            "la carpeta de OneDrive con MIO_DIRECTORIO_BASE_PBI."
        ),
    )
    subcarpeta_usos: str = Field(
        default="Datos_usos", description="Subcarpeta del origen con los parquets de demanda."
    )
    subcarpeta_dimensiones: str = Field(
        default="Datos_dimensiones", description="Subcarpeta del origen con las tablas maestras."
    )
    archivo_calendario: str = Field(
        default="dim_Calendario.xlsx", description="Nombre del calendario en el origen."
    )

    # --- Directorios de trabajo del proyecto (relativos a la raíz) ---
    data_raw_dir: Path = Field(default=Path("data/raw"))
    models_dir: Path = Field(default=Path("data/models"))
    config_dir: Path = Field(default=Path("config"))

    # --- MLflow (tracking local) ---
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: f"sqlite:///{(PROJECT_ROOT / 'mlruns.db').as_posix()}",
        description=(
            "Base de datos de MLflow. Absoluta por defecto para que no dependa de la "
            "carpeta desde donde se ejecute la aplicación."
        ),
    )
    mlflow_experiment: str = Field(default="mio-demanda")
    mlflow_ui_url: str = Field(
        default="http://127.0.0.1:5000",
        description=(
            "Dirección de la interfaz web de MLflow. La app solo enlaza ahí: el análisis "
            "detallado de los entrenamientos se hace en MLflow, no en la app."
        ),
    )

    # --- Entrenamiento ---
    random_state: int = Field(default=42)
    anio_inicio: int = Field(default=2022, description="Año mínimo de historia para entrenar.")
    n_splits_cv: int = Field(
        default=4, description="Número de particiones de la validación cruzada temporal."
    )
    n_trials_optuna: int = Field(
        default=25, description="Presupuesto de búsqueda de hiperparámetros por modelo."
    )

    # --- Ingeniería de características (ventanas de las anclas) ---
    dias_calentamiento: int = Field(
        default=120, description="Días de historia previa cargados para estimar las anclas."
    )
    ventana_reciente_dias: int = Field(
        default=56, description="Ventana del ancla reciente (nivel de la demanda)."
    )
    min_obs_ancla: int = Field(default=7, description="Observaciones mínimas para emitir un ancla.")
    min_cobertura_entidad: float = Field(
        default=0.5,
        description="Fracción mínima de días con datos para conservar una entidad en el modelo.",
    )
    dias_historia_perfil: int = Field(
        default=120, description="Días de historia usados para el perfil horario y la oferta."
    )
    hora_inicio_jornada: int = Field(
        default=4,
        description=(
            "Primera hora de la jornada operativa: se grafica y se exporta "
            "(la operación del MIO arranca de madrugada)."
        ),
    )
    hora_fin_jornada: int = Field(
        default=23,
        description="Última hora de la jornada operativa: se grafica y se exporta.",
    )
    dias_actividad_entidad: int = Field(
        default=90,
        description=(
            "Días recientes en los que una entidad debe tener usos para ofrecerse "
            "como proyectable (descarta rutas retiradas o nombres en desuso)."
        ),
    )
    max_filas_tuning_entidad: int = Field(
        default=25_000,
        description="Filas más recientes usadas en la afinación de alcances por entidad.",
    )

    # --- Proyección ---
    factor_banda_error: float = Field(
        default=1.2816,
        description="Factor del intervalo P10-P90 (1.2816 bajo normalidad) del rango esperado.",
    )
    coherencia_min: float = Field(
        default=0.5, description="Razón mínima aceptable entre lo proyectado y lo real reciente."
    )
    coherencia_max: float = Field(
        default=2.0, description="Razón máxima aceptable entre lo proyectado y lo real reciente."
    )
    dias_referencia_reciente: int = Field(
        default=30, description="Días reales usados como referencia de coherencia."
    )
    retrospectiva_dias: int = Field(
        default=15,
        description=(
            "Días hacia atrás que se proyectan con el modelo para compararlos con la "
            "demanda real ya observada (retrospectiva del vigilante)."
        ),
    )

    # --- Vigilancia del modelo (¿sigue sirviendo lo que entrenamos?) ---
    vigilancia_dias: int = Field(
        default=30, description="Días reales recientes usados para vigilar el modelo."
    )
    vigilancia_min_dias: int = Field(
        default=10, description="Días reales mínimos para emitir un veredicto de vigilancia."
    )
    vigilancia_factor_alerta: float = Field(
        default=1.5,
        description="Veces que el WAPE reciente puede superar al de validación antes de alertar.",
    )
    vigilancia_sesgo_alerta: float = Field(
        default=0.05, description="Sesgo relativo sostenido que dispara la alerta."
    )
    vigilancia_dias_validez: int = Field(
        default=7, description="Días de validez de la última revisión antes de sugerir otra."
    )
    vigilancia_max_predicciones: int = Field(
        default=60,
        description="Predicciones guardadas que se conservan para comparar con la realidad.",
    )
    umbral_dia_parcial: float = Field(
        default=0.5,
        description=(
            "Fracción de la mediana del mismo día de la semana por debajo de la cual el "
            "último día se considera a medio cargar y se descarta."
        ),
    )

    # --- Segunda opinión (TypeSafe Jev): no decide, solo aporta su predicción ---
    jev_api_key: SecretStr | None = Field(
        default=None,
        description="Clave de la API de TypeSafe (Jev) en claro. Prefiere MIO_JEV_API_KEY_CIFRADA.",
    )
    jev_api_key_cifrada: str = Field(
        default="",
        description=(
            "Clave de Jev cifrada con DPAPI (dpapi:...). Solo se descifra en tu usuario de "
            "Windows; se genera con 'uv run python -m mio_demanda.cli secreto'."
        ),
    )
    jev_base_url: str = Field(default="https://api.typesafe.ai", description="URL base de la API.")
    jev_modelo: str = Field(default="jev-latest", description="Modelo o alias a consultar.")
    jev_timeout_segundos: float = Field(default=90.0, description="Espera máxima por respuesta.")
    jev_max_dias: int = Field(
        default=31,
        description=(
            "Días por consulta a la segunda opinión. Un horizonte mayor se trocea en "
            "varias llamadas y se une el resultado."
        ),
    )
    jev_niveles: int = Field(
        default=9,
        description="Niveles de la rúbrica de demanda (la API admite hasta 10; impar = centrada).",
    )
    jev_amplitud: float = Field(
        default=0.15, description="Amplitud de la rúbrica a cada lado del nivel reciente."
    )

    # --- Acceso a la aplicación ---
    usuarios_acceso: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Usuarios de la aplicación y el hash PBKDF2 de su clave, en JSON "
            "({'maortiz': 'pbkdf2_sha256$...'}). En la nube se define en st.secrets."
        ),
    )
    usuarios_admin: list[str] = Field(
        default_factory=list,
        description=(
            "Usuarios que pueden entrar a la pestaña de entrenamiento. Si está vacío, "
            "nadie la ve (solo se proyecta). Se define con MIO_USUARIOS_ADMIN."
        ),
    )
    modo: Literal["local", "nube"] | None = Field(
        default=None,
        description=(
            "Fuerza el modo de la aplicación. Si se deja vacío se detecta solo: local "
            "cuando el sistema puede cifrar con DPAPI (Windows del profesional) y nube "
            "en cualquier otro caso. En modo nube se oculta el entrenamiento y se "
            "desactivan las funciones que necesitan secretos locales."
        ),
    )
    acceso_usuario: str = Field(
        default="",
        description="Usuario único heredado (se sigue aceptando junto a usuarios_acceso).",
    )
    acceso_clave: SecretStr | None = Field(
        default=None,
        description="Clave en claro (solo para pruebas). Prefiere MIO_ACCESO_CLAVE_HASH.",
    )
    acceso_clave_hash: str = Field(
        default="",
        description=(
            "Hash PBKDF2 de la clave (pbkdf2_sha256$iteraciones$sal$huella). "
            "Se genera con 'uv run python -m mio_demanda.cli secreto'."
        ),
    )

    calendario_anios_futuros: int = Field(
        default=10,
        description=(
            "Años hacia adelante que ofrece el calendario. Los eventos estándar se "
            "calculan por regla para cualquier año."
        ),
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # Secretos resueltos (hash de la clave y token descifrado)
    # ------------------------------------------------------------------
    @property
    def es_nube(self) -> bool:
        """Indica si la aplicación corre fuera del equipo del profesional.

        Returns:
            ``True`` en la nube (o cuando ``MIO_MODO=nube`` lo fuerza). Se detecta por
            la ausencia de DPAPI, que solo existe en Windows y está atado al usuario.
        """
        if self.modo is not None:
            return self.modo == "nube"
        return not dpapi_disponible()

    @property
    def modo_efectivo(self) -> str:
        """Modo con el que está funcionando la aplicación.

        Returns:
            ``"local"`` o ``"nube"``.
        """
        return "nube" if self.es_nube else "local"

    def credenciales(self) -> dict[str, str]:
        """Usuarios con clave configurada, con su hash (o texto plano heredado).

        Returns:
            Mapa ``usuario -> hash``. ``usuarios_acceso`` manda; el usuario único de
            ``acceso_usuario`` se añade si no está ya en el mapa.
        """
        usuarios = dict(self.usuarios_acceso)
        if self.acceso_usuario:
            if self.acceso_clave_hash:
                usuarios.setdefault(self.acceso_usuario, self.acceso_clave_hash)
            elif self.acceso_clave is not None:
                usuarios.setdefault(self.acceso_usuario, self.acceso_clave.get_secret_value())
        return usuarios

    def usuario_valido(self, usuario: str, clave: str) -> bool:
        """Comprueba un par usuario/contraseña contra los accesos configurados.

        Args:
            usuario: usuario escrito en el formulario.
            clave: contraseña escrita en el formulario.

        Returns:
            ``True`` solo si el usuario existe y la clave coincide con su hash.
        """
        guardado = self.credenciales().get(usuario.strip())
        if not guardado:
            return False
        if guardado.startswith("pbkdf2"):
            return verificar_hash(clave, guardado)
        return hmac.compare_digest(clave, guardado)

    def clave_acceso_valida(self, clave: str) -> bool:
        """Comprueba la contraseña de la consola contra el hash (o el texto plano).

        Args:
            clave: contraseña escrita por el usuario.

        Returns:
            ``True`` si coincide con el hash configurado o, en su defecto, con la
            clave en claro.
        """
        if self.acceso_clave_hash:
            return verificar_hash(clave, self.acceso_clave_hash)
        if self.acceso_clave is not None:
            return hmac.compare_digest(clave, self.acceso_clave.get_secret_value())
        return False

    def clave_jev(self) -> str | None:
        """Token de Jev listo para usar (descifrado si está cifrado en reposo).

        Returns:
            El token en claro, o ``None`` si no hay ninguno configurado.
        """
        if self.jev_api_key is not None:
            return self.jev_api_key.get_secret_value()
        if self.jev_api_key_cifrada:
            return descifrar(self.jev_api_key_cifrada)
        return None

    # ------------------------------------------------------------------
    # Validadores: convierten valores relativos en absolutos a la raíz.
    # ------------------------------------------------------------------
    @field_validator(
        "data_raw_dir",
        "models_dir",
        "config_dir",
        "directorio_base_pbi",
        mode="after",
    )
    @classmethod
    def _resolver_relativo(cls, valor: Path) -> Path:
        """Si la ruta es relativa, la ancla a la raíz del proyecto."""
        return valor if valor.is_absolute() else (PROJECT_ROOT / valor)

    # ------------------------------------------------------------------
    # Rutas derivadas del origen crudo (solo lectura).
    # ------------------------------------------------------------------
    @property
    def ruta_usos(self) -> Path:
        """Carpeta de parquets de usos (demanda) del sistema."""
        return self.directorio_base_pbi / self.subcarpeta_usos

    @property
    def ruta_dimensiones(self) -> Path:
        """Carpeta de tablas maestras (calendario, estaciones, vértices)."""
        return self.directorio_base_pbi / self.subcarpeta_dimensiones

    def rutas_de_datos(self) -> list[dict[str, object]]:
        """Lista las rutas que la aplicación lee y escribe, con su estado.

        Sirve para que la interfaz muestre de dónde salen los datos y qué falta, sin
        que nadie tenga que abrir el código ni el ``.env``.

        Returns:
            Lista de diccionarios con ``Nombre``, ``Ruta`` y ``Existe``.
        """
        candidatas: list[tuple[str, Path]] = [
            ("Origen (raíz)", self.directorio_base_pbi),
            ("Origen · usos", self.ruta_usos),
            ("Origen · dimensiones", self.ruta_dimensiones),
            ("Calendario de origen", self.ruta_dimensiones / self.archivo_calendario),
            ("Parquet de usos", self.parquet_usos),
            ("Parquet de calendario", self.parquet_calendario),
            ("Modelos entrenados", self.models_dir),
            ("Eventos del calendario", self.archivo_eventos_personalizados),
        ]
        return [
            {"Nombre": nombre, "Ruta": str(ruta), "Existe": ruta.exists()}
            for nombre, ruta in candidatas
        ]

    # ------------------------------------------------------------------
    # Rutas de artefactos normalizados dentro del proyecto.
    # ------------------------------------------------------------------
    @property
    def parquet_usos(self) -> Path:
        return self.data_raw_dir / "usos.parquet"

    @property
    def parquet_calendario(self) -> Path:
        return self.data_raw_dir / "calendario.parquet"

    @property
    def archivo_eventos_personalizados(self) -> Path:
        """Archivo JSON con los eventos de calendario creados por el profesional."""
        return self.config_dir / "eventos_personalizados.json"

    def ensure_directorios(self) -> None:
        """Crea los directorios de trabajo si no existen."""
        for carpeta in (
            self.data_raw_dir,
            self.models_dir,
            self.config_dir,
        ):
            carpeta.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuración única (cacheada) de la aplicación.

    Returns:
        Settings: instancia inmutable y validada de la configuración.
    """
    return Settings()
