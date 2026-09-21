"""
配置管理模块

配置以字典形式返回（与项目其余部分的 `config[...]` 访问方式一致）。

加载逻辑：
1. 读取 `.env`（若存在），把变量注入环境。
2. 根据环境变量 `RAG_ENV` 选择配置文件 `config/{RAG_ENV}.yaml`，
   默认 `laptop`（笔记本推荐配置）。若该文件不存在则回退到 `config/base.yaml`。
3. 递归替换配置中的 `${VAR}` 占位符为对应环境变量值。
"""
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import yaml

from app.utils.strict_dotenv import apply_release_env_file


def _load_dotenv_if_available(path: Path, *, override: bool = False) -> None:
    """Load an env file without making python-dotenv a hard import dependency."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv 未安装时降级
        return
    load_dotenv(path, override=override)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# 默认使用笔记本配置（详见 PROJECT_MEMORY.md 推荐）
DEFAULT_ENV = "laptop"

# 匹配 ${VAR} 占位符
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")
ALLOWED_QUEUE_PROVIDERS = frozenset({"memory", "celery"})
ALLOWED_PDF_ENGINES = frozenset({"auto", "mineru", "pymupdf"})
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def normalize_openai_base_url(value: str) -> str:
    """Normalize an OpenAI-compatible endpoint without exposing credentials.

    NewAPI deployments commonly provide a bare host while the OpenAI client
    expects the versioned API prefix.  Preserve an explicitly configured path
    and add ``/v1`` only when the endpoint has no path.
    """
    raw = str(value or "").strip().rstrip("/")
    if not raw or raw.startswith("${"):
        return raw
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM base_url must be an absolute http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base_url must not contain a query string or fragment")
    path = parsed.path.rstrip("/") or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def resolve_queue_provider(config: dict[str, Any]) -> str:
    """Resolve and validate the ingestion queue provider."""
    provider = str(
        os.getenv("QUEUE_PROVIDER") or config.get("queue", {}).get("provider", "memory")
    ).strip().lower()
    if provider == "rabbitmq":
        provider = "celery"
    if provider not in ALLOWED_QUEUE_PROVIDERS:
        allowed = ", ".join(sorted(ALLOWED_QUEUE_PROVIDERS))
        raise ValueError(f"Unsupported queue provider '{provider}'. Expected one of: {allowed}")
    return provider


def resolve_object_storage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve S3-compatible storage settings with process-env precedence."""
    section = config.get("object_storage", {})
    if not isinstance(section, dict):
        raise ValueError("object_storage must be a mapping")

    def value(env_name: str, config_name: str, default: Any = "") -> Any:
        if env_name in os.environ:
            return os.environ[env_name].strip()
        return section.get(config_name, default)

    enabled_value = value("OBJECT_STORAGE_ENABLED", "enabled", False)
    enabled = str(enabled_value).strip().lower() in TRUE_VALUES
    ca_bundle = str(value("S3_CA_BUNDLE", "ca_bundle", "")).strip()
    if ca_bundle.startswith("${") and ca_bundle.endswith("}"):
        ca_bundle = ""

    def positive_int(
        env_name: str,
        config_name: str,
        default: int,
        *,
        maximum: int,
    ) -> int:
        raw = value(env_name, config_name, default)
        try:
            resolved = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_name} must be an integer") from exc
        if resolved < 1 or resolved > maximum:
            raise ValueError(f"{env_name} must be between 1 and {maximum}")
        return resolved

    return {
        "enabled": enabled,
        "endpoint_url": str(value("S3_ENDPOINT_URL", "endpoint_url", "")).strip(),
        "bucket": str(value("S3_BUCKET", "bucket", "")).strip(),
        "region": str(value("S3_REGION", "region", "us-east-1")).strip(),
        "access_key_id": str(value("S3_ACCESS_KEY_ID", "access_key_id", "")).strip(),
        "secret_access_key": str(value("S3_SECRET_ACCESS_KEY", "secret_access_key", "")).strip(),
        "require_tls": str(value("S3_REQUIRE_TLS", "require_tls", "true")).strip().lower()
        in TRUE_VALUES,
        "addressing_style": str(value("S3_ADDRESSING_STYLE", "addressing_style", "path"))
        .strip()
        .lower(),
        "failed_prefix": str(
            value("S3_FAILED_PREFIX", "failed_prefix", "failed")
        ).strip("/"),
        "ca_bundle": ca_bundle,
        "connect_timeout_seconds": positive_int(
            "S3_CONNECT_TIMEOUT_SECONDS",
            "connect_timeout_seconds",
            3,
            maximum=60,
        ),
        "read_timeout_seconds": positive_int(
            "S3_READ_TIMEOUT_SECONDS",
            "read_timeout_seconds",
            15,
            maximum=300,
        ),
        "max_pool_connections": positive_int(
            "S3_MAX_POOL_CONNECTIONS",
            "max_pool_connections",
            20,
            maximum=1000,
        ),
    }


def validate_runtime_config(config: dict[str, Any]) -> None:
    """Fail fast when runtime settings would otherwise be silently ignored."""
    queue_provider = resolve_queue_provider(config)
    queue_config = config.get("queue", {})
    reconciliation = (
        queue_config.get("reconciliation", {}) if isinstance(queue_config, dict) else {}
    )
    if not isinstance(reconciliation, dict):
        raise ValueError("queue.reconciliation must be a mapping")
    reconciliation_enabled_raw = os.getenv("UPLOAD_RECONCILIATION_ENABLED")
    reconciliation_enabled = (
        str(reconciliation_enabled_raw).strip().lower() in TRUE_VALUES
        if reconciliation_enabled_raw is not None
        else bool(reconciliation.get("enabled", True))
    )
    if queue_provider == "celery" and not reconciliation_enabled:
        raise ValueError("upload reconciliation must be enabled for the Celery provider")
    reconciliation_limits = {
        "interval_seconds": ("UPLOAD_RECONCILIATION_INTERVAL_SECONDS", 3600),
        "redispatch_after_seconds": ("UPLOAD_REDISPATCH_AFTER_SECONDS", 86400),
        "batch_size": ("UPLOAD_RECONCILIATION_BATCH_SIZE", 1000),
        "dispatch_lease_seconds": ("UPLOAD_DISPATCH_LEASE_SECONDS", 3600),
        "max_dispatch_attempts": ("UPLOAD_MAX_DISPATCH_ATTEMPTS", 100),
        "completion_receipt_retention_seconds": (
            "UPLOAD_COMPLETION_RECEIPT_RETENTION_SECONDS",
            2592000,
        ),
        "producer_timeout_seconds": ("UPLOAD_PRODUCER_TIMEOUT_SECONDS", 60),
    }
    for key, (env_name, maximum) in reconciliation_limits.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            raw_value = reconciliation.get(key, 1)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_name} must be an integer") from exc
        if value < 1 or value > maximum:
            raise ValueError(f"{env_name} must be between 1 and {maximum}")
    pdf_engine = str(
        config.get("document_processing", {}).get("pdf", {}).get("engine", "auto")
    ).strip().lower()
    if pdf_engine not in ALLOWED_PDF_ENGINES:
        allowed = ", ".join(sorted(ALLOWED_PDF_ENGINES))
        raise ValueError(f"Unsupported PDF engine '{pdf_engine}'. Expected one of: {allowed}")

    logging_config = config.get("logging", {})
    level = str(logging_config.get("level", "INFO")).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Unsupported logging level: {level}")
    log_format = str(logging_config.get("format", "json")).lower()
    if log_format not in {"json", "text"}:
        raise ValueError(f"Unsupported logging format: {log_format}")

    performance = config.get("performance", {})
    for key in ("max_concurrent_requests", "gpu_inference_workers"):
        if int(performance.get(key, 1)) < 1:
            raise ValueError(f"performance.{key} must be at least 1")

    for section_name in ("embedding", "reranker"):
        accelerator = config.get(section_name, {})
        allow_cpu_fallback = accelerator.get("allow_cpu_fallback", True)
        if not isinstance(allow_cpu_fallback, bool):
            raise ValueError(f"{section_name}.allow_cpu_fallback must be a boolean")

    generation = config.get("rag", {}).get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("rag.generation must be a mapping")
    answer_contract = generation.get("answer_contract", {})
    if not isinstance(answer_contract, dict):
        raise ValueError("rag.generation.answer_contract must be a mapping")
    contract_enabled = bool(answer_contract.get("enabled", False))
    try:
        contract_retries = int(answer_contract.get("max_retries", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("rag.generation.answer_contract.max_retries must be 0 or 1") from exc
    if contract_retries not in {0, 1}:
        raise ValueError("rag.generation.answer_contract.max_retries must be 0 or 1")

    postgres = config.get("postgres", {})
    debug = bool(config.get("app", {}).get("debug", False))
    postgres_user = str(postgres.get("user", "")).strip()
    postgres_password = str(postgres.get("password", "")).strip()
    postgres_sslmode = str(postgres.get("sslmode", "")).strip().lower()
    if postgres_sslmode.startswith("${"):
        postgres_sslmode = ""
    allowed_ssl_modes = {"", "disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
    if postgres_sslmode not in allowed_ssl_modes:
        raise ValueError("POSTGRES_SSLMODE is not a supported libpq SSL mode")
    secure_mode_enabled = str(os.getenv("RAG_SECURE_MODE", "")).strip().lower() in TRUE_VALUES
    rag_env = os.getenv("RAG_ENV", DEFAULT_ENV).strip().lower()
    secure_database_required = secure_mode_enabled or (not debug and rag_env == "base")
    postgres_sslrootcert = str(postgres.get("sslrootcert", "")).strip()
    if postgres_sslrootcert.startswith("${"):
        postgres_sslrootcert = ""
    if not debug and postgres_user and postgres_user != "postgres":
        if not postgres_password or postgres_password.startswith("${"):
            raise ValueError(
                "POSTGRES_APP_PASSWORD must be configured for the PostgreSQL runtime user"
            )
    if secure_database_required:
        if postgres_sslmode != "verify-full":
            raise ValueError(
                "POSTGRES_SSLMODE must be verify-full in secure/production mode"
            )
        if not postgres_sslrootcert:
            raise ValueError(
                "POSTGRES_SSLROOTCERT must be configured in secure/production mode"
            )
    min_pool = int(postgres.get("min_pool_size", 1))
    max_pool = int(postgres.get("max_pool_size", 5))
    if min_pool < 1 or max_pool < min_pool:
        raise ValueError("postgres pool sizes must satisfy 1 <= min_pool_size <= max_pool_size")
    if queue_provider == "celery" and max_pool < 2:
        raise ValueError(
            "postgres.max_pool_size must be at least 2 for Celery advisory leases"
        )
    probes = int(postgres.get("ivfflat_probes", 10))
    if probes < 1:
        raise ValueError("postgres.ivfflat_probes must be at least 1")

    redis = config.get("redis", {})
    redis_url = str(os.getenv("REDIS_URL") or redis.get("url", "")).strip()
    configured_redis_url = bool(redis_url and not redis_url.startswith("${"))
    redis_validation_required = (
        queue_provider == "celery"
        or redis.get("enabled") is True
        or configured_redis_url
    )
    if not debug and redis_validation_required:
        if not redis_url or redis_url.startswith("${"):
            raise ValueError("REDIS_URL must be configured in production mode")
        parsed_redis = urlsplit(redis_url)
        if parsed_redis.scheme not in {"redis", "rediss"}:
            raise ValueError("REDIS_URL must use redis:// or rediss://")
        if not parsed_redis.password:
            raise ValueError("REDIS_URL must include a Redis password in production mode")
        require_tls = str(os.getenv("REDIS_REQUIRE_TLS") or redis.get("require_tls", "")).lower()
        secure_mode = str(os.getenv("RAG_SECURE_MODE", "")).strip().lower()
        tls_required = require_tls in {"1", "true", "yes", "on"} or secure_mode in {
            "1",
            "true",
            "yes",
            "on",
        }
        if tls_required and parsed_redis.scheme != "rediss":
            raise ValueError(
                "REDIS_URL must use rediss:// when REDIS_REQUIRE_TLS or "
                "RAG_SECURE_MODE is enabled"
            )

    object_storage = resolve_object_storage_config(config)
    if not debug and rag_env == "base" and not contract_enabled:
        raise ValueError("rag.generation.answer_contract.enabled must be true in production mode")
    if not debug and rag_env == "base" and queue_provider == "celery" and not object_storage["enabled"]:
        raise ValueError(
            "OBJECT_STORAGE_ENABLED must be true for Celery uploads in base/production mode"
        )
    if object_storage["enabled"]:
        for key in (
            "endpoint_url",
            "bucket",
            "region",
            "access_key_id",
            "secret_access_key",
        ):
            if not object_storage[key] or str(object_storage[key]).startswith("${"):
                raise ValueError(f"S3 {key} must be configured when object storage is enabled")
        parsed_s3 = urlsplit(str(object_storage["endpoint_url"]))
        if parsed_s3.scheme not in {"http", "https"} or not parsed_s3.netloc:
            raise ValueError("S3 endpoint_url must be an absolute http:// or https:// URL")
        if object_storage["require_tls"] and parsed_s3.scheme != "https":
            raise ValueError("S3 endpoint_url must use https:// when S3_REQUIRE_TLS is enabled")
        if object_storage["addressing_style"] not in {"path", "virtual"}:
            raise ValueError("S3_ADDRESSING_STYLE must be path or virtual")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(object_storage["failed_prefix"])):
            raise ValueError(
                "S3_FAILED_PREFIX must be a single safe path segment"
            )

    reranker_mode = str(config.get("reranker", {}).get("failure_mode", "closed")).lower()
    if reranker_mode not in {"closed", "open"}:
        raise ValueError("reranker.failure_mode must be either 'closed' or 'open'")


def _resolve_config_path() -> Path:
    """根据 RAG_ENV 决定加载哪个配置文件。"""
    env = os.getenv("RAG_ENV", DEFAULT_ENV).strip()

    candidate = CONFIG_DIR / f"{env}.yaml"
    if candidate.exists():
        return candidate

    # 回退到 base.yaml
    base = CONFIG_DIR / "base.yaml"
    if base.exists():
        return base

    raise FileNotFoundError(
        f"找不到配置文件：既没有 {CONFIG_DIR / f'{env}.yaml'}，也没有 {CONFIG_DIR / 'base.yaml'}"
    )


def _substitute_env_vars(obj: Any) -> Any:
    """递归地把字符串中的 ${VAR} 替换为环境变量值。

    若环境变量不存在，保留原始占位符（便于排查缺失的配置）。
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    if isinstance(obj, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return _ENV_VAR_PATTERN.sub(_replace, obj)
    return obj


def _apply_postgres_env_overrides(
    config: dict[str, Any], *, port_env_key: str = "POSTGRES_PORT"
) -> None:
    """Apply explicit PostgreSQL connection overrides after YAML substitution.

    ``POSTGRES_PORT`` is the service-to-service port inside Compose.  The
    laptop profile runs on the host and therefore uses the published port;
    callers can pass ``RAG_POSTGRES_PORT`` for that profile without changing
    the container contract.
    """
    postgres = config.setdefault("postgres", {})
    for config_key, env_key in (
        ("host", "POSTGRES_HOST"),
        ("port", port_env_key),
        ("database", "POSTGRES_DB"),
        ("user", "POSTGRES_USER"),
        ("password", "POSTGRES_PASSWORD"),
        ("sslmode", "POSTGRES_SSLMODE"),
        ("sslrootcert", "POSTGRES_SSLROOTCERT"),
    ):
        value = os.getenv(env_key)
        if value is not None and value.strip():
            postgres[config_key] = value.strip()


def _apply_accelerator_env_overrides(config: dict[str, Any]) -> None:
    """Apply per-workload device choices after the shared YAML is loaded."""
    mappings = (
        ("EMBEDDING_DEVICE", ("embedding", "device")),
        ("RERANKER_DEVICE", ("reranker", "device")),
        ("PDF_DEVICE", ("document_processing", "pdf", "device")),
        ("MULTIMODAL_DEVICE", ("multimodal", "image", "device")),
    )
    for env_name, path in mappings:
        value = os.getenv(env_name, "").strip().lower()
        if not value:
            continue
        if value not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"{env_name} must be one of: cpu, cuda, mps")
        section: dict[str, Any] = config
        for key in path[:-1]:
            nested = section.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"configuration section {'.'.join(path[:-1])} must be a mapping")
            section = cast(dict[str, Any], nested)
        section[path[-1]] = value


@lru_cache
def get_settings() -> dict[str, Any]:
    """
    获取配置（带缓存）。

    Returns:
        配置字典，其中的 ${VAR} 占位符已用环境变量替换。
    """
    # A protected release can point at a Secret Manager snapshot without
    # copying secrets into the checkout. Release mode uses the exact same
    # strict parser as validator/renderers and replaces ambient app settings.
    configured_env_file = os.getenv("RAG_ENV_FILE", "").strip()
    env_file = Path(configured_env_file) if configured_env_file else PROJECT_ROOT / ".env"
    if env_file.exists():
        release_snapshot = os.getenv("RAG_RELEASE_SNAPSHOT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if release_snapshot:
            apply_release_env_file(env_file)
        else:
            _load_dotenv_if_available(env_file)

    config_path = _resolve_config_path()

    with open(config_path, encoding="utf-8") as f:
        loaded_config: Any = yaml.safe_load(f)

    if loaded_config is None:
        config: dict[str, Any] = {}
    elif isinstance(loaded_config, dict):
        config = cast(dict[str, Any], loaded_config)
    else:
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    substituted_config = _substitute_env_vars(config)
    port_env_key = "RAG_POSTGRES_PORT" if config_path.stem == "laptop" else "POSTGRES_PORT"
    _apply_postgres_env_overrides(substituted_config, port_env_key=port_env_key)
    _apply_accelerator_env_overrides(substituted_config)
    if not isinstance(substituted_config, dict):
        raise ValueError(f"Configuration root must remain a mapping: {config_path}")
    config = cast(dict[str, Any], substituted_config)

    # 记录实际加载的配置来源，便于调试
    config.setdefault("_meta", {})["config_path"] = str(config_path)

    return config


def get_config_section(*path: str) -> dict[str, Any]:
    """
    获取嵌套配置段，缺失时返回空字典。

    直接链式调用 `get_settings().get("a", {}).get("b", {})` 会退化成 Any，
    类型检查形同虚设；这个入口保证返回值是真正的字典。

    Args:
        *path: 配置段路径，例如 get_config_section("rag", "retrieval")

    Returns:
        配置段字典；路径上任一层缺失时返回 {}

    Raises:
        TypeError: 路径上某一层存在但不是映射（配置写错，应尽早暴露而非静默降级）
    """
    section: dict[str, Any] = get_settings()
    for depth, key in enumerate(path):
        value = section.get(key, {})
        if not isinstance(value, dict):
            located = ".".join(path[: depth + 1])
            raise TypeError(
                f"Configuration section '{located}' must be a mapping, "
                f"got {type(value).__name__}"
            )
        section = cast(dict[str, Any], value)
    return section


def reload_settings() -> dict[str, Any]:
    """
    重新加载配置（清除缓存）。

    Returns:
        配置字典
    """
    get_settings.cache_clear()
    return get_settings()
