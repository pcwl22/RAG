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
from urllib.parse import urlsplit

import yaml


def _load_dotenv_if_available(path: Path) -> None:
    """Load an env file without making python-dotenv a hard import dependency."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv 未安装时降级
        return
    load_dotenv(path)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# 默认使用笔记本配置（详见 PROJECT_MEMORY.md 推荐）
DEFAULT_ENV = "laptop"

# 匹配 ${VAR} 占位符
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")
ALLOWED_QUEUE_PROVIDERS = frozenset({"memory", "celery"})
ALLOWED_PDF_ENGINES = frozenset({"auto", "mineru", "pymupdf"})


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


def validate_runtime_config(config: dict[str, Any]) -> None:
    """Fail fast when runtime settings would otherwise be silently ignored."""
    resolve_queue_provider(config)
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

    postgres = config.get("postgres", {})
    debug = bool(config.get("app", {}).get("debug", False))
    postgres_user = str(postgres.get("user", "")).strip()
    postgres_password = str(postgres.get("password", "")).strip()
    if not debug and postgres_user and postgres_user != "postgres":
        if not postgres_password or postgres_password.startswith("${"):
            raise ValueError(
                "POSTGRES_APP_PASSWORD must be configured for the PostgreSQL runtime user"
            )
    min_pool = int(postgres.get("min_pool_size", 1))
    max_pool = int(postgres.get("max_pool_size", 5))
    if min_pool < 1 or max_pool < min_pool:
        raise ValueError("postgres pool sizes must satisfy 1 <= min_pool_size <= max_pool_size")
    probes = int(postgres.get("ivfflat_probes", 10))
    if probes < 1:
        raise ValueError("postgres.ivfflat_probes must be at least 1")

    redis = config.get("redis", {})
    redis_enabled = bool(redis.get("enabled", True))
    redis_url = str(os.getenv("REDIS_URL") or redis.get("url", "")).strip()
    if not debug and redis_enabled:
        if not redis_url or redis_url.startswith("${"):
            raise ValueError("REDIS_URL must be configured in production mode")
        parsed_redis = urlsplit(redis_url)
        if parsed_redis.scheme not in {"redis", "rediss"}:
            raise ValueError("REDIS_URL must use redis:// or rediss://")
        if not parsed_redis.password:
            raise ValueError("REDIS_URL must include a Redis password in production mode")
        require_tls = str(os.getenv("REDIS_REQUIRE_TLS") or redis.get("require_tls", "")).lower()
        if require_tls in {"1", "true", "yes", "on"} and parsed_redis.scheme != "rediss":
            raise ValueError("REDIS_URL must use rediss:// when REDIS_REQUIRE_TLS is enabled")

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


@lru_cache
def get_settings() -> dict[str, Any]:
    """
    获取配置（带缓存）。

    Returns:
        配置字典，其中的 ${VAR} 占位符已用环境变量替换。
    """
    # 加载 .env（如果可用且存在）
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
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
