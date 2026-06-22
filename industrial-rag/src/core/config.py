"""
配置管理模块

配置以字典形式返回（与项目其余部分的 `config[...]` 访问方式一致）。

加载逻辑：
1. 读取 `.env`（若存在），把变量注入环境。
2. 根据环境变量 `RAG_ENV` 选择配置文件 `configs/{RAG_ENV}.yaml`，
   默认 `laptop`（笔记本推荐配置）。若该文件不存在则回退到 `configs/base.yaml`。
3. 递归替换配置中的 `${VAR}` 占位符为对应环境变量值。
"""
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 未安装时降级
    load_dotenv = None

# 项目根目录（src/core/config.py -> 上溯 3 级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 默认使用笔记本配置（详见 PROJECT_MEMORY.md 推荐）
DEFAULT_ENV = "laptop"

# 匹配 ${VAR} 占位符
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _resolve_config_path() -> Path:
    """根据 RAG_ENV 决定加载哪个配置文件。"""
    env = os.getenv("RAG_ENV", DEFAULT_ENV).strip()

    candidate = CONFIGS_DIR / f"{env}.yaml"
    if candidate.exists():
        return candidate

    # 回退到 base.yaml
    base = CONFIGS_DIR / "base.yaml"
    if base.exists():
        return base

    raise FileNotFoundError(
        f"找不到配置文件：既没有 {candidate}，也没有 {base}"
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


@lru_cache()
def get_settings() -> dict[str, Any]:
    """
    获取配置（带缓存）。

    Returns:
        配置字典，其中的 ${VAR} 占位符已用环境变量替换。
    """
    # 加载 .env（如果可用且存在）
    if load_dotenv is not None:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    config_path = _resolve_config_path()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = _substitute_env_vars(config)

    # 记录实际加载的配置来源，便于调试
    config.setdefault("_meta", {})["config_path"] = str(config_path)

    return config


def reload_settings() -> dict[str, Any]:
    """
    重新加载配置（清除缓存）。

    Returns:
        配置字典
    """
    get_settings.cache_clear()
    return get_settings()
