from pathlib import Path

import yaml


def test_base_llm_profile_uses_current_provider_schema():
    config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    text_config = config["llm"]["text"]
    provider = text_config["provider"]

    assert provider == "openai_compatible"
    assert text_config[provider]["model_name"] == "deepseek-chat"
    assert text_config[provider]["api_key"] == "${DEEPSEEK_API_KEY}"


def test_frontend_proxy_exposes_readiness_endpoint():
    config_path = Path(__file__).resolve().parents[2] / "rag-frontend" / "nginx" / "default.conf.template"
    config = config_path.read_text(encoding="utf-8")

    assert "location = /health/ready" in config
    assert "proxy_pass ${RAG_API_UPSTREAM}/health/ready;" in config
