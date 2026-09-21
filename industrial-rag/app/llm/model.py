"""
LLM 模型接口封装

支持多 Provider：
- Claude (Anthropic API)
- DeepSeek (OpenAI-compatible API)
- 智谱 AI (GLM API)

统一接口：generate() / generate_stream()，根据配置自动选择 Provider。
"""
import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm.request_policy import structured_output_request_options
from app.utils.config import get_config_section, normalize_openai_base_url
from app.utils.logger import get_logger
from app.utils.metrics import LLM_DURATION, LLM_REQUESTS

logger = get_logger(__name__)

# 全局 LLM 客户端实例
_llm_client: "LLMClient | None" = None


def _is_retryable_openai_error(exc: Exception) -> bool:
    """Identify provider failures that are safe to retry once more.

    Authentication, model and request-validation errors must fail fast.  A
    few OpenAI-compatible gateways also surface a stale internal function
    reference as HTTP 400 even though the request itself is valid; that error
    is transient and has been observed to succeed on the next gateway route.
    Transport failures are also safe to retry because no provider response was
    received and the request is idempotent from this client's perspective.
    """
    if exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    message = str(exc).lower()
    return "function id" in message and "not found" in message


def _llm_config() -> dict[str, Any]:
    """获取 LLM 配置。"""
    return get_config_section("llm", "text")


def _get_provider() -> str:
    """获取当前配置的 Provider。"""
    cfg = _llm_config()
    return str(cfg.get("provider", "claude")).lower()


class LLMClient:
    """LLM 客户端，根据配置选择不同的 Provider。"""

    def __init__(self) -> None:
        self.provider = _get_provider()
        self.config = _llm_config().get(self.provider, {})

        if not self.config:
            raise ValueError(
                f"LLM provider '{self.provider}' not configured in llm.text.{self.provider}"
            )

        logger.info(f"Initializing LLM client: {self.provider}")
        self._client = self._init_client()
        self._health_lock = asyncio.Lock()
        self._health_checked_at = 0.0
        self._health_available: bool = False

    def _record_health(self, available: bool) -> None:
        self._health_available = available
        self._health_checked_at = time.monotonic()

    def _invalidate_health(self) -> None:
        """Require a live probe without treating every request error as an outage."""
        self._health_checked_at = 0.0

    async def check_health(self, *, force: bool = False) -> bool:
        """Check upstream availability while caching the result for readiness probes."""
        ttl = max(1.0, float(self.config.get("healthcheck_ttl_seconds", 30)))
        now = time.monotonic()
        if not force and self._health_checked_at and now - self._health_checked_at < ttl:
            return self._health_available

        async with self._health_lock:
            now = time.monotonic()
            if not force and self._health_checked_at and now - self._health_checked_at < ttl:
                return self._health_available
            try:
                if self.provider == "claude":
                    await self._client.models.list(limit=1)
                else:
                    models = self._client.models.list()
                    # OpenAI's async SDK returns an AsyncPaginator here rather
                    # than an awaitable page. Keep compatibility with simple
                    # test doubles and older compatible clients as well.
                    if hasattr(models, "__aiter__"):
                        async for _ in models:
                            break
                    else:
                        await models
            except Exception:
                self._record_health(False)
                logger.warning("LLM upstream health probe failed", exc_info=True)
                return False
            self._record_health(True)
            return True

    def _init_client(self) -> Any:
        """根据 Provider 初始化对应的客户端。"""
        if self.provider == "claude":
            return self._init_claude()
        elif self.provider in ("openai_compatible", "deepseek", "zhipu"):
            return self._init_openai_compatible()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _init_claude(self) -> Any:
        """初始化 Claude (Anthropic) 客户端。"""
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        api_key = self.config.get("api_key")
        if not api_key or api_key.startswith("${"):
            raise ValueError(
                "CLAUDE_API_KEY not set. Please set it in .env or environment."
            )

        return AsyncAnthropic(
            api_key=api_key,
            base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout", 60),
        )

    def _init_openai_compatible(self) -> Any:
        """初始化 OpenAI-compatible 客户端 (DeepSeek 等)。"""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package not installed. Run: pip install openai") from exc

        api_key = self.config.get("api_key")
        if not api_key or api_key.startswith("${"):
            raise ValueError(
                f"{self.provider.upper()}_API_KEY not set. Please set it in .env or environment."
            )

        raw_base_url = str(self.config.get("base_url", "")).strip()
        if not raw_base_url or raw_base_url.startswith("${"):
            raise ValueError(
                "DEEPSEEK_API_URL not set. Please set it in .env or environment."
            )

        return AsyncOpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(raw_base_url),
            timeout=self.config.get("timeout", 60),
            # The application owns the bounded retry policy in
            # ``_generate_openai``. Leaving the SDK default enabled nests two
            # retry loops (up to nine network attempts for a configured value
            # of two) and makes latency and telemetry non-deterministic.
            max_retries=0,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        structured_output: bool = False,
    ) -> str:
        """
        生成响应（非流式）。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选，覆盖配置中的默认值）
            temperature: 温度（可选）
            max_tokens: 最大 token 数（可选）

        Returns:
            生成的文本
        """
        started = time.perf_counter()
        outcome = "success"
        try:
            if self.provider == "claude":
                result = await self._generate_claude(prompt, system_prompt, temperature, max_tokens)
            elif self.provider in ("openai_compatible", "deepseek", "zhipu"):
                result = await self._generate_openai(
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    structured_output=structured_output,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception:
            outcome = "error"
            # A request can fail because of user input, context limits, content
            # policy, or provider-side validation. Those are not readiness
            # signals. Force the next readiness check to perform its own bounded
            # upstream probe instead of taking the whole instance out of service.
            self._invalidate_health()
            raise
        else:
            self._record_health(True)
            return result
        finally:
            LLM_REQUESTS.labels(
                provider=self.provider,
                operation="generate",
                outcome=outcome,
            ).inc()
            LLM_DURATION.labels(
                provider=self.provider,
                operation="generate",
            ).observe(time.perf_counter() - started)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        structured_output: bool = False,
    ) -> AsyncIterator[str]:
        """
        生成响应（流式）。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大 token 数

        Yields:
            生成的文本片段
        """
        started = time.perf_counter()
        outcome = "success"
        try:
            if self.provider == "claude":
                async for chunk in self._generate_claude_stream(
                    prompt, system_prompt, temperature, max_tokens
                ):
                    yield chunk
            elif self.provider in ("openai_compatible", "deepseek", "zhipu"):
                async for chunk in self._generate_openai_stream(
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens,
                    structured_output=structured_output,
                ):
                    yield chunk
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception:
            outcome = "error"
            self._invalidate_health()
            raise
        else:
            self._record_health(True)
        finally:
            LLM_REQUESTS.labels(
                provider=self.provider,
                operation="stream",
                outcome=outcome,
            ).inc()
            LLM_DURATION.labels(
                provider=self.provider,
                operation="stream",
            ).observe(time.perf_counter() - started)

    # --------------------------------------------------------------------------
    # Claude 实现
    # --------------------------------------------------------------------------
    async def _generate_claude(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.config.get("model_name", "claude-sonnet-4-6"),
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.get("temperature", 0.7),
            "max_tokens": max_tokens if max_tokens is not None else self.config.get("max_tokens", 4096),
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return str(response.content[0].text)

    async def _generate_claude_stream(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.config.get("model_name", "claude-sonnet-4-6"),
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.get("temperature", 0.7),
            "max_tokens": max_tokens if max_tokens is not None else self.config.get("max_tokens", 4096),
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    # --------------------------------------------------------------------------
    # OpenAI-compatible 实现（DeepSeek）
    # --------------------------------------------------------------------------
    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        *,
        structured_output: bool = False,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = max(0, min(int(self.config.get("max_retries", 2)), 5))
        retry_backoff = max(0.0, float(self.config.get("retry_backoff_seconds", 0.5)))
        for attempt in range(max_retries + 1):
            try:
                request_options = (
                    structured_output_request_options(
                        provider=self.provider,
                        model_name=str(self.config.get("model_name", "deepseek-chat")),
                        base_url=str(self.config.get("base_url") or ""),
                    )
                    if structured_output
                    else {}
                )
                response = await self._client.chat.completions.create(
                    model=self.config.get("model_name", "deepseek-chat"),
                    messages=messages,
                    temperature=(
                        temperature
                        if temperature is not None
                        else self.config.get("temperature", 0.7)
                    ),
                    max_tokens=(
                        max_tokens
                        if max_tokens is not None
                        else self.config.get("max_tokens", 4096)
                    ),
                    **request_options,
                )
                break
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable_openai_error(exc):
                    raise
                delay = retry_backoff * (2**attempt)
                logger.warning(
                    "Retrying transient OpenAI-compatible request",
                    extra={"attempt": attempt + 1, "max_retries": max_retries},
                )
                if delay:
                    await asyncio.sleep(delay)
        return response.choices[0].message.content or ""

    async def _generate_openai_stream(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        *,
        structured_output: bool = False,
    ) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_options = (
            structured_output_request_options(
                provider=self.provider,
                model_name=str(self.config.get("model_name", "deepseek-chat")),
                base_url=str(self.config.get("base_url") or ""),
            )
            if structured_output
            else {}
        )
        stream = await self._client.chat.completions.create(
            model=self.config.get("model_name", "deepseek-chat"),
            messages=messages,
            temperature=temperature if temperature is not None else self.config.get("temperature", 0.7),
            max_tokens=max_tokens if max_tokens is not None else self.config.get("max_tokens", 4096),
            stream=True,
            **request_options,
        )
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                yield content

def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端实例（单例）。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
