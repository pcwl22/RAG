"""
LLM 模型接口封装

支持多 Provider：
- Claude (Anthropic API)
- DeepSeek (OpenAI-compatible API)
- 智谱 AI (GLM API)

统一接口：generate() / generate_stream()，根据配置自动选择 Provider。
"""
from collections.abc import AsyncIterator
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 全局 LLM 客户端实例
_llm_client: Any = None


def _llm_config() -> dict:
    """获取 LLM 配置。"""
    return get_settings().get("llm", {}).get("text", {})


def _get_provider() -> str:
    """获取当前配置的 Provider。"""
    cfg = _llm_config()
    return cfg.get("provider", "claude").lower()


class LLMClient:
    """LLM 客户端，根据配置选择不同的 Provider。"""

    def __init__(self):
        self.provider = _get_provider()
        self.config = _llm_config().get(self.provider, {})

        if not self.config:
            raise ValueError(
                f"LLM provider '{self.provider}' not configured in llm.text.{self.provider}"
            )

        logger.info(f"Initializing LLM client: {self.provider}")
        self._client = self._init_client()

    def _init_client(self) -> Any:
        """根据 Provider 初始化对应的客户端。"""
        if self.provider == "claude":
            return self._init_claude()
        elif self.provider in ("openai_compatible", "deepseek"):
            return self._init_openai_compatible()
        elif self.provider == "zhipu":
            return self._init_zhipu()
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

        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout", 60),
        )

    def _init_zhipu(self) -> Any:
        """初始化智谱 AI 客户端。"""
        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            raise ImportError("zhipuai package not installed. Run: pip install zhipuai") from exc

        api_key = self.config.get("api_key")
        if not api_key or api_key.startswith("${"):
            raise ValueError(
                "ZHIPU_API_KEY not set. Please set it in .env or environment."
            )

        # 智谱 AI SDK 可能是同步的，这里做个简单封装
        return ZhipuAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        if self.provider == "claude":
            return await self._generate_claude(prompt, system_prompt, temperature, max_tokens)
        elif self.provider in ("openai_compatible", "deepseek"):
            return await self._generate_openai(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "zhipu":
            return await self._generate_zhipu(prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        if self.provider == "claude":
            async for chunk in self._generate_claude_stream(
                prompt, system_prompt, temperature, max_tokens
            ):
                yield chunk
        elif self.provider in ("openai_compatible", "deepseek"):
            async for chunk in self._generate_openai_stream(
                prompt, system_prompt, temperature, max_tokens
            ):
                yield chunk
        elif self.provider == "zhipu":
            async for chunk in self._generate_zhipu_stream(
                prompt, system_prompt, temperature, max_tokens
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
            "temperature": temperature or self.config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.config.get("max_tokens", 4096),
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def _generate_claude_stream(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.config.get("model_name", "claude-sonnet-4-6"),
            "messages": messages,
            "temperature": temperature or self.config.get("temperature", 0.7),
            "max_tokens": max_tokens or self.config.get("max_tokens", 4096),
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
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.get("model_name", "deepseek-chat"),
            messages=messages,
            temperature=temperature or self.config.get("temperature", 0.7),
            max_tokens=max_tokens or self.config.get("max_tokens", 4096),
        )
        return response.choices[0].message.content or ""

    async def _generate_openai_stream(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self.config.get("model_name", "deepseek-chat"),
            messages=messages,
            temperature=temperature or self.config.get("temperature", 0.7),
            max_tokens=max_tokens or self.config.get("max_tokens", 4096),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # --------------------------------------------------------------------------
    # 智谱 AI 实现
    # --------------------------------------------------------------------------
    async def _generate_zhipu(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> str:
        """智谱 AI SDK 是同步的，这里用 asyncio.to_thread 包装。"""
        import asyncio

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            response = self._client.chat.completions.create(
                model=self.config.get("model_name", "glm-4-plus"),
                messages=messages,
                temperature=temperature or self.config.get("temperature", 0.7),
                max_tokens=max_tokens or self.config.get("max_tokens", 4096),
            )
            return response.choices[0].message.content or ""

        return await asyncio.to_thread(_call)

    async def _generate_zhipu_stream(
        self, prompt: str, system_prompt: str | None, temperature: float | None, max_tokens: int | None
    ) -> AsyncIterator[str]:
        """智谱 AI 流式输出（若 SDK 支持）。"""
        import asyncio

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _stream():
            response = self._client.chat.completions.create(
                model=self.config.get("model_name", "glm-4-plus"),
                messages=messages,
                temperature=temperature or self.config.get("temperature", 0.7),
                max_tokens=max_tokens or self.config.get("max_tokens", 4096),
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _worker() -> None:
            try:
                for chunk in _stream():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker = asyncio.create_task(asyncio.to_thread(_worker))
        completed = False
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    completed = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if completed or worker.done():
                await worker
            else:
                worker.cancel()


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端实例（单例）。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
