"""Run blocking local-model inference away from the asyncio event loop."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ParamSpec, TypeVar

from app.utils.config import get_settings

P = ParamSpec("P")
T = TypeVar("T")

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        performance = get_settings().get("performance", {})
        workers = max(1, int(performance.get("gpu_inference_workers", 1)))
        _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-inference")
    return _executor


async def run_inference(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Execute a synchronous embedding/reranking call in the bounded pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_executor(), partial(func, *args, **kwargs))


def close_inference_executor() -> None:
    """Release inference threads during application shutdown."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=True)
        _executor = None
