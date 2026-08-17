"""Shared request-boundary validation for model-facing text."""

from collections.abc import Iterable

from app.utils.config import get_settings


def validate_message_budget(contents: Iterable[str]) -> None:
    """Reject aggregate model input before it reaches retrieval or an LLM provider."""
    messages = list(contents)
    limits = get_settings().get("performance", {})
    total_chars = sum(len(content) for content in messages)
    max_chars = int(limits.get("max_chat_total_chars", 60000))
    if total_chars > max_chars:
        raise ValueError(f"chat input exceeds the {max_chars} character limit")

    # Provider tokenizers differ. Use the larger conservative estimate for
    # Latin text and UTF-8-heavy CJK text before making any paid model call.
    estimated_tokens = sum(
        max((len(content) + 3) // 4, (len(content.encode("utf-8")) + 2) // 3)
        for content in messages
    )
    max_tokens = int(limits.get("max_chat_input_tokens", 16000))
    if estimated_tokens > max_tokens:
        raise ValueError(f"chat input exceeds the estimated {max_tokens} token limit")
