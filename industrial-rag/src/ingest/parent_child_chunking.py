"""Parent-child chunking for RAG retrieval and generation."""
import re
from typing import Any

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。",
    "？",
    "！",
    ". ",
    "; ",
    ", ",
    " ",
    "",
]

ARTICLE_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千万零〇两0-9]+条[^\n]*")


def _chunking_config() -> dict:
    return get_settings().get("document_processing", {}).get("chunking", {})


def _parent_child_config() -> dict:
    return _chunking_config().get("parent_child", {})


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_by_separators(text: str, separators: list[str], target_size: int) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= target_size:
        return [text.strip()]
    if not separators:
        return [text[i : i + target_size].strip() for i in range(0, len(text), target_size)]

    separator = separators[0]
    if separator == "":
        return [text[i : i + target_size].strip() for i in range(0, len(text), target_size)]
    if separator not in text:
        return _split_by_separators(text, separators[1:], target_size)

    pieces: list[str] = []
    for index, split in enumerate(text.split(separator)):
        if not split.strip():
            continue
        piece = split if index == 0 else f"{separator}{split}"
        if len(piece) > target_size:
            pieces.extend(_split_by_separators(piece, separators[1:], target_size))
        else:
            pieces.append(piece.strip())
    return pieces


def _merge_chunks(splits: list[str], target_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for split in splits:
        split = split.strip()
        if not split:
            continue

        separator_len = 1 if current else 0
        if current and current_len + separator_len + len(split) > target_size:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)

            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                part_len = len(part) + (1 if overlap_parts else 0)
                if overlap_len + part_len > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += part_len

            current = overlap_parts
            current_len = sum(len(part) for part in current) + max(0, len(current) - 1)

        current.append(split)
        current_len += len(split) + separator_len

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _extract_legal_articles(text: str) -> list[str]:
    matches = list(ARTICLE_PATTERN.finditer(text))
    if len(matches) < 3:
        return []

    articles: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article = text[start:end].strip()
        if article:
            articles.append(article)
    return articles


def _split_long_article(article: str, child_size: int, child_overlap: int, separators: list[str]) -> list[str]:
    if len(article) <= child_size:
        return [article]
    splits = _split_by_separators(article, separators, child_size * 2)
    merged = _merge_chunks(splits, child_size, child_overlap)
    return merged or [article]


def _build_parent_child_from_articles(
    articles: list[str],
    parent_size: int,
    child_size: int,
    child_overlap: int,
    separators: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    parent_chunks: list[str] = []
    current_parent: list[str] = []
    current_parent_len = 0

    for article in articles:
        article_len = len(article) + (2 if current_parent else 0)
        if current_parent and current_parent_len + article_len > parent_size:
            parent_chunks.append("\n\n".join(current_parent).strip())
            current_parent = []
            current_parent_len = 0
        current_parent.append(article)
        current_parent_len += article_len

    if current_parent:
        parent_chunks.append("\n\n".join(current_parent).strip())

    for parent_index, parent_content in enumerate(parent_chunks):
        parent_articles = _extract_legal_articles(parent_content) or [parent_content]
        child_index = 0
        for article in parent_articles:
            child_parts = _split_long_article(article, child_size, child_overlap, separators)
            total_children = len(child_parts)
            for offset, child_content in enumerate(child_parts):
                results.append(
                    {
                        "parent_id": parent_index,
                        "parent_content": parent_content,
                        "child_id": f"{parent_index}_{child_index}",
                        "child_content": child_content,
                        "child_index": child_index,
                        "total_children": total_children,
                        "article_offset": offset,
                    }
                )
                child_index += 1

    logger.info(
        "Legal article chunking produced %s child chunks from %s parents",
        len(results),
        len(parent_chunks),
    )
    return results


def create_parent_child_chunks(
    text: str,
    parent_size: int = 3200,
    child_size: int = 800,
    child_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create child chunks with their parent context attached."""
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be greater than 0")
    if child_overlap >= child_size:
        child_overlap = max(0, child_size // 4)

    separators = list(separators or DEFAULT_SEPARATORS)
    if "" not in separators:
        separators.append("")

    normalized = _normalize_text(text)
    legal_articles = _extract_legal_articles(normalized)
    if legal_articles:
        return _build_parent_child_from_articles(
            legal_articles,
            parent_size=parent_size,
            child_size=child_size,
            child_overlap=child_overlap,
            separators=separators,
        )

    parent_splits = _split_by_separators(normalized, separators, parent_size * 2)
    parent_chunks = _merge_chunks(parent_splits, parent_size, overlap=0)

    if not parent_chunks and normalized:
        parent_chunks = [normalized]

    results: list[dict[str, Any]] = []
    for parent_index, parent_content in enumerate(parent_chunks):
        child_splits = _split_by_separators(parent_content, separators, child_size * 2)
        child_chunks = _merge_chunks(child_splits, child_size, child_overlap)

        if not child_chunks and parent_content:
            child_chunks = [parent_content]

        for child_index, child_content in enumerate(child_chunks):
            results.append(
                {
                    "parent_id": parent_index,
                    "parent_content": parent_content,
                    "child_id": f"{parent_index}_{child_index}",
                    "child_content": child_content,
                    "child_index": child_index,
                    "total_children": len(child_chunks),
                }
            )

    logger.info(
        "Parent-child chunking produced %s child chunks from %s parents",
        len(results),
        len(parent_chunks),
    )
    return results


def get_parent_context(child_chunks: list[dict], retrieved_child_ids: list[str]) -> list[dict]:
    """Resolve retrieved child IDs to unique parent chunks."""
    child_to_parent: dict[str, int] = {}
    parents: dict[int, dict] = {}

    for chunk in child_chunks:
        parent_id = int(chunk["parent_id"])
        child_to_parent[str(chunk["child_id"])] = parent_id
        parents.setdefault(
            parent_id,
            {"parent_id": parent_id, "content": chunk["parent_content"], "child_ids": []},
        )
        parents[parent_id]["child_ids"].append(chunk["child_id"])

    resolved_ids = {child_to_parent[cid] for cid in retrieved_child_ids if cid in child_to_parent}
    return [parents[parent_id] for parent_id in sorted(resolved_ids)]


def chunk_text_parent_child(text: str) -> dict[str, Any]:
    """Chunk text with configured parent-child sizes."""
    cfg = _parent_child_config()
    chunk_cfg = _chunking_config()
    separators = chunk_cfg.get("separators") or DEFAULT_SEPARATORS

    chunks = create_parent_child_chunks(
        text=text,
        parent_size=int(cfg.get("parent_size", 3200)),
        child_size=int(cfg.get("child_size", 800)),
        child_overlap=int(cfg.get("child_overlap", 200)),
        separators=separators,
    )

    parent_map: dict[int, str] = {}
    child_list: list[dict[str, Any]] = []
    for chunk in chunks:
        parent_id = int(chunk["parent_id"])
        parent_map.setdefault(parent_id, chunk["parent_content"])
        child_list.append(
            {
                "child_id": chunk["child_id"],
                "child_content": chunk["child_content"],
                "parent_id": parent_id,
                "parent_content": chunk["parent_content"],
                "child_index": chunk["child_index"],
                "total_children": chunk["total_children"],
            }
        )

    return {
        "parent_chunks": [{"id": pid, "content": content} for pid, content in parent_map.items()],
        "child_chunks": child_list,
    }
