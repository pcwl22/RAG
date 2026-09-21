"""Answer generation for retrieved RAG contexts."""
import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

from app.llm.answer_contract import (
    AnswerContractError,
    ValidatedAnswer,
    contract_context_id,
    recover_grounded_answer,
    serialize_validated_answer,
    validate_answer,
)
from app.llm.model import get_llm_client
from app.llm.prompt import (
    ANSWER_FORMAT_INSTRUCTIONS,
    STREAM_ANSWER_INSTRUCTIONS,
    build_rag_prompt,
    build_system_prompt,
    get_plain_output_instructions,
)
from app.retrieval.hybrid import explicit_legal_citations
from app.retrieval.result_merge import result_score
from app.utils.cache import SemanticCache
from app.utils.config import get_config_section, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

KNOWN_LAW_TITLES = [
    "民法典",
    "刑法",
    "劳动合同法",
    "公司法",
    "行政法",
    "行政处罚法",
    "食品安全法",
    "道路交通安全法",
    "农村土地承包法",
    "治安管理处罚法",
    "仲裁法",
    "民事诉讼法",
]


def _generation_config() -> dict[str, Any]:
    return get_config_section("rag", "generation")


def _doc_debug_summary(docs: list[dict], max_items: int = 5) -> list[dict]:
    """Return correlation-safe document diagnostics without legal text or names."""
    summary: list[dict] = []
    for doc in docs[:max_items]:
        metadata = doc.get("metadata") or {}
        content = (
            doc.get("content")
            or metadata.get("article_text")
            or metadata.get("parent_content")
            or doc.get("child_content", "")
        )
        content_text = str(content)
        document_id = str(doc.get("id") or "")
        source_name = str(metadata.get("filename") or metadata.get("document_id") or "")
        summary.append(
            {
                "document_id_sha256": hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16],
                "source_sha256": hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16],
                "score": round(float(doc.get("score", 0.0) or 0.0), 6),
                "rrf_score": round(float(doc.get("rrf_score", 0.0) or 0.0), 6),
                "chunk_index": metadata.get("chunk_index", doc.get("chunk_index")),
                "content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest()[:16],
                "content_length": len(content_text),
            }
        )
    return summary


def _doc_source(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    source = str(metadata.get("filename") or metadata.get("document_id") or "unknown")
    return _clean_source_name(source)


def _clean_source_name(source: str) -> str:
    return re.sub(r"^[0-9a-fA-F-]{32,}_", "", source)


def _doc_location(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    content = _doc_snippet(doc)
    parts: list[str] = []

    legal_citation = metadata.get("legal_citation")
    if legal_citation:
        parts.append("定位：" + str(legal_citation))
    else:
        legal_path = metadata.get("legal_path")
        law_name = metadata.get("law_name")
        if law_name or legal_path:
            path = " > ".join(str(item) for item in [law_name, legal_path] if item)
            parts.append("定位：" + path)

    section_numbers = metadata.get("section_numbers")
    if isinstance(section_numbers, list) and section_numbers:
        parts.append("章节：" + "、".join(str(item) for item in section_numbers[:3]))

    title_paths = metadata.get("title_paths")
    if isinstance(title_paths, list) and title_paths:
        title_path = title_paths[0]
        if isinstance(title_path, list):
            parts.append("路径：" + " > ".join(str(item) for item in title_path))
        else:
            parts.append("路径：" + str(title_path))

    article_number = metadata.get("article_number")
    if article_number:
        parts.append("条款：" + str(article_number))
    else:
        article_match = re.search(r"第[一二三四五六七八九十百千万零〇两0-9]+条", content)
        if article_match:
            parts.append("条款：" + article_match.group(0))

    page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_index")
    if page is not None:
        parts.append(f"页码：{page}")

    location = "；".join(parts) if parts else "未提供章节或页码 metadata"
    return f"{location}；\n内容：{content}"


def _doc_snippet(doc: dict, max_len: int = 180) -> str:
    metadata = doc.get("metadata") or {}
    content = str(
        doc.get("child_content")
        or doc.get("content")
        or metadata.get("article_text")
        or metadata.get("parent_content")
        or ""
    )
    content = re.sub(r"\s+", " ", content).strip()
    if len(content) <= max_len:
        return content
    return content[:max_len].rstrip() + "..."


def _doc_context_content(doc: dict, max_len: int) -> str:
    """Return the most precise content slice to include in the LLM prompt."""
    metadata = doc.get("metadata") or {}
    content = str(
        doc.get("child_content")
        or metadata.get("child_content")
        or doc.get("content")
        or metadata.get("article_text")
        or metadata.get("parent_content")
        or ""
    )
    content = re.sub(r"\s+", " ", content).strip()
    if max_len <= 0:
        return ""
    if len(content) <= max_len:
        return content
    if max_len <= 3:
        return content[:max_len]
    return content[: max_len - 3].rstrip() + "..."


def _article_numbers(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", text)


def _chinese_ngrams(text: str, min_size: int = 2, max_size: int = 6) -> list[str]:
    phrases = re.findall(rf"[\u4e00-\u9fff]{{{min_size},}}", text)
    grams: list[str] = []
    for phrase in phrases:
        upper = min(max_size, len(phrase))
        for size in range(upper, min_size - 1, -1):
            for index in range(0, len(phrase) - size + 1):
                grams.append(phrase[index : index + size])
    return grams


def _evidence_terms(query: str | None, answer: str | None) -> list[str]:
    text = "\n".join(part for part in [query, answer] if part)
    if not text:
        return []

    stop_terms = {
        "回答",
        "结论",
        "依据",
        "根据",
        "当前",
        "知识库",
        "问题",
        "情形",
        "如何",
        "可以",
        "直接",
        "需要",
        "如果",
        "则需",
        "或者",
        "以及",
        "中华人民共和国",
        "劳动者",
        "用人单位",
        "刑法",
        "民法典",
        "劳动合同法",
    }

    terms = _article_numbers(text)
    suffix_terms = re.findall(r"[\u4e00-\u9fff]{2,12}(?:罪|合同|工作|岗位|药品|毒品|电子烟)", text)
    terms.extend(suffix_terms)
    for term in suffix_terms:
        terms.extend(_chinese_ngrams(term, min_size=2, max_size=4))
    terms.extend(_chinese_ngrams(text))

    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        term = term.strip()
        if len(term) < 2 or term in stop_terms:
            continue
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique[:80]


def _filter_basis_docs(context_docs: list[dict], query: str | None, answer: str | None) -> list[dict]:
    if not context_docs:
        return []

    priority_docs = [
        doc for doc in context_docs if int(doc.get("mapped_article_priority") or 0) > 0
    ]
    priority_docs.sort(key=lambda doc: int(doc.get("mapped_article_priority") or 0), reverse=True)

    articles = _article_numbers("\n".join(part for part in [query, answer] if part))
    if articles:
        article_matches = [
            doc
            for doc in context_docs
            if any(article in _doc_context_content(doc, 800) for article in articles)
        ]
        if article_matches:
            return _prepend_priority_docs(priority_docs, article_matches)

    terms = _evidence_terms(query, answer)
    if not terms:
        return context_docs

    scored_docs: list[tuple[int, dict]] = []
    for doc in context_docs:
        searchable = _doc_context_content(doc, 800)
        score = sum(1 for term in terms if term in searchable)
        if score:
            scored_docs.append((score, doc))

    if not scored_docs:
        return _prepend_priority_docs(priority_docs, context_docs)

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    best_score = scored_docs[0][0]
    minimum_score = best_score if best_score <= 2 else best_score - 1
    return _prepend_priority_docs(
        priority_docs,
        [doc for score, doc in scored_docs if score >= minimum_score],
    )


def _prepend_priority_docs(priority_docs: list[dict], docs: list[dict]) -> list[dict]:
    ordered: list[dict] = []
    seen: set[str] = set()
    for doc in [*priority_docs, *docs]:
        key = str(doc.get("id") or id(doc))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(doc)
    return ordered


def _requested_laws(query: str) -> list[str]:
    return [title for title in KNOWN_LAW_TITLES if title in query]


def missing_requested_laws(query: str, docs: list[dict]) -> list[str]:
    """Return explicitly requested law titles that are absent from retrieved docs."""
    requested = _requested_laws(query)
    if not requested:
        return []

    source_text = "\n".join(
        " ".join(
            str(part)
            for part in [
                _doc_source(doc),
                (doc.get("metadata") or {}).get("law_name"),
                (doc.get("metadata") or {}).get("legal_citation"),
            ]
            if part
        )
        for doc in docs
    )
    missing: list[str] = []
    for title in requested:
        if title not in source_text:
            missing.append(title)
    return missing


def _normalized_law_name(value: object) -> str:
    return re.sub(r"[《》\s]", "", str(value or ""))


def missing_requested_citations(
    query: str,
    docs: list[dict],
) -> list[tuple[str, str]]:
    """Return named law/article pairs absent from the final answer context."""
    requested = explicit_legal_citations(query)
    if not requested:
        return []

    available: list[tuple[str, str]] = []
    for doc in docs:
        metadata = doc.get("metadata") or {}
        law_name = _normalized_law_name(metadata.get("law_name"))
        article = str(metadata.get("article_number") or "").strip()
        if law_name and article:
            available.append((law_name, article))

    missing: list[tuple[str, str]] = []
    for law_name, article in requested:
        normalized_law = _normalized_law_name(law_name)
        if not any(
            candidate_article == article
            and (
                candidate_law == normalized_law
                or candidate_law.endswith(normalized_law)
                or normalized_law.endswith(candidate_law)
            )
            for candidate_law, candidate_article in available
        ):
            missing.append((law_name, article))
    return missing


def format_missing_law_answer(missing_laws: list[str]) -> str:
    """Build a standard answer when the requested law is absent from context."""
    law_names = "、".join(f"《{law}》" for law in missing_laws)
    return f"""【回答】
结论：
当前知识库未检索到{law_names}相关文档，无法基于当前知识库可靠回答该问题。请先补充对应法律文件后再检索。

依据：
无匹配的请求法律文档。"""


def format_missing_citation_answer(missing: list[tuple[str, str]]) -> str:
    """Fail closed when any explicitly named provision is not in final context."""
    citations = "、".join(f"《{law_name}》{article}" for law_name, article in missing)
    return f"""【回答】
结论：
当前知识库未完整检索到用户明确指定的{citations}，为避免遗漏法条，不生成不完整结论。请补充对应条文或扩大检索后重试。

依据：
明确指定的法条未被完整覆盖。"""


def format_no_context_answer() -> str:
    """Return a deterministic refusal when retrieval produced no evidence."""
    return """【回答】
结论：
当前知识库没有找到足够相关的信息，无法基于当前知识库可靠回答该问题。请补充相关资料或调整问题后重试。

依据：
无直接支持结论的命中文档。"""


def _contract_retry_guidance(exc: Exception) -> str:
    """Return bounded, error-specific repair instructions without echoing model text."""

    reason = str(exc)
    common = (
        "上一次输出未通过服务端校验。只重新输出一个合法 JSON 对象，且顶层只能包含 "
        "conclusion、evidence、insufficient_context。不要重复上一次输出，也不要添加解释。"
    )
    if "every conclusion claim must have its own evidence" in reason:
        repair = (
            "把 conclusion 改成仅包含 evidence 已直接覆盖的一至三个短分句；删除其余分句。"
            "每个 evidence.claim 必须逐字等于 conclusion 中对应的完整分句。"
        )
    elif "claim is not grounded in its cited evidence" in reason:
        repair = (
            "用对应 quote 中的关键原词重写 claim 和 conclusion 分句，避免抽象推论、建议、"
            "风险或 quote 未出现的法律后果；claim 与分句必须完全相同。"
        )
    elif "insufficient-context answer must not include evidence" in reason:
        repair = (
            "资料不足模式必须使用 evidence: []，不得保留任何引用；conclusion 只说明资料不足。"
        )
    elif "response fields do not match the answer contract" in reason:
        repair = (
            "删除所有额外字段；每个 evidence 项也只能包含 claim、context_id、quote 三个字段。"
        )
    else:
        repair = (
            "修正所有不存在的 context_id，从对应上下文逐字复制连续 quote，并删除任何未被"
            "直接支持的结论分句。"
        )
    return common + repair


def _has_no_supported_conclusion(answer: str) -> bool:
    """Detect answers that explicitly decline due to missing or irrelevant context."""
    normalized = re.sub(r"\s+", "", answer)
    strong_markers = [
        "无法基于当前知识库回答",
        "无法基于当前知识库可靠回答",
        "无法基于提供的上下文回答",
        "无法根据提供的上下文回答",
        "无法回答该问题",
        "没有找到足够相关的信息",
        "未找到足够相关的信息",
    ]
    if any(marker in normalized for marker in strong_markers):
        return True

    supported_conclusion_markers = [
        "应首先",
        "应当向",
        "通常",
        "倾向",
        "基础规则",
        "可以确定",
    ]
    if any(marker in normalized for marker in supported_conclusion_markers) or _article_numbers(answer):
        return False

    return (
        ("检索到的上下文" in normalized or "上下文" in normalized)
        and ("未涉及" in normalized or "不包含" in normalized or "不足以回答" in normalized)
        and ("需要补充" in normalized or "无法" in normalized)
    )


class Generator:
    """Build RAG prompts and call the configured LLM."""

    def __init__(self) -> None:
        self.config = _generation_config()
        self.llm = get_llm_client()
        self.cache = SemanticCache()

    def _answer_contract_enabled(self) -> bool:
        contract = self.config.get("answer_contract", {})
        return isinstance(contract, dict) and bool(contract.get("enabled", False))

    def _context_template(self) -> str:
        return str(
            self.config.get(
                "context_template",
                "## 相关文档 {index}\n来源: {source}\n分数: {score:.4f}\n\n{content}\n",
            )
        )

    def _render_context_entry(
        self,
        index: int,
        context_id: str,
        doc: dict,
        content: str,
    ) -> str:
        return (
            f"[context_id={context_id}]\n"
            + self._context_template().format(
                index=index,
                source=_doc_source(doc),
                score=result_score(doc),
                content=content,
            )
        )

    def _context_entries(self, docs: list[dict]) -> list[tuple[str, dict, str]]:
        """Return the exact globally bounded passages shown to the model."""
        max_per_doc = max(300, int(self.config.get("max_context_per_doc", 1200)))
        max_total = max(0, int(self.config.get("max_context_length", 8000)))
        entries: list[tuple[str, dict, str]] = []
        rendered_length = 0
        for doc in docs:
            index = len(entries) + 1
            context_id = contract_context_id(index)
            separator_length = 1 if entries else 0
            available = max_total - rendered_length - separator_length
            if available <= 0:
                break
            content = _doc_context_content(doc, max_per_doc)
            if not content:
                continue
            rendered = self._render_context_entry(index, context_id, doc, content)
            # Entries are atomic at the global boundary. A partially included
            # legal provision would make "citation present" mean something
            # weaker than the text actually required for analysis.
            if len(rendered) > available:
                break
            entries.append((context_id, doc, content))
            rendered_length += separator_length + len(rendered)
        return entries

    def _context_map(self, docs: list[dict]) -> dict[str, str]:
        return {context_id: content for context_id, _doc, content in self._context_entries(docs)}

    def context_docs_for_generation(self, docs: list[dict]) -> list[dict]:
        """Return only documents represented in the final bounded prompt."""
        return [doc for _context_id, doc, _content in self._context_entries(docs)]

    def _build_context(self, docs: list[dict]) -> str:
        if not docs:
            return ""

        contexts: list[str] = []
        for index, (context_id, doc, content) in enumerate(self._context_entries(docs), 1):
            contexts.append(self._render_context_entry(index, context_id, doc, content))
        return "\n".join(contexts)

    def build_context(self, docs: list[dict]) -> str:
        """Build the same bounded context representation used for generation.

        Enhanced-query aggregation needs to ground its second LLM call in the
        exact representation used by the normal generator. Keep this small
        public wrapper instead of coupling that service to a private method.
        """
        return self._build_context(docs)

    def _build_matched_files(self, docs: list[dict]) -> list[str]:
        matched_files: list[str] = []
        for doc in docs:
            source = _doc_source(doc)
            if source and source not in matched_files:
                matched_files.append(source)
        return matched_files

    def _build_file_locations(self, docs: list[dict]) -> list[str]:
        locations: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            source = _doc_source(doc)
            location = _doc_location(doc)
            item = f"{source}：{location}"
            if item not in seen:
                seen.add(item)
                locations.append(item)
        return locations

    def format_basis_section(
        self,
        context_docs: list[dict],
        answer: str | None = None,
        query: str | None = None,
    ) -> str:
        """Format the citation/evidence section appended to answers."""
        if answer and _has_no_supported_conclusion(answer):
            return "\n\n依据：\n无直接支持结论的命中文档。"

        basis_docs = _filter_basis_docs(context_docs, query, answer)
        file_locations = self._build_file_locations(basis_docs)
        if file_locations:
            basis_text = "\n".join(f"{index}. {location}" for index, location in enumerate(file_locations, 1))
        else:
            basis_text = "无"

        return f"\n\n依据：\n{basis_text}"

    def format_verified_answer(
        self,
        answer: ValidatedAnswer,
        context_docs: list[dict],
        query: str | None = None,
    ) -> str:
        """Render only server-resolved locations for validated evidence."""
        if answer.insufficient_context or not answer.evidence:
            return format_no_context_answer()

        entries = {
            context_id: doc
            for context_id, doc, _content in self._context_entries(context_docs)
        }
        basis_lines: list[str] = []
        seen: set[str] = set()
        for evidence in answer.evidence:
            doc = entries.get(evidence.context_id)
            if doc is None:
                # This should be unreachable because validation uses the same
                # context map. Fail closed if a caller supplies mismatched docs.
                continue
            source = _doc_source(doc)
            location = _doc_location(doc)
            line = f"{source}：{location}；核验原文：{evidence.quote}"
            if line not in seen:
                seen.add(line)
                basis_lines.append(f"{len(basis_lines) + 1}. {line}")

        if not basis_lines:
            return format_no_context_answer()
        return f"【回答】\n结论：\n{answer.conclusion}\n\n依据：\n" + "\n".join(basis_lines)

    def validate_contract_answer(
        self,
        raw_answer: str,
        context_docs: list[dict],
        query: str | None = None,
    ) -> str:
        """Validate a structured answer and render its server-side evidence."""
        try:
            validated = validate_answer(raw_answer, self._context_map(context_docs), query or "")
        except AnswerContractError:
            logger.warning("Structured answer failed grounding validation")
            return format_no_context_answer()
        return self.format_verified_answer(validated, context_docs, query)

    def parse_contract_answer(
        self,
        raw_answer: str,
        context_docs: list[dict],
        query: str = "",
    ) -> ValidatedAnswer:
        """Expose strict parsing for the enhanced aggregation pipeline."""
        return validate_answer(raw_answer, self._context_map(context_docs), query)

    def recover_contract_answer(
        self,
        raw_answer: str,
        context_docs: list[dict],
        query: str = "",
    ) -> ValidatedAnswer:
        """Recover only independently grounded claims after bounded retries."""
        return recover_grounded_answer(raw_answer, self._context_map(context_docs), query)

    async def _generate_contract_answer(
        self,
        query: str,
        context_docs: list[dict],
        *,
        stream: bool = False,
    ) -> tuple[ValidatedAnswer | None, str | None]:
        """Generate, validate, and retry a structured answer at most once."""
        prompt = self._build_prompt(query, self._build_context(context_docs), stream=stream)
        retries = max(0, min(int(self.config.get("answer_contract", {}).get("max_retries", 1)), 1))
        raw_answer: str | None = None
        for attempt in range(retries + 1):
            try:
                if stream:
                    chunks = [
                        chunk
                        async for chunk in self.llm.generate_stream(
                            prompt=prompt,
                            system_prompt=self._get_system_prompt(),
                            temperature=0,
                            max_tokens=int(self.config.get("max_tokens", 2048)),
                            structured_output=True,
                        )
                    ]
                    raw_answer = "".join(chunks)
                else:
                    raw_answer = await self.llm.generate(
                        prompt=prompt,
                        system_prompt=self._get_system_prompt(),
                        temperature=0,
                        max_tokens=int(self.config.get("max_tokens", 2048)),
                        structured_output=True,
                    )
                validated = validate_answer(
                    raw_answer,
                    self._context_map(context_docs),
                    query,
                )
                return validated, raw_answer
            except (AnswerContractError, TypeError, ValueError) as exc:
                if attempt >= retries:
                    try:
                        recovered = recover_grounded_answer(
                            raw_answer or "",
                            self._context_map(context_docs),
                            query,
                        )
                    except (AnswerContractError, TypeError, ValueError):
                        logger.warning(
                            "Structured answer rejected after validation",
                            extra={"reason": str(exc)},
                        )
                        break
                    logger.info(
                        "Recovered independently grounded claims from structured answer",
                        extra={"claim_count": len(recovered.evidence)},
                    )
                    return recovered, serialize_validated_answer(recovered)
                prompt = prompt + "\n\n" + _contract_retry_guidance(exc)
            except Exception as exc:
                logger.warning(
                    "Structured answer generation failed",
                    extra={"error_type": type(exc).__name__},
                )
                break
        return None, None

    @staticmethod
    def _chunk_answer(answer: str, chunk_size: int = 256) -> list[str]:
        return [answer[index : index + chunk_size] for index in range(0, len(answer), chunk_size)]

    def format_answer(self, answer: str, context_docs: list[dict], query: str | None = None) -> str:
        """Normalize an LLM answer and append the standard evidence section."""
        answer = answer.strip()
        answer_match = re.search(r"(?:##\s*)?【回答】", answer)
        if answer_match:
            answer_body = answer[answer_match.end():]
            answer_body = re.split(
                r"\n\s*---\s*\n|\n\s*依据[:：]|\n(?:##\s*)?【匹配文件】|\n(?:##\s*)?【文件具体位置（文件章节页码）】|\n(?:##\s*)?【文件具体位置和内容（文件章节页码）】|\n(?:##\s*)?【文件具体位置和内容】",
                answer_body,
                maxsplit=1,
            )[0]
        else:
            answer_body = answer
        answer_body = re.sub(r"^\s*结论[:：]\s*", "", answer_body.strip())
        answer_body = re.sub(r"^根据(提供的)?上下文[，,]?\s*", "", answer_body.strip())
        answer_body = re.sub(r"\n+\s*来源：.*", "", answer_body).strip()

        return f"""【回答】
结论：
{answer_body}{self.format_basis_section(context_docs, answer_body, query)}"""

    def _build_prompt(self, query: str, context: str, stream: bool = False) -> str:
        if self._answer_contract_enabled():
            output_instructions = (
                STREAM_ANSWER_INSTRUCTIONS if stream else ANSWER_FORMAT_INSTRUCTIONS
            )
        else:
            output_instructions = get_plain_output_instructions(stream=stream)
        return build_rag_prompt(query, context, output_instructions)

    def _get_system_prompt(self) -> str:
        return build_system_prompt(self.config.get("system_prompt"))

    def _cache_context(self, docs: list[dict]) -> dict:
        """Describe everything that can materially change a generated answer."""
        settings = get_settings()
        llm_config = settings.get("llm", {}).get("text", {})
        provider = llm_config.get("provider", "local")
        provider_config = llm_config.get(provider, llm_config)
        context_docs = []
        for context_id, doc, content in self._context_entries(docs):
            metadata = doc.get("metadata") or {}
            context_docs.append(
                {
                    "context_id": context_id,
                    "id": str(doc.get("id") or metadata.get("chunk_id") or ""),
                    "partition": str(doc.get("partition") or metadata.get("partition") or ""),
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
        rendered_context = self._build_context(docs)
        prompt_material = "\n".join(
            [
                self._get_system_prompt(),
                self._build_prompt("__CACHE_QUERY__", "__CACHE_CONTEXT__"),
                str(self.config),
            ]
        )
        return {
            "provider": provider,
            "model": provider_config.get("model_name"),
            "prompt_sha256": hashlib.sha256(prompt_material.encode("utf-8")).hexdigest(),
            "rendered_context_sha256": hashlib.sha256(
                rendered_context.encode("utf-8")
            ).hexdigest(),
            "documents": context_docs,
        }

    async def generate(self, query: str, context_docs: list[dict], use_cache: bool = True) -> str:
        prompt_docs = self.context_docs_for_generation(context_docs)
        missing_citations = missing_requested_citations(query, prompt_docs)
        if missing_citations:
            return format_missing_citation_answer(missing_citations)
        missing_laws = missing_requested_laws(query, prompt_docs)
        if missing_laws:
            return format_missing_law_answer(missing_laws)

        if self._answer_contract_enabled() and not prompt_docs:
            return format_no_context_answer()

        if use_cache:
            try:
                cache_context = self._cache_context(context_docs)
                cached = await self.cache.get(query, cache_context)
                if cached:
                    if self._answer_contract_enabled():
                        try:
                            cached_validated = validate_answer(
                                cached,
                                self._context_map(context_docs),
                                query,
                            )
                            return self.format_verified_answer(cached_validated, context_docs, query)
                        except AnswerContractError:
                            logger.info("Ignoring an ungrounded semantic-cache entry")
                            cached = None
                    if not cached:
                        raise LookupError("ungrounded cache entry")
                    return self.format_answer(cached, context_docs, query)
            except Exception as exc:
                logger.warning("Semantic cache lookup skipped: %s", type(exc).__name__)
                cache_context = None
        else:
            cache_context = None

        logger.info(
            "Generating answer with retrieved contexts",
            extra={
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "context_doc_count": len(context_docs),
                "context_docs": _doc_debug_summary(context_docs),
            },
        )

        if self._answer_contract_enabled():
            validated, raw_answer = await self._generate_contract_answer(query, context_docs)
            if validated is None or raw_answer is None:
                return format_no_context_answer()
            answer = self.format_verified_answer(validated, context_docs, query)
            if use_cache and cache_context is not None:
                try:
                    await self.cache.set(query, cache_context, raw_answer)
                except Exception as exc:
                    logger.warning("Semantic cache write skipped: %s", type(exc).__name__)
            return answer

        prompt = self._build_prompt(query, self._build_context(context_docs))
        raw_answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=float(self.config.get("temperature", 0.3)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        )
        answer = self.format_answer(raw_answer, context_docs, query)

        if use_cache and cache_context is not None:
            try:
                await self.cache.set(query, cache_context, raw_answer)
            except Exception as exc:
                logger.warning("Semantic cache write skipped: %s", exc)
        return answer

    async def generate_stream(self, query: str, context_docs: list[dict]) -> AsyncIterator[str]:
        prompt_docs = self.context_docs_for_generation(context_docs)
        missing_citations = missing_requested_citations(query, prompt_docs)
        if missing_citations:
            yield format_missing_citation_answer(missing_citations)
            return
        missing_laws = missing_requested_laws(query, prompt_docs)
        if missing_laws:
            yield format_missing_law_answer(missing_laws)
            return

        if self._answer_contract_enabled() and not prompt_docs:
            yield format_no_context_answer()
            return

        if self._answer_contract_enabled():
            validated, _raw_answer = await self._generate_contract_answer(
                query, context_docs, stream=True
            )
            if validated is None:
                rendered = format_no_context_answer()
            else:
                rendered = self.format_verified_answer(validated, context_docs, query)
            for chunk in self._chunk_answer(rendered):
                yield chunk
            return

        prompt = self._build_prompt(query, self._build_context(context_docs), stream=True)
        streamed_chunks: list[str] = []
        yield "【回答】\n结论：\n"
        async for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=float(self.config.get("temperature", 0.3)),
            max_tokens=int(self.config.get("max_tokens", 2048)),
        ):
            streamed_chunks.append(chunk)
            yield chunk
        yield self.format_basis_section(context_docs, "".join(streamed_chunks), query)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = "\n\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        )
        return await self.llm.generate(
            prompt=f"{prompt}\n\nAssistant:",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        prompt = "\n\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        )
        async for chunk in self.llm.generate_stream(
            prompt=f"{prompt}\n\nAssistant:",
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
