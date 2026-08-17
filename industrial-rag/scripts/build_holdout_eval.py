"""Build a fact-pattern hold-out evaluation set that actually exercises retrieval.

The shipped ``legal_expanded_240.jsonl`` suite cannot measure retrieval: every one
of its 200 citation-bearing questions contains, verbatim, the article number it is
scored on (``请说明《民法典》第七百五十六条规定的主要内容。`` -> ``第七百五十六条``).
A system that echoes a number back scores 1.0 without retrieving anything.

This builder produces the opposite: a concrete fact pattern with no article number
and no law name, whose ground-truth citation is the article the facts fall under.
Ground truth comes from the statute text itself -- the generator is shown one
article and asked to describe a situation governed by it.

Every generated question is then screened, and anything that leaks the answer or
merely paraphrases the statute is rejected and retried:

* the article number must not appear in any form (第一百九十一条 / 191条 / ...)
* the law name must not appear (《中华人民共和国刑法》 ...)
* the longest substring shared with the statute text must stay short, so the
  question cannot be solved by lexical overlap alone

Known limitation -- read before trusting the number this set produces:
a fact pattern can legitimately be governed by more than one article, but each
record carries a single expected citation (the article the question was written
from). A retriever that returns an equally correct neighbouring article is scored
wrong. Treat hold-out citation_recall as a **lower bound**, not a point estimate.
Bare principle articles are filtered out below because they are the worst
offenders, but the effect cannot be removed entirely with single-label ground
truth.

Usage:
    python scripts/build_holdout_eval.py --limit 12 --out eval/legal_holdout_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parser.document_parser import parse_document  # noqa: E402
from app.parser.legal_parser import build_legal_article_chunks  # noqa: E402

WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_DIR = WORKSPACE_ROOT / "date"

# Longest run of characters a question may share with its own statute text before
# we treat it as a paraphrase rather than a fact pattern.
MAX_SHARED_SUBSTRING = 10
MIN_QUESTION_CHARS = 25
MAX_QUESTION_CHARS = 220

DOMAIN_BY_LAW = {
    "中华人民共和国民法典": "civil",
    "中华人民共和国刑法": "criminal",
    "中华人民共和国劳动合同法": "labor",
}

CN_DIGITS = "零一二三四五六七八九十百千"

SYSTEM_PROMPT = (
    "你是中国法律考试的命题人。你会看到一条法律条文，"
    "请据此写出一个**事实情景题**：描述一个具体的、可能发生的情境，"
    "该情境恰好由这条条文调整。"
)

USER_TEMPLATE = """条文出处（仅供你理解，禁止写进题目）：{law_name} {article_number}
条文正文：
{article_text}

请写一个事实情景题，严格满足：
1. 只描述事实：谁做了什么、发生了什么后果。用「甲」「乙」「某公司」「张某」这类称谓。
2. 结尾提出一个法律问题，例如「应如何处理？」「该行为如何定性？」「谁承担责任？」
3. **绝对不能出现任何条号**（如「第一百九十一条」「191条」），也不能出现法律名称。
4. **不要照抄条文措辞**。用日常语言重述情境，不要搬运条文里的专业短语。
5. 全文 30-120 字，只输出题目本身，不要解释、不要引号、不要编号。
"""


@dataclass
class Article:
    law_name: str
    article_number: str
    article_text: str
    semantic_chunk_id: str
    domain: str
    source_file: str


@dataclass
class Rejection:
    reason: str
    question: str


@dataclass
class BuildStats:
    accepted: int = 0
    rejections: list[Rejection] = field(default_factory=list)
    shared_lengths: list[int] = field(default_factory=list)


def _load_env(project_root: Path) -> None:
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def is_discriminative(article_text: str) -> bool:
    """Reject bare principle articles, which no fact pattern can uniquely identify.

    劳动合同法第三条 ("订立劳动合同，应当遵循合法、公平、平等自愿...的原则") governs
    almost every contract dispute, so a fact pattern written from it is equally
    well answered by whichever specific article actually decides the case.
    """
    return not ("原则" in article_text and len(article_text) < 200)


def load_articles(source_dir: Path) -> list[Article]:
    """Parse statutes straight from source documents, so no database is required."""
    articles: list[Article] = []
    for source in sorted(source_dir.glob("*")):
        if source.suffix.lower() not in {".docx", ".txt"}:
            continue
        try:
            text = parse_document(str(source))
        except Exception:  # pragma: no cover - a malformed source must not abort the run
            continue
        for chunk in build_legal_article_chunks(text, source.name):
            metadata = chunk.get("metadata") or {}
            law_name = str(metadata.get("law_name") or "")
            domain = next(
                (value for key, value in DOMAIN_BY_LAW.items() if key in law_name), ""
            )
            article_text = str(metadata.get("article_text") or chunk.get("content") or "")
            article_number = str(metadata.get("article_number") or "")
            chunk_id = str(metadata.get("semantic_chunk_id") or "")
            if not (domain and article_number and chunk_id) or len(article_text) < 60:
                continue
            if not is_discriminative(article_text):
                continue
            articles.append(
                Article(
                    law_name=law_name,
                    article_number=article_number,
                    article_text=article_text,
                    semantic_chunk_id=chunk_id,
                    domain=domain,
                    source_file=source.name,
                )
            )
    return articles


def article_number_variants(article_number: str) -> list[str]:
    """Every spelling of an article number a leaked question might use."""
    variants = {article_number}
    core = article_number.removeprefix("第").removesuffix("条")
    if core:
        variants.update({core, f"第{core}条", f"{core}条"})
        arabic = chinese_numeral_to_int(core)
        if arabic is not None:
            variants.update({str(arabic), f"第{arabic}条", f"{arabic}条"})
    return sorted(v for v in variants if v)


def chinese_numeral_to_int(text: str) -> int | None:
    """Convert 一百九十一 -> 191. Returns None when the text is not a plain numeral."""
    text = text.split("之")[0]
    if not text or any(ch not in CN_DIGITS for ch in text):
        return None
    digits = "零一二三四五六七八九"
    total = 0
    section = 0
    current = 0
    for ch in text:
        if ch in digits:
            current = digits.index(ch)
        elif ch == "十":
            section += (current or 1) * 10
            current = 0
        elif ch == "百":
            section += (current or 1) * 100
            current = 0
        elif ch == "千":
            section += (current or 1) * 1000
            current = 0
    total += section + current
    return total or None


def longest_shared_substring(left: str, right: str) -> int:
    """Length of the longest substring common to both strings (rolling DP)."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        left_char = left[i - 1]
        for j in range(1, len(right) + 1):
            if left_char == right[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def screen_question(question: str, article: Article) -> str | None:
    """Return a rejection reason, or None when the question is usable."""
    if not question:
        return "empty"
    if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
        return f"length {len(question)} outside [{MIN_QUESTION_CHARS},{MAX_QUESTION_CHARS}]"
    for variant in article_number_variants(article.article_number):
        # Bare arabic numbers are too common to ban outright; only ban the
        # spellings that actually identify an article.
        if variant.startswith("第") or variant.endswith("条"):
            if variant in question:
                return f"leaks article number ({variant})"
    if re.search(r"第[" + CN_DIGITS + r"]+条", question):
        return "contains some article number"
    if article.law_name and article.law_name in question:
        return "leaks law name"
    for short_name in ("民法典", "刑法", "劳动合同法"):
        if short_name in question:
            return f"leaks law name ({short_name})"
    shared = longest_shared_substring(question, article.article_text)
    if shared > MAX_SHARED_SUBSTRING:
        return f"paraphrases statute (shared run of {shared} chars)"
    return None


def generate_question(client: Any, article: Article, attempts: int = 3) -> tuple[str | None, list[Rejection]]:
    rejections: list[Rejection] = []
    for attempt in range(attempts):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        law_name=article.law_name,
                        article_number=article.article_number,
                        article_text=article.article_text[:1500],
                    ),
                },
            ],
            temperature=0.9 if attempt else 0.7,
            max_tokens=400,
        )
        question = (response.choices[0].message.content or "").strip()
        question = question.strip("“”\"'` \n")
        reason = screen_question(question, article)
        if reason is None:
            return question, rejections
        rejections.append(Rejection(reason=reason, question=question))
    return None, rejections


def build(
    articles: list[Article],
    limit: int,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], BuildStats]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    )

    rng = random.Random(seed)
    by_domain: dict[str, list[Article]] = {}
    for article in articles:
        by_domain.setdefault(article.domain, []).append(article)

    selected: list[Article] = []
    per_domain = max(1, limit // max(1, len(by_domain)))
    for domain in sorted(by_domain):
        pool = sorted(by_domain[domain], key=lambda a: a.semantic_chunk_id)
        rng.shuffle(pool)
        selected.extend(pool[:per_domain])
    rng.shuffle(selected)
    selected = selected[:limit]

    stats = BuildStats()
    records: list[dict[str, Any]] = []

    def work(article: Article) -> tuple[Article, str | None, list[Rejection]]:
        question, rejections = generate_question(client, article)
        return article, question, rejections

    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        for article, question, rejections in pool_executor.map(work, selected):
            stats.rejections.extend(rejections)
            if question is None:
                continue
            stats.accepted += 1
            stats.shared_lengths.append(
                longest_shared_substring(question, article.article_text)
            )
            records.append(
                {
                    "id": f"holdout-fact-{article.semantic_chunk_id}",
                    "query": question,
                    "expected_citations": [article.article_number],
                    "expected_sources": [article.law_name],
                    "expected_answer": article.article_text,
                    "metadata": {
                        "category": "holdout_fact_pattern",
                        "domain": article.domain,
                        "article_ids": [article.semantic_chunk_id],
                        "source_file": article.source_file,
                    },
                }
            )

    records.sort(key=lambda record: record["id"])
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    _load_env(PROJECT_ROOT)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required to generate hold-out questions")

    articles = load_articles(args.source_dir)
    if not articles:
        raise SystemExit(f"no statutes parsed from {args.source_dir}")

    records, stats = build(articles, args.limit, args.seed, args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    reason_counts: dict[str, int] = {}
    for rejection in stats.rejections:
        key = re.sub(r"\(.*?\)|\d+", "", rejection.reason).strip()
        reason_counts[key] = reason_counts.get(key, 0) + 1

    shared = stats.shared_lengths
    summary = {
        "articles_available": len(articles),
        "questions_written": len(records),
        "rejected_drafts": len(stats.rejections),
        "rejection_reasons": reason_counts,
        "shared_substring_chars": {
            "max": max(shared) if shared else 0,
            "mean": round(sum(shared) / len(shared), 2) if shared else 0,
            "threshold": MAX_SHARED_SUBSTRING,
        },
        "output": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
