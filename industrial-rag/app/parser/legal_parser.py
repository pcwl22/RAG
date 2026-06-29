"""
法律文档结构化解析器
识别法律文档的章节结构和条款编号
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

LEGAL_NUMBER_CHARS = "一二三四五六七八九十百千万零〇两0-9"
LEVEL_NAMES = {1: "编", 2: "分编", 3: "章", 4: "节", 5: "条"}


@dataclass
class LegalSection:
    """法律章节结构"""
    level: int  # 层级：1=编，2=分编，3=章，4=节，5=条
    number: str  # 编号：如"第一编"、"第二百零九条"
    title: str  # 标题
    content: str  # 内容
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    parent_path: List[str]  # 父级路径


class LegalDocumentParser:
    """法律文档解析器"""

    # 正则模式
    PATTERNS = {
        'bian': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+编)\s*(.*)$', re.MULTILINE),
        'fenbian': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+分编)\s*(.*)$', re.MULTILINE),
        'zhang': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+章)\s*(.*)$', re.MULTILINE),
        'jie': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+节)\s*(.*)$', re.MULTILINE),
        'tiao': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+条)\s*', re.MULTILINE),
    }

    def __init__(self):
        self.sections: List[LegalSection] = []
        self.current_path: List[str] = []

    def parse(self, text: str) -> List[LegalSection]:
        """
        解析法律文档，提取结构化信息

        Args:
            text: 法律文档全文

        Returns:
            结构化的章节列表
        """
        self.sections = []
        self.current_path = []

        # 查找所有结构标记
        matches = []

        # 1. 查找"编"
        for m in self.PATTERNS['bian'].finditer(text):
            matches.append(('bian', 1, m.start(), m.end(), m.group(1), m.group(2).strip()))

        # 2. 查找"分编"
        for m in self.PATTERNS['fenbian'].finditer(text):
            matches.append(('fenbian', 2, m.start(), m.end(), m.group(1), m.group(2).strip()))

        # 3. 查找"章"
        for m in self.PATTERNS['zhang'].finditer(text):
            matches.append(('zhang', 3, m.start(), m.end(), m.group(1), m.group(2).strip()))

        # 4. 查找"节"
        for m in self.PATTERNS['jie'].finditer(text):
            matches.append(('jie', 4, m.start(), m.end(), m.group(1), m.group(2).strip()))

        # 5. 查找"条"
        for m in self.PATTERNS['tiao'].finditer(text):
            matches.append(('tiao', 5, m.start(), m.end(), m.group(1), ''))

        # 按位置排序
        matches.sort(key=lambda x: x[2])

        # 提取每个章节的内容
        for i, match in enumerate(matches):
            typ, level, start, end, number, title = match

            # 找到内容的结束位置（下一个同级或更高级标记的开始）
            content_end = len(text)
            for j in range(i + 1, len(matches)):
                if matches[j][1] <= level:
                    content_end = matches[j][2]
                    break

            # 提取内容
            content = text[start:content_end].strip()

            # 更新当前路径
            self._update_path(level, number, title)

            # 创建章节对象
            section = LegalSection(
                level=level,
                number=number,
                title=title,
                content=content,
                start_pos=start,
                end_pos=content_end,
                parent_path=self.current_path.copy()
            )
            self.sections.append(section)

        return self.sections

    def _update_path(self, level: int, number: str, title: str):
        """更新当前路径"""
        # 移除比当前级别更低的路径
        self.current_path = [p for p in self.current_path if self._get_level_from_path(p) < level]

        # 添加当前级别
        if title:
            self.current_path.append(f"{number} {title}")
        else:
            self.current_path.append(number)

    def _get_level_from_path(self, path_item: str) -> int:
        """从路径项获取层级"""
        return _get_level_from_path_item(path_item)

    def get_sections_by_level(self, level: int) -> List[LegalSection]:
        """获取指定层级的所有章节"""
        return [s for s in self.sections if s.level == level]

    def get_articles(self) -> List[LegalSection]:
        """获取所有条款（最细粒度）"""
        return self.get_sections_by_level(5)

    def format_path(self, section: LegalSection) -> str:
        """格式化章节路径"""
        return ' > '.join(section.parent_path)


def infer_law_name(text: str, filename: str | None = None) -> str | None:
    """Infer the law title from filename or the first lines of parsed text."""
    candidates: list[str] = []
    if filename:
        stem = Path(filename).stem
        candidates.append(re.sub(r"[_-]?\d{6,8}$", "", stem))

    candidates.extend(line.strip() for line in text.splitlines()[:20] if line.strip())
    for candidate in candidates:
        normalized = re.sub(r"\s+", "", candidate)
        match = re.search(r"(中华人民共和国[\u4e00-\u9fff]{1,30}(?:法典|法|条例|办法|规定))", normalized)
        if not match:
            match = re.fullmatch(r"([\u4e00-\u9fff]{2,30}(?:法典|法|条例|办法|规定))", normalized)
        if match:
            law_name = match.group(1)
            if not law_name.startswith("第"):
                return law_name
    return None


def _get_level_from_path_item(path_item: str) -> int:
    if re.match(rf"^第[{LEGAL_NUMBER_CHARS}]+分编", path_item):
        return 2
    if re.match(rf"^第[{LEGAL_NUMBER_CHARS}]+编", path_item):
        return 1
    if re.match(rf"^第[{LEGAL_NUMBER_CHARS}]+章", path_item):
        return 3
    if re.match(rf"^第[{LEGAL_NUMBER_CHARS}]+节", path_item):
        return 4
    if re.match(rf"^第[{LEGAL_NUMBER_CHARS}]+条", path_item):
        return 5
    return 0


def _path_value(path: list[str], level: int) -> str:
    for item in path:
        if _get_level_from_path_item(item) == level:
            return item
    return ""


def legal_section_metadata(section: LegalSection, law_name: str | None = None) -> dict[str, Any]:
    """Build scalar metadata for a structured legal article."""
    path = [item for item in section.parent_path if _get_level_from_path_item(item) < 5]
    legal_path = " > ".join(path)
    citation_parts = [f"《{law_name}》" if law_name else "", legal_path, section.number]
    citation = " > ".join(part for part in citation_parts if part)

    return {
        "document_type": "legal_article",
        "law_name": law_name or "",
        "legal_level": LEVEL_NAMES.get(section.level, ""),
        "law_book": _path_value(path, 1),
        "law_subbook": _path_value(path, 2),
        "law_chapter": _path_value(path, 3),
        "law_section": _path_value(path, 4),
        "article_number": section.number,
        "article_text": section.content,
        "legal_path": legal_path,
        "legal_citation": citation,
    }


def build_legal_article_chunks(
    text: str,
    filename: str | None = None,
) -> list[dict[str, Any]]:
    """Return one chunk per legal article with structured metadata."""
    parser = LegalDocumentParser()
    sections = parser.parse(text)
    articles = [section for section in sections if section.level == 5 and section.content.strip()]
    law_name = infer_law_name(text, filename)

    if not articles or (not law_name and len(articles) < 3):
        return []

    return [
        {
            "content": article.content,
            "metadata": legal_section_metadata(article, law_name),
        }
        for article in articles
    ]


__all__ = [
    "LegalDocumentParser",
    "LegalSection",
    "build_legal_article_chunks",
    "infer_law_name",
    "legal_section_metadata",
]
