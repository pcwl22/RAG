"""
法律文档结构化解析器
识别法律文档的章节结构和条款编号
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGAL_NUMBER_CHARS = "一二三四五六七八九十百千万零〇两0-9"
LEVEL_NAMES = {1: "编", 2: "分编", 3: "章", 4: "节", 5: "条"}
NO_VALUE = "无"
ARTICLE_NUMBER_PATTERN = rf"第[{LEGAL_NUMBER_CHARS}]+条(?:之[{LEGAL_NUMBER_CHARS}]+)?"

CRIMINAL_LAW_CRIME_NAMES = {
    "第一百三十三条": "交通肇事罪",
    "第一百三十三条之一": "危险驾驶罪",
    "第二百三十二条": "故意杀人罪",
    "第二百三十四条": "故意伤害罪",
    "第二百三十八条": "非法拘禁罪",
    "第二百三十九条": "绑架罪",
    "第二百六十三条": "抢劫罪",
    "第二百六十四条": "盗窃罪",
    "第二百六十六条": "诈骗罪",
    "第二百六十七条": "抢夺罪",
    "第二百七十一条": "职务侵占罪",
    "第二百七十二条": "挪用资金罪",
    "第二百七十四条": "敲诈勒索罪",
    "第二百七十五条": "故意毁坏财物罪",
    "第二百八十条": "伪造、变造、买卖国家机关公文、证件、印章罪",
    "第三百零三条": "赌博罪",
    "第三百四十七条": "走私、贩卖、运输、制造毒品罪",
    "第三百五十八条": "组织卖淫罪",
    "第三百八十二条": "贪污罪",
    "第三百八十三条": "贪污罪",
    "第三百八十五条": "受贿罪",
}

CRIME_PHRASE_NAMES = {
    "盗窃公私财物": "盗窃罪",
    "诈骗公私财物": "诈骗罪",
    "抢夺公私财物": "抢夺罪",
    "以暴力、胁迫或者其他方法抢劫公私财物": "抢劫罪",
    "敲诈勒索公私财物": "敲诈勒索罪",
    "故意伤害他人身体": "故意伤害罪",
    "故意杀人": "故意杀人罪",
    "非法拘禁他人": "非法拘禁罪",
}

HIGH_SIGNAL_LEGAL_KEYWORDS = [
    "数额较大",
    "数额巨大",
    "数额特别巨大",
    "情节严重",
    "情节特别严重",
    "多次盗窃",
    "入户盗窃",
    "携带凶器盗窃",
    "扒窃",
    "严重违反",
    "严重失职",
    "不能胜任工作",
    "经济补偿",
    "赔偿金",
    "无效或者部分无效",
    "欺诈",
    "胁迫",
    "违背真实意思",
]


@dataclass
class LegalSection:
    """法律章节结构"""
    level: int  # 层级：1=编，2=分编，3=章，4=节，5=条
    number: str  # 编号：如"第一编"、"第二百零九条"
    title: str  # 标题
    content: str  # 内容
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    parent_path: list[str]  # 父级路径


class LegalDocumentParser:
    """法律文档解析器"""

    # 正则模式
    PATTERNS = {
        'bian': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+编)\s*(.*)$', re.MULTILINE),
        'fenbian': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+分编)\s*(.*)$', re.MULTILINE),
        'zhang': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+章)\s*(.*)$', re.MULTILINE),
        'jie': re.compile(rf'^(第[{LEGAL_NUMBER_CHARS}]+节)\s*(.*)$', re.MULTILINE),
        'tiao': re.compile(rf'^({ARTICLE_NUMBER_PATTERN})\s*', re.MULTILINE),
    }

    def __init__(self) -> None:
        self.sections: list[LegalSection] = []
        self.current_path: list[str] = []

    def parse(self, text: str) -> list[LegalSection]:
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

    def _update_path(self, level: int, number: str, title: str) -> None:
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

    def get_sections_by_level(self, level: int) -> list[LegalSection]:
        """获取指定层级的所有章节"""
        return [s for s in self.sections if s.level == level]

    def get_articles(self) -> list[LegalSection]:
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


def _normalize_space(text: str) -> str:
    return re.sub(r"[\s\u3000]+", " ", text).strip()


def _short_law_name(law_name: str | None) -> str:
    if not law_name:
        return ""
    short_name = re.sub(r"^中华人民共和国", "", law_name).strip()
    return short_name or law_name


def _infer_department(law_name: str | None) -> str:
    short_name = _short_law_name(law_name)
    department_markers = [
        ("刑法", "刑法"),
        ("民法典", "民法"),
        ("民法", "民法"),
        ("劳动", "劳动法"),
        ("公司", "公司法"),
        ("行政", "行政法"),
        ("刑事诉讼", "刑事诉讼法"),
        ("民事诉讼", "民事诉讼法"),
    ]
    for marker, department in department_markers:
        if marker in short_name:
            return department
    return short_name or NO_VALUE


def _law_date_from_filename(filename: str | None) -> str:
    if not filename:
        return ""
    match = re.search(r"(\d{8}|\d{6})", Path(filename).stem)
    return match.group(1) if match else ""


def _parent_law_id(law_name: str | None, filename: str | None) -> str:
    short_name = _short_law_name(law_name)
    if not short_name and filename:
        short_name = re.sub(r"[_-]?\d{6,8}$", "", Path(filename).stem)
        short_name = re.sub(r"^中华人民共和国", "", short_name).strip()
    date = _law_date_from_filename(filename)
    return "_".join(part for part in [short_name, date] if part) or NO_VALUE


def _path_label_without_number(path_item: str) -> str:
    if not path_item:
        return NO_VALUE
    label = re.sub(rf"^{ARTICLE_NUMBER_PATTERN}\s*", "", path_item).strip()
    label = re.sub(rf"^第[{LEGAL_NUMBER_CHARS}]+(?:编|分编|章|节)\s*", "", label).strip()
    label = _normalize_space(label)
    return label or path_item


def _chinese_number_to_int(text: str) -> int | None:
    if re.fullmatch(r"\d+", text):
        return int(text)

    digit_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {"十": 10, "百": 100, "千": 1000}

    total = 0
    section_total = 0
    number = 0
    for char in text:
        if char in digit_map:
            number = digit_map[char]
        elif char in unit_map:
            if number == 0:
                number = 1
            section_total += number * unit_map[char]
            number = 0
        elif char == "万":
            if number == 0 and section_total == 0:
                section_total = 1
            total += (section_total + number) * 10000
            section_total = 0
            number = 0
        else:
            return None
    return total + section_total + number


def _article_id_suffix(article_number: str) -> str:
    match = re.search(rf"第([{LEGAL_NUMBER_CHARS}]+)条(?:之([{LEGAL_NUMBER_CHARS}]+))?", article_number)
    if not match:
        return article_number
    value = _chinese_number_to_int(match.group(1))
    if value is None:
        return article_number
    suffix = match.group(2)
    return f"{value}条之{suffix}" if suffix else f"{value}条"


def _infer_crime_name(section: LegalSection, law_name: str | None) -> str:
    if "刑法" not in (law_name or ""):
        return NO_VALUE
    if section.number in CRIMINAL_LAW_CRIME_NAMES:
        return CRIMINAL_LAW_CRIME_NAMES[section.number]
    for phrase, crime_name in CRIME_PHRASE_NAMES.items():
        if phrase in section.content:
            return crime_name
    return NO_VALUE


def _extract_article_keywords(section: LegalSection, crime_name: str) -> list[str]:
    keywords: list[str] = []

    if crime_name and crime_name != NO_VALUE:
        base_crime = crime_name.removesuffix("罪")
        for item in [base_crime, crime_name]:
            if item and item not in keywords:
                keywords.append(item)

    for keyword in HIGH_SIGNAL_LEGAL_KEYWORDS:
        if keyword in section.content and keyword not in keywords:
            keywords.append(keyword)

    return keywords[:12]


def _normalized_path_value(path: list[str], level: int) -> str:
    value = _path_value(path, level)
    return _normalize_space(value) if value else NO_VALUE


def _semantic_chunk_id(
    section: LegalSection,
    law_name: str | None,
    filename: str | None,
    path: list[str],
    crime_name: str,
) -> str:
    short_name = _short_law_name(law_name) or _parent_law_id(law_name, filename)
    parts = [
        short_name,
        _path_label_without_number(_path_value(path, 1)),
        _path_label_without_number(_path_value(path, 3)),
    ]
    section_label = _path_label_without_number(_path_value(path, 4))
    if section_label != NO_VALUE:
        parts.append(section_label)
    if crime_name != NO_VALUE:
        parts.append(crime_name)
    parts.append(_article_id_suffix(section.number))
    cleaned = [
        re.sub(r"[^\w\u4e00-\u9fff]+", "", part)
        for part in parts
        if part and part != NO_VALUE
    ]
    return "_".join(part for part in cleaned if part)


def legal_section_metadata(
    section: LegalSection,
    law_name: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Build scalar metadata for a structured legal article."""
    path = [item for item in section.parent_path if _get_level_from_path_item(item) < 5]
    legal_path = " > ".join(path)
    citation_parts = [f"《{law_name}》" if law_name else "", legal_path, section.number]
    citation = " > ".join(part for part in citation_parts if part)
    crime_name = _infer_crime_name(section, law_name)

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
        "level_1_department": _infer_department(law_name),
        "level_2_part": _normalized_path_value(path, 1),
        "level_3_chapter": _normalized_path_value(path, 3),
        "level_4_section": _normalized_path_value(path, 4),
        "level_5_article": section.number,
        "level_6_crime_name": crime_name,
        "parent_law_id": _parent_law_id(law_name, filename),
        "keywords": _extract_article_keywords(section, crime_name),
        "semantic_chunk_id": _semantic_chunk_id(section, law_name, filename, path, crime_name),
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
            "metadata": legal_section_metadata(article, law_name, filename),
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
