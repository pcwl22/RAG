"""
法律文档结构化解析器
识别法律文档的章节结构和条款编号
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


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
        'bian': re.compile(r'^第[一二三四五六七八九十]+编\s+(.+)$', re.MULTILINE),
        'fenbian': re.compile(r'^第[一二三四五六七八九十]+分编\s+(.+)$', re.MULTILINE),
        'zhang': re.compile(r'^第[一二三四五六七八九十百千]+章\s+(.+)$', re.MULTILINE),
        'jie': re.compile(r'^第[一二三四五六七八九十百千]+节\s+(.+)$', re.MULTILINE),
        'tiao': re.compile(r'^第([一二三四五六七八九十百千零]+)条\s+', re.MULTILINE),
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
            matches.append(('bian', 1, m.start(), m.end(), m.group(0), m.group(1)))

        # 2. 查找"分编"
        for m in self.PATTERNS['fenbian'].finditer(text):
            matches.append(('fenbian', 2, m.start(), m.end(), m.group(0), m.group(1)))

        # 3. 查找"章"
        for m in self.PATTERNS['zhang'].finditer(text):
            matches.append(('zhang', 3, m.start(), m.end(), m.group(0), m.group(1)))

        # 4. 查找"节"
        for m in self.PATTERNS['jie'].finditer(text):
            matches.append(('jie', 4, m.start(), m.end(), m.group(0), m.group(1)))

        # 5. 查找"条"
        for m in self.PATTERNS['tiao'].finditer(text):
            matches.append(('tiao', 5, m.start(), m.end(), m.group(0).strip(), ''))

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
        if '编' in path_item and '分编' not in path_item:
            return 1
        elif '分编' in path_item:
            return 2
        elif '章' in path_item:
            return 3
        elif '节' in path_item:
            return 4
        elif '条' in path_item:
            return 5
        return 0

    def get_sections_by_level(self, level: int) -> List[LegalSection]:
        """获取指定层级的所有章节"""
        return [s for s in self.sections if s.level == level]

    def get_articles(self) -> List[LegalSection]:
        """获取所有条款（最细粒度）"""
        return self.get_sections_by_level(5)

    def format_path(self, section: LegalSection) -> str:
        """格式化章节路径"""
        return ' > '.join(section.parent_path)
