"""
Parent-Child分块器
使用结构化解析生成父子分块，并添加标题路径增强
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import tiktoken

from .legal_parser import LegalDocumentParser, LegalSection


@dataclass
class Chunk:
    """文档块"""
    id: str
    content: str
    metadata: Dict[str, Any]
    is_parent: bool
    parent_id: str | None = None


class ParentChildChunker:
    """Parent-Child分块器"""

    def __init__(
        self,
        parent_min_tokens: int = 1500,
        parent_max_tokens: int = 3000,
        child_min_tokens: int = 300,
        child_max_tokens: int = 800,
        model_name: str = "cl100k_base",
    ):
        """
        Args:
            parent_min_tokens: 父块最小token数
            parent_max_tokens: 父块最大token数
            child_min_tokens: 子块最小token数
            child_max_tokens: 子块最大token数
            model_name: tokenizer模型名称
        """
        self.parent_min_tokens = parent_min_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.parser = LegalDocumentParser()

    def count_tokens(self, text: str) -> int:
        """计算文本token数"""
        return len(self.tokenizer.encode(text))

    def chunk_document(
        self,
        text: str,
        document_id: str,
        filename: str,
        partition: str = "general",
    ) -> tuple[List[Chunk], List[Chunk]]:
        """
        对文档进行Parent-Child分块

        Args:
            text: 文档全文
            document_id: 文档ID
            filename: 文件名
            partition: 分区名

        Returns:
            (parent_chunks, child_chunks)
        """
        # 1. 结构化解析
        sections = self.parser.parse(text)

        if not sections:
            # 如果没有结构标记，回退到简单分块
            return self._fallback_chunking(text, document_id, filename, partition)

        # 2. 生成父块
        parent_chunks = self._create_parent_chunks(sections, document_id, filename, partition)

        # 3. 生成子块
        child_chunks = self._create_child_chunks(parent_chunks, document_id, filename, partition)

        return parent_chunks, child_chunks

    def _create_parent_chunks(
        self,
        sections: List[LegalSection],
        document_id: str,
        filename: str,
        partition: str,
    ) -> List[Chunk]:
        """创建父块：按结构层级聚合"""
        parent_chunks = []
        current_group = []
        current_tokens = 0
        chunk_index = 0

        for section in sections:
            section_tokens = self.count_tokens(section.content)

            # 如果当前组已经超过最小值，且加入新section会超过最大值，则切分
            if (
                current_tokens >= self.parent_min_tokens
                and current_tokens + section_tokens > self.parent_max_tokens
            ):
                # 保存当前组为父块
                parent_chunks.append(
                    self._build_parent_chunk(
                        current_group, document_id, filename, partition, chunk_index
                    )
                )
                chunk_index += 1
                current_group = []
                current_tokens = 0

            # 加入当前section
            current_group.append(section)
            current_tokens += section_tokens

        # 处理最后一组
        if current_group:
            parent_chunks.append(
                self._build_parent_chunk(
                    current_group, document_id, filename, partition, chunk_index
                )
            )

        return parent_chunks

    def _build_parent_chunk(
        self,
        sections: List[LegalSection],
        document_id: str,
        filename: str,
        partition: str,
        chunk_index: int,
    ) -> Chunk:
        """构建父块"""
        content = "\n\n".join(s.content for s in sections)

        # 提取标题路径
        title_paths = []
        for s in sections:
            path = self.parser.format_path(s)
            if path and path not in title_paths:
                title_paths.append(path)

        parent_id = f"{document_id}_parent_{chunk_index}"

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "partition": partition,
            "chunk_index": chunk_index,
            "chunk_type": "parent",
            "title_paths": title_paths,
            "section_levels": [s.level for s in sections],
            "section_numbers": [s.number for s in sections],
        }

        return Chunk(
            id=parent_id,
            content=content,
            metadata=metadata,
            is_parent=True,
            parent_id=None,
        )

    def _create_child_chunks(
        self,
        parent_chunks: List[Chunk],
        document_id: str,
        filename: str,
        partition: str,
    ) -> List[Chunk]:
        """从父块中提取子块"""
        child_chunks = []
        child_index = 0

        for parent in parent_chunks:
            parent_content = parent.content
            parent_tokens = self.count_tokens(parent_content)

            # 如果父块本身就很小，不再分割
            if parent_tokens <= self.child_max_tokens:
                child_chunks.append(
                    self._build_child_chunk(
                        parent_content,
                        parent,
                        document_id,
                        filename,
                        partition,
                        child_index,
                        0,
                    )
                )
                child_index += 1
                continue

            # 按段落分割子块
            paragraphs = parent_content.split("\n\n")
            current_child = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self.count_tokens(para)

                # 如果当前子块已经足够大，且加入新段落会超过最大值，则切分
                if (
                    current_tokens >= self.child_min_tokens
                    and current_tokens + para_tokens > self.child_max_tokens
                ):
                    child_content = "\n\n".join(current_child)
                    child_chunks.append(
                        self._build_child_chunk(
                            child_content,
                            parent,
                            document_id,
                            filename,
                            partition,
                            child_index,
                            len(child_chunks),
                        )
                    )
                    child_index += 1
                    current_child = []
                    current_tokens = 0

                current_child.append(para)
                current_tokens += para_tokens

            # 处理最后一个子块
            if current_child:
                child_content = "\n\n".join(current_child)
                child_chunks.append(
                    self._build_child_chunk(
                        child_content,
                        parent,
                        document_id,
                        filename,
                        partition,
                        child_index,
                        len(child_chunks),
                    )
                )
                child_index += 1

        return child_chunks

    def _build_child_chunk(
        self,
        content: str,
        parent: Chunk,
        document_id: str,
        filename: str,
        partition: str,
        child_index: int,
        child_offset: int,
    ) -> Chunk:
        """构建子块"""
        child_id = f"{document_id}_child_{child_index}"

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "partition": partition,
            "chunk_index": child_index,
            "chunk_type": "child",
            "parent_id": parent.id,
            "parent_content": parent.content,
            "title_paths": parent.metadata.get("title_paths", []),
            "child_offset": child_offset,
        }

        return Chunk(
            id=child_id,
            content=content,
            metadata=metadata,
            is_parent=False,
            parent_id=parent.id,
        )

    def _fallback_chunking(
        self,
        text: str,
        document_id: str,
        filename: str,
        partition: str,
    ) -> tuple[List[Chunk], List[Chunk]]:
        """回退分块策略：简单按token数切分"""
        parent_chunks = []
        child_chunks = []

        paragraphs = text.split("\n\n")
        current_parent = []
        current_tokens = 0
        parent_index = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if (
                current_tokens >= self.parent_min_tokens
                and current_tokens + para_tokens > self.parent_max_tokens
            ):
                parent_content = "\n\n".join(current_parent)
                parent_id = f"{document_id}_parent_{parent_index}"
                parent = Chunk(
                    id=parent_id,
                    content=parent_content,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "partition": partition,
                        "chunk_index": parent_index,
                        "chunk_type": "parent",
                        "title_paths": [],
                    },
                    is_parent=True,
                )
                parent_chunks.append(parent)
                parent_index += 1
                current_parent = []
                current_tokens = 0

            current_parent.append(para)
            current_tokens += para_tokens

        # 最后一个父块
        if current_parent:
            parent_content = "\n\n".join(current_parent)
            parent_id = f"{document_id}_parent_{parent_index}"
            parent = Chunk(
                id=parent_id,
                content=parent_content,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "partition": partition,
                    "chunk_index": parent_index,
                    "chunk_type": "parent",
                    "title_paths": [],
                },
                is_parent=True,
            )
            parent_chunks.append(parent)

        # 为每个父块生成子块
        child_chunks = self._create_child_chunks(parent_chunks, document_id, filename, partition)

        return parent_chunks, child_chunks
