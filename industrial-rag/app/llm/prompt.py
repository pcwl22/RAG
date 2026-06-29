"""Prompt template loading and rendering."""
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

_DEFAULT_PROMPTS = {
    "system.txt": "你是一个专业的本地知识库 RAG 问答助手。请严格基于上下文回答，不编造信息。",
    "business.txt": "业务规则：只使用提供的上下文；上下文不足时明确说明。",
    "citation.txt": "引用要求：依据必须来自命中的上下文，并给出文件、位置或原文片段。",
    "output.txt": (
        "必须严格按以下格式输出：\n\n"
        "【回答】\n"
        "结论：\n"
        "在这里直接给出答案正文。\n\n"
        "依据：\n"
        "1. 文件名或来源：章节/条款/页码；内容：命中的原文片段\n"
    ),
    "rag_template.jinja2": (
        "{{ context_block }}\n\n"
        "---\n\n"
        "用户问题：\n"
        "{{ query }}\n\n"
        "回答要求：\n"
        "{{ answer_guidance }}\n\n"
        "{{ output_instructions }}"
    ),
}


@lru_cache()
def load_prompt(filename: str) -> str:
    """Load a prompt file from the project-level prompts directory."""
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return _DEFAULT_PROMPTS.get(filename, "").strip()


def _render_template(template: str, values: dict[str, Any]) -> str:
    try:
        from jinja2 import Template
    except ImportError:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered.strip()

    return Template(template).render(**values).strip()


def build_system_prompt(default: str | None = None) -> str:
    """Build the system prompt from system, business, and citation prompt files."""
    parts = [
        load_prompt("system.txt") or default or "",
        load_prompt("business.txt"),
        load_prompt("citation.txt"),
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def get_output_instructions(stream: bool = False) -> str:
    """Return output instructions for normal or streaming answer generation."""
    output = load_prompt("output.txt")
    if not stream:
        return output

    return (
        f"{output}\n\n"
        "流式输出要求：\n"
        "1. 只输出最终答案正文，不要输出【回答】标题、结论：或依据：。\n"
        "2. 不要输出来源章节，系统会在答案后统一补充依据。\n"
        "3. 不要添加 Markdown 标题符号。"
    )


def build_rag_prompt(
    query: str,
    context: str,
    output_instructions: str | None = None,
) -> str:
    """Render the full RAG prompt from the project prompt template."""
    if context.strip():
        context_block = f"以下是从知识库检索到的上下文：\n{context.strip()}"
        answer_guidance = "请严格基于上述上下文回答。若上下文不充分，请明确指出不足之处。"
    else:
        context_block = "知识库中没有检索到相关上下文。"
        answer_guidance = "请直接说明无法基于当前知识库回答，并提示需要补充的资料类型。"

    return _render_template(
        load_prompt("rag_template.jinja2"),
        {
            "context_block": context_block,
            "query": query,
            "answer_guidance": answer_guidance,
            "output_instructions": output_instructions or ANSWER_FORMAT_INSTRUCTIONS,
        },
    )


ANSWER_FORMAT_INSTRUCTIONS = get_output_instructions(stream=False)
STREAM_ANSWER_INSTRUCTIONS = get_output_instructions(stream=True)

__all__ = [
    "ANSWER_FORMAT_INSTRUCTIONS",
    "STREAM_ANSWER_INSTRUCTIONS",
    "PROMPTS_DIR",
    "build_rag_prompt",
    "build_system_prompt",
    "get_output_instructions",
    "load_prompt",
]
