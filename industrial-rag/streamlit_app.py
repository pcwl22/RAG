"""Streamlit frontend for the local RAG knowledge QA system."""
import time

import httpx
import streamlit as st

st.set_page_config(
    page_title="RAG 本地知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000"

DOCUMENT_CATEGORIES = {
    "contract": "合同",
    "manual": "手册",
    "process": "流程",
    "policy": "制度",
    "general": "通用",
}


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("show_sources", True)
    st.session_state.setdefault("enable_coreference", True)
    st.session_state.setdefault("enable_decomposition", True)


def upload_document(file, category: str):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"partition": category, "metadata": "{}"}
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{API_BASE_URL}/api/v1/documents/ingest", files=files, data=data)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        st.error(f"上传失败: {exc}")
        return None


def get_documents() -> list[dict]:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_BASE_URL}/api/v1/documents")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload.get("documents", [])
            return payload
    except Exception as exc:
        st.warning(f"暂时无法获取文档列表: {exc}")
        return []


def delete_document(document_id: str) -> bool:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(f"{API_BASE_URL}/api/v1/documents/{document_id}")
            response.raise_for_status()
            return True
    except Exception as exc:
        st.error(f"删除失败: {exc}")
        return False


def enhanced_query(query: str, partition: str | None = None) -> dict | None:
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-8:]
    ]
    payload = {
        "query": query,
        "chat_history": history,
        "enable_coreference": st.session_state.enable_coreference,
        "enable_decomposition": st.session_state.enable_decomposition,
        "top_k": 5,
        "partition": partition,
    }
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(f"{API_BASE_URL}/api/v1/query/enhanced", json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        st.error(f"问答失败: {exc}")
        return None


def sidebar() -> str | None:
    with st.sidebar:
        st.title("📚 文档管理")

        uploaded_file = st.file_uploader(
            "选择文档",
            type=["pdf", "txt", "md", "docx", "doc", "xlsx", "xls"],
        )
        category = st.selectbox(
            "文档分类",
            options=list(DOCUMENT_CATEGORIES.keys()),
            format_func=lambda key: DOCUMENT_CATEGORIES[key],
        )

        if st.button("上传文档", disabled=uploaded_file is None, use_container_width=True):
            with st.spinner("正在上传并入库..."):
                result = upload_document(uploaded_file, category)
                if result:
                    st.success(f"任务已提交: {result.get('task_id', 'unknown')}")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        st.subheader("查询设置")
        selected_partition = st.selectbox(
            "限定分类",
            options=["all", *DOCUMENT_CATEGORIES.keys()],
            format_func=lambda key: "全部" if key == "all" else DOCUMENT_CATEGORIES[key],
        )
        st.checkbox("指代消解", key="enable_coreference")
        st.checkbox("复合问题拆分", key="enable_decomposition")
        st.checkbox("显示来源", key="show_sources")

        st.divider()
        st.subheader("已入库文档")
        docs = get_documents()
        if not docs:
            st.info("暂无文档")
        for doc in docs:
            document_id = doc.get("document_id", "")
            filename = doc.get("filename") or "unknown"
            partition = doc.get("partition") or "general"
            with st.expander(filename):
                st.write(f"分类: {DOCUMENT_CATEGORIES.get(partition, partition)}")
                st.write(f"分块数: {doc.get('chunk_count', 0)}")
                st.code(document_id)
                if st.button("删除", key=f"delete_{document_id}", use_container_width=True):
                    if delete_document(document_id):
                        st.success("已删除")
                        time.sleep(1)
                        st.rerun()

        return None if selected_partition == "all" else selected_partition


def render_sources(results: list[dict]) -> None:
    if not st.session_state.show_sources or not results:
        return
    with st.expander("来源上下文", expanded=False):
        for index, doc in enumerate(results, 1):
            metadata = doc.get("metadata") or {}
            source = metadata.get("filename") or metadata.get("document_id") or "unknown"
            score = doc.get("rrf_score", doc.get("score", 0))
            st.markdown(f"**{index}. {source}** · score `{score:.4f}`")
            st.caption(doc.get("child_content") or doc.get("content", "")[:500])


def main_chat(partition: str | None) -> None:
    st.title("RAG 本地知识库智能问答系统")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("输入你的问题...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索和生成..."):
            result = enhanced_query(prompt, partition=partition)

        if not result:
            answer = "请求失败，请检查后端服务是否已启动。"
            st.markdown(answer)
        else:
            answer = result.get("answer", "")
            st.markdown(answer)

            understanding = result.get("understanding") or {}
            if understanding.get("is_decomposed"):
                with st.expander("问题拆分", expanded=False):
                    for subquery in understanding.get("subqueries", []):
                        st.write(subquery)

            render_sources(result.get("results", []))

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    init_session_state()
    partition = sidebar()
    main_chat(partition)

    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"当前对话轮数: {len(st.session_state.messages) // 2}")
    with col2:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
