import os, sys, json, tempfile
from datetime import datetime
from typing import List, Dict, Literal

import streamlit as st
from dotenv import load_dotenv

# ---- add project ROOT to sys.path (.. / .. / .. from this file) ----
CURR  = os.path.dirname(os.path.abspath(__file__))          # .../teacher/agents/solution
PARENT= os.path.dirname(CURR)                                # .../teacher/agents
GRAND = os.path.dirname(PARENT)                              # .../teacher
ROOT  = os.path.dirname(GRAND)                               # .../llm-T  (프로젝트 루트)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

# 이제 패키지 임포트
from teacher.agents.solution.solution_agent import SolutionAgent

# ---------- Vector store (Milvus) ----------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections
from langchain_core.documents import Document

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "problems"

class NoRetrievalVectorStore:
    """Fallback vectorstore: no search, no add."""
    def similarity_search(self, query: str, k: int = 3):
        return []
    def add_documents(self, docs: List[Document]):
        return None

@st.cache_resource
def init_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus as LCMilvus

    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", model_kwargs={"device": "cpu"}
    )

    if "default" in connections.list_connections():
        connections.disconnect("default")
    # ✅ gRPC 전용 (host/port 말고 address 사용)
    connections.connect(alias="default", address=f"{MILVUS_HOST}:{MILVUS_PORT}", secure=False, timeout=5)

    # LangChain VectorStore (address만)
    return LCMilvus(
        embedding_function=embedding_model,
        collection_name=MILVUS_COLLECTION,
        connection_args={"address": f"{MILVUS_HOST}:{MILVUS_PORT}"},
    )


@st.cache_resource
def get_agent():
    return SolutionAgent()

def save_uploads(uploaded_files, subdir: str) -> List[str]:
    saved_paths = []
    base_dir = os.path.join(tempfile.gettempdir(), "agent_uploads", subdir)
    os.makedirs(base_dir, exist_ok=True)
    for uf in uploaded_files or []:
        suffix = os.path.splitext(uf.name)[1] or ""
        fd, path = tempfile.mkstemp(dir=base_dir, suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uf.read())
        saved_paths.append(path)
    return saved_paths

def render_results(results: List[Dict]):
    if not results:
        st.info("결과가 비어 있습니다.")
        return
    for i, item in enumerate(results, 1):
        with st.expander(f"문항 {i}: {'(검증 통과)' if item.get('validated') else '(검증 미통과)'}", expanded=(i==1)):
            st.markdown(f"**문제**: {item.get('question','')}")
            opts = item.get('options', []) or []
            if opts:
                st.markdown("**보기**")
                for idx, o in enumerate(opts, 1):
                    st.write(f"{idx}. {o}")
            st.markdown(f"**정답**: {item.get('generated_answer','')}")
            st.markdown("**풀이**")
            st.write(item.get('generated_explanation',''))

# ---------- UI ----------
st.set_page_config(page_title="🧠 문제 자동 해답 생성기 (Chat)", layout="wide")
st.title("🧠 문제 자동 해답 생성기 — Chat UI")

with st.sidebar:
    st.header("⚙️ 설정")
    # ✅ 오직 내부/외부 판단만
    source_type = st.radio("문제 원천", ["external", "internal"], horizontal=True, index=0)
    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state.pop("messages", None)
        st.session_state.pop("upload_key_imgs", None)
        st.session_state.pop("upload_key_files", None)
        st.rerun()

# Session state init
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "upload_key_imgs" not in st.session_state:
    st.session_state["upload_key_imgs"] = "imgs-0"
if "upload_key_files" not in st.session_state:
    st.session_state["upload_key_files"] = "files-0"

# Display history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Composer: attachments + chat input
with st.container(border=True):
    st.markdown("**📎 첨부파일 (선택)**")
    uploaded_images = st.file_uploader(
        "문제 이미지", type=["png","jpg","jpeg","webp"],
        accept_multiple_files=True, key=st.session_state["upload_key_imgs"]
    )
    uploaded_files = st.file_uploader(
        "문서 파일 (PDF/DOCX/TXT/MD 등)", type=["pdf","docx","txt","md"],
        accept_multiple_files=True, key=st.session_state["upload_key_files"]
    )
    stm_json = st.text_area(
        "STM JSON (옵션): 예) [{\"question\":\"...\",\"options\":[\"...\"]}]",
        height=100, value="",
        help="단발성 임시 내부 큐를 넣고 싶을 때 사용"
    )

user_text = st.chat_input("메시지를 입력하고 Enter를 누르세요")

def run_agent_once(
    *, user_text: str, kind: Literal["text","image","file","stm"],
    vectorstore, external_image_paths=None, external_file_paths=None, short_term_memory=None
):
    agent = get_agent()
    _user_input_txt = user_text or "문제를 풀어줘."
    return agent.execute(
        user_input_txt=_user_input_txt,
        source_type=source_type,   # 내부/외부만 전달
        input_kind=kind,
        external_image_paths=external_image_paths if kind == "image" else None,
        vectorstore=vectorstore,
        short_term_memory=short_term_memory if kind == "stm" else None,
        external_file_paths=external_file_paths if kind == "file" else None,
        exam_title="채팅 세션",
        difficulty="중급",
        subject="기타",
        recursion_limit=1000,
    )

if user_text is not None:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    vectorstore = init_vectorstore()  # ✅ 자동 Milvus 연결 (또는 폴백)

    short_term_memory = None
    if stm_json.strip():
        try:
            parsed = json.loads(stm_json)
            if isinstance(parsed, list):
                short_term_memory = parsed
            else:
                st.warning("STM JSON은 리스트여야 합니다. 무시합니다.")
        except Exception as e:
            st.warning(f"STM JSON 파싱 실패: {e}")

    image_paths = save_uploads(uploaded_images, "images") if uploaded_images else []
    file_paths = save_uploads(uploaded_files, "files") if uploaded_files else []

    final_states: List[Dict] = []
    try:
        if short_term_memory:
            final_states.append(run_agent_once(user_text=user_text, kind="stm",
                                               vectorstore=vectorstore,
                                               short_term_memory=short_term_memory))
        if image_paths and file_paths:
            final_states.append(run_agent_once(user_text=user_text, kind="image",
                                               vectorstore=vectorstore, external_image_paths=image_paths))
            final_states.append(run_agent_once(user_text=user_text, kind="file",
                                               vectorstore=vectorstore, external_file_paths=file_paths))
        elif image_paths:
            final_states.append(run_agent_once(user_text=user_text, kind="image",
                                               vectorstore=vectorstore, external_image_paths=image_paths))
        elif file_paths:
            final_states.append(run_agent_once(user_text=user_text, kind="file",
                                               vectorstore=vectorstore, external_file_paths=file_paths))
        if not short_term_memory and not image_paths and not file_paths:
            final_states.append(run_agent_once(user_text=user_text, kind="text", vectorstore=vectorstore))
    except Exception as e:
        st.error(f"에이전트 실행 중 오류: {e}")

    merged_results: List[Dict] = []
    for fs in final_states:
        if isinstance(fs, dict):
            results = fs.get("results", [])
            if results:
                merged_results.extend(results)

    with st.chat_message("assistant"):
        if merged_results:
            st.success(f"✅ 총 {len(merged_results)}개의 문항 결과가 생성되었습니다.")
            render_results(merged_results)
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "user_input_txt": user_text,
                "total_results": len(merged_results),
                "results": merged_results,
            }
            st.download_button(
                label="📥 결과 JSON 다운로드",
                data=json.dumps(results_data, ensure_ascii=False, indent=2),
                file_name=f"solution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            # 메시지에 상세 결과도 함께 추가
            details = "\n\n".join(
                [
                    f"문항 {i+1}:\n"
                    f"문제: {item.get('question','')}\n"
                    f"정답: {item.get('generated_answer','')}\n"
                    f"풀이: {item.get('generated_explanation','')}"
                    for i, item in enumerate(merged_results)
                ]
            )
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"총 {len(merged_results)}개의 문항 결과를 생성했습니다. 상세는 아래 확장영역을 확인하세요.\n\n{details}"
            })
        else:
            st.info("결과가 없습니다. 입력 텍스트 또는 첨부를 확인해 주세요.")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "결과가 없었습니다. 입력/첨부를 확인해 주세요."
            })

    st.session_state["upload_key_imgs"] = st.session_state["upload_key_imgs"] + "_r"
    st.session_state["upload_key_files"] = st.session_state["upload_key_files"] + "_r"
    st.rerun()
