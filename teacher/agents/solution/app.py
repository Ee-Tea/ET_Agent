# teacher/agents/solution/app.py
import os, sys, json
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# --- 패키지 임포트 안전화 ---
try:
    from teacher.agents.solution.solution_agent import SolutionAgent
except Exception:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))  # .../llm-T
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from teacher.agents.solution.solution_agent import SolutionAgent

# (Milvus는 옵션)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections
    from langchain_core.documents import Document
except Exception:
    Milvus = None  # 미설치여도 동작

load_dotenv()
GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED(page_title="🧠 문제 해답 생성기 (JSON)", layout="wide")
st.title("🧠 문제 해답 생성기 (JSON Batch)")

# -------- 사이드바: JSON 업로드 / 실행옵션 --------
with st.sidebar:
    st.subheader("🗂️ 문제 JSON 업로드")
    up = st.file_uploader("파일 선택 (.json)", type=["json"])
    st.caption("형식: [{ 'question': str, 'options': [str, ...] }, ...]")

    st.divider()
    st.subheader("⚙️ 실행 옵션")
    save_to_vector = st.checkbox("Milvus에 저장", value=False)
    if save_to_vector:
        milvus_host = st.text_input("Milvus Host", value="localhost")
        milvus_port = st.text_input("Milvus Port", value="19530")
        collection_name = st.text_input("Collection", value="problems")

# -------- 리소스: Milvus(옵션) / Agent --------
@st.cache_resource
def init_vectorstore(host: str, port: str, coll: str) -> Optional["Milvus"]:
    if not save_to_vector or Milvus is None:
        return None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_milvus import Milvus
        from pymilvus import connections
        emb = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask",
                                    model_kwargs={"device": "cpu"})
        if "default" in connections.list_connections():
            connections.disconnect("default")
        connections.connect(alias="default", host=host, port=port)
        vs = Milvus(embedding_function=emb,
                    collection_name=coll,
                    connection_args={"host": host, "port": port})
        return vs
    except Exception as e:
        st.sidebar.warning(f"Milvus 연결 실패: {e}")
        return None

def build_agent(use_store: bool) -> SolutionAgent:
    """
    use_store=False이면 store 노드를 결과만 적재하는 안전 스텁으로 교체 후 그래프 재컴파일
    """
    ag = SolutionAgent()

    if not use_store:
        def _store_stub(self, state: Dict[str, Any]) -> Dict[str, Any]:
            item = {
                "user_problem": state.get("user_problem", ""),
                "user_problem_options": state.get("user_problem_options", []),
                "generated_answer": state.get("generated_answer", ""),
                "generated_explanation": state.get("generated_explanation", ""),
                "generated_subject": state.get("generated_subject", ""),
                "validated": state.get("validated", False),
                "chat_history": state.get("chat_history", []),
            }
            state.setdefault("results", []).append(item)
            return state

        # 메서드 바인딩 후 그래프 재생성
        ag._store_to_vector_db = _store_stub.__get__(ag, SolutionAgent)
        ag.graph = ag._create_graph()
    return ag

if save_to_vector:
    vectorstore = init_vectorstore(milvus_host, milvus_port, collection_name)
else:
    vectorstore = None

agent = build_agent(use_store=save_to_vector and vectorstore is not None)

# -------- 입력: 지시문 --------
user_instr = st.text_input("✍️ 요구사항/지시문 (모든 문제에 공통 적용)", value="정답 번호와 풀이, 과목을 알려줘.")

# -------- JSON 파싱 / 미리보기 --------
problems: List[Dict[str, Any]] = []
if up:
    try:
        problems = json.loads(up.read().decode("utf-8"))
        assert isinstance(problems, list)
    except Exception as e:
        st.error(f"JSON 파싱 실패: {e}")
        problems = []

if problems:
    st.success(f"총 {len(problems)}문제 로드됨.")
    # 선택 실행
    idx = st.number_input("🔢 선택 실행 (1~N)", min_value=1, max_value=len(problems), value=1, step=1)
    sel = problems[idx - 1]
    st.markdown("**미리보기**")
    st.write(sel.get("question", ""))
    for i, o in enumerate(sel.get("options", []), 1):
        st.write(f"{i}. {o}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 선택 문제 풀이"):
            with st.spinner("실행 중..."):
                init_state = {
                    "user_input_txt": user_instr,
                    "user_problem": sel.get("question", ""),
                    "user_problem_options": sel.get("options", []),
                    "vectorstore": vectorstore,
                    "retrieved_docs": [],
                    "similar_questions_text": "",
                    "generated_answer": "",
                    "generated_explanation": "",
                    "generated_subject": "",
                    "validated": False,
                    "retry_count": 0,
                    "results": [],
                    "chat_history": [],
                    "source_type": "external" if (save_to_vector and vectorstore) else "internal",
                }
                final_state = agent.graph.invoke(init_state, config={"recursion_limit": 200})
                results = final_state.get("results", [])
                if results:
                    last = results[-1]
                    st.markdown(f"**정답(번호)**: {last.get('generated_answer','-')}")
                    st.markdown(f"**과목**: {last.get('generated_subject','-')}")
                    st.markdown("**풀이**")
                    st.write(last.get("generated_explanation","-"))
                else:
                    st.error("결과가 비어 있습니다.")

    with col2:
        if st.button("⏩ 전체 문제 일괄 풀이"):
            outs = []
            prog = st.progress(0)
            for i, p in enumerate(problems, 1):
                init_state = {
                    "user_input_txt": user_instr,
                    "user_problem": p.get("question", ""),
                    "user_problem_options": p.get("options", []),
                    "vectorstore": vectorstore,
                    "retrieved_docs": [],
                    "similar_questions_text": "",
                    "generated_answer": "",
                    "generated_explanation": "",
                    "generated_subject": "",
                    "validated": False,
                    "retry_count": 0,
                    "results": [],
                    "chat_history": [],
                    "source_type": "external" if (save_to_vector and vectorstore) else "internal",
                }
                final_state = agent.graph.invoke(init_state, config={"recursion_limit": 200})
                res = (final_state.get("results") or [{}])[-1]
                outs.append(res)
                prog.progress(i / len(problems))
            st.success(f"총 {len(outs)}건 완료")
            for i, r in enumerate(outs, 1):
                st.markdown(f"### 결과 #{i}")
                st.markdown(f"- **정답(번호)**: {r.get('generated_answer','-')}")
                st.markdown(f"- **과목**: {r.get('generated_subject','-')}")
                st.markdown("**풀이**")
                st.write(r.get("generated_explanation","-"))
else:
    st.info("좌측에서 JSON 파일을 업로드하세요.")

# -------- 키 안내 --------
if not GROQ_API_KEY=REDACTED("ℹ️ `.env`의 GROQ_API_KEY=REDACTED 호출 실패 가능성이 있습니다.")
