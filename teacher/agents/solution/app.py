# teacher/agents/solution/app.py
import os, sys, json
from typing import List, Dict, Any

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

# --- Milvus 필수 ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections
except Exception as e:
    st.error(f"Milvus 관련 라이브러리가 필요합니다: {e}\n\npip/uv로 다음을 설치하세요: langchain-milvus, pymilvus, langchain-huggingface")
    st.stop()

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="🧠 문제 해답 생성기 (JSON+Milvus)", layout="wide")
st.title("🧠 문제 해답 생성기 (JSON + Milvus 필수)")
st.caption("JSON 업로드 후 선택 실행/일괄 실행. 결과는 항상 Milvus에 저장됩니다.")

# -------- 사이드바: JSON 업로드 / Milvus 설정 --------
with st.sidebar:
    st.subheader("🗂️ 문제 JSON 업로드")
    up = st.file_uploader("파일 선택 (.json)", type=["json"])
    st.caption("형식: [{ 'question': str, 'options': [str, ...] }, ...]")

    st.divider()
    st.subheader("🗄️ Milvus 연결(필수)")
    milvus_host = st.text_input("Host", value="localhost")
    milvus_port = st.text_input("Port", value="19530")
    collection_name = st.text_input("Collection", value="problems")

# -------- 리소스: Milvus / Agent --------
@st.cache_resource
def init_vectorstore(host: str, port: str, coll: str) -> Milvus:
    try:
        emb = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
        )
        if "default" in connections.list_connections():
            connections.disconnect("default")
        connections.connect(alias="default", host=host, port=port)
        vs = Milvus(
            embedding_function=emb,
            collection_name=coll,
            connection_args={"host": host, "port": port},
        )
        return vs
    except Exception as e:
        st.error(f"Milvus 연결 실패: {e}")
        st.stop()

@st.cache_resource
def get_agent() -> SolutionAgent:
    return SolutionAgent()

vectorstore = init_vectorstore(milvus_host, milvus_port, collection_name)
agent = get_agent()

# -------- 입력: 공통 지시문 --------
user_instr = st.text_input("✍️ 공통 지시문", value="정답 번호와 풀이, 과목을 알려줘.")

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
    idx = st.number_input("🔢 선택 실행 (1~N)", min_value=1, max_value=len(problems), value=1, step=1)
    sel = problems[idx - 1]
    st.markdown("**미리보기**")
    st.write(sel.get("question", ""))
    for i, o in enumerate(sel.get("options", []), 1):
        st.write(f"{o}")

    col1, col2 = st.columns(2)

    def run_one(p: Dict[str, Any]):
        init_state = {
            "user_input_txt": user_instr,
            "user_problem": p.get("question", ""),
            "user_problem_options": p.get("options", []),
            "vectorstore": vectorstore,                 # ✅ 항상 연결
            "retrieved_docs": [],
            "similar_questions_text": "",
            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,
            "retry_count": 0,
            "results": [],
            "chat_history": [],
            "source_type": "external",                  # ✅ 항상 외부 저장
        }
        return agent.graph.invoke(init_state, config={"recursion_limit": 200})

    with col1:
        if st.button("▶️ 선택 문제 풀이"):
            with st.spinner("실행 중..."):
                final_state = run_one(sel)
                results = final_state.get("results", [])
                if results:
                    last = results[-1]
                    st.markdown(f"**정답(번호)**: {last.get('generated_answer','-')}")
                    st.markdown(f"**과목**: {last.get('generated_subject','-')}")
                    st.markdown(f"**풀이**: {last.get('generated_explanation','-')}")
                    st.write(last.get("generated_explanation","-"))
                else:
                    st.error("결과가 비어 있습니다.")

    with col2:
        if st.button("⏩ 전체 문제 일괄 풀이"):
            outs = []
            prog = st.progress(0)
            for i, p in enumerate(problems, 1):
                final_state = run_one(p)
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
if not GROQ_API_KEY:
    st.caption("ℹ️ `.env`의 GROQ_API_KEY가 비어있습니다. LLM 호출 실패 가능성이 있습니다.")
