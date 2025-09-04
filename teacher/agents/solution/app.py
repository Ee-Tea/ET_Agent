# teacher/agents/solution/app.py
import os, sys, json
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# --- 패키지 임포트 안전화 ---
from teacher.agents.solution.solution_agent import SolutionAgent

# --- asyncio 루프 보장 (Streamlit 스레드) ---
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    # Windows에서 pymilvus의 Async 클라이언트 경고 방지
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Milvus 필수 ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections
except Exception as e:
    st.error(
        f"Milvus 관련 라이브러리가 필요합니다: {e}\n\n"
        "pip/uv로 다음을 설치하세요: langchain-milvus, pymilvus, langchain-huggingface"
    )
    st.stop()

load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED(page_title="🧠 문제 해답 생성기 (JSON+Milvus)", layout="wide")
st.title("🧠 문제 해답 생성기 (JSON + Milvus 필수)")
st.caption("JSON 업로드 후 선택 실행/일괄 실행. 결과는 항상 Milvus에 저장됩니다.")

# -------- 사이드바: JSON 업로드 / Milvus 설정 --------
with st.sidebar:
    st.subheader("🗂️ 문제 JSON 업로드")
    up = st.file_uploader("파일 선택 (.json)", type=["json"])
    st.caption("형식: [{ 'question': str, 'options': [str, ...] }, ...]")

    st.divider()
    st.subheader("🗄️ Milvus 연결(필수)")
    milvus_host = st.text_input("Host", value=os.getenv("MILVUS_HOST", "localhost"))
    milvus_port = st.text_input("Port", value=os.getenv("MILVUS_PORT", "19530"))
    st.caption("collections: problems, concept_summary")


# --- 리소스: Milvus / Agent --------
@st.cache_resource
def init_vectorstore(host: str, port: str, coll: str,
                     *, text_field: str | None = None,
                     vector_field: str | None = None,
                     metric_type: str | None = None) -> Milvus:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections, Collection

    emb = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
    )

    if "default" not in connections.list_connections():
        connections.connect(alias="default", host=host, port=port)

    # 1) 컬렉션의 실제 인덱스 metric 확인
    actual_metric = metric_type
    try:
        col = Collection(coll)
        # 이미 인덱스가 있다면 그 metric을 사용
        if col.indexes:
            # 보통 하나만 존재
            params = col.indexes[0].params or {}
            # pymilvus 버전에 따라 키가 다를 수 있어 두 가지 모두 시도
            actual_metric = params.get("metric_type") or params.get("metric_type".upper())
    except Exception:
        pass

    if not actual_metric:
        # 인덱스가 없거나 조회 실패 시 기본값(당신의 환경에 맞게 선택)
        actual_metric = "L2"   # or "COSINE"

    # 2) langchain-milvus에 전달
    kwargs = {
        "embedding_function": emb,
        "collection_name": coll,
        "connection_args": {"host": host, "port": port},
        # 인덱스는 '이미 있는 것'을 그대로 쓰도록 index_params는 넣지 않음
        "search_params": {"metric_type": actual_metric, "params": {"nprobe": 10}},
    }
    if text_field is not None:
        kwargs["text_field"] = text_field
    if vector_field is not None:
        kwargs["vector_field"] = vector_field

    return Milvus(**kwargs)




@st.cache_resource
def get_agent() -> SolutionAgent:
    return SolutionAgent()

# problems 컬렉션: (LangChain로 만든 기본 스키마면) 필드 지정 없이
vectorstore_p = init_vectorstore(milvus_host, milvus_port, "problems")

# concept_summary 컬렉션: 사용자 정의 스키마 → content / embedding 필드 지정 필요
vectorstore_c = init_vectorstore(
    milvus_host, milvus_port, "concept_summary",
    text_field="content",
    vector_field="embedding", 
)


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
        st.write(f"{i}. {o}")

    col1, col2 = st.columns(2)

    def run_one(p: Dict[str, Any]):
        return agent.invoke(
            user_input_txt=user_instr,
            user_problem=p.get("question", ""),
            user_problem_options=p.get("options", []),
            vectorstore_p=vectorstore_p,              # ✅ 에이전트에 명시 전달
            vectorstore_c=vectorstore_c,              # ✅ 에이전트에 명시 전달
            recursion_limit=200,
        )

    with col1:
        if st.button("▶️ 선택 문제 풀이"):
            with st.spinner("실행 중..."):
                final_state = run_one(sel)
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
if not OPENAI_API_KEY=REDACTED("ℹ️ `.env`의 OPENAI_API_KEY=REDACTED 호출 실패 가능성이 있습니다.")
