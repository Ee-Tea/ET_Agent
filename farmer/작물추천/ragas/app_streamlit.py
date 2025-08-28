# -*- coding: utf-8 -*-
# app_streamlit.py
# ------------------------------------------------------------
# Streamlit UI for:
# 1) Chat (LangGraph + Groq + FAISS + HF Embeddings)
# 2) Golden-set Evaluation (CSV)
# 3) Amnesty QA RAGAS Evaluation (Ollama + ragas)
#
# Usage:
#   streamlit run app_streamlit.py
# ------------------------------------------------------------

import os, json, re, time
from typing import TypedDict, Optional, Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ===== User-requested path (do not change) =====
GOLDENSET_CSV = r"C:\Rookies_project\ET_Agent\ET_Agent-dev-eejang\farmer\작물추천\Goldenset_test\Goldenset_test1.csv"

# === Settings ===
VECTOR_DB_PATH   = os.getenv("VECTOR_DB_PATH", "faiss_pdf_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
SIM_THRESHOLD_DEFAULT = float(os.getenv("SIM_THRESHOLD", "0.75"))

if not GROQ_API_KEY:
    st.stop()  # Fail early in Streamlit

# LangChain / LangGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- Prompt ---
PROMPT_TMPL = """
당신은 대한민국 농업 작물재배 전문가입니다.
아래 '문맥'만 사용해 질문에 답하세요.

[문맥]
{context}

규칙:
- 문맥에 없는 정보/추측/한자 금지.
- 한글로만 작성.
- 단계/설명은 "한 문장씩 줄바꿈".
- 문맥에 근거 없으면: "주어진 정보로는 답변할 수 없습니다."

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)

# --- State ---
class GraphState(TypedDict):
    question: Optional[str]
    vectorstore: Optional[Any]
    context: Optional[str]
    answer: Optional[str]
    sources: Optional[List[Dict[str, Any]]]

# --- Common ---
def load_vectorstore(db_path: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_with_meta(vs: Any, question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question) or []
    ctx = "\n\n".join([d.page_content for d in docs])
    sources: List[Dict[str, Any]] = []
    for rank, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src  = meta.get("source") or meta.get("file_path") or meta.get("path")
        page = meta.get("page")
        sources.append({"rank": rank, "source": src, "page": page, "metadata": meta})
    return ctx, sources

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

def load_vs_node(state: GraphState) -> Dict[str, Any]:
    vs = load_vectorstore(VECTOR_DB_PATH)
    return {**state, "vectorstore": vs}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    if not state.get("vectorstore"):
        raise ValueError("vectorstore is missing.")
    q = state["question"] or ""
    ctx, sources = retrieve_with_meta(state["vectorstore"], q, k=5)
    return {**state, "context": ctx, "sources": sources}

def generate_node(state: GraphState) -> Dict[str, Any]:
    if not state.get("context") or not state.get("question"):
        raise ValueError("context/question missing")
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"]})
    return {**state, "answer": ans}

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_vs", load_vs_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge("load_vs", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.set_entry_point("load_vs")
    return g.compile()

def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float); vb = np.array(b, dtype=float)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    return float(np.dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else 0.0

def _search_similarity_max(vs: Any, embeddings: HuggingFaceEmbeddings, question: str, k: int = 20) -> float:
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question) or []
    if not docs:
        return 0.0
    qvec = embeddings.embed_query(question)
    dvecs = embeddings.embed_documents([d.page_content for d in docs])
    return max((_cosine(qvec, dv) for dv in dvecs), default=0.0)

def _sources_to_dataframe(sources: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for s in (sources or []):
        src = s.get("source")
        fname = os.path.basename(src) if src else "unknown"
        rows.append({"번호": s.get("rank"), "파일명": fname})
    return pd.DataFrame(rows, columns=["번호", "파일명"])

def _append_used_context_log(index: str, question: str, generated_answer: str, context_raw: str,
                             golden_answer: Optional[str] = None,
                             sources: Optional[List[Dict[str, Any]]] = None) -> None:
    try:
        with open("used_context_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[# {index}] --------------------------------------------\n")
            f.write(f"질문: {question}\n")
            if golden_answer is not None:
                f.write(f"골든셋 답변 A: {golden_answer}\n")
            f.write(f"LLM 답변: {generated_answer}\n")
            f.write("----- CONTEXT START (RAW) -----\n")
            f.write(context_raw if context_raw else "(컨텍스트 없음)")
            f.write("\n----- CONTEXT END -----\n\n")
            if sources:
                f.write("----- SOURCES (번호/파일명) -----\n")
                for s in sources:
                    src = s.get("source")
                    fname = os.path.basename(src) if src else "unknown"
                    f.write(f"{s.get('rank')}\t{fname}\n")
                f.write("\n")
    except Exception as e:
        st.warning(f"[경고] 컨텍스트 로그 저장 실패: {e}")

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="LangGraph Chat & Eval + RAGAS", layout="wide")
st.title("LangGraph Chat/Eval + RAGAS (GUI)")

with st.sidebar:
    st.header("설정")
    st.caption("`.env`로도 설정 가능")
    st.text_input("VECTOR_DB_PATH", value=VECTOR_DB_PATH, key="VECTOR_DB_PATH")
    st.text_input("EMBED_MODEL_NAME", value=EMBED_MODEL_NAME, key="EMBED_MODEL_NAME")
    st.text_input("GOLDENSET_CSV", value=GOLDENSET_CSV, key="GOLDENSET_CSV")
    st.text_input("GROQ_MODEL", value=GROQ_MODEL, key="GROQ_MODEL")
    st.number_input("TEMPERATURE", value=TEMPERATURE, step=0.1, key="TEMPERATURE")
    st.number_input("SIM_THRESHOLD", value=SIM_THRESHOLD_DEFAULT, step=0.05, key="SIM_THRESHOLD")

tabs = st.tabs(["채팅", "평가(골든셋)", "RAGAS (Amnesty QA)"])

# ----------- Tab 1: Chat -----------
with tabs[0]:
    st.subheader("채팅 (컨텍스트 RAW 및 소스 표시)")
    if "graph_app" not in st.session_state:
        st.session_state.graph_app = build_graph()

    user_q = st.text_input("질문을 입력하세요", key="chat_question")
    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("생성하기", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("지우기", use_container_width=True)

    if clear_btn:
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_context", None)
        st.session_state.pop("last_sources_df", None)
        st.experimental_rerun()

    if run_btn and user_q.strip():
        # Rebuild graph if settings changed
        global VECTOR_DB_PATH, EMBED_MODEL_NAME, GROQ_MODEL, TEMPERATURE
        VECTOR_DB_PATH = st.session_state["VECTOR_DB_PATH"]
        EMBED_MODEL_NAME = st.session_state["EMBED_MODEL_NAME"]
        GROQ_MODEL = st.session_state["GROQ_MODEL"]
        TEMPERATURE = float(st.session_state["TEMPERATURE"])
        st.session_state.graph_app = build_graph()

        with st.spinner("생성 중..."):
            state = st.session_state.graph_app.invoke({"question": user_q})
            ans = state.get("answer", "")
            raw_ctx = state.get("context", "")
            srcs = state.get("sources") or []

            src_df = _sources_to_dataframe(srcs)

            st.session_state.last_answer = ans
            st.session_state.last_context = raw_ctx
            st.session_state.last_sources_df = src_df

            _append_used_context_log(index=f"chat-{int(time.time())}", question=user_q, generated_answer=ans, context_raw=raw_ctx, sources=srcs)

    if st.session_state.get("last_answer") is not None:
        st.markdown("#### 답변")
        st.write(st.session_state.last_answer or "(답변 없음)")
        with st.expander("LLM에 전달된 컨텍스트 (RAW 전체 보기)"):
            st.code(st.session_state.last_context or "(컨텍스트 없음)")
        if st.session_state.get("last_sources_df") is not None and not st.session_state.last_sources_df.empty:
            st.markdown("#### 참고 소스 (번호/파일명)")
            st.dataframe(st.session_state.last_sources_df, use_container_width=True)

# ----------- Tab 2: Evaluate (Golden-set) -----------
with tabs[1]:
    st.subheader("골든셋 평가 (CSV)")
    csv_path = st.text_input("CSV 경로", value=st.session_state.get("GOLDENSET_CSV", GOLDENSET_CSV), key="eval_csv_path")
    threshold = st.number_input("유사도 임계값", value=float(st.session_state["SIM_THRESHOLD"]), step=0.05, min_value=0.0, max_value=1.0)
    sample_limit = st.number_input("샘플 제한 (0=전체)", value=int(os.getenv("EVAL_SAMPLE_LIMIT", "0")), step=1, min_value=0, key="eval_sample_limit")

    def evaluate_goldenset_gui(csv_path: str, threshold: float):
        # local copies
        emb_model = st.session_state["EMBED_MODEL_NAME"]
        vec_path = st.session_state["VECTOR_DB_PATH"]

        eval_emb = HuggingFaceEmbeddings(model_name=emb_model)
        try:
            vs = load_vectorstore(vec_path)
        except Exception as e:
            st.warning(f"벡터스토어 로드 실패: {e}")
            vs = None

        # read csv
        df = None
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                st.info(f"CSV 인코딩 감지: {enc}")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read()
                from io import StringIO
                df = pd.read_csv(StringIO(data))
                st.warning("utf-8(errors=ignore) 강제 사용")
            except Exception as e:
                st.error(f"CSV 읽기 실패: {e}")
                return None, None

        # ensure schema
        df0 = df.copy()
        cols = [str(c).strip() for c in df0.columns]
        df0.columns = cols
        if {"question", "answer"}.issubset(df0.columns):
            qa_df = df0[["question", "answer"]]
        elif "제목" in df0.columns:
            def norm(x):
                return "" if pd.isna(x) else str(x).strip()
            def build_answer(row):
                parts = []
                if "요약글" in df0.columns:
                    y = norm(row.get("요약글"))
                    if y:
                        parts.append(y)
                for i in range(1, 11):
                    sj = f"서브제목{i}"; sn1 = f"서브내용{i}"; sn2 = f" 서브내용{i}"
                    title = norm(row.get(sj)) if sj in df0.columns else ""
                    body  = norm(row.get(sn1)) if sn1 in df0.columns else norm(row.get(sn2)) if sn2 in df0.columns else ""
                    if title or body:
                        parts.append(f"{title}. {body}".strip(". ").strip())
                return "\n".join([p for p in parts if p])
            qa_df = pd.DataFrame({
                "question": df0["제목"].astype(str).str.strip(),
                "answer": df0.apply(build_answer, axis=1)
            })
        else:
            st.error(f"CSV에 question/answer 또는 '제목' 컬럼이 필요합니다. 현재 컬럼: {list(df0.columns)}")
            return None, None

        # build graph
        app = build_graph()

        details = []
        correct = 0
        for idx, row in qa_df.iterrows():
            q = str(row["question"]); g = str(row["answer"]); i = idx + 1
            try:
                state = app.invoke({"question": q})
                a = state.get("answer", "")
                raw_ctx = state.get("context", "")
                srcs = state.get("sources") or []

                _append_used_context_log(index=f"eval-{i}", question=q, generated_answer=a, context_raw=raw_ctx, sources=srcs)

                g_vec = eval_emb.embed_query(g)
                a_vec = eval_emb.embed_query(a)
                ans_sim = _cosine(g_vec, a_vec)

                if vs is not None:
                    srch_sim = _search_similarity_max(vs, eval_emb, q, k=20)
                else:
                    srch_sim = 0.0

                ok = bool(ans_sim >= threshold)
                correct += int(ok)

                details.append({
                    "index": i,
                    "question": q,
                    "golden_answer": g,
                    "generated_answer": a,
                    "answer_similarity": float(ans_sim),
                    "search_similarity_max": float(srch_sim),
                    "is_correct": ok
                })
            except Exception as e:
                details.append({
                    "index": i,
                    "question": q,
                    "golden_answer": g,
                    "generated_answer": f"[ERROR] {e}",
                    "answer_similarity": 0.0,
                    "search_similarity_max": 0.0,
                    "is_correct": False
                })

        total = len(details)
        acc = (correct / total) * 100 if total else 0.0
        summary = {
            "file": os.path.abspath(csv_path),
            "total": total,
            "correct": int(correct),
            "accuracy": round(acc, 2),
            "threshold": threshold,
            "embedding_model": emb_model,
        }
        report = {"summary": summary, "details": details}
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return summary, pd.DataFrame(details)

    run_eval = st.button("평가 실행", type="primary")
    if run_eval:
        with st.spinner("평가 중..."):
            s, df = evaluate_goldenset_gui(csv_path, threshold)
            if s is not None:
                st.success("평가 완료")
                st.json(s)
            if df is not None:
                st.markdown("#### 세부 결과")
                st.dataframe(df, use_container_width=True)

# ----------- Tab 3: RAGAS (Amnesty QA) -----------
with tabs[2]:
    st.subheader("RAGAS (Amnesty QA, Ollama 필요)")
    st.caption("llama3 모델이 Ollama에 설치되어 있어야 합니다. (ollama pull llama3)")

    run_ragas = st.button("RAGAS 평가 실행", type="primary")
    if run_ragas:
        try:
            from datasets import load_dataset
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
            )
            from ragas import evaluate
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
        except Exception as e:
            st.error(f"패키지 임포트 실패: {e}")
            st.info("pip install -U datasets ragas langchain langchain-community")
        else:
            try:
                ds = load_dataset("explodinggradients/amnesty_qa", "english_v2")
            except Exception as e1:
                st.warning(f"기본 로드 실패: {e1} → trust_remote_code=True로 재시도")
                try:
                    ds = load_dataset("explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True)
                except Exception as e2:
                    st.error(f"데이터셋 로드 실패: {e2}")
                    ds = None

            if ds is not None:
                subset = ds["eval"].select(range(2))
                try:
                    llm = ChatOllama(model="llama3")
                    embeddings = OllamaEmbeddings(model="llama3")
                    res = evaluate(
                        subset,
                        metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
                        llm=llm,
                        embeddings=embeddings,
                    )
                    st.success("RAGAS 평가 완료")
                    st.json(res)
                except Exception as e:
                    st.error(f"RAGAS 평가 실패: {e}")
