# run_graph.py  (채팅/평가 겸용)
# - 실행 시 메뉴에서 1) 채팅  2) 평가 선택
# - 두 모드 모두 LLM 참고 컨텍스트(원문 RAW)를 콘솔 출력 및 used_context_log.txt에 저장
# - LLM이 참고한 문서 출처를 번호/파일명 DataFrame으로 보여줌
import os, json, re
from typing import TypedDict, Optional, Any, Dict, List, Tuple
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ===== 사용자 요청: 경로/인자 바꾸지 않음 =====
GOLDENSET_CSV = r"C:\Rookies_project\ET_Agent\ET_Agent-dev-eejang\farmer\작물추천\Goldenset_test\Goldenset_test1.csv"

# === 설정 ===
VECTOR_DB_PATH   = os.getenv("VECTOR_DB_PATH", "faiss_pdf_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED   = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.75"))

if not GROQ_API_KEY=REDACTED ValueError("GROQ_API_KEY=REDACTED에 설정되어야 합니다.")

# === LangChain / LangGraph ===
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# 평가용
import numpy as np
import pandas as pd

# --- 프롬프트 ---
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

# --- 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    vectorstore: Optional[Any]
    context: Optional[str]
    answer: Optional[str]
    # LLM이 참고한 문서들의 메타(파일/페이지 등)
    sources: Optional[List[Dict[str, Any]]]

# --- 공통 함수 ---
def load_vectorstore(db_path: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_with_meta(vs: Any, question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """컨텍스트(문서 원문)와 메타데이터(파일 경로 등)를 함께 반환 — 절대 자르지 않음"""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question) or []
    ctx = "\n\n".join([d.page_content for d in docs])  # 원문 그대로 합침

    sources: List[Dict[str, Any]] = []
    for rank, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src  = meta.get("source") or meta.get("file_path") or meta.get("path")
        page = meta.get("page")
        sources.append({"rank": rank, "source": src, "page": page, "metadata": meta})
    return ctx, sources

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY=REDACTED LangGraph 노드 ---
def load_vs_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 벡터스토어 로드")
    vs = load_vectorstore(VECTOR_DB_PATH)
    return {**state, "vectorstore": vs}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 검색")
    if not state.get("vectorstore"):
        raise ValueError("vectorstore가 없습니다.")
    q = state["question"] or ""
    ctx, sources = retrieve_with_meta(state["vectorstore"], q, k=5)
    return {**state, "context": ctx, "sources": sources}

def generate_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 생성")
    if not state.get("context") or not state.get("question"):
        raise ValueError("context/question 누락")
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"]})
    return {**state, "answer": ans}

# --- 그래프 빌드 ---
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

# =======================
# CSV 읽기 & 스키마 변환
# =======================
def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] CSV 인코딩 감지: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    # 최후: 무시
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    print("[WARN] utf-8(errors=ignore) 강제 사용")
    from io import StringIO
    return pd.read_csv(StringIO(data))

def _ensure_qa_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼 공백 제거
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if {"question", "answer"}.issubset(df.columns):
        return df[["question", "answer"]]

    # 사용자가 주신 형식으로 자동 변환: 제목/요약글/서브제목i/서브내용i
    if "제목" not in df.columns:
        raise ValueError(f"CSV에 question/answer 컬럼이 없고, '제목' 컬럼도 없습니다. 현재 컬럼: {list(df.columns)}")

    def norm(x):
        return "" if pd.isna(x) else str(x).strip()

    def build_answer(row):
        parts = []
        if "요약글" in df.columns:
            y = norm(row.get("요약글"))
            if y:
                parts.append(y)
        # 서브제목/서브내용 1..10
        for i in range(1, 11):
            sj = f"서브제목{i}"
            sn1 = f"서브내용{i}"
            sn2 = f" 서브내용{i}"  # 공백 붙은 케이스
            title = norm(row.get(sj)) if sj in df.columns else ""
            body  = norm(row.get(sn1)) if sn1 in df.columns else norm(row.get(sn2)) if sn2 in df.columns else ""
            if title or body:
                parts.append(f"{title}. {body}".strip(". ").strip())
        return "\n".join([p for p in parts if p])

    out = pd.DataFrame({
        "question": df["제목"].astype(str).str.strip(),
        "answer": df.apply(build_answer, axis=1)
    })
    return out

# =======================
# (선택) 보기 좋게 다듬는 함수 — 현재 출력/로그에는 사용하지 않음
# =======================
def _pretty_context(text: str) -> str:
    """PDF 추출 잡음 제거하고 문단을 보기 좋게 다듬는다. (참고용)"""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    lines = [ln.strip() for ln in t.split("\n")]
    cleaned, seen = [], set()
    for ln in lines:
        if not ln:
            continue
        if re.match(r"^\s*<[^>]+>\s*$", ln):  # 순수 태그
            continue
        if re.match(r"^\d{1,4}\s*$", ln):     # 단독 숫자 라인
            continue
        if "농업기술길잡이" in ln:            # 머릿글/꼬릿글
            continue
        if re.match(r"^\s*(<\s*)?(표|그림)\s*\d+([\-.]\d+)*", ln):  # 캡션류
            continue
        if "|" in ln and len(ln) <= 80:       # 메타 헤더
            continue
        ln = re.sub(r"\s{2,}", " ", ln)
        if ln in seen:
            continue
        seen.add(ln)
        cleaned.append(ln)
    out = "\n".join(cleaned)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out).strip()
    return out

# =======================
# 공용: 소스 → DataFrame & 로그
# =======================
def _sources_to_dataframe(sources: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for s in (sources or []):
        src = s.get("source")
        fname = os.path.basename(src) if src else "unknown"
        rows.append({"번호": s.get("rank"), "파일명": fname})
    return pd.DataFrame(rows, columns=["번호", "파일명"])

def _append_used_context_log(index: Any, question: str, generated_answer: str, context_raw: str,
                             golden_answer: Optional[str] = None,
                             sources: Optional[List[Dict[str, Any]]] = None) -> None:
    """used_context_log.txt에 Q/A/컨텍스트(원문 RAW)와 참고 소스 기록"""
    try:
        with open("used_context_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[#{index}] ------------------------------------------------\n")
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
        print(f"[경고] 컨텍스트 로그 저장 실패: {e}")

# =======================
# 🔥 골든셋 평가 함수
# =======================
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

def evaluate_goldenset(app, csv_path: str, threshold: float = 0.75, out_path: str = "evaluation_report.json") -> None:
    # 1) 데이터 로드(+스키마 보정)
    raw = _read_csv_any(csv_path)
    df = _ensure_qa_columns(raw)

    # (옵션) 평가 샘플 제한: .env에 EVAL_SAMPLE_LIMIT=5 → 상위 N개만 평가
    try:
        sample_limit = int(os.getenv("EVAL_SAMPLE_LIMIT", "0"))
    except ValueError:
        sample_limit = 0

    original_total = len(df)
    if sample_limit > 0 and sample_limit < original_total:
        df = df.head(sample_limit)

    rows = df.to_dict("records")
    total = len(rows)

    print("\n=== 골든셋 평가 시작 ===")
    if sample_limit > 0:
        print(f"파일: {csv_path} | 샘플 수: {total} (원본 {original_total}, 제한 {sample_limit}) | 임계값: {threshold}\n")
    else:
        print(f"파일: {csv_path} | 샘플 수: {total} | 임계값: {threshold}\n")

    # 2) 평가 임베딩 로더 + 벡터스토어
    eval_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    try:
        vs = load_vectorstore(VECTOR_DB_PATH)
    except Exception as e:
        print(f"⚠️ 벡터스토어 로드 실패 → 검색 유사도는 0으로 처리합니다. 원인: {e}")
        vs = None

    # 3) 루프
    results, correct = [], 0
    for i, row in enumerate(rows, start=1):
        q = str(row["question"])
        g = str(row["answer"])
        try:
            state = app.invoke({"question": q})
            a = state.get("answer", "")
        except Exception as e:
            print(f"[{i}/{total}] ❌ 그래프 오류: {e}")
            state = {}
            a = ""

        # 컨텍스트(원문 RAW) & 소스 목록
        raw_ctx = state.get("context", "") if isinstance(state, dict) else ""
        srcs = state.get("sources") or []

        # --- 콘솔 출력 ---
        print(f"[{i}/{total}] ─────────────────────────────────────────")
        print(f"골든셋 Q: {q}")
        print(f"골든셋 답변 A: {g}")
        print(f"LLM 답변: {a}\n")

        print("--- ⬇ LLM에 전달된 컨텍스트(원문 RAW, 전체) ⬇ ---")
        print(raw_ctx if raw_ctx else "(컨텍스트 없음)")
        print("--- ⬆ 컨텍스트 끝 ⬆ ---\n")

        # 참고 소스 DataFrame(번호/파일명)
        src_df = _sources_to_dataframe(srcs)
        if not src_df.empty:
            print("=== 참고 소스(번호/파일명) ===")
            print(src_df.to_string(index=False))
            print()

        # --- 파일 로그 저장 ---
        _append_used_context_log(index=i, question=q, golden_answer=g, generated_answer=a, context_raw=raw_ctx, sources=srcs)

        # 유사도(골든↔생성)
        try:
            g_vec = eval_emb.embed_query(g)
            a_vec = eval_emb.embed_query(a)
            ans_sim = _cosine(g_vec, a_vec)
        except Exception as e:
            print(f"[{i}/{total}] ❌ 임베딩 오류: {e}")
            ans_sim = 0.0

        # 검색 유사도(옵션)
        if vs is not None:
            srch_sim = _search_similarity_max(vs, eval_emb, q, k=20)
        else:
            srch_sim = 0.0

        ok = bool(ans_sim >= threshold)
        correct += int(ok)

        print(f"평가결과: {'✅' if ok else '❌'}  sim={ans_sim:.4f}  search_sim_max={srch_sim:.4f}\n")

        results.append({
            "index": i,
            "question": q,
            "golden_answer": g,
            "generated_answer": a,
            "answer_similarity": float(ans_sim),
            "search_similarity_max": float(srch_sim),
            "is_correct": ok
        })

    acc = (correct / total) * 100 if total else 0.0
    print("=== 평가 완료 ===")
    print(f"정답 수: {correct}/{total}  정확도: {acc:.2f}%")

    # 4) JSON 저장
    report = {
        "summary": {
            "original_total": int(original_total),
            "total": int(total),
            "correct": int(correct),
            "accuracy": round(acc, 2),
            "threshold": threshold,
            "embedding_model": EMBED_MODEL_NAME,
            "sample_limit": int(sample_limit),
        },
        "details": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {os.path.abspath(out_path)}")

# =======================
# 채팅 모드 (골든셋 정보 비노출)
# =======================
def chat(app) -> None:
    """
    콘솔 채팅:
    - 사용자 질문 입력 → LLM 답변 출력
    - LLM이 참고한 컨텍스트(원문 RAW, 전체)와 참고 소스(번호/파일명) 표시
    - used_context_log.txt에 동일 포맷으로 저장 (골든셋 정보 비기록)
    종료: 빈 줄 입력 또는 'exit'/'quit'
    """
    print("\n=== 채팅 모드 시작 ===")
    print("질문을 입력하세요. (종료: 빈 줄 또는 exit/quit)\n")
    turn = 1
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[채팅 종료]")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("[채팅 종료]")
            break

        try:
            state = app.invoke({"question": q})
            a = state.get("answer", "")
            raw_ctx = state.get("context", "") if isinstance(state, dict) else ""
            srcs = state.get("sources") or []

            print("\n답변:")
            print(a if a else "(답변 없음)")
            print("\n--- ⬇ LLM에 전달된 컨텍스트(원문 RAW, 전체) ⬇ ---")
            print(raw_ctx if raw_ctx else "(컨텍스트 없음)")
            print("--- ⬆ 컨텍스트 끝 ⬆ ---\n")

            # 참고 소스 DataFrame(번호/파일명)
            src_df = _sources_to_dataframe(srcs)
            if not src_df.empty:
                print("=== 참고 소스(번호/파일명) ===")
                print(src_df.to_string(index=False))
                print()

            # 로그 저장 (골든셋 답변은 기록하지 않음)
            _append_used_context_log(index=f"chat-{turn}", question=q, generated_answer=a, context_raw=raw_ctx, sources=srcs)
            turn += 1
        except Exception as e:
            print(f"[오류] {e}")

# =======================
# 메인
# =======================
def main():
    print("==== 모드 선택 ====")
    print("1) 채팅")
    print("2) 평가")
    choice = input("번호를 입력하세요 [1/2]: ").strip()

    app = build_graph()

    if choice == "1":
        chat(app)
    elif choice == "2":
        print(f"[RUN] 골든셋 CSV: {GOLDEN셋_CSV}")
        evaluate_goldenset(app, csv_path=GOLDEN셋_CSV, threshold=SIM_THRESHOLD, out_path="evaluation_report.json")
    else:
        print("올바르지 않은 선택입니다. 기본값(평가)으로 실행합니다.")
        print(f"[RUN] 골든셋 CSV: {GOLDEN셋_CSV}")
        evaluate_goldenset(app, csv_path=GOLDEN셋_CSV, threshold=SIM_THRESHOLD, out_path="evaluation_report.json")

if __name__ == "__main__":
    # 변수명 오타 방지: 위 main()에서 사용할 상수 이름 교정
    GOLDEN셋_CSV = GOLDENSET_CSV
    main()
