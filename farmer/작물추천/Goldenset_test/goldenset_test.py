
# - 실행 시 메뉴에서 1) 채팅  2) 평가 선택
# - 두 모드 모두 LLM 참고 컨텍스트(원문 RAW)를 콘솔 출력 및 used_context_log.txt에 저장
# - LLM이 참고한 문서 출처를 번호/파일명 DataFrame으로 보여줌
import os, json, re
from typing import TypedDict, Optional, Any, Dict, List, Tuple
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from datetime import datetime
from pathlib import Path 
# =========================

# ===== 사용자 요청: 경로/인자 바꾸지 않음 =====
GOLDENSET_CSV = r"C:\Rookies_project\ET_Agent\farmer\작물추천\Goldenset_test\Goldenset_test1.csv"

# === 설정 ===
# 실행 파일 위치: Goldenset_test
# 벡터스토어 위치: ../Crop Recommedations DB/faiss_pdf_db
VECTOR_DB_PATH = Path("../faiss_pdf_db")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.75"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.")

# === 벡터스토어 경로 처리 ===
BASE_DIR = Path(__file__).resolve().parent        # Goldenset_test
vector_db_dir = (BASE_DIR / VECTOR_DB_PATH).resolve()

print("[VectorDB] 로드 경로 =", vector_db_dir)
if not vector_db_dir.exists():
    raise FileNotFoundError(f"벡터스토어 경로가 존재하지 않습니다: {vector_db_dir}")

print("실행 스크립트 위치(BASE_DIR):", BASE_DIR)
print("벡터DB 경로(VECTOR_DB_PATH):", VECTOR_DB_PATH.resolve())
print("index.faiss 존재:", (VECTOR_DB_PATH / "index.faiss").exists())
print("index.pkl 존재:", (VECTOR_DB_PATH / "index.pkl").exists())

# =========================
# [로그 관리 유틸 추가 - 자동 롤링 포함]
# - 채팅(1): 선택/유지되는 활성 로그 파일에 누적 (+ 크기 초과 시 자동 롤오버)
# - 평가(2): 실행마다 신규 파일로 교체 (+ 크기 초과 시 자동 롤오버)
#   환경변수:
#     LOG_MAX_MB   (기본 10MB)  — 파일 최대 크기
#     LOG_KEEP     (기본 50개)  — 보관할 로그 파일 개수(초과분은 오래된 것 삭제)
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 환경 변수
try:
    _LOG_MAX_MB = max(1, int(os.getenv("LOG_MAX_MB", "10")))
except Exception:
    _LOG_MAX_MB = 10

try:
    _LOG_KEEP = max(1, int(os.getenv("LOG_KEEP", "50")))
except Exception:
    _LOG_KEEP = 50

# 내부 상태: 활성 로그 파일명(디렉터리 제외)
_active_log_file: Optional[str] = None

def _unique_log_name() -> str:
    """충돌 방지를 위해 타임스탬프 + 카운터로 고유 파일명 생성"""
    base = datetime.now().strftime("used_context_log_%Y%m%d_%H%M%S")
    name = f"{base}.txt"
    if not os.path.exists(os.path.join(LOG_DIR, name)):
        return name
    counter = 1
    while True:
        name = f"{base}_{counter:02d}.txt"
        if not os.path.exists(os.path.join(LOG_DIR, name)):
            return name
        counter += 1

def _default_new_log_name() -> str:
    return _unique_log_name()

def _list_all_logs_sorted_newfirst() -> List[str]:
    files = [f for f in os.listdir(LOG_DIR) if f.startswith("used_context_log_") and f.endswith(".txt")]
    files.sort(reverse=True)  # 최신 우선
    return files

def _prune_old_logs(keep: int = _LOG_KEEP) -> None:
    """최신 keep개만 남기고 오래된 로그 삭제"""
    files = _list_all_logs_sorted_newfirst()
    for f in files[keep:]:
        try:
            os.remove(os.path.join(LOG_DIR, f))
        except Exception:
            pass

def _get_active_log_path() -> str:
    """활성 로그 파일 경로(없으면 새로 생성)"""
    global _active_log_file
    if _active_log_file is None:
        _active_log_file = _default_new_log_name()
        _prune_old_logs()
    return os.path.join(LOG_DIR, _active_log_file)

def _force_new_log_file() -> str:
    """항상 새 로그 파일로 활성 파일 교체 (평가 모드/롤오버에 사용)"""
    global _active_log_file
    _active_log_file = _default_new_log_name()
    path = os.path.join(LOG_DIR, _active_log_file)
    _prune_old_logs()
    return path

def _maybe_roll_log() -> None:
    """
    활성 로그 파일이 LOG_MAX_MB를 초과하면 자동으로 새 파일로 롤오버.
    - 활성 파일이 없으면 생성만 함.
    """
    path = _get_active_log_path()
    try:
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    except Exception:
        size_bytes = 0
    if size_bytes >= (_LOG_MAX_MB * 1024 * 1024):
        new_path = _force_new_log_file()
        print(f"[로그 롤오버] 최대 크기 {_LOG_MAX_MB}MB 초과 → 새 파일로 교체: {os.path.basename(new_path)}")

def list_log_files() -> List[str]:
    """로그 파일 목록(최신순)"""
    return _list_all_logs_sorted_newfirst()

def choose_log_file(index_or_name: Any) -> str:
    """인덱스(최신=0) 또는 파일명(.txt)으로 활성 파일 선택"""
    global _active_log_file
    files = list_log_files()
    if isinstance(index_or_name, int):
        if not files:
            raise FileNotFoundError("logs 폴더에 로그 파일이 없습니다.")
        if index_or_name < 0 or index_or_name >= len(files):
            raise IndexError(f"인덱스 범위 오류: 0~{len(files)-1}")
        _active_log_file = files[index_or_name]
    else:
        name = str(index_or_name)
        if not name.endswith(".txt"):
            raise ValueError("파일명은 .txt로 끝나야 합니다.")
        path = os.path.join(LOG_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"해당 파일이 없습니다: {name}")
        _active_log_file = name
    return _active_log_file

def read_log_file(filename: Optional[str] = None) -> str:
    target = filename if filename else _active_log_file
    if target is None:
        raise FileNotFoundError("활성 로그 파일이 없습니다. 먼저 선택하거나 기록을 남기세요.")
    path = os.path.join(LOG_DIR, target)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {target}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_active_log_file() -> Optional[str]:
    return _active_log_file

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
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

# --- LangGraph 노드 ---
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

def _append_used_context_log(index: Any,
                             question: str,
                             generated_answer: str,
                             context_raw: str,
                             golden_answer: Optional[str] = None,
                             sources: Optional[List[Dict[str, Any]]] = None) -> None:
    """활성 로그 파일에 Q/A/컨텍스트 RAW와 참고 소스 기록."""
    _maybe_roll_log()  # ✅ 크기 초과 시 자동 롤오버
    log_path = _get_active_log_path()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
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
import numpy as np
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
    # ✅ 평가 모드 시작 시: 항상 새 로그 파일로 교체
    new_log = _force_new_log_file()
    print(f"[LOG] 평가용 신규 로그 파일: {os.path.basename(new_log)}")

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

        # --- 파일 로그 저장 (평가 모드도 새 파일에 누적 기록) ---
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
    # ✅ 채팅 모드: 기존 로그 파일을 선택해 계속 누적할 수 있게 안내
    files = list_log_files()
    if files:
        print("\n[로그] 기존 파일 목록 (최신순, 인덱스 선택 가능):")
        for i, name in enumerate(files):
            print(f"  {i}: {name}")
        sel = input("기존 로그 사용? 인덱스 입력(엔터=새 파일): ").strip()
        if sel:
            try:
                choose_log_file(int(sel))
                print(f"[로그] 활성 파일: {get_active_log_file()}")
            except Exception as e:
                print(f"[로그] 선택 실패 → 새 파일 사용: {e}")
                _force_new_log_file()
                print(f"[로그] 활성 파일: {get_active_log_file()}")
        else:
            _force_new_log_file()
            print(f"[로그] 신규 파일 생성: {get_active_log_file()}")
    else:
        _force_new_log_file()
        print(f"[로그] 신규 파일 생성: {get_active_log_file()}")

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

            print("답변:")
            if a:
                for paragraph in a.strip().split("\n"):
                    if paragraph.strip():
                        print(f"\n{paragraph.strip()}")
            else:
                print("(답변 없음)")
            print("\n--- ⬇ LLM에 전달된 컨텍스트(원문 RAW, 전체) ⬇ ---")
            print(raw_ctx if raw_ctx else "(컨텍스트 없음)")
            print("--- ⬆ 컨텍스트 끝 ⬆ ---\n")

            # 참고 소스 DataFrame(번호/파일명)
            src_df = _sources_to_dataframe(srcs)
            if not src_df.empty:
                print("=== 참고 소스(번호/파일명) ===")
                print(src_df.to_string(index=False))
                print()

            # ✅ 채팅 모드만 로그 누적 기록 (자동 롤링 포함)
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
