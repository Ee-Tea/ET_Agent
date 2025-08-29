# (Milvus + Web Search Agent + Ragas Evaluation)
# ------------------------------------------------------------
# 1) Chat (Milvus + Tavily Web Search)
# 2) Golden-set Evaluation (custom similarity-based eval)
# 3) Ragas Evaluation (Amnesty QA dataset with Ollama)
# ------------------------------------------------------------
import os
import json
import re
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import numpy as np
import pandas as pd
import time
from numpy.linalg import norm  # ✅ 벡터 코사인 유사도용

# --- 환경 변수 로드 ---
load_dotenv(find_dotenv())

# --- 설정 ---
GOLDENSET_CSV = r"C:\Rookies_project\ET_Agent\farmer\작물추천\Goldenset_test\Goldenset_test1.csv"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hongyoungjun")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.75"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# 🔧 Milvus score → similarity 변환 관련 환경변수
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "cosine").lower()  # cosine | l2 | ip
MILVUS_SCORE_IS_DISTANCE = os.getenv("MILVUS_SCORE_IS_DISTANCE", "true").lower() == "true"
MILVUS_IP_RESCALE_01 = os.getenv("MILVUS_IP_RESCALE_01", "true").lower() == "true"

# --- 필수 API 키 확인 ---
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.")
if not TAVILY_API_KEY:
    print("⚠️ 경고: TAVILY_API_KEY가 .env에 설정되지 않았습니다. 웹 검색 기능이 비활성화됩니다.")

# --- 라이브러리 임포트 ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pymilvus import connections

# --- 상수/헬퍼 ---
NO_INFO_PHRASE = "주어진 정보로는 답변할 수 없습니다"

def normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    t = t.replace(".", "").strip()
    return t

# --- 프롬프트 정의 ---
RAG_PROMPT_TMPL = """당신은 대한민국 농업 작물 재배 방법 전문가입니다. 아래 '문맥'만 사용해 질문에 답하세요.
[문맥]: {context}
규칙:
- 문맥에 없는 정보/추측 금지. 한글로만 작성.
- 문맥에 근거 없으면: "주어진 정보로는 답변할 수 없습니다."라고만 답하세요.
질문: {question}
답변:"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

WEB_PROMPT_TMPL = """당신은 대한민국 농업 작물 재배 방법 전문가입니다. 아래 '웹 검색 결과'와 '질문'만을 근거로 답변을 생성하세요.
[웹 검색 결과]:
{search_results}

규칙:
- 반드시 '웹 검색 결과' 내 근거로만 답하세요. 한글로만 작성.
- 확실하지 않으면 "주어진 정보로는 답변할 수 없습니다."라고 작성하세요.
- 항상 답변 마지막에 "참고 링크:" 섹션을 만들고, 적절한 상위 1~3개 URL을 bullet로 첨부하세요.

질문: {question}
답변:"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL)

# --- 상태 정의 ---
class GraphState(TypedDict, total=False):
    question: Optional[str]
    # 분리된 컨텍스트
    context_internal: Optional[str]       # 내부 DB 컨텍스트
    context_web: Optional[str]            # 웹 검색 컨텍스트(정리된 결과)
    answer: Optional[str]
    web_search_results: Optional[str]
    answer_source: Optional[str]
    log_file: Optional[str]
    no_info: Optional[bool]
    force_web: Optional[bool]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    retrieval_time_ms: Optional[float]

# --- 공통 객체 ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": "cpu"})
llm = ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)
web_search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# --- 텍스트 코사인 유사도(임베딩 포함) ---
def cosine_sim(txt1: str, txt2: str) -> float:
    if not txt1 or not txt2:
        return 0.0
    v1 = np.array(embedding_model.embed_query(txt1))
    v2 = np.array(embedding_model.embed_query(txt2))
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

# 오타 방지용 별칭
def consine(txt1: str, txt2: str) -> float:
    return cosine_sim(txt1, txt2)

# --- Milvus score → similarity 변환 ---
def convert_score_to_similarity(raw: float) -> float:
    if MILVUS_METRIC == "cosine":
        sim = 1.0 - float(raw) if MILVUS_SCORE_IS_DISTANCE else float(raw)
        return max(0.0, min(1.0, sim))
    elif MILVUS_METRIC == "l2":
        d = float(raw) if MILVUS_SCORE_IS_DISTANCE else max(0.0, 1.0 - float(raw))
        sim = 1.0 / (1.0 + max(0.0, d))
        return max(0.0, min(1.0, sim))
    elif MILVUS_METRIC == "ip":
        val = -float(raw) if MILVUS_SCORE_IS_DISTANCE else float(raw)
        if MILVUS_IP_RESCALE_01:
            sim = (val + 1.0) / 2.0
            return max(0.0, min(1.0, sim))
        return val
    else:
        return max(0.0, min(1.0, float(raw)))

# --- 로그 유틸: 내부/웹 컨텍스트 분리 저장 ---
def append_log(state: GraphState):
    log_file = state.get("log_file")
    if not log_file:
        return
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": state.get("question"),
        "answer": state.get("answer"),
        "source": state.get("answer_source"),
        "context_internal": state.get("context_internal", ""),
        "context_web": state.get("context_web", ""),
    }
    history = []
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    history.append(log_entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# --- LangGraph 노드 ---
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 [1/4] Milvus DB에서 문서 검색 중...")
    question = state["question"]
    start = time.perf_counter()
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
        vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )
        docs_with_scores = vector_store.similarity_search_with_score(question, k=RETRIEVAL_K)

        retrieved_docs_dump = []
        for idx, (doc, score) in enumerate(docs_with_scores, start=1):
            sim = convert_score_to_similarity(float(score))
            full_text = doc.page_content or ""
            preview = full_text[:120].replace("\n", " ")
            print(f"  -> #{idx} raw={float(score):.6f} sim={sim:.4f} | {preview!r}")
            retrieved_docs_dump.append({
                "rank": idx,
                "raw_score": float(score),
                "similarity": float(sim),
                "metadata": getattr(doc, "metadata", {}),
                "text": full_text[:2000]
            })

        # 내부 DB 컨텍스트는 상위 k개 합침
        context_internal = "\n\n".join([d.page_content for d, _ in docs_with_scores])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"  -> {len(docs_with_scores)}개 문서 검색 완료. ({elapsed_ms:.1f} ms)")

        return {
            **state,
            "context_internal": context_internal,
            "retrieved_docs": retrieved_docs_dump,
            "retrieval_time_ms": elapsed_ms
        }
    finally:
        if "default" in connections.list_connections():
            connections.disconnect("default")

def generate_rag_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 [2/4] 내부 DB 정보로 1차 답변 생성 중...")
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state.get("context_internal", ""), "question": state["question"]})
    no_info = NO_INFO_PHRASE in normalize(answer)
    print("  -> 1차 답변 생성 완료.")
    return {**state, "answer": answer, "answer_source": "내부 DB", "no_info": no_info}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 [3/4] Tavily 웹 검색 중...")
    if not web_search_tool:
        print("  -> Tavily API 키가 없어 웹 검색을 건너뜁니다.")
        return {**state, "web_search_results": "Tavily API 키 없음", "context_web": ""}

    try:
        results = web_search_tool.invoke({"query": state["question"]}) or []
    except Exception as e:
        print(f"  -> Tavily 호출 오류: {e}")
        results = []

    lines = []
    for idx, r in enumerate(results, start=1):
        title = r.get("title") or r.get("url") or f"결과 {idx}"
        url = r.get("url") or ""
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:500]
        lines.append(f"- 제목: {title}\n  URL: {url}\n  요약: {snippet}")

    if not lines:
        search_results = "- 검색 결과가 없습니다."
        print("  -> 0개 웹 검색 결과 확인.")
    else:
        search_results = "\n".join(lines)
        print(f"  -> {len(results)}개 웹 검색 결과 확인.")

    # 웹 컨텍스트는 사람이 읽을 수 있는 요약 문자열(동일)
    return {**state, "web_search_results": search_results, "context_web": search_results}

def generate_web_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 [4/4] 웹 검색 정보로 2차 답변 생성 중...")
    chain = web_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": state["question"],
        "search_results": state.get("web_search_results", "- 검색 결과가 없습니다.")
    })
    print("  -> 2차 답변 생성 완료.")
    return {**state, "answer": answer, "answer_source": "웹 검색", "no_info": NO_INFO_PHRASE in normalize(answer)}

# --- 조건부 라우팅 ---
def should_web_search(state: GraphState) -> str:
    print("🧭 웹 검색 필요 여부 판단 중...")
    answer = state.get("answer", "")
    question = state.get("question", "")
    context_internal = state.get("context_internal", "") or ""
    no_info = state.get("no_info", False)

    if not TAVILY_API_KEY:
        print("  -> 결정: 웹 검색 안 함 (TAVILY_API_KEY 없음)")
        return "end"

    if state.get("force_web"):
        print("  -> 결정: 웹 검색 수행 (force_web)")
        return "continue"

    if no_info or (NO_INFO_PHRASE in normalize(answer)):
        print("  -> 결정: 웹 검색 수행 (내부 DB에 정보 없음)")
        return "continue"

    time_pattern = r"(최신|오늘|뉴스|가격|현재|최근|변경|개정|발표|공지|실시간|시세|업데이트)"
    if re.search(time_pattern, question):
        print("  -> 결정: 웹 검색 수행 (시간 민감 질문)")
        return "continue"

    if len(normalize(context_internal)) < 30:
        print("  -> 결정: 웹 검색 수행 (문맥 빈약)")
        return "continue"

    print("  -> 결정: 웹 검색 안 함 (내부 DB 정보로 충분)")
    return "end"

# --- 그래프 빌드 ---
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_rag", generate_rag_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_web", generate_web_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate_rag")
    g.add_conditional_edges(
        "generate_rag",
        should_web_search,
        {"continue": "web_search", "end": END}
    )
    g.add_edge("web_search", "generate_web")
    g.add_edge("generate_web", END)

    return g.compile()

# =======================
# 모드 1: 채팅
# =======================
def chat(app, log_file: str):
    print("\n=== 채팅 모드를 시작합니다. (종료: 빈 줄 Enter 또는 'exit') ===")
    while True:
        q = input("\n질문> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            final_state = app.invoke({"question": q, "log_file": log_file})
            answer = final_state.get("answer", "답변을 생성하지 못했습니다.")
            source = final_state.get("answer_source", "알 수 없음")
            print(f"\n🤖 답변 (출처: {source}):\n{answer}")
            append_log(final_state)
        except Exception as e:
            print(f"[오류] {e}")
    print("채팅을 종료합니다.")

# =======================
# 모드 2: 골든셋 평가
# =======================
def _calc_search_similarity_max_from_milvus(retrieved_docs: Optional[List[Dict[str, Any]]]) -> float:
    if not retrieved_docs:
        return 0.0
    sims = [float(x.get("similarity", 0.0)) for x in retrieved_docs]
    return max(sims) if sims else 0.0

def _calc_search_similarity_max_fallback(question: str, retrieved_docs: Optional[List[Dict[str, Any]]]) -> float:
    if not retrieved_docs:
        return 0.0
    sims = []
    for item in retrieved_docs:
        txt = item.get("text", "") or ""
        sims.append(cosine_sim(question, txt))
    return max(sims) if sims else 0.0

def evaluate_goldenset(app, csv_path: str, log_file: str):
    """
    골든셋 평가:
    - answer_similarity: golden_answer ↔ generated_answer 코사인 유사도 (벡터 기반)
    - search_similarity_max: Milvus 검색 유사도(변환값) 최댓값(없으면 질문↔문서 코사인 최대값)
    - sim / similarity: 최종 사용 컨텍스트 ↔ generated_answer 코사인 유사도 (벡터 기반)
      * 최종 출처가 '내부 DB'면 context_internal, '웹 검색'이면 context_web 사용
    - source: 최종 답변 출처 ('내부 DB' | '웹 검색')
    """
    # 이 함수 안에서만 쓰는 벡터 코사인 유사도
    def cosine_similarity(vec1, vec2):
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        denom = (norm(v1) * norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="cp949")

    rows = df.to_dict("records")
    total = len(rows)
    print(f"\n=== 골든셋 평가 시작: {total}개 샘플 | 임계값: {SIM_THRESHOLD} ===")

    results, correct = [], 0

    for i, row in enumerate(rows, start=1):
        q, g = str(row.get("question", "")), str(row.get("answer", ""))
        if not q or not g:
            continue

        print(f"\n[{i}/{total}] 질문 처리 중")
        print(f"질문: {q}")

        # 1차 시도
        used_second = False
        try:
            state1 = app.invoke({"question": q, "log_file": log_file})
            a1 = state1.get("answer", "")
            append_log(state1)
        except Exception as e:
            print(f"  -> ❌ 그래프 실행 오류(1차): {e}")
            state1, a1 = {}, ""

        # 벡터 임베딩 후 코사인 (golden ↔ generated)
        golden_vec_1 = embedding_model.embed_query(g) if g else []
        gen_vec_1 = embedding_model.embed_query(a1) if a1 else []
        sim1 = cosine_similarity(golden_vec_1, gen_vec_1)

        src1 = state1.get("answer_source", "N/A")
        no_info1 = state1.get("no_info", False)

        chosen_answer = a1
        chosen_sim = sim1
        chosen_src = src1
        chosen_state = state1

        # 2차(웹 강제)
        need_second_try = (src1 == "내부 DB") and (no_info1 or (sim1 < SIM_THRESHOLD))
        if need_second_try and TAVILY_API_KEY:
            print("  -> 2차 시도: 웹검색 강제 실행(force_web=True)")
            try:
                state2 = app.invoke({"question": q, "log_file": log_file, "force_web": True})
                a2 = state2.get("answer", "")
                append_log(state2)

                golden_vec_2 = golden_vec_1 or (embedding_model.embed_query(g) if g else [])
                gen_vec_2 = embedding_model.embed_query(a2) if a2 else []
                sim2 = cosine_similarity(golden_vec_2, gen_vec_2)

                if sim2 >= chosen_sim:
                    chosen_answer = a2
                    chosen_sim = sim2
                    chosen_src = state2.get("answer_source", "N/A")
                    chosen_state = state2
                    used_second = True
            except Exception as e:
                print(f"  -> ❌ 그래프 실행 오류(2차): {e}")

        # 검색 유사도 최대값
        search_similarity_max = _calc_search_similarity_max_from_milvus(chosen_state.get("retrieved_docs"))
        if search_similarity_max == 0.0:
            search_similarity_max = _calc_search_similarity_max_fallback(q, chosen_state.get("retrieved_docs"))

        # sim: (최종 컨텍스트 ↔ 최종 답변)
        ctx_int = chosen_state.get("context_internal", "") or ""
        ctx_web = chosen_state.get("context_web", "") or ""
        if chosen_src == "내부 DB":
            chosen_context = ctx_int
        elif chosen_src == "웹 검색":
            chosen_context = ctx_web
        else:
            chosen_context = ctx_int or ctx_web

        ctx_vec = embedding_model.embed_query(chosen_context) if chosen_context else []
        ans_vec = embedding_model.embed_query(chosen_answer) if chosen_answer else []
        sim_ctx_ans = cosine_similarity(ctx_vec, ans_vec)

        ok = chosen_sim >= SIM_THRESHOLD
        if ok:
            correct += 1

        print(f"  -> answer_similarity: {chosen_sim:.4f} | search_similarity_max: {search_similarity_max:.4f}")
        print(f"  -> sim(context↔answer): {sim_ctx_ans:.4f}")
        print(f"  -> 평가: {'✅ OK' if ok else '❌ FAIL'} | 출처: {chosen_src}{' (2차)' if used_second else ''}")

        # JSON record (✅ similarity & source 포함)
        results.append({
            "index": i,
            "question": q,
            "golden_answer": g,
            "generated_answer": chosen_answer,
            "answer_similarity": float(chosen_sim),           # golden ↔ generated
            "search_similarity_max": float(search_similarity_max),
            "sim": float(sim_ctx_ans),                        # 유지
            "similarity": float(sim_ctx_ans),                 # 요청: 동일값으로 표기
            "source": chosen_src,                             # 요청: '내부 DB' | '웹 검색'
            "is_correct": bool(ok)
        })

    acc = (correct / total) * 100 if total else 0.0
    print(f"\n=== 평가 완료 ===\n정확도: {acc:.2f}% ({correct}/{total})")

    report_path = "goldenset_evaluation_report.json"
    report = {"summary": {"total": total, "correct": correct, "accuracy": acc},
              "details": results}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"결과 리포트 저장: {os.path.abspath(report_path)}")

# =======================
# (옵션) 별도 간단 평가 유틸들
# =======================
def load_golden_dataset(file_path: str):
    """골든 데이터셋을 CSV 파일에서 로드하여 리스트로 반환합니다."""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df.to_dict('records')

def evaluate_chatbot(app, golden_dataset: List[Dict[str, str]]):
    """골든 데이터셋을 사용하여 간단 챗봇 성능 평가(JSON에는 similarity & source 포함)."""
    print("\n--- 챗봇 성능 평가 시작 (유사도 기반) ---")
    evaluation_results = []
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    SIMILARITY_THRESHOLD = 0.8

    for i, data in enumerate(golden_dataset):
        question = data['question']
        golden_answer = data['answer']

        print(f"\n[평가 {i+1}] 질문: {question}")
        print(f"  - 정답: {golden_answer}")

        try:
            final_state = app.invoke({"question": question})
            generated_answer = final_state.get('answer', '')
            src = final_state.get('answer_source', '')
            ctx_int = final_state.get('context_internal', '') or ''
            ctx_web = final_state.get('context_web', '') or ''
            chosen_context = ctx_int if src == "내부 DB" else (ctx_web if src == "웹 검색" else (ctx_int or ctx_web))

            # answer_similarity (golden ↔ generated) : 텍스트 임베딩 코사인
            g_vec = embeddings.embed_query(golden_answer) if golden_answer else []
            a_vec = embeddings.embed_query(generated_answer) if generated_answer else []
            denom = (norm(np.asarray(g_vec)) * norm(np.asarray(a_vec)))
            answer_similarity = float(np.dot(g_vec, a_vec) / denom) if denom else 0.0

            # similarity(sim): 최종 컨텍스트 ↔ 답변
            c_vec = embeddings.embed_query(chosen_context) if chosen_context else []
            denom2 = (norm(np.asarray(c_vec)) * norm(np.asarray(a_vec)))
            sim_ctx_ans = float(np.dot(c_vec, a_vec) / denom2) if denom2 else 0.0

            is_correct = bool(answer_similarity >= SIMILARITY_THRESHOLD)

            print(f"  - answer_similarity: {answer_similarity:.4f} (기준: {SIMILARITY_THRESHOLD})")
            print(f"  - similarity(context↔answer): {sim_ctx_ans:.4f}")
            print(f"  - 출처: {src}")
            print(f"  - 정답 여부: {'✅ 정답' if is_correct else '❌ 오답'}")

            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': generated_answer,
                'answer_similarity': float(answer_similarity),
                'similarity': float(sim_ctx_ans),   # JSON에 표기
                'source': src,                      # JSON에 표기
                'is_correct': is_correct
            })

        except Exception as e:
            print(f"  - 오류 발생: {e}")
            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': '오류 발생',
                'answer_similarity': 0.0,
                'similarity': 0.0,
                'source': '',
                'is_correct': False
            })

    print("\n--- 챗봇 성능 평가 완료 ---")
    total_questions = len(evaluation_results)
    correct_answers = sum(1 for res in evaluation_results if res['is_correct'])
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    print(f"\n총 질문 수: {total_questions}")
    print(f"정답 수: {correct_answers}")
    print(f"정확도: {accuracy:.2f}%")

    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {"total": total_questions, "correct": correct_answers, "accuracy": accuracy},
            "details": evaluation_results
        }, f, ensure_ascii=False, indent=4)
    print("상세 평가 결과가 'evaluation_report.json' 파일에 저장되었습니다.")

# =======================
# 모드 3: Ragas 평가
# =======================
def run_ragas_evaluation():
    print("\n=== Ragas 평가 모드를 시작합니다 ===")
    print("필요한 라이브러리를 로드합니다...")
    try:
        from datasets import load_dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError as e:
        print(f"오류: Ragas 실행에 필요한 라이브러리가 없습니다. ({e})")
        print("pip install -U datasets ragas langchain-community 를 실행하여 설치하세요.")
        return

    print("Amnesty QA 데이터셋을 로드합니다 (샘플 2개)...")
    try:
        amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
        amnesty_subset = amnesty_qa["eval"].select(range(2))
    except Exception as e:
        print(f"데이터셋 로드에 실패했습니다: {e}")
        return

    print("Ollama 모델을 초기화합니다 (Ollama 서버 실행 및 llama3 모델 필요)...")
    try:
        llm = ChatOllama(model="llama3")
        embeddings = OllamaEmbeddings(model="llama3")
        llm.invoke("Hi")
    except Exception as e:
        print(f"Ollama 모델 초기화 또는 연결에 실패했습니다: {e}")
        print("Ollama가 실행 중이고, 'ollama pull llama3' 명령어로 모델을 설치했는지 확인하세요.")
        return

    print("Ragas 평가를 시작합니다. 다소 시간이 걸릴 수 있습니다...")
    try:
        result = evaluate(
            amnesty_subset,
            metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        print("\n=== Ragas 평가 결과 ===")
        print(result)
    except Exception as e:
        print(f"Ragas 평가 중 오류가 발생했습니다: {e}")

# =======================
# 메인 실행 함수
# =======================
def main():
    print("="*50)
    print("Milvus & 웹 검색 에이전트")
    print("="*50)
    print("1) 채팅 모드")
    print("2) 골든셋 평가 모드 (자체 유사도 기반)")
    print("3) Ragas 평가 모드 (Amnesty QA 데이터셋, Ollama 필요)")
    choice = input("번호를 입력하세요 [1/2/3]: ").strip()

    if choice in ("1", "2"):
        log_dir = "agent_logs"
        os.makedirs(log_dir, exist_ok=True)
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"log_{session_ts}.json")
        print(f"이번 세션의 대화는 {log_file} 에 기록됩니다.")

        app = build_graph()

        if choice == "1":
            chat(app, log_file)
        elif choice == "2":
            evaluate_goldenset(app, GOLDENSET_CSV, log_file)
    elif choice == "3":
        run_ragas_evaluation()
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
