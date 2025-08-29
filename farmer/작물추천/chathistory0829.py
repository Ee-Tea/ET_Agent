# --- 라이브러리 설치 ---
# pip install langchain-huggingface langchain_community langchain-core langchain-groq langgraph pymilvus python-dotenv tavily-python ragas datasets

print("▶ [초기화] 라이브러리 임포트 시작...")
import os
import json
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from datetime import datetime
import re
import numpy as np  # ✅ 추가: 추출식 reference용 유사도 계산

# --- RAGAS 관련 라이브러리 임포트 ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 환경 변수 로드 ---
print("▶ [초기화] 환경 변수 로드...")
load_dotenv(find_dotenv())

# --- Milvus / Embedding 모델 설정 ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hongyoungjun")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# --- LLM 설정 ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# --- Web Search 설정 ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.")

# --- 라이브러리 임포트 ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pymilvus import connections
print("▶ [초기화] 모든 라이브러리 임포트 완료.")

# --- 프롬프트 정의 ---
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다.
아래 '문맥'만 사용해 질문에 답하세요.

[문맥]
{context}

규칙:
- 문맥에 없는 정보/추측/한자 금지.
- 한글으로만 작성.
- 단계/설명은 "한 문장씩 줄바꿈".
- 문맥에 근거 없으면: "주어진 정보로는 답변할 수 없습니다."

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

WEB_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다.
아래 '웹 검색 결과'와 '질문'을 바탕으로, 사용자가 이해하기 쉽게 답변을 종합하고 정리하여 설명해주세요.

[웹 검색 결과]
{search_results}

규칙:
- 검색 결과를 바탕으로 답변을 생성하되, 직접적인 내용이 없더라도 정보를 종합하여 최대한 유용한 답변을 만드세요.
- 내용은 명확하게 단계별로 설명해주세요.
- 검색 결과로 정말 답변이 불가능할 때만 "관련 정보를 찾을 수 없습니다."라고 답변하세요.
- [중요] 모든 답변은 반드시 한국어로 작성해야 합니다.

🟢 질문: {question}
✨ 답변:
"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL)

# (참고) 이전엔 LLM으로 합성 reference를 만들었지만, 이제 사용하지 않음.
SYNTHETIC_REF_PROMPT = ChatPromptTemplate.from_template("""
[UNUSED] 합성 reference 프롬프트 (현재는 LLM-free 추출식 reference 사용)
질문: {question}
컨텍스트: {context}
정답:
""")

# --- 상태 정의 ---
class GraphState(TypedDict, total=False):
    question: Optional[str]
    vectorstore: Optional[Milvus]
    context: Optional[str]
    answer: Optional[str]
    web_search_results: Optional[str]
    log_file: Optional[str]
    answer_source: Optional[str]
    ragas_score: Optional[float]
    rag_retry_count: Optional[int]
    original_docs: List[str]
    web_contexts: List[str]
    reference: Optional[str]
    rag_tokens: Optional[Dict[str, Any]]
    web_tokens: Optional[Dict[str, Any]]
    ragas_details: Optional[Dict[str, Any]]

# --- Embeddings 및 LLM ---
print("▶ [초기화] 임베딩 및 LLM 모델 객체 생성 시작...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)
llm = ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)
print("▶ [초기화] 임베딩 및 LLM 모델 객체 생성 완료.")

# --- 유틸: 대화 로그 저장 ---
def append_conversation_to_file(question: str, answer: str, source: str, score: Optional[float], token_usage: Optional[Dict[str, Any]], filename: str):
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s for s in sentences if s]
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": sentences,
        "source": source,
        "ragas_score": score,
        "token_usage": token_usage
    }
    if filename:
        try:
            hist = []
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, "r", encoding="utf-8") as f: hist = json.load(f)
            hist.append(data)
            with open(filename, "w", encoding="utf-8") as f: json.dump(hist, f, ensure_ascii=False, indent=4)
            print(f"       💾 대화 기록 저장 완료: {filename}")
        except Exception as e:
            print(f"       ❌ 대화 기록 저장 오류: {e}")

# --- 추출식 reference 유틸 (LLM-free) ---
def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # 마침표/물음표/느낌표/개행 기준 분할
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

def build_extractive_reference_from_contexts(contexts: List[str], question: str, embedder: HuggingFaceEmbeddings, top_k: int = 5, max_chars: int = 1800) -> str:
    """
    컨텍스트 원문에서 문장들을 추출하여, 질문-문장 임베딩 유사도 상위 top_k 문장으로 reference를 구성.
    - LLM 생성/요약 없음 (추출식, extractive)
    """
    sents: List[str] = []
    for c in contexts:
        sents.extend(_split_sentences(c))

    # 문장이 너무 많으면 앞부분 일부만 샘플링 (안정성)
    if len(sents) > 2000:
        sents = sents[:2000]

    if not sents:
        return ""

    try:
        q_emb = embedder.embed_query(question)
        s_embs = embedder.embed_documents(sents)
        q = np.array(q_emb, dtype=np.float32)
        S = np.array(s_embs, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        scores = S_norm @ q_norm  # cosine similarity
        idx = np.argsort(-scores)[:max(top_k, 1)]
        picked = [sents[i] for i in idx]
        reference = " ".join(picked)
        return reference[:max_chars]
    except Exception as e:
        print(f"       ⚠️ 추출식 reference 생성 중 오류: {e}")
        return " ".join(contexts)[:max_chars]  # 최후의 수단: 컨텍스트 앞부분

# --- LangGraph 노드 ---
def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- 🧩 노드 시작: Milvus 벡터스토어 로드 ---")
    if "default" not in connections.list_connections() or not connections.has_connection("default"):
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    try:
        vs = Milvus(embedding_model, collection_name=MILVUS_COLLECTION, connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN})
        print(f"       ✅ Milvus 로드 완료 (컬렉션: {MILVUS_COLLECTION})")
        return {**state, "vectorstore": vs}
    except Exception as e:
        print(f"       ❌ Milvus 로드 실패: {e}")
        raise ConnectionError("Milvus 벡터스토어 로드 실패")

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 문서 검색 ---")
    question = state.get("question")
    vectorstore = state.get("vectorstore")
    if not question or not vectorstore: raise ValueError("질문 또는 벡터스토어가 누락되었습니다.")
    
    print(f"       📥 검색 질문: '{question}'")
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=2)

    context = ""
    print(f"       📄 {len(docs_with_scores)}개 문서 검색 완료.")
    for i, (doc, score) in enumerate(docs_with_scores):
        preview = (doc.page_content or "")[:100].replace("\n", " ")
        print(f"         ▶ 문서 {i+1} (점수: {score:.4f}): '{preview}...'")
        context += f"\n\n{doc.page_content}"
    print(f"       📝 생성된 컨텍스트 길이: {len(context)} 자")
    
    original_docs = [doc.page_content for doc, score in docs_with_scores]
    return {**state, "context": context, "rag_retry_count": 0, "original_docs": original_docs}

def generate_rag_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: RAG 답변 생성 ---")
    rag_retries = state.get("rag_retry_count", 0) + 1
    print(f"       🔄 RAG 생성 시도: {rag_retries}번째")

    context = state.get("context", "")
    question = state.get("question")
    if not question: raise ValueError("질문이 누락되었습니다.")
    
    print(f"       ▶ 입력 컨텍스트: '{context[:100].replace('\n', ' ')}...'")
    
    response = llm.invoke(rag_prompt.format(context=context, question=question))
    ans = response.content
    token_usage = response.response_metadata.get("token_usage", {})

    print(f"       💬 생성된 답변: '{ans[:100].replace('\n', ' ')}...'")
    print(f"       📊 토큰 사용량 (RAG): prompt={token_usage.get('prompt_tokens', 0)}, completion={token_usage.get('completion_tokens', 0)}, total={token_usage.get('total_tokens', 0)}")
    
    return {**state, "answer": ans, "answer_source": "내부 DB", "rag_retry_count": rag_retries, "rag_tokens": token_usage}

def ragas_eval_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: RAGAS 답변 평가 ---")
    question = state.get("question")
    answer = state.get("answer", "")
    source = state.get("answer_source", "N/A")

    # 컨텍스트 확보
    if source == "웹 검색":
        eval_context = state.get("web_search_results", "")
        context_source_name = "웹 검색 결과"
        contexts_for_ragas = state.get("web_contexts", [])
    else:
        eval_context = state.get("context", "")
        context_source_name = "내부 DB 문서"
        contexts_for_ragas = state.get("original_docs", [])
        if not contexts_for_ragas and eval_context:
            contexts_for_ragas = [eval_context]

    print(f"       ▶ 평가 대상 ({source}): '{answer[:100].replace('\n', ' ')}...'")
    head = (eval_context[:100] if isinstance(eval_context, str) else str(eval_context)[:100]).replace('\n', ' ')
    print(f"       ▶ 평가 기준 ({context_source_name}): '{head}...'")

    if not question or not answer or not contexts_for_ragas:
        print("       ⚠️ 평가에 필요한 정보가 부족하여 건너뜁니다.")
        state["ragas_details"] = {
            "mode": "skipped",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "context_precision": None,
            "final_score": 0.0,
            "used_reference": False,
        }
        return {**state, "ragas_score": 0.0}

    # ✅ LLM-free: 컨텍스트 원문에서 '추출식 reference' 생성 (임베딩 유사도 top-k 문장)
    print("       ✂️ 컨텍스트에서 추출식 reference 생성(LLM 생성 금지)...")
    reference = build_extractive_reference_from_contexts(
        contexts_for_ragas, question, embedding_model, top_k=5, max_chars=1800
    )
    used_reference = bool(reference)
    state["reference"] = reference if used_reference else None
    if not used_reference:
        print("       ⚠️ 추출식 reference 비어있음 → 컨텍스트 앞부분을 대체 사용")
        reference = " ".join(contexts_for_ragas)[:1800]
        used_reference = bool(reference)
        state["reference"] = reference if used_reference else None

    # 데이터셋 구성
    dataset_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts_for_ragas],
        "reference": [reference],  # 4지표 필수
    }
    dataset = Dataset.from_dict(dataset_dict)

    # 유틸
    def harmonic_mean(vals: List[Optional[float]]) -> float:
        xs = [float(v) for v in vals if v is not None and v > 0.0]
        if not xs: return 0.0
        return len(xs) / sum(1.0 / v for v in xs)

    def to_df(result):
        if hasattr(result, "to_pandas"):
            return result.to_pandas()
        import pandas as pd
        try:
            return pd.DataFrame(result)
        except Exception:
            return pd.DataFrame()

    # 4지표 평가
    f = ar = cr = cp = None
    final_score = 0.0
    try:
        metrics_all = [faithfulness, answer_relevancy, context_recall, context_precision]
        result = evaluate(
            dataset,
            metrics=metrics_all,
            llm=llm,
            embeddings=embedding_model,
            raise_exceptions=False,
        )
        df = to_df(result)

        try:
            if "faithfulness" in df.columns:
                f = float(df["faithfulness"].iloc[0])
            if "answer_relevancy" in df.columns:
                ar = float(df["answer_relevancy"].iloc[0])
            if "context_recall" in df.columns:
                cr = float(df["context_recall"].iloc[0])
            if "context_precision" in df.columns:
                cp = float(df["context_precision"].iloc[0])
        except Exception as ex:
            print(f"       ⚠️ 점수 파싱 중 오류: {ex}")

        final_score = harmonic_mean([f, ar, cr, cp])

        def fmt(x):
            return "N/A" if x is None else f"{x:.4f}"
        print("\n--- 📊 RAGAS 평가 상세 점수 ---")
        print(f"    🟢 충실도 (Faithfulness):           {fmt(f)}")
        print(f"    🟢 관련성 (Answer Relevancy):       {fmt(ar)}")
        print(f"    🟢 문맥 재현율 (Context Recall):    {fmt(cr)}")
        print(f"    🟢 문맥 정밀도 (Context Precision): {fmt(cp)}")
        print(f"        📊 최종 RAGAS 조화평균 점수:     {final_score:.4f}")

    except Exception as e:
        print(f"       ❌ RAGAS 평가 중 오류: {e}")
        final_score = 0.0

    state["ragas_details"] = {
        "mode": "4-metrics" if all(v is not None for v in [f, ar, cr, cp]) else "partial",
        "faithfulness": f,
        "answer_relevancy": ar,
        "context_recall": cr,
        "context_precision": cp,
        "final_score": final_score,
        "used_reference": used_reference,
    }

    return {**state, "ragas_score": final_score}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 검색 ---")
    question = state.get("question")
    if not question:
        raise ValueError("질문이 누락되었습니다.")
    if not TAVILY_API_KEY:
        print("       ⚠️ TAVILY_API_KEY가 없어 웹 검색을 건너뜁니다.")
        return {**state, "web_search_results": "웹 검색 비활성화", "web_contexts": []}
    
    print(f"       🔍 웹 검색어: '{question}'")
    search_tool = TavilySearchResults(max_results=3)
    results = search_tool.invoke({"query": question}) 
    
    web_contexts: List[str] = []
    for r in results:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or r.get("snippet") or "").strip()
        url = (r.get("url") or "").strip()
        passage = f"{title}\n{content}\nURL: {url}".strip()
        web_contexts.append(passage)

    sr = json.dumps(results, ensure_ascii=False)
    print(f"       🌐 웹 검색 결과 {len(results)}개 수신 완료.")
    return {**state, "web_search_results": sr, "web_contexts": web_contexts}

def generate_web_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 기반 답변 생성 ---")
    question = state.get("question")
    search_results = state.get("web_search_results", "")
    if not question or not search_results or search_results == "웹 검색 비활성화":
        print("       ⚠️ 웹 검색 정보가 부족하여 답변 생성을 건너뜁니다.")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다.", "answer_source": "웹 검색 실패"}
    
    print(f"       ▶ 입력 웹 컨텍스트: '{search_results[:150].replace('\n', ' ')}...'")
    
    response = llm.invoke(web_prompt.format(question=question, search_results=search_results))
    ans = response.content
    token_usage = response.response_metadata.get("token_usage", {})
    
    print(f"       💬 생성된 답변: '{ans[:100].replace('\n', ' ')}...'")
    print(f"       📊 토큰 사용량 (Web): prompt={token_usage.get('prompt_tokens', 0)}, completion={token_usage.get('completion_tokens', 0)}, total={token_usage.get('total_tokens', 0)}")
    
    return {**state, "answer": ans, "answer_source": "웹 검색", "web_tokens": token_usage}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    answer = state.get("answer", "답변 생성 실패")
    source = state.get("answer_source", "알 수 없음")
    score = state.get("ragas_score")
    details = state.get("ragas_details", {})
    score_text = f"{score:.4f}" if score is not None else "N/A"
    
    token_usage = state.get("rag_tokens") or state.get("web_tokens")

    print("\n" + "="*50)
    print("🤖 최 종 답 변")
    print("="*50)
    print(f"✅ 답변 출처: {source} (RAGAS 점수: {score_text})")
    print(f"🔧 평가모드: {details.get('mode','N/A')}   |   reference 사용: {details.get('used_reference')}")
    print("📊 세부 지표:")
    def fmt(x): return "N/A" if x is None else f"{x:.4f}"
    print(f"   - Faithfulness:         {fmt(details.get('faithfulness'))}")
    print(f"   - Answer Relevancy:     {fmt(details.get('answer_relevancy'))}")
    print(f"   - Context Recall:       {fmt(details.get('context_recall'))}")
    print(f"   - Context Precision:    {fmt(details.get('context_precision'))}")
    print(f"   - Final (harmonic/avg): {fmt(details.get('final_score'))}")
    if token_usage:
        print("\n📈 이 턴의 토큰 사용량:")
        print(f"   - 프롬프트: {token_usage.get('prompt_tokens', 0)}")
        print(f"   - 응답:     {token_usage.get('completion_tokens', 0)}")
        print(f"   - 총량:     {token_usage.get('total_tokens', 0)}")
    print("\n" + answer)
    print("="*50)

    log_file = state.get("log_file") or ""
    q_for_log = state.get("question") or ""
    append_conversation_to_file(q_for_log, answer, source, score, token_usage, log_file)
    return state

# --- 라우터 ---
def master_router(state: GraphState) -> str:
    print("--- 🧭 라우터: 경로 결정 ---")
    score = state.get("ragas_score", 0.0)
    source = state.get("answer_source", "")
    rag_retries = state.get("rag_retry_count", 0)
    
    SCORE_THRESHOLD = 0.7
    RAG_RETRY_LIMIT = 2
    
    if source == "웹 검색":
        print("       ➡️ 결정: 웹 답변 평가 완료. 최종 답변으로 이동합니다.")
        return "end_journey"

    print(f"       📊 평가 점수 (내부 DB): {score:.4f} (임계값: {SCORE_THRESHOLD})")
    print(f"       🔄 RAG 재시도: {rag_retries}/{RAG_RETRY_LIMIT}")

    if score >= SCORE_THRESHOLD:
        print("       ➡️ 결정: RAG 답변 품질 통과. 최종 답변으로 이동합니다.")
        return "end_journey"
    else:
        if rag_retries < RAG_RETRY_LIMIT:
            print(f"       ➡️ 결정: RAG 답변 품질 미달. {rag_retries + 1}번째 답변 생성을 위해 돌아갑니다.")
            return "retry_rag"
        else:
            print(f"       ➡️ 결정: RAG 재시도 한도 도달. 웹 검색으로 보강합니다.")
            return "augment_with_web"

# --- 그래프 빌드 ---
def build_graph():
    print("▶ [그래프 설정] LangGraph 워크플로우 빌드 시작...")
    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_rag", generate_rag_node)
    g.add_node("ragas_eval", ragas_eval_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_web", generate_web_node)
    g.add_node("generate_answer", generate_answer_node)

    g.set_entry_point("load_milvus")
    g.add_edge("load_milvus", "retrieve")
    g.add_edge("retrieve", "generate_rag")
    g.add_edge("generate_rag", "ragas_eval")
    g.add_edge("web_search", "generate_web")
    g.add_edge("generate_web", "ragas_eval")

    g.add_conditional_edges(
        "ragas_eval",
        master_router,
        {
            "retry_rag": "generate_rag",
            "augment_with_web": "web_search",
            "end_journey": "generate_answer"
        }
    )
    
    g.add_edge("generate_answer", END)
    print("▶ [그래프 설정] LangGraph 워크플로우 빌드 완료.")
    return g.compile()

# --- 메인 실행 ---
if __name__ == "__main__":
    print("💬 Milvus 기반 LangGraph RAG + WebSearch (RAGAS 평가 포함) 시작 (종료: exit 또는 quit 입력)")
    
    log_dir = "milvusdb_crop65llm_logs"
    print(f"▶ [메인 실행] 로그 디렉토리 확인/생성: {log_dir}")
    Path(log_dir).mkdir(exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = f"{log_dir}/conversation_log_{session_timestamp}.json"

    app = build_graph()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_overall_tokens = 0

    print("▶ [메인 실행] 그래프 시각화 시도...")
    graph_image_path = "milvus_agent_workflow_llm.png"
    if os.path.exists(graph_image_path):
        print(f"\nℹ️ LangGraph 시각화 이미지 '{graph_image_path}'이(가) 이미 존재하여 생략합니다.")
    else:
        try:
            with open(graph_image_path, "wb") as f:
                f.write(app.get_graph().draw_mermaid_png())
            print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        except Exception as e:
            print(f"\n❌ 그래프 시각화 중 오류 발생: {e}")
            print("  (그래프 시각화를 위해서는 'mermaid-cli'가 필요할 수 있습니다.)")

    print("\n▶ [메인 실행] 대화 루프 시작. 질문을 입력해주세요.")
    while True:
        q = input("\n\n질문> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            print("💬 프로그램을 종료합니다. 🚪")
            print("\n" + "="*50)
            print("✨ 세션 종료: 총 토큰 사용량 요약 ✨")
            print("="*50)
            print(f"총 프롬프트 토큰: {total_prompt_tokens}")
            print(f"총 완료 토큰:     {total_completion_tokens}")
            print(f"총 사용 토큰:     {total_overall_tokens}")
            print("="*50)
            break

        print("-" * 60)
        print(f"🚀 새로운 질문 처리 시작: '{q}'")
        final_state = app.invoke({"question": q, "log_file": session_log_file})
        
        tokens_this_turn = final_state.get("rag_tokens") or final_state.get("web_tokens")
        if tokens_this_turn:
            total_prompt_tokens += tokens_this_turn.get("prompt_tokens", 0)
            total_completion_tokens += tokens_this_turn.get("completion_tokens", 0)
            total_overall_tokens += tokens_this_turn.get("total_tokens", 0)
            
        print(f"🏁 파이프라인 실행 완료.")
        print("-" * 60)
        
    print("\n프로그램이 종료되었습니다.")
