# --- 라이브러리 설치 ---
# 이 코드를 실행하기 전에 먼저 필요한 라이브러리를 설치해주세요.
# pip install langchain-huggingface langchain_community langchain-core langchain-groq langgraph pymilvus python-dotenv tavily-python ragas datasets

import os
import json
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from datetime import datetime
import re

# --- RAGAS 관련 라이브러리 임포트 ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# --- 환경 변수 로드 ---
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pymilvus import connections

# --- 프롬프트 정의 ---
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다.
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


# --- 🔥 변경: 상태 정의 ---
class GraphState(TypedDict, total=False):
    question: Optional[str]
    vectorstore: Optional[Milvus]
    context: Optional[str]
    answer: Optional[str]
    web_search_results: Optional[str]
    log_file: Optional[str]
    answer_source: Optional[str]
    ragas_score: Optional[float]
    rag_retry_count: Optional[int] # RAG 재시도 횟수만 카운트


# --- Embeddings 및 LLM ---
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)


# --- 유틸: 대화 로그 저장 ---
def append_conversation_to_file(question: str, answer: str, source: str, filename: str):
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s for s in sentences if s]
    data = {"timestamp": datetime.now().isoformat(), "question": question, "answer": sentences, "source": source}
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
    # RAG 재시도 카운터를 여기서 초기화
    return {**state, "context": context, "rag_retry_count": 0}

def generate_rag_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: RAG 답변 생성 ---")
    rag_retries = state.get("rag_retry_count", 0) + 1
    print(f"       🔄 RAG 생성 시도: {rag_retries}번째")

    context = state.get("context", "")
    question = state.get("question")
    if not question: raise ValueError("질문이 누락되었습니다.")
    
    print(f"       ▶ 입력 컨텍스트: '{context[:100].replace('\n', ' ')}...'")
    chain = (rag_prompt | make_llm() | StrOutputParser())
    ans = chain.invoke({"context": context, "question": question})
    print(f"       💬 생성된 답변: '{ans[:100].replace('\n', ' ')}...'")
    return {**state, "answer": ans, "answer_source": "내부 DB", "rag_retry_count": rag_retries}

def ragas_eval_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: RAGAS 답변 평가 ---")
    question = state.get("question")
    answer = state.get("answer", "")
    source = state.get("answer_source", "N/A")

    # 1. 답변 소스에 따라 평가 기준이 될 컨텍스트를 명확하게 선택합니다.
    if source == "웹 검색":
        eval_context = state.get("web_search_results", "")
        context_source_name = "웹 검색 결과"
    else: # "내부 DB" 또는 기타
        eval_context = state.get("context", "")
        context_source_name = "내부 DB 문서"

    print(f"       ▶ 평가 대상 ({source}): '{answer[:100].replace('\n', ' ')}...'")
    print(f"       ▶ 평가 기준 ({context_source_name}): '{eval_context[:100].replace('\n', ' ')}...'")
    print(f"       ▶ 기준 컨텍스트 길이: {len(eval_context)} 자")


    if not all([question, answer, eval_context]):
        print("       ⚠️ 평가에 필요한 정보가 부족하여 건너뜁니다.")
        return {**state, "ragas_score": 0.0}

    dataset_dict = {"question": [question], "answer": [answer], "contexts": [[eval_context]]}
    dataset = Dataset.from_dict(dataset_dict)

    try:
        # 2. 매번 LLM 객체를 새로 만들지 않고, 이미 생성된 전역 llm 객체를 재사용합니다.
        result = evaluate(
            dataset, 
            metrics=[faithfulness, answer_relevancy], 
            mllm=llm,  # 전역 llm 객체 사용
            embeddings=embedding_model, 
            raise_exceptions=False
        )
        score = result.get("ragas_score", 0.0)
        print(f"       📊 RAGAS 평가 점수: {score:.4f}")
    except Exception as e:
        print(f"       ❌ RAGAS 평가 중 오류: {e}")
        score = 0.0
        
    return {**state, "ragas_score": score}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 검색 ---")
    question = state.get("question")
    if not question: raise ValueError("질문이 누락되었습니다.")
    if not TAVILY_API_KEY:
        print("       ⚠️ TAVILY_API_KEY가 없어 웹 검색을 건너뜁니다.")
        return {**state, "web_search_results": "웹 검색 비활성화"}
    
    print(f"       🔍 웹 검색어: '{question}'")
    search_tool = TavilySearchResults(max_results=1)
    results = search_tool.invoke({"query": question})
    sr = "\n\n".join([json.dumps(r, ensure_ascii=False) for r in results])
    print(f"       🌐 웹 검색 결과 {len(results)}개 수신 완료.")
    return {**state, "web_search_results": sr}

def generate_web_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 기반 답변 생성 ---")
    question = state.get("question")
    search_results = state.get("web_search_results", "")
    if not question or not search_results or search_results == "웹 검색 비활성화":
        print("       ⚠️ 웹 검색 정보가 부족하여 답변 생성을 건너뜁니다.")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다.", "answer_source": "웹 검색 실패"}
    
    print(f"       ▶ 입력 웹 컨텍스트: '{search_results[:150].replace('\n', ' ')}...'")
    chain = (web_prompt | make_llm() | StrOutputParser())
    ans = chain.invoke({"question": question, "search_results": search_results})
    print(f"       💬 생성된 답변: '{ans[:100].replace('\n', ' ')}...'")
    return {**state, "answer": ans, "answer_source": "웹 검색"}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    answer = state.get("answer", "답변 생성 실패")
    source = state.get("answer_source", "알 수 없음")
    score = state.get("ragas_score")
    score_text = f"{score:.4f}" if score is not None else "N/A"
    
    print("\n" + "="*50)
    print("🤖 최 종 답 변")
    print("="*50)
    print(f"✅ 답변 출처: {source} (RAGAS 점수: {score_text})")
    print(f"\n{answer}")
    print("="*50)

    log_file = state.get("log_file") or ""
    q_for_log = state.get("question") or ""
    append_conversation_to_file(q_for_log, answer, source, log_file)
    return state

# --- 🔥 변경: 하이브리드 라우터 ---
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
            # --- 🔥 변경: 요청하신 print문 추가 ---
            print(f"       ➡️ 결정: RAG 답변 품질 미달. {rag_retries + 1}번째 답변 생성을 위해 돌아갑니다.")
            return "retry_rag"
        else:
            print(f"       ➡️ 결정: RAG 재시도 한도 도달. 웹 검색으로 보강합니다.")
            return "augment_with_web"


# --- 🔥 변경: 그래프 빌드 ---
def build_graph():
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
    g.add_edge("generate_web", "ragas_eval") # 웹 답변도 평가

    # 하이브리드 라우팅 로직
    g.add_conditional_edges(
        "ragas_eval",
        master_router,
        {
            "retry_rag": "generate_rag",        # RAG 재시도
            "augment_with_web": "web_search",   # 웹 검색으로 전환
            "end_journey": "generate_answer"    # 최종 답변으로 이동
        }
    )
    
    g.add_edge("generate_answer", END)
    return g.compile()

# --- 메인 실행 ---
if __name__ == "__main__":
    print("💬 Milvus 기반 LangGraph RAG + WebSearch (RAGAS 평가 포함) 시작 (종료: exit 또는 quit 입력)")

    log_dir = "milvusdb_crop65llm_logs"
    Path(log_dir).mkdir(exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = f"{log_dir}/conversation_log_{session_timestamp}.json"

    app = build_graph()

    try:
        graph_image_path = "milvus_agent_workflow_llm_with_ragas.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 그래프 시각화 중 오류 발생: {e}")
        print("   (그래프 시각화를 위해서는 'mermaid-cli'가 필요할 수 있습니다.)")

    while True:
        q = input("\n\n질문> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            print("💬 프로그램을 종료합니다. 🚪")
            break

        print("-" * 60)
        print(f"🚀 새로운 질문 처리 시작: '{q}'")
        app.invoke({"question": q, "log_file": session_log_file})
        print(f"🏁 파이프라인 실행 완료.")
        print("-" * 60)
        
    print("\n프로그램이 종료되었습니다.")