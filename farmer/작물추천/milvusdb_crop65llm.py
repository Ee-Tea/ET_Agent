import os
import json
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from datetime import datetime
import re


# --- 환경 변수 로드 ---
# .env 파일이 없다면 직접 환경 변수를 설정해주세요.
# 예: os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
load_dotenv(find_dotenv())

# --- Milvus / Embedding 모델 설정 ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hongyoungjun")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# --- LLM 설정 ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192") # 더 빠른 모델로 변경
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
아래 '웹 검색 결과'와 '질문'을 바탕으로 답변을 생성하세요.

[웹 검색 결과]
{search_results}

규칙:
- 검색 결과만 사용해 질문에 답하세요.
- 한글로만 작성.
- 답변이 불가능하면 "주어진 정보로는 답변할 수 없습니다."라고 작성하세요.

🟢 질문: {question}
✨ 답변:
"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL)


# --- 상태 정의 ---
class GraphState(TypedDict, total=False):
    question: Optional[str]           # 사용자 질문
    vectorstore: Optional[Milvus]     # Milvus 벡터스토어 객체
    context: Optional[str]            # Milvus에서 검색된 문서 내용
    answer: Optional[str]             # LLM이 생성한 최종 답변
    web_search_results: Optional[str] # 웹 검색 결과
    user_decision: Optional[str]      # "yes"|"no"
    decision_reason: Optional[str]    # 자동 판단 이유 로깅
    log_file: Optional[str]           # 로그 파일 경로
    answer_source: Optional[str]      # 답변 출처 ('내부 DB' 또는 '웹 검색')

# --- Embeddings 및 LLM ---
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

# 시간 민감 키워드
TIME_SENSITIVE = re.compile(
    r"(최신|오늘|방금|지금|실시간|변경|업데이트|뉴스|가격|환율|주가|일정|스케줄|예보|날씨|모집|채용|재고|판매|운항|발표)"
)

# --- 유틸: 대화 로그 저장 ---
def append_conversation_to_file(question: str, answer: str, source: str, filename: str):
    # answer 문장을 리스트로 분리
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s for s in sentences if s]  # 빈 문자열 제거

    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": sentences,   # 리스트로 저장
        "source": source
    }

    if filename:
        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        hist: List[Dict] = json.load(f)
                except json.JSONDecodeError:
                    print(f"       ⚠️ '{filename}' 손상 → 새 파일 시작")
                    hist = []
            else:
                hist = []
            hist.append(data)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(hist, f, ensure_ascii=False, indent=4)
            print(f"       ✅ 대화 기록 저장: {filename}")
        except Exception as e:
            print(f"       ❌ 대화 기록 저장 오류: {e}")


# --- LangGraph 노드 ---
def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: Milvus 벡터스토어 로드 ---")
    if "default" not in connections.list_connections() or not connections.has_connection("default"):
        print("       - Milvus 연결이 없어 새로 연결합니다.")
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)

    try:
        vs = Milvus(
            embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )
        print(f"       ✅ Milvus 로드 완료 (컬렉션: {MILVUS_COLLECTION})")
        return {**state, "vectorstore": vs}
    except Exception as e:
        print(f"       ❌ Milvus 로드 실패: {e}")
        raise ConnectionError("Milvus 벡터스토어 로드 실패")

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 문서 검색 ---")
    question = state.get("question")
    vectorstore = state.get("vectorstore")
    if not question or not vectorstore:
        raise ValueError("질문 또는 벡터스토어가 누락되었습니다.")
    print(f"       - 질문: '{question}'")
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)

    context = ""
    print(f"       ✅ {len(docs_with_scores)}개 문서 검색.")
    for i, (doc, score) in enumerate(docs_with_scores):
        preview = (doc.page_content or "")[:100].replace("\n", " ")
        print(f"       - 문서 {i+1} (점수: {score:.4f}): '{preview}...'")
        context += f"\n\n{doc.page_content}"
    return {**state, "context": context}

def generate_rag_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: RAG 답변 생성 ---")
    context = state.get("context")
    question = state.get("question")
    if not context or not question:
        raise ValueError("문맥 또는 질문이 누락되었습니다.")
    chain = (rag_prompt | make_llm() | StrOutputParser())
    ans = chain.invoke({"context": context, "question": question})
    print("       ✅ RAG 답변 생성 완료.")
    print(f"       - 미리보기: '{ans[:100]}...'")
    return {**state, "answer": ans, "answer_source": "내부 DB"}

def user_decision_node(state: GraphState) -> Dict[str, Any]:
    """
    ✅ 자동 판단:
        - Tavily 키 없음 → 'no'
        - RAG 실패문구 포함 → 'yes'
        - 시간민감 키워드 포함 → 'yes'
        - 그 외 → 'no'
    """
    print("--- 🧩 노드 시작: 사용자 결정(자동) ---")
    q = state.get("question", "") or ""
    ans = state.get("answer", "") or ""
    if not TAVILY_API_KEY:
        print("       ↪️ 웹검색 불가: no_tavily_api_key")
        return {**state, "user_decision": "no", "decision_reason": "no_tavily_api_key"}
    rag_failed = "주어진 정보로는 답변할 수 없습니다." in ans
    time_sensitive = bool(TIME_SENSITIVE.search(q))
    if rag_failed or time_sensitive:
        reason = "rag_failed" if rag_failed else "time_sensitive"
        print(f"       ↪️ 웹검색 진행: {reason}")
        return {**state, "user_decision": "yes", "decision_reason": reason}
    print("       ↪️ 웹검색 건너뜀: context_sufficient")
    return {**state, "user_decision": "no", "decision_reason": "context_sufficient"}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 검색 ---")
    question = state.get("question")
    if not question:
        raise ValueError("질문이 누락되었습니다.")
    if not TAVILY_API_KEY:
        print("       ⚠️ TAVILY_API_KEY 미설정 → 웹 검색 비활성화")
        return {**state, "web_search_results": "웹 검색 비활성화"}
    search_tool = TavilySearchResults(max_results=3)
    results = search_tool.invoke({"query": question})
    sr = "\n\n".join([json.dumps(r, ensure_ascii=False) for r in results])
    print("       ✅ 웹 검색 결과 수신:", len(results), "개")
    return {**state, "web_search_results": sr}

def generate_web_node(state: GraphState) -> Dict[str, Any]:
    print("--- 🧩 노드 시작: 웹 기반 답변 생성 ---")
    question = state.get("question")
    search_results = state.get("web_search_results")
    if not question or not search_results or search_results == "웹 검색 비활성화":
        print("       ⚠️ 웹 검색 정보 부족 → 웹기반 답변 불가")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다.", "answer_source": "웹 검색 실패"}
    chain = (web_prompt | make_llm() | StrOutputParser())
    ans = chain.invoke({"question": question, "search_results": search_results})
    print("       ✅ 웹 기반 답변 생성 완료.")
    print(f"       - 미리보기: '{ans[:100]}...'")
    return {**state, "answer": ans, "answer_source": "웹 검색"}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    ✅ 최종 답변 처리:
      - 답변을 출력하고 로그 파일에 저장합니다.
      - 이 노드는 그래프의 최종 단계 역할을 합니다.
    """
    print("\n--- 🤖 최종 답변 ---")
    answer = state.get("answer", "답변 생성 실패")
    source = state.get("answer_source", "알 수 없음")

    print(f"✅ 답변 출처: {source}")
    print(answer)
    print("---------------------\n")

    log_file = state.get("log_file") or ""
    q_for_log = state.get("question") or ""
    append_conversation_to_file(q_for_log, answer, source, log_file)

    return state

# --- 조건부 라우팅 ---
def route_to_web_search(state: GraphState) -> str:
    print("--- 🧭 라우터: RAG 결과 기반 분기 ---")
    answer = (state.get("answer") or "")
    if "주어진 정보로는 답변할 수 없습니다." in answer:
        print("       ↪️ RAG 실패 → user_decision")
        return "user_decision"
    print("       🎉 RAG 성공 → generate_answer")
    return "generate_answer"

def route_user_decision(state: GraphState) -> str:
    print("--- 🧭 라우터: 사용자 결정(자동) 처리 ---")
    return "do_web_search" if state.get("user_decision") == "yes" else "skip_web_search"

# --- 그래프 빌드 ---
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_rag", generate_rag_node)
    g.add_node("user_decision", user_decision_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_web", generate_web_node)
    g.add_node("generate_answer", generate_answer_node)

    g.set_entry_point("load_milvus")
    g.add_edge("load_milvus", "retrieve")
    g.add_edge("retrieve", "generate_rag")
    g.add_edge("generate_rag", "user_decision")
    g.add_conditional_edges("user_decision", route_user_decision,
                            {"do_web_search": "web_search", "skip_web_search": "generate_answer"})
    g.add_edge("web_search", "generate_web")
    g.add_edge("generate_web", "generate_answer")
    
    g.add_edge("generate_answer", END)
    
    return g.compile()

# --- 메인 실행 ---
if __name__ == "__main__":
    print("💬 Milvus 기반 LangGraph RAG + WebSearch 시작 (종료: exit 또는 quit 입력)")

    log_dir = "milvusdb_crop65llm_logs"
    Path(log_dir).mkdir(exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = f"{log_dir}/conversation_log_{session_timestamp}.json"

    app = build_graph()

    # --- ✍️ [수정됨] 그래프 시각화 ---
    try:
        graph_image_path = "milvus_agent_workflow_llm.png"
        with open(graph_image_path, "wb") as f:
            # 변수명을 agent_app에서 app으로 수정했습니다.
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        # Mermaid-CLI가 설치되지 않은 경우 오류가 발생할 수 있습니다.
        # https://mermaid-js.github.io/mermaid/getting-started/mermaid-cli.html
        print(f"❌ 그래프 시각화 중 오류 발생: {e}")
        print("   (그래프 시각화를 위해서는 'mermaid-cli'가 필요할 수 있습니다.)")

    # --- 대화 루프 실행 ---
    while True:
        q = input("\n질문> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            print("💬 파이프라인을 종료합니다. 🚪")
            break

        # 그래프 실행
        app.invoke({"question": q, "log_file": session_log_file})
        
    print("💬 파이프라인 종료. 🏁")