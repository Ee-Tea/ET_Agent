import os                                           # 운영체제 기능(파일 경로, 환경 변수)을 사용하기 위해 임포트
import json                                         # JSON 형식 데이터를 다루기 위해 임포트
from typing import TypedDict, Optional, Any, Dict, List # 타입 힌팅을 지원하기 위해 임포트
from dotenv import load_dotenv, find_dotenv         # .env 파일에서 환경 변수를 로드하기 위해 임포트
from pathlib import Path                            # 파일 시스템 경로를 객체처럼 다루기 위해 임포트
from datetime import datetime                       # 날짜와 시간을 다루기 위해 임포트
import re                                           # 정규표현식을 사용하기 위해 임포트


# --- 환경 변수 로드 ---
# .env 파일이 없다면 직접 환경 변수를 설정해주세요.
# 예: os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
load_dotenv(find_dotenv())                          # .env 파일을 찾아 환경 변수를 로드

# --- Milvus / Embedding 모델 설정 ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530") # Milvus 서버 주소 설정
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")       # Milvus 인증 토큰 설정
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hongyoungjun") # 사용할 Milvus 컬렉션 이름 설정
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask") # 사용할 한국어 임베딩 모델 이름 설정

# --- LLM 설정 ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")            # Groq API 키 설정
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192") # 사용할 Groq의 LLM 모델 설정
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7")) # 모델의 생성 온도(무작위성) 설정

# --- Web Search 설정 ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")          # Tavily 웹 검색 API 키 설정

if not GROQ_API_KEY:                                # Groq API 키가 없으면
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.") # 오류 발생

# --- 라이브러리 임포트 ---
from langchain_huggingface import HuggingFaceEmbeddings     # HuggingFace 임베딩 모델을 사용하기 위해 임포트
from langchain_community.vectorstores import Milvus         # Milvus 벡터스토어를 사용하기 위해 임포트
from langchain_community.tools.tavily_search import TavilySearchResults # Tavily 웹 검색 도구 임포트
from langchain_core.prompts import ChatPromptTemplate       # 채팅 프롬프트 템플릿을 만들기 위해 임포트
from langchain_core.output_parsers import StrOutputParser   # LLM의 출력을 문자열로 파싱하기 위해 임포트
from langchain_groq import ChatGroq                         # Groq 채팅 모델을 사용하기 위해 임포트
from langgraph.graph import StateGraph, END                 # LangGraph의 StateGraph와 END를 임포트
from pymilvus import connections                          # Milvus 연결 관리를 위해 임포트

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
""" # 내부 DB(Vector DB) 검색 결과를 기반으로 답변을 생성하는 RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL) # 프롬프트 템플릿 객체 생성

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
""" # 웹 검색 결과를 기반으로 답변을 생성하는 프롬프트
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL) # 프롬프트 템플릿 객체 생성


# --- 상태 정의 ---
class GraphState(TypedDict, total=False):           # LangGraph에서 노드 간에 전달될 데이터의 구조를 정의
    question: Optional[str]                         # 사용자 질문
    vectorstore: Optional[Milvus]                   # Milvus 벡터스토어 객체
    context: Optional[str]                          # Milvus에서 검색된 문서 내용
    answer: Optional[str]                           # LLM이 생성한 최종 답변
    web_search_results: Optional[str]               # 웹 검색 결과
    user_decision: Optional[str]                    # 웹 검색 여부에 대한 자동 판단 결과 ("yes"|"no")
    decision_reason: Optional[str]                  # 자동 판단의 이유
    log_file: Optional[str]                         # 로그 파일 경로
    answer_source: Optional[str]                    # 답변 출처 ('내부 DB' 또는 '웹 검색')

# --- Embeddings 및 LLM ---
embedding_model = HuggingFaceEmbeddings(            # HuggingFace 임베딩 모델 초기화
    model_name=EMBED_MODEL_NAME,                    # 사용할 모델 이름 지정
    model_kwargs={"device": "cpu"}                  # 모델을 CPU에서 실행하도록 설정
)

def make_llm() -> ChatGroq:                         # Groq LLM 객체를 생성하는 함수
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY) # 설정된 값으로 객체 생성 및 반환

# 시간 민감 키워드
TIME_SENSITIVE = re.compile(                        # 시간과 관련된 키워드를 찾기 위한 정규표현식 컴파일
    r"(최신|오늘|방금|지금|실시간|변경|업데이트|뉴스|가격|환율|주가|일정|스케줄|예보|날씨|모집|채용|재고|판매|운항|발표)"
)

# --- 유틸: 대화 로그 저장 ---
def append_conversation_to_file(question: str, answer: str, source: str, filename: str): # 대화 내용을 JSON 파일에 저장하는 함수
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip()) # 답변을 문장 단위로 분리
    sentences = [s for s in sentences if s]         # 빈 문장 제거

    data = {                                        # 저장할 데이터 구조 정의
        "timestamp": datetime.now().isoformat(),    # 현재 시간
        "question": question,                       # 질문
        "answer": sentences,                        # 문장 리스트로 분리된 답변
        "source": source                            # 답변 출처
    }

    if filename:                                    # 파일 이름이 유효하면
        try:                                        # 파일 입출력 오류 처리
            if os.path.exists(filename) and os.path.getsize(filename) > 0: # 파일이 존재하고 비어있지 않으면
                try:                                # JSON 파싱 오류 처리
                    with open(filename, "r", encoding="utf-8") as f: # 파일을 읽기 모드로 열기
                        hist: List[Dict] = json.load(f) # 기존 대화 기록 불러오기
                except json.JSONDecodeError:        # JSON 파일이 손상된 경우
                    print(f"       ⚠️ '{filename}' 손상 → 새 파일 시작")
                    hist = []                       # 빈 리스트로 새로 시작
            else:                                   # 파일이 없거나 비어있으면
                hist = []                           # 빈 리스트로 시작
            hist.append(data)                       # 현재 대화를 기록에 추가
            with open(filename, "w", encoding="utf-8") as f: # 파일을 쓰기 모드로 열기
                json.dump(hist, f, ensure_ascii=False, indent=4) # 전체 대화 기록을 파일에 저장
            print(f"       ✅ 대화 기록 저장: {filename}")
        except Exception as e:                      # 다른 오류 발생 시
            print(f"       ❌ 대화 기록 저장 오류: {e}")


# --- LangGraph 노드 ---
def load_milvus_node(state: GraphState) -> Dict[str, Any]: # Milvus 벡터스토어를 로드하는 노드
    print("--- 🧩 노드 시작: Milvus 벡터스토어 로드 ---")
    if "default" not in connections.list_connections() or not connections.has_connection("default"): # Milvus 연결이 없으면
        print("       - Milvus 연결이 없어 새로 연결합니다.")
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN) # 새로 연결

    try:                                            # 벡터스토어 로드 오류 처리
        vs = Milvus(                                # Milvus 벡터스토어 객체 초기화
            embedding_model,                        # 사용할 임베딩 모델
            collection_name=MILVUS_COLLECTION,      # 컬렉션 이름 지정
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN}, # 연결 정보 전달
        )
        print(f"       ✅ Milvus 로드 완료 (컬렉션: {MILVUS_COLLECTION})")
        return {**state, "vectorstore": vs}         # 상태에 벡터스토어 객체를 추가하여 반환
    except Exception as e:                          # 로드 실패 시
        print(f"       ❌ Milvus 로드 실패: {e}")
        raise ConnectionError("Milvus 벡터스토어 로드 실패") # 연결 오류 발생

def retrieve_node(state: GraphState) -> Dict[str, Any]: # Milvus에서 관련 문서를 검색하는 노드
    print("--- 🧩 노드 시작: 문서 검색 ---")
    question = state.get("question")                # 상태에서 질문 가져오기
    vectorstore = state.get("vectorstore")          # 상태에서 벡터스토어 객체 가져오기
    if not question or not vectorstore:             # 질문 또는 벡터스토어가 없으면
        raise ValueError("질문 또는 벡터스토어가 누락되었습니다.") # 오류 발생
    print(f"       - 질문: '{question}'")
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=5) # 질문과 유사한 문서 5개 검색

    context = ""                                    # 검색된 문서 내용을 합칠 변수
    print(f"       ✅ {len(docs_with_scores)}개 문서 검색.")
    for i, (doc, score) in enumerate(docs_with_scores): # 검색된 각 문서를 순회
        preview = (doc.page_content or "")[:100].replace("\n", " ") # 내용 미리보기 생성
        print(f"       - 문서 {i+1} (점수: {score:.4f}): '{preview}...'")
        context += f"\n\n{doc.page_content}"        # 문서 내용을 context에 추가
    return {**state, "context": context}            # 상태에 context를 추가하여 반환

def generate_rag_node(state: GraphState) -> Dict[str, Any]: # 검색된 문서를 기반으로 RAG 답변을 생성하는 노드
    print("--- 🧩 노드 시작: RAG 답변 생성 ---")
    context = state.get("context")                  # 상태에서 context 가져오기
    question = state.get("question")                # 상태에서 질문 가져오기
    if not context or not question:                 # context 또는 질문이 없으면
        raise ValueError("문맥 또는 질문이 누락되었습니다.") # 오류 발생
    chain = (rag_prompt | make_llm() | StrOutputParser()) # 프롬프트, LLM, 출력 파서를 연결하여 체인 생성
    ans = chain.invoke({"context": context, "question": question}) # 체인을 실행하여 답변 생성
    print("       ✅ RAG 답변 생성 완료.")
    print(f"       - 미리보기: '{ans[:100]}...'")
    return {**state, "answer": ans, "answer_source": "내부 DB"} # 상태에 답변과 출처를 추가하여 반환

def user_decision_node(state: GraphState) -> Dict[str, Any]: # 웹 검색을 할지 자동으로 결정하는 노드
    """
    ✅ 자동 판단:
        - Tavily 키 없음 → 'no'
        - RAG 실패문구 포함 → 'yes'
        - 시간민감 키워드 포함 → 'yes'
        - 그 외 → 'no'
    """
    print("--- 🧩 노드 시작: 사용자 결정(자동) ---")
    q = state.get("question", "") or ""             # 상태에서 질문 가져오기
    ans = state.get("answer", "") or ""             # 상태에서 RAG 답변 가져오기
    if not TAVILY_API_KEY:                          # Tavily API 키가 없으면
        print("       ↪️ 웹검색 불가: no_tavily_api_key")
        return {**state, "user_decision": "no", "decision_reason": "no_tavily_api_key"} # 웹 검색 안 함으로 결정
    rag_failed = "주어진 정보로는 답변할 수 없습니다." in ans # RAG 답변이 실패했는지 확인
    time_sensitive = bool(TIME_SENSITIVE.search(q)) # 질문에 시간 민감 키워드가 있는지 확인
    if rag_failed or time_sensitive:                # RAG가 실패했거나 시간 민감 질문이면
        reason = "rag_failed" if rag_failed else "time_sensitive" # 이유 설정
        print(f"       ↪️ 웹검색 진행: {reason}")
        return {**state, "user_decision": "yes", "decision_reason": reason} # 웹 검색을 하기로 결정
    print("       ↪️ 웹검색 건너뜀: context_sufficient")
    return {**state, "user_decision": "no", "decision_reason": "context_sufficient"} # 웹 검색을 안 하기로 결정

def web_search_node(state: GraphState) -> Dict[str, Any]: # 웹 검색을 수행하는 노드
    print("--- 🧩 노드 시작: 웹 검색 ---")
    question = state.get("question")                # 상태에서 질문 가져오기
    if not question:                                # 질문이 없으면
        raise ValueError("질문이 누락되었습니다.")   # 오류 발생
    if not TAVILY_API_KEY:                          # Tavily API 키가 없으면
        print("       ⚠️ TAVILY_API_KEY 미설정 → 웹 검색 비활성화")
        return {**state, "web_search_results": "웹 검색 비활성화"} # 웹 검색 결과 비활성화로 설정
    search_tool = TavilySearchResults(max_results=3) # Tavily 검색 도구 초기화 (최대 3개 결과)
    results = search_tool.invoke({"query": question}) # 검색 실행
    sr = "\n\n".join([json.dumps(r, ensure_ascii=False) for r in results]) # 검색 결과를 JSON 문자열로 변환
    print("       ✅ 웹 검색 결과 수신:", len(results), "개")
    return {**state, "web_search_results": sr}      # 상태에 웹 검색 결과를 추가하여 반환

def generate_web_node(state: GraphState) -> Dict[str, Any]: # 웹 검색 결과를 기반으로 답변을 생성하는 노드
    print("--- 🧩 노드 시작: 웹 기반 답변 생성 ---")
    question = state.get("question")                # 상태에서 질문 가져오기
    search_results = state.get("web_search_results") # 상태에서 웹 검색 결과 가져오기
    if not question or not search_results or search_results == "웹 검색 비활성화": # 필요한 정보가 부족하면
        print("       ⚠️ 웹 검색 정보 부족 → 웹기반 답변 불가")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다.", "answer_source": "웹 검색 실패"} # 실패 상태 반환
    chain = (web_prompt | make_llm() | StrOutputParser()) # 프롬프트, LLM, 출력 파서를 연결하여 체인 생성
    ans = chain.invoke({"question": question, "search_results": search_results}) # 체인을 실행하여 답변 생성
    print("       ✅ 웹 기반 답변 생성 완료.")
    print(f"       - 미리보기: '{ans[:100]}...'")
    return {**state, "answer": ans, "answer_source": "웹 검색"} # 상태에 답변과 출처를 추가하여 반환

def generate_answer_node(state: GraphState) -> Dict[str, Any]: # 최종 답변을 출력하고 로그를 저장하는 노드
    """
    ✅ 최종 답변 처리:
      - 답변을 출력하고 로그 파일에 저장합니다.
      - 이 노드는 그래프의 최종 단계 역할을 합니다.
    """
    print("\n--- 🤖 최종 답변 ---")
    answer = state.get("answer", "답변 생성 실패")   # 상태에서 최종 답변 가져오기
    source = state.get("answer_source", "알 수 없음") # 상태에서 답변 출처 가져오기

    print(f"✅ 답변 출처: {source}")                 # 답변 출처 출력
    print(answer)                                   # 최종 답변 출력
    print("---------------------\n")

    log_file = state.get("log_file") or ""          # 로그 파일 경로 가져오기
    q_for_log = state.get("question") or ""         # 질문 내용 가져오기
    append_conversation_to_file(q_for_log, answer, source, log_file) # 대화 내용 파일에 저장

    return state                                    # 최종 상태 반환

# --- 조건부 라우팅 ---
def route_to_web_search(state: GraphState) -> str:  # RAG 답변 결과에 따라 분기하는 라우터 함수
    print("--- 🧭 라우터: RAG 결과 기반 분기 ---")
    answer = (state.get("answer") or "")            # 상태에서 RAG 답변 가져오기
    if "주어진 정보로는 답변할 수 없습니다." in answer: # RAG 답변이 실패했다면
        print("       ↪️ RAG 실패 → user_decision")
        return "user_decision"                      # user_decision 노드로 이동
    print("       🎉 RAG 성공 → generate_answer")
    return "generate_answer"                        # 성공했다면 generate_answer 노드로 이동

def route_user_decision(state: GraphState) -> str:  # 자동 판단 결과에 따라 분기하는 라우터 함수
    print("--- 🧭 라우터: 사용자 결정(자동) 처리 ---")
    return "do_web_search" if state.get("user_decision") == "yes" else "skip_web_search" # 'yes'이면 웹 검색, 'no'이면 건너뛰기

# --- 그래프 빌드 ---
def build_graph():                                  # LangGraph 워크플로우를 정의하는 함수
    g = StateGraph(GraphState)                      # GraphState를 상태로 사용하는 그래프 생성
    g.add_node("load_milvus", load_milvus_node)     # 'load_milvus' 노드 추가
    g.add_node("retrieve", retrieve_node)           # 'retrieve' 노드 추가
    g.add_node("generate_rag", generate_rag_node)   # 'generate_rag' 노드 추가
    g.add_node("user_decision", user_decision_node) # 'user_decision' 노드 추가
    g.add_node("web_search", web_search_node)       # 'web_search' 노드 추가
    g.add_node("generate_web", generate_web_node)   # 'generate_web' 노드 추가
    g.add_node("generate_answer", generate_answer_node) # 'generate_answer' 노드 추가

    g.set_entry_point("load_milvus")                # 'load_milvus'를 시작 노드로 설정
    g.add_edge("load_milvus", "retrieve")           # 'load_milvus' 다음에 'retrieve' 실행
    g.add_edge("retrieve", "generate_rag")          # 'retrieve' 다음에 'generate_rag' 실행
    g.add_edge("generate_rag", "user_decision")     # 'generate_rag' 다음에 'user_decision' 실행 (기존 conditional_edges에서 변경)
    g.add_conditional_edges(                        # 'user_decision' 노드의 결과에 따라 분기
        "user_decision",                            # 분기 시작 노드
        route_user_decision,                        # 사용할 라우터 함수
        {                                           # 분기 경로 맵
            "do_web_search": "web_search",          # 라우터가 "do_web_search" 반환 시 'web_search'로 이동
            "skip_web_search": "generate_answer"    # 라우터가 "skip_web_search" 반환 시 'generate_answer'로 이동
        }
    )
    g.add_edge("web_search", "generate_web")        # 'web_search' 다음에 'generate_web' 실행
    g.add_edge("generate_web", "generate_answer")   # 'generate_web' 다음에 'generate_answer' 실행
    
    g.add_edge("generate_answer", END)              # 'generate_answer'가 끝나면 워크플로우 종료
    
    return g.compile()                              # 정의된 그래프를 컴파일하여 실행 가능한 객체로 반환

# --- 메인 실행 ---
if __name__ == "__main__":                          # 이 스크립트 파일이 직접 실행될 때
    print("💬 Milvus 기반 LangGraph RAG + WebSearch 시작 (종료: exit 또는 quit 입력)")

    log_dir = "milvusdb_crop65llm_logs"             # 로그를 저장할 디렉토리 이름
    Path(log_dir).mkdir(exist_ok=True)              # 로그 디렉토리 생성 (이미 있어도 오류 없음)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 시간을 기반으로 타임스탬프 생성
    session_log_file = f"{log_dir}/conversation_log_{session_timestamp}.json" # 세션 로그 파일 경로 생성

    app = build_graph()                             # 그래프를 빌드하여 실행 앱 생성

    # --- 그래프 시각화 ---
    try:                                            # 그래프 시각화 오류 처리
        graph_image_path = "milvus_agent_workflow_llm.png" # 저장할 이미지 파일 이름
        with open(graph_image_path, "wb") as f:     # 이미지 파일을 바이너리 쓰기 모드로 열기
            f.write(app.get_graph().draw_mermaid_png()) # 그래프 구조를 PNG로 그려 파일에 저장
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:                          # 오류 발생 시
        # Mermaid-CLI가 설치되지 않은 경우 오류가 발생할 수 있습니다.
        # https://mermaid-js.github.io/mermaid/getting-started/mermaid-cli.html
        print(f"❌ 그래프 시각화 중 오류 발생: {e}")
        print("   (그래프 시각화를 위해서는 'mermaid-cli'가 필요할 수 있습니다.)")

    # --- 대화 루프 실행 ---
    while True:                                     # 무한 루프 시작 (사용자가 종료할 때까지)
        q = input("\n질문> ").strip()               # 사용자로부터 질문 입력받기
        if not q or q.lower() in ("exit", "quit"):  # 사용자가 종료를 원하면
            print("💬 파이프라인을 종료합니다. 🚪")
            break                                   # 루프 탈출

        # 그래프 실행
        app.invoke({"question": q, "log_file": session_log_file}) # 질문과 로그 파일 경로를 초기 상태로 하여 그래프 실행
        
    print("💬 파이프라인 종료. 🏁")