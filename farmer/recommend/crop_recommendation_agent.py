# crop_recommendation_agent_optimized.py

# ==================== 라이브러리 불러오기 ====================
# 모든 필요한 라이브러리들을 코드 최상단에 모아두었습니다.
import os  # 운영체제(파일 경로, 환경 변수 등)와 상호작용합니다.
import re  # 정규 표현식을 사용하여 문자열을 조작합니다.
import argparse  # 명령행 인자를 파싱합니다.
from typing import List, Dict, Any, Optional, TypedDict  # 타입 힌트를 위한 모듈입니다.

from dotenv import load_dotenv, find_dotenv  # '.env' 파일에서 환경 변수를 로드합니다.
from langchain_core.prompts import ChatPromptTemplate  # 챗봇 프롬프트 템플릿을 정의합니다.
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 임베딩 모델을 사용합니다.
from langchain_milvus import Milvus as MilvusVectorStore  # Milvus 벡터 DB를 LangChain에 통합합니다.
from tavily import TavilyClient  # Tavily API를 사용하여 웹 검색을 수행합니다.
from langchain_openai import ChatOpenAI  # OpenAI 챗 모델을 사용합니다.
from langgraph.graph import StateGraph, END  # LangGraph의 상태 그래프와 종료 노드를 정의합니다.
from pymilvus import connections  # Milvus 서버와의 연결을 관리합니다.

# ==================== 환경 변수 로드 ====================
load_dotenv(find_dotenv())  # .env 파일의 환경 변수들을 로드합니다.

# ==================== 환경 설정 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # 환경 변수에서 Milvus 서버의 URI를 가져오거나 기본값을 사용합니다.
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")  # 환경 변수에서 Milvus 인증 토큰을 가져오거나 기본값을 사용합니다.
MILVUS_COLLECTION = "crop_info"  # Milvus에서 사용할 컬렉션 이름을 지정합니다.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")  # 환경 변수에서 임베딩 모델 이름을 가져오거나 기본값을 사용합니다.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 OpenAI API 키를 가져옵니다.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 환경 변수에서 사용할 OpenAI 모델 이름을 가져오거나 기본값을 사용합니다.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))  # 환경 변수에서 LLM의 응답 다양성을 조절하는 온도를 가져오거나 기본값을 사용합니다.

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # 환경 변수에서 Tavily API 키를 가져옵니다.

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 설정되어야 합니다.")  # OpenAI API 키가 없으면 오류를 발생시킵니다.

MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "800"))  # DB 컨텍스트의 최소 길이를 설정합니다.

# ==================== 전역 변수 및 초기화 ====================
_vectorstore = None  # Milvus 벡터스토어 객체를 저장할 전역 변수를 초기화합니다.
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)  # HuggingFace 임베딩 모델을 설정합니다.
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)  # OpenAI 챗 모델을 설정합니다.
agent_app = None  # LangGraph 애플리케이션 객체를 저장할 전역 변수를 초기화합니다.

# ==================== 프롬프트 템플릿 ====================
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천 전문가입니다.
아래 '문맥'을 참고하여 사용자의 질문에 맞는 작물을 추천하고 재배 방법을 안내해주세요.

[문맥]
{context}

규칙:
- 문맥에 있는 정보만 사용하여 답변하세요.
- 추천 작물과 그 이유를 명확하게 설명하세요.
- 재배 조건, 시기, 관리 방법 등을 구체적으로 안내하세요.
- 한글로만 작성하고, 단계별로 설명하세요.
- 재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.
- 문맥에 근거가 없으면 웹 검색을 통해 얻은 정보로 답변을 보강하세요.

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)  # RAG(검색 증강 생성) 프롬프트 템플릿을 생성합니다.

WEB_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천 전문가입니다.
아래 '웹 검색 결과'를 바탕으로 사용자의 질문에 맞는 작물을 추천하고 재배 정보를 종합하여 안내해주세요.

[웹 검색 결과]
{search_results}

규칙:
- 검색 결과를 바탕으로 작물 추천과 재배 방법을 종합하여 답변하세요.
- 추천 작물의 선택 이유와 장점을 명확하게 설명하세요.
- 재배 조건, 시기, 관리 방법 등을 단계별로 정리하세요.
- 재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.
- 검색 결과로 답변이 불가능할 때만 "관련 정보를 찾을 수 없습니다."라고 답변하세요.
- 모든 답변은 반드시 한국어로 작성하세요.

🟢 질문: {question}
✨ 답변:
"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL)  # 웹 검색 결과를 활용한 답변 생성을 위한 프롬프트 템플릿을 생성합니다.

class GraphState(TypedDict, total=False):  # LangGraph의 상태를 정의하는 딕셔너리 타입입니다.
    question: Optional[str]  # 사용자의 질문을 저장하는 필드입니다.
    db_context: Optional[str]  # DB 검색 결과를 저장하는 필드입니다.
    web_context: Optional[str]  # 웹 검색 결과를 저장하는 필드입니다.
    context: Optional[str]  # 최종적으로 결합된 컨텍스트를 저장하는 필드입니다.
    answer: Optional[str]  # 최종 답변을 저장하는 필드입니다.
    answer_draft: Optional[str]  # 답변 초안을 저장하는 필드입니다.
    answer_source: Optional[str]  # 답변의 출처를 저장하는 필드입니다.

# ==================== 노드 분기 함수 ====================
def route_after_retrieve(state: "GraphState") -> str:  # DB 검색 결과에 따라 다음 노드를 결정하는 분기 함수입니다.
    """DB 컨텍스트 충분/불충분에 따라 분기 라벨을 반환합니다."""
    db = (state.get("db_context") or "").strip()  # 상태에서 DB 컨텍스트를 가져와 공백을 제거합니다.
    if (not db) or ("관련 문서를 찾을 수 없습니다." in db) or (len(db) < MIN_DB_CONTEXT_CHARS):  # DB 컨텍스트가 없거나 불충분한지 확인합니다.
        return "need_web"  # 웹 검색이 필요하다고 판단하여 라벨을 반환합니다.
    return "have_db"  # DB 컨텍스트가 충분하다고 판단하여 라벨을 반환합니다.

# ==================== 노드 함수들 ====================
def load_milvus_node(state: GraphState) -> Dict[str, Any]:  # Milvus 벡터스토어를 로드하는 노드 함수입니다.
    print("\n--- 노드: Milvus 로드 ---")  # 노드 실행 시작을 알립니다.
    global _vectorstore  # 전역 변수 _vectorstore에 접근을 선언합니다.
    if "default" not in connections.list_connections() or not connections.has_connection("default"):  # Milvus 연결이 존재하지 않으면
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)  # 새로운 연결을 생성합니다.
    try:
        _vectorstore = MilvusVectorStore(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )  # Milvus 벡터스토어 객체를 생성합니다.
        print(f"✅ Milvus 로드 완료: {MILVUS_COLLECTION}")  # 로드 성공 메시지를 출력합니다.
        return {**state}  # 현재 상태를 반환합니다.
    except Exception as e:
        print(f"❌ Milvus 로드 실패: {e}")  # 실패 메시지를 출력합니다.
        raise ConnectionError("Milvus 벡터스토어 로드 실패")  # Milvus 연결 실패 에러를 발생시킵니다.

def retrieve_node(state: GraphState) -> Dict[str, Any]:  # 문서를 검색하는 노드 함수입니다.
    print("--- 노드: 문서 검색 ---")  # 노드 실행 시작을 알립니다.
    question = state.get("question")  # 상태에서 질문을 가져옵니다.
    global _vectorstore  # 전역 변수 _vectorstore에 접근을 선언합니다.
    vectorstore = _vectorstore  # 지역 변수에 할당합니다.
    if not question or not vectorstore:
        raise ValueError("질문 또는 벡터스토어가 누락되었습니다.")  # 필수 인자가 없으면 에러를 발생시킵니다.
    
    enhanced_query = f"{question} 재배 방법 키우기 팁"  # 검색 정확도를 높이기 위해 쿼리를 보강합니다.
    docs_with_scores = vectorstore.similarity_search_with_score(enhanced_query, k=10)  # 유사도 검색을 수행합니다.
    
    filtered_docs = []  # 필터링된 문서를 저장할 리스트를 생성합니다.
    for doc, score in docs_with_scores:  # 검색된 문서와 점수를 반복합니다.
        content = doc.page_content or ""  # 문서 내용을 가져옵니다.
        if score > 0.5 and len(content.strip()) > 100:  # 점수가 0.5보다 높고 내용이 100자 이상일 경우
            filtered_docs.append(doc)  # 필터링된 문서 리스트에 추가합니다.
    
    final_docs = filtered_docs[:5]  # 최종적으로 상위 5개의 문서만 선택합니다.

    context = ""  # 컨텍스트 문자열을 초기화합니다.
    if final_docs:  # 최종 문서가 존재하면
        print(f"✅ {len(final_docs)}개 문서 검색 완료.")  # 검색 완료 메시지를 출력합니다.
        for i, doc in enumerate(final_docs):
            preview = (doc.page_content or "")[:100].replace("\n", " ")
            print(f"   - 문서 {i+1}: '{preview}...'")  # 문서 미리보기를 출력합니다.
            context += f"\n\n{doc.page_content}"  # 문서 내용을 컨텍스트에 추가합니다.
    else:  # 문서가 없으면
        print("⚠️ 검색된 문서가 없습니다.")  # 경고 메시지를 출력합니다.
        context = "관련 문서를 찾을 수 없습니다."  # 컨텍스트에 실패 메시지를 저장합니다.
        
    print(f"   - 컨텍스트 길이: {len(context)}자")  # 컨텍스트 길이를 출력합니다.
    return {**state, "db_context": context}  # 상태에 DB 컨텍스트를 추가하여 반환합니다.

def combine_context_node(state: GraphState) -> Dict[str, Any]:  # 컨텍스트를 결합하는 노드 함수입니다.
    print("--- 노드: 컨텍스트 결합 ---")  # 노드 실행 시작을 알립니다.
    db_context = state.get("db_context", "")  # DB 컨텍스트를 가져옵니다.
    web_context = state.get("web_context", "")  # 웹 컨텍스트를 가져옵니다.
    final_context = db_context  # 최종 컨텍스트를 DB 컨텍스트로 초기화합니다.
    if web_context and web_context != "웹 검색 비활성화":  # 웹 컨텍스트가 존재하고 활성화 상태이면
        print("✅ DB와 웹 컨텍스트를 결합합니다.")  # 결합 메시지를 출력합니다.
        final_context = f"[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}"  # 두 컨텍스트를 결합합니다.
    else:  # 웹 컨텍스트가 없으면
        print("ℹ️ DB 컨텍스트만 사용합니다.")  # DB 컨텍스트만 사용함을 알립니다.
    return {**state, "context": final_context}  # 최종 컨텍스트를 상태에 추가하여 반환합니다.

def generate_draft_node(state: GraphState) -> Dict[str, Any]:  # 답변 초안을 생성하는 노드 함수입니다.
    print("--- 노드: 초안 생성 ---")  # 노드 실행 시작을 알립니다.
    context = state.get("context", "")  # 최종 컨텍스트를 가져옵니다.
    question = state.get("question", "")  # 질문을 가져옵니다.
    
    if not context:  # 컨텍스트가 없으면
        print("❌ 컨텍스트가 없어 답변을 생성할 수 없습니다.")  # 에러 메시지를 출력합니다.
        return {**state, "answer_draft": "주어진 정보로는 답변할 수 없습니다."}  # 실패 메시지를 초안에 저장하고 반환합니다.

    response = llm.invoke(rag_prompt.format(context=context, question=question))  # LLM을 호출하여 답변을 생성합니다.
    ans = response.content  # 생성된 답변 내용을 가져옵니다.
    
    print("✅ 답변 초안 생성 완료.")  # 생성 완료 메시지를 출력합니다.
    return {**state, "answer_draft": ans, "answer_source": "내부 DB/웹 검색"}  # 상태에 답변 초안과 출처를 추가하여 반환합니다.

def refine_answer_node(state: GraphState) -> Dict[str, Any]:  # 최종 답변을 확정하는 노드 함수입니다.
    print("--- 노드: 최종 답변 확정 ---")  # 노드 실행 시작을 알립니다.
    answer = state.get("answer_draft", "")  # 답변 초안을 가져옵니다.
    print("✅ 최종 답변 확정 완료.")  # 확정 완료 메시지를 출력합니다.
    return {**state, "answer": answer}  # 답변 초안을 최종 답변으로 저장하여 반환합니다.

def web_search_node(state: GraphState) -> Dict[str, Any]:  # 웹 검색을 수행하는 노드 함수입니다.
    print("--- 노드: 웹 검색 ---")  # 노드 실행 시작을 알립니다.
    question = state.get("question")  # 질문을 가져옵니다.
    if not question:  # 질문이 없으면
        raise ValueError("질문이 누락되었습니다.")  # 에러를 발생시킵니다.
    if not TAVILY_API_KEY:  # Tavily API 키가 없으면
        print("⚠️ TAVILY API 키가 없어 웹 검색을 건너뜁니다.")  # 경고 메시지를 출력합니다.
        return {**state, "web_context": "웹 검색 비활성화"}  # 웹 검색 비활성화 상태를 반환합니다.
    
    search_tool = TavilyClient(api_key=TAVILY_API_KEY)  # Tavily 클라이언트 객체를 생성합니다.
    web_context_parts = []  # 웹 컨텍스트 부분을 저장할 리스트를 생성합니다.
    
    try:
        response = search_tool.search(query=question, max_results=5)  # Tavily를 사용하여 웹 검색을 수행합니다.
        
        if isinstance(response, dict) and "results" in response:  # 응답 형식을 확인합니다.
            results = response["results"]  # 검색 결과 리스트를 가져옵니다.
            for r in results:  # 각 결과에 대해 반복합니다.
                if isinstance(r, dict):
                    title = (r.get("title") or "").strip()  # 제목을 가져옵니다.
                    content = (r.get("content") or r.get("snippet") or "").strip()  # 내용을 가져옵니다.
                    url = (r.get("url") or "").strip()  # URL을 가져옵니다.
                    web_context_parts.append(f"- 출처: {url or 'N/A'}\n 내용: {content}")  # 형식에 맞춰 리스트에 추가합니다.
        else:
            print(f"⚠️ 예상치 못한 Tavily 응답 형식: {type(response)}")  # 응답 형식 오류를 경고합니다.
            web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 응답 형식 오류")  # 오류 메시지를 추가합니다.
    except Exception as e:
        print(f"❌ Tavily 검색 오류: {e}")  # 검색 중 발생한 예외를 출력합니다.
        web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 실패 - {str(e)}")  # 실패 메시지를 추가합니다.

    web_context = "\n\n".join(web_context_parts)  # 모든 웹 컨텍스트 부분을 하나의 문자열로 결합합니다.
    print("✅ 웹 검색 완료.")  # 검색 완료를 알립니다.
    return {**state, "web_context": web_context}  # 웹 컨텍스트를 상태에 추가하여 반환합니다.

def build_graph():  # LangGraph 그래프를 구축하는 함수입니다.
    global agent_app  # 전역 변수 agent_app에 접근을 선언합니다.
    if agent_app is not None:  # 이미 그래프가 빌드되었으면
        return agent_app  # 기존 객체를 반환합니다.
    
    g = StateGraph(GraphState)  # StateGraph 객체를 생성합니다.
    
    g.add_node("load_milvus", load_milvus_node)  # "load_milvus" 노드를 추가합니다.
    g.add_node("retrieve", retrieve_node)  # "retrieve" 노드를 추가합니다.
    g.add_node("web_search", web_search_node)  # "web_search" 노드를 추가합니다.
    g.add_node("combine_context", combine_context_node)  # "combine_context" 노드를 추가합니다.
    g.add_node("generate_draft", generate_draft_node)  # "generate_draft" 노드를 추가합니다.
    g.add_node("refine_answer", refine_answer_node)  # "refine_answer" 노드를 추가합니다.

    g.set_entry_point("load_milvus")  # 그래프의 시작점을 "load_milvus"로 설정합니다.
    g.add_edge("load_milvus", "retrieve")  # "load_milvus"에서 "retrieve"로 연결합니다.

    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "need_web": "web_search",
            "have_db": "combine_context",
        },
    )  # "retrieve" 노드 이후에 조건부 엣지를 추가합니다.

    g.add_edge("web_search", "combine_context")  # "web_search"에서 "combine_context"로 연결합니다.
    g.add_edge("combine_context", "generate_draft")  # "combine_context"에서 "generate_draft"로 연결합니다.
    g.add_edge("generate_draft", "refine_answer")  # "generate_draft"에서 "refine_answer"로 연결합니다.
    g.add_edge("refine_answer", END)  # "refine_answer"에서 그래프를 종료합니다.

    agent_app = g.compile()  # 그래프를 컴파일하여 실행 가능한 객체로 만듭니다.
    return agent_app  # 컴파일된 객체를 반환합니다.

def run(state: dict) -> dict:  # 에이전트를 실행하는 함수입니다.
    try:
        query = state.get("query", "")  # 상태에서 쿼리를 가져옵니다.
        if not query:  # 쿼리가 없으면
            return {"agent_answer": "질문이 제공되지 않았습니다. 작물추천 관련 질문을 해주세요."}  # 오류 메시지를 반환합니다.
        print(f"[작물추천_agent] 질문 처리 시작: {query}")  # 질문 처리 시작을 알립니다.

        # 그래프를 빌드하고 실행
        app = build_graph()
        final_state = app.invoke({"question": query})  # LangGraph 애플리케이션을 호출하여 그래프를 실행합니다.

        if isinstance(final_state, dict):  # 최종 상태가 딕셔너리이면
            answer = final_state.get("answer", "답변 생성에 실패했습니다.")  # 'answer'를 가져옵니다.
        elif isinstance(final_state, str):  # 최종 상태가 문자열이면
            answer = final_state  # 그대로 사용합니다.
        else:  # 다른 형식의 상태이면
            answer = "답변 형식이 올바르지 않습니다."  # 오류 메시지를 반환합니다.

        print(f"[작물추천_agent] 답변 생성 완료: {len(answer)}자")  # 답변 생성 완료를 알립니다.
        return {"agent_answer": answer}  # 최종 답변을 반환합니다.
    except Exception as e:
        error_msg = f"작물추천 에이전트 실행 중 오류가 발생했습니다: {e}"  # 예외 발생 시 에러 메시지를 생성합니다.
        print(f"[작물추천_agent] 오류: {e}")  # 오류 메시지를 출력합니다.
        return {"agent_answer": error_msg}  # 에러 메시지를 반환합니다.

def remove_markdown_and_special_chars(text: str) -> str:  # 마크다운과 특수 문자를 제거하는 유틸리티 함수입니다.
    text = re.sub(r'#{1,6}\s', '', text)  # 마크다운 헤더(#)를 제거합니다.
    text = re.sub(r'[\*\-]', '', text)  # '*', '-' 문자를 제거합니다.
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # 마크다운 링크를 제거합니다.
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 단일 공백으로 대체합니다.
    return text.strip()  # 양 끝의 공백을 제거하고 반환합니다.

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 실행되는 코드 블록입니다.
    parser = argparse.ArgumentParser(description="RAG 파이프라인 - 농업 작물 추천 시스템 (검증 제거 + 조건부 웹검색)")  # 명령행 인자 파서를 설정합니다.
    parser.add_argument("-q", "--query", type=str, help="한 번만 실행할 질문 (예: -q '주말농장에 키울 작물 추천해줘')")  # 단일 쿼리 인자를 추가합니다.
    args = parser.parse_args()  # 인자를 파싱합니다.
    
    agent_app = build_graph()  # 그래프를 빌드합니다.
    
    # 그래프 시각화
    try:
        graph_image_path = "agent_workflow.png"  # 이미지 파일 경로를 설정합니다.
        with open(graph_image_path, "wb") as f:  # 바이너리 쓰기 모드로 파일을 엽니다.
            f.write(agent_app.get_graph().draw_mermaid_png())  # 그래프를 PNG로 그려 파일에 저장합니다.
        print(f"\n:흰색_확인_표시: LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")  # 성공 메시지를 출력합니다.
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")  # 시각화 실패 시 오류를 출력합니다.
        
    
    if args.query:  # 쿼리 인자가 제공되면
        print(f"\n질문: '{args.query}'")  # 질문을 출력합니다.
        print("-" * 20)
        final_state = agent_app.invoke({"question": args.query})  # 그래프를 한 번 실행합니다.
        answer = final_state.get("answer", "답변 생성에 실패했습니다.")  # 최종 답변을 가져옵니다.
        cleaned_answer = remove_markdown_and_special_chars(answer)  # 답변을 정리합니다.
        print("\n최종 답변:")
        print("=" * 20)
        print(cleaned_answer)  # 최종 답변을 출력합니다.
        print("=" * 20)
    else:  # 쿼리 인자가 없으면
        print("(종료: exit/quit)")  # 대화형 모드 안내를 출력합니다.
        while True:  # 무한 루프를 시작합니다.
            q = input("\n질문> ").strip()  # 사용자로부터 입력을 받습니다.
            if not q or q.lower() in ("exit", "quit"):  # 'exit' 또는 'quit' 입력 시
                break  # 루프를 종료합니다.
            final_state = agent_app.invoke({"question": q})  # 그래프를 실행합니다.
            answer = final_state.get("answer", "답변 생성에 실패했습니다.")
            cleaned_answer = remove_markdown_and_special_chars(answer)
            print("\n최종 답변:")
            print("=" * 20)
            print(cleaned_answer)
            print("=" * 20)