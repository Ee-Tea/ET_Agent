import os
import sys
import re
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict

# Langchain 및 LangGraph 관련 라이브러리
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# Milvus 관련 라이브러리 임포트
from langchain_community.vectorstores import Milvus as LangChainMilvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.retrievers import EnsembleRetriever

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# API 키가 설정되지 않았을 경우 경고 메시지를 출력하고 종료합니다.
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit()
if not TAVILY_API_KEY:
    print("오류: TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit()

# Milvus 연결 정보 및 컬렉션 이름 정의
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME_INFO = "crop_info"
COLLECTION_NAME_GROW = "crop_grow"

# --- 2. LLM 및 프롬프트 설정 ---
# 답변 생성을 위한 LLM
llm = ChatGroq(model_name="llama3-70b-8192",
                temperature=0.7,
                api_key=OPENAI_API_KEY)

# 복합 질문 분류를 위한 프롬프트 (농약 주제 제거)
MULTI_CLASSIFY_PROMPT_TEMPLATE = """
당신은 사용자의 질문이 어떤 주제에 관한 것인지 분류하는 전문가입니다.

질문을 분석하여 다음 규칙에 따라 답변하세요.
- 질문에 '농작물' 재배 또는 관리 관련 내용이 있다면, 'crop_growth'를 포함하세요.
- 그 외의 모든 일반적인 질문이라면, 'general'를 포함하세요.
- 여러 주제가 포함된 경우, 쉼표(,)로 구분하여 답변하세요.
- 답변은 오직 주제 키워드만 포함해야 합니다.

질문: {question}
답변:
"""
multi_classify_prompt = ChatPromptTemplate.from_template(MULTI_CLASSIFY_PROMPT_TEMPLATE)

# DB + 웹 검색 결과를 요약하기 위한 새로운 프롬프트 (API 부분 제거)
DB_AND_WEB_SEARCH_PROMPT_TEMPLATE = """
당신은 검색 전문가입니다.
다음 검색 결과들을 활용하여 사용자의 질문에 가장 정확하고 완전한 답변을 제공해 주세요.

# DB 검색 결과:
{db_context}

# 웹 검색 결과:
{web_search_results}

답변 규칙
1. **친절하고 자연스럽게**: 친근하고 명확한 문체로 작성해 주세요.
2. **정보의 출처 명시**: DB와 웹 검색 결과에 제시된 정보만을 사용하세요. 만약 질문에 대한 답변이 검색 결과에 없다면, '검색 결과에 해당 정보가 없습니다.'라고 명확하게 말해야 합니다.
3. **핵심 요약 및 정리**: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
4. **구체적이고 상세하게**: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명 등)를 포함하여 작성해 주세요.
5. **한글로만 답변**: 모든 답변은 한글로만 제공해야 합니다.
6. **내부 DB 정보를 우선적으로 활용**: 내부 DB에 관련된 내용이 있다면 이를 우선적으로 사용하고, 부족한 부분을 웹 검색 결과로 보충하세요.

질문: {question}
답변:
"""
db_and_web_search_prompt = ChatPromptTemplate.from_template(DB_AND_WEB_SEARCH_PROMPT_TEMPLATE)

tavily_tool = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    retriever: Optional[EnsembleRetriever]
    answer: Optional[str]
    topics: Optional[List[str]]
    db_context: Optional[str]
    web_sources: Optional[List[Dict[str, Any]]]
    db_sources: Optional[List[Dict[str, Any]]]

# --- 4. 핵심 기능 함수 정의 ---
def create_retriever() -> EnsembleRetriever:
    """두 개의 Milvus 컬렉션에 연결하여 EnsembleRetriever를 생성합니다."""
    print("---기능: Milvus 컬렉션 연결 및 EnsembleRetriever 생성 시작---")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

        vectorstore_info = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_INFO,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_INFO}' 컬렉션에 연결했습니다.")

        vectorstore_grow = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_GROW,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_GROW}' 컬렉션에 연결했습니다.")

        retriever_info = vectorstore_info.as_retriever(search_kwargs={"k": 3})
        retriever_grow = vectorstore_grow.as_retriever(search_kwargs={"k": 3})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_info, retriever_grow],
            weights=[0.5, 0.5]
        )
        
        print("✅ EnsembleRetriever가 성공적으로 생성되었습니다.")
        return ensemble_retriever
    except Exception as e:
        print(f"Milvus 연결 또는 EnsembleRetriever 생성 중 오류 발생: {e}")
        raise

def retrieve_relevant_chunks(retriever: EnsembleRetriever, question: str) -> Dict[str, Any]:
    """EnsembleRetriever를 사용하여 두 컬렉션에서 관련 문서를 검색합니다."""
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    db_sources = [{"source": doc.metadata.get('source'), "page": doc.metadata.get('page'), "content": doc.page_content} for doc in docs]
    print(f"검색된 총 청크 수: {len(docs)}개")
    return {"context": context, "db_sources": db_sources}

# --- 5. LangGraph 노드 함수 정의 ---
def load_and_merge_dbs_node(state: GraphState) -> Dict[str, Any]:
    """Milvus의 EnsembleRetriever를 생성합니다."""
    print("\n---노드: Milvus EnsembleRetriever 생성 실행---")
    retriever = create_retriever()
    print("Milvus EnsembleRetriever 로드 완료.\n")
    return {**state, "retriever": retriever}

def multi_classify_question_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 복합 질문 분류 실행---")
    question = state["question"]
    chain = multi_classify_prompt | llm | StrOutputParser()
    classification_str = chain.invoke({"question": question}).strip()
    topics = [topic.strip() for topic in classification_str.split(',') if topic.strip()]
    print(f"질문이 다음 주제들로 분류되었습니다: {topics}")
    return {**state, "topics": topics}
    
def process_topics_and_retrieve_content_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 주제별 정보 검색 및 통합 실행---")
    question = state["question"]
    topics = state.get("topics", [])
    
    db_context = ""
    db_sources = []
    web_sources = []

    # 1. DB 검색 (농작물 재배 관련)
    if "crop_growth" in topics:
        print("🔍 '농작물 재배' 주제 관련 DB 정보 검색 중...")
        retrieval_result = retrieve_relevant_chunks(state["retriever"], question)
        db_context = retrieval_result["context"]
        db_sources = retrieval_result["db_sources"]
        print("✅ DB 검색 완료.")

    # 2. 웹 검색 (DB 외 일반 정보 또는 보충 정보)
    # 'general' 주제가 있거나, DB 결과가 부족할 경우 웹 검색을 수행
    if "general" in topics or not db_context:
        print("🌐 '일반' 주제 또는 정보 보충을 위한 웹 검색 중...")
        search_results = tavily_tool.invoke({"query": question})
        web_sources = [{"url": res["url"], "content": res["content"]} for res in search_results]
        print("✅ 웹 검색 완료.")
    
    return {**state, "db_context": db_context, "db_sources": db_sources, "web_sources": web_sources}

def generate_final_answer_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 최종 답변 생성 실행---")
    question = state["question"]
    db_context = state.get("db_context", "내부 DB에서 검색된 정보가 없습니다.")
    web_search_results = "\n".join([str(res) for res in state.get("web_sources", [])])
    
    if not web_search_results:
        web_search_results = "웹 검색 결과가 없습니다."
        
    inputs = {
        "question": question,
        "db_context": db_context,
        "web_search_results": web_search_results
    }
    
    final_chain = db_and_web_search_prompt | llm | StrOutputParser()
    answer = final_chain.invoke(inputs)
    return {**state, "answer": answer}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_initial_setup_graph():
    """초기 문서 로딩 및 벡터스토어 구축을 위한 그래프를 빌드합니다."""
    initial_builder = StateGraph(GraphState)
    initial_builder.add_node("load_and_merge_dbs", load_and_merge_dbs_node)
    initial_builder.set_entry_point("load_and_merge_dbs")
    initial_builder.add_edge("load_and_merge_dbs", END)
    return initial_builder.compile()

def build_query_graph():
    """질문 분류, RAG, 웹 검색을 통합한 메인 질의 그래프를 빌드합니다."""
    query_builder = StateGraph(GraphState)
    
    # 새로운 노드들 추가
    query_builder.add_node("multi_classify_question", multi_classify_question_node)
    query_builder.add_node("process_topics_and_retrieve_content", process_topics_and_retrieve_content_node)
    query_builder.add_node("generate_final_answer", generate_final_answer_node)

    # 새로운 워크플로우를 단순하게 연결
    query_builder.set_entry_point("multi_classify_question")
    query_builder.add_edge("multi_classify_question", "process_topics_and_retrieve_content")
    query_builder.add_edge("process_topics_and_retrieve_content", "generate_final_answer")
    query_builder.add_edge("generate_final_answer", END)
    
    return query_builder.compile()

# --- 7. 메인 실행 로직 ---
if __name__ == "__main__":
    print("🌱 농작물 챗봇 에이전트 시작...")
    print("--------------------------------------------------")
    
    print("챗봇 시스템을 준비하는 중입니다... (Milvus EnsembleRetriever 생성)")
    setup_graph = build_initial_setup_graph()
    initial_state = {"question": "setup"}
    try:
        setup_result = setup_graph.invoke(initial_state)
        retriever = setup_result.get("retriever")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        exit()
        
    print("챗봇 시스템 준비 완료!\n")
    
    rag_app = build_query_graph()

    print("이제 질문을 입력하세요. (종료하려면 'exit' 또는 'quit' 입력)")
    print("--------------------------------------------------")

    while True:
        prompt = input("질문을 입력하세요: ")
        if prompt.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        
        print("답변을 생성하는 중...")
        try:
            # 'retriever' 객체를 상태에 전달합니다.
            final_state = rag_app.invoke({"question": prompt, "retriever": retriever})
            response = final_state.get('answer', "죄송합니다. 답변을 생성하지 못했습니다.")
            
            # --- 참고 자료 출력 로직 추가 ---
            db_sources = final_state.get('db_sources', [])
            web_sources = final_state.get('web_sources', [])

            print("\n------------------- 답변 -------------------")
            print(response)
            print("-------------------------------------------\n")

            if db_sources:
                print("--- 참고한 DB 내용 ---")
                for i, source in enumerate(db_sources, 1):
                    file_name = os.path.basename(source.get('source', '')).rsplit('.', 1)[0]
                    page_num = source.get('page')
                    print(f"**[{i}]** 출처: {file_name}", end="")
                    if page_num is not None:
                        print(f", 페이지: {page_num + 1}", end="")
                    print(f"\n내용: {source.get('content', '내용 없음')[:100]}...\n")
            
            if web_sources:
                print("--- 참고한 웹 검색 결과 ---")
                for i, source in enumerate(web_sources, 1):
                    print(f"**[{i}]** URL: {source.get('url', 'URL 없음')}")
                    print(f"내용: {source.get('content', '내용 없음')[:100]}...\n")

            print("-------------------------------------------\n")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")